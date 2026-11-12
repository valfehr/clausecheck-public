"""Build Phase 1 eval suites from CUAD.

severity  (80 = 8 rules x 10): take a real lawyer-labelled clause, have Haiku rewrite it into a
          target label (3 green, 3 yellow, 2 red/contradicts) or delete it (2 red/missing), and
          splice it back into the contract. Label is known by construction; a 50-case sample
          is hand-checked (scripts/export_review_sample.py).
retrieval (60 = 8 rules x 6 present + 12 absent): unmodified contracts. Present cases score
          citations against CUAD spans only. Absent cases (CUAD lawyers found no such clause)
          are natural red/missing labels.

Splits are by contract FAMILY (License, Distributor, ...), so test contracts come from contract
types the prompt was never tuned on. Holdout is reserved for promote/block decisions.

Cost: ~130 small Haiku calls, a few cents. Re-runs are free (disk cache).
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck.cuad import Contract, Span, load_contracts  # noqa: E402
from clausecheck.llm import AnthropicAdapter, spent_usd  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.schemas import EvalCase, RedReason, Rule, Status  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "evals"
MAX_DOC_CHARS = 120_000  # ~40k tokens: caps eval cost (bias noted in PLAN.md)
SEVERITY_PLAN = ["green"] * 3 + ["yellow"] * 3 + ["red_contradicts"] * 2 + ["red_missing"] * 2
RETRIEVAL_PRESENT, RETRIEVAL_ABSENT = 6, 1  # per rule; +4 extra absent spread below to reach 60


def family_split(family: str) -> str:
    h = int(hashlib.md5(family.encode()).hexdigest(), 16) % 10
    return "dev" if h < 5 else "test" if h < 8 else "holdout"


class Rewrite(BaseModel):
    client_party: str
    rewritten_clause: str


GEN_SYSTEM = """You create evaluation data for a contract-review system.
You get a golden rule (with a green/yellow/red rubric written from "Client"'s side) and a real clause from a contract.

1. Decide which party named in the clause is best cast as "Client": the customer/licensee/recipient of goods or services. Return that party's name exactly as written in the clause (e.g. "PC Financial", "Company"). If unclear, pick the party the rule most naturally protects.
2. If a target label is given, rewrite the clause so that, from that Client's perspective, it clearly and unambiguously matches the rubric for that label and NOT the other labels. Keep the original's style, defined terms, party names and approximate length. Output only the clause text, no commentary.
   - red_contradicts: the clause exists but its legal effect is the opposite of what the rule requires.
3. If the target is "party_only", set rewritten_clause to an empty string."""


def gen_prompt(rule: Rule, clause: str, target: str) -> str:
    return (
        f"Rule: {rule.title}\nIdeal text: {rule.ideal_text}\n"
        f"Rubric:\n- green: {rule.rubric.green}\n- yellow: {rule.rubric.yellow}\n- red: {rule.rubric.red}\n\n"
        f"Original clause:\n<<<\n{clause}\n>>>\n\nTarget: {target}"
    )


def merged_spans(spans: list[Span]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s in sorted(spans, key=lambda s: s.start):
        if out and s.start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], s.end))
        else:
            out.append((s.start, s.end))
    return out


def splice(text: str, spans: list[tuple[int, int]], keep: tuple[int, int], replacement: str | None):
    """Replace `keep` span with replacement (None = delete it too); delete the other spans."""
    new_span = None
    for a, b in sorted(spans, reverse=True):
        if (a, b) == keep and replacement is not None:
            text = text[:a] + replacement + text[b:]
            new_span = (a, a + len(replacement))
        else:
            text = text[:a] + text[b:]
            if new_span and a < new_span[0]:
                new_span = (new_span[0] - (b - a), new_span[1] - (b - a))
    return text, new_span


def main() -> None:
    adapter = AnthropicAdapter(tag="phase1-evalgen")
    rules = load_rules()
    contracts = [c for c in load_contracts() if len(c.text) <= MAX_DOC_CHARS and c.family != "Unknown"]
    rng = random.Random(42)
    used: set[tuple[str, str]] = set()
    severity: list[EvalCase] = []
    retrieval: list[EvalCase] = []

    def party_for(rule: Rule, clause: str) -> str:
        r = adapter.complete(system=GEN_SYSTEM, user=gen_prompt(rule, clause, "party_only"), schema=Rewrite, max_tokens=1500)
        return r.parsed.client_party

    for rule in rules:
        cat = rule.cuad_category
        present = [c for c in contracts if c.labels.get(cat)]
        absent = [c for c in contracts if not c.labels.get(cat)]
        rng.shuffle(present)
        rng.shuffle(absent)

        # --- severity suite ---
        for i, target in enumerate(SEVERITY_PLAN):
            c: Contract = present[i]
            used.add((c.doc_id, rule.id))
            spans = merged_spans(c.labels[cat])
            keep = max(spans, key=lambda s: s[1] - s[0])
            original = c.text[keep[0]:keep[1]]
            if target == "red_missing":
                party = party_for(rule, original)
                text, new_span = splice(c.text, spans, keep, None)
                status, reason = Status.RED, RedReason.MISSING
            else:
                r = adapter.complete(system=GEN_SYSTEM, user=gen_prompt(rule, original, target), schema=Rewrite,
                                     max_tokens=3000, effort="medium")
                party = r.parsed.client_party
                text, new_span = splice(c.text, spans, keep, " " + r.parsed.rewritten_clause.strip() + " ")
                status = Status.RED if target == "red_contradicts" else Status(target)
                reason = RedReason.CONTRADICTS if target == "red_contradicts" else None
            severity.append(EvalCase(
                case_id=f"sev-{rule.id}-{i:02d}", suite="severity", split=family_split(c.family), doc_id=c.doc_id,
                family=c.family, rule_id=rule.id, client_party=party, text=text, expected_status=status,
                expected_red_reason=reason, expected_spans=[new_span] if new_span else [], source="synthetic",
                note=f"target={target}; original clause: {original[:400]}",
            ))

        # --- retrieval suite ---
        n_absent = RETRIEVAL_ABSENT + (1 if rules.index(rule) < 4 else 0)
        picks = [c for c in present if (c.doc_id, rule.id) not in used][:RETRIEVAL_PRESENT]
        for j, c in enumerate(picks):
            spans = merged_spans(c.labels[cat])
            keep = max(spans, key=lambda s: s[1] - s[0])
            retrieval.append(EvalCase(
                case_id=f"ret-{rule.id}-p{j}", suite="retrieval", split=family_split(c.family), doc_id=c.doc_id,
                family=c.family, rule_id=rule.id, client_party=party_for(rule, c.text[keep[0]:keep[1]]), text=c.text,
                expected_status=None, expected_spans=spans, source="cuad",
            ))
        for j, c in enumerate(absent[:n_absent]):
            retrieval.append(EvalCase(
                case_id=f"ret-{rule.id}-a{j}", suite="retrieval", split=family_split(c.family), doc_id=c.doc_id,
                family=c.family, rule_id=rule.id, client_party="the customer / recipient of goods or services",
                text=c.text, expected_status=Status.RED, expected_red_reason=RedReason.MISSING, source="cuad",
                note="CUAD lawyers found no clause of this category",
            ))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, cases in [("severity", severity), ("retrieval", retrieval)]:
        with (OUT_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for case in cases:
                f.write(case.model_dump_json() + "\n")
        splits = {s: sum(1 for c in cases if c.split == s) for s in ("dev", "test", "holdout")}
        print(f"{name}: {len(cases)} cases, splits {splits}")
    print(f"Families by split: { {s: sorted({c.family for c in severity + retrieval if c.split == s}) for s in ('dev', 'test', 'holdout')} }")
    print(f"Spend so far: ${spent_usd():.4f}")


if __name__ == "__main__":
    main()
