"""Long-contract severity suite (`severity_long`): eval cases for the RAG branch of the length switch.

Same method as Phase 1 (scripts/build_evalset.py), on test-family contracts over LONG_CHARS characters:
per rule, a real lawyer-labelled clause rewritten into green / yellow / red-contradicts, or deleted
(red-missing), and spliced back into the long contract; plus CUAD-absent contracts (natural red-missing).
Labels are then blind-audited (scripts/audit_labels.py --suite severity_long --drop).

Cost guard: own ledger tag, stops at CAP_USD. ~40 small Haiku calls (the clause, not the contract).

  uv run python scripts/build_long_suite.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_evalset import GEN_SYSTEM, Rewrite, family_split, gen_prompt, merged_spans, splice  # noqa: E402
from clausecheck.cuad import load_contracts  # noqa: E402
from clausecheck.llm import AnthropicAdapter, ledger_entries  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.schemas import EvalCase, RedReason, Status  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evals" / "severity_long.jsonl"
LONG_CHARS = 135_000  # ≈ 45k estimated tokens: clearly above the 40k full-context limit
TARGETS = ["green", "yellow", "red_contradicts", "red_missing"]
TAG, CAP_USD = "phase6-longsuite-gen", 0.30
GENERIC_CLIENT = "the customer / recipient of goods or services"


def spent() -> float:
    return sum(e["cost_usd"] for e in ledger_entries() if e.get("tag") == TAG)


def main() -> None:
    adapter = AnthropicAdapter(tag=TAG)
    rng = random.Random(7)
    contracts = [c for c in load_contracts()
                 if len(c.text) > LONG_CHARS and c.family != "Unknown" and family_split(c.family) == "test"]
    cases: list[EvalCase] = []
    for rule in load_rules():
        cat = rule.cuad_category
        present = [c for c in contracts if c.labels.get(cat)]
        absent = [c for c in contracts if not c.labels.get(cat)]
        rng.shuffle(present)
        rng.shuffle(absent)
        for i, target in enumerate(TARGETS):
            if spent() >= CAP_USD:
                sys.exit(f"STOP: ${CAP_USD} cap reached")
            c = present[i % len(present)]
            spans = merged_spans(c.labels[cat])
            keep = max(spans, key=lambda s: s[1] - s[0])
            original = c.text[keep[0]:keep[1]]
            r = adapter.complete(system=GEN_SYSTEM, user=gen_prompt(rule, original, "party_only" if target == "red_missing" else target),
                                 schema=Rewrite, max_tokens=3000, effort="medium")
            party = r.parsed.client_party
            if target == "red_missing":
                text, new_span = splice(c.text, spans, keep, None)
                status, reason = Status.RED, RedReason.MISSING
            else:
                text, new_span = splice(c.text, spans, keep, " " + r.parsed.rewritten_clause.strip() + " ")
                status = Status.RED if target == "red_contradicts" else Status(target)
                reason = RedReason.CONTRADICTS if target == "red_contradicts" else None
            cases.append(EvalCase(
                case_id=f"long-{rule.id}-{target}", suite="severity_long", split="test", doc_id=c.doc_id,
                family=c.family, rule_id=rule.id, client_party=party, text=text, expected_status=status,
                expected_red_reason=reason, expected_spans=[new_span] if new_span else [], source="synthetic",
                note=f"target={target}; long contract {len(c.text):,} chars; original clause: {original[:400]}",
            ))
        if absent:
            c = absent[0]
            cases.append(EvalCase(
                case_id=f"long-{rule.id}-cuad_absent", suite="severity_long", split="test", doc_id=c.doc_id,
                family=c.family, rule_id=rule.id, client_party=GENERIC_CLIENT, text=c.text,
                expected_status=Status.RED, expected_red_reason=RedReason.MISSING, source="cuad",
                note=f"target=cuad_absent; long contract {len(c.text):,} chars; CUAD lawyers found no such clause",
            ))
        print(f"{rule.id}: {sum(x.rule_id == rule.id for x in cases)} cases, spend ${spent():.4f}", flush=True)

    OUT.write_text("\n".join(c.model_dump_json() for c in cases) + "\n", encoding="utf-8")
    lens = sorted(len(c.text) for c in cases)
    print(f"\n{len(cases)} cases → {OUT.relative_to(ROOT)}; contract length {lens[0]:,}–{lens[-1]:,} chars "
          f"(≈ {int(lens[0] * 1.35 / 4):,}–{int(lens[-1] * 1.35 / 4):,} tokens); spend ${spent():.4f}")


if __name__ == "__main__":
    main()
