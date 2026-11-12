"""Hard severity cases: the traps a careless reviewer (or model) falls into.

Per rule (8 rules x 4 traps = 32 synthetic):
  buried_carveout     → yellow: reads like green, but a carve-out inside the clause guts it
  override_elsewhere  → yellow: green clause, plus a "Notwithstanding Section X..." sentence placed far away
  crossref_definition → yellow: clause relies on a defined term; the definition (placed near the top) narrows it
  flipped_direction   → red/contradicts: identical shape to green, but the right/obligation runs the wrong way
Plus distractor-missing (6 rules): real CUAD contracts WITHOUT the clause but WITH a related, tempting
one (e.g. termination for convenience absent, renewal-termination present) → red/missing.

All labels are then blind-audited (scripts/audit_labels.py --suite severity_hard); cases the auditor
disagrees with are dropped, not argued. Cost: ~40 Haiku calls, ~$0.05.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_evalset import family_split, merged_spans, splice  # noqa: E402
from clausecheck.cuad import load_contracts  # noqa: E402
from clausecheck.llm import AnthropicAdapter, spent_usd  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.schemas import EvalCase, RedReason, Rule, Status  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evals" / "severity_hard.jsonl"
MAX_DOC_CHARS = 45_000  # ~15k tokens; keeps the full-contract audit cheap

TRAPS = {
    "buried_carveout": ("yellow", None, "Write a clause that reads like a GREEN clause at first glance, but contains a carve-out, exception or condition inside it that materially weakens it so it only meets the YELLOW rubric. Set extra_text to empty."),
    "override_elsewhere": ("yellow", None, "Write a clause that fully meets the GREEN rubric. Then, in extra_text, write ONE separate sentence for a 'Miscellaneous' section (starting 'Notwithstanding Section ...' or similar) that overrides it so that, read together, the agreement only meets the YELLOW rubric."),
    "crossref_definition": ("yellow", None, "Write a clause that meets the GREEN rubric only if a defined term you introduce (in initial capitals) is read broadly. In extra_text, write the definition of that term for the definitions section, worded so the term is narrow and the agreement, read together, only meets the YELLOW rubric."),
    "flipped_direction": ("red", "contradicts", "Write a clause with the same shape and vocabulary as a GREEN clause, but where the right or obligation runs the wrong way (it protects the counterparty against Client, or burdens Client instead of the counterparty), so it meets the RED rubric. Set extra_text to empty."),
}
PLACEMENT = {"override_elsewhere": 0.9, "crossref_definition": 0.05}

# Rule → CUAD category whose presence makes a tempting distractor when the rule's own clause is absent.
DISTRACTOR = {
    "termination_for_convenience": "Notice Period To Terminate Renewal",
    "change_of_control": "Anti-Assignment",
    "cap_on_liability": "Liquidated Damages",
    "audit_rights": "Revenue/Profit Sharing",
    "insurance": "Warranty Duration",
    "anti_assignment": "Non-Transferable License",
}


class HardOut(BaseModel):
    client_party: str
    clause: str
    extra_text: str


SYSTEM = """You create hard evaluation cases for a contract-review system. You get a golden rule with a
green/yellow/red rubric written from "Client"'s side, and a real clause from a contract for style reference.
First decide which party named in the reference clause is "Client" (the customer/licensee/recipient); return its exact name.
Then write the case as instructed. Keep the reference's style, defined terms and party names. Output legal text only, no commentary."""


def insert_at_boundary(text: str, frac: float, snippet: str) -> tuple[str, tuple[int, int]]:
    """Insert snippet at the sentence boundary nearest to frac * len(text)."""
    target = int(len(text) * frac)
    bounds = [m.end() for m in re.finditer(r"[.;]\s+(?=[A-Z0-9(])", text)] or [len(text)]
    pos = min(bounds, key=lambda b: abs(b - target))
    ins = snippet.strip() + " "
    return text[:pos] + ins + text[pos:], (pos, pos + len(ins))


def main() -> None:
    adapter = AnthropicAdapter(tag="phase1-hardgen")
    rules = load_rules()
    contracts = [c for c in load_contracts() if len(c.text) <= MAX_DOC_CHARS and c.family != "Unknown"]
    rng = random.Random(1234)
    cases: list[EvalCase] = []

    for rule in rules:
        cat = rule.cuad_category
        present = [c for c in contracts if c.labels.get(cat)]
        rng.shuffle(present)
        for i, (trap, (status, reason, instr)) in enumerate(TRAPS.items()):
            c = present[i]
            spans = merged_spans(c.labels[cat])
            keep = max(spans, key=lambda s: s[1] - s[0])
            ref = c.text[keep[0]:keep[1]]
            r = adapter.complete(
                system=SYSTEM,
                user=(f"Rule: {rule.title}\nRubric:\n- green: {rule.rubric.green}\n- yellow: {rule.rubric.yellow}\n"
                      f"- red: {rule.rubric.red}\n\nReference clause:\n<<<\n{ref}\n>>>\n\nInstruction: {instr}"),
                schema=HardOut, max_tokens=4000, effort="medium",
            )
            o: HardOut = r.parsed
            text, main_span = splice(c.text, spans, keep, " " + o.clause.strip() + " ")
            exp = [main_span]
            if trap in PLACEMENT and o.extra_text.strip():
                text, extra = insert_at_boundary(text, PLACEMENT[trap], o.extra_text)
                # shift the main span if the insertion landed before it
                if extra[0] <= main_span[0]:
                    shift = extra[1] - extra[0]
                    exp = [(main_span[0] + shift, main_span[1] + shift)]
                exp.append(extra)
            cases.append(EvalCase(
                case_id=f"hard-{rule.id}-{trap}", suite="severity_hard", split=family_split(c.family), doc_id=c.doc_id,
                family=c.family, rule_id=rule.id, client_party=o.client_party, text=text,
                expected_status=Status(status), expected_red_reason=RedReason(reason) if reason else None,
                expected_spans=exp, source="synthetic-hard", note=f"trap={trap}; extra={o.extra_text[:300]}",
            ))

        if rule.id in DISTRACTOR:
            pool = [c for c in contracts if not c.labels.get(cat) and c.labels.get(DISTRACTOR[rule.id])]
            rng.shuffle(pool)
            for j, c in enumerate(pool[:1]):
                cases.append(EvalCase(
                    case_id=f"hard-{rule.id}-distractor{j}", suite="severity_hard", split=family_split(c.family),
                    doc_id=c.doc_id, family=c.family, rule_id=rule.id,
                    client_party="the customer / recipient of goods or services", text=c.text,
                    expected_status=Status.RED, expected_red_reason=RedReason.MISSING, source="cuad-distractor",
                    note=f"trap=distractor_missing; CUAD: no '{cat}' clause, but '{DISTRACTOR[rule.id]}' present",
                ))

    OUT.write_text("\n".join(c.model_dump_json() for c in cases) + "\n", encoding="utf-8")
    splits = {s: sum(1 for c in cases if c.split == s) for s in ("dev", "test", "holdout")}
    print(f"severity_hard: {len(cases)} cases, splits {splits}; spend so far ${spent_usd():.4f}")


if __name__ == "__main__":
    main()
