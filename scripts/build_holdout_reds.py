"""B2: more red cases for the holdout (gates on 12 reds per run swing on one case). Protocol: PLAN.md leftovers.

Per rule, on holdout-family contracts not already used for that rule (≤ 120k characters, like the other suites):
  - 1 CUAD natural absence (lawyers found no such clause): a clean red/missing label (11/11 clean in Phase 1)
  - 1 red/contradicts: the labelled clause rewritten by Haiku to the opposite legal effect (blind audit afterwards)
Written to a separate suite, severity_reds.jsonl, so earlier results stay comparable.

  uv run python scripts/build_holdout_reds.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_evalset import GEN_SYSTEM, MAX_DOC_CHARS, Rewrite, family_split, gen_prompt, merged_spans, splice  # noqa: E402
from clausecheck.cuad import load_contracts  # noqa: E402
from clausecheck.llm import AnthropicAdapter  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.schemas import EvalCase, RedReason, Status  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evals" / "severity_reds.jsonl"
GENERIC = "the customer / recipient of goods or services"


def main() -> None:
    used = set()
    for s in ("severity", "severity_hard", "retrieval"):
        for l in (ROOT / "data" / "evals" / f"{s}.jsonl").read_text(encoding="utf-8").split("\n"):
            if l.strip():
                r = json.loads(l)
                used.add((r["doc_id"], r["rule_id"]))
    pool = [c for c in load_contracts() if family_split(c.family) == "holdout" and c.family != "Unknown"
            and len(c.text) <= MAX_DOC_CHARS]
    adapter, rng, cases = AnthropicAdapter(tag="phase7-leftovers-reds"), random.Random(17), []
    for rule in load_rules():
        cat = rule.cuad_category
        absent = [c for c in pool if not c.labels.get(cat) and (c.doc_id, rule.id) not in used]
        present = [c for c in pool if c.labels.get(cat) and (c.doc_id, rule.id) not in used]
        rng.shuffle(absent)
        rng.shuffle(present)
        if absent:
            c = absent[0]
            cases.append(EvalCase(case_id=f"red-{rule.id}-absent", suite="severity_reds", split="holdout",
                                  doc_id=c.doc_id, family=c.family, rule_id=rule.id, client_party=GENERIC, text=c.text,
                                  expected_status=Status.RED, expected_red_reason=RedReason.MISSING, source="cuad",
                                  note="target=cuad_absent; CUAD lawyers found no such clause"))
        if present:
            c = present[0]
            spans = merged_spans(c.labels[cat])
            keep = max(spans, key=lambda s: s[1] - s[0])
            original = c.text[keep[0]:keep[1]]
            r = adapter.complete(system=GEN_SYSTEM, user=gen_prompt(rule, original, "red_contradicts"), schema=Rewrite,
                                 max_tokens=3000, effort="medium")
            text, new_span = splice(c.text, spans, keep, " " + r.parsed.rewritten_clause.strip() + " ")
            cases.append(EvalCase(case_id=f"red-{rule.id}-contradicts", suite="severity_reds", split="holdout",
                                  doc_id=c.doc_id, family=c.family, rule_id=rule.id, client_party=r.parsed.client_party,
                                  text=text, expected_status=Status.RED, expected_red_reason=RedReason.CONTRADICTS,
                                  expected_spans=[new_span] if new_span else [], source="synthetic",
                                  note=f"target=red_contradicts; original clause: {original[:400]}"))
        print(f"{rule.id}: absent {bool(absent)}, contradicts {bool(present)}", flush=True)
    OUT.write_text("\n".join(c.model_dump_json() for c in cases) + "\n", encoding="utf-8")
    print(f"{len(cases)} red cases → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
