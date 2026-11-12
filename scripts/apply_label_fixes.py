"""Apply adjudicated label decisions to the severity suite. Costs $0.

Keeps the original label in the case note, so every change is traceable (labels are versioned data).
Idempotent: re-running applies the same decisions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck.schemas import EvalCase, RedReason, Status  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "data/evals/severity.jsonl"
DECISIONS = ROOT / "reviews/label_audit_decisions.jsonl"


def main() -> None:
    lines = [l for l in DECISIONS.read_text(encoding="utf-8").split("\n") if l.strip()]
    decisions = {d["case_id"]: d for d in map(json.loads, lines)}
    cases = [EvalCase.model_validate_json(l) for l in SUITE.read_text(encoding="utf-8").split("\n") if l.strip()]
    changed = 0
    for c in cases:
        d = decisions.get(c.case_id)
        if not d or d["decision"] != "relabel" or "[relabelled" in c.note:
            continue
        old = c.expected_status.value + (f"/{c.expected_red_reason.value}" if c.expected_red_reason else "")
        c.expected_status = Status(d["new_status"])
        c.expected_red_reason = RedReason(d["new_red_reason"]) if d["new_red_reason"] else None
        c.source = "synthetic+audited"
        c.note = f"[relabelled from {old} after blind audit: {d['why']}] " + c.note
        changed += 1
    SUITE.write_text("\n".join(c.model_dump_json() for c in cases) + "\n", encoding="utf-8")
    print(f"Relabelled {changed} cases")


if __name__ == "__main__":
    main()
