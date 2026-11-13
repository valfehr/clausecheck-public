"""B1 follow-up: apply the label corrections from the blind Opus audit (reviews/audit_disagreements.jsonl).

Rule (PLAN.md): a label changes only when the blind auditor sides with the model at medium/high confidence and
doesn't call the rubric ambiguous. Every change is logged in reviews/label_audit_decisions.jsonl with the reason.
Then re-scores the stored full-contract runs on the corrected labels ($0).

  uv run python scripts/apply_audit_relabels.py
"""

from __future__ import annotations

import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck.schemas import EvalCase, RedReason, Status  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    audits = [json.loads(l) for l in (ROOT / "reviews" / "audit_disagreements.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
    fix = {a["case"]: a for a in audits if a["verdict"] == "sides with the model" and not a["ambiguous"]
           and a["confidence"] in ("medium", "high")}
    log = ROOT / "reviews" / "label_audit_decisions.jsonl"
    changed = []
    for suite in ("severity", "severity_hard"):
        path = ROOT / "data" / "evals" / f"{suite}.jsonl"
        cases = [EvalCase.model_validate_json(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]
        for c in cases:
            a = fix.get(c.case_id)
            if not a or c.expected_status.value == a["opus"]:
                continue
            old = c.expected_status.value
            c.expected_status = Status(a["opus"])
            c.expected_red_reason = (RedReason.CONTRADICTS if c.expected_spans else RedReason.MISSING) if a["opus"] == "red" else None
            c.note += f"; relabelled {old}->{a['opus']} 2026-11-12 (blind Opus audit, sided with the model)"
            changed.append(c.case_id)
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"case_id": c.case_id, "from": old, "to": a["opus"], "decided_by": "blind Opus audit",
                                    "reason": a["rationale"][:600], "ts": time.time()}) + "\n")
        path.write_text("\n".join(c.model_dump_json() for c in cases) + "\n", encoding="utf-8")
    print("relabelled:", changed)

    labels = {c.case_id: c.expected_status.value for s in ("severity", "severity_hard") for c in
              (EvalCase.model_validate_json(l) for l in (ROOT / "data" / "evals" / f"{s}.jsonl").read_text(encoding="utf-8").split("\n") if l.strip())}
    for split, names in (("test", ["SC-run1", "SC-run2", "SC-run3"]), ("holdout", ["HO-SC-run1", "HO-SC-run2", "HO-SC-run3"])):
        old_rr = new_rr = old_n = new_n = old_acc = new_acc = n = crit_new = 0
        for nm in names:
            f = sorted(glob.glob(str(ROOT / "runs" / f"*_{nm}.jsonl")))[-1]
            for r in map(json.loads, open(f, encoding="utf-8")):
                if not r.get("pred_status") or r["case_id"] not in labels:
                    continue
                old, new, p = r["expected_status"], labels[r["case_id"]], r["pred_status"]
                n += 1
                old_acc += p == old
                new_acc += p == new
                if old == "red":
                    old_n += 1
                    old_rr += p == "red"
                if new == "red":
                    new_n += 1
                    new_rr += p == "red"
                    crit_new += p == "green"
        print(f"{split} (full contract, 3 runs): red recall {old_rr}/{old_n} → {new_rr}/{new_n}; "
              f"severity accuracy {old_acc}/{n} → {new_acc}/{n}; critical misses now {crit_new}")


if __name__ == "__main__":
    main()
