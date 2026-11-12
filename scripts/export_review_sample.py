"""Export 50 synthetic severity cases for a human check. Costs $0.

Opens in Excel. Fill `your_label` (green / yellow / red) WITHOUT looking at `generated_as`
(hide that column first), then `comment` if something's off. Then run:
  uv run python scripts/export_review_sample.py --score
to get agreement between you and the generator (Cohen's kappa) — the label-quality number.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.schemas import EvalCase  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reviews" / "severity_sample.csv"
FIELDS = ["case_id", "rule", "rubric", "client_party", "clause_in_contract", "your_label", "comment", "generated_as"]


def export() -> None:
    rules = {r.id: r for r in load_rules()}
    cases = [EvalCase.model_validate_json(l) for l in (ROOT / "data/evals/severity.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
    sample = random.Random(7).sample(cases, 50)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for c in sorted(sample, key=lambda c: c.case_id):
            r = rules[c.rule_id]
            clause = c.text[c.expected_spans[0][0]:c.expected_spans[0][1]].strip() if c.expected_spans else "(clause removed from contract)"
            w.writerow({
                "case_id": c.case_id, "rule": r.title,
                "rubric": f"GREEN: {r.rubric.green}\nYELLOW: {r.rubric.yellow}\nRED: {r.rubric.red}",
                "client_party": c.client_party, "clause_in_contract": clause, "your_label": "", "comment": "",
                "generated_as": c.expected_status.value + (f" ({c.expected_red_reason.value})" if c.expected_red_reason else ""),
            })
    print(f"Wrote {OUT}")


def score() -> None:
    rows = list(csv.DictReader(OUT.open(encoding="utf-8-sig")))
    pairs = [(r["generated_as"].split()[0], r["your_label"].strip().lower()) for r in rows if r["your_label"].strip()]
    n = len(pairs)
    agree = sum(a == b for a, b in pairs) / n
    labels = ["green", "yellow", "red"]
    pe = sum((sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n) for l in labels)
    kappa = (agree - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"Labelled: {n}; raw agreement {agree:.0%}; Cohen's kappa {kappa:.2f}")
    disagreements = [r for r in rows if r["your_label"].strip() and r["generated_as"].split()[0] != r["your_label"].strip().lower()]
    for r in disagreements:
        print(f"  {r['case_id']}: generated {r['generated_as']}, you said {r['your_label']} — {r['comment']}")


if __name__ == "__main__":
    score() if "--score" in sys.argv else export()
