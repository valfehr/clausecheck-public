"""Stability eval: does the same config give the same verdict when asked again? Costs $0.

Input: 2+ eval runs of the SAME config made without the disk cache (run_eval.py --fresh, or a
run whose calls were all new). Compares verdicts case by case.

  uv run python scripts/stability_eval.py --split test E0v11-haiku-low E0v11-repeat E0v11-repeat2

Reports
- flip rate: share of cases where the runs don't all agree
- RED <-> GREEN flips: the dangerous kind (a problem that sometimes gets signed off)
- flips by rule, and by the model's stated confidence (does "low" predict instability?)
- single-run accuracy vs majority-vote accuracy (what self-consistency would buy)
- gate: flip rate <= 5% and zero RED <-> GREEN flips (criteria fixed in PLAN.md before the run)
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import latest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "stability.md"
MAX_FLIP_RATE = 0.05


def main() -> None:
    args = sys.argv[1:]
    split = None
    if "--split" in args:
        i = args.index("--split")
        split = args[i + 1]
        args = args[:i] + args[i + 2:]
    names = args
    if len(names) < 2:
        sys.exit("Need at least 2 runs of the same config.")
    runs = [{r["case_id"]: r for r in latest(n) if "error" not in r and (split is None or r.get("split") == split)} for n in names]
    ids = sorted(set.intersection(*(set(r) for r in runs)))

    flips, rg, by_rule, by_conf = [], [], Counter(), Counter()
    conf_total = Counter()
    single_correct, majority_correct, scored = [0] * len(runs), 0, 0
    for cid in ids:
        preds = [r[cid]["pred_status"] for r in runs]
        conf = min(r[cid]["confidence"] for r in runs)  # least confident run
        conf_label = {0.3: "low", 0.6: "medium", 0.9: "high"}.get(conf, str(conf))
        conf_total[conf_label] += 1
        if len(set(preds)) > 1:
            flips.append((cid, preds))
            by_rule[runs[0][cid]["rule_id"]] += 1
            by_conf[conf_label] += 1
            if {"red", "green"} <= set(preds):
                rg.append((cid, preds))
        exp = runs[0][cid].get("expected_status")
        if exp:
            scored += 1
            for k, r in enumerate(runs):
                single_correct[k] += r[cid]["pred_status"] == exp
            majority = Counter(preds).most_common(1)[0][0]
            majority_correct += majority == exp

    n = len(ids)
    rate = len(flips) / n
    passed = rate <= MAX_FLIP_RATE and not rg
    lines = [
        f"# Stability eval: {len(runs)} identical runs, {n} cases" + (f" ({split} split)" if split else ""),
        "",
        f"Runs: {', '.join(f'`{x}`' for x in names)}",
        "",
        f"- **Flip rate** (runs disagree): **{len(flips)}/{n} ({100 * rate:.1f}%)**",
        f"- **RED ↔ GREEN flips**: **{len(rg)}**",
        f"- Single-run accuracy: {', '.join(f'{c}/{scored}' for c in single_correct)}; "
        f"**majority of {len(runs)}**: {majority_correct}/{scored}",
        f"- Flips by least-confident stated confidence: "
        + ", ".join(f"{k} {by_conf[k]}/{conf_total[k]}" for k in ("low", "medium", "high") if conf_total[k]),
        f"- Flips by rule: {', '.join(f'{k} {v}' for k, v in by_rule.most_common()) or 'none'}",
        f"- Flagged for human review (any run): "
        f"{sum(1 for cid in ids if any(r[cid].get('needs_human') for r in runs))}/{n}; "
        f"avg $/case per run: {sum(r[cid]['cost_usd'] for r in runs for cid in ids) / (n * len(runs)):.5f}",
        "",
        f"**Gate (flip rate ≤ {MAX_FLIP_RATE:.0%}, zero RED ↔ GREEN): {'PASSED' if passed else 'FAILED'}**",
        "",
        "## Flipped cases",
        "",
        *[f"- `{cid}`: {' / '.join(p)} (expected {runs[0][cid].get('expected_status') or 'n/a'})" for cid, p in flips],
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
