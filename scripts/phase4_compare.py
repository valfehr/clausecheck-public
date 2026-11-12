"""Phase 4 swap eval: local models vs the incumbent, same retrieval, same gates. Costs $0.

  uv run python scripts/phase4_compare.py [--prefix HO-]

--prefix: compare runs named `<prefix>RAG10-sc-run{1,2,3}` and `<prefix>L-<model>-run{1,2,3}` instead
(e.g. HO- = holdout runs); writes reports/phase5_holdout.md.

Incumbent: Claude Haiku 5.5 + RAG top-10 + re-ask (`RAG10-sc`, test split).
Candidates: `L-<model>-run{1,2,3}` (3 fresh runs each, test split).
Gates (PLAN.md Phase 4, fixed before the runs): red recall ≥ 90%, zero critical misses (red → green)
in ANY run, citations verified ≥ 90%, stability flip rate ≤ 5% with zero RED ↔ GREEN.

GPU cost: wall-clock of each run × an assumed cloud GPU price (on-demand list price of an
AWS g5.xlarge, 1× A10G 24 GB, ≈ $1.01/h; an assumption, check current pricing). Laptop
speed ≠ A10G speed, so treat it as an order of magnitude.
"""

from __future__ import annotations

import datetime as dt
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import latest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase4_swap_eval.md"
GPU_USD_PER_HOUR = 1.01
PREFIX = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else ""
if PREFIX:
    OUT = ROOT / "reports" / "phase5_holdout.md"
INCUMBENT = "RAG10-sc"
INC_NAMES = ([f"{PREFIX}{INCUMBENT}-run{i}" for i in (1, 2, 3)]
             if list((ROOT / "runs").glob(f"*_{PREFIX}{INCUMBENT}-run3.jsonl")) else [INCUMBENT])
CANDIDATES = [m for m in ["llama-3.1-8b", "gemma-3-4b", "gemma-3-4b-tuned", "gemma-3-4b-tuned-greedy",
                                "gemma-3-4b-tuned-e2", "gemma-3-4b-tuned-e2-greedy", "gemma-3-4b-tuned-parties",
                                "gemma-3-4b-tuned-ptyline"]
              if list((Path(__file__).resolve().parents[1] / "runs").glob(f"*_{PREFIX}L-{m}-run3.jsonl"))]
SEV = ("severity", "severity_hard")


def wall_seconds(names: list[str]) -> float:
    """run_eval stamps a run when it FINISHES, so a run's duration = gap to the previous run's stamp.
    The 3 runs were back to back: use the mean gap (includes ~15 s of process start-up per run)."""
    stamps = [dt.datetime.strptime(sorted((ROOT / "runs").glob(f"*_{n}.md"))[-1].name.split("_")[0], "%Y%m%d-%H%M%S")
              for n in names]
    gaps = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
    return statistics.mean(gaps)


def gate_stats(runs: list[list[dict]]) -> dict:
    ok_runs = [[r for r in run if "error" not in r] for run in runs]
    errors = sum(len(run) - len(ok) for run, ok in zip(runs, ok_runs))
    pooled = [r for run in ok_runs for r in run]
    reds = [r for r in pooled if r.get("expected_status") == "red"]
    sev = [r for r in pooled if r["suite"] in SEV and r.get("expected_status")]
    out = {
        "red_recall": sum(r["pred_status"] == "red" for r in reds) / max(len(reds), 1),
        "critical": sum(r["pred_status"] == "green" for r in reds),
        "verified": sum(r["citation_valid"] for r in pooled) / max(len(pooled), 1),
        "accuracy": sum(r["expected_status"] == r["pred_status"] for r in sev) / max(len(sev), 1),
        "n_sev": len(sev), "n_red": len(reds), "errors": errors,
        "flagged": sum(bool(r.get("needs_human")) for r in pooled) / max(len(pooled), 1),
        "latency_p50": statistics.median([r["latency_s"] for r in pooled if not r["from_cache"]] or [0]),
    }
    if len(runs) > 1:
        by_id = [{r["case_id"]: r for r in run} for run in ok_runs]
        ids = set.intersection(*(set(b) for b in by_id))
        flips = [i for i in ids if len({b[i]["pred_status"] for b in by_id}) > 1]
        out["flip_rate"] = len(flips) / max(len(ids), 1)
        out["rg_flips"] = sum(1 for i in flips if {"red", "green"} <= {b[i]["pred_status"] for b in by_id})
    return out


def per_rule(runs: list[list[dict]]) -> dict[str, tuple[int, int]]:
    acc: dict[str, list[int]] = {}
    for run in runs:
        for r in run:
            if "error" in r or r["suite"] not in SEV or not r.get("expected_status"):
                continue
            acc.setdefault(r["rule_id"], [0, 0])
            acc[r["rule_id"]][0] += r["expected_status"] == r["pred_status"]
            acc[r["rule_id"]][1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


def confusions(runs: list[list[dict]]) -> Counter:
    c = Counter()
    for run in runs:
        for r in run:
            if "error" not in r and r.get("expected_status") and r["expected_status"] != r["pred_status"]:
                c[f"{r['expected_status']}→{r['pred_status']}"] += 1
    return c


def main() -> None:
    cols = {"Claude Haiku 5.5 (incumbent)": ([latest(n) for n in INC_NAMES], INC_NAMES)}
    for m in CANDIDATES:
        names = [f"{PREFIX}L-{m}-run{i}" for i in (1, 2, 3)]
        cols[m] = ([latest(n) for n in names], names)
    stats = {k: gate_stats(runs) for k, (runs, _) in cols.items()}
    rules = {k: per_rule(runs) for k, (runs, _) in cols.items()}
    conf = {k: confusions(runs) for k, (runs, _) in cols.items()}

    def gate(s: dict) -> str:
        checks = [s["red_recall"] >= 0.9, s["critical"] == 0, s["verified"] >= 0.9]
        if "flip_rate" in s:
            checks += [s["flip_rate"] <= 0.05, s["rg_flips"] == 0]
        return "✅ PASS" if all(checks) else "❌ BLOCK"

    names = list(cols)
    f = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    split = "HOLDOUT split" if PREFIX == "HO-" else "test split"
    lines = [f"# Model swap eval ({split}, RAG top-10, re-ask-3x; only the model changes)", "",
             "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    rows = [
        ("runs", lambda k, s: str(len(cols[k][0]))),
        ("**Red recall** (gate ≥ 90%)", lambda k, s: f"{f(s['red_recall'])} of {s['n_red']}"),
        ("**Critical misses red→green** (gate 0)", lambda k, s: str(s["critical"])),
        ("**Citations verified** (gate ≥ 90%)", lambda k, s: f(s["verified"])),
        ("**Flip rate over 3 runs** (gate ≤ 5%)", lambda k, s: f(s["flip_rate"]) if "flip_rate" in s else "n/a (1 run)"),
        ("**RED↔GREEN flips** (gate 0)", lambda k, s: str(s["rg_flips"]) if "rg_flips" in s else "n/a"),
        ("Severity accuracy (std + hard)", lambda k, s: f"{f(s['accuracy'])} of {s['n_sev']}"),
        ("Flagged for human review", lambda k, s: f(s["flagged"])),
        ("Errors (schema/server)", lambda k, s: str(s["errors"])),
        ("p50 latency per call", lambda k, s: f"{s['latency_p50']:.1f}s"),
        ("**Gate**", lambda k, s: gate(s)),
    ]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(k, stats[k]) for k in names) + " |")

    lines += ["", "## Cost per correct verdict", "",
              "| | $ per correct verdict | how |", "|---|---|---|"]
    inc = latest(INC_NAMES[0])
    inc_sev = [r for r in inc if "error" not in r and r["suite"] in SEV and r.get("expected_status")]
    inc_cost = sum(r["cost_usd"] for r in inc if "error" not in r)
    inc_correct = sum(r["expected_status"] == r["pred_status"] for r in inc_sev)
    lines.append(f"| Claude Haiku 5.5 | ${inc_cost / max(inc_correct, 1):.5f} | API bill, one run, all {len(inc)} cases |")
    for m in CANDIDATES:
        names_m = cols[m][1]
        walls = [wall_seconds(names_m)] * len(names_m)
        correct = [sum(r["expected_status"] == r["pred_status"] for r in run if "error" not in r
                       and r["suite"] in SEV and r.get("expected_status")) for run in cols[m][0]]
        usd = [w / 3600 * GPU_USD_PER_HOUR / max(c, 1) for w, c in zip(walls, correct)]
        lines.append(f"| {m} | ${statistics.mean(usd):.5f} | {statistics.mean(walls) / 60:.1f} min/run of GPU time "
                     f"× ${GPU_USD_PER_HOUR}/h, only if the GPU is busy 100% of the time |")

    lines += ["", "## Accuracy per rule (severity cases, pooled over runs)", "",
              "| rule | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    all_rules = sorted({r for k in names for r in rules[k]})
    for rid in all_rules:
        cells = []
        for k in names:
            a, t = rules[k].get(rid, (0, 0))
            cells.append(f"{a}/{t} ({100 * a / t:.0f}%)" if t else "–")
        lines.append(f"| {rid} | " + " | ".join(cells) + " |")

    lines += ["", "## Error types (expected → predicted, pooled)", ""]
    for k in names:
        lines.append(f"- **{k}**: " + (", ".join(f"{e} ×{n}" for e, n in conf[k].most_common()) or "none"))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
