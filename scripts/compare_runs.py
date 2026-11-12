"""Side-by-side table of eval runs. Costs $0.

Usage: uv run python scripts/compare_runs.py E0-haiku-low E1-haiku-medium ...   (latest run with each name)
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase1_experiments.md"


def latest(name: str) -> list[dict]:
    f = sorted((ROOT / "runs").glob(f"*_{name}.jsonl"))[-1]
    return [json.loads(l) for l in f.read_text(encoding="utf-8").split("\n") if l.strip()]


def stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    sev = [r for r in ok if r["suite"] in ("severity", "severity_hard") and r["expected_status"]]
    hard = [r for r in sev if r["suite"] == "severity_hard"]
    reds = [r for r in ok if r["expected_status"] == "red"]
    cit = [r for r in ok if r["citation_hit"] is not None]
    correct = sum(r["expected_status"] == r["pred_status"] for r in sev)
    cost = sum(r["cost_usd"] for r in ok)
    paid = [r["latency_s"] for r in ok if not r["from_cache"]] or [0]
    f = lambda n, d: f"{100 * n / d:.0f}% ({n}/{d})" if d else "–"  # noqa: E731
    return {
        "severity acc (std + hard)": f(correct, len(sev)),
        "hard-case acc": f(sum(r["expected_status"] == r["pred_status"] for r in hard), len(hard)),
        "red recall (all reds)": f(sum(r["pred_status"] == "red" for r in reds), len(reds)),
        "critical miss red→green": f(sum(r["pred_status"] == "green" for r in reds), len(reds)),
        "citation hit": f(sum(r["citation_hit"] for r in cit), len(cit)),
        "citation verified": f(sum(r["citation_valid"] for r in ok), len(ok)),
        "$ per case": f"${cost / max(len(ok), 1):.5f}",
        "$ per correct verdict": f"${sum(r['cost_usd'] for r in sev) / max(correct, 1):.5f}",
        "p50 latency": f"{statistics.median(paid):.1f}s",
        "cases / errors": f"{len(ok)} / {len(rows) - len(ok)}",
    }


def main() -> None:
    # optional: --split test  → only compare cases from that split
    args = sys.argv[1:]
    split = None
    if "--split" in args:
        i = args.index("--split")
        split = args[i + 1]
        args = args[:i] + args[i + 2:]
    names = args
    cols = {n: stats([r for r in latest(n) if split is None or r.get("split") == split]) for n in names}
    metrics = list(next(iter(cols.values())))
    lines = ["# Phase 1 — experiments (test split; one change at a time vs E0)", "",
             "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {m} | " + " | ".join(cols[n][m] for n in names) + " |" for m in metrics]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
