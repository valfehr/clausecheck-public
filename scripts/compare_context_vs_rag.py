"""Full contract vs RAG top-10, Claude Haiku 5.5 + re-ask-3x, 3 fresh runs per split, same gates. $0.

  uv run python scripts/compare_context_vs_rag.py   → reports/context_vs_rag.md

Stability uses the gate adopted 2026-11-12 (PLAN.md decision log): 0 RED↔GREEN flips, 0 unflagged odd answers
(a run whose verdict differs from the cross-run majority must have flagged it), ≤ 15% of checks flagged.
Eval runs disable prompt caching, so $ here is the uncached eval cost; production $/contract is in Phase 1/2.
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import latest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "context_vs_rag.md"
CONFIGS = {
    "Full contract": {"test": ["SC-run1", "SC-run2", "SC-run3"], "holdout": ["HO-SC-run1", "HO-SC-run2", "HO-SC-run3"],
                      "long": ["LONG-SC-run1", "LONG-SC-run2", "LONG-SC-run3"]},
    "RAG top-10": {"test": ["RAG10-sc", "RAG10-sc-run2", "RAG10-sc-run3"],
                   "holdout": ["HO-RAG10-sc-run1", "HO-RAG10-sc-run2", "HO-RAG10-sc-run3"],
                   "long": ["LONG-RAG10-sc-run1", "LONG-RAG10-sc-run2", "LONG-RAG10-sc-run3"]},
}
SEV = ("severity", "severity_hard", "severity_long")


def stats(names: list[str]) -> dict:
    runs = [{r["case_id"]: r for r in latest(n) if "error" not in r} for n in names]
    ids = sorted(set.intersection(*(set(r) for r in runs)))
    pooled = [r[i] for r in runs for i in ids]
    reds = [x for x in pooled if x.get("expected_status") == "red"]
    sev = [x for x in pooled if x["suite"] in SEV and x.get("expected_status")]
    missing = [x for x in pooled if x["suite"] == "retrieval" and x.get("expected_status") == "red"]
    flips = rg = unflagged = 0
    for i in ids:
        s = [r[i]["pred_status"] for r in runs]
        if len(set(s)) == 1:
            continue
        flips += 1
        rg += {"red", "green"} <= set(s)
        maj = Counter(s).most_common(1)[0][0]
        unflagged += any(s[k] != maj and not runs[k][i].get("needs_human") for k in range(len(runs)))
    return {
        "n_cases": len(ids), "n_red": len(reds),
        "red_recall": sum(x["pred_status"] == "red" for x in reds) / max(len(reds), 1),
        "critical": sum(x["pred_status"] == "green" for x in reds),
        "missing_found": f"{sum(x['pred_status'] == 'red' for x in missing)}/{len(missing)}",
        "verified": sum(x["citation_valid"] for x in pooled) / len(pooled),
        "accuracy": sum(x["pred_status"] == x["expected_status"] for x in sev) / max(len(sev), 1), "n_sev": len(sev),
        "flips": flips, "rg": rg, "unflagged": unflagged,
        "flagged": sum(bool(x.get("needs_human")) for x in pooled) / len(pooled),
        "usd_check": statistics.mean(x["cost_usd"] for x in pooled),
        "in_tok": statistics.median(x["input_tokens"] for x in pooled),
        "p50": statistics.median(x["latency_s"] for x in pooled if not x["from_cache"]),
    }


def main() -> None:
    cols = [(c, sp) for sp in ("test", "holdout", "long") for c in CONFIGS]
    S = {(c, sp): stats(CONFIGS[c][sp]) for c, sp in cols}
    pct = lambda v: f"{100 * v:.0f}%"  # noqa: E731

    def gate(s: dict) -> str:
        ok = [s["red_recall"] >= 0.9, s["critical"] == 0, s["verified"] >= 0.9,
              s["rg"] == 0, s["unflagged"] == 0, s["flagged"] <= 0.15]
        return "✅ PASS" if all(ok) else "❌ BLOCK"

    rows = [
        ("Cases per run", lambda s: str(s["n_cases"])),
        ("**Red recall** (≥ 90%), pooled over 3 runs", lambda s: f"{pct(s['red_recall'])} of {s['n_red']}"),
        ("**Critical misses** red → green (0)", lambda s: str(s["critical"])),
        ("Clause genuinely absent → found red (retrieval suite)", lambda s: s["missing_found"]),
        ("**Citations verified** (≥ 90%)", lambda s: pct(s["verified"])),
        ("Severity accuracy (std + hard)", lambda s: f"{pct(s['accuracy'])} of {s['n_sev']}"),
        ("Cases whose verdict changed across runs", lambda s: f"{s['flips']} ({pct(s['flips'] / s['n_cases'])})"),
        ("**RED ↔ GREEN flips** (0)", lambda s: str(s["rg"])),
        ("**Odd answers shown without a flag** (0)", lambda s: str(s["unflagged"])),
        ("**Checks flagged for a human** (≤ 15%)", lambda s: pct(s["flagged"])),
        ("Input tokens per check (median)", lambda s: f"{s['in_tok']:,.0f}"),
        ("$ per check, eval (no prompt caching)", lambda s: f"${s['usd_check']:.5f}"),
        ("p50 latency per check", lambda s: f"{s['p50']:.1f}s"),
        ("**Gate**", gate),
    ]
    head = " | ".join(f"{c} ({sp})" for c, sp in cols)
    lines = ["# Full contract vs RAG top-10 (Claude Haiku 5.5, re-ask-3x, 3 fresh runs per split)", "",
             f"| | {head} |", "|---|" + "---|" * len(cols)]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(S[k]) for k in cols) + " |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
