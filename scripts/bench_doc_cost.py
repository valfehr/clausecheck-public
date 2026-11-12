"""Real $/doc for a full review: all 8 golden rules on one contract, with prompt caching.

Production call pattern per contract:
  rule 1 alone   → writes the contract into the prompt cache (1.25x input price)
  rules 2-8 in parallel → read the contract from cache (0.1x input price)
Sending all 8 at once would race: several calls would miss the cache and each pay the write.

Picks 10 unmodified contracts spread across the length distribution (dev/test families only,
never holdout). Bypasses the disk cache so every call is a real, billed API call.
The "no prompt caching" column is computed from the same token counts (no extra spend).

Cost: ~80 Haiku calls ≈ $0.05.
"""

from __future__ import annotations

import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_evalset import family_split  # noqa: E402
from clausecheck.cuad import load_contracts  # noqa: E402
from clausecheck.judge import judge  # noqa: E402
from clausecheck.llm import AnthropicAdapter, cost_usd, spent_usd  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.segment import segment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase1_doc_cost.md"
MODEL = "claude-haiku-5-5"
CLIENT = "the customer / recipient of goods or services"
QUANTILES = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.85, 0.9, 0.95, 0.99]


def main() -> None:
    rules = load_rules()
    adapter = AnthropicAdapter(tag="phase1-doc-cost")
    pool = sorted(
        (c for c in load_contracts() if family_split(c.family) in ("dev", "test")),
        key=lambda c: len(c.text),
    )
    docs = [pool[int(q * (len(pool) - 1))] for q in QUANTILES]
    before = spent_usd()

    rows = []
    for c in docs:
        t_seg = time.perf_counter()
        clauses = segment(c.doc_id, c.text)
        seg_ms = (time.perf_counter() - t_seg) * 1000

        t0 = time.perf_counter()
        first = judge(adapter, clauses, rules[0], CLIENT, model=MODEL, disk_cache=False)
        with ThreadPoolExecutor(7) as ex:
            rest = list(ex.map(lambda r: judge(adapter, clauses, r, CLIENT, model=MODEL, disk_cache=False), rules[1:]))
        wall = time.perf_counter() - t0

        calls = [first, *rest]
        llm = [x.llm for x in calls]
        inp = sum(x["input_tokens"] for x in llm)
        cr = sum(x["cache_read_tokens"] for x in llm)
        cw = sum(x["cache_write_tokens"] for x in llm)
        out = sum(x["output_tokens"] for x in llm)
        actual = sum(x["cost_usd"] for x in llm)
        no_cache = sum(cost_usd(MODEL, x["input_tokens"] + x["cache_read_tokens"] + x["cache_write_tokens"], x["output_tokens"]) for x in llm)
        rows.append({
            "doc": c.doc_id[:40], "family": c.family, "tokens": first.llm["input_tokens"] + first.llm["cache_write_tokens"] + first.llm["cache_read_tokens"],
            "cache_hits": sum(1 for x in llm[1:] if x["cache_read_tokens"] > 0), "cw": cw, "cr": cr, "inp": inp, "out": out,
            "actual": actual, "no_cache": no_cache, "out_cost": cost_usd(MODEL, 0, out), "wall": wall, "seg_ms": seg_ms,
            "statuses": "".join(x.verdict.status.value[0].upper() for x in calls),
        })
        print(f"{rows[-1]['doc']:<40} {rows[-1]['tokens']:>7,} tok  ${actual:.4f} (no cache ${no_cache:.4f})  "
              f"cache hits {rows[-1]['cache_hits']}/7  {wall:.1f}s  {rows[-1]['statuses']}")

    tot_actual = sum(r["actual"] for r in rows)
    tot_nc = sum(r["no_cache"] for r in rows)
    tot_out = sum(r["out_cost"] for r in rows)
    lines = [
        "# Phase 1 — real cost per document (8 golden rules, Haiku 5.5, effort low)",
        "",
        "Call pattern: rule 1 writes the contract to the prompt cache, rules 2-8 run in parallel and read it.",
        "",
        f"- **Median $/doc: ${statistics.median(r['actual'] for r in rows):.4f}** (mean ${tot_actual / len(rows):.4f}); "
        f"without prompt caching: median ${statistics.median(r['no_cache'] for r in rows):.4f}",
        f"- Caching saved **{100 * (1 - tot_actual / tot_nc):.0f}%** of total cost; cache hits on rules 2-8: "
        f"{sum(r['cache_hits'] for r in rows)}/{7 * len(rows)}",
        f"- Output (incl. reasoning) is **{100 * tot_out / tot_actual:.0f}%** of the cached bill; most of the rest is the first full read of the contract (the cache write)",
        f"- Wall-clock for a full 8-rule review: median **{statistics.median(r['wall'] for r in rows):.1f}s**, max {max(r['wall'] for r in rows):.1f}s",
        f"- Segmentation (local Python, once per doc): median {statistics.median(r['seg_ms'] for r in rows):.0f} ms, $0",
        f"- At list price, 1,000 contracts/month ≈ **${1000 * tot_actual / len(rows):.2f}** with caching vs ${1000 * tot_nc / len(rows):.2f} without",
        "",
        "| contract | family | tokens | cache hits | $/doc cached | $/doc no cache | wall s | verdicts (8 rules) |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {r['doc']} | {r['family']} | {r['tokens']:,} | {r['cache_hits']}/7 | ${r['actual']:.4f} | ${r['no_cache']:.4f} | "
          f"{r['wall']:.1f} | {r['statuses']} |" for r in rows],
        "",
        f"Actual spend for this benchmark: ${spent_usd() - before:.4f}",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
