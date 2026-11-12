"""Phase 0 exploration: contract lengths, families, clause prevalence. Costs $0 (no API calls).

Token counts here are a chars/4 estimate; scripts/validate_tokens.py checks them
against the (free) token-counting endpoint on a sample.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck.cuad import load_contracts  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "phase0_cuad_stats.md"


def main() -> None:
    contracts = load_contracts()
    docs = pd.DataFrame(
        {
            "doc_id": c.doc_id,
            "family": c.family,
            "chars": len(c.text),
            # 2.97 chars/token measured on 20 contracts with the real tokenizer (validate_tokens.py);
            # the usual chars/4 rule undercounted legal text by 35%.
            "est_tokens": int(len(c.text) / 2.97),
        }
        for c in contracts
    )
    clauses = pd.DataFrame(
        {"doc_id": c.doc_id, "category": cat, "present": bool(spans), "n_spans": len(spans),
         "span_chars": sum(s.end - s.start for s in spans)}
        for c in contracts
        for cat, spans in c.labels.items()
    )

    q = docs["est_tokens"].quantile([0.1, 0.5, 0.9, 0.99]).astype(int)
    fam = docs.groupby("family").agg(n=("doc_id", "count"), median_tokens=("est_tokens", "median")).sort_values("n", ascending=False)
    prev = (
        clauses.groupby("category")
        .agg(present_pct=("present", "mean"), median_span_chars=("span_chars", lambda s: int(s[s > 0].median()) if (s > 0).any() else 0))
        .sort_values("present_pct", ascending=False)
    )
    prev["present_pct"] = (prev["present_pct"] * 100).round(0).astype(int)

    # KV-cache fit on the local GPU (see PLAN.md Phase 3): ~12-16k tokens of KV for an 8B 4-bit model on 8 GB
    fits_local = (docs["est_tokens"] <= 12_000).mean() * 100

    lines = [
        "# Phase 0 — CUAD exploration (tokens = chars / 2.97, calibrated on the real tokenizer)",
        "",
        f"- Contracts over 100k tokens (Haiku 5.5 long-prompt pricing applies): **{(docs['est_tokens'] > 100_000).sum()}**",
        "",
        f"- Contracts: **{len(docs)}**, families: **{docs['family'].nunique()}** (unknown family: {(docs['family'] == 'Unknown').sum()})",
        f"- Tokens p10 / p50 / p90 / p99: **{q[0.1]:,} / {q[0.5]:,} / {q[0.9]:,} / {q[0.99]:,}**, max {docs['est_tokens'].max():,}",
        f"- Total corpus: ~{docs['est_tokens'].sum() / 1e6:.1f}M tokens",
        f"- Contracts that fit in ~12k tokens (rough KV budget of a local 8B 4-bit model on 8 GB): **{fits_local:.0f}%**",
        "",
        "## Cost of one full long-context pass (input only, list prices, no caching)",
        "",
        "| Model | $/M input | One read of the whole corpus |",
        "|---|---|---|",
    ]
    total = docs["est_tokens"].sum()
    for name, price in [("Haiku 5.5", 0.10), ("Sonnet 5.5", 2.00), ("Opus 5.5", 4.00)]:
        lines.append(f"| {name} | ${price:.2f} | ${total / 1e6 * price:,.2f} |")
    lines += ["", "## Families", "", fam.to_markdown(), "", "## Clause prevalence (% of contracts where lawyers found the clause)", "", prev.to_markdown()]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"\nFull report: {OUT}")


if __name__ == "__main__":
    main()
