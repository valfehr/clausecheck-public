"""Check the chars/4 token estimate against the real tokenizer on 20 contracts.

Uses the token-counting endpoint, which is free (rate-limited, not billed): $0 spend.
Optional: --smoke also sends ONE tiny structured-output request to Haiku 5.5 (~$0.0005)
to prove the adapter, ledger and cache work end to end.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck.cuad import load_contracts  # noqa: E402
from clausecheck.llm import AnthropicAdapter, spent_usd  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "phase0_token_validation.md"


class Ping(BaseModel):
    governing_law_state: str | None


def main() -> None:
    adapter = AnthropicAdapter(tag="phase0")
    contracts = load_contracts()
    random.seed(0)
    sample = random.sample(contracts, 20)

    rows = []
    for c in sample:
        real = adapter.count_tokens(system="", user=c.text)
        rows.append((c.doc_id[:50], c.family, len(c.text) // 4, real, real / max(len(c.text), 1)))
    ratio = sum(r[3] for r in rows) / sum(r[2] for r in rows)
    chars_per_token = sum(len(c.text) for c in sample) / sum(r[3] for r in rows)

    lines = [
        "# Phase 0 — token estimate vs real tokenizer (20 random contracts, free endpoint)",
        "",
        f"- Real tokens / (chars/4) estimate: **{ratio:.2f}x**",
        f"- Measured chars per token: **{chars_per_token:.2f}**",
        "",
        "| contract | family | est (chars/4) | real |",
        "|---|---|---|---|",
        *[f"| {r[0]} | {r[1]} | {r[2]:,} | {r[3]:,} |" for r in rows],
    ]

    if "--smoke" in sys.argv:
        c = next(c for c in sample if c.labels.get("Governing Law"))
        clause = c.labels["Governing Law"][0].text
        res = adapter.complete(
            system="Extract the US state whose law governs the contract. Null if none or non-US.",
            user=clause,
            schema=Ping,
            max_tokens=500,
        )
        again = adapter.complete(
            system="Extract the US state whose law governs the contract. Null if none or non-US.",
            user=clause,
            schema=Ping,
            max_tokens=500,
        )
        lines += [
            "",
            "## Smoke test (Haiku 5.5, structured output)",
            f"- Parsed: `{res.parsed}`; tokens in/out {res.input_tokens}/{res.output_tokens}; "
            f"cost ${res.cost_usd:.5f}; latency {res.latency_s}s",
            f"- Second identical call served from disk cache: **{again.from_cache}** (cost ${again.cost_usd})",
        ]

    lines += ["", f"Total project spend so far: ${spent_usd():.4f}"]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
