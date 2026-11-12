"""Ingestion step: extract who's-who for each eval contract and store it as document metadata.

One Haiku call per distinct contract (synthetic edits sit in the body, so the preamble is shared by all
cases of a contract), then one metadata doc per case doc_key in the `contract_meta` index.
Cost guard: own ledger tag, stops at $0.10.

  uv run python scripts/index_parties.py --splits test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck import parties, search  # noqa: E402
from clausecheck.llm import AnthropicAdapter, ledger_entries  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG, CAP = "phase5-parties", 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--suites", nargs="+", default=["severity", "severity_hard", "retrieval"])
    a = ap.parse_args()
    es, adapter = search.client(), AnthropicAdapter(tag=TAG)
    by_doc: dict[str, parties.Parties] = {}
    n = matched = 0
    for suite in a.suites:
        for line in (ROOT / "data" / "evals" / f"{suite}.jsonl").read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            case = json.loads(line)  # split on \n only: contract text contains U+2028, which splitlines() breaks on
            if case["split"] not in a.splits:
                continue
            if sum(e["cost_usd"] for e in ledger_entries() if e.get("tag") == TAG) >= CAP:
                sys.exit(f"STOP: ${CAP} cap reached")
            if case["doc_id"] not in by_doc:
                by_doc[case["doc_id"]] = parties.extract(adapter, case["text"])
            p = by_doc[case["doc_id"]]
            parties.store(es, case["case_id"], p)
            cp = parties.counterparty_for(p, case["client_party"])
            n += 1
            matched += bool(cp)
            print(f"{case['case_id']:<45} client {case['client_party'][:28]:<28} → counterparty {cp or '(no match)'}")
    spent = sum(e["cost_usd"] for e in ledger_entries() if e.get("tag") == TAG)
    print(f"\n{n} cases, {len(by_doc)} contracts, counterparty resolved for {matched}; spend ${spent:.4f}")


if __name__ == "__main__":
    main()
