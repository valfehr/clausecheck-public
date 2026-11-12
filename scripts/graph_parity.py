"""E6.1 parity: the LangGraph version of the workflow must reproduce the workflow's verdicts.

Runs the graph on test-split severity cases (one rule per case, like the eval), with the eval call pattern
(no prompt-cache flag), so identical requests hit the stored answers in the disk cache. That is the point of
the check: if the graph sends byte-identical requests, it can't change behaviour. Any mismatch means the graph
built a different prompt or picked different excerpts.

Also exercises the plumbing: SQLite checkpoints, the human-review interrupt (resumed by a stub reviewer that
records nothing), the verdict store (a second pass must make zero model calls).

  uv run python scripts/graph_parity.py [--limit 20]
"""

from __future__ import annotations

import argparse
import json
import uuid
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from clausecheck.graph import CHECKPOINTS, build  # noqa: E402
from clausecheck.llm import ledger_entries  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG = "phase6-parity"
STORED = {"full": "SC-run3", "rag": "RAG10-sc-run3"}  # latest runs = what the disk cache holds


def stored(name: str) -> dict:
    f = sorted((ROOT / "runs").glob(f"*_{name}.jsonl"))[-1]
    return {r["case_id"]: r for r in map(json.loads, f.read_text(encoding="utf-8").split("\n")) if r.get("pred_status")}


def spend() -> tuple[int, float]:
    e = [x for x in ledger_entries() if x.get("tag") == TAG]
    return len(e), sum(x["cost_usd"] for x in e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all test cases")
    ap.add_argument("--force", choices=["full", "rag"], default=None, help="force the context strategy (test the RAG branch)")
    a = ap.parse_args()
    cases = [json.loads(l) for f in ("severity", "severity_hard")
             for l in (ROOT / "data" / "evals" / f"{f}.jsonl").read_text(encoding="utf-8").split("\n")
             if l.strip() and '"split":"test"' in l.replace(" ", "")][: a.limit or None]
    ref = {k: stored(v) for k, v in STORED.items()}
    rows, reused, (calls0, usd0) = [], 0, spend()

    with SqliteSaver.from_conn_string(str(CHECKPOINTS)) as cp:
        graph = build(cp)
        for pass_no in (1, 2):
            for c in cases:
                cfg = {"configurable": {"thread_id": f"parity-{a.force or 'auto'}-{pass_no}-{c['case_id']}-{uuid.uuid4().hex[:8]}"}}
                out = graph.invoke({"doc_key": c["case_id"], "doc_id": c["doc_id"], "text": c["text"],
                                    "client": c["client_party"], "rule_ids": [c["rule_id"]], "verdicts": [],
                                    "ledger_tag": TAG, "prompt_cache": False, "disk_cache": True,
                                    **({"force_strategy": a.force} if a.force else {})}, cfg)
                assert "__interrupt__" in out, "graph should pause for human review"
                out = graph.invoke(Command(resume={"reviewer": "parity-stub", "decisions": {}}), cfg)
                reused += pass_no == 2 and out["reused"]
                if pass_no == 1:
                    v = out["verdicts"][0]
                    want = ref[out["strategy"]].get(c["case_id"], {}).get("pred_status")
                    rows.append((c["case_id"], out["strategy"], v["status"], want))
            if pass_no == 1:
                calls1, usd1 = spend()
    calls2, usd2 = spend()

    match = sum(g == w for _, _, g, w in rows)
    for cid, strat, got, want in rows:
        print(f"{'OK ' if got == want else 'DIFF'} {cid:45} {strat:4} graph={got:6} workflow={want}")
    print(f"\nparity: {match}/{len(rows)} identical verdicts "
          f"({sum(s == 'full' for _, s, _, _ in rows)} full contract, {sum(s == 'rag' for _, s, _, _ in rows)} RAG)")
    print(f"pass 1: {calls1 - calls0} paid API calls, ${usd1 - usd0:.4f}")
    print(f"pass 2: {reused}/{len(cases)} served from the verdict store, {calls2 - calls1} paid API calls, ${usd2 - usd1:.4f}")


if __name__ == "__main__":
    main()
