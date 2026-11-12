"""Phase 7.2: one review's trace as a tree, plus cost and latency per rule and per model.

Reads data/traces/spans.jsonl (written by clausecheck.telemetry). Default: the latest "review.run" trace.

  uv run python scripts/trace_report.py [--trace TRACE_ID] [--real-review DOC_ID_PREFIX]

--real-review runs one fresh review through the API in-process (real judge, ~$0.01) and reports its trace.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ROOT = Path(__file__).resolve().parents[1]
SPANS = ROOT / "data" / "traces" / "spans.jsonl"


def real_review(prefix: str) -> None:
    import secrets

    from fastapi.testclient import TestClient

    from clausecheck import api
    from clausecheck.cuad import load_contracts

    os.environ["CLAUSECHECK_SYNC_JOBS"] = "1"
    key = secrets.token_urlsafe(16)
    api.add_user("trace-demo", "trace-firm", key)
    c, h = TestClient(api.app), {"X-API-Key": key}
    doc = next(x for x in load_contracts() if x.doc_id.startswith(prefix))
    d = c.post("/documents", json={"name": doc.doc_id[:60], "text": doc.text}, headers=h).json()
    job = c.post("/reviews", json={"document_id": d["document_id"]}, headers=h).json()["job_id"]
    s = c.get(f"/reviews/{job}", headers=h).json()
    c.post(f"/reviews/{job}/decisions", json={"decisions": {v["rule_id"]: {"action": "approve", "reason_code": "approved"}
                                                            for v in s["verdicts"]}}, headers=h)
    print(f"reviewed {doc.doc_id[:50]} ({len(doc.text):,} chars): {c.get(f'/reviews/{job}', headers=h).json()['status']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--real-review")
    a = ap.parse_args()
    if a.real_review:
        real_review(a.real_review)
        from opentelemetry import trace
        trace.get_tracer_provider().force_flush()
    spans = [json.loads(l) for l in SPANS.read_text(encoding="utf-8").split("\n") if l.strip()]
    roots = [s for s in spans if s["name"] == "review.run"]
    tid = a.trace or max(roots, key=lambda s: s["start_ns"])["trace_id"]
    t = sorted((s for s in spans if s["trace_id"] == tid), key=lambda s: s["start_ns"])
    kids = defaultdict(list)
    for s in t:
        kids[s["parent_id"]].append(s)
    t0 = t[0]["start_ns"]

    def show(s, depth=0):
        at = s["attrs"]
        bits = [f"{k.split('.')[-1]}={v}" for k, v in at.items()
                if k.split(".")[-1] in ("strategy", "reused", "rule_id", "status", "needs_human", "input_tokens",
                                        "output_tokens", "cache_read_tokens", "cache_write_tokens", "cost_usd", "sample")]
        print(f"{'  ' * depth}{s['name']:<28} +{(s['start_ns'] - t0) / 1e6:7.0f} ms  {s['ms']:7.0f} ms  {' '.join(bits)}")
        for k in kids[s["span_id"]]:
            show(k, depth + 1)

    for r in kids[None]:
        show(r)
    llm = [s for s in t if s["name"].startswith("chat ")]
    rules = [s for s in t if s["name"] == "graph.judge_rule"]
    per_rule = defaultdict(lambda: [0, 0.0, 0])
    by_id = {s["span_id"]: s for s in t}
    for s in llm:
        p = by_id.get(s["parent_id"])
        while p and p["name"] != "graph.judge_rule":
            p = by_id.get(p["parent_id"])
        r = p["attrs"].get("clausecheck.rule_id") if p else "?"
        per_rule[r][0] += 1
        per_rule[r][1] += s["attrs"].get("clausecheck.cost_usd", 0)
        per_rule[r][2] += s["attrs"].get("clausecheck.cache_read_tokens", 0) > 0
    print(f"\ntrace {tid}: {len(t)} spans, {len(llm)} model calls, {len(rules)} rule nodes, "
          f"wall {(max(s['end_ns'] for s in t) - t0) / 1e9:.1f}s, "
          f"cost ${sum(s['attrs'].get('clausecheck.cost_usd', 0) for s in llm):.4f}")
    print("orphan model calls (no rule parent):", per_rule.pop("?", [0])[0])
    for r, (n, c, hits) in sorted(per_rule.items()):
        print(f"  {r:30} calls {n}  ${c:.5f}  cache hits {hits}/{n}")


if __name__ == "__main__":
    main()
