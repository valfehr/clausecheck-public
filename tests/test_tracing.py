"""Regression test (Phase 7.2): re-asks run in worker threads, and their spans must stay in the review's trace.

Measured bug before the fix: 4 of 12 model calls of a real review landed in separate, parentless traces, so the
review's trace under-reported its calls and cost by a third. $0: the model is stubbed.
"""

from __future__ import annotations

import json

from clausecheck import judge, telemetry
from clausecheck.llm import LLMResult
from clausecheck.schemas import Clause, Rule


class StubAdapter:
    """Answers 'medium' confidence so judge_consistent re-asks twice; emits a span like the real adapter."""

    def complete(self, **kw):
        with telemetry.tracer().start_as_current_span("chat stub") as span:
            span.set_attribute("clausecheck.sample", kw.get("sample", 0))
            out = judge.JudgeOut(cited_clause_ids=["c0"], quote="Supplier shall maintain insurance",
                                 rationale="stub", status="yellow", red_reason="none", confidence="medium")
            return LLMResult(text="", parsed=out, model="stub", input_tokens=1, output_tokens=1, cache_read_tokens=0,
                             cache_write_tokens=0, cost_usd=0.0, latency_s=0.0, from_cache=False, request_id=None)


def test_reask_spans_share_the_review_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "TRACE_DIR", tmp_path)
    clauses = [Clause(id="c0", doc_id="d", path=[], text="Supplier shall maintain insurance at all times.",
                      start_char=0, end_char=47)]
    rule = Rule.model_validate({"id": "insurance", "title": "Insurance", "cuad_category": "Insurance",
                                "doc_types": [], "ideal_text": "x",
                                "rubric": {"green": "g", "yellow": "y", "red": "r"}})
    with telemetry.tracer().start_as_current_span("review.run") as root:
        res = judge.judge_consistent(StubAdapter(), clauses, rule, "Client", disk_cache=False)
        root_trace = f"{root.get_span_context().trace_id:032x}"
    assert res.consistency["samples"] == ["yellow"] * 3  # 1 call + 2 re-asks happened
    spans = [json.loads(l) for l in (tmp_path / "spans.jsonl").read_text().split("\n") if l.strip()]
    chats = [s for s in spans if s["name"] == "chat stub"]
    assert len(chats) == 3
    assert {s["trace_id"] for s in chats} == {root_trace}, "re-ask spans escaped the review's trace"
