"""Phase 7.2: OpenTelemetry tracing for reviews, graph nodes and model calls.

Span names and attributes follow the OpenTelemetry GenAI semantic conventions where they exist
(gen_ai.operation.name, gen_ai.request.model, gen_ai.usage.input_tokens / output_tokens), plus clausecheck.* for
what the conventions don't cover (cost, cache tokens, verdict, flag, ledger tag).

Privacy rule: spans carry METADATA only (ids, rule, model, tokens, cost, status). Never contract text, prompts,
quotes or rationales: traces often go to a third-party backend with weaker access control than the documents.

Exporters:
- default: JSONL files in data/traces/ (local, $0), read by scripts/trace_report.py
- LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY set → OTLP/HTTP to Langfuse as well (needs the user's own account and
  `opentelemetry-exporter-otlp-proto-http`; not set up yet)
"""

from __future__ import annotations

import functools
import json
import os
import threading
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult

ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "data" / "traces"
_lock = threading.Lock()
_ready = False


class JsonlExporter(SpanExporter):
    def export(self, spans) -> SpanExportResult:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        lines = []
        for s in spans:
            lines.append(json.dumps({
                "trace_id": f"{s.context.trace_id:032x}", "span_id": f"{s.context.span_id:016x}",
                "parent_id": f"{s.parent.span_id:016x}" if s.parent else None, "name": s.name,
                "start_ns": s.start_time, "end_ns": s.end_time, "ms": round((s.end_time - s.start_time) / 1e6, 1),
                "status": s.status.status_code.name, "attrs": dict(s.attributes or {})}))
        with _lock, (TRACE_DIR / "spans.jsonl").open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return SpanExportResult.SUCCESS


def setup() -> None:
    global _ready
    with _lock:
        if _ready:
            return
        provider = TracerProvider(resource=Resource.create({"service.name": "clausecheck"}))
        provider.add_span_processor(SimpleSpanProcessor(JsonlExporter()))  # simple: spans survive a crash
        if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
            import base64

            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            auth = base64.b64encode(f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()).decode()
            host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                endpoint=f"{host}/api/public/otel/v1/traces", headers={"Authorization": f"Basic {auth}"})))
        trace.set_tracer_provider(provider)
        _ready = True


def tracer():
    setup()
    return trace.get_tracer("clausecheck")


def record_llm_call(span, *, model: str, input_tokens: int, output_tokens: int, cache_read: int, cache_write: int,
                    cost_usd: float, from_cache: bool, tag: str, operation: str = "chat") -> None:
    span.set_attribute("gen_ai.operation.name", operation)
    span.set_attribute("gen_ai.provider.name", "anthropic")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    span.set_attribute("clausecheck.cache_read_tokens", cache_read)
    span.set_attribute("clausecheck.cache_write_tokens", cache_write)
    span.set_attribute("clausecheck.cost_usd", round(cost_usd, 6))
    span.set_attribute("clausecheck.from_disk_cache", from_cache)
    span.set_attribute("clausecheck.ledger_tag", tag)


def node(name: str, attrs_from=None):
    """Decorator: one span per graph node. attrs_from(input, output) -> dict of metadata attributes."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(arg):
            with tracer().start_as_current_span(f"graph.{name}") as span:
                out = fn(arg)
                if attrs_from:
                    for k, v in (attrs_from(arg, out) or {}).items():
                        if v is not None:
                            span.set_attribute(f"clausecheck.{k}", v)
                return out
        return inner
    return wrap
