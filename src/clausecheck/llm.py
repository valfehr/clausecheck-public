"""Model adapter: the only place that talks to an LLM provider.

Three jobs besides calling the model:
1. **Swappability**: everything else depends on `ModelAdapter`, not on a vendor SDK.
   Bedrock (Phase 8) and a local vLLM endpoint (Phase 3) become other implementations.
2. **Hard spend cap**: before each call, the worst-case cost (input estimate + max_tokens of output)
   is checked against the budget in `.env`. Every real call is appended to `data/ledger.jsonl`.
3. **Response cache**: identical requests are served from disk for $0, so re-running evals
   or scripts never pays twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "data" / "ledger.jsonl"
CACHE_DIR = ROOT / "data" / "cache" / "llm"

load_dotenv(ROOT / ".env")

# USD per 1M tokens: (input, output, cache_read). Cache writes are billed at 1.25x input.
# Source: list prices as of 2026-11. Haiku cache-read price assumed at 0.1x input.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
}
# Haiku 5.5 is priced higher for prompts over 100K tokens.
HAIKU_LONG_PROMPT = (100_000, (0.50, 2.50, 0.05))

DEFAULT_MODEL = "claude-haiku-5-5"  # cheapest; use Sonnet only where an eval shows it's needed


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    parsed: Any
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float
    latency_s: float
    from_cache: bool
    request_id: str | None = None


class ModelAdapter(Protocol):
    def complete(
        self,
        *,
        system: str,
        user: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2000,
        schema: type[BaseModel] | None = None,
        effort: str = "low",
        cache_system: bool = False,
        disk_cache: bool = True,
        sample: int = 0,
    ) -> LLMResult: ...

    def count_tokens(self, *, system: str, user: str, model: str = DEFAULT_MODEL) -> int: ...


def price_for(model: str, prompt_tokens: int) -> tuple[float, float, float]:
    if model == "claude-haiku-5-5" and prompt_tokens > HAIKU_LONG_PROMPT[0]:
        return HAIKU_LONG_PROMPT[1]
    return PRICES[model]


def cost_usd(model: str, inp: int, out: int, cache_read: int = 0, cache_write: int = 0) -> float:
    p_in, p_out, p_cr = price_for(model, inp + cache_read + cache_write)
    return (inp * p_in + out * p_out + cache_read * p_cr + cache_write * p_in * 1.25) / 1e6


LEDGER_LOCK = threading.Lock()


def ledger_entries() -> list[dict]:
    """All ledger records. Strict on purpose: a corrupt line raises, so budget checks fail CLOSED
    (stop spending) instead of silently under-counting. Reads take the same lock as writes."""
    if not LEDGER.exists():
        return []
    with LEDGER_LOCK:
        text = LEDGER.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def spent_usd() -> float:
    return sum(e["cost_usd"] for e in ledger_entries())


def budget_usd() -> float:
    return float(os.environ.get("CLAUSECHECK_BUDGET_USD", "15"))


class AnthropicAdapter:
    def __init__(self, tag: str = "dev") -> None:
        self.client = anthropic.Anthropic(max_retries=3)
        self.tag = tag  # recorded in the ledger, e.g. "phase1-eval"

    def count_tokens(self, *, system: str, user: str, model: str = DEFAULT_MODEL) -> int:
        """Free endpoint (rate-limited, not billed)."""
        r = self.client.messages.count_tokens(
            model=model, system=system, messages=[{"role": "user", "content": user}]
        )
        return r.input_tokens

    def complete(self, **kw) -> LLMResult:
        """Traced wrapper (Phase 7.2): one GenAI span per call, metadata only, never prompt or output text."""
        from clausecheck import telemetry

        model = kw.get("model", DEFAULT_MODEL)
        with telemetry.tracer().start_as_current_span(f"chat {model}") as span:
            span.set_attribute("clausecheck.sample", kw.get("sample", 0))
            r = self._complete(**kw)
            telemetry.record_llm_call(span, model=model, input_tokens=r.input_tokens, output_tokens=r.output_tokens,
                                      cache_read=r.cache_read_tokens, cache_write=r.cache_write_tokens,
                                      cost_usd=r.cost_usd, from_cache=r.from_cache, tag=self.tag)
            return r

    def _complete(
        self,
        *,
        system: str,
        user: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2000,
        schema: type[BaseModel] | None = None,
        effort: str = "low",
        cache_system: bool = False,
        disk_cache: bool = True,  # False = always call the API (benchmarks that must pay real prices)
        sample: int = 0,  # >0 = an independent re-ask of the same request, cached under its own key
    ) -> LLMResult:
        system_block: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if cache_system:
            system_block[0]["cache_control"] = {"type": "ephemeral"}
        request = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_block,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": effort},
        }

        key = hashlib.sha256(
            json.dumps({**request, "schema": schema.model_json_schema() if schema else None,
                        **({"sample": sample} if sample else {})}, sort_keys=True).encode()
        ).hexdigest()
        cache_file = CACHE_DIR / f"{key}.json"
        if disk_cache and cache_file.exists():
            hit = json.loads(cache_file.read_text(encoding="utf-8"))
            parsed = schema.model_validate(hit["parsed"]) if schema and hit["parsed"] is not None else hit["parsed"]
            return LLMResult(**{**hit, "parsed": parsed, "cost_usd": 0.0, "latency_s": 0.0, "from_cache": True})

        # Hard budget gate on the worst case: ~3.5 chars/token for input, full max_tokens of output.
        est_in = (len(system) + len(user)) // 3 + 50
        worst = cost_usd(model, est_in, max_tokens)
        spent = spent_usd()
        if spent + worst > budget_usd():
            raise BudgetExceeded(f"Spent ${spent:.4f}; this call could cost up to ${worst:.4f}; budget ${budget_usd():.2f}")

        t0 = time.perf_counter()
        if schema:
            resp = self.client.messages.parse(**request, output_format=schema)
            parsed = resp.parsed_output
        else:
            resp = self.client.messages.create(**request)
            parsed = None
        latency = time.perf_counter() - t0

        if resp.stop_reason == "refusal":
            raise RuntimeError(f"Model refused: {getattr(resp, 'stop_details', None)}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        cr, cw = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
        result = LLMResult(
            text=text,
            parsed=parsed,
            model=model,
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=cr,
            cache_write_tokens=cw,
            cost_usd=cost_usd(model, u.input_tokens, u.output_tokens, cr, cw),
            latency_s=round(latency, 3),
            from_cache=False,
            request_id=resp._request_id,
        )

        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": time.time(), "tag": self.tag, "model": model, "cost_usd": result.cost_usd,
                           "in": u.input_tokens, "out": u.output_tokens, "cache_read": cr, "cache_write": cw,
                           "latency_s": result.latency_s, "request_id": result.request_id}) + "\n"
        with LEDGER_LOCK, LEDGER.open("a", encoding="utf-8") as f:  # one writer at a time (threads interleaved lines)
            f.write(line)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        stored = asdict(result)
        stored["parsed"] = parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else parsed
        cache_file.write_text(json.dumps(stored), encoding="utf-8")
        return result

    def tool_turn(self, *, system: str, messages: list[dict], tools: list[dict], model: str = DEFAULT_MODEL,
                  max_tokens: int = 2000, effort: str = "low", tool_choice: dict | None = None,
                  disk_cache: bool = True) -> tuple[list[dict], str, float, int, int]:
        """One step of a tool-use loop (agents): same budget gate, ledger and disk cache as complete().
        Returns (content blocks as dicts, stop_reason, cost, input tokens, output tokens)."""
        request: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages,
                                   "tools": tools, "output_config": {"effort": effort}}
        if tool_choice:
            request["tool_choice"] = tool_choice
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        cache_file = CACHE_DIR / f"tool-{key}.json"
        if disk_cache and cache_file.exists():
            hit = json.loads(cache_file.read_text(encoding="utf-8"))
            return hit["content"], hit["stop_reason"], 0.0, hit["in"], hit["out"]
        est_in = len(json.dumps(request)) // 3 + 50
        spent = spent_usd()
        if spent + cost_usd(model, est_in, max_tokens) > budget_usd():
            raise BudgetExceeded(f"Spent ${spent:.4f}; budget ${budget_usd():.2f}")
        t0 = time.perf_counter()
        resp = self.client.messages.create(**request)
        latency = time.perf_counter() - t0
        u = resp.usage
        cost = cost_usd(model, u.input_tokens, u.output_tokens)
        content = [b.model_dump(mode="json", exclude_none=True) for b in resp.content]
        with LEDGER_LOCK, LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "tag": self.tag, "model": model, "cost_usd": cost,
                                "in": u.input_tokens, "out": u.output_tokens, "cache_read": 0, "cache_write": 0,
                                "latency_s": round(latency, 3), "request_id": resp._request_id}) + "\n")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({"content": content, "stop_reason": resp.stop_reason,
                                          "in": u.input_tokens, "out": u.output_tokens}), encoding="utf-8")
        return content, resp.stop_reason, cost, u.input_tokens, u.output_tokens
