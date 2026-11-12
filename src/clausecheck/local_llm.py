"""Local model adapter: an OpenAI-compatible server (llama.cpp here; vLLM exposes the same API).

Implements the same `complete()` interface as AnthropicAdapter, so the judge, verifier and
self-consistency code run unchanged. Differences that matter:
- Structured output via `response_format: json_schema`: llama.cpp compiles the schema into a grammar,
  so the model literally cannot emit invalid JSON (constrained decoding). Pydantic still validates.
- `effort` has no meaning here; temperature + seed control sampling. The seed is derived from
  (sample, effort), so re-asks and retries get different samples while each one is reproducible.
- No API bill → nothing goes to the spend ledger. GPU cost is accounted separately (see PLAN.md Phase 4).
- Prompt caching: llama.cpp reuses a slot's KV for a matching prompt prefix (`cache_prompt`).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from pydantic import BaseModel, ValidationError

from clausecheck.llm import CACHE_DIR, LLMResult

TEMPERATURE = 0.6
_EFFORT_SEED = {"low": 0, "medium": 1000, "high": 2000, "xhigh": 3000, "max": 4000}


class LocalAdapter:
    def __init__(self, base_url: str, model_name: str, tag: str = "local", temperature: float = TEMPERATURE) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name  # recorded in results and cache keys
        self.tag = tag
        self.temperature = temperature  # 0 = greedy: always the most likely token (re-asks become pointless)

    def count_tokens(self, *, system: str, user: str, model: str = "") -> int:
        body = json.dumps({"content": system + "\n" + user}).encode()
        req = urllib.request.Request(self.base_url + "/tokenize", data=body, headers={"Content-Type": "application/json"})
        return len(json.load(urllib.request.urlopen(req, timeout=60))["tokens"])

    def complete(self, *, system: str, user: str, model: str = "", max_tokens: int = 2000,
                 schema: type[BaseModel] | None = None, effort: str = "low", cache_system: bool = False,
                 disk_cache: bool = True, sample: int = 0) -> LLMResult:
        seed = _EFFORT_SEED.get(effort, 0) + sample
        body = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": min(max_tokens, 1500), "temperature": self.temperature, "seed": seed, "cache_prompt": True,
        }
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": True}}
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        cache_file = CACHE_DIR / f"local-{key}.json"
        if disk_cache and cache_file.exists():
            hit = json.loads(cache_file.read_text(encoding="utf-8"))
            parsed = schema.model_validate(hit["parsed"]) if schema and hit["parsed"] is not None else None
            return LLMResult(**{**hit, "parsed": parsed, "latency_s": 0.0, "from_cache": True})

        t0 = time.perf_counter()
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = json.load(urllib.request.urlopen(req, timeout=600))
        latency = time.perf_counter() - t0
        text = resp["choices"][0]["message"]["content"] or ""
        parsed = None
        if schema:
            try:
                parsed = schema.model_validate_json(text)
            except ValidationError as e:  # grammar guarantees syntax, not e.g. truncation at max_tokens
                raise RuntimeError(f"local model output failed schema validation: {str(e)[:200]}") from e
        u = resp.get("usage", {})
        result = LLMResult(
            text=text, parsed=parsed, model=self.model_name, input_tokens=u.get("prompt_tokens", 0),
            output_tokens=u.get("completion_tokens", 0), cache_read_tokens=0, cache_write_tokens=0,
            cost_usd=0.0, latency_s=round(latency, 3), from_cache=False, request_id=None,
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        stored = asdict(result)
        stored["parsed"] = parsed.model_dump(mode="json") if parsed is not None else None
        cache_file.write_text(json.dumps(stored), encoding="utf-8")
        return result
