"""Load test an OpenAI-compatible endpoint (llama.cpp server, vLLM, or the lb.py proxy). Costs $0.

  uv run python scripts/loadtest.py --url http://127.0.0.1:8080 --concurrency 1 4 16 --requests 32 --name gemma-1rep

Each request = a realistic RAG-sized prompt (~1.5k tokens: rule + contract excerpts) asking for a short
structured answer (max 200 output tokens), streamed, so we can time:
  TTFT     time to first token (queueing + prefill). What a user feels as "is it doing anything?"
  tok/s    decode speed of ONE request after its first token. Drops as more requests share the GPU.
  throughput  total output tokens / wall-clock across ALL requests. What capacity planning uses.
Writes reports/loadtest_<name>.md and appends to reports/loadtest_all.jsonl.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck.cuad import load_contracts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def build_prompts(n: int) -> list[str]:
    docs = [c for c in load_contracts() if 20_000 < len(c.text) < 80_000][:n]
    out = []
    for i, d in enumerate(docs):
        excerpt = " ".join(d.text.split())[i * 300: i * 300 + 4500]  # ~1.5k tokens, different per request
        out.append(
            "You are a contract lawyer. Golden rule: the counterparty must carry insurance with stated minimum "
            "limits. Using only the excerpt, answer in 2-3 sentences: is the rule green, yellow or red, and why?\n\n"
            f"Excerpt:\n{excerpt}"
        )
    return out


def one_request(url: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, n_tok, timings, backend = None, 0, {}, None
    with urllib.request.urlopen(req, timeout=600) as resp:
        backend = resp.headers.get("X-Backend")
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("timings"):
                timings = ev["timings"]
            for ch in ev.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_tok += 1
            if ev.get("usage"):
                n_tok = ev["usage"].get("completion_tokens", n_tok)
    total = time.perf_counter() - t0
    decode = total - (ttft or total)
    return {"ttft": ttft or total, "total": total, "out_tokens": n_tok,
            "tok_s": (n_tok - 1) / decode if decode > 0 and n_tok > 1 else 0.0,
            "prompt_tokens": timings.get("prompt_n"), "backend": backend}


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 16])
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    prompts = build_prompts(a.requests)
    one_request(a.url, prompts[0], 8)  # warm-up

    lines = [f"# Load test `{a.name}`", "", f"Endpoint {a.url}; {a.requests} requests per level; "
             f"~1.5k-token prompts, max {a.max_tokens} output tokens, streamed.", "",
             "| concurrency | TTFT p50 | TTFT p95 | per-request tok/s (p50) | throughput tok/s | req/s | backends |",
             "|---|---|---|---|---|---|---|"]
    for c in a.concurrency:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            res = list(ex.map(lambda p: one_request(a.url, p, a.max_tokens), prompts))
        wall = time.perf_counter() - t0
        tput = sum(r["out_tokens"] for r in res) / wall
        backends = {}
        for r in res:
            backends[r["backend"] or "direct"] = backends.get(r["backend"] or "direct", 0) + 1
        row = {"name": a.name, "concurrency": c, "ttft_p50": pct([r["ttft"] for r in res], .5),
               "ttft_p95": pct([r["ttft"] for r in res], .95), "tok_s_p50": statistics.median(r["tok_s"] for r in res),
               "throughput": tput, "rps": len(res) / wall,
               "prompt_tokens": statistics.mean(r["prompt_tokens"] or 0 for r in res), "backends": backends}
        lines.append(f"| {c} | {row['ttft_p50']:.2f}s | {row['ttft_p95']:.2f}s | {row['tok_s_p50']:.1f} | "
                     f"{tput:.1f} | {row['rps']:.2f} | {', '.join(f'{k.split(':')[-1]}:{v}' for k, v in backends.items())} |")
        print(lines[-1], flush=True)
        with (ROOT / "reports" / "loadtest_all.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    (ROOT / "reports" / f"loadtest_{a.name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
