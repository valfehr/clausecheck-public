"""What does the silent VRAM spill cost? Same model, same requests, context that fits vs one that spills. $0.

Starts llama-server at each context size, sends the same 8 requests one at a time (concurrency 1),
reports median TTFT and decode tok/s.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from loadtest import build_prompts, one_request  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tools" / "llama.cpp" / "llama-server.exe"
MODEL = ROOT / "models" / "gguf" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PORT = 8098


def run(ctx: int) -> tuple[float, float]:
    p = subprocess.Popen([str(SERVER), "-m", str(MODEL), "-c", str(ctx), "-ngl", "99", "-np", "1",
                          "--port", str(PORT), "--host", "127.0.0.1"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            time.sleep(1)
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                pass
        url = f"http://127.0.0.1:{PORT}"
        prompts = build_prompts(8)
        one_request(url, prompts[0], 8)  # warm-up
        res = [one_request(url, p, 200) for p in prompts]
        return statistics.median(r["ttft"] for r in res), statistics.median(r["tok_s"] for r in res)
    finally:
        p.terminate()
        p.wait(timeout=30)


if __name__ == "__main__":
    for ctx in (12288, 32768):
        ttft, toks = run(ctx)
        print(f"ctx {ctx:>6}: TTFT p50 {ttft:.2f}s, decode {toks:.1f} tok/s", flush=True)
