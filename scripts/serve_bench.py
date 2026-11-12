"""Replica experiment on ONE GPU, Gemma-3-4B Q4_K_M. Costs $0.

  A  1 server, 1 slot            (no batching: requests queue one at a time)
  B  1 server, 8 slots           (continuous batching inside one process)
  C  2 servers × 4 slots + lb.py (two replicas sharing the same GPU, least-outstanding routing)

Same total KV budget (16k tokens, q8_0, flash attention) in every setup, so only the topology changes.
Each setup gets the load test at concurrency 1, 4, 16 (32 requests per level).
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tools" / "llama.cpp" / "llama-server.exe"
MODEL = ROOT / "models" / "gguf" / "google_gemma-3-4b-it-Q4_K_M.gguf"
PY = [sys.executable]


def start_server(port: int, ctx: int, slots: int) -> subprocess.Popen:
    log = (ROOT / "tools" / f"bench_{port}.log").open("w", encoding="utf-8")
    return subprocess.Popen([str(SERVER), "-m", str(MODEL), "-c", str(ctx), "-np", str(slots), "-ngl", "99",
                             "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--port", str(port), "--host", "127.0.0.1"],
                            stdout=log, stderr=subprocess.STDOUT)


def wait_ready(port: int) -> None:
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"server on {port} never became ready")


def loadtest(url: str, name: str) -> None:
    subprocess.run(PY + [str(ROOT / "scripts" / "loadtest.py"), "--url", url, "--concurrency", "1", "4", "16",
                         "--requests", "32", "--name", name], check=True)


def stop(*procs: subprocess.Popen) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> None:
    which = sys.argv[1:] or ["A", "B", "C"]
    if "A" in which:
        s = start_server(8081, 16384, 1)
        try:
            wait_ready(8081)
            loadtest("http://127.0.0.1:8081", "gemma4b-A-1server-1slot")
        finally:
            stop(s)
    if "B" in which:
        s = start_server(8081, 16384, 8)
        try:
            wait_ready(8081)
            loadtest("http://127.0.0.1:8081", "gemma4b-B-1server-8slots")
        finally:
            stop(s)
    if "C" in which:
        s1, s2 = start_server(8081, 8192, 4), start_server(8082, 8192, 4)
        lb = subprocess.Popen(PY + [str(ROOT / "scripts" / "lb.py"), "--port", "8080", "--backends",
                                    "http://127.0.0.1:8081", "http://127.0.0.1:8082", "--policy", "least"])
        try:
            wait_ready(8081)
            wait_ready(8082)
            time.sleep(1)
            loadtest("http://127.0.0.1:8080", "gemma4b-C-2replicas-lb")
        finally:
            stop(lb, s1, s2)


if __name__ == "__main__":
    main()
