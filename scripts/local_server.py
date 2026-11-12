"""Start / stop the local llama.cpp server used by Phase 4 evals. Costs $0.

  uv run python scripts/local_server.py start llama-3.1-8b      # or gemma-3-4b
  uv run python scripts/local_server.py stop

Config (pinned for the swap eval): full GPU offload, 16k-token KV pool in q8_0 (+ flash attention),
3 parallel slots (≈5.4k tokens each: RAG top-10 prompt ~4k + answer), port 8081.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tools" / "llama.cpp" / "llama-server.exe"
MODELS = {
    "llama-3.1-8b": ROOT / "models" / "gguf" / "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "gemma-3-4b": ROOT / "models" / "gguf" / "google_gemma-3-4b-it-Q4_K_M.gguf",
}
PIDFILE = ROOT / "tools" / "local_server.pid"
PORT = 8081


LORAS = {  # tuned variants: same GGUF base + a LoRA adapter applied at load time (llama.cpp --lora)
    "gemma-3-4b-tuned": ("gemma-3-4b", ROOT / "models" / "lora" / "gemma-3-4b-clausecheck.gguf"),
    "gemma-3-4b-tuned-e2": ("gemma-3-4b", ROOT / "models" / "lora" / "gemma-3-4b-clausecheck-e2.gguf"),
    "gemma-3-4b-tuned-parties": ("gemma-3-4b", ROOT / "models" / "lora" / "gemma-3-4b-clausecheck-parties.gguf"),
}


def start(name: str) -> None:
    stop()
    base, extra = name, []
    if name in LORAS:
        base, adapter = LORAS[name]
        extra = ["--lora", str(adapter)]
    log = (ROOT / "tools" / f"local_server_{name}.log").open("w", encoding="utf-8")
    p = subprocess.Popen([str(SERVER), "-m", str(MODELS[base]), "-c", "16384", "-np", "3", "-ngl", "99",
                          "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--port", str(PORT), "--host", "127.0.0.1",
                          "--alias", name, *extra], stdout=log, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    PIDFILE.write_text(str(p.pid))
    for _ in range(180):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            print(f"{name} ready on :{PORT} (pid {p.pid})")
            return
        except Exception:
            time.sleep(1)
    raise SystemExit("server did not become ready; see tools/local_server_*.log")


def stop() -> None:
    if PIDFILE.exists():
        pid = PIDFILE.read_text().strip()
        subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)
        PIDFILE.unlink()
        time.sleep(2)


if __name__ == "__main__":
    {"start": lambda: start(sys.argv[2]), "stop": stop}[sys.argv[1]]()
