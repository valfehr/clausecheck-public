"""Check the KV-cache math against what llama.cpp actually allocates. Costs $0.

For each context size: start llama-server fully on the GPU, wait until it's serving (or fails),
read the buffer sizes it logs (model weights, KV cache, compute) and GPU memory in use, then stop it.

  uv run python scripts/kv_probe.py --model models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
      --ctx 8192 16384 24576 32768 --kv f16 q8_0

Prediction for Llama-3.1-8B (PLAN.md Phase 3):
  KV bytes/token = 2 (K,V) × 32 layers × 8 KV heads × 128 head dim × 2 bytes (f16) = 131,072 = 128 KiB
  q8_0 KV halves that (~1 byte/value + small scale overhead) → ~68 KiB/token
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "tools" / "llama.cpp" / "llama-server.exe"
OUT = ROOT / "reports" / "phase3_kv_probe.md"
PAT = {
    "model_mib": r"CUDA0 model buffer size\s*=\s*([\d.]+) MiB",
    "kv_mib": r"CUDA0 KV buffer size\s*=\s*([\d.]+) MiB",
    "compute_mib": r"CUDA0 compute buffer size\s*=\s*([\d.]+) MiB",
}


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def probe(model: Path, ctx: int, kv: str, port: int = 8099) -> dict:
    log = ROOT / "tools" / f"probe_{ctx}_{kv}.log"
    args = [str(SERVER), "-m", str(model), "-c", str(ctx), "-ngl", "99", "-np", "1", "--port", str(port),
            "-ctk", kv, "-ctv", kv, "--host", "127.0.0.1", "-lv", "4"]  # verbose: logs buffer sizes
    if kv != "f16":
        args += ["-fa", "on"]  # quantized V cache needs flash attention
    base = gpu_used_mib()
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
    ok, used = False, None
    for _ in range(120):
        time.sleep(1)
        if p.poll() is not None:
            break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            ok, used = True, gpu_used_mib()
            break
        except Exception:
            pass
    p.terminate()
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        p.kill()
    text = log.read_text(encoding="utf-8", errors="ignore")
    # llama.cpp checks free VRAM itself; with -ngl 99 it warns and loads anyway, and the Windows driver
    # then spills the overflow into shared system RAM (no OOM, just much slower). Treat that as "doesn't fit".
    fits = "failed to fit params to free device memory" not in text if (text := log.read_text(encoding="utf-8", errors="ignore")) else True
    row = {"ctx": ctx, "kv": kv, "ok": ok and fits, "gpu_mib": (used - base) if used else None}
    if ok and not fits:
        row["error"] = "loaded, but llama.cpp says it doesn't fit in free VRAM → spills to system RAM"
    for k, pat in PAT.items():
        vals = [float(x) for x in re.findall(pat, text)]
        row[k] = sum(vals) if vals else None
    if not row["ok"] and "error" not in row:
        err = [l for l in text.splitlines() if re.search(r"(?i)out of memory|failed to allocate|error", l)]
        row["error"] = err[-1][:160] if err else "did not start"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", nargs="+", type=int, required=True)
    ap.add_argument("--kv", nargs="+", default=["f16"])
    a = ap.parse_args()
    model = ROOT / a.model
    rows = []
    for kv in a.kv:
        for ctx in a.ctx:
            r = probe(model, ctx, kv)
            rows.append(r)
            per_tok = f"{r['kv_mib'] * 1024 / ctx:.1f} KiB" if r.get("kv_mib") else "–"
            print(f"ctx {ctx:>6} kv {kv:<5} ok={r['ok']!s:<5} model {r['model_mib']} MiB  KV {r['kv_mib']} MiB "
                  f"({per_tok}/token)  compute {r['compute_mib']} MiB  GPU +{r['gpu_mib']} MiB  {r.get('error', '')}",
                  flush=True)
    lines = [f"# Phase 3 — KV cache probe: `{model.name}` on RTX 3070 Laptop (8 GB)", "",
             "| KV type | context | starts? | weights MiB | KV cache MiB | KiB per token | compute MiB | GPU memory used MiB | note |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        per_tok = f"{r['kv_mib'] * 1024 / r['ctx']:.1f}" if r.get("kv_mib") else "–"
        lines.append(f"| {r['kv']} | {r['ctx']:,} | {'✅' if r['ok'] else '❌'} | {r['model_mib'] or '–'} | {r['kv_mib'] or '–'} | "
                     f"{per_tok} | {r['compute_mib'] or '–'} | {r['gpu_mib'] or '–'} | {r.get('error', '')} |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
