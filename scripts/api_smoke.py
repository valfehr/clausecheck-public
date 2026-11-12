"""Phase 7.1 check against the RUNNING API (uv run uvicorn clausecheck.api:app --port 8090): real model calls.

Creates a local dev user (random key, saved to data/dev_users.json, which git ignores), uploads two test-family
CUAD contracts, runs both reviews through the async job flow, approves the verdicts, and compares one review with
the verdicts the graph stored for the same contract in E6.4 (same client + text → served from the verdict store).
Cost: one new 8-rule review ≈ $0.01.
"""

from __future__ import annotations

import json
import secrets
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck import api  # noqa: E402
from clausecheck.cuad import load_contracts  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8090"
CLIENT = "the customer / recipient of goods or services"


def wait(c: httpx.Client, job: str, h: dict, until: set[str]) -> dict:
    for _ in range(240):
        s = c.get(f"{BASE}/reviews/{job}", headers=h).json()
        if s["status"] in until | {"failed"}:
            return s
        time.sleep(1)
    raise TimeoutError(job)


def main() -> None:
    users = ROOT / "data" / "dev_users.json"
    keys = json.loads(users.read_text()) if users.exists() else {}
    if "dev-lawyer" not in keys:
        keys["dev-lawyer"] = secrets.token_urlsafe(24)
        users.write_text(json.dumps(keys))
        api.add_user("dev-lawyer", "dev-firm", keys["dev-lawyer"])
    h = {"X-API-Key": keys["dev-lawyer"]}
    contracts = {c.doc_id: c for c in load_contracts()}
    picks = [next(d for d in contracts if d.startswith("GOLDRESOURCECORP_12_11_2008")),
             next(d for d in sorted(contracts) if d.startswith("ON4COMMUNICATIONS"))]
    with httpx.Client(timeout=60) as c:
        print("health:", c.get(f"{BASE}/health").json())
        for doc_id in picks:
            text = contracts[doc_id].text
            d = c.post(f"{BASE}/documents", json={"name": doc_id[:60], "text": text}, headers=h).json()
            t0 = time.time()
            job = c.post(f"{BASE}/reviews", json={"document_id": d["document_id"], "client": CLIENT},
                         headers={**h, "Idempotency-Key": f"smoke-{d['document_id']}"}).json()
            s = wait(c, job["job_id"], h, {"awaiting_review", "completed"})
            took = time.time() - t0
            print(f"\n{doc_id[:45]} ({len(text):,} chars): {s['status']} in {took:.1f}s, context={s.get('context')}, "
                  f"from storage={s.get('served_from_storage')}, duplicate request={job['duplicate']}")
            if s["status"] == "failed":
                print("  error:", s["error"])
                continue
            print("  verdicts:", {v["rule_id"]: v["status"] for v in s["verdicts"]})
            print("  needs a closer look:", s["needs_closer_look"])
            if s["status"] == "awaiting_review":
                dec = {v["rule_id"]: {"action": "approve", "reason_code": "approved"} for v in s["verdicts"]}
                c.post(f"{BASE}/reviews/{job['job_id']}/decisions", json={"decisions": dec}, headers=h)
                print("  after decisions:", wait(c, job["job_id"], h, {"completed"})["status"])


if __name__ == "__main__":
    main()
