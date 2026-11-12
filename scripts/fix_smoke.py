"""One real "suggest a fix" through the API in-process (real Haiku drafter + Sonnet checker, ~$0.10).

Reviews a dev contract (served from the verdict store if reviewed before), asks for a fix on the first red verdict,
prints the proposal, then records a lawyer-style decision.
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["CLAUSECHECK_SYNC_JOBS"] = "1"
from fastapi.testclient import TestClient  # noqa: E402

from clausecheck import api  # noqa: E402
from clausecheck.cuad import load_contracts  # noqa: E402

key = secrets.token_urlsafe(16)
api.add_user("fix-demo", "fix-firm", key)
c, h = TestClient(api.app), {"X-API-Key": key}
doc = next(x for x in load_contracts() if x.doc_id.startswith("AMERICASSHOPPINGMALLINC_12_10_1999"))
d = c.post("/documents", json={"name": doc.doc_id[:60], "text": doc.text}, headers=h).json()
job = c.post("/reviews", json={"document_id": d["document_id"]}, headers=h).json()["job_id"]
s = c.get(f"/reviews/{job}", headers=h).json()
print("verdicts:", {v["rule_id"]: v["status"] for v in s["verdicts"]}, "| from storage:", s["served_from_storage"])
rule = next(v["rule_id"] for v in s["verdicts"] if v["status"] == "red")
fix = c.post(f"/reviews/{job}/verdicts/{rule}/suggest-fix", headers=h).json()["fix_id"]
f = c.get(f"/fixes/{fix}", headers=h).json()
print(f"\nsuggest-fix for {rule}: status={f['status']} verified={f.get('verified')} rounds={f.get('rounds')} "
      f"cost=${f.get('cost_usd')} error={f.get('error')}")
if f.get("redline"):
    print("proposed:", f["redline"]["action"], f["redline"]["segment_id"], "→", f["redline"]["new_text"][:400])
print("why no fix:", f.get("why_no_fix"))
print("suggestions:", [x[:120] for x in f.get("suggestions", [])][:3])
dec = ({"action": "accept", "reason_code": "ok"} if f.get("verified") else
       {"action": "reject", "reason_code": "other", "comment": "no verified fix; will draft manually"})
print("decision:", c.post(f"/fixes/{fix}/decision", json=dec, headers=h).json())
