"""Phase 7.1 API tests. $0: the judge is stubbed, storage goes to a temp folder, jobs run inline.

  uv run pytest -q
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from clausecheck import api, graph, telemetry

TEXT = ("MASTER SERVICES AGREEMENT between Acme Corp (\"Customer\") and Beta LLC (\"Supplier\").\n"
        "1. Governing Law. This Agreement is governed by the laws of the State of New York.\n"
        "2. Assignment. Supplier may not assign this Agreement without Customer's prior written consent.\n" * 3)


def fake_judge(state, rule_id):
    status = "red" if rule_id == "insurance" else "green"
    return {"rule_id": rule_id, "status": status, "red_reason": "missing" if status == "red" else None,
            "cited": [], "quote": "", "rationale": f"stub verdict for {rule_id}", "confidence": 0.9,
            "citation_valid": True, "needs_human": rule_id == "audit_rights",
            "why_flagged": "the model gave different answers when asked again" if rule_id == "audit_rights" else "",
            "samples": [status], "cost_usd": 0.0}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(graph, "DB", tmp_path / "test.db")
    monkeypatch.setattr(graph, "CHECKPOINTS", tmp_path / "checkpoints.db")
    monkeypatch.setattr(graph, "ROOT", tmp_path)
    monkeypatch.setattr(telemetry, "TRACE_DIR", tmp_path / "traces")  # tests don't pollute real traces
    calls = []
    monkeypatch.setattr(graph, "_judge_one", lambda s, r: calls.append(r) or fake_judge(s, r))
    monkeypatch.setenv("CLAUSECHECK_SYNC_JOBS", "1")
    api.add_user("alice", "firm-a", "key-alice")
    api.add_user("bob", "firm-b", "key-bob")
    c = TestClient(api.app)
    c.calls = calls
    return c


A, B = {"X-API-Key": "key-alice"}, {"X-API-Key": "key-bob"}


def upload(c, headers=A):
    r = c.post("/documents", json={"name": "msa.txt", "text": TEXT}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["document_id"]


def test_auth_required(client):
    assert client.post("/documents", json={"name": "x", "text": TEXT}).status_code == 401
    assert client.post("/documents", json={"name": "x", "text": TEXT}, headers={"X-API-Key": "nope"}).status_code == 401


def test_review_pauses_for_lawyer_then_completes(client):
    doc = upload(client)
    r = client.post("/reviews", json={"document_id": doc, "client": "Acme Corp"}, headers=A)
    assert r.status_code == 202
    job = r.json()["job_id"]
    s = client.get(f"/reviews/{job}", headers=A).json()
    assert s["status"] == "awaiting_review"
    assert len(s["verdicts"]) == 8
    assert [x["rule_id"] for x in s["needs_closer_look"]] == ["audit_rights"]
    assert client.get(f"/reviews/{job}/verdicts/insurance", headers=A).json()["status"] == "red"

    d = {"decisions": {"insurance": {"action": "approve", "reason_code": "approved"},
                       "audit_rights": {"action": "change_severity", "new_status": "yellow", "reason_code": "wrong_severity",
                                        "comment": "audit right exists but no notice period"}}}
    assert client.post(f"/reviews/{job}/decisions", json=d, headers=A).status_code == 202
    assert client.get(f"/reviews/{job}", headers=A).json()["status"] == "completed"
    import sqlite3
    with sqlite3.connect(graph.DB) as con:
        rows = con.execute("SELECT rule_id, reviewer, action, new_status, reason_code FROM feedback ORDER BY rule_id").fetchall()
    assert rows == [("audit_rights", "alice", "change_severity", "yellow", "wrong_severity"),
                    ("insurance", "alice", "approve", None, "approved")]


def test_decisions_only_when_awaiting_review(client):
    doc = upload(client)
    job = client.post("/reviews", json={"document_id": doc}, headers=A).json()["job_id"]
    ok = {"decisions": {"insurance": {"action": "approve", "reason_code": "approved"}}}
    client.post(f"/reviews/{job}/decisions", json=ok, headers=A)
    assert client.post(f"/reviews/{job}/decisions", json=ok, headers=A).status_code == 409
    bad = {"decisions": {"insurance": {"action": "change_severity", "reason_code": "wrong_severity"}}}
    job2 = client.post("/reviews", json={"document_id": doc, "client": "other"}, headers=A).json()["job_id"]
    assert client.post(f"/reviews/{job2}/decisions", json=bad, headers=A).status_code == 422


def test_idempotency_key_returns_same_job(client):
    doc = upload(client)
    h = {**A, "Idempotency-Key": "abc-123"}
    first = client.post("/reviews", json={"document_id": doc}, headers=h).json()
    second = client.post("/reviews", json={"document_id": doc}, headers=h).json()
    assert second["job_id"] == first["job_id"] and second["duplicate"] is True
    assert len(client.calls) == 8  # the duplicate request made no model calls


def test_other_tenant_gets_404_not_403(client):
    doc = upload(client, A)
    job = client.post("/reviews", json={"document_id": doc}, headers=A).json()["job_id"]
    assert client.post("/reviews", json={"document_id": doc}, headers=B).status_code == 404
    assert client.get(f"/reviews/{job}", headers=B).status_code == 404
    assert client.get(f"/reviews/{job}/verdicts/insurance", headers=B).status_code == 404


def test_verdict_store_reuses_same_contract(client):
    doc = upload(client)
    j1 = client.post("/reviews", json={"document_id": doc, "client": "Acme Corp"}, headers=A).json()["job_id"]
    client.post(f"/reviews/{j1}/decisions", json={"decisions": {}}, headers=A)
    n = len(client.calls)
    j2 = client.post("/reviews", json={"document_id": doc, "client": "Acme Corp"}, headers=A).json()["job_id"]
    s = client.get(f"/reviews/{j2}", headers=A).json()
    assert s["served_from_storage"] is True and len(client.calls) == n


def _review(client):
    doc = upload(client)
    job = client.post("/reviews", json={"document_id": doc, "client": "Acme Corp"}, headers=A).json()["job_id"]
    return job


@pytest.fixture()
def drafter(monkeypatch):
    from clausecheck import drafting_v2
    from clausecheck.drafting_v2 import DraftResultV2, RedlineV2

    state = {"verified": True}

    def fake_propose(adapter, es, doc_key, clauses, rule, client_name, verdict):
        red = RedlineV2(action="insert_after", segment_id=clauses[0].id, new_text="Supplier shall maintain insurance.",
                        explanation="adds the missing duty", optional_suggestions=["name Customer as additional insured"])
        if state["verified"]:
            return DraftResultV2(accepted=True, rounds=1, redline=red, history=[{"suggestions": red.optional_suggestions}],
                                 cost_usd=0.0)
        return DraftResultV2(accepted=False, rounds=3, redline=None, cost_usd=0.0,
                             history=[{"judge": "yellow", "outside_changes": ["x"], "suggestions": ["y"]}])

    monkeypatch.setattr(drafting_v2, "propose", fake_propose)
    monkeypatch.setattr(api, "_search_client", lambda: None)
    return state


def test_suggest_fix_then_lawyer_edits(client, drafter):
    job = _review(client)
    assert client.post(f"/reviews/{job}/verdicts/governing_law/suggest-fix", headers=A).status_code == 409  # green
    fix = client.post(f"/reviews/{job}/verdicts/insurance/suggest-fix", headers=A).json()["fix_id"]
    f = client.get(f"/fixes/{fix}", headers=A).json()
    assert f["status"] == "ready" and f["verified"] is True
    assert f["redline"]["new_text"].startswith("Supplier shall maintain")
    assert f["suggestions"] == ["name Customer as additional insured"]
    assert client.get(f"/fixes/{fix}", headers=B).status_code == 404  # other tenant
    bad = {"action": "edit", "reason_code": "partial_fix"}
    assert client.post(f"/fixes/{fix}/decision", json=bad, headers=A).status_code == 422  # edit needs wording
    d = {"action": "edit", "final_text": "Supplier shall maintain insurance of at least $1,000,000.",
         "reason_code": "partial_fix", "comment": "needs a minimum limit"}
    r = client.post(f"/fixes/{fix}/decision", json=d, headers=A).json()
    assert r["status"] == "decided:edit" and 0 < r["lawyer_edit_ratio"] < 1
    assert client.post(f"/fixes/{fix}/decision", json=d, headers=A).status_code == 409  # decided once


def test_unverified_fix_cannot_be_accepted(client, drafter):
    drafter["verified"] = False
    job = _review(client)
    fix = client.post(f"/reviews/{job}/verdicts/insurance/suggest-fix", headers=A).json()["fix_id"]
    f = client.get(f"/fixes/{fix}", headers=A).json()
    assert f["verified"] is False and f["redline"] is None and "checker" in f["why_no_fix"]
    assert client.post(f"/fixes/{fix}/decision", json={"action": "accept", "reason_code": "ok"},
                       headers=A).status_code == 422


def test_completed_review_returns_lawyer_decisions(client):
    job = _review(client)
    d = {"decisions": {"insurance": {"action": "change_severity", "new_status": "yellow", "reason_code": "wrong_severity"}}}
    client.post(f"/reviews/{job}/decisions", json=d, headers=A)
    s = client.get(f"/reviews/{job}", headers=A).json()
    assert s["status"] == "completed" and s["decisions"]["insurance"]["new_status"] == "yellow"


def test_failures_are_shown_in_plain_words():
    msg = api._user_message(RuntimeError("Error code: 400 - Your credit balance is too low to access the Anthropic API"))
    assert "usage limit" in msg and "400" not in msg
    assert "logged" in api._user_message(KeyError("internal detail"))


def test_ui_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "<title>ClauseCheck</title>" in r.text


def test_unknown_rule_rejected(client):
    doc = upload(client)
    assert client.post("/reviews", json={"document_id": doc, "rules": ["made_up"]}, headers=A).status_code == 422
