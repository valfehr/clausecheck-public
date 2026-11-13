"""Phase 7.1: the review service as an HTTP API (FastAPI), async job pattern, per-tenant access control.

    POST /documents                      upload a contract (text) → document_id
    POST /reviews                        start a review job (Idempotency-Key header) → job_id, status "queued"
    GET  /reviews/{job_id}               status: queued → running → awaiting_review (verdicts ready) → completed
    POST /reviews/{job_id}/decisions     the lawyer's decision per rule → resumes the graph, stores feedback
    GET  /reviews/{job_id}/verdicts/{rule_id}   rationale, quote, citations, why flagged
    GET  /health

Local stand-ins for the AWS design (docs/aws-topology.md): an in-process thread pool for SQS + worker, SQLite for
RDS Postgres, data/inbox/<tenant>/ for S3. Auth: one API key per user (stored hashed), each user in one tenant.
Every read is filtered by tenant; another tenant's document or job answers 404, not 403, so ids can't be probed.

  uv run uvicorn clausecheck.api:app --port 8090
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from clausecheck import graph
from clausecheck.rules import load_rules

app = FastAPI(title="ClauseCheck", version="0.7")
_pool = ThreadPoolExecutor(2)  # stand-in for SQS + autoscaled workers
ACTIONS = ("approve", "change_severity", "wrong_passage", "not_applicable")
REASONS = ("wrong_severity", "wrong_passage", "rule_gap", "approved", "other")


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(graph.DB, timeout=30)
    con.execute("CREATE TABLE IF NOT EXISTS users (key_hash TEXT PRIMARY KEY, user_id TEXT, tenant TEXT)")
    con.execute("""CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, tenant TEXT, owner TEXT, name TEXT,
                   path TEXT, chars INTEGER, created REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, tenant TEXT, user_id TEXT, doc_id TEXT,
                   client TEXT, rule_ids TEXT, status TEXT, idem_key TEXT, result TEXT, error TEXT, created REAL,
                   updated REAL, UNIQUE (user_id, idem_key))""")
    return con


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def add_user(user_id: str, tenant: str, api_key: str) -> None:
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO users VALUES (?,?,?)", (_hash(api_key), user_id, tenant))


class User(BaseModel):
    user_id: str
    tenant: str


def current_user(x_api_key: str = Header(default="")) -> User:
    with _db() as con:
        row = con.execute("SELECT user_id, tenant FROM users WHERE key_hash=?", (_hash(x_api_key),)).fetchone()
    if not x_api_key or not row:
        raise HTTPException(401, "missing or invalid X-API-Key")
    return User(user_id=row[0], tenant=row[1])


# ---------------------------------------------------------------- documents

class DocumentIn(BaseModel):
    name: str = Field(max_length=200)
    text: str = Field(min_length=200, max_length=2_000_000)


@app.post("/documents", status_code=201)
def upload(doc: DocumentIn, user: User = Depends(current_user)) -> dict:
    doc_id = hashlib.sha256(f"{user.tenant}\n{doc.text}".encode()).hexdigest()[:20]
    folder = graph.ROOT / "data" / "inbox" / user.tenant
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{doc_id}.txt"
    path.write_text(doc.text, encoding="utf-8")
    with _db() as con:
        con.execute("INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?,?)",
                    (doc_id, user.tenant, user.user_id, doc.name, str(path), len(doc.text), time.time()))
    return {"document_id": doc_id, "chars": len(doc.text)}


def _document(doc_id: str, user: User) -> tuple[str, str]:
    with _db() as con:
        row = con.execute("SELECT path, name FROM documents WHERE doc_id=? AND tenant=?", (doc_id, user.tenant)).fetchone()
    if not row:
        raise HTTPException(404, "document not found")  # same answer whether it doesn't exist or isn't yours
    return row[0], row[1]


# ---------------------------------------------------------------- reviews

class ReviewIn(BaseModel):
    document_id: str
    client: str = Field(default="the customer / recipient of goods or services", max_length=200)
    rules: list[str] | None = None


def _set(job_id: str, **fields) -> None:
    sets = ", ".join(f"{k}=?" for k in fields) + ", updated=?"
    with _db() as con:
        con.execute(f"UPDATE jobs SET {sets} WHERE job_id=?", (*fields.values(), time.time(), job_id))


def _graph_config(job_id: str) -> dict:
    return {"configurable": {"thread_id": f"api-{job_id}"}}


def _run(job_id: str, payload: dict | None, resume: dict | None) -> None:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    from clausecheck.telemetry import tracer

    try:
        _set(job_id, status="running")
        # one root span per job step: the review run, and later the resume after the lawyer's decisions
        with tracer().start_as_current_span("review.resume" if resume else "review.run") as span, \
                SqliteSaver.from_conn_string(str(graph.CHECKPOINTS)) as cp:
            span.set_attribute("clausecheck.job_id", job_id)
            span.set_attribute("clausecheck.rules", len((payload or {}).get("rule_ids", [])))
            g = graph.build(cp)
            out = g.invoke(Command(resume=resume) if resume else payload, _graph_config(job_id))
            span.set_attribute("clausecheck.outcome", "awaiting_review" if "__interrupt__" in out else "completed")
        if "__interrupt__" in out:
            pending = out["__interrupt__"][0].value
            _set(job_id, status="awaiting_review", result=json.dumps({
                "strategy": pending["strategy"], "verdicts": pending["verdicts"],
                # the reason travels with the flag: the UI shows *why* a verdict needs a closer look
                "needs_closer_look": [{"rule_id": v["rule_id"], "why": v["why_flagged"]}
                                      for v in pending["verdicts"] if v["needs_human"]],
                "served_from_storage": bool(out.get("reused"))}))
        else:
            res = json.loads(_get_job(job_id)["result"] or "{}")
            res["review"] = out.get("review")
            _set(job_id, status="completed", result=json.dumps(res))
    except Exception as e:  # noqa: BLE001 — the job records the failure; the API never crashes on a job
        _set(job_id, status="failed", error=_user_message(e))


def _submit(job_id: str, payload: dict | None = None, resume: dict | None = None) -> None:
    if os.environ.get("CLAUSECHECK_SYNC_JOBS") == "1":  # tests: run inline, deterministic
        _run(job_id, payload, resume)
    else:
        _pool.submit(_run, job_id, payload, resume)


def _get_job(job_id: str, user: User | None = None) -> dict:
    q, args = "SELECT * FROM jobs WHERE job_id=?", [job_id]
    if user:
        q, args = q + " AND tenant=?", [job_id, user.tenant]
    with _db() as con:
        con.row_factory = sqlite3.Row
        row = con.execute(q, args).fetchone()
    if not row:
        raise HTTPException(404, "review not found")
    return dict(row)


@app.post("/reviews", status_code=202)
def start_review(body: ReviewIn, user: User = Depends(current_user),
                 idempotency_key: str = Header(default="")) -> dict:
    path, name = _document(body.document_id, user)
    valid = [r.id for r in load_rules()]
    rule_ids = body.rules or valid
    if bad := [r for r in rule_ids if r not in valid]:
        raise HTTPException(422, f"unknown rule ids {bad}")
    if idempotency_key:
        with _db() as con:
            row = con.execute("SELECT job_id, status FROM jobs WHERE user_id=? AND idem_key=?",
                              (user.user_id, idempotency_key)).fetchone()
        if row:
            return {"job_id": row[0], "status": row[1], "duplicate": True}
    job_id = uuid.uuid4().hex[:16]
    with _db() as con:
        con.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (job_id, user.tenant, user.user_id, body.document_id, body.client, json.dumps(rule_ids), "queued",
                     idempotency_key or None, None, None, time.time(), time.time()))
    text = open(path, encoding="utf-8").read()
    _submit(job_id, payload={"doc_key": f"api::{user.tenant}::{body.document_id}", "doc_id": body.document_id,
                             "text": text, "client": body.client, "rule_ids": rule_ids, "verdicts": [],
                             "ledger_tag": "phase7-api"})
    return {"job_id": job_id, "status": "queued", "duplicate": False}


@app.get("/reviews/{job_id}")
def get_review(job_id: str, user: User = Depends(current_user)) -> dict:
    j = _get_job(job_id, user)
    res = json.loads(j["result"] or "{}")
    return {"job_id": job_id, "status": j["status"], "document_id": j["doc_id"], "client": j["client"],
            "error": j["error"], **({"verdicts": [{k: v[k] for k in ("rule_id", "status", "red_reason", "cited",
                                                                     "needs_human", "why_flagged")}
                                                   for v in res.get("verdicts", [])],
                                     "needs_closer_look": res.get("needs_closer_look", []),
                                     "context": res.get("strategy"),
                                     "served_from_storage": res.get("served_from_storage"),
                                     # the lawyer's decisions, once submitted: the UI shows these over the AI's verdicts
                                     "decisions": (res.get("review") or {}).get("decisions", {})} if res else {}),
            "note": "First-pass verdicts; a lawyer validates each one."}


class Decision(BaseModel):
    action: Literal[ACTIONS]  # type: ignore[valid-type]
    new_status: Literal["green", "yellow", "red"] | None = None
    reason_code: Literal[REASONS]  # type: ignore[valid-type]
    comment: str = Field(default="", max_length=2000)


class DecisionsIn(BaseModel):
    decisions: dict[str, Decision]


@app.post("/reviews/{job_id}/decisions", status_code=202)
def decide(job_id: str, body: DecisionsIn, user: User = Depends(current_user)) -> dict:
    j = _get_job(job_id, user)
    if j["status"] != "awaiting_review":
        raise HTTPException(409, f"review is {j['status']}, not awaiting_review")
    rules = set(json.loads(j["rule_ids"]))
    if bad := [r for r in body.decisions if r not in rules]:
        raise HTTPException(422, f"rules not in this review: {bad}")
    for rid, d in body.decisions.items():
        if d.action == "change_severity" and not d.new_status:
            raise HTTPException(422, f"{rid}: change_severity needs new_status")
    _set(job_id, status="resuming")
    _submit(job_id, resume={"reviewer": user.user_id,
                            "decisions": {k: v.model_dump() for k, v in body.decisions.items()}})
    return {"job_id": job_id, "status": "resuming"}


@app.get("/reviews/{job_id}/verdicts/{rule_id}")
def explain(job_id: str, rule_id: str, user: User = Depends(current_user)) -> dict:
    from clausecheck.segment import segment

    j = _get_job(job_id, user)
    res = json.loads(j["result"] or "{}")
    v = next((v for v in res.get("verdicts", []) if v["rule_id"] == rule_id), None)
    if not v:
        raise HTTPException(404, "no verdict for that rule yet")
    path, _ = _document(j["doc_id"], user)
    by_id = {c.id: c.text for c in segment(j["doc_id"], open(path, encoding="utf-8").read())}
    return {**{k: v.get(k) for k in ("rule_id", "status", "red_reason", "rationale", "quote", "cited", "samples",
                                      "needs_human", "why_flagged", "confidence")},
            "cited_text": {i: by_id.get(i, "") for i in v.get("cited", [])}}


@app.get("/rules")
def rules(user: User = Depends(current_user)) -> list[dict]:
    return [{"id": r.id, "title": r.title, "green": r.rubric.green, "yellow": r.rubric.yellow, "red": r.rubric.red}
            for r in load_rules()]


@app.get("/reviews")
def list_reviews(user: User = Depends(current_user)) -> list[dict]:
    with _db() as con:
        rows = con.execute("SELECT j.job_id, j.status, j.client, j.created, d.name, j.doc_id FROM jobs j "
                           "JOIN documents d ON d.doc_id = j.doc_id AND d.tenant = j.tenant WHERE j.tenant=? "
                           "ORDER BY j.created DESC LIMIT 50", (user.tenant,)).fetchall()
    return [{"job_id": r[0], "status": r[1], "client": r[2], "created": r[3], "document": r[4], "document_id": r[5]}
            for r in rows]


# ---------------------------------------------------------------- suggest a fix (drafting agent, Phase 6 E6.3b)

FIX_REASONS = ("ok", "over_reach", "partial_fix", "removed_protection", "wrong_party", "style", "other")


def _fix_db() -> sqlite3.Connection:
    con = _db()
    con.execute("""CREATE TABLE IF NOT EXISTS fixes (fix_id TEXT PRIMARY KEY, job_id TEXT, tenant TEXT, rule_id TEXT,
                   status TEXT, result TEXT, error TEXT, created REAL, updated REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS redline_feedback (fix_id TEXT PRIMARY KEY, tenant TEXT, reviewer TEXT,
                   rule_id TEXT, contract_hash TEXT, proposed TEXT, final TEXT, action TEXT, reason_code TEXT,
                   comment TEXT, edit_ratio REAL, created REAL)""")
    return con


def _search_client():
    from clausecheck import search
    return search.client()


def _run_fix(fix_id: str, job: dict, rule_id: str) -> None:
    from clausecheck import drafting_v2, search
    from clausecheck.llm import AnthropicAdapter
    from clausecheck.schemas import Status, Verdict
    from clausecheck.segment import segment
    from clausecheck.telemetry import tracer

    def upd(**f):
        sets = ", ".join(f"{k}=?" for k in f) + ", updated=?"
        with _fix_db() as con:
            con.execute(f"UPDATE fixes SET {sets} WHERE fix_id=?", (*f.values(), time.time(), fix_id))

    try:
        upd(status="drafting")
        res = json.loads(job["result"])
        v = next(x for x in res["verdicts"] if x["rule_id"] == rule_id)
        with _db() as con:
            path = con.execute("SELECT path FROM documents WHERE doc_id=?", (job["doc_id"],)).fetchone()[0]
        text = open(path, encoding="utf-8").read()
        clauses = segment(job["doc_id"], text)
        doc_key = f"api::{job['tenant']}::{job['doc_id']}"
        es = _search_client()
        if es is not None and not es.count(index=search.INDEX, query={"term": {"doc_key": doc_key}})["count"]:
            search.index_document(es, doc_key, job["doc_id"], clauses)
            es.indices.refresh(index=search.INDEX)
        verdict = Verdict(rule_id=rule_id, status=Status(v["status"]), rationale=v["rationale"],
                          confidence=v["confidence"], cited_clause_ids=v["cited"])
        rule = next(r for r in load_rules() if r.id == rule_id)
        with tracer().start_as_current_span("draft.suggest_fix") as span:
            out = drafting_v2.propose(AnthropicAdapter(tag="phase7-fix"), es, doc_key, clauses, rule, job["client"],
                                      verdict)
            span.set_attribute("clausecheck.rule_id", rule_id)
            span.set_attribute("clausecheck.rounds", out.rounds)
            span.set_attribute("clausecheck.accepted", out.accepted)
            span.set_attribute("clausecheck.cost_usd", round(out.cost_usd, 6))
        last = out.history[-1] if out.history else {}
        by_id = {c.id: c.text for c in clauses}
        red = out.redline.model_dump() if out.redline else None
        upd(status="ready", result=json.dumps({
            "verified": out.accepted, "rounds": out.rounds, "cost_usd": round(out.cost_usd, 4),
            "redline": red, "original_text": by_id.get(red["segment_id"], "") if red else None,
            "suggestions": last.get("suggestions", []),
            "why_no_fix": None if out.accepted else (
                f"after {out.rounds} drafts the checker still found problems: reviewer rated it "
                f"{last.get('judge')}; changes outside the rule: {last.get('outside_changes') or last.get('added_outside') or 'none'}")}))
    except Exception as e:  # noqa: BLE001
        upd(status="failed", error=_user_message(e))


def _user_message(e: Exception) -> str:
    """What a lawyer sees when a job fails: plain words, no provider internals (those go to the server log)."""
    import logging

    logging.getLogger("clausecheck").warning("job failed: %s: %s", type(e).__name__, str(e)[:500])
    text = str(e).lower()
    if "credit balance" in text or type(e).__name__ == "BudgetExceeded":
        return "The AI service's usage limit has been reached, so nothing new can be drafted right now. Try again later."
    if "rate" in text and "limit" in text or "overloaded" in text:
        return "The AI service is busy. Try again in a minute."
    return "Something went wrong on our side; the error has been logged."


@app.post("/reviews/{job_id}/verdicts/{rule_id}/suggest-fix", status_code=202)
def suggest_fix(job_id: str, rule_id: str, user: User = Depends(current_user)) -> dict:
    """The AI drafts a redline (Haiku drafts, Sonnet checks, up to 3 rounds, ~$0.05-0.15). Nothing is applied: the
    lawyer accepts, edits or rejects it via POST /fixes/{fix_id}/decision."""
    job = _get_job(job_id, user)
    res = json.loads(job["result"] or "{}")
    v = next((x for x in res.get("verdicts", []) if x["rule_id"] == rule_id), None)
    if not v:
        raise HTTPException(404, "no verdict for that rule yet")
    if v["status"] == "green":
        raise HTTPException(409, "this verdict is green: nothing to fix")
    fix_id = uuid.uuid4().hex[:16]
    with _fix_db() as con:
        con.execute("INSERT INTO fixes VALUES (?,?,?,?,?,?,?,?,?)",
                    (fix_id, job_id, user.tenant, rule_id, "queued", None, None, time.time(), time.time()))
    if os.environ.get("CLAUSECHECK_SYNC_JOBS") == "1":
        _run_fix(fix_id, job, rule_id)
    else:
        _pool.submit(_run_fix, fix_id, job, rule_id)
    return {"fix_id": fix_id, "status": "queued"}


def _get_fix(fix_id: str, user: User) -> dict:
    with _fix_db() as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM fixes WHERE fix_id=? AND tenant=?", (fix_id, user.tenant)).fetchone()
    if not row:
        raise HTTPException(404, "fix not found")
    return dict(row)


@app.get("/fixes/{fix_id}")
def get_fix(fix_id: str, user: User = Depends(current_user)) -> dict:
    f = _get_fix(fix_id, user)
    return {"fix_id": fix_id, "status": f["status"], "rule_id": f["rule_id"], "error": f["error"],
            **(json.loads(f["result"]) if f["result"] else {}),
            "note": "A proposal only: nothing is applied to the contract until a lawyer accepts it."}


class FixDecision(BaseModel):
    action: Literal["accept", "edit", "reject"]
    final_text: str | None = Field(default=None, max_length=20000)  # required for "edit": the lawyer's wording
    reason_code: Literal[FIX_REASONS]  # type: ignore[valid-type]
    comment: str = Field(default="", max_length=2000)


@app.post("/fixes/{fix_id}/decision")
def decide_fix(fix_id: str, body: FixDecision, user: User = Depends(current_user)) -> dict:
    import difflib

    f = _get_fix(fix_id, user)
    if f["status"] != "ready":
        raise HTTPException(409, f"fix is {f['status']}")
    res = json.loads(f["result"])
    proposed = (res.get("redline") or {}).get("new_text") or ""
    if body.action == "edit" and not body.final_text:
        raise HTTPException(422, "edit needs final_text (the lawyer's wording)")
    if body.action == "accept" and not proposed:
        raise HTTPException(422, "there is no verified redline to accept; edit or reject instead")
    final = body.final_text if body.action == "edit" else proposed if body.action == "accept" else None
    # how much the lawyer had to change: 0 = accepted as is, 1 = rewrote everything (product metric, PLAN.md Phase 7)
    ratio = 0.0 if body.action == "accept" else 1.0 if body.action == "reject" else \
        round(1 - difflib.SequenceMatcher(None, proposed, final or "").ratio(), 3)
    job = _get_job(f["job_id"], user)
    with _fix_db() as con:
        con.execute("INSERT OR REPLACE INTO redline_feedback VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (fix_id, user.tenant, user.user_id, f["rule_id"], job["doc_id"], proposed, final, body.action,
                     body.reason_code, body.comment, ratio, time.time()))
        con.execute("UPDATE fixes SET status=?, updated=? WHERE fix_id=?", (f"decided:{body.action}", time.time(), fix_id))
    return {"fix_id": fix_id, "status": f"decided:{body.action}", "lawyer_edit_ratio": ratio}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/", include_in_schema=False)
def ui():
    """The web UI: one static page calling this API (no build step). Same file goes to S3 + CloudFront on AWS."""
    from pathlib import Path

    from fastapi.responses import FileResponse

    return FileResponse(Path(__file__).parent / "ui" / "index.html")
