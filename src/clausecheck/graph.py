"""Phase 6 step 1: the review WORKFLOW expressed as a LangGraph graph (E6.1 parity).

No agent decides anything here yet: the graph runs the same fixed path as the workflow, so its verdicts must
match the stored workflow runs. What the graph adds is structure: a checkpoint after every node (resume after a
crash without re-paying finished rules), a human-in-the-loop pause, and a verdict store.

    START → load_or_route ─(stored verdicts)──────────────────────────────→ human_review → persist → END
                          └─(new)→ warm_cache → judge_rule × (other rules, parallel) ─┘

- load_or_route: hash the contract; if verdicts for (hash, prompt version, model) exist, reuse them. Else pick
  the context strategy: full contract whenever it fits the model's context window, RAG top-10 only when it doesn't.
- warm_cache: judge the FIRST rule alone so it writes the contract into the prompt cache; the other rules then
  fan out in parallel and read it (sending all 8 at once would race and pay the cache write several times).
- human_review: interrupt(). The graph pauses; the reviewer's decisions come back via Command(resume=...).
  Every decision is stored in the feedback table (capture half of the feedback pipeline; routing is Phase 7).
- persist: write verdicts to the verdict store.

Local storage: SQLite (data/clausecheck.db for verdicts + feedback, data/checkpoints.db for LangGraph).
On AWS both become Postgres (RDS); the LangGraph checkpointer interface is the same.
"""

from __future__ import annotations

import hashlib
import json
import operator
import sqlite3
import time
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from clausecheck import search
from clausecheck.judge import PROMPT_VERSION, judge_consistent
from clausecheck.llm import DEFAULT_MODEL
from clausecheck.rules import load_rules
from clausecheck.segment import segment

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "clausecheck.db"
CHECKPOINTS = ROOT / "data" / "checkpoints.db"
# Decision 2026-11-12 (revised after the long-contract suite): full contract whenever it FITS, RAG only when it
# doesn't. Budget = the model's context window minus room for instructions, rubric, re-asks' output and the
# estimate's error. 200k is the context window assumed for Claude Haiku here: check the current model's limit.
MODEL_CONTEXT_TOKENS = 200_000
FULL_CONTEXT_MAX_TOKENS = int(MODEL_CONTEXT_TOKENS * 0.8)  # 160k: 20% margin for prompt, output, estimate error
RAG_K = 10
TOKENS_PER_CHAR = 1.35 / 4  # Phase 0 calibration: real tokens ≈ 1.35 × chars/4 on legal text


class ReviewState(TypedDict, total=False):
    doc_key: str  # search-index key for this document (case_id for eval cases)
    doc_id: str
    text: str
    client: str
    rule_ids: list[str]
    contract_hash: str
    est_tokens: int
    strategy: Literal["full", "rag"]
    reused: bool  # verdicts came from the verdict store, no model calls
    verdicts: Annotated[list[dict], operator.add]  # parallel judge nodes append here; accumulates across runs on
    # the SAME thread, so every review must use a fresh thread_id (one per job / call)
    review: dict
    # run settings (not contract data)
    ledger_tag: str  # API spend is booked under this tag
    prompt_cache: bool  # production call pattern; evals turn it off to reuse stored answers
    disk_cache: bool
    force_strategy: str  # tests only: "full" / "rag" regardless of length
    injection: list[str]  # kinds of AI-directed text found in the contract (Phase 7.4)


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS verdicts (contract_hash TEXT, prompt_version TEXT, model TEXT,
                   strategy TEXT, rule_id TEXT, verdict TEXT, created REAL,
                   PRIMARY KEY (contract_hash, prompt_version, model, rule_id))""")
    con.execute("""CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, contract_hash TEXT,
                   doc_key TEXT, rule_id TEXT, prompt_version TEXT, model TEXT, model_status TEXT,
                   needs_human INTEGER, reviewer TEXT, action TEXT, new_status TEXT, reason_code TEXT,
                   comment TEXT, created REAL)""")
    return con


# ---------------------------------------------------------------- nodes

def load_or_route(state: ReviewState) -> dict:
    h = hashlib.sha256(f"{state['client']}\n{state['text']}".encode()).hexdigest()[:24]
    with _db() as con:
        rows = con.execute("SELECT rule_id, verdict, strategy FROM verdicts WHERE contract_hash=? AND prompt_version=? "
                           "AND model=?", (h, PROMPT_VERSION, DEFAULT_MODEL)).fetchall()
    if state.get("force_strategy"):
        rows = [r for r in rows if r[2] == state["force_strategy"]]
    stored = {r[0]: json.loads(r[1]) for r in rows}
    if rows and all(rid in stored for rid in state["rule_ids"]):
        return {"contract_hash": h, "reused": True, "strategy": rows[0][2],
                "verdicts": [stored[rid] for rid in state["rule_ids"]]}
    est = int(len(state["text"]) * TOKENS_PER_CHAR)
    strategy = state.get("force_strategy") or ("full" if est <= FULL_CONTEXT_MAX_TOKENS else "rag")
    from clausecheck.injection import scan
    kinds = sorted({h["kind"] for h in scan(state["text"])})
    return {"contract_hash": h, "reused": False, "est_tokens": est, "strategy": strategy, "injection": kinds}


def _inject_failure(rule_id: str) -> None:
    """E6.4 test hook: CLAUSECHECK_FAIL_ONCE=<rule_id> makes that rule's node crash the first time it runs
    (simulates a worker dying mid-review). A marker file makes it succeed on the resume."""
    import os

    if os.environ.get("CLAUSECHECK_FAIL_ONCE") != rule_id:
        return
    marker = ROOT / "data" / f".failed_once_{rule_id}"
    if not marker.exists():
        marker.touch()
        raise RuntimeError(f"injected failure while judging {rule_id}")


def _judge_one(state: ReviewState, rule_id: str) -> dict:
    from clausecheck.llm import AnthropicAdapter

    _inject_failure(rule_id)
    rule = next(r for r in load_rules() if r.id == rule_id)
    clauses = segment(state["doc_id"], state["text"])
    full = state["strategy"] == "full"
    shown = clauses
    if not full:
        es = search.client()
        if not es.count(index=search.INDEX, query={"term": {"doc_key": state["doc_key"]}})["count"]:
            search.index_document(es, state["doc_key"], state["doc_id"], clauses)
            es.indices.refresh(index=search.INDEX)
        hits = {sid for sid, _ in search.hybrid(es, state["doc_key"], search.rule_query(rule), RAG_K)}
        shown = [c for c in clauses if c.id in hits]
    r = judge_consistent(AnthropicAdapter(tag=state.get("ledger_tag", "phase6-graph")), shown, rule, state["client"],
                         cache_contract=full and state.get("prompt_cache", True), excerpts_only=not full,
                         disk_cache=state.get("disk_cache", True))
    return {"rule_id": rule_id, "status": r.verdict.status.value,
            "red_reason": r.verdict.red_reason.value if r.verdict.red_reason else None,
            "cited": r.verdict.cited_clause_ids, "quote": r.llm.get("quote", ""), "rationale": r.verdict.rationale,
            "confidence": r.verdict.confidence, "citation_valid": r.citation_valid,
            "needs_human": bool(r.consistency.get("needs_human")) or not r.citation_valid,
            "why_flagged": ("the model gave different answers when asked again" if r.consistency.get("needs_human")
                            else "the cited text couldn't be found in the contract" if not r.citation_valid else ""),
            "samples": r.consistency.get("samples"), "cost_usd": r.llm["cost_usd"]}


def _with_injection_flag(state: ReviewState, v: dict) -> dict:
    # The judge resisted every planted attack (Phase 7.4), but a contract that talks to "any AI reviewer" is itself
    # something the lawyer must see: flag the verdict and say why.
    if state.get("injection"):
        v = {**v, "needs_human": True,
             "why_flagged": "the contract contains text addressed to an AI reviewer (" + ", ".join(state["injection"])
                            + ")" + (f"; also: {v['why_flagged']}" if v.get("why_flagged") else "")}
    return v


def warm_cache(state: ReviewState) -> dict:
    return {"verdicts": [_with_injection_flag(state, _judge_one(state, state["rule_ids"][0]))]}


def judge_rule(payload: dict) -> dict:
    return {"verdicts": [_with_injection_flag(payload["state"], _judge_one(payload["state"], payload["rule_id"]))]}


def fan_out(state: ReviewState):
    rest = state["rule_ids"][1:]
    if not rest:
        return "human_review"
    return [Send("judge_rule", {"state": state, "rule_id": rid}) for rid in rest]


def after_route(state: ReviewState) -> str:
    return "human_review" if state["reused"] else "warm_cache"


def human_review(state: ReviewState) -> dict:
    """Pause for the lawyer. Resume value: {"reviewer": str, "decisions": {rule_id: {action, new_status,
    reason_code, comment}}}; rules without a decision count as not yet reviewed (nothing stored)."""
    answer = interrupt({"contract_hash": state["contract_hash"], "strategy": state["strategy"],
                        "verdicts": state["verdicts"],
                        "needs_closer_look": [v["rule_id"] for v in state["verdicts"] if v["needs_human"]]})
    by_rule = {v["rule_id"]: v for v in state["verdicts"]}
    with _db() as con:
        for rid, d in answer.get("decisions", {}).items():
            v = by_rule[rid]
            con.execute("INSERT INTO feedback (contract_hash, doc_key, rule_id, prompt_version, model, model_status, "
                        "needs_human, reviewer, action, new_status, reason_code, comment, created) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (state["contract_hash"], state["doc_key"], rid, PROMPT_VERSION, DEFAULT_MODEL, v["status"],
                         int(v["needs_human"]), answer.get("reviewer", "unknown"), d.get("action"),
                         d.get("new_status"), d.get("reason_code"), d.get("comment", ""), time.time()))
    return {"review": answer}


def persist(state: ReviewState) -> dict:
    # forced-strategy runs are tests: they must never overwrite production verdicts in the store
    # (found when a forced-RAG parity run replaced 42 full-contract verdicts)
    if not state["reused"] and not state.get("force_strategy"):
        with _db() as con:
            for v in state["verdicts"]:
                con.execute("INSERT OR REPLACE INTO verdicts VALUES (?,?,?,?,?,?,?)",
                            (state["contract_hash"], PROMPT_VERSION, DEFAULT_MODEL, state["strategy"], v["rule_id"],
                             json.dumps(v), time.time()))
    return {}


def build(checkpointer=None):
    g = StateGraph(ReviewState)
    from clausecheck.telemetry import node

    def verdict_attrs(_inp, out):  # metadata only: never rationale, quote or contract text in a span
        v = out["verdicts"][0]
        return {"rule_id": v["rule_id"], "status": v["status"], "needs_human": v["needs_human"],
                "rule_cost_usd": round(v.get("cost_usd", 0.0), 6)}

    g.add_node("load_or_route", node("load_or_route", lambda i, o: {
        "strategy": o.get("strategy"), "reused": o.get("reused"), "est_tokens": o.get("est_tokens"),
        "rules": len(i["rule_ids"])})(load_or_route))
    g.add_node("warm_cache", node("judge_rule", verdict_attrs)(warm_cache))
    g.add_node("judge_rule", node("judge_rule", verdict_attrs)(judge_rule))
    g.add_node("human_review", human_review)  # not traced: interrupt() pauses by raising, which a span reads as an error
    g.add_node("persist", node("persist")(persist))
    g.add_edge(START, "load_or_route")
    g.add_conditional_edges("load_or_route", after_route, ["warm_cache", "human_review"])
    g.add_conditional_edges("warm_cache", fan_out, ["judge_rule", "human_review"])
    g.add_edge("judge_rule", "human_review")
    g.add_edge("human_review", "persist")
    g.add_edge("persist", END)
    return g.compile(checkpointer=checkpointer)
