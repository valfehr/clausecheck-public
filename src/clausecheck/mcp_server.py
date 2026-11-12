"""E6.5: ClauseCheck as an MCP server (stdio). Any MCP client (Claude Code, Claude Desktop) can call it.

Tools (descriptions are written for a model reader: what it does, what it costs, what it returns, what it won't do):
- list_rules              the 8 golden rules (ids for the other tools)
- check_contract          run the review graph on one contract; returns verdicts + which need a closer look
- find_clause             search one contract's segments (local, $0)
- explain_verdict         the stored reasoning, quote and citations behind one verdict ($0, no model call)

Least privilege: documents come only from the CUAD corpus (by id) or from files inside data/inbox/;
paths outside it are refused. Contract text is data, never instructions, in every prompt behind these tools.
Verdicts are a first pass: the response says that a lawyer must validate each one.

  uv run python -m clausecheck.mcp_server          (stdio; registered in a client, not run by hand)
"""

from __future__ import annotations

import hashlib
import json
import uuid
import sqlite3
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError  # message reaches the caller; other exceptions stay hidden

from clausecheck import search
from clausecheck.graph import CHECKPOINTS, DB, ROOT, build
from clausecheck.judge import PROMPT_VERSION
from clausecheck.llm import DEFAULT_MODEL
from clausecheck.rules import load_rules
from clausecheck.segment import segment

INBOX = ROOT / "data" / "inbox"
DEFAULT_CLIENT = "the customer / recipient of goods or services"

server = MCPServer(
    name="clausecheck",
    instructions=("Contract review against 8 golden rules (green / yellow / red, with citations). Start with "
                  "list_rules; check_contract costs about $0.01 per contract and is cached per contract version; "
                  "find_clause and explain_verdict are free. Every verdict is a first pass that a lawyer validates."),
)


def _resolve(document: str) -> tuple[str, str, str]:
    """(doc_key, doc_id, text) for a CUAD id (or unique id fragment) or a file name inside data/inbox/."""
    candidate = (INBOX / document).resolve()
    if any(s in document for s in ("/", "\\", "..")) and INBOX.resolve() not in candidate.parents:
        # check containment BEFORE touching the file system, so the refusal reveals nothing about other paths
        raise ToolError("documents must be a CUAD contract id or a file name inside data/inbox/")
    if candidate.is_file() and INBOX.resolve() in candidate.parents:
        text = candidate.read_text(encoding="utf-8", errors="replace")
        return f"file::{hashlib.sha256(text.encode()).hexdigest()[:16]}", candidate.stem, text
    if any(s in document for s in ("/", "\\", "..")):
        raise ToolError("unknown document; give a CUAD contract id or a file name inside data/inbox/")
    from clausecheck.cuad import load_contracts

    hits = [c for c in load_contracts() if document.lower() in c.doc_id.lower()]
    if not hits:
        raise ToolError(f"no contract matches '{document}'")
    if len(hits) > 1 and not any(c.doc_id == document for c in hits):
        raise ToolError(f"'{document}' matches {len(hits)} contracts, e.g. {[c.doc_id for c in hits[:3]]}; be more specific")
    c = next((c for c in hits if c.doc_id == document), hits[0])
    return f"cuad::{c.doc_id}", c.doc_id, c.text


@server.tool(description="List the golden rules: id, title, and what green / yellow / red mean. Free.")
def list_rules() -> list[dict]:
    return [{"id": r.id, "title": r.title, "green": r.rubric.green, "yellow": r.rubric.yellow, "red": r.rubric.red}
            for r in load_rules()]


@server.tool(description=(
    "Review one contract against the golden rules on behalf of `client` (the party we represent; default: the "
    "customer / recipient). `document` = a CUAD contract id (a unique fragment is enough) or a file name in "
    "data/inbox/. `rules` = rule ids (default: all 8). Costs ~$0.01 per contract with Claude Haiku; a contract "
    "already reviewed with the same prompt/model version is served from storage for $0. Returns each verdict "
    "(status, cited segment ids, short rationale) and the list that needs a closer look, with the reason. "
    "These are first-pass verdicts for a lawyer to validate, not legal advice."))
def check_contract(document: str, client: str = DEFAULT_CLIENT, rules: list[str] | None = None) -> dict:
    doc_key, doc_id, text = _resolve(document)
    valid = [r.id for r in load_rules()]
    rule_ids = rules or valid
    unknown = [r for r in rule_ids if r not in valid]
    if unknown:
        raise ToolError(f"unknown rule ids {unknown}; see list_rules")
    from langgraph.checkpoint.sqlite import SqliteSaver

    thread = hashlib.sha256(f"{doc_key}|{client}|{','.join(rule_ids)}|{PROMPT_VERSION}".encode()).hexdigest()[:20]
    with SqliteSaver.from_conn_string(str(CHECKPOINTS)) as cp:
        out = build(cp).invoke({"doc_key": doc_key, "doc_id": doc_id, "text": text, "client": client,
                                "rule_ids": rule_ids, "verdicts": [], "ledger_tag": "phase6-mcp"},
                               # one thread per CALL: the graph's verdict list accumulates within a thread, so reusing
                               # a thread for the same contract returned old + new verdicts (found in Phase 7.4)
                               {"configurable": {"thread_id": f"mcp-{thread}-{uuid.uuid4().hex[:8]}"}})
    vs = out.get("verdicts", [])
    return {
        "document": doc_id, "client": client, "context": out.get("strategy"),
        "served_from_storage": bool(out.get("reused")),
        "cost_usd": round(sum(v.get("cost_usd", 0) for v in vs), 4) if not out.get("reused") else 0.0,
        "verdicts": [{"rule_id": v["rule_id"], "status": v["status"], "red_reason": v["red_reason"],
                      "cited_segments": v["cited"], "rationale": v["rationale"][:400],
                      "needs_closer_look": v["needs_human"], "why": v["why_flagged"]} for v in vs],
        "needs_closer_look": [{"rule_id": v["rule_id"], "why": v["why_flagged"]} for v in vs if v["needs_human"]],
        "note": "First-pass verdicts; a lawyer validates each one before it is relied on.",
    }


@server.tool(description=(
    "Search one contract for passages about a topic (keyword + semantic search, local, free). `document` as in "
    "check_contract. Returns up to `k` segments: id, section text (truncated to 600 chars). Use it to show a "
    "clause or check what the contract says; it does not judge anything."))
def find_clause(document: str, query: str, k: int = 5) -> list[dict]:
    doc_key, doc_id, text = _resolve(document)
    clauses = segment(doc_id, text)
    es = search.client()
    if not es.count(index=search.INDEX, query={"term": {"doc_key": doc_key}})["count"]:
        search.index_document(es, doc_key, doc_id, clauses)
        es.indices.refresh(index=search.INDEX)
    by_id = {c.id: c for c in clauses}
    hits = search.hybrid(es, doc_key, query[:300], max(1, min(k, 10)))
    return [{"segment_id": s, "text": " ".join(by_id[s].text.split())[:600]} for s, _ in hits if s in by_id]


@server.tool(description=(
    "Explain a verdict already produced by check_contract: the rationale, the exact quote from the contract, the "
    "cited segment ids, the re-ask samples, and why it was flagged (if it was). Free: reads storage, no model call. "
    "Fails if the contract/rule hasn't been checked yet."))
def explain_verdict(document: str, rule_id: str, client: str = DEFAULT_CLIENT) -> dict:
    doc_key, doc_id, text = _resolve(document)
    h = hashlib.sha256(f"{client}\n{text}".encode()).hexdigest()[:24]
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT verdict, strategy, created FROM verdicts WHERE contract_hash=? AND prompt_version=? "
                          "AND model=? AND rule_id=?", (h, PROMPT_VERSION, DEFAULT_MODEL, rule_id)).fetchone()
    if not row:
        raise ToolError(f"no stored verdict for {rule_id} on this contract/client; run check_contract first")
    v = json.loads(row[0])
    by_id = {c.id: c for c in segment(doc_id, text)}
    return {"rule_id": rule_id, "status": v["status"], "red_reason": v["red_reason"], "rationale": v["rationale"],
            "quote": v["quote"], "cited_segments": {i: " ".join(by_id[i].text.split())[:500] for i in v["cited"] if i in by_id},
            "re_ask_samples": v.get("samples"), "needs_closer_look": v["needs_human"], "why": v["why_flagged"],
            "context": row[1], "prompt_version": PROMPT_VERSION, "model": DEFAULT_MODEL}


if __name__ == "__main__":
    server.run("stdio")
