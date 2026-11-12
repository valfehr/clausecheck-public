"""E6.5 check: start the ClauseCheck MCP server as a real MCP client would (stdio subprocess) and call every tool.

  uv run python scripts/mcp_smoke.py

Uses the contract reviewed in E6.4 so check_contract is served from the verdict store ($0).
Also checks the refusals: a path outside data/inbox/ and explain_verdict before a review.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp.client.client import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]
DOC = "GOLDRESOURCECORP_12_11_2008"


def text_of(result) -> str:
    return "\n".join(getattr(c, "text", "") for c in result.content)


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "clausecheck.mcp_server"], cwd=str(ROOT),
                                   env={"PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8",
                                        "HF_HUB_OFFLINE": "1"})  # models are already in models/; no network needed
    async with Client(params, read_timeout_seconds=300) as c:
        tools = await c.list_tools()
        print("tools:", [t.name for t in tools.tools])
        r = await c.call_tool("list_rules", {})
        print("list_rules:", len(json.loads(text_of(r)) if text_of(r).startswith("[") else r.structured_content["result"]), "rules")
        r = await c.call_tool("check_contract", {"document": DOC})
        out = r.structured_content or json.loads(text_of(r))
        print(f"check_contract: served_from_storage={out['served_from_storage']} cost=${out['cost_usd']} "
              f"statuses={ {v['rule_id']: v['status'] for v in out['verdicts']} } closer_look={out['needs_closer_look']}")
        pick = next(v["rule_id"] for v in out["verdicts"] if v["cited_segments"])  # a verdict with a citation
        r = await c.call_tool("explain_verdict", {"document": DOC, "rule_id": pick})
        ex = r.structured_content or json.loads(text_of(r))
        print(f"explain_verdict({pick}): quote={ex['quote'][:120]!r} cited={list(ex['cited_segments'])}")
        r = await c.call_tool("find_clause", {"document": DOC, "query": "insurance coverage limits", "k": 3})
        fc = r.structured_content["result"] if r.structured_content else json.loads(text_of(r))
        print("find_clause:", [(h["segment_id"], h["text"][:70]) for h in fc])
        r = await c.call_tool("find_clause", {"document": "..\\..\\.env", "query": "key"})
        print("refusal (path outside inbox):", r.is_error, text_of(r)[:120])
        r = await c.call_tool("explain_verdict", {"document": DOC, "rule_id": "insurance", "client": "Someone Else"})
        print("refusal (not reviewed):", r.is_error, text_of(r)[:120])


if __name__ == "__main__":
    asyncio.run(main())
