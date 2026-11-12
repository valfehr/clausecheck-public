"""E6.4: a real 8-rule review crashes mid-run and resumes from the checkpoint without re-paying finished rules.

Setup: one unmodified test-family CUAD contract, all 8 rules, production call pattern (prompt cache on,
no disk cache: every call is real and billed under its own ledger tag). CLAUSECHECK_FAIL_ONCE makes one rule's
node crash the first time. LangGraph saves the writes of the parallel nodes that DID finish in that step
("pending writes"), so resuming should run only the crashed rule.

Pass bar (PLAN.md Phase 6 protocol): the resume makes model calls for the crashed rule only.
Cost: one review ≈ $0.01.

  uv run python scripts/e64_resume.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_evalset import family_split  # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from clausecheck.cuad import load_contracts  # noqa: E402
from clausecheck.graph import CHECKPOINTS, ROOT, build  # noqa: E402
from clausecheck.llm import ledger_entries  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402

TAG = "phase6-resume"
FAIL_RULE = "insurance"
CLIENT = "the customer / recipient of goods or services"


def calls() -> list[dict]:
    return [e for e in ledger_entries() if e.get("tag") == TAG]


def main() -> None:
    os.environ["CLAUSECHECK_FAIL_ONCE"] = FAIL_RULE
    (ROOT / "data" / f".failed_once_{FAIL_RULE}").unlink(missing_ok=True)
    pool = sorted((c for c in load_contracts() if family_split(c.family) == "test" and 30_000 < len(c.text) < 60_000),
                  key=lambda c: c.doc_id)
    c = random.Random(3).choice(pool)
    rules = [r.id for r in load_rules()]
    print(f"contract {c.doc_id[:50]} ({len(c.text):,} chars), crash injected in '{FAIL_RULE}'")

    n0 = len(calls())
    cfg = {"configurable": {"thread_id": f"e64-{c.doc_id[:40]}-{random.random():.6f}"}}
    with SqliteSaver.from_conn_string(str(CHECKPOINTS)) as cp:
        graph = build(cp)
        try:
            graph.invoke({"doc_key": f"cuad::{c.doc_id}", "doc_id": c.doc_id, "text": c.text, "client": CLIENT,
                          "rule_ids": rules, "verdicts": [], "ledger_tag": TAG, "prompt_cache": True,
                          "disk_cache": False}, cfg)
            print("no crash?")
        except RuntimeError as e:
            print(f"crashed: {e}")
        n1 = len(calls())
        snap = graph.get_state(cfg)
        done = sorted({v["rule_id"] for v in snap.values.get("verdicts", [])})
        print(f"before the crash: {n1 - n0} model calls; verdicts in the committed state: {done}; "
              f"next node(s): {snap.next}")

        out = graph.invoke(None, cfg)  # resume from the last checkpoint
        n2 = len(calls())
        assert "__interrupt__" in out
        out = graph.invoke(Command(resume={"reviewer": "e64-stub", "decisions": {}}), cfg)

    usd_before = sum(e["cost_usd"] for e in calls()[n0:n1])
    usd_resume = sum(e["cost_usd"] for e in calls()[n1:n2])
    got = sorted(v["rule_id"] for v in out["verdicts"])
    print(f"resume: {n2 - n1} model calls (${usd_resume:.4f}); before crash ${usd_before:.4f}")
    print(f"final: {len(got)} verdicts, unique rules {len(set(got))}/8, duplicates: {len(got) - len(set(got))}")
    print("statuses:", {v['rule_id']: v['status'] for v in out['verdicts']})
    calls_per_rule = 3 if any(v["rule_id"] == FAIL_RULE and len(v.get("samples") or []) > 1 for v in out["verdicts"]) else 1
    verdict = "PASS" if (n2 - n1) <= calls_per_rule and len(set(got)) == 8 and len(got) == 8 else "FAIL"
    print(f"E6.4 {verdict}: resume made {n2 - n1} call(s); the crashed rule needs {calls_per_rule}")


if __name__ == "__main__":
    main()
