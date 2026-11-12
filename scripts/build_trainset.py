"""Distillation training set: Claude (teacher) labels dev-family contracts; the student learns from it.

For each dev-family CUAD contract (test/holdout families excluded) and each golden rule:
  index the contract → hybrid top-10 excerpts → judge_consistent with Claude Haiku (the shipped incumbent)
  → keep the example only if the citation verified AND the teacher wasn't uncertain (needs_human)
  → save {system, user, assistant(JSON)} in the exact format the judge uses at serving time.

COST GUARD (PLAN.md Phase 5): hard cap $3.00 for this script.
  1. Pilot: first PILOT checks, then project the total from their real cost; stop if > cap.
  2. Before every contract: real spend of this script's ledger tag; stop at the cap.
Re-runs are free for already-labelled examples (disk cache).

  uv run python scripts/build_trainset.py [--pilot-only]
"""

from __future__ import annotations

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_evalset import family_split  # noqa: E402
from clausecheck import search  # noqa: E402
from clausecheck.cuad import load_contracts  # noqa: E402
from clausecheck.judge import build_prompts, judge_consistent  # noqa: E402
from clausecheck.llm import AnthropicAdapter, ledger_entries  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.segment import segment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "train" / "distill_haiku_rag10.jsonl"
TAG = "phase5-teacher"
CAP_USD = 3.00
PILOT = 40
CLIENT = "the customer / recipient of goods or services"
K = 10


def tag_spend() -> float:
    return sum(e["cost_usd"] for e in ledger_entries() if e.get("tag") == TAG)


def answer_json(res) -> str:
    v = res.verdict
    conf = {0.3: "low", 0.6: "medium", 0.9: "high"}.get(v.confidence, "medium")
    return json.dumps({"cited_clause_ids": v.cited_clause_ids, "quote": res.llm.get("quote", ""),
                       "rationale": v.rationale, "status": v.status.value,
                       "red_reason": v.red_reason.value if v.red_reason else "none", "confidence": conf},
                      ensure_ascii=False)


def main() -> None:
    rules = load_rules()
    contracts = [c for c in load_contracts() if family_split(c.family) == "dev"]
    random.Random(5).shuffle(contracts)
    es = search.client()
    search.create_index(es)
    adapter = AnthropicAdapter(tag=TAG)
    start_spend = tag_spend()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in OUT.read_text(encoding="utf-8").splitlines()} if OUT.exists() else set()
    stats = {"kept": 0, "dropped_unverified": 0, "dropped_uncertain": 0, "checks": 0}
    projected_ok = start_spend > 0  # resuming a run that already passed the pilot

    with OUT.open("a", encoding="utf-8") as out:
        for ci, c in enumerate(contracts):
            if tag_spend() >= CAP_USD:
                print(f"STOP: teacher spend ${tag_spend():.3f} reached the ${CAP_USD:.2f} cap")
                break
            doc_key = f"cuad::{c.doc_id}"
            clauses = segment(c.doc_id, c.text)
            if not es.count(index=search.INDEX, query={"term": {"doc_key": doc_key}})["count"]:
                search.index_document(es, doc_key, c.doc_id, clauses)
                es.indices.refresh(index=search.INDEX)
            todo = [r for r in rules if f"{c.doc_id}::{r.id}" not in done]

            def label(rule):
                hits = {sid for sid, _ in search.hybrid(es, doc_key, search.rule_query(rule), K)}
                shown = [x for x in clauses if x.id in hits]
                return rule, shown, judge_consistent(adapter, shown, rule, CLIENT, cache_contract=False,
                                                     excerpts_only=True)

            with ThreadPoolExecutor(8) as ex:  # the 8 rules of one contract in parallel; cap checked per contract
                results = list(ex.map(label, todo))
            for rule, shown, res in results:
                key = f"{c.doc_id}::{rule.id}"
                stats["checks"] += 1
                if not res.citation_valid:
                    stats["dropped_unverified"] += 1
                elif res.consistency.get("needs_human"):
                    stats["dropped_uncertain"] += 1
                else:
                    system, user = build_prompts(shown, rule, CLIENT, excerpts_only=True)
                    out.write(json.dumps({"key": key, "family": c.family, "rule_id": rule.id,
                                          "status": res.verdict.status.value, "system": system, "user": user,
                                          "assistant": answer_json(res)}, ensure_ascii=False) + "\n")
                    out.flush()
                    stats["kept"] += 1
                if not projected_ok and stats["checks"] == PILOT:
                    per = (tag_spend() - start_spend) / PILOT
                    proj = tag_spend() + per * (len(contracts) * len(rules) - PILOT)
                    print(f"PILOT: {PILOT} checks cost ${tag_spend() - start_spend:.4f} (${per:.5f}/check); "
                          f"projected total ${proj:.2f} vs cap ${CAP_USD:.2f}", flush=True)
                    if proj > CAP_USD or "--pilot-only" in sys.argv:
                        print("Stopping after pilot." if proj <= CAP_USD else "STOP: projection exceeds the cap.")
                        return
                    projected_ok = True
            if (ci + 1) % 20 == 0:
                print(f"  {ci + 1}/{len(contracts)} contracts, {stats}, teacher spend ${tag_spend():.3f}", flush=True)
    print(f"Done: {stats}; teacher spend ${tag_spend():.3f}; file {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
