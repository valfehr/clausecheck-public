"""Phase 5 follow-up: teacher labels with a NAMED client party, from both sides of the same contract.

Why: every critical miss of the tuned Gemma was one case where it called the client "the counterparty".
The first training set always used the generic client "the customer / recipient of goods or services",
while 40 of the 42 severity test cases name the client ("MD Anderson"). Train/serve mismatch: the student
never practised working out which named party is the client.

Fix: for N dev contracts (already indexed, same RAG top-10), find the two parties' names, then label the
direction-dependent rules twice, once with each party as the client. Same contract and excerpts, opposite
client → often opposite verdicts: a contrastive pair that teaches the student to track who is who.

COST GUARD: hard cap $0.60 (own ledger tag), pilot projection first, real spend re-checked per contract.

  uv run python scripts/build_trainset_parties.py [--contracts 18]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_evalset import family_split  # noqa: E402
from build_trainset import K, answer_json  # noqa: E402
from clausecheck import search  # noqa: E402
from clausecheck.cuad import load_contracts  # noqa: E402
from clausecheck.judge import build_prompts, judge_consistent  # noqa: E402
from clausecheck.llm import AnthropicAdapter, ledger_entries  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.segment import segment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "train" / "distill_named_parties.jsonl"
TAG = "phase5-teacher-parties"
CAP_USD = 0.60
PILOT_CONTRACTS = 2
SKIP_RULES = {"governing_law"}  # the only rule whose verdict doesn't depend on which side you're on


class Parties(BaseModel):
    party_a: str
    party_b: str


PARTY_SYSTEM = """You read the opening of a contract and the party mentions a lawyer highlighted.
Return the two main contracting parties as the SHORT names the contract body uses for them
(the defined term if there is one, e.g. "Company", "iVillage", "MD Anderson"), one per field."""


def tag_spend() -> float:
    return sum(e["cost_usd"] for e in ledger_entries() if e.get("tag") == TAG)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", type=int, default=18)
    a = ap.parse_args()

    rules = [r for r in load_rules() if r.id not in SKIP_RULES]
    contracts = [c for c in load_contracts() if family_split(c.family) == "dev"]
    random.Random(5).shuffle(contracts)  # same order as build_trainset → these are already indexed
    contracts = contracts[: a.contracts]
    es = search.client()
    adapter = AnthropicAdapter(tag=TAG)
    done = {json.loads(l)["key"] for l in OUT.read_text(encoding="utf-8").splitlines()} if OUT.exists() else set()
    stats = {"kept": 0, "dropped_unverified": 0, "dropped_uncertain": 0, "checks": 0, "pairs_disagree": 0}
    start = tag_spend()

    with OUT.open("a", encoding="utf-8") as out:
        for ci, c in enumerate(contracts):
            if tag_spend() >= CAP_USD:
                print(f"STOP: spend ${tag_spend():.3f} reached the ${CAP_USD:.2f} cap")
                break
            mentions = sorted({c.text[s.start:s.end] for s in c.labels.get("Parties", [])}, key=len)[:12]
            p = adapter.complete(system=PARTY_SYSTEM, schema=Parties, max_tokens=300,
                                 user=f"Highlighted party mentions: {mentions}\n\nOpening:\n{c.text[:2500]}").parsed
            doc_key = f"cuad::{c.doc_id}"
            clauses = segment(c.doc_id, c.text)
            if not es.count(index=search.INDEX, query={"term": {"doc_key": doc_key}})["count"]:
                search.index_document(es, doc_key, c.doc_id, clauses)
                es.indices.refresh(index=search.INDEX)
            jobs = [(r, side, client) for r in rules for side, client in (("a", p.party_a), ("b", p.party_b))
                    if f"{c.doc_id}::{r.id}::{side}" not in done]

            def label(job):
                rule, side, client = job
                hits = {sid for sid, _ in search.hybrid(es, doc_key, search.rule_query(rule), K)}
                shown = [x for x in clauses if x.id in hits]
                return rule, side, client, shown, judge_consistent(adapter, shown, rule, client,
                                                                   cache_contract=False, excerpts_only=True)

            with ThreadPoolExecutor(7) as ex:
                results = list(ex.map(label, jobs))
            by_rule: dict[str, list[str]] = {}
            for rule, side, client, shown, res in results:
                stats["checks"] += 1
                by_rule.setdefault(rule.id, []).append(res.verdict.status.value)
                if not res.citation_valid:
                    stats["dropped_unverified"] += 1
                    continue
                if res.consistency.get("needs_human"):
                    stats["dropped_uncertain"] += 1
                    continue
                system, user = build_prompts(shown, rule, client, excerpts_only=True)
                out.write(json.dumps({"key": f"{c.doc_id}::{rule.id}::{side}", "family": c.family,
                                      "rule_id": rule.id, "client": client, "status": res.verdict.status.value,
                                      "system": system, "user": user, "assistant": answer_json(res)},
                                     ensure_ascii=False) + "\n")
                out.flush()
                stats["kept"] += 1
            stats["pairs_disagree"] += sum(1 for v in by_rule.values() if len(v) == 2 and v[0] != v[1])
            print(f"  {ci + 1}/{len(contracts)} {c.doc_id[:40]} ({p.party_a} / {p.party_b}): {stats}, "
                  f"spend ${tag_spend():.3f}", flush=True)
            if ci + 1 == PILOT_CONTRACTS and start == 0:
                proj = tag_spend() / PILOT_CONTRACTS * len(contracts)
                print(f"PILOT: projected ${proj:.2f} vs cap ${CAP_USD:.2f}", flush=True)
                if proj > CAP_USD:
                    print("STOP: projection exceeds the cap.")
                    return
    print(f"Done: {stats}; spend ${tag_spend():.3f}; file {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
