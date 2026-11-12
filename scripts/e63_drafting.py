"""E6.3: drafting agent on real red/yellow verdicts. Pass bars and metrics: PLAN.md Phase 6 protocol.

Input: test-split severity + hard cases whose label is red or yellow AND where the stored full-contract run
(SC-run3) also said red/yellow (the drafter works from the tool's own verdict, as in production).
For each: propose → judge re-check (green?) → blind side-effect check → up to 3 rounds.
Accepted redlines get an independent Sonnet grade (eval only) to measure judge gaming.

Cost guard: ledger tag phase6-drafting, stops at CAP_USD.

  uv run python scripts/e63_drafting.py [--limit 16]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck import search  # noqa: E402
from clausecheck.drafting import grade, propose  # noqa: E402
from clausecheck.llm import AnthropicAdapter, ledger_entries  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.schemas import EvalCase, Status, Verdict  # noqa: E402
from clausecheck.segment import segment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG, CAP_USD = "phase6-drafting", 1.00
OUT = ROOT / "reports" / "e63_drafting.md"


def spent() -> float:
    return sum(e["cost_usd"] for e in ledger_entries() if e.get("tag") == TAG)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=16)
    a = ap.parse_args()
    rules = {r.id: r for r in load_rules()}
    run = {r["case_id"]: r for r in map(json.loads, sorted((ROOT / "runs").glob("*_SC-run3.jsonl"))[-1]
                                        .read_text(encoding="utf-8").split("\n")) if r.get("pred_status")}
    cases = [EvalCase.model_validate_json(l) for s in ("severity", "severity_hard")
             for l in (ROOT / "data" / "evals" / f"{s}.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
    cases = [c for c in cases if c.split == "test" and c.expected_status in (Status.RED, Status.YELLOW)
             and run.get(c.case_id, {}).get("pred_status") in ("red", "yellow")]
    # one case per (rule, label) first, for spread
    seen, picked = set(), []
    for c in sorted(cases, key=lambda c: c.case_id):
        k = (c.rule_id, c.expected_status.value)
        if k not in seen:
            seen.add(k)
            picked.append(c)
    picked += [c for c in cases if c not in picked]
    picked = picked[: a.limit]
    adapter, es = AnthropicAdapter(tag=TAG), search.client()
    rows = []
    for c in picked:
        if spent() >= CAP_USD:
            print(f"STOP: ${CAP_USD} cap reached")
            break
        rule, r = rules[c.rule_id], run[c.case_id]
        clauses = segment(c.doc_id, c.text)
        idx = {x.id: i for i, x in enumerate(clauses)}
        want = set()
        for cid in r["cited_clause_ids"]:
            if cid in idx:
                want.update(range(max(idx[cid] - 1, 0), min(idx[cid] + 2, len(clauses))))
        if not want:  # missing clause: show the search hits so the drafter can pick an anchor
            want = {idx[s] for s, _ in search.hybrid(es, c.case_id, search.rule_query(rule), 10) if s in idx}
        context = [clauses[i] for i in sorted(want | {0, 1})]  # + opening segments: who the parties are
        v = Verdict(rule_id=c.rule_id, status=Status(r["pred_status"]), rationale=r["rationale"],
                    confidence=r["confidence"], cited_clause_ids=r["cited_clause_ids"])
        res = propose(adapter, clauses, context, rule, c.client_party, v)
        g, gcost = (None, 0.0)
        if res.accepted:
            old = next((x.text for x in clauses if x.id == res.redline.segment_id), "")
            g, gcost = grade(adapter, rule, c.client_party, res.redline, old if res.redline.action == "replace" else
                             f"(inserted after:) {old}")
        rows.append({"case_id": c.case_id, "rule": c.rule_id, "label": c.expected_status.value,
                     "tool_verdict": r["pred_status"], "accepted": res.accepted, "rounds": res.rounds,
                     "calls": res.calls, "cost": res.cost_usd, "grade_cost": gcost,
                     "action": res.redline.action if res.redline else None,
                     "redline": res.redline.new_text if res.redline else None,
                     "grade": g.model_dump() if g else None, "history": res.history})
        print(f"{c.case_id:45} {r['pred_status']:6} accepted={res.accepted!s:5} rounds={res.rounds} "
              f"${res.cost_usd:.4f}" + (f" grade green={g.meets_green} coherent={g.legally_coherent}" if g else ""),
              flush=True)

    (ROOT / "reports" / "e63_drafting.jsonl").write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    acc = [x for x in rows if x["accepted"]]
    graded_ok = sum(x["grade"]["meets_green"] and x["grade"]["legally_coherent"] for x in acc)
    rejections = Counter()
    for x in rows:
        for h in x["history"]:
            if not h.get("accepted"):
                if h.get("problem"):
                    rejections["bad segment id"] += 1
                    continue
                if h["judge"] != "green":
                    rejections[f"judge said {h['judge']}"] += 1
                se = h["side_effects"]
                if se["other_obligations_changed"] or se["client_protection_removed"]:
                    rejections["side effects"] += 1
    lines = [
        "# E6.3 drafting agent", "",
        f"- Cases: {len(rows)} red/yellow verdicts (test split)",
        f"- **Accepted: {len(acc)}/{len(rows)} ({len(acc) / max(len(rows), 1):.0%})**; rounds used by accepted: "
        f"{dict(Counter(x['rounds'] for x in acc))}",
        f"- **Independent grade of accepted redlines (Sonnet, eval only): {graded_ok}/{len(acc)} meet green and are coherent**",
        f"- Rejections by reason (all rounds): {dict(rejections)}",
        f"- $ per case: mean ${statistics.mean(x['cost'] for x in rows):.4f}; $ per accepted redline: "
        f"${sum(x['cost'] for x in rows) / max(len(acc), 1):.4f}; model calls per case: "
        f"{statistics.mean(x['calls'] for x in rows):.1f}",
        f"- Total spend (incl. grading): ${spent():.4f}", "",
        "| case | tool verdict | accepted | rounds | grade | redline (start) |", "|---|---|---|---|---|---|",
        *[f"| {x['case_id']} | {x['tool_verdict']} | {'✅' if x['accepted'] else '❌'} | {x['rounds']} | "
          f"{('✅' if x['grade']['meets_green'] and x['grade']['legally_coherent'] else '⚠️ ' + x['grade']['rationale'][:80]) if x['grade'] else ''} | "
          f"{(x['redline'] or '')[:90].replace('|', '/')} |" for x in rows],
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
