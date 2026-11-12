"""E6.3b: playbook drafter + independent verifiers, on the same 16 cases as E6.3. Pass bars: PLAN.md E6.3b.

Part 1  verifier test: for one case per rule, Sonnet writes 3 deliberately flawed redlines (over-reach, partial fix,
        removed protection); the verifier must reject each.
Part 2  drafting: same 16 red/yellow verdicts as E6.3; accepted redlines get an independent Opus grade.

Cost guard: ledger tag phase6-drafting-v2, hard cap $1.75 (shared with build_playbook.py).

  uv run python scripts/e63b_drafting.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck import search  # noqa: E402
from clausecheck.drafting_v2 import bad_variants, grade, propose, verify  # noqa: E402
from clausecheck.llm import AnthropicAdapter, ledger_entries  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.schemas import EvalCase, Status, Verdict  # noqa: E402
from clausecheck.segment import segment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAG, CAP = "phase6-drafting-v2", 1.75
OUT = ROOT / "reports" / "e63b_drafting.md"


def spent() -> float:
    return sum(e["cost_usd"] for e in ledger_entries() if e.get("tag") == TAG)


def main() -> None:
    rules = {r.id: r for r in load_rules()}
    prev = [json.loads(l) for l in (ROOT / "reports" / "e63_drafting.jsonl").read_text(encoding="utf-8").split("\n") if l.strip()]
    run = {r["case_id"]: r for r in map(json.loads, sorted((ROOT / "runs").glob("*_SC-run3.jsonl"))[-1]
                                        .read_text(encoding="utf-8").split("\n")) if r.get("pred_status")}
    allc = {c.case_id: c for s in ("severity", "severity_hard") for c in
            (EvalCase.model_validate_json(l) for l in (ROOT / "data" / "evals" / f"{s}.jsonl").read_text(encoding="utf-8").split("\n") if l.strip())}
    cases = [allc[p["case_id"]] for p in prev]  # the same 16 as E6.3
    adapter, es = AnthropicAdapter(tag=TAG), search.client()

    def verdict_of(c: EvalCase) -> Verdict:
        r = run[c.case_id]
        return Verdict(rule_id=c.rule_id, status=Status(r["pred_status"]), rationale=r["rationale"],
                       confidence=r["confidence"], cited_clause_ids=r["cited_clause_ids"])

    # ---- part 1: verifier on deliberately flawed redlines
    caught = Counter()
    total = Counter()
    one_per_rule = list({c.rule_id: c for c in reversed(cases)}.values())
    for c in one_per_rule:
        if spent() >= CAP:
            print("STOP: cap reached")
            break
        clauses = segment(c.doc_id, c.text)
        v = verdict_of(c)
        sid = v.cited_clause_ids[0] if v.cited_clause_ids else \
            next(s for s, _ in search.hybrid(es, c.case_id, search.rule_query(rules[c.rule_id]), 1))
        seg = next(x for x in clauses if x.id == sid)
        bad, _ = bad_variants(adapter, rules[c.rule_id], c.client_party, seg, missing=not v.cited_clause_ids)
        start = {s for s, _ in search.hybrid(es, c.case_id, search.rule_query(rules[c.rule_id]), 10)}
        for kind in ("over_reach", "partial_fix", "removed_protection"):
            res = verify(adapter, clauses, getattr(bad, kind), rules[c.rule_id], c.client_party, start)
            total[kind] += 1
            caught[kind] += not res["ok"]
            print(f"  verifier test {c.case_id:45} {kind:18} rejected={not res['ok']} "
                  f"(judge {res.get('judge')}, audit ok {res.get('ok_audit')})", flush=True)

    # ---- part 2: drafting on the same 16 cases
    rows = []
    for c in cases:
        if spent() >= CAP:
            print("STOP: cap reached")
            break
        clauses = segment(c.doc_id, c.text)
        res = propose(adapter, es, c.case_id, clauses, rules[c.rule_id], c.client_party, verdict_of(c))
        g, gc = None, 0.0
        if res.accepted:
            old = next((x.text for x in clauses if x.id == res.redline.segment_id), "")
            g, gc = grade(adapter, rules[c.rule_id], c.client_party, res.redline,
                          old if res.redline.action == "replace" else f"(inserted after:) {old}")
        good = bool(g and g.meets_green and g.legally_coherent and not g.over_reaches)
        rows.append({"case_id": c.case_id, "accepted": res.accepted, "rounds": res.rounds, "cost": res.cost_usd,
                     "grade_cost": gc, "confirmed": good, "grade": g.model_dump() if g else None,
                     "redline": res.redline.model_dump() if res.redline else None, "history": res.history})
        print(f"{c.case_id:45} accepted={res.accepted!s:5} rounds={res.rounds} ${res.cost_usd:.4f}"
              + (f" opus: green={g.meets_green} coherent={g.legally_coherent} over_reach={g.over_reaches}" if g else ""),
              flush=True)

    (ROOT / "reports" / "e63b_drafting.jsonl").write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")
    acc = [x for x in rows if x["accepted"]]
    rej = Counter()
    for x in rows:
        for h in x["history"]:
            if not h.get("ok"):
                if h.get("problem"):
                    rej["bad segment id"] += 1
                if h.get("ok_judge") is False:
                    rej[f"verifier judge said {h.get('judge')}"] += 1
                if h.get("ok_audit") is False:
                    rej["side-effect audit"] += 1
    prev_acc = sum(p["accepted"] for p in prev)
    prev_ok = sum(bool(p["accepted"] and p["grade"]["meets_green"] and p["grade"]["legally_coherent"]) for p in prev)
    lines = [
        "# E6.3b drafting agent: playbook drafter + independent verifiers", "",
        "## Verifier test (deliberately flawed redlines must be rejected)", "",
        *[f"- {k}: rejected **{caught[k]}/{total[k]}**" for k in total],
        f"- **Overall: {sum(caught.values())}/{sum(total.values())} ({sum(caught.values()) / max(sum(total.values()), 1):.0%})** (bar ≥ 90%)", "",
        "## Drafting, same 16 cases as E6.3", "",
        "| | E6.3 (Haiku drafts + Haiku accepts) | E6.3b (playbook + Sonnet verifiers) |", "|---|---|---|",
        f"| Accepted | {prev_acc}/16 | {len(acc)}/{len(rows)} |",
        f"| Accepted AND independently confirmed | {prev_ok}/{prev_acc} ({prev_ok / max(prev_acc, 1):.0%}), grader Sonnet | "
        f"**{sum(x['confirmed'] for x in acc)}/{len(acc)} ({sum(x['confirmed'] for x in acc) / max(len(acc), 1):.0%})**, grader Opus (bar ≥ 90%) |",
        f"| Confirmed redlines out of 16 | {prev_ok} | {sum(x['confirmed'] for x in acc)} |",
        f"| $ per case | ${statistics.mean(p['cost'] for p in prev):.4f} | ${statistics.mean(x['cost'] for x in rows):.4f} |",
        f"| $ per confirmed redline | ${sum(p['cost'] for p in prev) / max(prev_ok, 1):.4f} | "
        f"${sum(x['cost'] for x in rows) / max(sum(x['confirmed'] for x in acc), 1):.4f} |", "",
        f"- Rejections by reason (all rounds): {dict(rej)}",
        f"- Optional suggestions offered (not applied): {sum(len(h.get('suggestions') or []) for x in rows for h in x['history'])}",
        f"- Total spend under {TAG}: ${spent():.4f}",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
