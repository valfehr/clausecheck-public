"""Run the judge over eval suites and report metrics.

Usage:
  uv run python scripts/run_eval.py --name baseline --splits dev test [--model claude-haiku-5-5] [--effort low]
  uv run python scripts/run_eval.py --name release --splits holdout --release   # only for promote/block

Metrics
- severity: confusion matrix, accuracy, RED RECALL (true reds caught), CRITICAL MISS (true red → green),
  red-reason accuracy
- retrieval: citation hit rate (a cited segment overlaps a CUAD lawyer span), absent-clause → red/missing rate
- all: citation verification failure rate, $/case, latency p50/p95, tokens
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck.judge import PROMPT_VERSION, judge, judge_consistent  # noqa: E402
from clausecheck.llm import AnthropicAdapter, BudgetExceeded, spent_usd  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.schemas import EvalCase  # noqa: E402
from clausecheck.segment import covering_clauses, segment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["green", "yellow", "red"]


def load_cases(suites: list[str], splits: list[str]) -> list[EvalCase]:
    cases = []
    for s in suites:
        # split("\n"), not splitlines(): contract text contains U+2028 / form feeds that splitlines() breaks on
        for line in (ROOT / "data" / "evals" / f"{s}.jsonl").read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            c = EvalCase.model_validate_json(line)
            if c.split in splits:
                cases.append(c)
    return cases


def build_fewshot(rules: dict) -> dict[str, str]:
    """One green, one yellow, one red/contradicts clause per rule, from DEV severity cases only."""
    cases = load_cases(["severity"], ["dev"])
    out: dict[str, str] = {}
    for rid in rules:
        picks = []
        for status, reason in [("green", None), ("yellow", None), ("red", "contradicts")]:
            c = next((c for c in cases if c.rule_id == rid and c.expected_status.value == status and c.expected_spans
                      and (c.expected_red_reason.value if c.expected_red_reason else None) == reason), None)
            if c:
                a, b = c.expected_spans[0]
                picks.append(f'- {status.upper()}{" (" + reason + ")" if reason else ""}: "{c.text[a:b].strip()[:700]}"')
        out[rid] = "\n".join(picks)
    return out


FRESH = False
SELF_CONSISTENCY = False
RAG_K = 0
ES = None
PARTIES = False
AGENTIC = False


def run_case(adapter, rules, case: EvalCase, model: str, effort: str, examples: dict[str, str]) -> dict:
    clauses = segment(case.doc_id, case.text)
    expected_ids = sorted({i for a, b in case.expected_spans for i in covering_clauses(clauses, a, b)})
    shown = clauses
    if RAG_K:  # RAG: the judge only sees the top-K hybrid search hits, in document order
        hit_ids = {sid for sid, _ in search.hybrid(ES, case.case_id, search.rule_query(rules[case.rule_id]), RAG_K)}
        shown = [c for c in clauses if c.id in hit_ids]
    agent = None
    if AGENTIC:  # E6.2: an agent adds evidence to the same top-K before the (unchanged) judge sees it
        from clausecheck.retrieval_agent import gather
        agent = gather(adapter, ES, case.case_id, clauses, rules[case.rule_id], case.client_party, k=RAG_K,
                       disk_cache=not FRESH)
        shown = agent.shown
    counterparty = ""
    if PARTIES:  # who's-who metadata stored at ingestion (scripts/index_parties.py)
        from clausecheck import parties
        counterparty = parties.counterparty_for(parties.load(ES, case.case_id), case.client_party)
    for attempt in range(3):
        try:
            fn = judge_consistent if SELF_CONSISTENCY else judge
            r = fn(adapter, shown, rules[case.rule_id], case.client_party, model=model, effort=effort,
                   cache_contract=False, examples=examples.get(case.rule_id, ""), disk_cache=not FRESH,
                   counterparty=counterparty, **({"excerpts_only": True} if RAG_K else {}))
            break
        except BudgetExceeded:
            raise
        except Exception as e:  # noqa: BLE001 — rate limits beyond SDK retries, transient errors
            if attempt == 2:
                return {"case_id": case.case_id, "split": case.split, "suite": case.suite, "error": repr(e)}
            time.sleep(20 * (attempt + 1))
    v = r.verdict
    return {
        "case_id": case.case_id, "suite": case.suite, "split": case.split, "rule_id": case.rule_id,
        "family": case.family, "trap": case.note.split("trap=")[1].split(";")[0] if "trap=" in case.note else None,
        "expected_status": case.expected_status.value if case.expected_status else None,
        "expected_red_reason": case.expected_red_reason.value if case.expected_red_reason else None,
        "counterparty": counterparty,
        "pred_status": v.status.value, "pred_red_reason": v.red_reason.value if v.red_reason else None,
        "expected_clause_ids": expected_ids, "cited_clause_ids": v.cited_clause_ids,
        "citation_hit": bool(set(expected_ids) & set(v.cited_clause_ids)) if expected_ids else None,
        "citation_valid": r.citation_valid, "citation_problem": r.citation_problem,
        "confidence": v.confidence, "rationale": v.rationale, **r.llm,
        "samples": r.consistency.get("samples"), "agreement": r.consistency.get("agreement"),
        "needs_human": r.consistency.get("needs_human", not r.citation_valid) or not r.citation_valid,
        **({"agent_added": agent.added, "agent_tools": agent.tool_calls, "agent_calls": agent.model_calls,
            "agent_cost_usd": agent.cost_usd, "agent_in": agent.input_tokens,
            "judge_cost_usd": r.llm["cost_usd"], "cost_usd": r.llm["cost_usd"] + agent.cost_usd,
            "input_tokens": r.llm["input_tokens"] + agent.input_tokens} if agent else {}),
    }


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}% ({n}/{d})" if d else "n/a"


def summarize(rows: list[dict], header: str) -> list[str]:
    ok = [r for r in rows if "error" not in r]
    sev = [r for r in ok if r["expected_status"]]
    lines = [f"### {header}", ""]
    if sev:
        cm = {(e, p): 0 for e in LABELS for p in LABELS}
        for r in sev:
            cm[(r["expected_status"], r["pred_status"])] += 1
        reds = [r for r in sev if r["expected_status"] == "red"]
        lines += [
            f"- Severity accuracy: **{pct(sum(cm[(l, l)] for l in LABELS), len(sev))}**",
            f"- **Red recall**: {pct(sum(1 for r in reds if r['pred_status'] == 'red'), len(reds))}",
            f"- **Critical miss (red → green)**: {pct(sum(1 for r in reds if r['pred_status'] == 'green'), len(reds))}",
            f"- Red reason correct (when red caught): "
            f"{pct(sum(1 for r in reds if r['pred_status'] == 'red' and r['pred_red_reason'] == r['expected_red_reason']), sum(1 for r in reds if r['pred_status'] == 'red'))}",
            "",
            "| expected ↓ / predicted → | green | yellow | red |",
            "|---|---|---|---|",
            *[f"| {e} | " + " | ".join(str(cm[(e, p)]) for p in LABELS) + " |" for e in LABELS],
            "",
        ]
        correct = sum(r["expected_status"] == r["pred_status"] for r in sev)
        lines.append(f"- **Cost per correct verdict**: ${sum(r['cost_usd'] for r in sev) / max(correct, 1):.5f}")
        traps = sorted({r["trap"] for r in sev if r.get("trap")})
        for t in traps:
            sub = [r for r in sev if r.get("trap") == t]
            lines.append(f"- trap `{t}`: {pct(sum(r['expected_status'] == r['pred_status'] for r in sub), len(sub))}")
        lines.append("")
    cit = [r for r in ok if r["citation_hit"] is not None]
    if cit:
        lines.append(f"- Citation hit (cited segment overlaps the expected clause): **{pct(sum(r['citation_hit'] for r in cit), len(cit))}**")
    lines.append(f"- Citation verification passed: {pct(sum(r['citation_valid'] for r in ok), len(ok))}")
    paid = [r for r in ok if not r["from_cache"]]
    lat = sorted(r["latency_s"] for r in paid) or [0]
    lines += [
        f"- Cost: ${sum(r['cost_usd'] for r in ok):.4f} total, ${sum(r['cost_usd'] for r in ok) / max(len(ok), 1):.5f}/case "
        f"(cached replays count $0)",
        f"- Latency (uncached calls): p50 {statistics.median(lat):.1f}s, p95 {lat[int(0.95 * (len(lat) - 1))]:.1f}s",
        f"- Tokens/case: input {statistics.mean(r['input_tokens'] + r['cache_read_tokens'] + r['cache_write_tokens'] for r in ok):,.0f}, "
        f"output {statistics.mean(r['output_tokens'] for r in ok):,.0f}",
        f"- Errors: {len(rows) - len(ok)}",
        "",
    ]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--suites", nargs="+", default=["severity", "severity_hard", "retrieval"])
    ap.add_argument("--fewshot", action="store_true", help="add dev-split labelled examples (test/holdout only)")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--release", action="store_true", help="required to touch the holdout split")
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (smoke test)")
    ap.add_argument("--fresh", action="store_true", help="skip the disk cache (measure run-to-run noise)")
    ap.add_argument("--sc", action="store_true", help="self-consistency: re-ask non-high-confidence verdicts 3x")
    ap.add_argument("--rag", type=int, default=0, help="RAG: judge sees only the top-K hybrid search hits")
    ap.add_argument("--local-url", default="http://127.0.0.1:8081", help="OpenAI-compatible server for --model local:NAME")
    ap.add_argument("--agentic", action="store_true", help="E6.2: retrieval agent adds evidence to the top-K (needs --rag)")
    ap.add_argument("--parties", action="store_true", help="state the counterparty from party metadata (needs --rag)")
    ap.add_argument("--temperature", type=float, default=None, help="local models only (default 0.6; 0 = greedy)")
    args = ap.parse_args()
    if "holdout" in args.splits and not args.release:
        sys.exit("Holdout is reserved for release decisions. Pass --release if that's what this run is.")

    global FRESH, SELF_CONSISTENCY, RAG_K, ES, PARTIES, AGENTIC
    FRESH = args.fresh
    PARTIES = args.parties
    AGENTIC = args.agentic
    if AGENTIC and not args.rag:
        sys.exit("--agentic needs --rag K (the agent starts from the top-K hits)")
    SELF_CONSISTENCY = args.sc
    RAG_K = args.rag
    if RAG_K:
        from clausecheck import search as _search
        globals()["search"] = _search
        ES = _search.client()
    if args.fewshot and "dev" in args.splits:
        sys.exit("Few-shot examples come from dev; evaluate few-shot on test/holdout only (no leakage).")
    rules = {r.id: r for r in load_rules()}
    examples = build_fewshot(rules) if args.fewshot else {}
    cases = load_cases(args.suites, args.splits)
    if args.limit:
        cases = cases[: args.limit]
    if args.model.startswith("local:"):  # e.g. local:llama-3.1-8b → llama.cpp/vLLM server at --local-url
        from clausecheck.local_llm import TEMPERATURE, LocalAdapter
        adapter = LocalAdapter(args.local_url, args.model[len("local:"):], tag=f"eval-{args.name}",
                               temperature=TEMPERATURE if args.temperature is None else args.temperature)
    else:
        adapter = AnthropicAdapter(tag=f"phase1-{args.name}")
    spent_before = spent_usd()
    print(f"{len(cases)} cases, model {args.model}, effort {args.effort}, prompt {PROMPT_VERSION}")
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda c: run_case(adapter, rules, c, args.model, args.effort, examples), cases))

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs"
    run_dir.mkdir(exist_ok=True)
    (run_dir / f"{stamp}_{args.name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    lines = [f"# Eval run `{args.name}` ({stamp})", "",
             f"model `{args.model}`, effort `{args.effort}`, prompt `{PROMPT_VERSION}{'+fewshot' if args.fewshot else ''}"
             f"{'+sc3' if args.sc else ''}{'+parties' if args.parties else ''}{'+agentic' if args.agentic else ''}{f'+rag{args.rag}' if args.rag else ''}`, "
             f"splits {args.splits}", ""]
    for split in args.splits:
        for suite in args.suites:
            sub = [r for r in rows if r.get("split") == split and r.get("suite") == suite]
            if sub:
                lines += summarize(sub, f"{suite} / {split}")
    lines.append(f"Actual spend this run: ${spent_usd() - spent_before:.4f}; project total ${spent_usd():.4f}")
    (run_dir / f"{stamp}_{args.name}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
