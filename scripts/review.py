"""Review one contract against the golden rules: the tool as a user would run it.

Examples:
  uv run python scripts/review.py --list                         # some contracts you can try
  uv run python scripts/review.py --cuad "NETGEAR"               # a CUAD contract, by part of its name
  uv run python scripts/review.py --file my_contract.txt --client "Acme Corp"
  uv run python scripts/review.py --cuad "NETGEAR" --rules governing_law insurance

Prints a Green/Yellow/Red verdict per rule with the cited excerpt and the reason, and saves
a Markdown report in reports/reviews/. Uses the shipped config (Haiku 5.5, low effort) and the
production call pattern: rule 1 caches the contract, the other rules run in parallel on the cache.
Typical cost: under $0.01 per contract. Re-running the same contract is free (disk cache).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_evalset import family_split  # noqa: E402
from clausecheck.cuad import load_contracts  # noqa: E402
from clausecheck.judge import judge_consistent as judge  # noqa: E402  (re-asks non-high-confidence verdicts 3x)
from clausecheck.llm import AnthropicAdapter, spent_usd  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.segment import segment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "reviews"
DEFAULT_CLIENT = "the customer / recipient of goods or services"
ICON = {"green": "🟢", "yellow": "🟡", "red": "🔴"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def pick_cuad(query: str):
    hits = [c for c in load_contracts() if query.lower() in c.doc_id.lower()]
    if not hits:
        sys.exit(f"No CUAD contract matches '{query}'. Try --list.")
    if len(hits) > 1:
        print(f"{len(hits)} matches, using the first: {hits[0].doc_id}")
    return hits[0].doc_id, hits[0].text, hits[0].family


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="path to a plain-text contract")
    src.add_argument("--cuad", help="part of a CUAD contract name")
    src.add_argument("--list", action="store_true", help="list some test-split contracts to try")
    ap.add_argument("--client", default=DEFAULT_CLIENT, help="the party you represent, as named in the contract")
    ap.add_argument("--rules", nargs="+", help="rule ids to check (default: all 8)")
    args = ap.parse_args()

    if args.list:
        test = [c for c in load_contracts() if family_split(c.family) == "test" and 15_000 < len(c.text) < 60_000]
        for c in test[:15]:
            print(f"  {c.family:<20} ~{int(len(c.text) / 2.97):>6,} tokens   {c.doc_id}")
        return

    if args.file:
        path = Path(args.file)
        doc_id, text, family = path.stem, path.read_text(encoding="utf-8", errors="replace"), "-"
    else:
        doc_id, text, family = pick_cuad(args.cuad)

    rules = [r for r in load_rules() if not args.rules or r.id in args.rules]
    if not rules:
        sys.exit("No matching rules. Ids: " + ", ".join(r.id for r in load_rules()))
    clauses = segment(doc_id, text)
    by_id = {c.id: c for c in clauses}
    adapter = AnthropicAdapter(tag="review-cli")
    before = spent_usd()

    print(f"\nReviewing: {doc_id}\nFamily: {family} · ~{int(len(text) / 2.97):,} tokens · {len(clauses)} segments · "
          f"on behalf of: {args.client}\n")
    t0 = time.perf_counter()
    first = judge(adapter, clauses, rules[0], args.client)
    with ThreadPoolExecutor(7) as ex:
        rest = list(ex.map(lambda r: judge(adapter, clauses, r, args.client), rules[1:]))
    wall = time.perf_counter() - t0

    md = [f"# Review: {doc_id}", "", f"- On behalf of: **{args.client}**", f"- Family: {family}",
          f"- Model: Haiku 5.5, effort low, prompt judge-v1", ""]
    counts = {"green": 0, "yellow": 0, "red": 0}
    for rule, res in zip(rules, [first, *rest]):
        v = res.verdict
        counts[v.status.value] += 1
        reason = f" ({v.red_reason.value})" if v.red_reason else ""
        flag = "" if res.citation_valid else f"  ⚠ needs human check: {res.citation_problem}"
        if res.llm.get("retried") and res.citation_valid:
            flag = "  (first answer was invalid; retried at higher effort)"
        c = res.consistency
        if len(c.get("samples", [])) > 1:
            note = f"asked {len(c['samples'])}x: {'/'.join(c['samples'])} → {c['agreement']}"
            flag += f"  ⚠ needs human check ({note})" if c["needs_human"] else f"  ({note})"
        cited = " ".join(by_id[i].text for i in v.cited_clause_ids if i in by_id)
        print(f"{ICON[v.status.value]} {v.status.value.upper():<6}{reason:<14} {rule.title}{flag}")
        if res.llm["quote"]:
            print(f"     quote: \"{res.llm['quote']}\"  [{', '.join(v.cited_clause_ids)}]")
        print(f"     why:   {v.rationale}\n")
        md += [f"## {ICON[v.status.value]} {rule.title}: **{v.status.value.upper()}**{reason}", "",
               f"**Why:** {v.rationale}", ""]
        if flag:
            md += [f"> {flag.strip()}", ""]
        if cited:
            md += [f"**Cited excerpt** ({', '.join(v.cited_clause_ids)}):", "", f"> {cited[:1500]}", ""]

    cost = spent_usd() - before
    summary = (f"{counts['green']} green · {counts['yellow']} yellow · {counts['red']} red   |   "
               f"${cost:.4f}{' (served from cache)' if cost == 0 else ''} · {wall:.1f}s")
    print(summary)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{doc_id[:60].replace(' ', '_')}.md"
    md.insert(5, f"- Result: {summary}")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"Report: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
