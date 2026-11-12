"""Retrieval recall: for each (contract, rule) with a known relevant clause, does search find it? $0.

Ground truth = segments overlapping the expected spans (CUAD lawyer spans, or the synthetic
clause + any override/definition placed elsewhere). Cases without spans (missing clauses) are
skipped here: they test the judge, not retrieval.

  hit@k   any expected segment in the top k     (enough for most rules)
  all@k   ALL expected segments in the top k    (hard cases: clause + far-away override)
  MRR     1 / rank of the first expected segment

Only dev and test are scored: holdout stays untouched for release decisions.
  uv run python scripts/eval_retrieval.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clausecheck import search  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.segment import covering_clauses, segment  # noqa: E402
from run_eval import load_cases  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "phase2_retrieval.md"
KS = [1, 3, 5, 10]


def main() -> None:
    es = search.client()
    rules = {r.id: r for r in load_rules()}
    # --real: only unmodified CUAD contracts with lawyer spans (no synthetic text sharing the rule's wording)
    real = "--real" in sys.argv
    global OUT
    suites = ["retrieval"] if real else ["severity", "severity_hard", "retrieval"]
    if real:
        OUT = ROOT / "reports" / "phase2_retrieval_real_cuad.md"
    cases = [c for c in load_cases(suites, ["dev", "test"]) if c.expected_spans]
    lines = ["# Phase 2 — retrieval recall (Elasticsearch, per contract, per rule)", "",
             f"{len(cases)} (contract, rule) cases with a known relevant clause; dev + test only"
             + (" — **real CUAD lawyer spans only**" if real else " (synthetic + real)") + ".", ""]
    table = ["| method | split | hit@1 | hit@3 | hit@5 | hit@10 | all@5 (multi-span) | MRR | ms/query |",
             "|---|---|---|---|---|---|---|---|---|"]
    by_rule_rows = []
    for name, fn in search.METHODS.items():
        for split in ("dev", "test"):
            sub = [c for c in cases if c.split == split]
            hits = {k: 0 for k in KS}
            all5, multi, rr, ms = 0, 0, [], []
            per_rule: dict[str, list[int]] = {}
            for c in sub:
                expected = set()
                for a, b in c.expected_spans:
                    expected.update(covering_clauses(segment(c.doc_id, c.text), a, b))
                q = search.rule_query(rules[c.rule_id])
                t0 = time.perf_counter()
                got = [sid for sid, _ in fn(es, c.case_id, q, 10)]
                ms.append((time.perf_counter() - t0) * 1000)
                for k in KS:
                    hits[k] += bool(expected & set(got[:k]))
                first = next((i for i, s in enumerate(got, 1) if s in expected), None)
                rr.append(1 / first if first else 0.0)
                if len(c.expected_spans) > 1:
                    multi += 1
                    all5 += expected <= set(got[:5])
                per_rule.setdefault(c.rule_id, [0, 0])
                per_rule[c.rule_id][0] += bool(expected & set(got[:5]))
                per_rule[c.rule_id][1] += 1
            n = len(sub)
            table.append(f"| {name} | {split} | " + " | ".join(f"{100 * hits[k] / n:.0f}%" for k in KS)
                         + f" | {f'{all5}/{multi}' if multi else '–'} | {statistics.mean(rr):.2f} | {statistics.median(ms):.0f} |")
            if split == "test":
                by_rule_rows.append((name, per_rule))
        print(table[-2]); print(table[-1])
    rule_ids = list(rules)
    lines += table + ["", "## hit@5 by rule (test)", "",
                      "| method | " + " | ".join(rule_ids) + " |", "|---|" + "---|" * len(rule_ids)]
    for name, pr in by_rule_rows:
        lines.append(f"| {name} | " + " | ".join(f"{pr[r][0]}/{pr[r][1]}" if r in pr else "–" for r in rule_ids) + " |")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
