"""How well do segments line up with lawyer-labelled CUAD spans? Costs $0.

For every labelled span of our 8 rule categories:
- n_segments: how many segments the span touches (1 = clean, the excerpt is one unit)
- excess: (chars of covering segments) / (span chars), i.e. how much extra text an excerpt drags along
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from clausecheck.cuad import load_contracts  # noqa: E402
from clausecheck.rules import load_rules  # noqa: E402
from clausecheck.segment import covering_clauses, segment  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "phase1_segmenter.md"


def main() -> None:
    cats = {r.cuad_category for r in load_rules()}
    seg_counts, excess, seg_lens = [], [], []
    for c in load_contracts():
        clauses = segment(c.doc_id, c.text)
        seg_lens += [len(x.text) for x in clauses]
        by_id = {x.id: x for x in clauses}
        for cat in cats:
            for s in c.labels.get(cat, []):
                ids = covering_clauses(clauses, s.start, s.end)
                seg_counts.append(len(ids))
                excess.append(sum(len(by_id[i].text) for i in ids) / max(s.end - s.start, 1))

    n = len(seg_counts)
    pct = lambda k: 100 * sum(1 for x in seg_counts if x <= k) / n  # noqa: E731
    lines = [
        "# Phase 1 — segmenter vs CUAD lawyer spans (8 rule categories)",
        "",
        f"- Labelled spans checked: **{n}**",
        f"- Span inside 1 segment: **{pct(1):.0f}%**, within ≤2: **{pct(2):.0f}%**, within ≤3: {pct(3):.0f}%",
        f"- Excess context (covering segment chars / span chars): median **{statistics.median(excess):.1f}x**, "
        f"p90 {sorted(excess)[int(0.9 * n)]:.1f}x",
        f"- Segment length: median {statistics.median(seg_lens):.0f} chars, p90 {sorted(seg_lens)[int(0.9 * len(seg_lens))]} chars, "
        f"total segments {len(seg_lens):,}",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
