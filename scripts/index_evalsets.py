"""Index every eval case's contract into Elasticsearch (one doc_key per case: synthetic cases
have their own modified text). Costs $0: local embeddings.

  uv run python scripts/index_evalsets.py [--recreate] [--suites severity_long]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clausecheck import search  # noqa: E402
from clausecheck.segment import segment  # noqa: E402
from run_eval import load_cases  # noqa: E402


def main() -> None:
    es = search.client()
    search.create_index(es, recreate="--recreate" in sys.argv)
    suites = (sys.argv[sys.argv.index("--suites") + 1:] if "--suites" in sys.argv
              else ["severity", "severity_hard", "retrieval"])
    cases = load_cases(suites, ["dev", "test", "holdout"])
    t0, n = time.perf_counter(), 0
    for i, c in enumerate(cases, 1):
        n += search.index_document(es, c.case_id, c.doc_id, segment(c.doc_id, c.text))
        if i % 25 == 0:
            print(f"  {i}/{len(cases)} contracts, {n:,} segments, {time.perf_counter() - t0:.0f}s")
    es.indices.refresh(index=search.INDEX)
    stats = es.indices.stats(index=search.INDEX)["indices"][search.INDEX]["total"]
    print(f"Indexed {len(cases)} contracts, {n:,} segments in {time.perf_counter() - t0:.0f}s; "
          f"index size {stats['store']['size_in_bytes'] / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
