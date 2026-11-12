"""Exit non-zero if an eval run fails the release gates (used by CI's paid eval workflow).

Gates (PLAN.md, fixed since Phase 1): red recall ≥ 90%, zero critical misses (red → green), citations verified ≥ 90%.

  uv run python scripts/check_gates.py RUN_NAME
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    name = sys.argv[1]
    f = sorted((ROOT / "runs").glob(f"*_{name}.jsonl"))[-1]
    rows = [r for r in map(json.loads, f.read_text(encoding="utf-8").split("\n")) if r.get("pred_status")]
    errors = sum(1 for r in map(json.loads, f.read_text(encoding="utf-8").split("\n")) if "error" in r)
    reds = [r for r in rows if r.get("expected_status") == "red"]
    recall = sum(r["pred_status"] == "red" for r in reds) / max(len(reds), 1)
    critical = sum(r["pred_status"] == "green" for r in reds)
    verified = sum(r["citation_valid"] for r in rows) / max(len(rows), 1)
    checks = {"red recall ≥ 90%": recall >= 0.9, "0 critical misses": critical == 0,
              "citations verified ≥ 90%": verified >= 0.9, "no errors": errors == 0}
    print(f"{name}: red recall {recall:.0%} ({len(reds)} reds), critical misses {critical}, "
          f"citations verified {verified:.0%}, errors {errors}")
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
