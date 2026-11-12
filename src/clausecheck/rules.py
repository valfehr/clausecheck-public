from __future__ import annotations

from pathlib import Path

import yaml

from clausecheck.schemas import Rule

RULES_PATH = Path(__file__).resolve().parents[2] / "data" / "rules" / "golden_rules.yaml"


def load_rules(path: Path = RULES_PATH) -> list[Rule]:
    return [Rule(**r) for r in yaml.safe_load(path.read_text(encoding="utf-8"))["rules"]]
