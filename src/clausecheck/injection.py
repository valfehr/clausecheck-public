"""Phase 7.4: spot text in a contract that is addressed to an AI reviewer, and neutralise our own prompt delimiters.

The judge resisted all 40 planted attacks (PLAN.md 7.4), but never told the lawyer the contract contained them.
A contract that talks to "any AI reviewer" is itself a finding a lawyer should see, so this detector flags the whole
review for a closer look. Deterministic and free; tuned on CUAD for false alarms (target: 0 on 510 real contracts).
"""

from __future__ import annotations

import re

PATTERNS = {
    "addresses an AI": r"\b(?:AI|artificial intelligence|language model|LLM|automated)\s+(?:system|reviewer|review|model|assistant|tool)s?\b"
                       r"(?=.{0,200}\b(?:ignore|mark|output|return|treat|approve|disregard)\b)",
    "override instruction": r"\b(?:ignore|disregard)\s+(?:all\s+|any\s+|your\s+|the\s+|previous\s+|prior\s+)*(?:rubric|instructions?|rules?|criteria|guidelines)\b",
    "prompt structure": r"</?\s*(?:agreement|system|instructions?)\s*>",
    "ready-made verdict": r"\"(?:status|red_reason|confidence)\"\s*:\s*\"",
    "reviewer directive": r"\b(?:any|the)\s+reviewer[^.]{0,80}\bshall\s+(?:treat|mark|deem|consider)\b",
    "verdict directive": r"\b(?:return|output|mark|rate|classify)\b[^.\]]{0,40}\b(?:green|yellow|red)\b[^.\]]{0,40}"
                         r"\b(?:rules?|confidence|status)\b",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE | re.MULTILINE) for k, v in PATTERNS.items()}
# role markers are matched case-SENSITIVELY: "SYSTEM:" is an attack pattern, "System:" is a heading in real
# contracts (one CUAD false alarm with the case-insensitive version)
_COMPILED["role marker"] = re.compile(r"^\s*(?:SYSTEM|ASSISTANT|USER)\s*:", re.MULTILINE)


def scan(text: str) -> list[dict]:
    """Matches as [{kind, excerpt}], empty if clean."""
    hits = []
    for kind, rx in _COMPILED.items():
        for m in rx.finditer(text):
            a, b = max(m.start() - 60, 0), min(m.end() + 60, len(text))
            hits.append({"kind": kind, "excerpt": " ".join(text[a:b].split())})
    return hits


def neutralise_delimiters(text: str) -> str:
    """Contract text can't open or close our prompt's own data block."""
    return re.sub(r"</?\s*agreement\s*>", "[tag removed]", text, flags=re.IGNORECASE)
