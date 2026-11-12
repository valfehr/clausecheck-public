"""Split a contract into clause-sized segments with exact character offsets.

CUAD text comes from PDF extraction. 504/510 contracts keep line breaks plus long runs of layout
spaces; 6 are a single line. (An earlier version assumed "single line" from the first contract
inspected; that missed section numbers at line starts and fed 30-space gaps to the model.)
Structure signals: numbering "1.", "1.1", "12.3(b)", "Section 4", "ARTICLE IV" at a line start or
after a sentence boundary.
Strategy:
1. Cut before numbering markers that start a line or follow a sentence/heading boundary.
2. Merge tiny fragments (bare headings like "1. DEFINITIONS") into the next segment.
3. Split oversized segments on sentence boundaries, to cap how much surrounding text an
   excerpt drags along (the "not too much before or after" problem).

The judge cites segment IDs, so excerpt boundaries always come from here, never from the LLM.
"""

from __future__ import annotations

import re

from clausecheck.schemas import Clause

# A numbering marker: "Section 4.2", "ARTICLE IV", "12.3", "7." — preceded by start or a boundary char.
_MARKER = re.compile(
    r"""(?:(?<=^)|(?<=[.;:"”)\]]\s)|(?<=[A-Z]{3}\s)|(?<=\n))  # start, sentence end, ALL-CAPS heading word, or line start
        [ \t]*                                                 # indentation before the number
        (?=(?:(?:Section|SECTION|Article|ARTICLE)\s+[0-9IVXLivxl]+(?:\.\d+)*
            |\d{1,2}\.(?:\d{1,2}\.?){0,3}\s+(?=[A-Z"“(])))""",
    re.VERBOSE,
)
_LABEL = re.compile(r"^((?:Section|SECTION|Article|ARTICLE)\s+[0-9IVXLivxl]+(?:\.\d+)*|\d{1,2}(?:\.\d{1,2}){0,3}\.?)")
_SENTENCE_END = re.compile(r"(?<=[.;])\s+(?=[A-Z(“\"])")

MIN_CHARS = 120
# Measured on 3,810 CUAD spans (reports/phase1_segmenter.md):
#   1500 → 95% of spans in one segment, 4.4x excess text | 900 → 94%, 3.0x | 600 → 92%, 2.3x
# 900 keeps nearly all clauses whole while cutting excess by a third.
MAX_CHARS = 900


def _split_long(start: int, text: str) -> list[tuple[int, int]]:
    """Split [start, start+len(text)) into ~MAX_CHARS pieces on sentence boundaries."""
    if len(text) <= MAX_CHARS:
        return [(start, start + len(text))]
    cuts = [0]
    last = 0
    for m in _SENTENCE_END.finditer(text):
        if m.start() - last >= MAX_CHARS * 0.6:
            cuts.append(m.end())
            last = m.end()
    cuts.append(len(text))
    return [(start + a, start + b) for a, b in zip(cuts, cuts[1:]) if b > a]


def segment(doc_id: str, text: str) -> list[Clause]:
    starts = sorted({0, *(m.start() for m in _MARKER.finditer(text))})
    raw = [(a, b) for a, b in zip(starts, starts[1:] + [len(text)]) if text[a:b].strip()]

    # Merge short fragments (headings, stray numbers) forward.
    merged: list[tuple[int, int]] = []
    carry: int | None = None
    for a, b in raw:
        a = carry if carry is not None else a
        if b - a < MIN_CHARS:
            carry = a
            continue
        merged.append((a, b))
        carry = None
    if carry is not None:
        if merged:
            merged[-1] = (merged[-1][0], len(text))
        else:
            merged.append((carry, len(text)))

    spans: list[tuple[int, int]] = []
    for a, b in merged:
        spans.extend(_split_long(a, text[a:b]))

    clauses = []
    for i, (a, b) in enumerate(spans):
        chunk = text[a:b]
        # Trim whitespace but keep offsets exact.
        lead = len(chunk) - len(chunk.lstrip())
        trail = len(chunk) - len(chunk.rstrip())
        a2, b2 = a + lead, b - trail
        body = text[a2:b2]
        label = _LABEL.match(body)
        clauses.append(
            Clause(id=f"c{i}", doc_id=doc_id, path=[label.group(1)] if label else [], text=body, start_char=a2, end_char=b2)
        )
    return clauses


def render_for_prompt(clauses: list[Clause]) -> str:
    """Contract as the model sees it: one segment per line, prefixed by its ID, whitespace collapsed.

    Collapsing matters: PDF layout spaces made prompts ~29% longer and produced degenerate
    (empty) verdicts. Offsets stay exact because only the rendered copy is normalized; the
    quote verifier normalizes whitespace the same way.
    """
    return "\n".join(f"[{c.id}] {re.sub(r'\s+', ' ', c.text)}" for c in clauses)


def covering_clauses(clauses: list[Clause], start: int, end: int) -> list[str]:
    """IDs of segments overlapping a character span."""
    return [c.id for c in clauses if c.start_char < end and c.end_char > start]
