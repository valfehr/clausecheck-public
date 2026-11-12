"""Loader for the CUAD v1 dataset (Atticus Project).

CUAD_v1.json is SQuAD-format: one entry per contract, the full text as `context`,
and one question per clause category with lawyer-highlighted answer spans
(`answer_start` = char offset into `context`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

CUAD_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "cuad" / "CUAD_v1"

# Folder names differ slightly between Part_I/II/III; normalize to one family label.
_FAMILY_ALIASES = {
    "affiliate agreements": "Affiliate",
    "affiliate agreement": "Affiliate",
    "agency agreements": "Agency",
    "co branding": "Co-Branding",
    "consulting agreements": "Consulting",
    "endorsement agreement": "Endorsement",
    "joint venture filing": "Joint Venture",
    "license agreements": "License",
    "non compete non solicit": "Non-Compete/Non-Solicit",
}


def _normalize_family(folder: str) -> str:
    key = re.sub(r"[_\s]+", " ", folder).strip().lower()
    return _FAMILY_ALIASES.get(key, key.title())


@dataclass
class Span:
    start: int
    end: int
    text: str


@dataclass
class Contract:
    doc_id: str          # CUAD title (file stem)
    family: str          # contract type, used for train/test splits
    text: str
    labels: dict[str, list[Span]] = field(default_factory=dict)  # category -> spans (empty = clause absent)


@lru_cache(maxsize=1)
def _family_by_stem() -> dict[str, str]:
    """PDFs live in Part_X/<family>/..., txt files don't; map stem -> family."""
    out: dict[str, str] = {}
    for pdf in (CUAD_DIR / "full_contract_pdf").rglob("*.pdf"):
        family_folder = pdf.relative_to(CUAD_DIR / "full_contract_pdf").parts[1]
        out[_normalize_family_stem(pdf.stem)] = _normalize_family(family_folder)
    return out


def _normalize_family_stem(stem: str) -> str:
    return re.sub(r"\W+", "", stem).lower()


def load_contracts() -> list[Contract]:
    raw = json.loads((CUAD_DIR / "CUAD_v1.json").read_text(encoding="utf-8"))
    families = _family_by_stem()
    contracts = []
    for entry in raw["data"]:
        para = entry["paragraphs"][0]
        text = para["context"]
        labels: dict[str, list[Span]] = {}
        for qa in para["qas"]:
            category = qa["id"].split("__")[-1]
            labels[category] = [
                Span(a["answer_start"], a["answer_start"] + len(a["text"]), a["text"]) for a in qa["answers"]
            ]
        contracts.append(
            Contract(
                doc_id=entry["title"],
                family=families.get(_normalize_family_stem(entry["title"]), "Unknown"),
                text=text,
                labels=labels,
            )
        )
    return contracts
