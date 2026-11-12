"""Who's who per contract: extracted once at ingestion, stored as document metadata, injected into every prompt.

Why (Phase 5): the tuned 4B model kept calling the client "the counterparty". The preamble that defines the
parties is rarely among the RAG top-10 excerpts, and the prompt named only the client, so the model had to
infer who the counterparty was. Stable per-document facts like this shouldn't depend on retrieval luck:
pull them out once, store them next to the index, and state them explicitly at query time.

Storage: a small Elasticsearch index `contract_meta` (one doc per doc_key). In production this would be a
document-level table or a field on the parent document; the point is that it's structured, not retrieved.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

META_INDEX = "contract_meta"


class Party(BaseModel):
    name: str  # full legal name as written, e.g. "Adaptimmune Limited"
    aliases: list[str]  # short names / defined terms used in the body, e.g. ["Adaptimmune Limited"]


class Parties(BaseModel):
    parties: list[Party]


EXTRACT_SYSTEM = """You read the opening of a contract. List every contracting party (not affiliates merely
mentioned, not witnesses). For each: its full name as written, and every short name or defined term the
contract uses for it (e.g. "MD Anderson", "Company", "Supplier"). Keep names exactly as written."""


def extract(adapter, text: str) -> Parties:
    """Empty Parties on failure: the prompt then falls back to naming the client only (never a guess)."""
    try:
        return adapter.complete(system=EXTRACT_SYSTEM, user=text[:3500], schema=Parties, max_tokens=3000).parsed
    except Exception as e:  # noqa: BLE001 — e.g. output cut off at max_tokens
        print(f"  party extraction failed ({str(e)[:80]}); no metadata for this contract")
        return Parties(parties=[])


def store(es, doc_key: str, parties: Parties) -> None:
    if not es.indices.exists(index=META_INDEX):
        es.indices.create(index=META_INDEX, mappings={"properties": {"parties": {"type": "object", "enabled": False}}})
    es.index(index=META_INDEX, id=doc_key, document={"parties": parties.model_dump()["parties"]})


def load(es, doc_key: str) -> Parties | None:
    if not es.exists(index=META_INDEX, id=doc_key):
        return None
    return Parties(parties=es.get(index=META_INDEX, id=doc_key)["_source"]["parties"])


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def counterparty_for(parties: Parties | None, client: str) -> str:
    """Names of every party that isn't the client, or "" if the client can't be matched (then the prompt
    stays as before rather than guessing)."""
    if not parties:
        return ""
    c = _norm(client)
    is_client = [any(c and (c == _norm(n) or c in _norm(n) or _norm(n) in c) for n in [p.name, *p.aliases] if _norm(n))
                 for p in parties.parties]
    if sum(is_client) != 1:
        return ""
    others = [p for p, m in zip(parties.parties, is_client) if not m]
    return "; ".join(p.name + (f' ("{p.aliases[0]}")' if p.aliases and p.aliases[0] != p.name else "") for p in others)
