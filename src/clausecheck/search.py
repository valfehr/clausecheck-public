"""Clause retrieval on Elasticsearch: BM25, dense vectors, hybrid (RRF) and reranked hybrid.

Index layout (one Elasticsearch document per contract segment):
  doc_key   keyword   which contract (an eval case id, or a reviewed file's id): every query filters on it
  seg_id    keyword   segment id from the segmenter ("c12"): what the judge cites
  header    text      contextual chunk header: contract name + section label
  text      text      segment text (english analyzer: stemming, stopwords) → BM25
  embedding dense_vector(384, cosine, HNSW) of header + text → semantic search

On AWS the same mapping works on Amazon OpenSearch (k-NN plugin), with IAM/SigV4 instead of no auth.
"""

from __future__ import annotations

import re

from elasticsearch import Elasticsearch, helpers

from clausecheck.embed import EMBED_DIMS, embed_passages, embed_query, rerank
from clausecheck.schemas import Clause, Rule

INDEX = "clauses"
RRF_K = 60  # standard reciprocal-rank-fusion constant


def client() -> Elasticsearch:
    return Elasticsearch("http://127.0.0.1:9200", request_timeout=60)


def create_index(es: Elasticsearch, recreate: bool = False) -> None:
    if es.indices.exists(index=INDEX):
        if not recreate:
            return
        es.indices.delete(index=INDEX)
    es.indices.create(
        index=INDEX,
        settings={"number_of_shards": 1, "number_of_replicas": 0},
        mappings={"properties": {
            "doc_key": {"type": "keyword"},
            "seg_id": {"type": "keyword"},
            "seg_idx": {"type": "integer"},
            "header": {"type": "text", "analyzer": "english"},
            "text": {"type": "text", "analyzer": "english"},
            "embedding": {"type": "dense_vector", "dims": EMBED_DIMS, "index": True, "similarity": "cosine"},
        }},
    )


def header_for(title: str, clause: Clause) -> str:
    sec = f"Section {clause.path[0]}" if clause.path else "Unnumbered passage"
    return f"{title[:120]}. {sec}."


def index_document(es: Elasticsearch, doc_key: str, title: str, clauses: list[Clause]) -> int:
    texts = [re.sub(r"\s+", " ", c.text) for c in clauses]
    headers = [header_for(title, c) for c in clauses]
    vectors = embed_passages([h + " " + t for h, t in zip(headers, texts)])
    actions = [
        {"_index": INDEX, "_id": f"{doc_key}::{c.id}", "_source": {
            "doc_key": doc_key, "seg_id": c.id, "seg_idx": i, "header": h, "text": t, "embedding": v}}
        for i, (c, h, t, v) in enumerate(zip(clauses, headers, texts, vectors))
    ]
    helpers.bulk(es, actions)
    return len(actions)


def rule_query(rule: Rule) -> str:
    return f"{rule.title}. {rule.ideal_text}"


def _hits(resp) -> list[tuple[str, str]]:
    return [(h["_source"]["seg_id"], h["_source"]["text"]) for h in resp["hits"]["hits"]]


def bm25(es: Elasticsearch, doc_key: str, query: str, k: int) -> list[tuple[str, str]]:
    resp = es.search(index=INDEX, size=k, source=["seg_id", "text"], query={"bool": {
        "must": {"multi_match": {"query": query, "fields": ["text", "header^0.5"]}},
        "filter": {"term": {"doc_key": doc_key}}}})
    return _hits(resp)


def dense(es: Elasticsearch, doc_key: str, query: str, k: int) -> list[tuple[str, str]]:
    resp = es.search(index=INDEX, size=k, source=["seg_id", "text"], knn={
        "field": "embedding", "query_vector": embed_query(query), "k": k, "num_candidates": 200,
        "filter": {"term": {"doc_key": doc_key}}})
    return _hits(resp)


def hybrid(es: Elasticsearch, doc_key: str, query: str, k: int, pool: int = 30) -> list[tuple[str, str]]:
    """Reciprocal rank fusion: score = sum over rankers of 1 / (RRF_K + rank). Rank-based, so BM25 and
    cosine scores never need to be put on the same scale."""
    lists = [bm25(es, doc_key, query, pool), dense(es, doc_key, query, pool)]
    scores: dict[str, float] = {}
    text: dict[str, str] = {}
    for hits in lists:
        for rank, (sid, t) in enumerate(hits, start=1):
            scores[sid] = scores.get(sid, 0.0) + 1.0 / (RRF_K + rank)
            text[sid] = t
    return [(sid, text[sid]) for sid in sorted(scores, key=scores.get, reverse=True)[:k]]


def hybrid_rerank(es: Elasticsearch, doc_key: str, query: str, k: int, pool: int = 20) -> list[tuple[str, str]]:
    cands = hybrid(es, doc_key, query, pool)
    scores = rerank(query, [t for _, t in cands])
    return [c for _, c in sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)[:k]]


METHODS = {"bm25": bm25, "dense": dense, "hybrid": hybrid, "hybrid+rerank": hybrid_rerank}
