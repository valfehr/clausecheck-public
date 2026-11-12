"""Local embedding + reranking models (free, CPU). Models are cached in <project>/models/.

- Embeddings: BAAI/bge-small-en-v1.5 (384 dims, ~130 MB). bge models want a short instruction
  prefix on QUERIES only (not on passages).
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (~90 MB): scores (query, passage) pairs jointly,
  slower than vectors but more precise; used on the top-N candidates only.
Both are general-domain models; whether they work on legal text is something we measure.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "models"))

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EMBED_DIMS = 384
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, device=_device())


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL, device=_device())


# One model call at a time: concurrent calls from several threads on the same GPU model crashed with
# "CUDA error: an illegal memory access" (it was fine on CPU). Each call takes milliseconds, so a lock is cheap.
_GPU_LOCK = threading.Lock()


def embed_passages(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    with _GPU_LOCK:
        return _embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                  show_progress_bar=False).tolist()


def embed_query(text: str) -> list[float]:
    with _GPU_LOCK:
        return _embedder().encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()


def rerank(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    with _GPU_LOCK:
        return [float(s) for s in _reranker().predict([(query, p) for p in passages])]
