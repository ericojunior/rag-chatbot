from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .chunking import Chunk
from .index import VectorIndex


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def retrieve(index: VectorIndex, query_embedding: np.ndarray, *, top_k: int = 5) -> List[RetrievedChunk]:
    """
    Recupera top_k chunks via similaridade coseno (vetores já normalizados).
    """
    if top_k <= 0:
        raise ValueError("top_k deve ser > 0")

    q = query_embedding.astype(np.float32).reshape(1, -1)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    # Como está tudo normalizado, cos sim = produto escalar
    scores = (index.embeddings @ q.T).reshape(-1)
    k = min(top_k, scores.shape[0])
    top_idx = np.argpartition(-scores, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    out: List[RetrievedChunk] = []
    for i in top_idx:
        out.append(RetrievedChunk(chunk=index.chunks[int(i)], score=float(scores[int(i)])))
    return out

