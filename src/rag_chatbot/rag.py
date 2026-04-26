from __future__ import annotations

from typing import Any, Dict
from functools import lru_cache

import numpy as np

from .index import VectorIndex
from .llm import fallback_answer, generate_answer, has_llm
from .retrieval import retrieve


@lru_cache(maxsize=4)
def _get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name)


def _embed_query(model_name: str, query: str) -> np.ndarray:
    model = _get_embedding_model(model_name)
    v = model.encode([query], show_progress_bar=False)
    return np.asarray(v[0], dtype=np.float32)


def answer_question(index: VectorIndex, question: str, *, top_k: int = 5) -> Dict[str, Any]:
    q_emb = _embed_query(index.model_name, question)
    hits = retrieve(index, q_emb, top_k=top_k)

    sources = [
        {"page": h.chunk.page, "score": h.score, "text": h.chunk.text}
        for h in hits
    ]
    contexts = [s["text"] for s in sources]

    if has_llm():
        answer = generate_answer(question=question, contexts=contexts)
    else:
        answer = fallback_answer(question=question, contexts=contexts)

    return {"answer": answer, "sources": sources}
