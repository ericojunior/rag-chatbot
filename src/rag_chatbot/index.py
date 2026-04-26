from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from functools import lru_cache
from typing import BinaryIO, List

import numpy as np

from .chunking import Chunk, chunk_text_by_chars
from .pdf_text import extract_pdf_text


@dataclass(frozen=True)
class VectorIndex:
    chunks: List[Chunk]
    embeddings: np.ndarray  # shape: (n_chunks, dim), normalizado
    model_name: str


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


@lru_cache(maxsize=4)
def _get_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name)


def build_index(
    *,
    pdf_file: BinaryIO,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> VectorIndex:
    pages = extract_pdf_text(pdf_file)

    all_chunks: List[Chunk] = []
    next_id = 0
    for p in pages:
        pcs = chunk_text_by_chars(
            p.page,
            p.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_id=next_id,
        )
        all_chunks.extend(pcs)
        next_id += len(pcs)

    if not all_chunks:
        raise ValueError("Não foi possível extrair texto do PDF (ou o PDF está vazio).")

    model = _get_embedding_model(embedding_model)
    texts = [c.text for c in all_chunks]
    emb = model.encode(texts, batch_size=32, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    emb = _normalize_rows(emb)

    return VectorIndex(chunks=all_chunks, embeddings=emb, model_name=embedding_model)
