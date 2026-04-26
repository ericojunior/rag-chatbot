from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    id: int
    page: int
    text: str


def chunk_text_by_chars(
    page: int,
    text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    start_id: int = 0,
) -> List[Chunk]:
    """
    Quebra texto em chunks por caracteres com overlap (simples e robusto).

    - chunk_size: tamanho aproximado do chunk
    - chunk_overlap: quantos caracteres “voltam” entre chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap deve ser >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap deve ser menor que chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: List[Chunk] = []
    i = 0
    chunk_id = start_id
    step = chunk_size - chunk_overlap

    while i < len(cleaned):
        piece = cleaned[i : i + chunk_size].strip()
        if piece:
            chunks.append(Chunk(id=chunk_id, page=page, text=piece))
            chunk_id += 1
        i += step

    return chunks

