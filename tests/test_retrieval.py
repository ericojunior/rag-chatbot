import numpy as np

from rag_chatbot.chunking import Chunk
from rag_chatbot.index import VectorIndex
from rag_chatbot.retrieval import retrieve


def test_retrieval_picks_best_match():
    chunks = [
        Chunk(id=0, page=1, text="primeiro"),
        Chunk(id=1, page=2, text="segundo"),
        Chunk(id=2, page=3, text="terceiro"),
    ]

    # embeddings já normalizados
    emb = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ],
        dtype=np.float32,
    )
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    idx = VectorIndex(chunks=chunks, embeddings=emb, model_name="dummy")

    q = np.array([0.0, 1.0], dtype=np.float32)
    hits = retrieve(idx, q, top_k=1)
    assert hits[0].chunk.id == 1
