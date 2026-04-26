from rag_chatbot.chunking import chunk_text_by_chars


def test_chunking_overlap_basic():
    text = "a" * 1200
    chunks = chunk_text_by_chars(1, text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 3
    assert chunks[0].page == 1
    # garante que overlap realmente acontece
    assert chunks[0].text[-50:] == chunks[1].text[:50]

