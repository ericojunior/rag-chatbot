# Chatbot RAG (Perguntas & Respostas com PDF)
Trabalho de MBA

Mini projeto (MVP) de **RAG – Retrieval Augmented Generation**: você faz upload de um **PDF**, o app:

1. Extrai o texto por página
2. Quebra em *chunks* (trechos)
3. Gera embeddings
4. Faz busca semântica (cosine similarity)
5. (Opcional) Chama um LLM para montar a resposta com base nos trechos recuperados

## Stack escolhida

- Python 3.10+
- Streamlit (UI)
- sentence-transformers (embeddings)
- scikit-learn + numpy (similaridade)
- pdfplumber/pypdf (extração de texto)
- OpenAI (opcional, via `OPENAI_API_KEY`)

## Como rodar

```bash
cd rag-chatbot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --break-system-packages

streamlit run app.py
```

Abra o link que o Streamlit mostrar, faça upload de um PDF e pergunte.

## Configuração (opcional: LLM)

Crie um arquivo `.env` (veja `.env.example`):

```env
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://api.openai.com/v1
```

Se **não** houver chave, o app entra em modo *fallback* e retorna os trechos mais relevantes do PDF (com páginas), sem gerar texto por LLM.

## Estrutura

```
rag-chatbot/
  app.py
  src/rag_chatbot/
    pdf_text.py
    chunking.py
    index.py
    retrieval.py
    llm.py
    rag.py
  tests/
```

## Testes

```bash
pytest -q
```

---

## Autor

**Érico Júnior de Morais**  
*Projeto desenvolvido para o MBA em Engenharia de Software com Inteligência Artificial*  
*CEIA - UFG - GOIÁS - BRASIL*
**Abril de 2026**
