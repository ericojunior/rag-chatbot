# Chatbot RAG (Perguntas & Respostas com PDF)
Desenvolvido com o **Antigravity IDE** 🚀


Mini projeto (MVP) de **RAG – Retrieval Augmented Generation**: você faz upload de um **PDF**, o app:

1. Extrai o texto por página
2. Quebra em *chunks* (trechos)
3. Gera embeddings
4. Faz busca semântica (cosine similarity)
5. (Opcional) Chama um LLM (OpenAI ou DeepSeek) para montar a resposta com base nos trechos recuperados

### Destaques desta versão:
- **Identidade HyFit:** Interface personalizada com cores e logo da HyFit.
- **Provedores Dinâmicos:** Troca automática de logos e modelos baseada na configuração do `.env`.
- **Modo Fallback Inteligente:** Funciona 100% offline (para busca) caso a chave de API não esteja disponível.
- **Tratamento de Erros:** Mensagens amigáveis para problemas de saldo ou conexão com a API.

## Stack escolhida

- Python 3.10+
- Streamlit (UI)
- sentence-transformers (embeddings)
- scikit-learn + numpy (similaridade)
- pdfplumber/pypdf (extração de texto)
- OpenAI (opcional, via `OPENAI_API_KEY`)

## Instalação e Execução

Para rodar o projeto localmente, siga os passos abaixo:

1. **Clonar o repositório:**
   ```bash
   git clone <url-do-repositorio>
   cd rag-chatbot
   ```

2. **Configurar o Ambiente Virtual:**
   ```bash
   # No macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate

   # No Windows:
   # python -m venv .venv
   # .venv\Scripts\activate
   ```

3. **Instalar Dependências:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Rodar o App:**
   ```bash
   streamlit run app.py
   ```

Abra o link que o Streamlit mostrar, faça upload de um PDF e pergunte.

## Configuração de LLM

O app suporta múltiplos provedores. Crie/edite o arquivo `.env` na raiz do projeto:

### Para usar OpenAI:
```env
LLM_PROVIDER=openai
LLM_API_KEY=sua_chave_aqui
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

### Para usar DeepSeek:
```env
LLM_PROVIDER=deepseek
LLM_API_KEY=sua_chave_aqui
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

> [!TIP]
> A interface do app (logo e provedor) mudará automaticamente assim que você salvar o arquivo `.env`.

### Modo Fallback (Sem API Key)

Se nenhuma chave de API for configurada no `.env`, o sistema entra automaticamente em **Modo Fallback**. 
Neste modo:
1. O PDF é processado e indexado normalmente via **Embeddings Locais** (`sentence-transformers`).
2. A busca semântica continua funcionando via **Cosine Similarity**.
3. Em vez de uma resposta gerada por IA, o app retorna os **Top-K trechos mais relevantes** diretamente do documento, indicando as páginas originais.

Isso garante que o projeto seja testável mesmo sem saldo em provedores externos.

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
*Desenvolvido utilizando o **Antigravity IDE** da Google DeepMind.*

**Abril de 2026**
