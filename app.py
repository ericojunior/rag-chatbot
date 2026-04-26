import os
import sys
from pathlib import Path
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

# Garante que `src/` esteja no PYTHONPATH ao rodar via `streamlit run app.py`
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag_chatbot.index import build_index  # noqa: E402
from rag_chatbot.rag import answer_question  # noqa: E402


load_dotenv()

st.set_page_config(page_title="Chatbot RAG (PDF) - HyFit", page_icon="📄", layout="wide")

# Estilo HyFit
st.markdown("""
    <style>
    .stButton>button {
        background-color: #2B8BFF;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1A73E8;
        color: white;
    }
    h1 {
        color: #2B8BFF;
    }
    .sidebar-logo {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Layout de Cabeçalho com Logo
col1, col2 = st.columns([1, 5])
with col1:
    if Path("assets/logo.svg").exists():
        st.image("assets/logo.svg", width=100)
with col2:
    st.title("HyFit Chatbot RAG")
    st.caption("Faça upload de um PDF, indexe e pergunte. (Powered by HyFit Tech)")


with st.sidebar:
    # Logo no topo da sidebar
    if Path("assets/logo.svg").exists():
        st.image("assets/logo.svg", width="stretch")
    
    st.header("Configurações")
    chunk_size = st.slider("Tamanho do chunk (caracteres)", 400, 2000, 1000, 50)
    chunk_overlap = st.slider("Overlap (caracteres)", 0, 500, 200, 25)
    top_k = st.slider("Top-K trechos recuperados", 1, 10, 5, 1)
    st.divider()
    
    # Seção LLM com Logo Dinâmica
    provider = (os.getenv("LLM_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "")).strip().lower()
    
    st.subheader("LLM (Inteligência)")
    
    # Escolha da logo baseada no provedor
    llm_logo = None
    if provider == "deepseek" and Path("assets/DeepSeek_logo.svg.png").exists():
        llm_logo = "assets/DeepSeek_logo.svg.png"
    elif provider == "openai" and Path("assets/openai-white-lockup.png").exists():
        llm_logo = "assets/openai-white-lockup.png"
    
    if llm_logo:
        st.image(llm_logo, width=180)
    
    has_key = bool(os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))
    model = (
        os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or ("deepseek-chat" if provider == "deepseek" else "gpt-4o-mini")
    )
    st.write("Provedor:", provider.capitalize())
    st.write("API key:", "✅ configurada" if has_key else "❌ não configurada (modo fallback)")
    st.write("Modelo:", model)


uploaded = st.file_uploader("Envie um PDF", type=["pdf"])

if "index" not in st.session_state:
    st.session_state.index = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


def _build_from_upload(data: bytes, filename: str):
    with st.spinner("Indexando PDF (extração → chunks → embeddings)…"):
        idx = build_index(
            pdf_file=BytesIO(data),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    st.session_state.index = idx
    st.session_state.pdf_name = filename


if uploaded is not None:
    data = uploaded.getvalue()
    if st.session_state.pdf_name != uploaded.name:
        _build_from_upload(data, uploaded.name)


if st.session_state.index is None:
    st.info("Envie um PDF para começar.")
    st.stop()

st.success(f"PDF indexado: **{st.session_state.pdf_name}**  •  chunks: **{len(st.session_state.index.chunks)}**")

question = st.text_input("Pergunta", placeholder="Ex.: Faça um resumo do documento ou pergunte sobre pontos específicos...")
ask = st.button("Perguntar", type="primary", disabled=not question.strip())

if ask:
    try:
        with st.spinner("Buscando trechos relevantes e gerando resposta…"):
            result = answer_question(st.session_state.index, question, top_k=top_k)

        st.subheader("Resposta")
        st.write(result["answer"])

        st.subheader("Fontes (trechos recuperados)")
        for i, src in enumerate(result["sources"], start=1):
            with st.expander(f"#{i} • página {src['page']} • score {src['score']:.3f}"):
                st.write(src["text"])
    except Exception as e:
        if "402" in str(e) or "balance" in str(e).lower():
            st.error("⚠️ **Erro de Saldo:** Sua chave de API (DeepSeek/OpenAI) está sem créditos. Por favor, recarregue sua conta ou use outra chave no arquivo .env.")
        else:
            st.error(f"Ocorreu um erro inesperado: {e}")
