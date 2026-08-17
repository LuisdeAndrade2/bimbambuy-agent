"""
app.py — Interface web do Agente de IA BimBam Buy (Streamlit).

Execute com:
    streamlit run app.py

Funcionalidades:
- Chat com histórico de conversa;
- Inicialização automática: carrega e indexa os documentos na 1ª execução;
- Exibição das fontes citadas (arquivo, página, seção) com prévia do trecho;
- Métricas por resposta: tempo, chunks recuperados e utilizados;
- Botão "Limpar conversa" e opção de reindexar a base na barra lateral;
- Indicador de processamento enquanto o pipeline RAG roda.
"""

import os
import sys
import streamlit as st

# ---------------------------------------------------------------------------
# Compatibilidade ChromaDB x Streamlit Cloud
# O ChromaDB exige SQLite >= 3.35; o Streamlit Cloud às vezes tem uma versão
# mais antiga. O pacote pysqlite3-binary resolve isso — este truque só é
# ativado quando o pysqlite3 está instalado (localmente não faz diferença).
# ---------------------------------------------------------------------------
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

from utils import (
    load_documents_from_folder,
    get_llm,
    setup_logging,
    format_time,
)
from text_processor import TextProcessor
from vector_store import VectorStore
from reranker import Reranker
from rag_chain import RAGChain

# Pasta onde ficam os documentos da base de conhecimento
DOCS_FOLDER = os.getenv("DOCS_FOLDER", "data/documents")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Agente BimBam Buy",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Inicialização do sistema RAG (com cache para não reindexar a cada clique)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def initialize_system(force_reindex: bool = False):
    """
    Monta todo o pipeline: carrega documentos, gera chunks, indexa no
    ChromaDB e prepara o reranker + LLM.

    O decorador @st.cache_resource garante que isso roda UMA vez por sessão
    do servidor — recarregar a página ou conversar não reindexa nada.
    Como o ChromaDB é persistente, nas próximas execuções a indexação é
    pulada automaticamente (a coleção já existe em disco).

    IMPORTANTE: NUNCA chame st.error(), st.stop() ou qualquer comando
    de UI dentro desta função cacheada. Isso quebra o Streamlit.
    """
    setup_logging()

    vector_store = VectorStore(persist_dir=CHROMA_DIR)
    documents = []

    if force_reindex or vector_store.is_empty():
        if force_reindex:
            vector_store.clear()

        documents = load_documents_from_folder(DOCS_FOLDER)

        if documents:
            processor = TextProcessor()
            chunks = []
            for doc in documents:
                chunks.extend(processor.create_chunks(doc))
            vector_store.index_documents(chunks)

    llm = get_llm()  # levanta ValueError amigável se a chave faltar
    reranker = Reranker()
    chain = RAGChain(llm=llm, vector_store=vector_store, reranker=reranker)

    return chain, vector_store, documents


# ---------------------------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------------------------
st.title("🛒 Agente BimBam Buy")
st.caption(
    "Assistente corporativo que responde perguntas com base nos documentos "
    "internos da empresa. As respostas citam as fontes consultadas."
)

# ---------------------------------------------------------------------------
# Inicialização com indicador de progresso
# ---------------------------------------------------------------------------
# Garante que a flag force_reindex existe no session_state
if "force_reindex" not in st.session_state:
    st.session_state["force_reindex"] = False

try:
    with st.spinner(
        "Inicializando a base de conhecimento... "
        "(na 1ª execução isso pode levar alguns minutos)"
    ):
        chain, vector_store, documents = initialize_system(
            force_reindex=st.session_state.get("force_reindex", False)
        )
        # Reseta a flag após usar
        st.session_state["force_reindex"] = False

    # Validação de documentos FORA da função cacheada
    if not documents and vector_store.is_empty():
        st.error(
            f"Nenhum documento encontrado em `{DOCS_FOLDER}`. "
            "Adicione os arquivos da base de conhecimento e reinicie."
        )
        st.info(
            "Certifique-se de que a pasta `data/documents/` existe no "
            "repositório e contém os documentos corporativos."
        )
        st.stop()

except ValueError as exc:
    # Erro típico: chave de API ausente
    st.error(f"⚠️ Configuração incompleta: {exc}")
    st.info(
        "**Como resolver (Streamlit Cloud):**\n"
        "1. Acesse o painel do seu app em share.streamlit.io\n"
        "2. Vá em **Settings → Secrets**\n"
        "3. Cole suas variáveis no formato TOML:\n"
        "```\n"
        'LLM_PROVIDER = "gemini"\n'
        'GEMINI_API_KEY = "sua_chave_aqui"\n'
        'DOCS_FOLDER = "data/documents"\n'
        'CHROMA_DIR = "chroma_db"\n'
        "```"
    )
    st.stop()
except Exception as exc:
    st.error(f"Erro inesperado na inicialização: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Barra lateral: status do sistema e ações
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📚 Base de conhecimento")
    st.metric("Chunks indexados", vector_store.count())

    # Lista os documentos disponíveis na pasta
    if os.path.isdir(DOCS_FOLDER):
        files = sorted(os.listdir(DOCS_FOLDER))
        st.markdown("**Documentos disponíveis:**")
        for f in files:
            st.markdown(f"- `{f}`")
    else:
        st.warning(f"Pasta `{DOCS_FOLDER}` não encontrada.")

    st.divider()
    provider = os.getenv("LLM_PROVIDER", "gemini")
    st.markdown(f"**LLM ativo:** `{provider}`")

    if st.button("🔄 Reindexar documentos", use_container_width=True):
        st.session_state["force_reindex"] = True
        st.cache_resource.clear()
        st.rerun()

    if st.button("🧹 Limpar conversa", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.caption(
        "As respostas usam **somente** os documentos indexados. "
        "Se a informação não existir na base, o agente dirá isso "
        "explicitamente."
    )

# ---------------------------------------------------------------------------
# Histórico de conversa (guardado no estado da sessão)
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    # Mensagem de boas-vindas
    st.session_state["messages"].append({
        "role": "assistant",
        "content": (
            "Olá! Sou o assistente da BimBam Buy. 👋\n\n"
            "Posso responder perguntas sobre nossos documentos internos, como:\n"
            "- Política de reembolsos e devoluções\n"
            "- Métodos de pagamento\n"
            "- Garantia de produtos\n"
            "- Programa de afiliados\n"
            "- Prazos e custos de envio\n\n"
            "Como posso ajudar?"
        ),
        "sources": [],
        "metrics": None,
    })


def render_sources(sources: list):
    """Exibe as fontes citadas dentro de um expansor."""
    if not sources:
        return
    with st.expander(f"📎 Fontes consultadas ({len(sources)})"):
        for src in sources:
            parts = [f"**[Fonte {src['id']}]** `{src['file_name']}`"]
            if src.get("page", -1) > 0:
                parts.append(f"página {src['page']}")
            if src.get("section"):
                parts.append(f'seção "{src["section"]}"')
            st.markdown(" — ".join(parts))
            if src.get("excerpt"):
                st.caption(f"_{src['excerpt']}..._")


# Renderiza todo o histórico
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])

# ---------------------------------------------------------------------------
# Entrada do usuário e geração da resposta
# ---------------------------------------------------------------------------
if query := st.chat_input("Digite sua pergunta sobre os documentos internos..."):
    # Mostra a pergunta do usuário
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Gera a resposta com indicador de processamento
    with st.chat_message("assistant"):
        with st.spinner("Processando sua pergunta..."):
            result = chain.generate_response(query)

        st.markdown(result["answer"])
        render_sources(result["sources"])

        if result["is_fallback"]:
            st.info(
                "💡 Esta pergunta parece estar fora do escopo dos documentos "
                "indexados. Tente reformular ou verifique se o documento "
                "relevante foi adicionado à pasta `data/documents/`.",
                icon="ℹ️",
            )

    # Salva a resposta no histórico
    st.session_state["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "is_fallback": result["is_fallback"],
    })
