"""
utils.py — Funções auxiliares do projeto.

Reúne utilidades usadas pelo app.py e pelo pipeline:
- load_documents_from_folder: varre a pasta de documentos e carrega tudo;
- get_llm: configura o LLM gratuito (Gemini, Groq ou Hugging Face);
- setup_logging: configura o logging do projeto;
- format_time: formata durações para exibição na interface.
"""

import os
import glob
import logging

from document_loader import DocumentLoader, SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Carregamento em lote dos documentos
# ----------------------------------------------------------------------
def load_documents_from_folder(folder_path: str) -> list:
    """
    Carrega todos os documentos suportados de uma pasta (não recursivo).

    Arquivos com extensão não suportada são ignorados com um aviso no log,
    sem interromper o processo. Arquivos que falharem na leitura também são
    pulados com aviso — um arquivo corrompido não derruba a indexação.

    Args:
        folder_path: caminho da pasta (ex.: "data/documents").

    Returns:
        Lista de documentos no formato do DocumentLoader.
    """
    loader = DocumentLoader()
    documents = []

    if not os.path.isdir(folder_path):
        logger.error("Pasta de documentos não encontrada: %s", folder_path)
        return documents

    for file_path in sorted(glob.glob(os.path.join(folder_path, "*"))):
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.warning("Ignorando arquivo não suportado: %s", file_path)
            continue

        try:
            documents.append(loader.load_document(file_path))
        except Exception as exc:
            logger.error("Falha ao carregar '%s': %s", file_path, exc)

    logger.info(
        "%d documentos carregados de '%s'.", len(documents), folder_path
    )
    return documents


# ----------------------------------------------------------------------
# Configuração do LLM gratuito
# ----------------------------------------------------------------------
def get_llm():
    """
    Retorna o LLM configurado conforme as variáveis de ambiente.

    Provedores suportados (defina LLM_PROVIDER no .env):

    - "gemini" (padrão): Google Gemini — chave gratuita em
      https://aistudio.google.com/apikey
      Variável: GEMINI_API_KEY
    - "groq": Groq (Llama 3.1, muito rápido) — chave em
      https://console.groq.com/keys
      Variável: GROQ_API_KEY
    - "huggingface": Hugging Face Inference API — token em
      https://huggingface.co/settings/tokens
      Variável: HUGGINGFACEHUB_API_TOKEN

    A temperatura baixa (0.1) deixa as respostas mais fiéis ao contexto,
    o que é essencial para um sistema RAG corporativo.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY não encontrada. Crie o arquivo .env com sua "
                "chave gratuita de https://aistudio.google.com/apikey "
                "(veja .env.example)."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=4096,
        )

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY não encontrada. Crie uma chave gratuita em "
                "https://console.groq.com/keys e adicione ao .env."
            )
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            groq_api_key=api_key,
            temperature=0.1,
            max_tokens=1024,
        )

    if provider == "huggingface":
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        if not token:
            raise ValueError(
                "HUGGINGFACEHUB_API_TOKEN não encontrado. Gere um token em "
                "https://huggingface.co/settings/tokens e adicione ao .env."
            )
        from langchain_huggingface import HuggingFaceEndpoint

        return HuggingFaceEndpoint(
            repo_id=os.getenv(
                "HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"
            ),
            huggingfacehub_api_token=token,
            temperature=0.1,
            max_new_tokens=1024,
        )

    raise ValueError(
        f"LLM_PROVIDER desconhecido: '{provider}'. "
        "Use 'gemini', 'groq' ou 'huggingface'."
    )


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
def setup_logging(level: int = logging.INFO):
    """
    Configura o logging do projeto no console.

    Formato: horário | nível | módulo | mensagem
    Chame uma única vez, no início da aplicação.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # sobrescreve configs anteriores (útil no Streamlit)
    )


# ----------------------------------------------------------------------
# Formatação de tempo
# ----------------------------------------------------------------------
def format_time(seconds: float) -> str:
    """
    Formata uma duração em segundos para exibição amigável.

    Exemplos: 0.85 -> "850 ms" | 2.34 -> "2,34 s" | 75.2 -> "1 min 15 s"
    """
    if seconds < 1:
        return f"{int(seconds * 1000)} ms"
    if seconds < 60:
        return f"{seconds:.2f}".replace(".", ",") + " s"
    minutes = int(seconds // 60)
    rest = int(seconds % 60)
    return f"{minutes} min {rest} s"
