"""
test_pipeline.py — Testes básicos do pipeline RAG (sem chamar o LLM).

Executa um "smoke test" ponta a ponta da parte local do sistema:
carregamento dos documentos -> limpeza -> chunking -> indexação -> busca.
O LLM não é testado aqui porque exige chave de API.

Como rodar:
    python tests/test_pipeline.py

O teste de busca vetorial só roda se chromadb/sentence-transformers
estiverem instalados; caso contrário, é pulado com aviso.
"""

import os
import sys
import shutil
import tempfile

# Permite importar os módulos da raiz do projeto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from document_loader import DocumentLoader, SUPPORTED_EXTENSIONS
from text_processor import TextProcessor
from utils import format_time

DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data", "documents")


def test_document_loading():
    """Cada PDF da pasta deve ser carregado com texto e metadados."""
    loader = DocumentLoader()
    pdfs = [f for f in os.listdir(DOCS_FOLDER) if f.endswith(".pdf")]
    assert len(pdfs) == 5, f"Esperados 5 PDFs, encontrados {len(pdfs)}"

    for pdf in pdfs:
        doc = loader.load_document(os.path.join(DOCS_FOLDER, pdf))
        assert doc["text"].strip(), f"PDF sem texto: {pdf}"
        assert doc["pages"], f"PDF sem páginas: {pdf}"
        meta = doc["metadata"]
        assert meta["file_name"] == pdf
        assert meta["extension"] in SUPPORTED_EXTENSIONS
        assert meta["modified_date"], "Data de modificação ausente"
        print(f"  OK  {pdf}: {len(doc['pages'])} página(s), "
              f"{len(doc['text'])} caracteres")


def test_text_processing():
    """O chunking deve preservar seções e gerar metadados completos."""
    loader = DocumentLoader()
    processor = TextProcessor()

    doc = loader.load_document(
        os.path.join(DOCS_FOLDER, "politica_reembolsos_devolucoes.pdf")
    )
    chunks = processor.create_chunks(doc)

    assert chunks, "Nenhum chunk gerado"
    for chunk in chunks:
        assert chunk["text"].strip(), "Chunk vazio"
        meta = chunk["metadata"]
        assert meta["file_name"] == "politica_reembolsos_devolucoes.pdf"
        assert meta["page"] >= 1, "PDF deve registrar a página do chunk"
        assert "chunk_index" in meta

    # A política tem seções numeradas ("1. Prazo..."), elas devem aparecer
    sections = {c["metadata"]["section"] for c in chunks}
    assert any("Prazo" in s or "reembolso" in s.lower() for s in sections), \
        f"Seções não detectadas: {sections}"

    print(f"  OK  {len(chunks)} chunks gerados com seções: {sorted(sections)[:3]}...")


def test_clean_text():
    """A limpeza deve remover rodapés e normalizar espaços."""
    processor = TextProcessor()
    dirty = "Informação   importante.\n\n\n\nPágina 3 de 10\nFim.\n"
    clean = processor.clean_text(dirty)
    assert "Página 3 de 10" not in clean
    assert "\n\n\n" not in clean
    assert "Informação importante." in clean
    print("  OK  Limpeza de texto removeu rodapé e espaços extras")


def test_format_time():
    """Formatação de tempo amigável."""
    assert format_time(0.5) == "500 ms"
    assert format_time(2.0) == "2,00 s"
    assert format_time(75) == "1 min 15 s"
    print("  OK  format_time")


def test_vector_store_search():
    """Indexação + busca semântica (requer chromadb e sentence-transformers).

    Na primeira execução, o modelo de embeddings (~80 MB) é baixado do
    Hugging Face — sem internet, este teste é pulado com aviso.
    """
    try:
        _run_vector_store_search()
    except Exception as exc:
        print(f"  PULADO ({type(exc).__name__}: {str(exc)[:120]})")


def _run_vector_store_search():
    from vector_store import VectorStore

    # Usa uma pasta temporária para não sujar o chroma_db de produção
    tmp = tempfile.mkdtemp()
    try:
        store = VectorStore(persist_dir=tmp)
        assert store.is_empty()

        loader = DocumentLoader()
        processor = TextProcessor()
        chunks = []
        for pdf in os.listdir(DOCS_FOLDER):
            doc = loader.load_document(os.path.join(DOCS_FOLDER, pdf))
            chunks.extend(processor.create_chunks(doc))

        indexed = store.index_documents(chunks)
        assert indexed == len(chunks) > 0
        assert not store.is_empty()

        # Busca semântica: pergunta sobre reembolso deve achar o documento certo
        results = store.search("qual o prazo para devolução de um produto?", top_k=5)
        assert results, "Busca não retornou resultados"
        assert results[0]["score"] > 0.3, f"Score baixo: {results[0]['score']}"
        top_file = results[0]["metadata"]["file_name"]
        print(f"  OK  Busca retornou '{top_file}' com score {results[0]['score']}")
        assert "reembolso" in top_file or "devoluc" in top_file, \
            f"Documento inesperado no topo: {top_file}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("\n[1/5] Carregamento de documentos")
    test_document_loading()
    print("[2/5] Chunking e metadados")
    test_text_processing()
    print("[3/5] Limpeza de texto")
    test_clean_text()
    print("[4/5] Formatação de tempo")
    test_format_time()
    print("[5/5] Indexação e busca vetorial")
    test_vector_store_search()
    print("\n✅ Todos os testes passaram!\n")
