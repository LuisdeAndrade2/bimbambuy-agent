"""
vector_store.py — Banco de dados vetorial com ChromaDB.

Responsável por:
- Gerar embeddings dos chunks com o modelo sentence-transformers
  all-MiniLM-L6-v2 (gratuito, roda localmente, sem API externa);
- Indexar os chunks em uma coleção ChromaDB com PERSISTÊNCIA LOCAL
  (pasta chroma_db/), para não precisar reindexar a cada inicialização;
- Executar buscas semânticas (similaridade de cosseno).

Cada resultado de busca é um dicionário:
    {"text": ..., "metadata": {...}, "score": 0.0 a 1.0}
quanto maior o score, mais similar o chunk é à pergunta.
"""

import os
import hashlib
import logging

logger = logging.getLogger(__name__)

# Modelo de embeddings: leve (80 MB), multilíngue o suficiente para PT-BR
# e totalmente gratuito/local (não consome cota de nenhuma API).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "bimbambuy_knowledge_base"


class VectorStore:
    """Interface simples sobre o ChromaDB para indexar e buscar chunks."""

    def __init__(self, persist_dir: str = "chroma_db"):
        """
        Inicializa o cliente ChromaDB com persistência em disco.

        Args:
            persist_dir: pasta onde o banco vetorial fica salvo.
        """
        import chromadb
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir

        # A função de embedding é registrada na coleção: o próprio Chroma
        # aplica o modelo tanto na indexação quanto na busca.
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},  # similaridade de cosseno
        )
        logger.info(
            "ChromaDB inicializado em '%s' — %d chunks já indexados.",
            persist_dir,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def get_embeddings(self):
        """
        Retorna a função de embedding usada pela coleção.

        Útil se você precisar gerar embeddings manualmente, por exemplo:
            fn = store.get_embeddings()
            vetores = fn(["texto 1", "texto 2"])
        """
        return self._embedding_fn

    # ------------------------------------------------------------------
    # Indexação
    # ------------------------------------------------------------------
    def is_empty(self) -> bool:
        """True se a coleção ainda não tem nenhum chunk indexado."""
        return self._collection.count() == 0

    def count(self) -> int:
        """Número total de chunks indexados."""
        return self._collection.count()

    def _chunk_id(self, chunk: dict) -> str:
        """
        Gera um ID determinístico para o chunk (hash do arquivo + índice).
        IDs determinísticos permitem reindexar sem criar duplicatas.
        """
        key = f"{chunk['metadata'].get('file_name')}::{chunk['metadata'].get('chunk_index')}::{chunk['text'][:80]}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def index_documents(self, chunks: list, batch_size: int = 64) -> int:
        """
        Gera embeddings e indexa uma lista de chunks no ChromaDB.

        A indexação é feita em lotes para economizar memória. Como os IDs
        são determinísticos, rodar duas vezes não duplica dados
        (o Chroma substitui registros com o mesmo ID via upsert).

        Returns:
            Número de chunks indexados.
        """
        if not chunks:
            logger.warning("Nenhum chunk para indexar.")
            return 0

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self._collection.upsert(
                ids=[self._chunk_id(c) for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )
            total += len(batch)
            logger.info("Indexados %d/%d chunks...", total, len(chunks))

        return total

    def clear(self):
        """Remove todos os chunks da coleção (recria a coleção vazia)."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Coleção '%s' limpa.", COLLECTION_NAME)

    # ------------------------------------------------------------------
    # Busca semântica
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: dict = None,
    ) -> list:
        """
        Busca os chunks mais similares à pergunta.

        Args:
            query: pergunta do usuário em linguagem natural.
            top_k: quantos chunks retornar.
            filter_metadata: filtro opcional do ChromaDB, ex.:
                {"file_name": "politica_reembolsos_devolucoes.pdf"}

        Returns:
            Lista de {"text", "metadata", "score"} ordenada por relevância.
            O score é a similaridade de cosseno (0 a 1, maior = melhor).
        """
        if self.is_empty():
            logger.warning("Busca ignorada: coleção vazia.")
            return []

        params = {
            "query_texts": [query],
            "n_results": min(top_k, self._collection.count()),
        }
        if filter_metadata:
            params["where"] = filter_metadata

        results = self._collection.query(**params)

        # Converte o formato bruto do Chroma em lista de dicionários
        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for text, meta, dist in zip(docs, metas, dists):
            output.append({
                "text": text,
                "metadata": meta or {},
                # Chroma retorna DISTÂNCIA de cosseno (0 = idêntico);
                # convertemos para similaridade (1 = idêntico).
                "score": round(1 - dist, 4),
            })

        return output
