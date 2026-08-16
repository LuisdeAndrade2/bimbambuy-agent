"""
reranker.py — Reordenação de resultados por relevância (reranking).

A busca vetorial (primeira etapa) recupera candidatos rapidamente, mas a
ordem nem sempre é perfeita. O reranker refina essa ordem usando um modelo
cross-encoder, que avalia o par (pergunta, trecho) em conjunto — é mais
preciso, porém mais lento, por isso só rodamos nos top-k já recuperados.

Modelo padrão: cross-encoder/ms-marco-MiniLM-L-6-v2
- Gratuito, roda localmente, leve (~90 MB);
- Treinado justamente para ranquear pares pergunta/documento;
- Funciona razoavelmente em português, apesar de treinado em inglês.

Fallback: se o cross-encoder falhar (sem internet para baixar o modelo,
memória insuficiente etc.), caímos para a ordem original da busca vetorial,
garantindo que o sistema continue funcionando.
"""

import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Reordena chunks candidatos usando um cross-encoder local."""

    def __init__(self, model_name: str = DEFAULT_MODEL, enabled: bool = True):
        """
        Args:
            model_name: modelo cross-encoder do Hugging Face.
            enabled: se False, o reranker vira pass-through (útil para testes).
        """
        self.model_name = model_name
        self.enabled = enabled
        self._model = None  # carregado preguiçosamente na primeira chamada

    # ------------------------------------------------------------------
    # Carregamento preguiçoso do modelo (lazy loading)
    # ------------------------------------------------------------------
    def _load_model(self):
        """
        Carrega o cross-encoder apenas quando necessário.

        O download acontece uma única vez; depois o modelo fica em cache
        local (~/.cache/huggingface).
        """
        if self._model is None:
            logger.info("Carregando cross-encoder '%s'...", self.model_name)
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder carregado.")

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------
    def rerank(self, query: str, documents: list, top_n: int = 5) -> list:
        """
        Reordena os documentos candidatos por relevância à pergunta.

        Args:
            query: pergunta do usuário.
            documents: lista de chunks retornados pelo VectorStore.search
                       (dicionários com "text", "metadata" e "score").
            top_n: quantos documentos manter após o reranking.

        Returns:
            Os top_n documentos mais relevantes, com o campo extra
            "rerank_score" (quanto maior, mais relevante).
        """
        if not documents:
            return []

        # Se desabilitado ou poucos documentos, mantém a ordem original
        if not self.enabled or len(documents) <= 1:
            return documents[:top_n]

        try:
            self._load_model()

            # O cross-encoder recebe pares (pergunta, texto) e devolve
            # um score de relevância para cada par
            pairs = [(query, doc["text"]) for doc in documents]
            scores = self._model.predict(pairs)

            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)

            ranked = sorted(
                documents, key=lambda d: d["rerank_score"], reverse=True
            )
            logger.info(
                "Reranking concluído. Melhor score: %.3f | Pior: %.3f",
                ranked[0]["rerank_score"],
                ranked[-1]["rerank_score"],
            )
            return ranked[:top_n]

        except Exception as exc:
            # Fallback seguro: usa a ordem da busca vetorial original
            logger.warning(
                "Cross-encoder indisponível (%s). "
                "Usando ordem da busca vetorial como fallback.",
                exc,
            )
            ranked = sorted(
                documents, key=lambda d: d.get("score", 0), reverse=True
            )
            return ranked[:top_n]
