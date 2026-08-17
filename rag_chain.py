"""
rag_chain.py — Pipeline completo de RAG (Retrieval-Augmented Generation).

Fluxo para cada pergunta do usuário:

    1. Busca semântica no ChromaDB (top 10 candidatos);
    2. Reranking com cross-encoder (mantém os top 5 mais relevantes);
    3. Montagem do prompt: contexto numerado + metadados + instruções
       rígidas para o LLM responder APENAS com base no contexto;
    4. Chamada ao LLM (Gemini, Groq ou Hugging Face);
    5. Extração das fontes citadas na resposta.

O ponto anti-alucinação é o PROMPT DO SISTEMA: ele instrui o modelo a
nunca usar conhecimento próprio e a declarar explicitamente quando a
informação não estiver nos documentos.
"""

import re
import time
import logging

logger = logging.getLogger(__name__)

# Mensagem padrão quando a informação não existe na base de conhecimento
FALLBACK_MESSAGE = (
    "Não encontrei essa informação nos documentos disponíveis."
)

# ---------------------------------------------------------------------------
# Prompt do sistema — o coração da estratégia anti-alucinação
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Você é o assistente corporativo da BimBam Buy, um e-commerce da América Latina.
Sua função é responder perguntas de colaboradores usando EXCLUSIVAMENTE os
trechos de documentos internos fornecidos abaixo como CONTEXTO.

REGRAS OBRIGATÓRIAS:
1. IDIOMA: responda SEMPRE em português brasileiro, em linguagem natural,
   com frases completas — como um colega explicando para outro. NUNCA use
   inglês ou outro idioma na resposta, mesmo que termos em inglês apareçam
   no contexto (traduza-os).
2. FORMATO: comece a resposta com a resposta direta à pergunta, em texto
   corrido. Nunca comece com ":", marcadores soltos ou cabeçalhos. Se
   houver mais de um caso (ex.: prazos diferentes por situação), use uma
   lista com hífen (-) DEPOIS da resposta direta. Não use tabelas.
3. Responda APENAS com base no contexto fornecido. NUNCA use conhecimento
   externo, suposições ou informações que não estejam no contexto.
4. Se a resposta não estiver no contexto, diga exatamente:
   "{fallback}".
5. Se a pergunta for ambígua, responda com a interpretação mais provável
   e mencione a alternativa apenas se houver evidência no contexto.
"""

# Limiares do pipeline (fáceis de ajustar)
RETRIEVAL_TOP_K = 10   # candidatos da busca vetorial
RERANK_TOP_N = 5       # chunks que entram de fato no prompt
MIN_SCORE = 0.25       # similaridade mínima aceitável (abaixo disso, ignoramos)


class RAGChain:
    """Orquestra busca -> reranking -> prompt -> LLM -> citações."""

    def __init__(self, llm, vector_store, reranker):
        """
        Args:
            llm: modelo de linguagem LangChain (ChatGoogleGenerativeAI etc.).
            vector_store: instância de VectorStore já indexada.
            reranker: instância de Reranker.
        """
        self.llm = llm
        self.vector_store = vector_store
        self.reranker = reranker

    # ------------------------------------------------------------------
    # Pipeline completo
    # ------------------------------------------------------------------
    def generate_response(self, query: str) -> dict:
        """
        Executa o pipeline RAG completo para uma pergunta.

        Returns:
            {
                "answer": "texto da resposta",
                "sources": [ {"id": 1, "file_name": ..., "page": ...,
                               "section": ..., "score": ...} ],
                "chunks_retrieved": 10,   # quantos vieram da busca vetorial
                "chunks_used": 5,         # quantos entraram no prompt
                "elapsed_seconds": 1.23,  # tempo total do pipeline
                "is_fallback": False,     # True se não achou informação
            }
        """
        start = time.time()

        # ---- Etapa 1: busca semântica (recuperação ampla) -------------
        candidates = self.vector_store.search(query, top_k=RETRIEVAL_TOP_K)

        # Descarta resultados com similaridade muito baixa: provavelmente
        # a pergunta está fora do escopo dos documentos
        candidates = [c for c in candidates if c.get("score", 0) >= MIN_SCORE]

        if not candidates:
            return self._fallback_result(start)

        # ---- Etapa 2: reranking (refinamento da ordem) -----------------
        top_chunks = self.reranker.rerank(query, candidates, top_n=RERANK_TOP_N)

        # ---- Etapa 3: montagem do prompt ------------------------------
        context = self.format_context(top_chunks)
        prompt = self._build_prompt(query, context)

        # ---- Etapa 4: chamada ao LLM -----------------------------------
        try:
            response = self.llm.invoke(prompt)
            # LangChain retorna um objeto AIMessage; extraímos o texto
            answer = (
                response.content if hasattr(response, "content") else str(response)
            )
            # O Gemini pode devolver o conteúdo como lista de blocos
            if isinstance(answer, list):
                answer = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in answer
                )
        except Exception as exc:
            logger.error("Erro na chamada ao LLM: %s", exc)
            answer = (
                "Ocorreu um erro ao gerar a resposta. "
                f"Detalhes técnicos: {exc}"
            )
            top_chunks = top_chunks  # mantém as fontes para referência

        # ---- Etapa 5: extração das fontes citadas ----------------------
        sources = self.extract_sources(answer, top_chunks)
        
        # Remove os marcadores [Fonte N] da resposta final
        clean_answer = self.remove_source_markers(answer)

        elapsed = time.time() - start
        
        # É fallback se:
        # 1. Não encontrou nenhum chunk relevante inicialmente, OU
        # 2. Nenhuma fonte foi citada, OU
        # 3. A resposta contém a mensagem de fallback
        is_fallback = (
            not candidates 
            or not sources 
            or FALLBACK_MESSAGE.lower() in answer.lower()
        )

        return {
            "answer": clean_answer,
            "sources": sources,
            "chunks_retrieved": len(candidates),
            "chunks_used": len(top_chunks),
            "elapsed_seconds": round(elapsed, 2),
            "is_fallback": is_fallback,
        }

    # ------------------------------------------------------------------
    # Formatação do contexto para o prompt
    # ------------------------------------------------------------------
    def format_context(self, chunks: list) -> str:
        """
        Converte os chunks selecionados em blocos numerados com metadados,
        prontos para serem colados no prompt. Exemplo:

            [Fonte 1] politica_reembolsos_devolucoes.pdf — página 1 — seção "4. Prazos de reembolso"
            "Após o recebimento e conferência do produto..."
        """
        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            file_name = meta.get("file_name", "desconhecido")
            page = meta.get("page", -1)
            section = meta.get("section", "")

            header = f"[Fonte {i}] {file_name}"
            if page and page > 0:
                header += f" — página {page}"
            if section:
                header += f' — seção "{section}"'

            blocks.append(f"{header}\n{chunk['text']}")

        return "\n\n---\n\n".join(blocks)

    def _build_prompt(self, query: str, context: str) -> str:
        """Monta o prompt final: sistema + contexto + pergunta."""
        system = SYSTEM_PROMPT.format(fallback=FALLBACK_MESSAGE)
        return (
            f"{system}\n\n"
            f"CONTEXTO (trechos de documentos internos):\n"
            f"{'=' * 50}\n{context}\n{'=' * 50}\n\n"
            f"PERGUNTA DO COLABORADOR: {query}\n\n"
            f"RESPOSTA (cite as fontes no formato [Fonte N]):"
        )

    # ------------------------------------------------------------------
    # Extração das fontes citadas na resposta
    # ------------------------------------------------------------------
    def extract_sources(self, response: str, chunks: list) -> list:
        """
        Identifica quais fontes o LLM citou na resposta.

        Procura marcadores do tipo [Fonte 1], [Fonte 2]... e mapeia cada
        número para o chunk correspondente (a numeração do contexto é a
        mesma da lista `chunks`). Se o LLM não citou nada (resposta de
        fallback ou desobediência ao prompt), retornamos todos os chunks
        usados como "fontes consultadas", para transparência.

        Returns:
            Lista de dicionários com metadados das fontes.
        """
        cited_ids = re.findall(r"\[Fonte\s*(\d+)\]", response, re.IGNORECASE)
        cited_ids = [int(n) for n in cited_ids]

        sources = []
        seen = set()
        for n in cited_ids:
            if 1 <= n <= len(chunks) and n not in seen:
                seen.add(n)
                sources.append(self._source_dict(n, chunks[n - 1]))

        # Se nada foi citado, mostramos as fontes consultadas mesmo assim
        if not sources:
            for i, chunk in enumerate(chunks, start=1):
                sources.append(self._source_dict(i, chunk))

        return sources

    def remove_source_markers(self, text: str) -> str:
        """
        Remove os marcadores [Fonte N] do texto da resposta.
        
        Captura tanto marcadores simples [Fonte 1] quanto múltiplos
        [Fonte 1, Fonte 2] ou [Fonte 1 e Fonte 2].
        
        Mantém apenas o conteúdo expandido, sem os identificadores
        entre chaves.
        """
        # Captura [Fonte ...] incluindo tudo dentro dos colchetes
        return re.sub(r"\[Fonte[^\]]*\]", "", text, flags=re.IGNORECASE).strip()

    def _source_dict(self, source_id: int, chunk: dict) -> dict:
        """Converte um chunk em um dicionário de fonte amigável para a UI."""
        meta = chunk.get("metadata", {})
        return {
            "id": source_id,
            "file_name": meta.get("file_name", "desconhecido"),
            "page": meta.get("page", -1),
            "section": meta.get("section", ""),
            "score": chunk.get("score", 0),
            "rerank_score": chunk.get("rerank_score"),
            "excerpt": chunk.get("text", "")[:300],  # prévia para a UI
        }

    # ------------------------------------------------------------------
    # Resultado de fallback (nenhum chunk relevante encontrado)
    # ------------------------------------------------------------------
    def _fallback_result(self, start: float) -> dict:
        """Monta a resposta padrão quando a busca não retorna nada útil."""
        return {
            "answer": (
                f"{FALLBACK_MESSAGE}\n\n"
                "Sugestões:\n"
                "- Verifique se o documento sobre o assunto foi adicionado "
                "à pasta `data/documents/`;\n"
                "- Tente reformular a pergunta com outras palavras-chave;\n"
                "- Documentos disponíveis cobrem: reembolsos e devoluções, "
                "métodos de pagamento, garantia de produtos, programa de "
                "afiliados e prazos/custos de envio."
            ),
            "sources": [],
            "chunks_retrieved": 0,
            "chunks_used": 0,
            "elapsed_seconds": round(time.time() - start, 2),
            "is_fallback": True,
        }
