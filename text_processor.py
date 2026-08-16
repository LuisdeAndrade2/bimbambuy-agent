"""
text_processor.py — Limpeza, chunking inteligente e extração de metadados.

O "chunking" é o processo de dividir documentos longos em pedaços (chunks)
menores, que são indexados no banco vetorial. A estratégia aqui é:

1. Limpar o texto (ruídos, espaços, cabeçalhos/rodapés de página);
2. Tentar dividir por SEÇÕES (títulos numerados, linhas em maiúsculas etc.),
   porque seções mantêm o contexto semântico completo;
3. Se uma seção for grande demais, subdividi-la por tamanho fixo com
   SOBREPOSIÇÃO (overlap), para não cortar ideias no meio;
4. Se não houver seções detectáveis, dividir tudo por tamanho fixo.

Cada chunk gerado carrega metadados: arquivo, página (quando houver),
seção e índice do chunk — isso alimenta o sistema de citações.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Tamanho padrão do chunk (em caracteres) e sobreposição entre chunks
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200

# Chunks menores que isso geralmente são só títulos soltos — fundimos com o próximo
MIN_CHUNK_SIZE = 80

# Padrões que indicam o início de uma seção/título
SECTION_PATTERNS = [
    r"^\s*\d+(\.\d+)*[\.\)]\s+\S",                    # "1. Título", "2.3 Título", "1) Título"
    r"^\s*(SEÇÃO|CAPÍTULO|ARTIGO|CLÁUSULA)\s+\d+",    # "Seção 2", "Artigo 5"
    r"^\s*#{1,6}\s+\S",                               # títulos Markdown: "## Título"
    r"^\s*[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9][A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9\s\-:,&]{5,}$",  # linha toda em MAIÚSCULAS
]
_SECTION_RE = re.compile("|".join(f"({p})" for p in SECTION_PATTERNS))

# Padrões de rodapé/cabeçalho comuns que podem ser removidos
FOOTER_PATTERNS = [
    r"Página\s+\d+\s+de\s+\d+",      # "Página 3 de 10"
    r"^\s*\d+\s*/\s*\d+\s*$",        # "3/10"
    r"^\s*-\s*\d+\s*-\s*$",          # "- 3 -"
]
_FOOTER_RE = re.compile("|".join(FOOTER_PATTERNS), re.IGNORECASE)


class TextProcessor:
    """Limpa textos e os divide em chunks com metadados ricos."""

    # ------------------------------------------------------------------
    # Limpeza de texto
    # ------------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        """
        Remove ruídos típicos de texto extraído de documentos:
        - caracteres de controle invisíveis;
        - rodapés do tipo "Página X de Y";
        - hifenização de fim de linha ("infor- mação" -> "informação");
        - espaços e quebras de linha excessivos.
        """
        if not text:
            return ""

        # Remove caracteres de controle (mantém \n e \t)
        text = re.sub(r"[^\S\n\t]+", " ", text)
        text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")

        # Junta palavras quebradas por hífen no fim da linha
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        # Remove linhas que são só rodapé/cabeçalho de paginação
        lines = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and _FOOTER_RE.search(stripped):
                continue  # descarta a linha de rodapé
            lines.append(line.rstrip())
        text = "\n".join(lines)

        # Normaliza quebras de linha: no máximo 2 seguidas
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove espaços duplicados
        text = re.sub(r"[ \t]{2,}", " ", text)

        return text.strip()

    # ------------------------------------------------------------------
    # Chunking por seção (estratégia prioritária)
    # ------------------------------------------------------------------
    def chunk_by_section(self, text: str) -> list:
        """
        Divide o texto em seções com base em títulos/numeração.

        Returns:
            Lista de dicionários: {"section": título, "text": conteúdo}.
            Retorna lista vazia se nenhum título for detectado.
        """
        sections = []
        current_title = "Introdução"
        current_lines = []
        found_any = False

        for line in text.split("\n"):
            if _SECTION_RE.match(line):
                # Encontrou um novo título: fecha a seção anterior
                found_any = True
                if current_lines:
                    sections.append({
                        "section": current_title,
                        "text": "\n".join(current_lines).strip(),
                    })
                current_title = line.strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Fecha a última seção
        if current_lines:
            sections.append({
                "section": current_title,
                "text": "\n".join(current_lines).strip(),
            })

        if not found_any:
            return []  # sem títulos detectáveis — o chamador usa chunk_by_size

        # Funde seções muito pequenas com a seguinte (evita chunks de 1 linha)
        merged = []
        buffer = None
        for sec in sections:
            if buffer is None:
                buffer = sec
                continue
            if len(buffer["text"]) < MIN_CHUNK_SIZE:
                # Seção anterior era minúscula: junta com a atual
                buffer = {
                    "section": buffer["section"],
                    "text": (buffer["text"] + "\n" + sec["section"] + "\n" + sec["text"]).strip(),
                }
            else:
                merged.append(buffer)
                buffer = sec
        if buffer is not None:
            merged.append(buffer)

        return [s for s in merged if s["text"]]

    # ------------------------------------------------------------------
    # Chunking por tamanho fixo com sobreposição (fallback)
    # ------------------------------------------------------------------
    def chunk_by_size(
        self,
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> list:
        """
        Divide o texto em pedaços de tamanho fixo com sobreposição.

        A divisão tenta respeitar limites de parágrafo/frase para não
        cortar ideias no meio. A sobreposição garante que uma frase
        dividida apareça completa em pelo menos um dos chunks.
        """
        if not text:
            return []
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size

            if end < len(text):
                # Tenta terminar o chunk num ponto natural (parágrafo, frase ou espaço)
                window = text[start:end]
                cut = max(
                    window.rfind("\n\n"),
                    window.rfind(". "),
                    window.rfind("\n"),
                    window.rfind(" "),
                )
                # Só usa o corte natural se ele não encolher demais o chunk
                if cut > chunk_size * 0.5:
                    end = start + cut + 1

            chunks.append(text[start:end].strip())
            # Avança com sobreposição para manter contexto entre chunks
            start = end - overlap if end < len(text) else len(text)

        return [c for c in chunks if c]

    # ------------------------------------------------------------------
    # Extração de metadados do chunk
    # ------------------------------------------------------------------
    def extract_metadata(self, text: str, file_name: str) -> dict:
        """
        Extrai metadados descritivos de um trecho de texto.

        - title: primeira linha que parece um título;
        - section: título da seção detectada (se houver).
        """
        title = ""
        section = ""

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if not title:
                title = stripped[:120]  # primeira linha não vazia = título provável
            if _SECTION_RE.match(line) and not section:
                section = stripped[:120]
            if title and section:
                break

        return {
            "file_name": file_name,
            "title": title,
            "section": section,
        }

    # ------------------------------------------------------------------
    # Pipeline completo: documento -> lista de chunks com metadados
    # ------------------------------------------------------------------
    def create_chunks(
        self,
        document: dict,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> list:
        """
        Transforma um documento carregado (ver document_loader.py) em uma
        lista de chunks prontos para indexação.

        Estratégia:
        - PDFs são processados página por página (para registrar a página
          em cada chunk e alimentar as citações);
        - Em cada página (ou no texto inteiro, para outros formatos),
          tentamos chunking por seção; se não houver seções, usamos
          chunking por tamanho com overlap.

        Returns:
            Lista de dicionários: {"text": str, "metadata": {...}}
        """
        base_meta = document.get("metadata", {})
        file_name = base_meta.get("file_name", "desconhecido")
        chunks = []

        # PDFs trazem a lista de páginas; demais formatos, só o texto
        pages = document.get("pages") or [(None, document.get("text", ""))]

        for page_num, page_text in pages:
            cleaned = self.clean_text(page_text)
            if not cleaned:
                continue

            # 1) Tenta dividir por seções
            sections = self.chunk_by_section(cleaned)

            if sections:
                # 2) Seções grandes demais são subdivididas por tamanho
                for sec in sections:
                    if len(sec["text"]) <= chunk_size:
                        chunk_texts = [sec["text"]]
                    else:
                        chunk_texts = self.chunk_by_size(sec["text"], chunk_size, overlap)
                    for ct in chunk_texts:
                        chunks.append(self._make_chunk(
                            ct, file_name, page_num, sec["section"], base_meta
                        ))
            else:
                # 3) Sem seções detectáveis: chunking por tamanho puro
                for ct in self.chunk_by_size(cleaned, chunk_size, overlap):
                    meta = self.extract_metadata(ct, file_name)
                    chunks.append(self._make_chunk(
                        ct, file_name, page_num, meta.get("section", ""), base_meta
                    ))

        # Numera os chunks sequencialmente dentro do documento
        for i, chunk in enumerate(chunks):
            chunk["metadata"]["chunk_index"] = i

        logger.info(
            "Documento '%s' gerou %d chunks", file_name, len(chunks)
        )
        return chunks

    # ------------------------------------------------------------------
    # Helper: monta o dicionário de um chunk
    # ------------------------------------------------------------------
    def _make_chunk(self, text, file_name, page_num, section, base_meta) -> dict:
        """Monta um chunk no formato padrão usado pelo restante do sistema."""
        metadata = {
            "file_name": file_name,
            "section": section or "",
            # page=-1 significa "não se aplica" (formatos sem paginação).
            # ChromaDB exige valores simples (str/int/float/bool).
            "page": page_num if page_num is not None else -1,
            "modified_date": base_meta.get("modified_date", ""),
        }
        return {"text": text, "metadata": metadata}
