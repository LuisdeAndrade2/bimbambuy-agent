# 🛒 Agente de IA BimBam Buy — Base de Conhecimento Corporativo

Agente de IA acessível a todos os colaboradores da **BimBam Buy** (e-commerce
LATAM), capaz de responder perguntas com base **exclusivamente** nos documentos
internos da empresa. Construído com **RAG (Retrieval-Augmented Generation)**,
interface web em **Streamlit** e componentes **100% gratuitos**.

O agente responde sobre os 5 documentos corporativos incluídos:

| Documento | Domínio |
|---|---|
| Política de Reembolsos e Devoluções | Pós-venda / Operacional |
| Perguntas Frequentes sobre Métodos de Pagamento | Financeiro / CX |
| Manual de Garantia de Produtos | Pós-venda / Legal |
| Programa de Afiliados | Marketing / Parcerias |
| Guia de Prazos e Custos de Envio | Logística / Operacional |

---

## ✨ Funcionalidades

- 💬 **Chat web simples e funcional** (Streamlit), com histórico de conversa
- 🔎 **RAG completo**: busca semântica → reranking → geração com LLM
- 📄 **Múltiplos formatos**: PDF, Word, Excel, PowerPoint, Markdown, CSV, JSON e HTML
- ✂️ **Chunking inteligente**: divisão por seções/títulos, com fallback para
  tamanho fixo com sobreposição (overlap)
- 🏆 **Reranking** com cross-encoder para refinar a relevância dos trechos
- 📎 **Citação de fontes** com metadados: nome do arquivo, página e seção,
  com prévia do trecho utilizado
- 🚫 **Anti-alucinação**: o LLM é instruído a responder APENAS com base no
  contexto recuperado, com fallback claro ("Não encontrei essa informação
  nos documentos disponíveis")
- 📊 **Métricas por resposta**: tempo, chunks recuperados e utilizados
- 💾 **Persistência local** do banco vetorial (não reindexa a cada execução)
- 🔄 **Reindexação** da base com um clique na barra lateral

---

## 🧰 Tecnologias

| Componente | Escolha | Por quê |
|---|---|---|
| Interface | Streamlit | Simples, gratuito, deploy fácil |
| Orquestração RAG | LangChain | Padrão de mercado, integrações prontas |
| LLM | Google Gemini 3.6 Flash (gratuito) | Chave free tier generosa; Groq e Hugging Face como alternativas |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, gratuito, leve (~80 MB), sem consumir cota de API |
| Banco vetorial | ChromaDB (persistente, local) | Zero configuração de servidor, gratuito |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local, gratuito, melhora a precisão da recuperação |
| Extração de PDF | pdfplumber (fallback: pypdf) | Boa fidelidade de layout |

---

## 📋 Pré-requisitos

- **Python 3.9 ou superior** (recomendado: 3.10–3.12)
- Conexão com a internet na primeira execução (para baixar os modelos de
  embedding/reranking, ~170 MB, e para chamar a API do LLM)
- ~2 GB de RAM livres

---

## 🚀 Instalação (passo a passo)

### 1. Clone ou baixe o projeto

```bash
git clone https://github.com/SEU-USUARIO/bimbambuy-rag-agent.git
cd bimbambuy-rag-agent
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure a chave da API (gratuita)

**Opção recomendada — Google Gemini:**

1. Acesse https://aistudio.google.com/apikey
2. Faça login com sua conta Google e clique em **"Create API key"**
3. Copie a chave gerada

Depois, crie o arquivo `.env` a partir do exemplo:

```bash
cp .env.example .env
```

Edite o `.env` e cole sua chave:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=sua_chave_aqui
```

> **Alternativas gratuitas:** troque `LLM_PROVIDER` para `groq`
> (chave em https://console.groq.com/keys) ou `huggingface`
> (token em https://huggingface.co/settings/tokens) e preencha a
> variável correspondente no `.env`.

### 5. Execute o agente

```bash
streamlit run app.py
```

O navegador abrirá automaticamente em `http://localhost:8501`.

> ⏳ **Primeira execução:** o sistema baixa os modelos de IA locais e indexa
> os documentos — pode levar alguns minutos. Nas execuções seguintes, a
> inicialização é quase instantânea (o banco vetorial fica salvo em disco).

---

## 🔄 Fluxo de funcionamento

```
                 ┌──────────────────────────────────────────────────┐
                 │              INICIALIZAÇÃO (1 vez)               │
                 │                                                  │
                 │  PDFs/DOCX/... ──► Limpeza ──► Chunking por      │
                 │  seção (overlap) ──► Embeddings ──► ChromaDB     │
                 └──────────────────────────────────────────────────┘

 Pergunta do colaborador
        │
        ▼
 ┌─────────────┐    top 10     ┌───────────────┐   top 5   ┌─────────────┐
 │ Busca       │ ────────────► │ Reranking     │ ────────► │ Prompt com  │
 │ semântica   │               │ cross-encoder │           │ contexto +  │
 │ (ChromaDB)  │               │               │           │ metadados   │
 └─────────────┘               └───────────────┘           └──────┬──────┘
                                                                  │
                                                                  ▼
 ┌────────────────────────────────────────────────────────────────────┐
 │  LLM (Gemini) com regras rígidas:                                  │
 │  "Responda APENAS com base no contexto. Cite [Fonte N].            │
 │   Se não souber, diga que não encontrou."                          │
 └────────────────────────────────────────────────────────────────────┘
        │
        ▼
 Resposta + fontes citadas (arquivo, página, seção) + métricas
```

---

## 📸 O agente em funcionamento


**Perguntas de exemplo para demonstrar o sistema:**

- `Qual o prazo para solicitar a devolução de um produto?`
- `Quais métodos de pagamento a BimBam Buy aceita?`
- `O que a garantia não cobre?`
- `Como funciona o pagamento das comissões de afiliados?`
- `Existe frete grátis? Qual o valor mínimo?`
- `Qual a capital da França?` → *deve responder que não encontrou a
  informação nos documentos (teste de fallback)*

---

## 📄 Licença

Projeto educacional — uso livre. A "BimBam Buy" é uma empresa fictícia
criada para fins de demonstração.
