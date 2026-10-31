# DocQA — Architecture Notes

## Component Responsibilities

### backend/
- `main.py`      — FastAPI app, routes
- `ingest.py`    — CLI to parse PDFs, chunk, embed, upsert to Qdrant
- `chunker.py`   — Text chunking with configurable size/overlap
- `embedder.py`  — Wraps sentence-transformers
- `retriever.py` — Qdrant search + optional reranking
- `llm.py`       — LLM abstraction (Gemini / OpenAI)
- `config.py`    — Loads env vars, single source of truth

### eval/
- `questions.json` — Ground-truth Q&A pairs
- `run_eval.py`    — Hit-rate@k, MRR computation
- `results.md`     — Experiment results (auto-generated)

### data/
- `pdfs/`          — Input PDF files (git-ignored)

## Data Flow

1. **Ingestion**: PDF → pages → chunks (tokens) → embeddings → Qdrant
2. **Query**: question → embedding → ANN → [rerank] → LLM → response
3. **Eval**: questions.json → retrieval only → metrics

## Design Decisions

- Qdrant chosen over pgvector for simplicity (no Postgres schema needed).
- tiktoken used for token counting (language-agnostic, fast).
- Reranker runs only at query time; embeddings are always stored.
- LLM prompt enforces grounded answers ("answer only from context").
