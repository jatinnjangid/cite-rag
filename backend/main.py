"""
main.py — FastAPI application: /health and /ask endpoints.

Start with:
    uvicorn main:app --reload --port 8000

Endpoints:
    GET  /health   → liveness check, reports Qdrant connectivity
    POST /ask      → RAG question answering with cited sources
"""

import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Ensure backend/ is on sys.path when launched from repo root
sys.path.insert(0, str(Path(__file__).parent))

from config import cfg
import llm                  # import module, not function — keeps mock patches working
from retriever import Retriever

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocQA",
    description="RAG question-answering over PDFs with cited sources.",
    version="0.1.0",
)

# Allow the Vite dev server (port 5173) and any Vercel deployment to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared retriever instance (loads embedding model once at startup)
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Lazy-init the retriever so startup failures surface cleanly in logs."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# ── Request / Response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=1000,
        description="The question to answer from the PDF corpus."
    )
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Number of source chunks to retrieve (1–20)."
    )

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class SourceItem(BaseModel):
    doc:   str   = Field(description="Source document filename.")
    page:  int   = Field(description="1-based page number.")
    text:  str   = Field(description="Retrieved chunk text.")
    score: float = Field(description="Relevance score (higher = more relevant).")


class AskResponse(BaseModel):
    answer:  str              = Field(description="LLM-generated grounded answer.")
    sources: List[SourceItem] = Field(description="Ranked source chunks used to answer.")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="Liveness and Qdrant connectivity check")
def health():
    """
    Returns 200 if the API process is running.
    Also reports Qdrant connectivity and indexed chunk count.
    A Qdrant error does NOT make this endpoint fail — the API itself is still up.
    """
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
        info = client.get_collection(cfg.qdrant_collection)
        qdrant_status = "ok"
        indexed_chunks = info.points_count
    except Exception as exc:
        qdrant_status = f"unavailable ({exc})"
        indexed_chunks = None

    return {
        "status": "ok",
        "qdrant": qdrant_status,
        "indexed_chunks": indexed_chunks,
        "collection": cfg.qdrant_collection,
        "llm_provider": cfg.llm_provider,
        "embed_model": cfg.embed_model,
        "reranker_enabled": cfg.reranker_enabled,
    }


@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Answer a question from the PDF corpus",
)
def ask(request: AskRequest):
    """
    Two-stage RAG pipeline:
    1. Retrieve the top-k most relevant chunks from Qdrant.
    2. Pass chunks + question to the LLM with a grounding prompt.

    The LLM is instructed to answer ONLY from provided context.
    If no relevant context is found it returns "not found in documents".
    """
    # ── Stage 1: Retrieval ────────────────────────────────────────────────────
    try:
        retriever = get_retriever()
        chunks = retriever.retrieve(request.question, top_k=request.top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Retrieval failed — is Qdrant running and the corpus ingested? "
                f"Error: {exc}"
            ),
        )

    # ── Stage 2: Generation ───────────────────────────────────────────────────
    try:
        answer = llm.generate_answer(request.question, chunks)
    except EnvironmentError as exc:
        # Missing API key — actionable by the user
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    # ── Build response ────────────────────────────────────────────────────────
    sources = [
        SourceItem(
            doc=c["doc_name"],
            page=c["page_num"],
            text=c["text"],
            score=round(c["score"], 4),
        )
        for c in chunks
    ]

    return AskResponse(answer=answer, sources=sources)
