"""
retriever.py — Qdrant ANN search + optional cross-encoder reranking.

Two-stage retrieval:
  1. ANN search via Qdrant (fast, approximate) → top fetch_k candidates
  2. [Optional] Cross-encoder reranking → re-score and return top_k

The reranker is lazily loaded and only runs when RERANKER_ENABLED=true.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from qdrant_client import QdrantClient

from config import cfg
from embedder import Embedder


class Retriever:
    """
    Handles embedding a query and fetching the most relevant chunks.

    Usage:
        retriever = Retriever()
        results = retriever.retrieve("What is attention?", top_k=5)
    """

    def __init__(self):
        self._embedder = Embedder()
        self._client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)

        # Lazy-load reranker only if enabled (heavy model ~80 MB)
        self._reranker = None
        if cfg.reranker_enabled:
            self._reranker = _load_reranker(cfg.reranker_model)

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        fetch_multiplier: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Embed `question`, search Qdrant, optionally rerank, return top_k results.

        Args:
            question:         The user's question string.
            top_k:            Final number of results to return.
                              Defaults to cfg.default_top_k.
            fetch_multiplier: ANN fetches top_k * this before reranking.
                              Ignored when reranker is disabled.

        Returns:
            List of dicts (ordered best-first):
                doc_name  — source filename
                page_num  — 1-based page number
                text      — chunk text
                score     — similarity / reranker score (float)
                chunk_id  — unique chunk identifier
        """
        top_k = top_k or cfg.default_top_k

        # Step 1: embed the question
        query_vec = self._embedder.encode(question)[0]  # shape (dim,)

        # Step 2: ANN search — fetch more candidates when reranking
        fetch_k = top_k * fetch_multiplier if self._reranker else top_k
        hits = self._client.search(
            collection_name=cfg.qdrant_collection,
            query_vector=query_vec.tolist(),
            limit=fetch_k,
            with_payload=True,
        )

        if not hits:
            return []

        candidates = [
            {
                "doc_name": h.payload["doc_name"],
                "page_num": h.payload["page_num"],
                "text":     h.payload["text"],
                "score":    float(h.score),           # cosine similarity
                "chunk_id": h.payload["chunk_id"],
            }
            for h in hits
        ]

        # Step 3: optional cross-encoder reranking
        if self._reranker:
            candidates = _rerank(self._reranker, question, candidates, top_k)

        return candidates[:top_k]


# ── Reranker helpers (module-level so the model is only loaded once) ──────────

def _load_reranker(model_name: str):
    """Load a CrossEncoder model. Called at most once per process."""
    from sentence_transformers import CrossEncoder
    print(f"[reranker] Loading '{model_name}' …")
    model = CrossEncoder(model_name)
    print("[reranker] Loaded.")
    return model


def _rerank(
    reranker,
    question: str,
    candidates: List[Dict],
    top_k: int,
) -> List[Dict]:
    """
    Score each (question, chunk_text) pair with the cross-encoder,
    replace the cosine score with the reranker score, and sort descending.
    """
    pairs = [[question, c["text"]] for c in candidates]
    scores: np.ndarray = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["score"] = float(score)

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_k]
