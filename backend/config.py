"""
config.py — Central configuration loaded from environment variables.

All other modules import `cfg` from here; no module calls os.getenv() directly.
Defaults make local development work without a .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv  # gracefully ignored if .env absent

# Load .env from the project root (two levels up from backend/)
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


class _Config:
    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "docqa")

    # ── Embedding model ───────────────────────────────────────────────────────
    embed_model: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "false").lower() == "true"
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" | "openai"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# Single shared instance — import this everywhere
cfg = _Config()
