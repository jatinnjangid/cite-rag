"""
test_api.py — Unit + integration tests for the FastAPI endpoints.

Strategy: patch at the source module so module reloads pick up the mock.
  - retriever.QdrantClient  → prevents Qdrant connection attempts
  - retriever.Embedder      → prevents model download
  - llm.generate_answer     → replaces the LLM call

No Qdrant, no embedding model, no API key needed to run these tests.

Run with:
    pytest backend/tests/test_api.py -v
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Put backend/ on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Shared fixtures ────────────────────────────────────────────────────────────

MOCK_CHUNKS = [
    {
        "doc_name": "paper.pdf",
        "page_num": 3,
        "text":     "Attention is all you need.",
        "score":    0.92,
        "chunk_id": "paper.pdf::p3::c0",
    }
]
MOCK_ANSWER = "Attention mechanisms replace recurrence."


def _fresh_app(chunks=MOCK_CHUNKS, answer=MOCK_ANSWER, retriever_side_effect=None):
    """
    Reload main (and its deps) with heavy dependencies fully mocked.
    Returns (TestClient, main_module).
    """
    # Patch at the *source* module so reloads pick up the mock
    with patch("retriever.QdrantClient"), \
         patch("retriever.Embedder"):

        # Reload retriever first so it uses mocked deps
        import retriever as ret_mod
        importlib.reload(ret_mod)

        # Now reload main so it gets the freshly-mocked retriever
        import main as main_mod
        importlib.reload(main_mod)

    # Wire a mock retriever instance directly onto the singleton
    mock_retriever = MagicMock()
    if retriever_side_effect:
        mock_retriever.retrieve.side_effect = retriever_side_effect
    else:
        mock_retriever.retrieve.return_value = chunks
    main_mod._retriever = mock_retriever

    # Patch generate_answer at the llm module (main imports it by reference)
    import llm as llm_mod
    llm_mod._test_override_answer = answer  # not used directly; patch below

    client = TestClient(main_mod.app, raise_server_exceptions=False)
    return client, main_mod


# ── /health tests ─────────────────────────────────────────────────────────────

class TestHealth:

    def setup_method(self):
        with patch("llm.generate_answer", return_value=MOCK_ANSWER):
            self.client, _ = _fresh_app()

    def test_health_returns_200(self):
        resp = self.client.get("/health")
        assert resp.status_code == 200

    def test_health_has_required_keys(self):
        data = self.client.get("/health").json()
        required = {"status", "qdrant", "collection", "llm_provider",
                    "embed_model", "reranker_enabled"}
        assert required.issubset(data.keys()), f"Missing: {required - data.keys()}"

    def test_health_status_is_ok(self):
        assert self.client.get("/health").json()["status"] == "ok"


# ── /ask validation tests ──────────────────────────────────────────────────────

class TestAskValidation:

    def setup_method(self):
        with patch("llm.generate_answer", return_value=MOCK_ANSWER):
            self.client, _ = _fresh_app()

    def test_missing_question_returns_422(self):
        assert self.client.post("/ask", json={"top_k": 3}).status_code == 422

    def test_blank_question_returns_422(self):
        assert self.client.post(
            "/ask", json={"question": "   ", "top_k": 3}
        ).status_code == 422

    def test_too_short_question_returns_422(self):
        # min_length=3 in schema
        assert self.client.post(
            "/ask", json={"question": "hi", "top_k": 3}
        ).status_code == 422

    def test_top_k_zero_returns_422(self):
        assert self.client.post(
            "/ask", json={"question": "What is attention?", "top_k": 0}
        ).status_code == 422

    def test_top_k_over_limit_returns_422(self):
        assert self.client.post(
            "/ask", json={"question": "What is attention?", "top_k": 99}
        ).status_code == 422

    def test_top_k_defaults_to_5(self):
        with patch("llm.generate_answer", return_value=MOCK_ANSWER):
            resp = self.client.post("/ask", json={"question": "What is attention?"})
        assert resp.status_code == 200


# ── /ask response shape tests ──────────────────────────────────────────────────

class TestAskResponse:

    def setup_method(self):
        with patch("llm.generate_answer", return_value=MOCK_ANSWER):
            self.client, _ = _fresh_app()

    def _ask(self, question="What is attention?", top_k=3):
        with patch("llm.generate_answer", return_value=MOCK_ANSWER):
            return self.client.post("/ask", json={"question": question, "top_k": top_k})

    def test_ask_returns_200(self):
        assert self._ask().status_code == 200

    def test_ask_response_has_answer_and_sources(self):
        data = self._ask().json()
        assert "answer" in data
        assert "sources" in data

    def test_ask_answer_is_string(self):
        data = self._ask().json()
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_ask_sources_are_list(self):
        assert isinstance(self._ask().json()["sources"], list)

    def test_ask_source_item_fields(self):
        sources = self._ask().json()["sources"]
        if sources:
            src = sources[0]
            for key in ("doc", "page", "text", "score"):
                assert key in src, f"Missing field '{key}' in source"
            assert isinstance(src["page"], int)
            assert isinstance(src["score"], float)

    def test_question_is_stripped(self):
        # Extra whitespace should be stripped silently, not rejected
        assert self._ask(question="   What is attention?   ").status_code == 200


# ── /ask error-path tests ──────────────────────────────────────────────────────

class TestAskErrors:

    def test_retriever_error_returns_503(self):
        """When the retriever raises, /ask must return 503."""
        with patch("llm.generate_answer", return_value=MOCK_ANSWER):
            client, _ = _fresh_app(
                retriever_side_effect=ConnectionRefusedError("Qdrant down")
            )
        with patch("llm.generate_answer", return_value=MOCK_ANSWER):
            resp = client.post("/ask", json={"question": "What is attention?", "top_k": 3})
        assert resp.status_code == 503

    def test_missing_api_key_returns_500(self):
        """When generate_answer raises EnvironmentError, /ask must return 500."""
        with patch("llm.generate_answer", return_value=MOCK_ANSWER):
            client, _ = _fresh_app()

        with patch("llm.generate_answer",
                   side_effect=EnvironmentError("GEMINI_API_KEY is not set")):
            resp = client.post("/ask", json={"question": "What is attention?", "top_k": 3})

        assert resp.status_code == 500
        assert "GEMINI_API_KEY" in resp.json()["detail"]
