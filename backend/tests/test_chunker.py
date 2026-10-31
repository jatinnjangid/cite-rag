"""
test_chunker.py — Unit tests for chunker.py

These tests are pure Python + tiktoken (no Qdrant, no embedding model).
Run with: pytest backend/tests/test_chunker.py -v
"""

import sys
from pathlib import Path

# Make sure backend/ is importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tiktoken

from chunker import chunk_text, chunk_page


# ── Shared helper ──────────────────────────────────────────────────────────────

def _token_count(text: str) -> int:
    """Count tokens in a string using the same encoder as chunker.py."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ── chunk_text tests ───────────────────────────────────────────────────────────

class TestChunkText:

    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_whitespace_only_returns_empty_list(self):
        assert chunk_text("   \n\t  ") == []

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size must produce exactly one chunk."""
        text = "Hello world, this is a short sentence."
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0]["chunk_idx"] == 0
        assert chunks[0]["n_tokens"] <= 500

    def test_chunk_text_round_trips(self):
        """Joining all chunks should reconstruct all tokens (with overlap)."""
        # Build a long deterministic text (~800 tokens)
        text = ("The quick brown fox jumps over the lazy dog. " * 80).strip()
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=20)

        assert len(chunks) >= 2
        # Every token in the original must appear in at least one chunk
        enc = tiktoken.get_encoding("cl100k_base")
        original_tokens = enc.encode(text)
        covered_tokens = set()
        for chunk in chunks:
            c_start = chunk["token_start"]
            c_end = chunk["token_end"]
            for idx in range(c_start, c_end):
                covered_tokens.add(idx)

        assert covered_tokens == set(range(len(original_tokens))), (
            "Some tokens are not covered by any chunk"
        )

    def test_chunk_size_respected(self):
        """Every chunk must have n_tokens <= chunk_size."""
        text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100)
        chunk_size = 150
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=20)

        for chunk in chunks:
            assert chunk["n_tokens"] <= chunk_size, (
                f"Chunk {chunk['chunk_idx']} has {chunk['n_tokens']} tokens > {chunk_size}"
            )

    def test_overlap_is_shared(self):
        """
        The last `chunk_overlap` tokens of chunk[i] should equal
        the first `chunk_overlap` tokens of chunk[i+1].
        """
        text = ("word " * 300).strip()
        chunk_size = 100
        overlap = 20
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=overlap)

        enc = tiktoken.get_encoding("cl100k_base")
        all_tokens = enc.encode(text)

        for i in range(len(chunks) - 1):
            tail_tokens = all_tokens[chunks[i]["token_end"] - overlap: chunks[i]["token_end"]]
            head_tokens = all_tokens[chunks[i + 1]["token_start"]: chunks[i + 1]["token_start"] + overlap]
            assert tail_tokens == head_tokens, (
                f"Overlap mismatch between chunk {i} and {i+1}"
            )

    def test_chunk_indices_are_sequential(self):
        """chunk_idx must be 0, 1, 2, ... with no gaps."""
        text = ("sentence number one. " * 200).strip()
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        indices = [c["chunk_idx"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_token_start_end_are_consistent(self):
        """token_end - token_start must equal n_tokens for every chunk."""
        text = ("data " * 250).strip()
        chunks = chunk_text(text, chunk_size=120, chunk_overlap=15)
        for chunk in chunks:
            assert chunk["token_end"] - chunk["token_start"] == chunk["n_tokens"]

    def test_invalid_overlap_raises(self):
        """chunk_overlap >= chunk_size must raise ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("some text", chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_text("some text", chunk_size=100, chunk_overlap=200)

    def test_text_exactly_one_chunk_size(self):
        """Text with exactly chunk_size tokens must produce exactly 1 chunk."""
        enc = tiktoken.get_encoding("cl100k_base")
        chunk_size = 50
        # Build a token list of exactly chunk_size tokens, decode back to text
        base_token = enc.encode("word ")[0]
        tokens = [base_token] * chunk_size
        text = enc.decode(tokens)

        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=5)
        assert len(chunks) == 1
        assert chunks[0]["n_tokens"] == chunk_size

    def test_last_chunk_may_be_smaller(self):
        """The final chunk is allowed to be smaller than chunk_size."""
        text = ("token " * 210).strip()  # ~210 tokens
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        last = chunks[-1]
        assert last["n_tokens"] <= 100  # ≤ chunk_size


# ── chunk_page tests ───────────────────────────────────────────────────────────

class TestChunkPage:

    def test_metadata_attached(self):
        """chunk_page must attach doc_name, page_num, and chunk_id."""
        text = ("This is page text. " * 50).strip()
        chunks = chunk_page(
            page_text=text,
            doc_name="paper.pdf",
            page_num=3,
            chunk_size=100,
            chunk_overlap=10,
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["doc_name"] == "paper.pdf"
            assert chunk["page_num"] == 3
            assert chunk["chunk_id"].startswith("paper.pdf::p3::c")

    def test_chunk_id_uniqueness_within_page(self):
        """All chunk_ids within a single page must be unique."""
        text = ("unique token stream here. " * 100).strip()
        chunks = chunk_page(
            page_text=text,
            doc_name="doc.pdf",
            page_num=1,
            chunk_size=80,
            chunk_overlap=10,
        )
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids detected"

    def test_empty_page_returns_empty_list(self):
        chunks = chunk_page(
            page_text="",
            doc_name="empty.pdf",
            page_num=1,
        )
        assert chunks == []

    def test_chunk_id_format(self):
        """chunk_id format: '{doc_name}::p{page_num}::c{chunk_idx}'"""
        text = ("content " * 10).strip()
        chunks = chunk_page(
            page_text=text,
            doc_name="my_doc.pdf",
            page_num=7,
            chunk_size=500,
            chunk_overlap=50,
        )
        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "my_doc.pdf::p7::c0"
