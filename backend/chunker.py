"""
chunker.py — Token-based sliding-window text chunker.

Strategy:
  1. Encode the full text into tokens using tiktoken (cl100k_base).
  2. Slide a window of `chunk_size` tokens, stepping `chunk_size - chunk_overlap`
     tokens each time.
  3. Decode each window back to a string.

This produces chunks whose token counts are exact and predictable, which is
important for staying within LLM context limits.
"""

from typing import List, Dict
import tiktoken

# Use OpenAI's cl100k_base tokenizer — language-agnostic, fast, and consistent
# with how most modern LLMs count tokens.
_ENCODING_NAME = "cl100k_base"


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """
    Split `text` into overlapping token windows.

    Args:
        text:          Raw text to chunk (already extracted from a page).
        chunk_size:    Maximum tokens per chunk.
        chunk_overlap: Number of tokens shared between consecutive chunks.

    Returns:
        List of dicts, each containing:
            text        — decoded chunk string
            chunk_idx   — 0-based index within this page/document
            token_start — index of first token in the original token list
            token_end   — index one past the last token (exclusive)
            n_tokens    — number of tokens in this chunk

    Raises:
        ValueError: if chunk_overlap >= chunk_size (would cause infinite loop).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )

    text = text.strip()
    if not text:
        return []

    enc = tiktoken.get_encoding(_ENCODING_NAME)
    tokens: List[int] = enc.encode(text)

    if not tokens:
        return []

    chunks: List[Dict] = []
    step = chunk_size - chunk_overlap  # always > 0 due to the check above
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        chunks.append(
            {
                "text": enc.decode(chunk_tokens),
                "chunk_idx": len(chunks),
                "token_start": start,
                "token_end": end,
                "n_tokens": end - start,
            }
        )

        # If we reached the end of the token list, stop
        if end == len(tokens):
            break

        start += step

    return chunks


def chunk_page(
    page_text: str,
    doc_name: str,
    page_num: int,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """
    Chunk a single page and attach document/page metadata.

    Returns:
        List of chunk dicts with extra metadata fields:
            doc_name  — filename of the source PDF
            page_num  — 1-based page number
            chunk_id  — globally unique string "{doc_name}::p{page}::c{idx}"
    """
    raw_chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for chunk in raw_chunks:
        chunk["doc_name"] = doc_name
        chunk["page_num"] = page_num
        # Unique ID used as Qdrant point ID payload (not the UUID — that's separate)
        chunk["chunk_id"] = f"{doc_name}::p{page_num}::c{chunk['chunk_idx']}"

    return raw_chunks
