"""
ingest.py — CLI to parse PDFs, chunk text, embed chunks, and upsert to Qdrant.

Usage:
    python ingest.py --dir data/pdfs
    python ingest.py --dir data/pdfs --chunk-size 300 --chunk-overlap 30

The script is idempotent: re-running with the same PDFs will overwrite
existing points (matched by chunk_id in payload) rather than duplicating.
"""

import argparse
import hashlib
import sys
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

# Add backend/ to sys.path so sibling imports work when called from repo root
sys.path.insert(0, str(Path(__file__).parent))

from chunker import chunk_page
from config import cfg
from embedder import Embedder


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text page by page from a PDF.

    Returns a list of dicts:
        { "doc_name": str, "page_num": int (1-based), "text": str }

    Pages with no extractable text are skipped (scanned / image-only pages).
    """
    doc = fitz.open(str(pdf_path))
    pages = []
    for page_idx in range(len(doc)):
        text = doc[page_idx].get_text("text")  # plain-text extraction
        text = text.strip()
        if text:
            pages.append(
                {
                    "doc_name": pdf_path.name,
                    "page_num": page_idx + 1,  # 1-based
                    "text": text,
                }
            )
    doc.close()
    return pages


def _deterministic_uuid(chunk_id: str) -> str:
    """
    Generate a UUID5 (name-based) from the chunk_id string.
    Same chunk_id always maps to the same UUID, enabling idempotent upserts.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def _ensure_collection(client: QdrantClient, dim: int) -> None:
    """Create the Qdrant collection if it doesn't exist yet."""
    existing = {c.name for c in client.get_collections().collections}
    if cfg.qdrant_collection not in existing:
        client.create_collection(
            collection_name=cfg.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"[qdrant] Created collection '{cfg.qdrant_collection}' (dim={dim})")
    else:
        print(f"[qdrant] Collection '{cfg.qdrant_collection}' already exists")


# ── Main ingestion logic ───────────────────────────────────────────────────────

def ingest_directory(
    pdf_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int = 64,
) -> None:
    """
    Walk `pdf_dir`, parse every PDF, chunk, embed, and upsert to Qdrant.
    """
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[ingest] No PDF files found in {pdf_dir}")
        return

    print(f"[ingest] Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")
    print(f"[ingest] Chunk size={chunk_size} tokens, overlap={chunk_overlap} tokens")

    # Initialise embedding model (downloads on first run)
    print(f"[embed]  Loading model '{cfg.embed_model}' …")
    embedder = Embedder()
    print(f"[embed]  Model loaded. Embedding dim = {embedder.dim}")

    # Connect to Qdrant
    client = QdrantClient(host=cfg.qdrant_host, port=cfg.qdrant_port)
    _ensure_collection(client, embedder.dim)

    total_chunks = 0

    for pdf_path in tqdm(pdf_files, desc="PDFs", unit="file"):
        print(f"\n[parse]  Processing '{pdf_path.name}' …")
        pages = _parse_pdf(pdf_path)
        print(f"[parse]  {len(pages)} text pages extracted")

        # Collect all chunks from all pages in this PDF
        all_chunks = []
        for page in pages:
            chunks = chunk_page(
                page_text=page["text"],
                doc_name=page["doc_name"],
                page_num=page["page_num"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            print(f"[ingest] No chunks produced for '{pdf_path.name}', skipping.")
            continue

        print(f"[chunk]  {len(all_chunks)} chunks produced")

        # Embed in batches
        texts = [c["text"] for c in all_chunks]
        vectors = embedder.encode(texts, batch_size=batch_size, show_progress=True)

        # Build Qdrant point structs
        points = []
        for chunk, vector in zip(all_chunks, vectors):
            point_id = _deterministic_uuid(chunk["chunk_id"])
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "doc_name":  chunk["doc_name"],
                        "page_num":  chunk["page_num"],
                        "chunk_id":  chunk["chunk_id"],
                        "chunk_idx": chunk["chunk_idx"],
                        "text":      chunk["text"],
                        "n_tokens":  chunk["n_tokens"],
                        # Store config used for ingestion — helps debugging
                        "chunk_size":    chunk_size,
                        "chunk_overlap": chunk_overlap,
                    },
                )
            )

        # Upsert in batches (avoids hitting gRPC message size limits)
        upsert_batch = 128
        for i in range(0, len(points), upsert_batch):
            client.upsert(
                collection_name=cfg.qdrant_collection,
                points=points[i : i + upsert_batch],
            )

        total_chunks += len(all_chunks)
        print(f"[qdrant] Upserted {len(all_chunks)} points for '{pdf_path.name}'")

    print(f"\n[ingest] Done. Total chunks upserted: {total_chunks}")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs into the DocQA vector database."
    )
    parser.add_argument(
        "--dir",
        required=True,
        type=Path,
        help="Directory containing PDF files (e.g. data/pdfs).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=cfg.chunk_size,
        help=f"Tokens per chunk (default: {cfg.chunk_size}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=cfg.chunk_overlap,
        help=f"Overlapping tokens between chunks (default: {cfg.chunk_overlap}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size (default: 64).",
    )
    args = parser.parse_args()

    if not args.dir.is_dir():
        print(f"[error] '{args.dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    ingest_directory(
        pdf_dir=args.dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
