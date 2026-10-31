"""
embedder.py — Sentence-transformer embedding wrapper.

Downloads the model on first use (cached in ~/.cache/huggingface).
Normalises embeddings to unit length so cosine similarity == dot product,
which matches Qdrant's Cosine distance metric.
"""

from typing import List, Optional, Union
import numpy as np

from config import cfg


class Embedder:
    """
    Thin wrapper around SentenceTransformer.

    Usage:
        embedder = Embedder()
        vecs = embedder.encode(["hello world", "another sentence"])
        # vecs.shape == (2, 384) for MiniLM
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        Args:
            model_name: Override the model from config (useful in tests).
        """
        self._model_name = model_name or cfg.embed_model
        # Import lazily so tests can mock this without installing the library
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode one or more texts into L2-normalised embeddings.

        Args:
            texts:         A single string or a list of strings.
            batch_size:    Number of texts per forward pass.
            show_progress: Show tqdm progress bar (useful for large ingestion).

        Returns:
            np.ndarray of shape (n_texts, embedding_dim), float32, unit vectors.
        """
        if isinstance(texts, str):
            texts = [texts]

        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalise → cosine = dot product
            convert_to_numpy=True,
        )

    @property
    def dim(self) -> int:
        """Embedding dimensionality (e.g. 384 for MiniLM-L6-v2)."""
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name
