"""Embedding model wrapper.

Prefers a local `sentence-transformers` model (no paid API required for
document search). If the package or model weights aren't available in
the current environment (e.g. no internet access to download weights),
falls back to a small deterministic hashing-based embedding so the rest
of the retrieval pipeline (chunking -> FAISS -> search) still works
end-to-end for development and tests. This fallback is clearly logged;
it is NOT a semantic embedding and should not be relied on in
production — swap in a real sentence-transformers model there.
"""
from __future__ import annotations

import hashlib
import logging
import math

logger = logging.getLogger(__name__)

FALLBACK_DIM = 256


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._backend = None
        self.dim = FALLBACK_DIM
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._backend = SentenceTransformer(self.model_name)
            self.dim = self._backend.get_sentence_embedding_dimension()
            logger.info("Loaded sentence-transformers model '%s' (dim=%s)", self.model_name, self.dim)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Falling back to a deterministic hashing embedding (sentence-transformers "
                "unavailable: %s). Set EMBEDDING_MODEL / install sentence-transformers for "
                "real semantic search.",
                exc,
            )
            self._backend = None
            self.dim = FALLBACK_DIM

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self._backend is not None:
            vectors = self._backend.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vectors]
        return [_hashing_embedding(t, self.dim) for t in texts]


def _hashing_embedding(text: str, dim: int) -> list[float]:
    """Deterministic bag-of-tokens hashing embedding, L2-normalized.
    Good enough to exercise the retrieval pipeline without network access
    or heavyweight ML dependencies; not a substitute for real embeddings.
    """
    vec = [0.0] * dim
    tokens = text.lower().split()
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
