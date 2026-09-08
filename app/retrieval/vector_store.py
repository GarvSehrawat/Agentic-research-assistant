"""A small FAISS wrapper that also persists chunk metadata alongside the
vector index, since FAISS itself only stores vectors and integer ids."""
from __future__ import annotations

import json
import logging
import os
import threading

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        self._lock = threading.Lock()
        os.makedirs(index_path, exist_ok=True)
        self._index_file = os.path.join(index_path, "index.faiss")
        self._meta_file = os.path.join(index_path, "metadata.json")

        import faiss  # imported here so importing this module doesn't
        # require faiss unless a vector store is actually created.

        self._faiss = faiss
        self._metadata: list[dict] = []
        self._index = faiss.IndexFlatL2(dim)
        self.load()

    def add_documents(self, vectors: list[list[float]], metadatas: list[dict]) -> None:
        if not vectors:
            return
        with self._lock:
            arr = np.array(vectors, dtype="float32")
            self._index.add(arr)
            self._metadata.extend(metadatas)

    def search(self, query_vector: list[float], k: int = 5, document_id: str | None = None) -> list[dict]:
        if self._index.ntotal == 0:
            return []
        with self._lock:
            arr = np.array([query_vector], dtype="float32")
            # Over-fetch a bit when filtering by document_id so we still
            # return k relevant results.
            fetch_k = min(self._index.ntotal, max(k * 4, k))
            distances, indices = self._index.search(arr, fetch_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx]
            if document_id and meta.get("document_id") != document_id:
                continue
            results.append({**meta, "score": float(dist)})
            if len(results) >= k:
                break
        return results

    def save(self) -> None:
        with self._lock:
            self._faiss.write_index(self._index, self._index_file)
            with open(self._meta_file, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f)

    def load(self) -> None:
        if os.path.exists(self._index_file) and os.path.exists(self._meta_file):
            try:
                self._index = self._faiss.read_index(self._index_file)
                with open(self._meta_file, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                logger.info("Loaded FAISS index with %s vectors from %s", self._index.ntotal, self.index_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load existing FAISS index, starting fresh: %s", exc)
                self._index = self._faiss.IndexFlatL2(self.dim)
                self._metadata = []

    def delete_document(self, document_id: str) -> None:
        """Rebuild the index without the given document's chunks.
        IndexFlatL2 doesn't support in-place deletion by id."""
        with self._lock:
            keep = [(m, v) for m, v in zip(self._metadata, self._reconstruct_all()) if m.get("document_id") != document_id]
            self._index = self._faiss.IndexFlatL2(self.dim)
            self._metadata = [m for m, _ in keep]
            if keep:
                arr = np.array([v for _, v in keep], dtype="float32")
                self._index.add(arr)

    def _reconstruct_all(self) -> list[list[float]]:
        return [self._index.reconstruct(i).tolist() for i in range(self._index.ntotal)]

    @property
    def size(self) -> int:
        return self._index.ntotal
