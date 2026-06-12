"""
VectorStore — Wrapper FAISS pour la recherche sémantique (RAG).

- Embeddings : sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- Index      : FAISS IndexFlatIP (cosine similarity via vecteurs L2-normalisés)
- Persistance : fichiers .faiss + .meta.json par collection
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Dict, List, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class _Collection:
    """Un index FAISS + ses métadonnées sur disque."""

    def __init__(self, name: str, persist_dir: str, embed_fn) -> None:
        self.name = name
        self._embed_fn = embed_fn
        self._lock = threading.Lock()

        self._index_path = os.path.join(persist_dir, f"{name}.faiss")
        self._meta_path = os.path.join(persist_dir, f"{name}.meta.json")

        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self._index = faiss.read_index(self._index_path)
            with open(self._meta_path, "r", encoding="utf-8") as f:
                self._meta: list[dict] = json.load(f)
            logger.info("FAISS '%s' chargée (%d docs).", name, len(self._meta))
        else:
            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self._meta: list[dict] = []
            logger.info("FAISS '%s' créée (vide).", name)

    def count(self) -> int:
        return len(self._meta)

    # ── Encode + normalise L2 ────────────────────────────────────────────

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(self._embed_fn(texts), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    # ── Persistance ──────────────────────────────────────────────────────

    def _save(self) -> None:
        faiss.write_index(self._index, self._index_path)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, ensure_ascii=False, indent=2)

    # ── Upsert ───────────────────────────────────────────────────────────

    def upsert(self, doc_id: str, content: str, metadata: dict | None = None) -> None:
        with self._lock:
            # Supprimer l'ancien doc si présent, puis reconstruire l'index
            old_len = len(self._meta)
            self._meta = [m for m in self._meta if m["id"] != doc_id]
            if len(self._meta) < old_len:
                self._rebuild()

            vec = self._encode([content])
            self._index.add(vec)
            self._meta.append({"id": doc_id, "content": content, "metadata": metadata or {}})
            self._save()

    def _rebuild(self) -> None:
        new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if self._meta:
            new_index.add(self._encode([m["content"] for m in self._meta]))
        self._index = new_index

    # ── Recherche ────────────────────────────────────────────────────────

    def query(self, text: str, n_results: int = 5, where: dict | None = None) -> list[dict]:
        if self._index.ntotal == 0:
            return []

        # Filtrage par métadonnées → sous-index temporaire
        if where:
            docs = [m for m in self._meta
                    if all(m["metadata"].get(k) == v for k, v in where.items())]
            if not docs:
                return []

            sub_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            sub_index.add(self._encode([d["content"] for d in docs]))

            k = min(n_results, len(docs))
            scores, indices = sub_index.search(self._encode([text]), k)

            return [
                {"id": docs[i]["id"], "content": docs[i]["content"],
                 "metadata": docs[i]["metadata"], "distance": float(1.0 - s)}
                for s, i in zip(scores[0], indices[0]) if i >= 0
            ]

        # Recherche globale
        k = min(n_results, self._index.ntotal)
        scores, indices = self._index.search(self._encode([text]), k)

        return [
            {"id": self._meta[i]["id"], "content": self._meta[i]["content"],
             "metadata": self._meta[i]["metadata"], "distance": float(1.0 - s)}
            for s, i in zip(scores[0], indices[0]) if 0 <= i < len(self._meta)
        ]


class VectorStore:
    """
    Point d'entrée FAISS. Lazy-init du modèle d'embedding.
    API : index_document / search.
    """

    def __init__(self, persist_directory: str = "./faiss_db"):
        self._persist_dir = persist_directory
        self._embedding_model = None
        self._collections: dict[str, _Collection] = {}
        self._init_lock = threading.Lock()

    def _ensure_init(self) -> None:
        if self._embedding_model is not None:
            return
        with self._init_lock:
            if self._embedding_model is not None:
                return
            from sentence_transformers import SentenceTransformer
            os.makedirs(self._persist_dir, exist_ok=True)
            logger.info("Chargement du modèle d'embedding : %s", EMBEDDING_MODEL)
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._embedding_model.encode(texts, convert_to_numpy=True)

    def _collection(self, name: str) -> _Collection:
        self._ensure_init()
        if name not in self._collections:
            self._collections[name] = _Collection(name, self._persist_dir, self._embed)
        return self._collections[name]

    # ── API publique ─────────────────────────────────────────────────────

    def index_document(self, collection_name: str, doc_id: str,
                       content: str, metadata: dict | None = None) -> None:
        """Indexe ou met à jour un document."""
        self._collection(collection_name).upsert(doc_id, content, metadata)

    def search(self, collection_name: str, query: str,
               n_results: int = 3, where: dict | None = None) -> list[dict]:
        """Recherche sémantique. Retourne [{id, content, metadata, distance}]."""
        coll = self._collection(collection_name)
        if coll.count() == 0:
            return []
        return coll.query(query, min(n_results, coll.count()), where)


# Singleton
vector_store = VectorStore()
