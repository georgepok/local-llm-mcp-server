"""Minimal in-memory vector DB with hashed-TF-IDF embeddings.

Dependency-free (stdlib + numpy). Serves as the baseline vector retrieval
leg for GraphRAG experiments. Production users would swap in ChromaDB,
FAISS, or a managed service; the interface is intentionally compatible.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, List, Optional


_TOK = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")
_STOP = frozenset(
    "a an the and or of to in for on with by as at from is it be are was "
    "were this that these those we you i he she they them our your their".split())


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOK.findall(text or "") if t.lower() not in _STOP
            and len(t) > 2]


def _hash_dim(token: str, dim: int) -> int:
    return int(hashlib.md5(token.encode()).hexdigest(), 16) % dim


def _tf_idf_vector(tokens: List[str], df_table: Dict[int, int],
                   total_docs: int, dim: int) -> List[float]:
    """Hashed TF-IDF vector of length `dim`. Deterministic given vocabulary."""
    tf: Dict[int, int] = {}
    for t in tokens:
        h = _hash_dim(t, dim)
        tf[h] = tf.get(h, 0) + 1
    if not tf:
        return [0.0] * dim
    max_tf = max(tf.values())
    vec = [0.0] * dim
    for h, f in tf.items():
        tf_val = 0.5 + 0.5 * (f / max_tf)                  # normalized TF
        idf = math.log((total_docs + 1) / (1 + df_table.get(h, 0)))
        vec[h] = tf_val * idf
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    return num  # both pre-normalized → dot == cosine


class VectorDB:
    """In-memory TF-IDF vector store.

    add(chunk_text, metadata) → stores the chunk, indexes its tokens.
    query(text, k) → returns top-k chunks by cosine similarity.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.chunks: List[Dict[str, Any]] = []  # {text, metadata, tokens, vector}
        self.df: Dict[int, int] = {}             # hashed doc-frequency

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        tokens = _tokenize(text)
        unique = set(_hash_dim(t, self.dim) for t in tokens)
        for h in unique:
            self.df[h] = self.df.get(h, 0) + 1
        self.chunks.append({
            "text": text,
            "metadata": dict(metadata or {}),
            "tokens": tokens,
            "vector": None,  # recomputed lazily when query needs current IDF
        })
        return len(self.chunks) - 1

    def _vectors(self) -> List[List[float]]:
        """Compute vectors for all chunks using the current IDF table."""
        total = len(self.chunks)
        out = []
        for c in self.chunks:
            v = _tf_idf_vector(c["tokens"], self.df, total, self.dim)
            c["vector"] = v
            out.append(v)
        return out

    def query(self, text: str, k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k chunks by cosine similarity to query text."""
        if not self.chunks:
            return []
        vecs = self._vectors()
        q_tokens = _tokenize(text)
        q_vec = _tf_idf_vector(q_tokens, self.df, len(self.chunks), self.dim)
        scored = []
        for i, v in enumerate(vecs):
            scored.append((i, _cosine(q_vec, v)))
        scored.sort(key=lambda x: -x[1])
        out = []
        for i, s in scored[:k]:
            c = self.chunks[i]
            out.append({
                "chunk_id": i,
                "text": c["text"],
                "metadata": c["metadata"],
                "score": s,
            })
        return out

    def __len__(self) -> int:
        return len(self.chunks)
