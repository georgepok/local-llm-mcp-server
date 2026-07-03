"""PatternLibrary — persistent metric-signature store for cross-session transfer.

Spec: GEOMETRIC_NAVIGATOR_SPEC.md §5 (Pattern Library).

Each pattern = one signature (list of floats, produced by
GraphOutputHead.signature → 64-d by default, topology-invariant) plus a
label and metadata. Lookup by cosine similarity. Patterns persist to a
single JSON file so they survive MCP restarts.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional


def _cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


class PatternLibrary:
    """List of {signature, label, metadata, count} records on disk.

    Small by design — we expect O(10²) patterns, not a large ANN index.
    """

    def __init__(self, library_path: str):
        self.library_path = library_path
        self.patterns: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_nearest(self, signature: List[float],
                     threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        if not self.patterns or not signature:
            return None
        best_sim = -1.0
        best = None
        for pattern in self.patterns:
            sim = _cosine(signature, pattern["signature"])
            if sim > best_sim:
                best_sim = sim
                best = pattern
        if best is None or best_sim < threshold:
            return None
        return {**best, "similarity": best_sim}

    def store(self, signature: List[float], metadata: Dict[str, Any]) -> None:
        # If we already have an almost-identical signature (cosine > 0.99),
        # increment its count rather than duplicate.
        for p in self.patterns:
            if _cosine(signature, p["signature"]) > 0.99:
                p["count"] = int(p.get("count", 1)) + 1
                p["last_seen"] = time.time()
                self._save()
                return
        self.patterns.append({
            "signature": list(signature),
            "label": metadata.get("label", "unnamed"),
            "source_query": metadata.get("source_query"),
            "timestamp": metadata.get("timestamp", time.time()),
            "last_seen": metadata.get("timestamp", time.time()),
            "count": 1,
        })
        self._save()

    def reset(self) -> None:
        self.patterns = []
        self._save()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.library_path)) or ".",
                    exist_ok=True)
        tmp = self.library_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.patterns, f)
        os.replace(tmp, self.library_path)

    def _load(self) -> None:
        if not os.path.exists(self.library_path):
            return
        try:
            with open(self.library_path) as f:
                self.patterns = json.load(f)
        except Exception as exc:
            print(f"[PatternLibrary] load failed: {exc}", flush=True)
            self.patterns = []
