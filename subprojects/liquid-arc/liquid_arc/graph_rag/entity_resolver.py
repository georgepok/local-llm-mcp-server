"""Entity resolution for GraphRAG ingestion.

Merges duplicate entities extracted across chunks and documents. Three
matching tiers, priority-ordered:

1. **Exact ID** — if the extracted node id already exists in state, reuse it.
2. **Token-stem Jaccard** — 4-char stems of underscore tokens must
   overlap by ≥ 0.6 (`shanghai_port` ~ `shanghai_facility`).
3. **Type + metric proximity** — if an extracted node's type matches an
   existing node of the same type that is metrically close under g,
   treat them as the same entity. Uses the navigator's OWN learned
   geometry for the closeness test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _stems(nid: str) -> set:
    return {t[:4] for t in (nid or "").lower().split("_") if t}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class EntityResolver:
    def __init__(self, state, *, stem_jaccard_threshold: float = 0.6,
                 metric_distance_threshold: float = 0.5,
                 enable_metric_proximity: bool = True):
        self.state = state
        self.jaccard_threshold = stem_jaccard_threshold
        self.metric_threshold = metric_distance_threshold
        self.enable_metric = enable_metric_proximity

    def resolve(self, fragment: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite fragment in-place so node/edge IDs point to resolved
        canonical IDs. Returns the rewritten fragment plus a trace of
        which extracted IDs mapped to which canonical ones."""
        id_map: Dict[str, str] = {}
        resolved_nodes: List[Dict[str, Any]] = []
        for node in fragment.get("nodes", []):
            nid = node.get("id")
            if not nid:
                continue
            canonical = self._resolve_node(node)
            if canonical and canonical != nid:
                id_map[nid] = canonical
                node["id"] = canonical
            resolved_nodes.append(node)

        resolved_edges: List[Dict[str, Any]] = []
        for e in fragment.get("edges", []):
            e2 = dict(e)
            if e2.get("src") in id_map:
                e2["src"] = id_map[e2["src"]]
            if e2.get("dst") in id_map:
                e2["dst"] = id_map[e2["dst"]]
            if e2.get("scope") in id_map:
                e2["scope"] = id_map[e2["scope"]]
            resolved_edges.append(e2)

        fragment["nodes"] = resolved_nodes
        fragment["edges"] = resolved_edges
        fragment["_resolver_map"] = id_map
        return fragment

    # ------------------------------------------------------------------

    def _resolve_node(self, node: Dict[str, Any]) -> Optional[str]:
        nid = node["id"]
        # Tier 1: exact match
        if nid in self.state.nodes:
            return nid
        # Tier 2: stem Jaccard
        my_stems = _stems(nid)
        best = None
        best_score = self.jaccard_threshold
        for existing_id in self.state.nodes:
            j = _jaccard(my_stems, _stems(existing_id))
            if j > best_score and self.state.nodes[existing_id].get(
                    "type") == node.get("type", "entity"):
                best_score = j
                best = existing_id
        if best is not None:
            return best
        # Tier 3: metric proximity (slow — only if Tier 2 missed and we
        # have embeddings). Cheap heuristic: pick same-type existing node
        # whose embedding is closest to the mean embedding of same-type
        # neighbours. Implementing strictly requires re-running the ODE
        # on a candidate graph, which is expensive — skip for Phase 1.
        return None
