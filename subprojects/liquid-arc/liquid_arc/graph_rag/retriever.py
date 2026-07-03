"""GraphRAGRetriever — multi-modal retrieval.

At query time:
  1. QueryRouter decides which modes to run
  2. VectorDB returns top-k vector-similar chunks (always)
  3. Navigator (optional modes):
       graph    — causal-chain ancestors via state.query_relevant(mode='graph')
       metric   — type/role/topology-similar nodes via mode='metric'
       topology — emits topology_digest + top-SPOF nodes
       scope    — filter context_nodes by scope attribute
       connection — runs analyze_graph connection_check
  4. For each selected node, pull associated text segments from
     state.retrieve_text_for_nodes
  5. Merge vector + graph chunk sets; dedupe by text; keep union
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .router import QueryRouter
from .vector_db import VectorDB


class GraphRAGRetriever:
    def __init__(self, navigator, vector_db: VectorDB,
                 router: Optional[QueryRouter] = None):
        self.navigator = navigator
        self.vector_db = vector_db
        self.router = router or QueryRouter()

    def retrieve(self, query_text: str,
                 k_vector: int = 10, k_graph: int = 10,
                 extracted_fragment: Optional[Dict[str, Any]] = None,
                 scope: Optional[str] = None,
                 ) -> Dict[str, Any]:
        modes = self.router.route(query_text, extracted_fragment)
        # Pull more than k_vector when a scope filter will drop some.
        raw_k = k_vector * 4 if scope else k_vector
        vector_chunks = self.vector_db.query(query_text, k=raw_k)
        # Apply scope filter to vector leg as well — this is the
        # load-bearing difference from vector-only RAG: once the user
        # supplies a scope, only in-scope vector chunks survive.
        if scope:
            vector_chunks = [
                c for c in vector_chunks
                if (c.get("metadata") or {}).get("scope") == scope
            ][:k_vector]

        # Run navigator only if graph-style mode is requested. If the
        # query is simple factual ("what is X"), we skip navigator.
        graph_chunks: List[Dict[str, Any]] = []
        nav_result: Dict[str, Any] = {}
        structural_hint: Optional[str] = None
        pattern_match = None
        active_nodes: List[str] = []

        if any(m in modes for m in ("graph", "metric", "topology",
                                      "scope", "connection")):
            # Use the extracted fragment if supplied; otherwise the
            # navigator's _topology_digest path handles empty fragments.
            pre = extracted_fragment or {"nodes": [], "edges": []}
            nav_result = self.navigator.process_interaction(
                query_text, pre_extracted=pre)
            structural_hint = nav_result.get("rendered_hint")
            pattern_match = nav_result.get("pattern_match")

            context_nodes = nav_result.get("context_nodes", []) or []
            # Filter by scope if requested: keep nodes whose stored
            # metadata.scope matches the requested scope.
            if "scope" in modes and scope:
                context_nodes = [
                    n for n in context_nodes
                    if self._node_in_scope(n["id"], scope)
                ]
            active_nodes = [n["id"] for n in context_nodes[:k_graph]]
            segs = self.navigator.state.retrieve_text_for_nodes(
                active_nodes, max_segments=k_graph)
            for s in segs:
                graph_chunks.append({
                    "chunk_id": None,
                    "text": s["text"],
                    "metadata": {"from": "graph",
                                 "source_nodes": s.get("source_nodes", []),
                                 "interaction_index": s.get("interaction_index")},
                    "score": None,
                })

        merged = self._merge_and_dedup(vector_chunks, graph_chunks)
        return {
            "query": query_text,
            "modes": modes,
            "chunks": merged,
            "structural_hint": structural_hint,
            "pattern_match": pattern_match,
            "nav_analysis": nav_result.get("analysis"),
            "active_nodes": active_nodes,
            "retrieval_stats": {
                "vector_only": len(vector_chunks),
                "graph_only": len(graph_chunks),
                "merged": len(merged),
            },
        }

    def _node_in_scope(self, node_id: str, scope: str) -> bool:
        """Does the node participate in an edge gated by `scope`?
        If an edge from/to this node has edge['scope'] == scope, it's
        in-scope. Otherwise fall back to node type/role = scope match."""
        for e in self.navigator.state.edges:
            if e.get("scope") == scope and (
                    e["src"] == node_id or e["dst"] == node_id):
                return True
        meta = self.navigator.state.nodes.get(node_id, {})
        return meta.get("role") == "scope" and node_id == scope

    @staticmethod
    def _merge_and_dedup(vector: List[Dict[str, Any]],
                          graph: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_texts = set()
        merged: List[Dict[str, Any]] = []
        for c in vector + graph:
            t = (c.get("text") or "").strip()
            if not t or t in seen_texts:
                continue
            seen_texts.add(t)
            merged.append(c)
        return merged
