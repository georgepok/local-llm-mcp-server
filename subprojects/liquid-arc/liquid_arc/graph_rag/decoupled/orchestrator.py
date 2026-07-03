"""DecoupledGraphRAG — routes queries to graph-DB (cheap) or ODE (costly).

Design invariants:
  - Ingestion NEVER runs the ODE.
  - Causal / scope / connection queries NEVER run the ODE (graph DB only).
  - Topology / pattern queries invoke the ODE exactly once on a
    <=max_subgraph-sized subgraph.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import networkx as nx

from ..chunker import Chunker
from ..entity_resolver import EntityResolver
from ..router import QueryRouter
from ..vector_db import VectorDB
from .graph_db import KnowledgeGraphDB
from .ode_engine import SubgraphODEEngine


class _ShimState:
    """EntityResolver expects `state.nodes` as a dict. KnowledgeGraphDB
    exposes `G` (networkx). Shim with a minimal dict-ish interface."""

    def __init__(self, db: KnowledgeGraphDB):
        self._db = db

    @property
    def nodes(self) -> Dict[str, Dict[str, Any]]:
        return {n: data for n, data in self._db.G.nodes(data=True)}


class DecoupledGraphRAG:
    def __init__(self,
                 graph_db: KnowledgeGraphDB,
                 ode_engine: Optional[SubgraphODEEngine],
                 vector_db: VectorDB,
                 extractor,
                 pattern_library=None,
                 *,
                 chunker: Optional[Chunker] = None,
                 resolver: Optional[EntityResolver] = None,
                 router: Optional[QueryRouter] = None,
                 max_subgraph_nodes: int = 200,
                 doc_signature_enabled: bool = False):
        self.db = graph_db
        self.ode = ode_engine
        self.vector_db = vector_db
        self.extractor = extractor
        self.patterns = pattern_library
        self.chunker = chunker or Chunker()
        self.resolver = resolver or EntityResolver(_ShimState(graph_db))
        self.router = router or QueryRouter()
        self.max_subgraph_nodes = max_subgraph_nodes
        self.doc_signature_enabled = doc_signature_enabled

    # ------------------------------------------------------------------
    # Ingestion — no ODE
    # ------------------------------------------------------------------

    def ingest(self, doc_text: str,
               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        t0 = time.time()
        chunks = self.chunker.chunk(doc_text, metadata=metadata)
        n_frags = 0
        all_nodes_added: List[str] = []
        for ci, chunk in enumerate(chunks):
            chunk_meta = dict(chunk.get("metadata") or {})
            chunk_meta["chunk_id"] = chunk["chunk_id"]
            self.vector_db.add(chunk["text"], metadata=chunk_meta)
            try:
                fragment = self.extractor.extract(chunk["text"])
            except Exception:
                fragment = None
            if not fragment or not fragment.get("nodes"):
                continue
            fragment = self.resolver.resolve(fragment)
            pre = set(self.db.G.nodes)
            self.db.add_fragment(fragment, source_text=chunk["text"],
                                  chunk_id=f"c{ci}",
                                  doc_metadata=metadata,
                                  autosave=False)
            all_nodes_added.extend(
                [n for n in self.db.G.nodes if n not in pre])
            n_frags += 1
        # One persistence pass per document (avoid N writes per chunk).
        self.db._save()

        sig_stored = False
        if (self.doc_signature_enabled and self.patterns is not None
                and self.ode is not None and n_frags > 0):
            # Per-document signature: build a subgraph around the nodes
            # introduced by this doc and compute its signature.
            seeds = set(all_nodes_added) if all_nodes_added else set()
            if seeds:
                neigh = self.db.get_neighbors(seeds, hops=2)
                sub = self.db.extract_subgraph(neigh, max_nodes=50)
                if len(sub["nodes"]) >= 2:
                    try:
                        sig = self.ode.compute_signature(sub)
                        label = f"doc:{(metadata or {}).get('title','unknown')}"
                        self.patterns.store(sig, {"label": label,
                                                   "source_query": metadata})
                        sig_stored = True
                    except Exception:
                        sig_stored = False

        return {
            "elapsed_s": time.time() - t0,
            "n_chunks": len(chunks),
            "n_fragments_merged": n_frags,
            "signature_stored": sig_stored,
            "stats": self.db.stats(),
        }

    # ------------------------------------------------------------------
    # Query — ODE only when required
    # ------------------------------------------------------------------

    def query(self, query_text: str, *, scope: Optional[str] = None,
              k_vector: int = 10, k_graph: int = 10,
              extracted_fragment: Optional[Dict[str, Any]] = None,
              _extractor_override: Any = None,
              ) -> Dict[str, Any]:
        t0 = time.time()

        # 1) Always: vector leg
        raw_k = k_vector * 4 if scope else k_vector
        vector_chunks = self.vector_db.query(query_text, k=raw_k)
        if scope:
            vector_chunks = [
                c for c in vector_chunks
                if (c.get("metadata") or {}).get("scope") == scope
            ][:k_vector]

        # 2) Attempt to pull entities from the query (cheap extraction).
        query_fragment = extracted_fragment
        if query_fragment is None:
            try:
                extractor = _extractor_override or self.extractor
                query_fragment = extractor.extract(query_text)
            except Exception:
                query_fragment = None
        query_nodes = [n["id"] for n in (query_fragment or {}).get("nodes", [])
                        if n.get("id") in self.db.G]

        # 3) Route
        modes = self.router.route(query_text, query_fragment)
        if scope and "scope" not in modes:
            modes.append("scope")
        timings: Dict[str, float] = {}
        graph_result: Dict[str, Any] = {}
        graph_chunks: List[Dict[str, Any]] = []
        structural_hint_lines: List[str] = []

        # 4) Causal tracing — graph DB only (no ODE)
        if "graph" in modes or "causal" in modes:
            t_c = time.time()
            for qn in query_nodes:
                chain = self.db.trace_causal_chain(qn)
                if chain["root"]:
                    graph_result["causal_chain"] = chain
                    structural_hint_lines.append(
                        f"Root cause: {chain['root']} "
                        f"({chain['hops']} hops; path: "
                        f"{' → '.join(chain['path'])})")
                    for seg in self.db.retrieve_text(chain["path"],
                                                      max_segments=5):
                        graph_chunks.append(self._text_to_chunk(seg, "graph"))
                    break
            timings["causal_ms"] = (time.time() - t_c) * 1000

        # 5) Scope filtering — graph DB only
        if "scope" in modes:
            t_s = time.time()
            if scope and query_nodes:
                reach: set = set()
                g_filt = self.db.scope_filter(scope)
                for qn in query_nodes:
                    if qn in g_filt:
                        try:
                            reach.update(nx.descendants(g_filt, qn))
                        except Exception:
                            pass
                graph_result["scope_filtered_nodes"] = list(reach)
                # IMPORTANT: only retrieve text for scope-reachable nodes
                # (not query_nodes themselves). Shared nodes like a topic
                # anchor have text segments spanning every scope — pulling
                # those would contaminate the scope answer. The reachable
                # nodes are scope-specific authorities and therefore only
                # have in-scope text.
                for seg in self.db.retrieve_text(list(reach),
                                                  max_segments=5):
                    graph_chunks.append(self._text_to_chunk(seg, "scope"))
            timings["scope_ms"] = (time.time() - t_s) * 1000

        # 6) Connection (reachability) — graph DB only
        if "connection" in modes and len(query_nodes) >= 2:
            t_cc = time.time()
            src, dst = query_nodes[0], query_nodes[1]
            try:
                connected = nx.has_path(self.db.G.to_undirected(), src, dst)
            except Exception:
                connected = False
            graph_result["connection"] = {
                "src": src, "dst": dst, "connected": connected}
            structural_hint_lines.append(
                f"{src} ↔ {dst}: {'connected' if connected else 'not connected'}")
            timings["connection_ms"] = (time.time() - t_cc) * 1000

        # 7) Topology — ODE needed
        ode_invoked = False
        if "topology" in modes:
            t_t = time.time()
            if query_nodes:
                hood = self.db.get_neighbors(query_nodes, hops=3)
            else:
                comms = self.db.find_communities()
                hood = max(comms, key=len) if comms else set(self.db.G.nodes)
            subgraph = self.db.extract_subgraph(
                hood, max_nodes=self.max_subgraph_nodes)
            if self.ode is not None and len(subgraph["nodes"]) >= 2:
                diag: Dict[str, Any] = {}
                try:
                    diag = self.ode.compute_diagnostics(subgraph)
                    ode_invoked = True
                except Exception as exc:
                    diag = {"error": str(exc)}
                raw_centrality = diag.get("per_node_centrality_metric_space", {})
                centrality = raw_centrality if isinstance(raw_centrality, dict) else {}
                # Always also compute reach-based SPOFs as a graph-DB
                # cross-check (matches the Phase 2 topology-digest fix).
                reach_spof = self._reach_spof(subgraph)
                top_metric = sorted(centrality.items(),
                                     key=lambda kv: -kv[1])[:5]
                graph_result["topology"] = {
                    "subgraph_size": len(subgraph["nodes"]),
                    "cv_g": diag.get("cv_g"),
                    "tau_mean": diag.get("tau_mean"),
                    "top_metric_centrality": top_metric,
                    "top_reach_spof": reach_spof[:5],
                    "communities": diag.get("metric_clusters"),
                }
                spof_ids = [n for n, _ in reach_spof[:5]]
                if spof_ids:
                    structural_hint_lines.append(
                        "Top SPOFs (by downstream reach): " +
                        ", ".join(spof_ids))
                for seg in self.db.retrieve_text(spof_ids, max_segments=5):
                    graph_chunks.append(self._text_to_chunk(seg, "topology"))
            timings["topology_ms"] = (time.time() - t_t) * 1000

        # 8) Pattern — ODE invoked to compute signature; cosine search is cheap
        if "metric" in modes and self.patterns is not None and self.ode is not None:
            t_p = time.time()
            if query_nodes:
                hood = self.db.get_neighbors(query_nodes, hops=2)
                sub = self.db.extract_subgraph(hood, max_nodes=50)
                if len(sub["nodes"]) >= 2:
                    try:
                        sig = self.ode.compute_signature(sub)
                        ode_invoked = True
                        match = self.patterns.find_nearest(sig, threshold=0.85)
                        if match:
                            graph_result["pattern_match"] = match
                            structural_hint_lines.append(
                                f"Matches pattern: {match['label']} "
                                f"(cosine={match['similarity']:.3f})")
                    except Exception as exc:
                        graph_result["pattern_error"] = str(exc)
            timings["pattern_ms"] = (time.time() - t_p) * 1000

        merged = self._merge_dedup(vector_chunks, graph_chunks)
        return {
            "query": query_text,
            "modes": modes,
            "chunks": merged[:k_vector + k_graph],
            "structural_hint": "\n".join(structural_hint_lines) or None,
            "graph_result": graph_result,
            "stats": {
                "vector_chunks": len(vector_chunks),
                "graph_chunks": len(graph_chunks),
                "db_nodes": self.db.G.number_of_nodes(),
                "db_edges": self.db.G.number_of_edges(),
                "ode_invoked": ode_invoked,
                "timings_ms": timings,
            },
            "elapsed_ms": (time.time() - t0) * 1000,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reach_spof(self, subgraph: Dict[str, Any]) -> List[Any]:
        """Downstream-reach SPOFs computed directly on the subgraph."""
        g = nx.DiGraph()
        for n in subgraph["nodes"]:
            g.add_node(n["id"])
        for e in subgraph["edges"]:
            g.add_edge(e["src"], e["dst"])
        reach = [(n, len(nx.descendants(g, n))) for n in g.nodes]
        reach.sort(key=lambda kv: -kv[1])
        return [(nid, r) for nid, r in reach if r > 0]

    def _text_to_chunk(self, seg: Dict[str, Any], source: str
                        ) -> Dict[str, Any]:
        return {
            "chunk_id": seg.get("chunk_id"),
            "text": seg["text"],
            "metadata": {
                "from": source,
                "timestamp": seg.get("timestamp"),
                "doc_metadata": seg.get("doc_metadata", {}),
            },
            "score": None,
        }

    @staticmethod
    def _merge_dedup(vector: List[Dict[str, Any]],
                     graph: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for c in vector + graph:
            t = (c.get("text") or "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(c)
        return out
