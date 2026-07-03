"""GeometricState — persistent graph + metric landscape across interactions.

Spec: GEOMETRIC_NAVIGATOR_SPEC.md §1 (Persistent h_state Manager).

Stores graph structure and the ODE-processed positions of each node. Never
stores raw text. Every merge re-runs the frozen ContinuousDynamics over the
accumulated graph so embeddings and the metric reflect the full history.

Typical flow:
  state = GeometricState(path, engine, max_nodes=512)
  state.merge_fragment({"nodes":[...], "edges":[...]})   # extracted by LLM
  nearest = state.query_relevant(["node_x"], k=10)
  sig = state.get_signature()
"""

from __future__ import annotations

import collections
import json
import os
import time
from typing import Any, Dict, List, Optional

import networkx as nx
import torch

from .graph_engine_inference import (
    GraphEngine,
    _normalize_edges,
    _normalize_nodes,
)


class GeometricState:
    """Persistent geometric state over accumulated graph interactions.

    All ODE / MetricNet calls are delegated to `engine._forward_graph` so
    there is exactly one source of truth for the learned geometry.
    """

    def __init__(self, state_path: str, engine: GraphEngine,
                 max_nodes: int = 512):
        self.state_path = state_path
        self.engine = engine
        self.max_nodes = max_nodes

        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.clusters: List[Dict[str, Any]] = []
        self.signatures: List[Dict[str, Any]] = []

        # Fix 3: text segments per node (short snippets that mentioned it).
        self.text_segments: Dict[str, List[Dict[str, Any]]] = {}
        self._interaction_count: int = 0

        # Cached metric tensor g [N, d_metric] and order used to compute it.
        # Refreshed on every _recompute_geometry.
        self._g: Optional[torch.Tensor] = None
        self._order: List[str] = []

        if os.path.exists(self.state_path):
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge_fragment(self, fragment: Dict[str, Any],
                       source_text: Optional[str] = None) -> Dict[str, Any]:
        """Merge a new graph fragment into persistent state.

        Args:
            fragment:    {"nodes": [...], "edges": [...]}
            source_text: optional natural-language source; when provided,
                         it's indexed against each node in the fragment so
                         retrieve_text_for_nodes can later recover it.

        Returns a small report: how many new nodes/edges were added and
        whether a prune happened.
        """
        nodes_in = fragment.get("nodes") or []
        edges_in = fragment.get("edges") or []
        self._interaction_count += 1

        added_nodes = 0
        for n in nodes_in:
            nid = n.get("id")
            if not nid:
                continue
            if nid in self.nodes:
                self.nodes[nid]["mention_count"] = int(
                    self.nodes[nid].get("mention_count", 1)) + 1
                self.nodes[nid]["last_seen"] = time.time()
                # Allow role/type to be refined on later mentions
                if n.get("type"):
                    self.nodes[nid]["type"] = n["type"]
                if n.get("role"):
                    self.nodes[nid]["role"] = n["role"]
            else:
                self.nodes[nid] = {
                    "type": n.get("type", "entity"),
                    "role": n.get("role", "intermediate"),
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                    "mention_count": 1,
                }
                added_nodes += 1

        added_edges = 0
        for e in edges_in:
            src, dst = e.get("src"), e.get("dst")
            if not src or not dst:
                continue
            if src not in self.nodes or dst not in self.nodes:
                continue
            et = e.get("type", "related_to")
            scope = e.get("scope", None)
            existing = self._find_edge(src, dst, et, scope)
            if existing:
                existing["weight"] = int(existing.get("weight", 1)) + 1
            else:
                self.edges.append({
                    "src": src, "dst": dst, "type": et,
                    "scope": scope, "weight": 1,
                })
                added_edges += 1

        pruned = 0
        if len(self.nodes) > self.max_nodes:
            pruned = self._prune_oldest(len(self.nodes) - self.max_nodes)

        # Fix 3: index source text under every node in this fragment.
        if source_text:
            source_text = source_text.strip()
            for n in nodes_in:
                nid = n.get("id")
                if not nid or nid not in self.nodes:
                    continue
                bucket = self.text_segments.setdefault(nid, [])
                bucket.append({
                    "text": source_text,
                    "timestamp": time.time(),
                    "interaction_index": self._interaction_count,
                })
                # Keep per-node history bounded to avoid runaway growth.
                if len(bucket) > 8:
                    del bucket[:-8]

        # Recompute embeddings + metric landscape for the full graph.
        self._recompute_geometry()
        self._save()

        return {
            "added_nodes": added_nodes,
            "added_edges": added_edges,
            "pruned": pruned,
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
        }

    def query_relevant(self, query_nodes: List[str], k: int = 10,
                       mode: str = "metric") -> List[Dict[str, Any]]:
        """Return the k nodes most relevant to the query set.

        Args:
            query_nodes: anchor node IDs (must exist in state)
            k:           how many to return
            mode:        'metric' → type/role/topology similarity (under g)
                         'graph'  → causal-chain adjacency (shortest path on edges)
                         'both'   → union, deduped, scored by reciprocal-rank fusion

        The 'graph' mode was added in Phase 2 (NAVIGATOR_CONTINUATION_SPEC.md
        Fix 1) to complement 'metric'. Phase 1 showed the trained metric
        clusters by type/role, not chain adjacency; both retrievals are
        legitimate and serve different query shapes.
        """
        mode = (mode or "metric").lower()
        if mode == "graph":
            return self._query_graph_adjacent(query_nodes, k)
        if mode == "both":
            metric = self._query_metric(query_nodes, k=k)
            graph = self._query_graph_adjacent(query_nodes, k=k)
            return self._merge_results(metric, graph, k)
        return self._query_metric(query_nodes, k)

    def _query_metric(self, query_nodes: List[str], k: int
                      ) -> List[Dict[str, Any]]:
        if not query_nodes or not self.embeddings or self._g is None:
            return []

        q_embs = [self.embeddings[n] for n in query_nodes if n in self.embeddings]
        if not q_embs:
            return []
        q_vec = torch.stack(q_embs).mean(dim=0)

        # Build a mean-g approximation for distances (average of per-node g's
        # in the query set). Close enough for ranking.
        q_gs = []
        for n in query_nodes:
            if n in self._order:
                idx = self._order.index(n)
                q_gs.append(self._g[idx])
        q_g = torch.stack(q_gs).mean(dim=0) if q_gs else self._g.mean(dim=0)
        d_m = q_g.shape[-1]

        distances: List[tuple] = []
        for nid, emb in self.embeddings.items():
            if nid in query_nodes:
                continue
            # Diagonal metric: D² = sum_d g_d * (x_d - y_d)^2
            # Use average of g_query and g_node as symmetric metric.
            if nid in self._order:
                n_idx = self._order.index(nid)
                g_pair = 0.5 * (q_g + self._g[n_idx])
            else:
                g_pair = q_g
            diff = (q_vec[:d_m] - emb[:d_m]).to(g_pair.dtype)
            d_sq = float((diff * diff * g_pair).sum().item())
            distances.append((nid, d_sq))

        distances.sort(key=lambda x: x[1])
        out = []
        for nid, d in distances[:k]:
            out.append({
                "id": nid,
                "type": self.nodes[nid]["type"],
                "role": self.nodes[nid]["role"],
                "mention_count": self.nodes[nid].get("mention_count", 1),
                "metric_distance": d,
                "cluster_id": self._get_cluster(nid),
                "source": "metric",
            })
        return out

    def _query_graph_adjacent(self, query_nodes: List[str], k: int
                              ) -> List[Dict[str, Any]]:
        """Return up to k nodes reachable from the query set, ranked by
        undirected shortest-path distance on the accumulated edge set.

        Unlike the metric mode (type/role/topology similarity), this mode
        answers 'what's in the same narrative chain as X'.
        """
        query_set = set(query_nodes) & set(self.nodes)
        if not query_set:
            return []
        g = nx.Graph()
        for nid in self.nodes:
            g.add_node(nid)
        for e in self.edges:
            if e["src"] in self.nodes and e["dst"] in self.nodes:
                g.add_edge(e["src"], e["dst"])

        # Multi-source BFS: distance to the nearest query node.
        dist: Dict[str, int] = {q: 0 for q in query_set}
        frontier = collections.deque(query_set)
        while frontier:
            u = frontier.popleft()
            for v in g.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    frontier.append(v)

        ranked = sorted(
            ((nid, d) for nid, d in dist.items() if nid not in query_set),
            key=lambda x: x[1])
        out = []
        for nid, d in ranked[:k]:
            out.append({
                "id": nid,
                "type": self.nodes[nid]["type"],
                "role": self.nodes[nid]["role"],
                "mention_count": self.nodes[nid].get("mention_count", 1),
                "graph_distance": int(d),
                "cluster_id": self._get_cluster(nid),
                "source": "graph",
            })
        return out

    @staticmethod
    def _merge_results(metric: List[Dict[str, Any]],
                       graph: List[Dict[str, Any]],
                       k: int) -> List[Dict[str, Any]]:
        """Reciprocal-rank fusion: score(node) = 1/(60+rank_metric) +
        1/(60+rank_graph). The '60' damping is the standard RRF default.
        """
        rank_m = {n["id"]: i for i, n in enumerate(metric)}
        rank_g = {n["id"]: i for i, n in enumerate(graph)}
        by_id: Dict[str, Dict[str, Any]] = {}
        for n in metric + graph:
            by_id.setdefault(n["id"], n)
        scored = []
        for nid, n in by_id.items():
            score = 0.0
            if nid in rank_m:
                score += 1.0 / (60 + rank_m[nid])
            if nid in rank_g:
                score += 1.0 / (60 + rank_g[nid])
            scored.append((score, n))
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, n in scored[:k]:
            merged = dict(n)
            merged["rrf_score"] = score
            merged["source"] = (
                "both" if n["id"] in rank_m and n["id"] in rank_g
                else n.get("source", "?"))
            out.append(merged)
        return out

    def retrieve_text_for_nodes(self, node_ids: List[str],
                                max_segments: int = 5) -> List[Dict[str, Any]]:
        """Return deduped text snippets associated with the given nodes.

        Sorted by recency (most recent interaction first). Each item:
        {text, timestamp, interaction_index, source_nodes}.
        """
        by_text: Dict[str, Dict[str, Any]] = {}
        for nid in node_ids:
            for seg in self.text_segments.get(nid, []):
                t = seg["text"]
                if t not in by_text:
                    by_text[t] = {**seg, "source_nodes": [nid]}
                else:
                    if nid not in by_text[t]["source_nodes"]:
                        by_text[t]["source_nodes"].append(nid)
                    # Keep the most recent timestamp/index for this text
                    by_text[t]["timestamp"] = max(
                        by_text[t]["timestamp"], seg["timestamp"])
                    by_text[t]["interaction_index"] = max(
                        by_text[t]["interaction_index"], seg["interaction_index"])
        ordered = sorted(by_text.values(),
                         key=lambda s: -s["interaction_index"])
        return ordered[:max_segments]

    def get_signature(self) -> Optional[List[float]]:
        """Compute the current 52-dim + projection signature via the head.

        Returns a Python list for JSON portability. Returns None when the
        graph is too small to produce a meaningful signature.
        """
        if len(self.nodes) < 2:
            return None
        nodes_rec, edges_rec = self._records()
        nodes_norm = _normalize_nodes(nodes_rec)
        edges_norm = _normalize_edges(edges_rec)
        state = self.engine._forward_graph(nodes_norm, edges_norm)
        with torch.no_grad():
            sig = self.engine.head.signature(
                state["h_out"], state["g"], state["tau"],
                state["node_mask"], struct_features=state["struct"])
        return sig.squeeze(0).detach().cpu().float().tolist()

    def to_graph_dict(self) -> Dict[str, Any]:
        """Full graph in the shape analyze_graph expects."""
        nodes_rec, edges_rec = self._records()
        return {"nodes": nodes_rec, "edges": edges_rec}

    def reset(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.embeddings.clear()
        self.clusters = []
        self.signatures = []
        self.text_segments.clear()
        self._interaction_count = 0
        self._g = None
        self._order = []
        self._save()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _records(self):
        nodes_rec = [
            {"id": nid, "type": meta["type"], "role": meta["role"]}
            for nid, meta in self.nodes.items()
        ]
        edges_rec = [
            {k: v for k, v in e.items() if v is not None}
            for e in self.edges
        ]
        return nodes_rec, edges_rec

    def _find_edge(self, src: str, dst: str, et: str,
                   scope: Optional[str]) -> Optional[Dict[str, Any]]:
        for e in self.edges:
            if (e["src"] == src and e["dst"] == dst
                    and e["type"] == et and e.get("scope") == scope):
                return e
        return None

    def _prune_oldest(self, n: int) -> int:
        """Remove the n least-recently-seen nodes and their incident edges."""
        if n <= 0 or not self.nodes:
            return 0
        order = sorted(self.nodes.items(),
                       key=lambda x: x[1].get("last_seen", 0))
        victims = {nid for nid, _ in order[:n]}
        for nid in victims:
            self.nodes.pop(nid, None)
            self.embeddings.pop(nid, None)
            self.text_segments.pop(nid, None)
        self.edges = [e for e in self.edges
                      if e["src"] not in victims and e["dst"] not in victims]
        return len(victims)

    def _recompute_geometry(self) -> None:
        """Run ODE on the full accumulated graph and refresh embeddings."""
        if len(self.nodes) == 0:
            self._g = None
            self._order = []
            self.embeddings = {}
            self.clusters = []
            return

        nodes_rec, edges_rec = self._records()
        nodes_norm = _normalize_nodes(nodes_rec)
        edges_norm = _normalize_edges(edges_rec)

        try:
            state = self.engine._forward_graph(nodes_norm, edges_norm)
        except Exception as exc:
            # Geometry recompute is best-effort; keep old embeddings on failure.
            print(f"[GeometricState] _recompute_geometry failed: {exc}",
                  flush=True)
            return

        h_out = state["h_out"][0].detach()             # [N, d]
        g_tensor = state["g"][0].detach()              # [N, d_metric]
        order = state["order"]

        # Cluster computation uses the still-on-device tensors (fast matmul).
        self.clusters = self._compute_clusters(h_out, g_tensor, order)

        # Store on CPU so query_relevant / signature work without device churn.
        self._g = g_tensor.cpu()
        self._order = list(order)
        self.embeddings = {nid: h_out[i].cpu() for i, nid in enumerate(order)}

    def _compute_clusters(self, h: torch.Tensor, g: torch.Tensor,
                          order: List[str]) -> List[Dict[str, Any]]:
        N = h.shape[0]
        if N < 2:
            return [{"cluster_id": 0, "members": order, "size": N}] if N else []

        d_m = g.shape[-1]
        diff = h[:, :d_m].unsqueeze(1) - h[:, :d_m].unsqueeze(0)         # [N,N,d_m]
        g_avg = 0.5 * (g.unsqueeze(1) + g.unsqueeze(0))                  # [N,N,d_m]
        D2 = (diff * diff * g_avg).sum(dim=-1)                           # [N,N]
        iu, ju = torch.triu_indices(N, N, offset=1, device=h.device)
        off = D2[iu, ju]
        median = float(off.median().item()) if off.numel() else 0.0
        threshold = max(median * 0.5, 1e-6)

        assigned = [-1] * N
        clusters: List[Dict[str, Any]] = []
        cluster_id = 0
        for i in range(N):
            if assigned[i] != -1:
                continue
            stack = [i]
            members: List[str] = []
            while stack:
                u = stack.pop()
                if assigned[u] != -1:
                    continue
                assigned[u] = cluster_id
                members.append(order[u])
                for v in range(N):
                    if assigned[v] == -1 and float(D2[u, v].item()) < threshold:
                        stack.append(v)
            clusters.append({
                "cluster_id": cluster_id,
                "members": members,
                "size": len(members),
            })
            cluster_id += 1
        return clusters

    def _get_cluster(self, nid: str) -> Optional[int]:
        for c in self.clusters:
            if nid in c["members"]:
                return c["cluster_id"]
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        state = {
            "nodes": self.nodes,
            "edges": self.edges,
            "embeddings": {k: v.tolist() for k, v in self.embeddings.items()},
            "clusters": self.clusters,
            "signatures": self.signatures,
            "text_segments": self.text_segments,
            "interaction_count": self._interaction_count,
            "order": self._order,
            "g": self._g.tolist() if self._g is not None else None,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.state_path)) or ".",
                    exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, self.state_path)

    def _load(self) -> None:
        try:
            with open(self.state_path) as f:
                state = json.load(f)
        except Exception as exc:
            print(f"[GeometricState] load failed: {exc}", flush=True)
            return
        self.nodes = state.get("nodes", {})
        self.edges = state.get("edges", [])
        self.embeddings = {
            k: torch.tensor(v) for k, v in state.get("embeddings", {}).items()
        }
        self.clusters = state.get("clusters", [])
        self.signatures = state.get("signatures", [])
        self.text_segments = state.get("text_segments", {})
        self._interaction_count = int(state.get("interaction_count", 0))
        self._order = state.get("order", [])
        g_list = state.get("g")
        self._g = torch.tensor(g_list) if g_list is not None else None
