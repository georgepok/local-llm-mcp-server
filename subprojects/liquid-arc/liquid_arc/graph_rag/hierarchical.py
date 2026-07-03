"""HierarchicalGraphRAG — community detection → subgraph → navigator.

Two-level processing for scale:

Level 1 (fast, scales to thousands of nodes):
  - Louvain (or greedy) community detection on the full graph
  - Identify which communities contain the query nodes
  - Expand 1 hop via community adjacency
  - Cap at `max_subgraph_nodes`

Level 2 (accurate, bounded):
  - Run the navigator's ODE pipeline on the selected subgraph only
  - Delegate analyze_graph / get_graph_diagnostics to the engine
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

import networkx as nx


class HierarchicalGraphRAG:
    def __init__(self, navigator, *, max_subgraph_nodes: int = 200):
        self.navigator = navigator
        self.max_subgraph_nodes = max_subgraph_nodes

    # ------------------------------------------------------------------
    # Community detection + selection
    # ------------------------------------------------------------------

    def _full_graph(self) -> nx.Graph:
        g = nx.Graph()
        for nid in self.navigator.state.nodes:
            g.add_node(nid)
        for e in self.navigator.state.edges:
            if e["src"] in self.navigator.state.nodes and e["dst"] in self.navigator.state.nodes:
                g.add_edge(e["src"], e["dst"])
        return g

    def _communities(self, g: nx.Graph) -> List[Set[str]]:
        # Louvain if available (networkx ≥ 3.0 ships nx.community.louvain_communities)
        try:
            return [set(c) for c in nx.community.louvain_communities(g, seed=0)]
        except Exception:
            # Fallback: greedy modularity
            try:
                return [set(c) for c in nx.community.greedy_modularity_communities(g)]
            except Exception:
                # Last resort: each weakly-connected component is a community
                return [set(c) for c in nx.connected_components(g)]

    def select_subgraph(self, query_nodes: List[str]) -> Dict[str, Any]:
        """Return {nodes, edges, communities_selected, coverage_stats}."""
        g = self._full_graph()
        if g.number_of_nodes() <= self.max_subgraph_nodes:
            return {
                "nodes": list(g.nodes),
                "edges": [{"src": u, "dst": v}
                          for u, v in nx.DiGraph(g).edges],
                "communities_selected": 1,
                "coverage_stats": {
                    "total_nodes": g.number_of_nodes(),
                    "subgraph_nodes": g.number_of_nodes(),
                    "capped": False,
                },
            }

        communities = self._communities(g)
        # Seed: communities containing any query node
        seed_ids = set()
        for i, c in enumerate(communities):
            if any(q in c for q in query_nodes):
                seed_ids.add(i)
        # If no hit, pick the two largest communities as a default
        if not seed_ids:
            sized = sorted(enumerate(communities), key=lambda x: -len(x[1]))
            seed_ids = {sized[0][0], sized[1][0]} if len(sized) > 1 else {0}

        # Expand 1-hop in the community graph: a community adjacent to a
        # seed shares at least one edge with it.
        community_of: Dict[str, int] = {}
        for i, c in enumerate(communities):
            for n in c:
                community_of[n] = i
        adj: Dict[int, Set[int]] = {i: set() for i in range(len(communities))}
        for u, v in g.edges:
            cu, cv = community_of.get(u), community_of.get(v)
            if cu is not None and cv is not None and cu != cv:
                adj[cu].add(cv)
                adj[cv].add(cu)
        expanded = set(seed_ids)
        for i in seed_ids:
            expanded |= adj[i]

        # Assemble subgraph, capped
        subgraph_nodes: Set[str] = set()
        ordered = sorted(expanded,
                          key=lambda i: -len(communities[i] & set(query_nodes)))
        for i in ordered:
            subgraph_nodes |= communities[i]
            if len(subgraph_nodes) > self.max_subgraph_nodes:
                break
        # Ensure all anchor query nodes are included even if we overshot
        for q in query_nodes:
            if q in self.navigator.state.nodes:
                subgraph_nodes.add(q)

        sg_dir = nx.DiGraph()
        for nid in subgraph_nodes:
            sg_dir.add_node(nid)
        for e in self.navigator.state.edges:
            if e["src"] in subgraph_nodes and e["dst"] in subgraph_nodes:
                sg_dir.add_edge(e["src"], e["dst"])

        return {
            "nodes": list(sg_dir.nodes),
            "edges": [
                {"src": u, "dst": v, **self._edge_attrs(u, v)}
                for u, v in sg_dir.edges
            ],
            "communities_selected": len(expanded),
            "coverage_stats": {
                "total_nodes": g.number_of_nodes(),
                "subgraph_nodes": sg_dir.number_of_nodes(),
                "capped": sg_dir.number_of_nodes() >= self.max_subgraph_nodes,
            },
        }

    def _edge_attrs(self, u: str, v: str) -> Dict[str, Any]:
        for e in self.navigator.state.edges:
            if e["src"] == u and e["dst"] == v:
                return {k: val for k, val in e.items() if k not in ("src", "dst")}
        return {}

    # ------------------------------------------------------------------
    # Query path: community-select → engine.analyze_graph
    # ------------------------------------------------------------------

    def process_query(self, query_nodes: List[str],
                      query_spec: Dict[str, Any]) -> Dict[str, Any]:
        sub = self.select_subgraph(query_nodes)
        graph_json = json.dumps({
            "nodes": [
                {"id": nid,
                 "type": self.navigator.state.nodes[nid]["type"],
                 "role": self.navigator.state.nodes[nid]["role"]}
                for nid in sub["nodes"]
                if nid in self.navigator.state.nodes
            ],
            "edges": sub["edges"],
        })
        try:
            raw = self.navigator.engine.analyze_graph(
                graph_json, json.dumps(query_spec))
            result = json.loads(raw)
        except Exception as exc:
            result = {"error": f"engine: {exc}"}
        result["_subgraph_stats"] = sub["coverage_stats"]
        return result
