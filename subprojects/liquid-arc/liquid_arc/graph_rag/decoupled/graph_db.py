"""KnowledgeGraphDB — NetworkX-backed knowledge store.

All graph-algorithmic operations (BFS, scope filter, community
detection, neighborhood expansion, subgraph extraction, text retrieval)
happen here. No ODE. No navigator. Scales to millions of nodes.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Set

import networkx as nx


CAUSAL_EDGE_TYPES = frozenset({"causes", "precedes", "enables", "depends_on",
                                "requires", "blocks", "related_to", "is_a"})
BFS_CAUSAL_BACKWARDS = frozenset({"causes", "precedes", "enables", "depends_on"})


class KnowledgeGraphDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.G: nx.DiGraph = nx.DiGraph()
        self.text_segments: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Write path (no ODE)
    # ------------------------------------------------------------------

    def add_fragment(self, fragment: Dict[str, Any],
                     source_text: Optional[str] = None,
                     chunk_id: Optional[str] = None,
                     doc_metadata: Optional[Dict[str, Any]] = None,
                     autosave: bool = True) -> Dict[str, int]:
        """Merge a typed-graph fragment. O(|V|+|E|). Returns added counts."""
        now = time.time()
        added_nodes = 0
        for n in fragment.get("nodes", []):
            nid = n.get("id")
            if not nid:
                continue
            if self.G.has_node(nid):
                self.G.nodes[nid]["mention_count"] = (
                    self.G.nodes[nid].get("mention_count", 1) + 1)
                self.G.nodes[nid]["last_seen"] = now
                if n.get("type"):
                    self.G.nodes[nid]["type"] = n["type"]
                if n.get("role"):
                    self.G.nodes[nid]["role"] = n["role"]
            else:
                self.G.add_node(
                    nid,
                    type=n.get("type", "entity"),
                    role=n.get("role", "intermediate"),
                    first_seen=now,
                    last_seen=now,
                    mention_count=1,
                    doc_metadata=dict(doc_metadata) if doc_metadata else {},
                )
                added_nodes += 1
            if source_text:
                self.text_segments.setdefault(nid, []).append({
                    "text": source_text,
                    "timestamp": now,
                    "chunk_id": chunk_id,
                    "doc_metadata": dict(doc_metadata) if doc_metadata else {},
                })

        added_edges = 0
        for e in fragment.get("edges", []):
            src, dst = e.get("src"), e.get("dst")
            if not src or not dst:
                continue
            if not self.G.has_node(src) or not self.G.has_node(dst):
                continue
            et = e.get("type", "related_to")
            scope = e.get("scope")
            if self.G.has_edge(src, dst) and self.G[src][dst].get("type") == et \
                    and self.G[src][dst].get("scope") == scope:
                self.G[src][dst]["weight"] = self.G[src][dst].get("weight", 1) + 1
            else:
                self.G.add_edge(src, dst, type=et, scope=scope, weight=1)
                added_edges += 1

        if autosave:
            self._save()
        return {"added_nodes": added_nodes, "added_edges": added_edges}

    # ------------------------------------------------------------------
    # Read path — deterministic graph algorithms (no ODE)
    # ------------------------------------------------------------------

    def trace_causal_chain(self, target: str, max_hops: int = 10) -> Dict[str, Any]:
        """BFS backward from target through causal edges."""
        if target not in self.G:
            return {"root": None, "path": [], "hops": 0}
        path: List[str] = [target]
        current = target
        visited = {target}
        for _ in range(max_hops):
            preds: List[str] = []
            for src, _, data in self.G.in_edges(current, data=True):
                if (data.get("type") in BFS_CAUSAL_BACKWARDS
                        and src not in visited):
                    preds.append(src)
            if not preds:
                break
            preds.sort(key=lambda n: -self.G.nodes[n].get("mention_count", 1))
            current = preds[0]
            visited.add(current)
            path.append(current)
        path.reverse()
        return {"root": path[0], "path": path, "hops": max(0, len(path) - 1)}

    def scope_filter(self, scope: Optional[str]) -> nx.DiGraph:
        """Return a DiGraph view where out-of-scope edges are removed."""
        if not scope:
            return self.G
        filtered = nx.DiGraph()
        filtered.add_nodes_from(self.G.nodes(data=True))
        for u, v, data in self.G.edges(data=True):
            edge_scope = data.get("scope")
            if edge_scope is None or edge_scope == scope:
                filtered.add_edge(u, v, **data)
        return filtered

    def get_reachable(self, source: str, scope: Optional[str] = None,
                      max_hops: int = 5) -> Set[str]:
        """All descendants of `source` within max_hops, scope-filtered."""
        g = self.scope_filter(scope)
        if source not in g:
            return set()
        reachable: Set[str] = set()
        frontier: Set[str] = {source}
        for _ in range(max_hops):
            nxt: Set[str] = set()
            for node in frontier:
                for succ in g.successors(node):
                    if succ not in reachable and succ != source:
                        nxt.add(succ)
                        reachable.add(succ)
            frontier = nxt
            if not frontier:
                break
        return reachable

    def find_communities(self, min_size: int = 3) -> List[Set[str]]:
        """Louvain (fallback: greedy modularity → connected components)."""
        if self.G.number_of_nodes() < 2:
            return []
        u = self.G.to_undirected()
        try:
            comms = nx.community.louvain_communities(u, seed=0)
        except Exception:
            try:
                comms = nx.community.greedy_modularity_communities(u)
            except Exception:
                comms = nx.connected_components(u)
        return [set(c) for c in comms if len(c) >= min_size]

    def get_neighbors(self, node_ids: Iterable[str],
                      hops: int = 2,
                      direction: str = "both") -> Set[str]:
        """k-hop neighborhood. direction ∈ {'both','forward','backward'}."""
        seeds = {n for n in node_ids if n in self.G}
        if not seeds:
            return set()
        hood: Set[str] = set(seeds)
        frontier: Set[str] = set(seeds)
        for _ in range(hops):
            nxt: Set[str] = set()
            for node in frontier:
                if direction in ("both", "forward"):
                    nxt.update(self.G.successors(node))
                if direction in ("both", "backward"):
                    nxt.update(self.G.predecessors(node))
            frontier = nxt - hood
            hood.update(frontier)
            if not frontier:
                break
        return hood

    def extract_subgraph(self, node_ids: Iterable[str],
                         max_nodes: int = 200) -> Dict[str, Any]:
        """Extract a subgraph for ODE processing. Cap at max_nodes,
        prioritizing highest-mention nodes when capping is required."""
        nodes = {n for n in node_ids if n in self.G}
        if len(nodes) > max_nodes:
            scored = sorted(
                nodes,
                key=lambda n: -self.G.nodes[n].get("mention_count", 1),
            )
            nodes = set(scored[:max_nodes])
        out_nodes = [
            {"id": n,
             "type": self.G.nodes[n]["type"],
             "role": self.G.nodes[n]["role"]}
            for n in nodes
        ]
        out_edges = []
        for u, v, data in self.G.edges(data=True):
            if u in nodes and v in nodes:
                ed = {"src": u, "dst": v, "type": data.get("type", "related_to")}
                if data.get("scope") is not None:
                    ed["scope"] = data["scope"]
                out_edges.append(ed)
        return {"nodes": out_nodes, "edges": out_edges}

    def retrieve_text(self, node_ids: Iterable[str],
                      max_segments: int = 10) -> List[Dict[str, Any]]:
        """Text snippets linked to any of `node_ids`, recency-ordered, deduped.

        Dedupe uses a composite key `(text, chunk_id, doc_metadata)` so
        two segments that happen to share identical text but come from
        different documents / chunks aren't incorrectly collapsed
        (found by Qwen3-Next-80B agent, Stage 2c).
        """
        segments: List[Dict[str, Any]] = []
        for nid in node_ids:
            segments.extend(self.text_segments.get(nid, []))
        segments.sort(key=lambda s: -s.get("timestamp", 0))
        seen: set = set()
        unique: List[Dict[str, Any]] = []
        for s in segments:
            t = s.get("text", "")
            chunk_id = s.get("chunk_id")
            doc_md = frozenset((s.get("doc_metadata") or {}).items())
            key = (t, chunk_id, doc_md)
            if t and key not in seen:
                seen.add(key)
                unique.append(s)
            if len(unique) >= max_segments:
                break
        return unique

    # ------------------------------------------------------------------
    # Stats / persistence
    # ------------------------------------------------------------------

    def stats(self, compute_communities: bool = False) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "n_nodes": self.G.number_of_nodes(),
            "n_edges": self.G.number_of_edges(),
            "n_text_segments": sum(len(v) for v in self.text_segments.values()),
            "node_types": dict(Counter(
                data.get("type", "unknown")
                for _, data in self.G.nodes(data=True)
            )),
        }
        if compute_communities:
            stats["n_communities"] = len(self.find_communities())
        return stats

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)) or ".",
                    exist_ok=True)
        payload = {
            "nodes": [
                {"id": n, **data} for n, data in self.G.nodes(data=True)
            ],
            "edges": [
                {"src": u, "dst": v, **data}
                for u, v, data in self.G.edges(data=True)
            ],
            "text_segments": self.text_segments,
        }
        tmp = self.db_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, self.db_path)

    def _load(self) -> None:
        if not os.path.exists(self.db_path):
            return
        try:
            with open(self.db_path) as f:
                payload = json.load(f)
        except Exception:
            return
        for n in payload.get("nodes", []):
            attrs = {k: v for k, v in n.items() if k != "id"}
            self.G.add_node(n["id"], **attrs)
        for e in payload.get("edges", []):
            src, dst = e.get("src"), e.get("dst")
            if not src or not dst:
                continue
            attrs = {k: v for k, v in e.items() if k not in ("src", "dst")}
            self.G.add_edge(src, dst, **attrs)
        self.text_segments = payload.get("text_segments", {})

    def clear(self) -> None:
        self.G.clear()
        self.text_segments.clear()
        self._save()
