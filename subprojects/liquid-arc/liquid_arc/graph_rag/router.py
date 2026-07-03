"""QueryRouter — pick retrieval modes for a given query.

Heuristic triggers from Phase 2 observations:
  - Causal language → graph (ancestor traversal)
  - Topology language → topology digest
  - Scope language → scope-filtered retrieval
  - Pattern language → metric signature / pattern library
Vector retrieval always runs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


CAUSAL_TRIGGERS = ("cause", "causes", "why", "why did", "root", "root cause",
                   "led to", "resulted in", "because", "triggered by",
                   "what happened", "trace", "upstream")

TOPOLOGY_TRIGGERS = ("critical", "most important", "most fragile", "risk",
                      "single point", "spof", "hub", "bottleneck",
                      "what are the", "across everything", "globally",
                      "most central", "most affected")

SCOPE_TRIGGERS = ("for role", "as a", "for a", "department", "scope",
                   "authorized", "tier", "under the",
                   "jurisdiction", "for senior", "for junior",
                   "in production", "in staging")

PATTERN_TRIGGERS = ("similar to", "like before", "like the", "pattern",
                     "reminds me", "seen this", "precedent", "analogous")

CONNECTION_TRIGGERS = ("related to", "connected to", "has to do with",
                        "linked to", "connection between")


def _hit(query: str, triggers) -> bool:
    q = query.lower()
    return any(t in q for t in triggers)


class QueryRouter:
    def route(self, query_text: str,
              extracted_fragment: Optional[Dict[str, Any]] = None,
              ) -> List[str]:
        modes: List[str] = ["vector"]
        if _hit(query_text, CAUSAL_TRIGGERS):
            modes.append("graph")
        if _hit(query_text, TOPOLOGY_TRIGGERS):
            modes.append("topology")
        if _hit(query_text, SCOPE_TRIGGERS):
            modes.append("scope")
        if _hit(query_text, PATTERN_TRIGGERS):
            modes.append("metric")
        if _hit(query_text, CONNECTION_TRIGGERS):
            modes.append("connection")
        # Disconnected components in the fragment → connection test
        if extracted_fragment and self._has_disconnected(extracted_fragment):
            if "connection" not in modes:
                modes.append("connection")
        return modes

    @staticmethod
    def _has_disconnected(frag: Dict[str, Any]) -> bool:
        import networkx as nx
        g = nx.Graph()
        for n in frag.get("nodes", []):
            g.add_node(n["id"])
        for e in frag.get("edges", []):
            g.add_edge(e["src"], e["dst"])
        if g.number_of_nodes() < 2:
            return False
        try:
            return nx.number_connected_components(g) > 1
        except Exception:
            return False
