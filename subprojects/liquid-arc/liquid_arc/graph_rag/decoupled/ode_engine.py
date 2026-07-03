"""SubgraphODEEngine — wraps GraphEngine, enforces the ≤200-node cap.

The only change vs direct GraphEngine use: input comes from
KnowledgeGraphDB.extract_subgraph() (already capped). No persistent
state is maintained between calls — each invocation is stateless.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from ...graph_engine_inference import GraphEngine


class SubgraphODEEngine:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.engine = GraphEngine(checkpoint_path, device=device,
                                  corrections_log=None)

    def compute_diagnostics(self, subgraph: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(self.engine.get_graph_diagnostics(json.dumps(subgraph)))

    def compute_signature(self, subgraph: Dict[str, Any]) -> Any:
        """Return the head's 64-d signature for the subgraph (as list)."""
        import torch
        nodes = subgraph["nodes"]
        edges = subgraph["edges"]
        from ...graph_engine_inference import _normalize_nodes, _normalize_edges
        norm_nodes = _normalize_nodes(nodes)
        norm_edges = _normalize_edges(edges)
        state = self.engine._forward_graph(norm_nodes, norm_edges)
        with torch.no_grad():
            sig = self.engine.head.signature(
                state["h_out"], state["g"], state["tau"],
                state["node_mask"], struct_features=state["struct"])
        return sig.squeeze(0).detach().cpu().float().tolist()

    def analyze(self, subgraph: Dict[str, Any],
                query: Dict[str, Any]) -> Dict[str, Any]:
        raw = self.engine.analyze_graph(json.dumps(subgraph), json.dumps(query))
        return json.loads(raw)
