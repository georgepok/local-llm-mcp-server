"""Per-node structural features for LiquidARC graph processing.

Spec: GRAPH_REASONING_ENGINE_SPEC.md lines 85-105.

These are input-INDEPENDENT topology features that the MetricNet uses
alongside the type/role embeddings to produce its routing pattern.

Dimensions of the 16-d feature vector per node:
    0: in_degree (normalized)
    1: out_degree (normalized)
    2: total_degree (normalized)
    3: closeness_centrality
    4: is_root (1.0 if in_degree == 0 else 0.0)
    5: is_leaf (1.0 if out_degree == 0 else 0.0)
    6: depth_from_root (normalized by longest shortest path length)
    7: depth_to_leaf (normalized by longest shortest path length)
    8: betweenness_centrality
    9: eigenvector_centrality (falls back to degree centrality on failure)
    10: pagerank
    11: clustering_coefficient (on underlying undirected view)
    12: avg_in_weight (mean weight of incoming edges, normalized)
    13: avg_out_weight (mean weight of outgoing edges, normalized)
    14: cycle_participation (1.0 if node is part of a simple cycle else 0.0)
    15: local_density (edges_in_2hop_neighborhood / possible_edges)
"""

from typing import Optional

import math
import networkx as nx
import torch


STRUCT_FEATURE_DIM = 16


def _norm(values: dict, fallback: float = 0.0) -> dict:
    """Normalize a dict of node→float to [0, 1] by its max; empty-safe."""
    if not values:
        return {}
    vmax = max(abs(v) for v in values.values())
    if vmax == 0:
        return {k: fallback for k in values}
    return {k: v / vmax for k, v in values.items()}


def _depth_from_root(g: nx.DiGraph) -> dict:
    """Distance from each node to the nearest root (in_degree==0) node.
    Unreachable → longest finite distance + 1, then normalized in the caller."""
    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    if not roots:
        # No root: use the node with minimum in_degree
        min_in = min(g.in_degree(n) for n in g.nodes)
        roots = [n for n in g.nodes if g.in_degree(n) == min_in]
    dist = {n: float('inf') for n in g.nodes}
    for r in roots:
        single = nx.single_source_shortest_path_length(g, r)
        for n, d in single.items():
            if d < dist[n]:
                dist[n] = d
    finite = [d for d in dist.values() if math.isfinite(d)]
    if not finite:
        return {n: 0.0 for n in g.nodes}
    max_finite = max(finite) if finite else 1.0
    return {n: (d if math.isfinite(d) else max_finite + 1) for n, d in dist.items()}


def _depth_to_leaf(g: nx.DiGraph) -> dict:
    """Distance from each node to the nearest leaf (out_degree==0) node."""
    leaves = [n for n in g.nodes if g.out_degree(n) == 0]
    if not leaves:
        min_out = min(g.out_degree(n) for n in g.nodes)
        leaves = [n for n in g.nodes if g.out_degree(n) == min_out]
    g_rev = g.reverse(copy=False)
    dist = {n: float('inf') for n in g.nodes}
    for leaf in leaves:
        single = nx.single_source_shortest_path_length(g_rev, leaf)
        for n, d in single.items():
            if d < dist[n]:
                dist[n] = d
    finite = [d for d in dist.values() if math.isfinite(d)]
    if not finite:
        return {n: 0.0 for n in g.nodes}
    max_finite = max(finite) if finite else 1.0
    return {n: (d if math.isfinite(d) else max_finite + 1) for n, d in dist.items()}


def _safe_closeness(g: nx.DiGraph) -> dict:
    try:
        return nx.closeness_centrality(g)
    except Exception:
        return {n: 0.0 for n in g.nodes}


def _safe_betweenness(g: nx.DiGraph) -> dict:
    try:
        return nx.betweenness_centrality(g)
    except Exception:
        return {n: 0.0 for n in g.nodes}


def _safe_eigenvector(g: nx.DiGraph) -> dict:
    try:
        return nx.eigenvector_centrality_numpy(g)
    except Exception:
        # Fallback: degree centrality
        try:
            return nx.degree_centrality(g)
        except Exception:
            return {n: 0.0 for n in g.nodes}


def _safe_pagerank(g: nx.DiGraph) -> dict:
    try:
        return nx.pagerank(g, alpha=0.85, max_iter=200)
    except Exception:
        return {n: 1.0 / max(1, g.number_of_nodes()) for n in g.nodes}


def _clustering_coeff(g: nx.DiGraph) -> dict:
    u = g.to_undirected()
    try:
        return nx.clustering(u)
    except Exception:
        return {n: 0.0 for n in g.nodes}


def _avg_edge_weight(g: nx.DiGraph, direction: str) -> dict:
    out = {}
    for n in g.nodes:
        edges = g.in_edges(n, data=True) if direction == 'in' else g.out_edges(n, data=True)
        weights = [data.get('weight', 1.0) for _, _, data in edges] if direction == 'in' else [
            data.get('weight', 1.0) for _, _, data in edges]
        out[n] = float(sum(weights) / max(1, len(weights)))
    return out


def _cycle_participation(g: nx.DiGraph) -> dict:
    """1.0 if node participates in any simple cycle, else 0.0."""
    participants = set()
    try:
        for cyc in nx.simple_cycles(g):
            for n in cyc:
                participants.add(n)
    except Exception:
        pass
    return {n: (1.0 if n in participants else 0.0) for n in g.nodes}


def _local_density(g: nx.DiGraph) -> dict:
    """Ratio of edges within 2-hop neighborhood to max possible (undirected view)."""
    u = g.to_undirected()
    out = {}
    for n in u.nodes:
        neigh = set(nx.single_source_shortest_path_length(u, n, cutoff=2).keys())
        neigh.discard(n)
        k = len(neigh)
        if k < 2:
            out[n] = 0.0
            continue
        sub = u.subgraph(neigh | {n})
        m = sub.number_of_edges()
        possible = k * (k + 1) / 2
        out[n] = float(m) / float(possible) if possible > 0 else 0.0
    return out


def compute_structural_features(graph: nx.DiGraph,
                                 node_order: Optional[list] = None
                                 ) -> torch.Tensor:
    """Extract a [N, 16] tensor of topology features per node.

    Args:
        graph:      networkx.DiGraph with optional 'weight' attribute on edges
        node_order: optional explicit ordering of nodes; defaults to graph.nodes order

    Returns:
        feats: [N, 16] float tensor. Features normalized to roughly [0, 1].
    """
    nodes = list(node_order) if node_order is not None else list(graph.nodes)
    N = len(nodes)
    if N == 0:
        return torch.zeros(0, STRUCT_FEATURE_DIM, dtype=torch.float32)

    in_deg = _norm({n: graph.in_degree(n) for n in nodes})
    out_deg = _norm({n: graph.out_degree(n) for n in nodes})
    tot_deg = _norm({n: graph.degree(n) for n in nodes})
    closeness = _safe_closeness(graph)
    betweenness = _safe_betweenness(graph)
    eigen = _safe_eigenvector(graph)
    pr = _safe_pagerank(graph)
    clustering = _clustering_coeff(graph)
    depth_from = _norm(_depth_from_root(graph))
    depth_to = _norm(_depth_to_leaf(graph))
    avg_in_w = _norm(_avg_edge_weight(graph, 'in'))
    avg_out_w = _norm(_avg_edge_weight(graph, 'out'))
    cycle_part = _cycle_participation(graph)
    density = _local_density(graph)

    feats = torch.zeros(N, STRUCT_FEATURE_DIM, dtype=torch.float32)
    for i, n in enumerate(nodes):
        feats[i, 0] = in_deg.get(n, 0.0)
        feats[i, 1] = out_deg.get(n, 0.0)
        feats[i, 2] = tot_deg.get(n, 0.0)
        feats[i, 3] = closeness.get(n, 0.0)
        feats[i, 4] = 1.0 if graph.in_degree(n) == 0 else 0.0
        feats[i, 5] = 1.0 if graph.out_degree(n) == 0 else 0.0
        feats[i, 6] = depth_from.get(n, 0.0)
        feats[i, 7] = depth_to.get(n, 0.0)
        feats[i, 8] = betweenness.get(n, 0.0)
        feats[i, 9] = eigen.get(n, 0.0)
        feats[i, 10] = pr.get(n, 0.0)
        feats[i, 11] = clustering.get(n, 0.0)
        feats[i, 12] = avg_in_w.get(n, 0.0)
        feats[i, 13] = avg_out_w.get(n, 0.0)
        feats[i, 14] = cycle_part.get(n, 0.0)
        feats[i, 15] = density.get(n, 0.0)
    return feats


if __name__ == "__main__":
    print("Testing compute_structural_features...")
    g = nx.DiGraph()
    g.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('B', 'E'), ('E', 'D')],
                     weight=1.0)
    feats = compute_structural_features(g, node_order=['A', 'B', 'C', 'D', 'E'])
    assert feats.shape == (5, 16), f"expected (5,16), got {feats.shape}"
    # Root A should have is_root=1, is_leaf=0, depth_from_root=0
    assert feats[0, 4].item() == 1.0, "A should be root"
    assert feats[0, 5].item() == 0.0, "A should not be leaf"
    assert feats[0, 6].item() == 0.0, "A depth_from_root should be 0"
    # Leaf D should have is_leaf=1
    assert feats[3, 5].item() == 1.0, "D should be leaf"
    print(f"  shape: {feats.shape}")
    print(f"  feature ranges: min={feats.min():.3f} max={feats.max():.3f}")
    print(f"  A (root): {feats[0].tolist()}")
    print(f"  D (leaf): {feats[3].tolist()}")
    print("compute_structural_features OK")
