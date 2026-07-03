"""Edge mask builder for LiquidARC graph processing.

Spec: GRAPH_REASONING_ENGINE_SPEC.md lines 111-129.

The heat kernel routes between ALL positions by default. Graph edges constrain
routing: connected nodes get mask = 0 (route allowed), disconnected nodes get
mask = -inf (route blocked). With k-hop relaxation we allow routing within k
hops rather than direct edges only — this lets information propagate through
chains within a single ODE step.
"""

from typing import Optional

import networkx as nx
import torch


def build_edge_mask(graph: nx.DiGraph,
                    n_nodes: Optional[int] = None,
                    node_order: Optional[list] = None,
                    k_hops: int = 2,
                    undirected: bool = True,
                    as_bool: bool = False,
                    active_scope: Optional[int] = None) -> torch.Tensor:
    """Build an attention mask from graph edges.

    Args:
        graph:      networkx.DiGraph whose nodes are the positions
        n_nodes:    optional explicit size; if None, uses len(graph)
        node_order: optional explicit node ordering (required to align with
                    embedding tensor). If None, uses graph.nodes order.
        k_hops:     number of edge hops to allow (spec recommends 2).
        undirected: if True, treat adjacency as symmetric for routing purposes
                    (the heat kernel is bidirectional in nature).
        as_bool:    if True, return a bool mask where True means BLOCKED
                    (the form ContinuousDynamics.set_mask expects via
                    torch.masked_fill_). Default False (spec form).

    Returns:
        mask: [N, N] tensor.
              - default (spec form, as_bool=False):
                  float tensor. mask[i, j] = 0 where route i→j is allowed,
                  -inf where blocked. Diagonal is always 0.
              - as_bool=True:
                  bool tensor. mask[i, j] = True where route i→j is BLOCKED,
                  False where allowed. Diagonal is always False.
    """
    nodes = list(node_order) if node_order is not None else list(graph.nodes)
    N = n_nodes if n_nodes is not None else len(nodes)

    # Build adjacency matrix in the requested node order.
    # If active_scope is provided, edges with a 'scope' attribute that does
    # not match active_scope are treated as absent (scope-gated).
    idx = {n: i for i, n in enumerate(nodes)}
    adj = torch.zeros(N, N, dtype=torch.float32)
    for u, v, data in graph.edges(data=True):
        if u not in idx or v not in idx:
            continue
        if active_scope is not None:
            edge_scope = data.get('scope', None) if isinstance(data, dict) else None
            if edge_scope is not None and edge_scope != active_scope:
                continue    # gated out by non-matching scope
        adj[idx[u], idx[v]] = 1.0
        if undirected:
            adj[idx[v], idx[u]] = 1.0

    # k-hop adjacency: iterated boolean power of adj, clipped to {0, 1}.
    # adj_k = (adj + adj@adj + adj@adj@adj + ... up to k).clip(0, 1)
    reach = adj.clone()
    power = adj.clone()
    for _ in range(max(1, k_hops) - 1):
        power = (power @ adj).clamp(0.0, 1.0)
        reach = (reach + power).clamp(0.0, 1.0)

    if as_bool:
        # ContinuousDynamics.set_mask form: True = blocked (for masked_fill_)
        blocked = (reach == 0)
        blocked.fill_diagonal_(False)
        return blocked
    # Spec form: float with -inf on blocked, 0 on allowed
    mask = torch.zeros(N, N, dtype=torch.float32)
    mask[reach == 0] = float('-inf')
    mask.fill_diagonal_(0.0)
    return mask


if __name__ == "__main__":
    print("Testing build_edge_mask...")

    # 1) Linear chain A→B→C→D→E
    g = nx.DiGraph()
    g.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')])
    order = ['A', 'B', 'C', 'D', 'E']
    m = build_edge_mask(g, node_order=order, k_hops=2)
    assert m.shape == (5, 5)
    # A can reach B (1-hop) and C (2-hops), but not D or E with k=2
    assert torch.isfinite(m[0, 1]), "A should reach B"
    assert torch.isfinite(m[0, 2]), "A should reach C (2-hop)"
    assert torch.isinf(m[0, 3]), "A should NOT reach D (3-hop)"
    assert torch.isinf(m[0, 4]), "A should NOT reach E (4-hop)"
    # Self-attention always allowed
    assert torch.isfinite(m[2, 2]), "diagonal should be 0"

    # 2) Two disconnected components (Task B scenario)
    g2 = nx.DiGraph()
    g2.add_edges_from([('A', 'B'), ('C', 'D')])
    order2 = ['A', 'B', 'C', 'D']
    m2 = build_edge_mask(g2, node_order=order2, k_hops=3)
    assert torch.isfinite(m2[0, 1]), "A-B connected within component 1"
    assert torch.isinf(m2[0, 2]), "A and C in different components"
    assert torch.isinf(m2[0, 3]), "A and D in different components"
    assert torch.isfinite(m2[2, 3]), "C-D connected within component 2"

    # 3) Larger graph with branching
    g3 = nx.DiGraph()
    g3.add_edges_from([
        ('F', 'J'), ('F', 'G'), ('J', 'E'), ('E', 'A'),
        ('G', 'H'), ('H', 'A'),
    ])
    order3 = ['F', 'J', 'G', 'E', 'H', 'A']
    m3 = build_edge_mask(g3, node_order=order3, k_hops=2)
    # F reaches J, G directly (1-hop), E, H via 2-hop
    assert torch.isfinite(m3[0, 1])  # F-J
    assert torch.isfinite(m3[0, 2])  # F-G
    assert torch.isfinite(m3[0, 3])  # F-E (2-hop via J)
    assert torch.isfinite(m3[0, 4])  # F-H (2-hop via G)
    assert torch.isinf(m3[0, 5])     # F-A (3-hop, blocked at k=2)

    print(f"  chain mask [5,5] passes 1-hop and 2-hop reachability")
    print(f"  disconnected components are isolated")
    print(f"  branching graph respects k-hop boundary")
    print("build_edge_mask OK")
