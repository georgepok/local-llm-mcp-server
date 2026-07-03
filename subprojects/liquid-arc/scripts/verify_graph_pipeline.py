"""Verify Phase 1 plumbing: graph encoding + mask + existing ContinuousDynamics.

End-to-end smoke test (no training, random init):
    build graph → GraphNodeEmbedding → build_edge_mask →
    ContinuousDynamics (via set_context with mask) → euler_solve →
    check output shape and finiteness.

If this runs clean, Phase 1 infrastructure is wired correctly and we can
proceed to Experiment 1 (ARC checkpoint transfer test).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import torch

from liquid_arc.config import LiquidARCConfig
from liquid_arc.context_pool import ContextPool
from liquid_arc.dynamics import ContinuousDynamics
from liquid_arc.solver import euler_solve
from liquid_arc.graph_embed import GraphNodeEmbedding
from liquid_arc.graph_features import compute_structural_features
from liquid_arc.graph_mask import build_edge_mask


def build_test_graph():
    """Causal chain + branching: (bridge_closure→truck_reroute→landslide→
    road_blocked→food_shortage) plus a parallel side-path."""
    g = nx.DiGraph()
    nodes = [
        ('bridge_closure', {'type': 0, 'role': 0}),       # event, root
        ('truck_reroute',  {'type': 1, 'role': 1}),       # consequence, intermediate
        ('landslide',      {'type': 0, 'role': 1}),       # event, intermediate
        ('road_blocked',   {'type': 2, 'role': 1}),       # state, intermediate
        ('food_shortage',  {'type': 1, 'role': 2}),       # consequence, terminal
    ]
    for n, attrs in nodes:
        g.add_node(n, **attrs)
    g.add_edges_from([
        ('bridge_closure', 'truck_reroute'),
        ('truck_reroute',  'landslide'),
        ('landslide',      'road_blocked'),
        ('road_blocked',   'food_shortage'),
    ])
    return g, nodes


def main():
    print("=" * 70)
    print("PHASE 1 VERIFICATION — graph pipeline through ContinuousDynamics")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  device: {device}")

    g, node_meta = build_test_graph()
    order = [n for n, _ in node_meta]
    node_types = torch.tensor([[attr['type'] for _, attr in node_meta]],
                              dtype=torch.long, device=device)
    roles = torch.tensor([[attr['role'] for _, attr in node_meta]],
                         dtype=torch.long, device=device)
    struct = compute_structural_features(g, node_order=order).unsqueeze(0).to(device)
    print(f"  graph: {len(order)} nodes, {g.number_of_edges()} edges")
    print(f"  node_types shape: {node_types.shape}")
    print(f"  struct features shape: {struct.shape}")

    # Embedding
    config = LiquidARCConfig()
    # Keep d_model modest for smoke-test speed
    config.d_model = 256
    config.d_metric = 64
    emb = GraphNodeEmbedding(d_model=config.d_model,
                             n_node_types=32, n_edge_types=16, n_roles=8).to(device)
    h0 = emb(node_types, roles, struct)
    print(f"  h0 shape: {h0.shape}, finite: {torch.isfinite(h0).all().item()}")

    # Edge mask (bool form for dynamics)
    mask = build_edge_mask(g, node_order=order, k_hops=2, as_bool=True).to(device)
    print(f"  mask shape: {mask.shape}, blocked count: {int(mask.sum().item())}")

    # Dynamics + context
    dynamics = ContinuousDynamics(config).to(device).eval()
    ctx_pool = ContextPool(config).to(device).eval()
    with torch.no_grad():
        ones_mask = torch.ones(1, len(order), dtype=torch.bool, device=device)
        context = ctx_pool(h0, ones_mask)
        print(f"  context shape: {context.shape}")

        dynamics.set_context(context, mask=mask)
        dynamics.set_n_steps(16)
        h_out = euler_solve(dynamics, h0, t_span=(0.0, 2.0), n_steps=16)

    print(f"  h_out shape: {h_out.shape}")
    print(f"  h_out finite: {torch.isfinite(h_out).all().item()}")
    print(f"  h_out norm per node: {h_out[0].norm(dim=-1).tolist()}")

    # Sanity check: did the state change?
    delta = (h_out - h0).norm().item()
    print(f"  ‖h_out - h0‖ = {delta:.4f}  (nonzero → dynamics is active)")
    assert delta > 0, "ODE produced no change — dynamics not connected"
    print("\nPhase 1 pipeline wired correctly.")


if __name__ == '__main__':
    main()
