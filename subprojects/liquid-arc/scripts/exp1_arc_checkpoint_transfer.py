"""Experiment 1 — ARC checkpoint transfer to graph input.

Spec: GRAPH_REASONING_ENGINE_SPEC.md lines 474-484.

Load d=768 post-transition ARC checkpoint. Feed GRAPH node embeddings
(no retraining). Measure:
  - CV of learned metric g(h) over graph input
  - D²/4τ (criticality ratio)
  - within-type vs across-type D² (cluster structure)

Pass criteria (from spec line 528):
  - CV > 3.0 (vs ~7 on ARC input)
  - D²/4τ near 18
  - same-type nodes cluster (low within-type D²)
  - different-type nodes separate (high across-type D²)

If pass → ARC routing transfers directly to graph input, no retraining
needed. Proceed to Task A evaluation.
If fail → MetricNet needs graph-specific training (Phase 3).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import torch
import torch.nn.functional as F

from liquid_arc.config import LiquidARCConfig
from liquid_arc.context_pool import ContextPool
from liquid_arc.dynamics import ContinuousDynamics
from liquid_arc.solver import euler_solve
from liquid_arc.graph_embed import GraphNodeEmbedding
from liquid_arc.graph_features import compute_structural_features
from liquid_arc.graph_mask import build_edge_mask


# ─────────────────────────────────────────────────────────────────────
# Test graphs with known structure
# ─────────────────────────────────────────────────────────────────────

def graph_linear_chain():
    """Single 5-node linear causal chain (event → consequence → state →
    consequence → state). Three node types produce three clusters."""
    g = nx.DiGraph()
    nodes = [
        ('n0', 0, 0),  # type 0 (event), role 0 (root)
        ('n1', 1, 1),  # type 1 (consequence), role 1 (intermediate)
        ('n2', 2, 1),  # type 2 (state), role 1
        ('n3', 1, 1),  # type 1 (consequence)
        ('n4', 2, 2),  # type 2 (state), role 2 (terminal)
    ]
    for name, _, _ in nodes:
        g.add_node(name)
    g.add_edges_from([('n0', 'n1'), ('n1', 'n2'), ('n2', 'n3'), ('n3', 'n4')])
    return g, nodes, 'linear_chain'


def graph_parallel_chains():
    """Two disconnected 4-node chains, interleaved node order to make
    the within/across-cluster signal clear."""
    g = nx.DiGraph()
    nodes = [
        ('A0', 0, 0), ('B0', 0, 0),   # roots of chain A, chain B
        ('A1', 1, 1), ('B1', 1, 1),
        ('A2', 2, 1), ('B2', 2, 1),
        ('A3', 1, 2), ('B3', 1, 2),
    ]
    for name, _, _ in nodes:
        g.add_node(name)
    g.add_edges_from([('A0','A1'),('A1','A2'),('A2','A3'),
                      ('B0','B1'),('B1','B2'),('B2','B3')])
    return g, nodes, 'parallel_chains'


def graph_clustered_types():
    """10 nodes with 3 clear type clusters, edges connecting clusters:
    3 nodes of type 0, 4 of type 1, 3 of type 2. Edges only between
    consecutive type groups (0→1→2)."""
    g = nx.DiGraph()
    nodes = []
    # Type 0 cluster
    for i in range(3):
        nodes.append((f'T0_{i}', 0, 0))
    # Type 1 cluster
    for i in range(4):
        nodes.append((f'T1_{i}', 1, 1))
    # Type 2 cluster
    for i in range(3):
        nodes.append((f'T2_{i}', 2, 2))
    for name, _, _ in nodes:
        g.add_node(name)
    # Edges 0→1
    for i in range(3):
        for j in range(2):
            g.add_edge(f'T0_{i}', f'T1_{(i+j) % 4}')
    # Edges 1→2
    for i in range(4):
        for j in range(1):
            g.add_edge(f'T1_{i}', f'T2_{(i+j) % 3}')
    return g, nodes, 'clustered_types'


# ─────────────────────────────────────────────────────────────────────
# Checkpoint loading (strip _orig_mod prefix from compiled checkpoint)
# ─────────────────────────────────────────────────────────────────────

def load_arc_checkpoint(ckpt_path, config, device):
    print(f"  loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt['model']

    dyn_sd = {}
    ctx_sd = {}
    for k, v in sd.items():
        if k.startswith('dynamics._orig_mod.'):
            dyn_sd[k[len('dynamics._orig_mod.'):]] = v
        elif k.startswith('dynamics.'):
            dyn_sd[k[len('dynamics.'):]] = v
        elif k.startswith('context_pool.'):
            ctx_sd[k[len('context_pool.'):]] = v

    dynamics = ContinuousDynamics(config).to(device)
    ctx_pool = ContextPool(config).to(device)
    dyn_missing, dyn_unexp = dynamics.load_state_dict(dyn_sd, strict=False)
    ctx_missing, ctx_unexp = ctx_pool.load_state_dict(ctx_sd, strict=False)
    print(f"  dynamics: {len(dyn_sd)} loaded, "
          f"{len(dyn_missing)} missing, {len(dyn_unexp)} unexpected")
    print(f"  context_pool: {len(ctx_sd)} loaded, "
          f"{len(ctx_missing)} missing, {len(ctx_unexp)} unexpected")
    return dynamics.eval(), ctx_pool.eval()


# ─────────────────────────────────────────────────────────────────────
# Diagnostics — CV, D², D²/4τ, within/across-type D²
# ─────────────────────────────────────────────────────────────────────

def compute_metric_diagnostics(dynamics, h_input, context, mask,
                               node_type_ids: torch.Tensor):
    """Replicate the metric computation from ContinuousDynamics.forward and
    extract CV(g), D²(i,j), τ(h). Split D² into within-type / across-type.

    Args:
        dynamics:       loaded ContinuousDynamics
        h_input:        [B, N, d] hidden states to probe
        context:        [B, d] pooled context
        mask:           [N, N] bool (True = blocked) — unused here; diagnostics
                        are per-position quantities
        node_type_ids:  [N] long — integer type per node (for within/across split)

    Returns:
        dict with CV, D2_mean, D2_crit_ratio, within_type_D2, across_type_D2,
        tau_mean, tau_log_spread
    """
    B, N, d = h_input.shape
    assert B == 1, "single-graph diagnostic"
    device = h_input.device

    # Run the same metric computation ContinuousDynamics.forward does
    h_normed = dynamics.norm_geo(h_input)
    ctx_exp = context.unsqueeze(1).expand(B, N, d)
    mi = torch.cat([h_normed, ctx_exp], dim=-1)
    hidden = F.gelu(dynamics.metric_net_linear1(mi))
    g = F.softplus(dynamics.metric_net_linear2_diag(hidden))  # [B, N, d_or_metric]

    # CV of g (same statistic used during ARC training)
    cv = (g.std() / (g.mean().clamp(min=1e-8))).item()

    # Pairwise D² under the diagonal metric (average metric between endpoints)
    # D²(i,j) = Σ_k 0.5*(g_i + g_j)[k] * (h_i - h_j)[k]²
    h_i = h_normed.unsqueeze(2)  # [B, N, 1, d]
    h_j = h_normed.unsqueeze(1)  # [B, 1, N, d]
    g_i = g.unsqueeze(2)
    g_j = g.unsqueeze(1)
    g_avg = 0.5 * (g_i + g_j)
    diff = h_i - h_j
    # If metric_rank < d, g is [B,N,d_metric] < d. Only take first d_metric dims.
    d_metric = g.shape[-1]
    d_eff = min(d_metric, d)
    D2 = ((diff[..., :d_eff] ** 2) * g_avg[..., :d_eff]).sum(dim=-1)  # [B, N, N]
    D2 = D2[0]  # [N, N]

    # τ(h)
    tau = dynamics.compute_tau(h_input).squeeze(-1)  # [B, N] or [B]
    tau_mean = tau.mean().item()
    log_tau = (tau + 1e-8).log()
    tau_log_spread = log_tau.std().item()

    # D²/4τ criticality ratio (mean over non-diagonal pairs)
    offdiag = ~torch.eye(N, dtype=torch.bool, device=device)
    D2_mean = D2[offdiag].mean().item()
    D2_crit_ratio = D2_mean / (4.0 * tau_mean + 1e-8)

    # Within-type vs across-type D²
    type_ids = node_type_ids.to(device)
    same_type = (type_ids.unsqueeze(0) == type_ids.unsqueeze(1)) & offdiag
    diff_type = (type_ids.unsqueeze(0) != type_ids.unsqueeze(1)) & offdiag
    within = D2[same_type].mean().item() if same_type.any() else float('nan')
    across = D2[diff_type].mean().item() if diff_type.any() else float('nan')

    return {
        'cv_g': cv,
        'D2_mean': D2_mean,
        'D2_crit_ratio': D2_crit_ratio,
        'tau_mean': tau_mean,
        'tau_log_spread': tau_log_spread,
        'within_type_D2': within,
        'across_type_D2': across,
        'separation_ratio': (across / within) if within > 0 else float('inf'),
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',
                   default='/workspace/liquid-arc/output_30m/checkpoints/step_10000.pt')
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='/workspace/liquid-arc/exp1_arc_transfer.json')
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("EXPERIMENT 1 — ARC checkpoint transfer to graph input")
    print("=" * 70)
    print(f"  device: {device}")

    # Config must match the checkpoint (d=768, d_metric=192, d_ffn=1536, n_ode_steps=16)
    config = LiquidARCConfig()
    config.d_model = 768
    config.d_metric = 192
    config.d_ffn = 1536
    config.n_ode_steps = 16

    dynamics, ctx_pool = load_arc_checkpoint(args.checkpoint, config, device)
    # Freeze everything — we are measuring zero-shot transfer
    for p_ in dynamics.parameters():
        p_.requires_grad_(False)
    for p_ in ctx_pool.parameters():
        p_.requires_grad_(False)

    # Graph node embedding (random init — we're testing whether the ODE's
    # routing still structures random embeddings along the type axis)
    emb = GraphNodeEmbedding(d_model=config.d_model).to(device).eval()
    for p_ in emb.parameters():
        p_.requires_grad_(False)

    all_results = {}

    for graph_builder in (graph_linear_chain,
                          graph_parallel_chains,
                          graph_clustered_types):
        g, nodes, name = graph_builder()
        order = [nm for nm, _, _ in nodes]
        node_types = torch.tensor([[t for _, t, _ in nodes]],
                                  dtype=torch.long, device=device)
        roles = torch.tensor([[r for _, _, r in nodes]],
                             dtype=torch.long, device=device)
        struct = compute_structural_features(g, node_order=order).unsqueeze(0).to(device)

        h0 = emb(node_types, roles, struct)

        mask = build_edge_mask(g, node_order=order, k_hops=2, as_bool=True).to(device)

        with torch.no_grad():
            ones = torch.ones(1, len(order), dtype=torch.bool, device=device)
            context = ctx_pool(h0, ones)
            dynamics.set_context(context, mask=mask)
            dynamics.set_n_steps(16)
            # Diagnostics BEFORE the ODE (on embedding input)
            pre = compute_metric_diagnostics(
                dynamics, h0, context, mask, node_types[0])
            # Run ODE
            h_out = euler_solve(dynamics, h0, t_span=(0.0, 2.0), n_steps=16)
            # Diagnostics AFTER the ODE (on evolved state)
            post = compute_metric_diagnostics(
                dynamics, h_out, context, mask, node_types[0])

        print(f"\n--- graph: {name} ({len(order)} nodes, "
              f"{g.number_of_edges()} edges) ---")
        print(f"  pre-ODE:   CV={pre['cv_g']:>6.3f}  "
              f"D²/4τ={pre['D2_crit_ratio']:>7.2f}  "
              f"within-D²={pre['within_type_D2']:>8.3f}  "
              f"across-D²={pre['across_type_D2']:>8.3f}  "
              f"sep={pre['separation_ratio']:>5.2f}×  "
              f"τ={pre['tau_mean']:>5.3f}±{pre['tau_log_spread']:>4.2f}")
        print(f"  post-ODE:  CV={post['cv_g']:>6.3f}  "
              f"D²/4τ={post['D2_crit_ratio']:>7.2f}  "
              f"within-D²={post['within_type_D2']:>8.3f}  "
              f"across-D²={post['across_type_D2']:>8.3f}  "
              f"sep={post['separation_ratio']:>5.2f}×  "
              f"τ={post['tau_mean']:>5.3f}±{post['tau_log_spread']:>4.2f}")

        all_results[name] = {'pre': pre, 'post': post,
                             'n_nodes': len(order),
                             'n_edges': g.number_of_edges()}

    # Pass/fail assessment (spec lines 481-484, 528)
    print("\n" + "=" * 70)
    print("PASS CRITERIA (Experiment 1)")
    print("=" * 70)

    cvs_post = [r['post']['cv_g'] for r in all_results.values()]
    crit_post = [r['post']['D2_crit_ratio'] for r in all_results.values()]
    sep_post = [r['post']['separation_ratio'] for r in all_results.values()]

    mean_cv = sum(cvs_post) / len(cvs_post)
    mean_crit = sum(crit_post) / len(crit_post)
    # For separation ratio, we want across > within (ratio > 1)
    mean_sep = sum(sep_post) / len(sep_post)

    def pf(cond):
        return "PASS" if cond else "FAIL"
    criterion_cv = mean_cv > 3.0
    criterion_crit = abs(mean_crit - 18.0) < 18.0  # within 100% of target
    criterion_cluster = mean_sep > 1.05   # at least 5% separation
    print(f"  (1) CV > 3.0                  : {pf(criterion_cv)}  (mean CV = {mean_cv:.3f})")
    print(f"  (2) D²/4τ near 18             : {pf(criterion_crit)}  (mean = {mean_crit:.2f})")
    print(f"  (3) same-type clusters (sep>1): {pf(criterion_cluster)}  (mean sep = {mean_sep:.3f}×)")

    transfers = criterion_cv and criterion_cluster
    print(f"\n  OVERALL: ARC routing {'TRANSFERS' if transfers else 'DOES NOT TRANSFER'} to graph input")
    if transfers:
        print("    → Proceed to Task A evaluation without retraining")
    else:
        print("    → MetricNet needs graph-specific training (Phase 3)")

    summary = {
        'mean_cv_post': mean_cv,
        'mean_crit_post': mean_crit,
        'mean_sep_post': mean_sep,
        'criterion_cv_pass': criterion_cv,
        'criterion_crit_pass': criterion_crit,
        'criterion_cluster_pass': criterion_cluster,
        'transfers': transfers,
        'per_graph': all_results,
        'checkpoint': args.checkpoint,
    }
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  → saved {args.out}")


if __name__ == '__main__':
    main()
