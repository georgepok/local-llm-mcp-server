"""Train LiquidARC graph reasoning engine.

Spec: GRAPH_REASONING_ENGINE_SPEC.md Phase 3 (lines 322-380).

Training loop exactly matching the spec:
  - Three param groups: embed lr=1e-3, dynamics lr=1e-4, head lr=1e-3
  - Multi-task losses: CE for root_cause & scoped_logic, BCE for connection,
    contrastive for analogy
  - Criticality scaffolding: D²/4τ→18 (weight 0.01), tau quality (weight 0.05)
  - MetricNet initialization: Option A (ARC d=768 post-transition checkpoint)

Task-homogeneous batches (each step uses one task) because analogy needs
two graphs per example and the other tasks have different output shapes.
Tasks are sampled proportionally to their dataset sizes.

Run:
  python3 scripts/train_graph_engine.py \
    --data_dir /workspace/liquid-arc/data/graph_engine \
    --arc_checkpoint /workspace/liquid-arc/output_30m/checkpoints/step_10000.pt \
    --output_dir /workspace/liquid-arc/output_graph_engine \
    --max_steps 10000 --batch_size 8
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from liquid_arc.config import LiquidARCConfig
from liquid_arc.context_pool import ContextPool
from liquid_arc.dynamics import ContinuousDynamics
from liquid_arc.solver import euler_solve
from liquid_arc.graph_embed import GraphNodeEmbedding
from liquid_arc.graph_features import compute_structural_features
from liquid_arc.graph_mask import build_edge_mask
from liquid_arc.graph_output_head import GraphOutputHead
from liquid_arc.sustained_criticality import (
    compute_criticality_loss, compute_tau_quality_loss,
)


# ─────────────────────────────────────────────────────────────────────
# JSONL reader and per-record feature extraction
# ─────────────────────────────────────────────────────────────────────

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def record_to_graph(record: Dict[str, Any],
                    nodes_key: str = 'nodes',
                    edges_key: str = 'edges') -> nx.DiGraph:
    g = nx.DiGraph()
    for n in record[nodes_key]:
        g.add_node(n['id'])
    for e in record[edges_key]:
        g.add_edge(e['src'], e['dst'])
    return g


def featurize_graph(record_nodes, record_edges, device,
                    active_scope=None):
    """Return (node_types[N], roles[N], struct[N,16], mask[N,N] bool, N).

    If active_scope is provided, the edge mask excludes edges whose 'scope'
    attribute does not match — used for scoped_logic queries where the mask
    must depend on the query context.
    """
    g = nx.DiGraph()
    ids = [n['id'] for n in record_nodes]
    for nid in ids:
        g.add_node(nid)
    for e in record_edges:
        attrs = {k: v for k, v in e.items() if k not in ('src', 'dst')}
        g.add_edge(e['src'], e['dst'], **attrs)
    order = ids
    N = len(order)
    types = torch.tensor([n['type'] for n in record_nodes], dtype=torch.long, device=device)
    roles = torch.tensor([n['role'] for n in record_nodes], dtype=torch.long, device=device)
    struct = compute_structural_features(g, node_order=order).to(device)
    mask = build_edge_mask(g, node_order=order, k_hops=3, as_bool=True,
                           active_scope=active_scope).to(device)
    return types, roles, struct, mask, N


# ─────────────────────────────────────────────────────────────────────
# Batch assembly with padding
# ─────────────────────────────────────────────────────────────────────

def collate_single_graph(records, device, per_record_scope=None):
    """Pad N variable graphs in a batch to max_N. Produce node_mask.

    Args:
        records: list of dicts with 'nodes' and 'edges'
        per_record_scope: optional list of active_scope ints per record; used
                          for scoped_logic so each query gets its own scope-
                          filtered edge mask. None → no scope filtering.
    """
    B = len(records)
    max_N = max(len(r['nodes']) for r in records)

    types = torch.zeros(B, max_N, dtype=torch.long, device=device)
    roles = torch.zeros(B, max_N, dtype=torch.long, device=device)
    struct = torch.zeros(B, max_N, 16, dtype=torch.float32, device=device)
    mask = torch.ones(B, max_N, max_N, dtype=torch.bool, device=device)
    node_mask = torch.zeros(B, max_N, dtype=torch.bool, device=device)

    for i, r in enumerate(records):
        scope_i = per_record_scope[i] if per_record_scope is not None else None
        t, ro, s, m, N = featurize_graph(r['nodes'], r['edges'], device,
                                         active_scope=scope_i)
        types[i, :N] = t
        roles[i, :N] = ro
        struct[i, :N] = s
        mask[i, :N, :N] = m
        node_mask[i, :N] = True

    return {
        'types': types, 'roles': roles, 'struct': struct,
        'mask': mask, 'node_mask': node_mask, 'max_N': max_N,
    }


# ─────────────────────────────────────────────────────────────────────
# Checkpoint loader (ARC Option A init)
# ─────────────────────────────────────────────────────────────────────

def load_arc_init(ckpt_path: str, dynamics: nn.Module, ctx_pool: nn.Module):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"  ARC checkpoint not provided/missing — starting from scratch")
        return 0
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)

    dyn_sd = {}
    ctx_sd = {}
    for k, v in sd.items():
        if k.startswith('dynamics._orig_mod.'):
            dyn_sd[k[len('dynamics._orig_mod.'):]] = v
        elif k.startswith('dynamics.'):
            dyn_sd[k[len('dynamics.'):]] = v
        elif k.startswith('context_pool.'):
            ctx_sd[k[len('context_pool.'):]] = v

    dyn_missing, _ = dynamics.load_state_dict(dyn_sd, strict=False)
    ctx_missing, _ = ctx_pool.load_state_dict(ctx_sd, strict=False)
    print(f"  ARC init: dynamics {len(dyn_sd)} loaded ({len(dyn_missing)} missing), "
          f"context_pool {len(ctx_sd)} loaded ({len(ctx_missing)} missing)")
    return len(dyn_sd) + len(ctx_sd)


# ─────────────────────────────────────────────────────────────────────
# Single ODE forward pass (returns h_out, g, tau for criticality losses)
# ─────────────────────────────────────────────────────────────────────

def forward_graph_batch(emb, ctx_pool, dynamics, batch, n_steps=16):
    """Run embedding → context → ODE on a batch, with per-example edge masks.

    Passes a [B, N, N] mask directly into ContinuousDynamics (requires the
    dynamics.py fix that accepts 3-D masks). Fully batched — 5-8× faster than
    the per-example loop.
    """
    h0 = emb(batch['types'], batch['roles'], batch['struct'])
    context = ctx_pool(h0, batch['node_mask'])
    B, max_N, d = h0.shape

    # Per-example mask: combine graph mask with pad-block so padding positions
    # are isolated and can't route anywhere.
    graph_mask = batch['mask']                                      # [B, N, N]
    valid = batch['node_mask']                                      # [B, N]
    pad_block = (~valid.unsqueeze(1)) | (~valid.unsqueeze(2))       # [B, N, N]
    mask_3d = graph_mask | pad_block
    # Ensure self-attention allowed on real nodes
    eye = torch.eye(max_N, dtype=torch.bool, device=mask_3d.device).unsqueeze(0)
    mask_3d = mask_3d & ~eye

    dynamics.set_context(context, mask=mask_3d)
    dynamics.set_n_steps(n_steps)
    h_out = euler_solve(dynamics, h0, t_span=(0.0, 2.0), n_steps=n_steps)

    # Compute g and tau for criticality losses and analogy signature
    h_normed = dynamics.norm_geo(h_out)
    ctx_exp = context.unsqueeze(1).expand(B, max_N, d)
    mi = torch.cat([h_normed, ctx_exp], dim=-1)
    hidden = F.gelu(dynamics.metric_net_linear1(mi))
    g = F.softplus(dynamics.metric_net_linear2_diag(hidden))
    tau = dynamics.compute_tau(h_out)
    if tau.dim() == 2:
        tau = tau.unsqueeze(-1)
    return h_out, g, tau


# ─────────────────────────────────────────────────────────────────────
# Per-task loss
# ─────────────────────────────────────────────────────────────────────

def task_loss_root_cause(output_head, h_out, records, batch, device):
    B = h_out.shape[0]
    query_node = torch.tensor([r['query']['target'] for r in records],
                              dtype=torch.long, device=device)
    target_node = torch.tensor([r['target'] for r in records],
                               dtype=torch.long, device=device)
    logits = output_head.root_cause(h_out, query_node, batch['node_mask'])
    loss = F.cross_entropy(logits, target_node)
    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == target_node).float().mean().item()
    return loss, {'acc': acc}


def task_loss_connection(output_head, h_out, records, batch, device):
    B = h_out.shape[0]
    src = torch.tensor([r['query']['src'] for r in records],
                       dtype=torch.long, device=device)
    dst = torch.tensor([r['query']['dst'] for r in records],
                       dtype=torch.long, device=device)
    targets = torch.tensor([1.0 if r['target'] else 0.0 for r in records],
                           dtype=torch.float32, device=device)
    logits = output_head.connection(h_out, src, dst)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == targets).float().mean().item()
    return loss, {'acc': acc}


def task_loss_implication(output_head, h_out, records, batch, device):
    B = h_out.shape[0]
    scope = torch.tensor([r['query']['context_scope'] for r in records],
                         dtype=torch.long, device=device)
    premise = torch.tensor([r['query']['premise'] for r in records],
                           dtype=torch.long, device=device)
    concl = torch.tensor([r['query']['conclusion'] for r in records],
                         dtype=torch.long, device=device)
    targets = torch.tensor([1 if r['target'] else 0 for r in records],
                           dtype=torch.long, device=device)
    logits = output_head.implication(h_out, scope, premise, concl,
                                     batch['node_mask'])
    loss = F.cross_entropy(logits, targets)
    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == targets).float().mean().item()
    return loss, {'acc': acc}


def task_loss_analogy(output_head, sig_a, sig_b, records, device):
    targets = torch.tensor([1.0 if r['target'] else 0.0 for r in records],
                           dtype=torch.float32, device=device)
    # Cosine similarity between signatures
    sim = F.cosine_similarity(sig_a, sig_b, dim=-1)
    # Target: sim=1 when isomorphic, sim=-1 (or 0) when not
    desired = targets * 2.0 - 1.0   # {-1, 1}
    # Margin loss: for positive pairs want sim close to 1; for negative close to -1
    loss = F.mse_loss(sim, desired)
    with torch.no_grad():
        preds = (sim > 0.0).float()
        acc = (preds == targets).float().mean().item()
    return loss, {'acc': acc, 'mean_sim_pos': sim[targets > 0.5].mean().item() if (targets > 0.5).any() else 0,
                  'mean_sim_neg': sim[targets < 0.5].mean().item() if (targets < 0.5).any() else 0}


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='/workspace/liquid-arc/data/graph_engine')
    p.add_argument('--arc_checkpoint',
                   default='/workspace/liquid-arc/output_30m/checkpoints/step_10000.pt')
    p.add_argument('--output_dir', default='/workspace/liquid-arc/output_graph_engine')
    p.add_argument('--max_steps', type=int, default=10000)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--save_every', type=int, default=2000)
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--d_model', type=int, default=768)
    p.add_argument('--d_metric', type=int, default=192)
    p.add_argument('--d_ffn', type=int, default=1536)
    p.add_argument('--n_ode_steps', type=int, default=16)
    p.add_argument('--lr_embed', type=float, default=1e-3)
    p.add_argument('--lr_dynamics', type=float, default=5e-5)
    p.add_argument('--lr_head', type=float, default=1e-3)
    p.add_argument('--crit_weight', type=float, default=0.01)
    p.add_argument('--tau_weight', type=float, default=0.05)
    p.add_argument('--freeze_dynamics', action='store_true',
                   help='freeze MetricNet/TauNet/ContextPool/Embedding; train only output_head')
    p.add_argument('--task_weights', default='',
                   help='comma-separated per-task weights root_cause,connection,analogy,implication; '
                        'overrides the default uniform sampling')
    p.add_argument('--resume_checkpoint', default=None,
                   help='path to a prior graph-engine checkpoint to resume from')
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("LIQUIDARC GRAPH ENGINE TRAINING")
    print("=" * 70)
    print(f"  device={device}  max_steps={args.max_steps}  batch_size={args.batch_size}")
    print(f"  d_model={args.d_model}  d_metric={args.d_metric}  d_ffn={args.d_ffn}")
    print(f"  lr: embed={args.lr_embed} dyn={args.lr_dynamics} head={args.lr_head}")
    print(f"  criticality_weight={args.crit_weight}  tau_weight={args.tau_weight}")

    # Load datasets
    print("\nloading datasets...")
    task_pools = {
        'root_cause': read_jsonl(os.path.join(args.data_dir, 'causal_chains.jsonl')),
        'connection_check': read_jsonl(os.path.join(args.data_dir, 'parallel_chains.jsonl')),
        'analogy': read_jsonl(os.path.join(args.data_dir, 'analogy_pairs.jsonl')),
        'implication_check': read_jsonl(os.path.join(args.data_dir, 'scoped_logic.jsonl')),
    }
    for k, v in task_pools.items():
        print(f"  {k}: {len(v)} records")

    # Split: last 5% as eval
    eval_pools = {k: v[-max(1, len(v)//20):] for k, v in task_pools.items()}
    train_pools = {k: v[:-max(1, len(v)//20)] for k, v in task_pools.items()}
    print("  eval split: 5% per task")

    # Uniform task sampling by default; --task_weights overrides.
    task_names = list(train_pools.keys())
    if args.task_weights:
        weights = [float(w) for w in args.task_weights.split(',')]
        assert len(weights) == len(task_names), \
            f"Expected {len(task_names)} task weights, got {len(weights)}"
        total = sum(weights)
        task_probs = [w / total for w in weights]
        print(f"  task sampling (weighted): {dict(zip(task_names, task_probs))}")
    else:
        task_probs = [1.0 / len(task_names)] * len(task_names)
        print(f"  task sampling (uniform): {dict(zip(task_names, task_probs))}")

    # Build model
    config = LiquidARCConfig()
    config.d_model = args.d_model
    config.d_metric = args.d_metric
    config.d_ffn = args.d_ffn
    config.n_ode_steps = args.n_ode_steps

    emb = GraphNodeEmbedding(d_model=config.d_model).to(device)
    dynamics = ContinuousDynamics(config).to(device)
    ctx_pool = ContextPool(config).to(device)
    output_head = GraphOutputHead(d_model=config.d_model).to(device)

    print(f"\n  emb params:      {sum(p.numel() for p in emb.parameters()):,}")
    print(f"  dynamics params: {sum(p.numel() for p in dynamics.parameters()):,}")
    print(f"  ctx_pool params: {sum(p.numel() for p in ctx_pool.parameters()):,}")
    print(f"  head params:     {sum(p.numel() for p in output_head.parameters()):,}")

    # Resume from a prior graph-engine checkpoint (if provided) — handles
    # architecture changes with shape-mismatched head params skipped so the
    # affected sub-modules train from scratch while dynamics/ctx/emb resume.
    if args.resume_checkpoint and os.path.exists(args.resume_checkpoint):
        print(f"  resuming from {args.resume_checkpoint}")
        rc = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=False)
        emb.load_state_dict(rc['emb'], strict=False)
        dynamics.load_state_dict(rc['dynamics'], strict=False)
        ctx_pool.load_state_dict(rc['context_pool'], strict=False)
        head_ckpt = rc['output_head']
        head_sd = output_head.state_dict()
        filtered = {k: v for k, v in head_ckpt.items()
                    if k in head_sd and head_sd[k].shape == v.shape}
        skipped = [k for k in head_ckpt if k not in filtered]
        output_head.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"    output_head: skipped {len(skipped)} shape-mismatched "
                  f"params (will retrain): {skipped[:5]}")
    else:
        load_arc_init(args.arc_checkpoint, dynamics, ctx_pool)

    # Optionally freeze everything except output_head (used when refining the
    # head architecture on a pre-trained dynamics/ctx/emb stack).
    if args.freeze_dynamics:
        print("  freezing emb/dynamics/ctx_pool — only output_head trains")
        for module in (emb, dynamics, ctx_pool):
            for p in module.parameters():
                p.requires_grad_(False)
        optimizer = torch.optim.Adam(output_head.parameters(), lr=args.lr_head)
    else:
        # Three param groups per spec
        optimizer = torch.optim.Adam([
            {'params': list(emb.parameters()) + list(ctx_pool.parameters()),
             'lr': args.lr_embed},
            {'params': dynamics.parameters(), 'lr': args.lr_dynamics},
            {'params': output_head.parameters(), 'lr': args.lr_head},
        ])

    if args.freeze_dynamics:
        emb.eval()
        dynamics.eval()
        ctx_pool.eval()
    else:
        emb.train()
        dynamics.train()
        ctx_pool.train()
    output_head.train()

    log_path = os.path.join(args.output_dir, 'train.log')
    log_f = open(log_path, 'w')

    t0 = time.time()
    running = {t: {'loss': [], 'acc': [], 'crit': [], 'tau': [], 'cv': []}
               for t in task_names}

    for step in range(1, args.max_steps + 1):
        # Sample task
        task = random.choices(task_names, weights=task_probs, k=1)[0]

        # Sample batch
        pool = train_pools[task]
        batch_recs = random.sample(pool, min(args.batch_size, len(pool)))

        if task == 'analogy':
            # Two graphs per record
            records_a = [{'nodes': r['graph_a']['nodes'],
                          'edges': r['graph_a']['edges']} for r in batch_recs]
            records_b = [{'nodes': r['graph_b']['nodes'],
                          'edges': r['graph_b']['edges']} for r in batch_recs]
            batch_a = collate_single_graph(records_a, device)
            batch_b = collate_single_graph(records_b, device)
            h_out_a, g_a, tau_a = forward_graph_batch(
                emb, ctx_pool, dynamics, batch_a, n_steps=args.n_ode_steps)
            h_out_b, g_b, tau_b = forward_graph_batch(
                emb, ctx_pool, dynamics, batch_b, n_steps=args.n_ode_steps)
            sig_a = output_head.signature(h_out_a, g_a, tau_a,
                                          batch_a['node_mask'],
                                          struct_features=batch_a['struct'])
            sig_b = output_head.signature(h_out_b, g_b, tau_b,
                                          batch_b['node_mask'],
                                          struct_features=batch_b['struct'])
            loss, metrics = task_loss_analogy(
                output_head, sig_a, sig_b, batch_recs, device)
            # Criticality signal averaged across both passes
            crit_loss_a, _ = compute_criticality_loss(
                h_out_a, g_a, tau_a, dynamics.t_diffusion, target_ratio=18.0)
            crit_loss_b, _ = compute_criticality_loss(
                h_out_b, g_b, tau_b, dynamics.t_diffusion, target_ratio=18.0)
            crit_loss = 0.5 * (crit_loss_a + crit_loss_b)
            tau_loss = 0.5 * (compute_tau_quality_loss(tau_a) +
                              compute_tau_quality_loss(tau_b))
            cv_val = ((g_a.std() / (g_a.mean() + 1e-8)).item() +
                      (g_b.std() / (g_b.mean() + 1e-8)).item()) / 2
        else:
            # For implication_check, pass per-record active_scope so the edge
            # mask filters out edges gated by a different scope.
            per_scope = None
            if task == 'implication_check':
                per_scope = [r['query']['context_scope'] for r in batch_recs]
            batch = collate_single_graph(batch_recs, device,
                                         per_record_scope=per_scope)
            h_out, g, tau = forward_graph_batch(
                emb, ctx_pool, dynamics, batch, n_steps=args.n_ode_steps)
            if task == 'root_cause':
                loss, metrics = task_loss_root_cause(
                    output_head, h_out, batch_recs, batch, device)
            elif task == 'connection_check':
                loss, metrics = task_loss_connection(
                    output_head, h_out, batch_recs, batch, device)
            elif task == 'implication_check':
                loss, metrics = task_loss_implication(
                    output_head, h_out, batch_recs, batch, device)
            crit_loss, _ = compute_criticality_loss(
                h_out, g, tau, dynamics.t_diffusion, target_ratio=18.0)
            tau_loss = compute_tau_quality_loss(tau)
            cv_val = (g.std() / (g.mean() + 1e-8)).item()

        # Accuracy-aware task loss weight: saturated tasks (acc → 1) contribute
        # less to total loss, so their gradient doesn't oscillate the dynamics.
        # Unsaturated tasks (acc near chance) get full weight.
        # window of last 100 steps per-task running accuracy
        recent_acc = running[task]['acc'][-100:] if running[task]['acc'] else [0.5]
        acc_estimate = sum(recent_acc) / max(1, len(recent_acc))
        task_weight = max(0.1, 1.0 - acc_estimate)   # floor at 0.1 so gradient never fully vanishes
        weighted_task_loss = task_weight * loss

        total_loss = weighted_task_loss + args.crit_weight * crit_loss + args.tau_weight * tau_loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  [step {step}] NaN/Inf — skipping")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(emb.parameters()) + list(dynamics.parameters())
            + list(ctx_pool.parameters()) + list(output_head.parameters()),
            max_norm=1.0)
        optimizer.step()

        running[task]['loss'].append(loss.item())
        running[task]['acc'].append(metrics.get('acc', 0.0))
        running[task]['crit'].append(crit_loss.item())
        running[task]['tau'].append(tau_loss.item())
        running[task]['cv'].append(cv_val)

        if step % args.log_every == 0:
            # Per-task rolling averages
            parts = [f"step={step}/{args.max_steps}"]
            for t in task_names:
                rl = running[t]['loss'][-args.log_every:]
                ra = running[t]['acc'][-args.log_every:]
                if rl:
                    parts.append(f"{t[:4]}:L={sum(rl)/len(rl):.3f},A={sum(ra)/len(ra):.2f}")
            # Last-step criticality signals
            parts.append(f"CV={cv_val:.2f}")
            parts.append(f"crit={crit_loss.item():.2f}")
            parts.append(f"tau={tau_loss.item():.2f}")
            parts.append(f"t={time.time()-t0:.0f}s")
            msg = " | ".join(parts)
            print(msg)
            log_f.write(msg + '\n')
            log_f.flush()

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_path = os.path.join(args.output_dir, 'checkpoints', f'step_{step}.pt')
            torch.save({
                'step': step,
                'emb': emb.state_dict(),
                'dynamics': dynamics.state_dict(),
                'context_pool': ctx_pool.state_dict(),
                'output_head': output_head.state_dict(),
                'config': vars(config),
            }, ckpt_path)
            print(f"  >> saved {ckpt_path}")

    log_f.close()
    # Final artifact
    final_path = os.path.join(args.output_dir, 'checkpoints', 'final.pt')
    torch.save({
        'step': args.max_steps,
        'emb': emb.state_dict(),
        'dynamics': dynamics.state_dict(),
        'context_pool': ctx_pool.state_dict(),
        'output_head': output_head.state_dict(),
        'config': vars(config),
    }, final_path)
    print(f"\ntraining complete. final → {final_path}")


if __name__ == '__main__':
    main()
