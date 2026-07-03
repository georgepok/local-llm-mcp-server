"""GraphEngine — inference wrapper around the trained LiquidARC graph engine.

Loads a checkpoint produced by scripts/train_graph_engine.py and exposes
three query operations (analyze_graph, compare_graphs, get_diagnostics) as
methods returning JSON-serializable dicts. Used by the MCP server wrapper
to serve graph-reasoning queries to the LLM.

Spec: GRAPH_REASONING_ENGINE_SPEC.md Phase 4 (lines 384-430).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import networkx as nx
import torch
import torch.nn.functional as F

from .config import LiquidARCConfig
from .context_pool import ContextPool
from .dynamics import ContinuousDynamics
from .graph_embed import GraphNodeEmbedding
from .graph_features import compute_structural_features
from .graph_mask import build_edge_mask
from .graph_output_head import GraphOutputHead
from .solver import euler_solve


# ─────────────────────────────────────────────────────────────────────
# String ↔ integer vocabularies for MCP-friendly input
#
# MCP callers (including LLMs) naturally pass node types and roles as
# string names ("event", "root"). Internally the ODE consumes integer
# slots (indices into the TypeEmbed / RoleEmbed lookup tables). These
# tables map between the two.
# ─────────────────────────────────────────────────────────────────────

_NODE_TYPE_VOCAB: Dict[str, int] = {
    'event': 0, 'consequence': 1, 'state': 2, 'cause': 3,
    'role': 4, 'credential': 5, 'requirement': 6, 'prerequisite': 7,
    'trigger': 8, 'step': 9, 'outcome': 10,
    # Generic fallbacks
    'entity': 11, 'node': 12, 'concept': 13, 'attribute': 14, 'other': 15,
}

_ROLE_VOCAB: Dict[str, int] = {
    'root': 0, 'intermediate': 1, 'terminal': 2, 'scope': 3,
    'query': 4, 'other': 5, 'source': 6, 'target': 7,
}

_EDGE_TYPE_VOCAB: Dict[str, int] = {
    'causes': 0, 'requires': 1, 'precedes': 2, 'enables': 3,
    'depends_on': 4, 'related_to': 5, 'blocks': 6, 'is_a': 7,
}


def _coerce_int(value: Any, vocab: Dict[str, int], default: int = 0) -> int:
    """Accept either an integer index or a string name; return the int slot."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in vocab:
            return vocab[key]
        # Hash-based fallback so unknown names still produce a stable int slot
        # within [0, 31) (fits TypeEmbed n_node_types=32).
        return (hash(key) & 0x7fffffff) % 32
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _normalize_nodes(nodes: List[Dict]) -> List[Dict]:
    out = []
    for n in nodes:
        out.append({
            'id': n['id'],
            'type': _coerce_int(n.get('type', 0), _NODE_TYPE_VOCAB, default=0),
            'role': _coerce_int(n.get('role', 0), _ROLE_VOCAB, default=0),
        })
    return out


def _normalize_edges(edges: List[Dict]) -> List[Dict]:
    out = []
    for e in edges:
        ed = {
            'src': e['src'],
            'dst': e['dst'],
            'type': _coerce_int(e.get('type', 0), _EDGE_TYPE_VOCAB, default=0),
        }
        if 'scope' in e and e['scope'] is not None:
            # Scope stays as-is (it's a node id, usually string)
            ed['scope'] = e['scope']
        out.append(ed)
    return out


def _records_to_graph(nodes: List[Dict], edges: List[Dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n['id'])
    for e in edges:
        attrs = {k: v for k, v in e.items() if k not in ('src', 'dst')}
        g.add_edge(e['src'], e['dst'], **attrs)
    return g


class GraphEngine:
    """Inference + online-correction wrapper around the trained graph engine.

    Reliable tasks (from training): root_cause, implication_check.
    Partial tasks: connection_check, analogy.

    Online learning: callers can submit corrections via `correct_answer(...)`.
    The output head receives a gradient step per correction; dynamics and
    ctx_pool stay frozen (protects the learned geometry). A rolling replay
    buffer of recent corrections is available for session-scale retention.
    Corrections are appended to a JSONL log on disk so the model can be
    reloaded with all prior corrections after restart.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cuda',
                 corrections_log: str = None,
                 correction_lr: float = 1e-5,
                 replay_buffer_size: int = 128,
                 max_corrections_per_example: int = 3,
                 replay_per_correction: int = 8,
                 health_check_variance_threshold: float = 0.02):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        cfg_dict = ckpt.get('config', {})
        self.config = LiquidARCConfig()
        for k in ('d_model', 'd_metric', 'd_ffn', 'n_ode_steps'):
            if k in cfg_dict:
                setattr(self.config, k, cfg_dict[k])
        self.n_steps = getattr(self.config, 'n_ode_steps', 16)

        self.emb = GraphNodeEmbedding(d_model=self.config.d_model).to(self.device)
        self.dynamics = ContinuousDynamics(self.config).to(self.device)
        self.ctx_pool = ContextPool(self.config).to(self.device)
        self.head = GraphOutputHead(d_model=self.config.d_model).to(self.device)

        self.emb.load_state_dict(ckpt['emb'])
        self.dynamics.load_state_dict(ckpt['dynamics'])
        self.ctx_pool.load_state_dict(ckpt['context_pool'])
        # Head architecture may have evolved (e.g. impl_head gained premise
        # input → shape changed). Filter out any shape-mismatched keys; the
        # affected head sub-module starts fresh and learns from corrections.
        head_ckpt = ckpt['output_head']
        head_sd = self.head.state_dict()
        filtered = {}
        skipped = []
        for k, v in head_ckpt.items():
            if k in head_sd and head_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        self.head.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"[GraphEngine] skipped {len(skipped)} shape-mismatched head "
                  f"params (will be trained via corrections): {skipped[:5]}...",
                  flush=True)

        # FREEZE embedding, dynamics, ctx_pool — protect learned geometry.
        # Only the HEAD trains on corrections.
        for module in (self.emb, self.dynamics, self.ctx_pool):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)
        # Head is kept in eval mode by default but with trainable params.
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad_(True)

        # Optimizer for online corrections
        self.correction_lr = correction_lr
        self.optimizer = torch.optim.Adam(
            self.head.parameters(), lr=correction_lr)

        # Replay buffer + persistent log
        self.replay_buffer: list = []
        self.replay_buffer_size = replay_buffer_size
        self.n_corrections_applied = 0
        self.corrections_log = corrections_log

        # Safeguards
        self.max_corrections_per_example = max_corrections_per_example
        self.replay_per_correction = replay_per_correction
        self.health_check_variance_threshold = health_check_variance_threshold
        self.example_correction_count: Dict[str, int] = {}

        # Pristine snapshot for rollback if the head collapses
        import copy
        self._head_pristine_state = copy.deepcopy(self.head.state_dict())

        if corrections_log and os.path.exists(corrections_log):
            self._replay_from_log(corrections_log)

    # ─────────────────────────────────────────────────────────────
    # Forward pipeline: parse JSON graph → embed → ODE → features
    # ─────────────────────────────────────────────────────────────

    def _forward_graph(self, nodes: List[Dict], edges: List[Dict],
                       active_scope: Optional[int] = None):
        g = _records_to_graph(nodes, edges)
        order = [n['id'] for n in nodes]
        N = len(order)
        node_types = torch.tensor([[n.get('type', 0) for n in nodes]],
                                  dtype=torch.long, device=self.device)
        roles = torch.tensor([[n.get('role', 0) for n in nodes]],
                             dtype=torch.long, device=self.device)
        struct = compute_structural_features(g, node_order=order).unsqueeze(0).to(self.device)
        mask = build_edge_mask(g, node_order=order, k_hops=3,
                               as_bool=True, active_scope=active_scope).to(self.device)
        mask_3d = mask.unsqueeze(0)
        node_mask = torch.ones(1, N, dtype=torch.bool, device=self.device)

        h0 = self.emb(node_types, roles, struct)
        context = self.ctx_pool(h0, node_mask)
        # Ensure diagonal allowed
        mask_3d = mask_3d.clone()
        for i in range(N):
            mask_3d[0, i, i] = False
        self.dynamics.set_context(context, mask=mask_3d)
        self.dynamics.set_n_steps(self.n_steps)
        h_out = euler_solve(self.dynamics, h0,
                            t_span=(0.0, 2.0), n_steps=self.n_steps)

        # Compute g and tau for analogy signature
        h_normed = self.dynamics.norm_geo(h_out)
        ctx_exp = context.unsqueeze(1).expand(1, N, self.config.d_model)
        mi = torch.cat([h_normed, ctx_exp], dim=-1)
        hidden = F.gelu(self.dynamics.metric_net_linear1(mi))
        g_metric = F.softplus(self.dynamics.metric_net_linear2_diag(hidden))
        tau = self.dynamics.compute_tau(h_out)
        if tau.dim() == 2:
            tau = tau.unsqueeze(-1)

        id_to_idx = {nid: i for i, nid in enumerate(order)}
        return {
            'h_out': h_out, 'node_mask': node_mask, 'context': context,
            'g': g_metric, 'tau': tau, 'struct': struct,
            'id_to_idx': id_to_idx, 'N': N, 'order': order, 'graph': g,
        }

    # ─────────────────────────────────────────────────────────────
    # Tool: analyze_graph
    # ─────────────────────────────────────────────────────────────

    def analyze_graph(self, graph_json: str, query_json: str) -> str:
        graph = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
        query = json.loads(query_json) if isinstance(query_json, str) else query_json
        nodes = _normalize_nodes(graph['nodes'])
        edges = _normalize_edges(graph['edges'])
        qtype = query.get('type')

        if qtype == 'root_cause':
            return self._root_cause(nodes, edges, query)
        if qtype == 'connection_check':
            return self._connection_check(nodes, edges, query)
        if qtype == 'implication_check':
            return self._implication_check(nodes, edges, query)
        if qtype == 'shortest_path':
            return self._shortest_path(nodes, edges, query)
        return json.dumps({'error': f'unsupported query type: {qtype}'})

    def _root_cause(self, nodes, edges, query) -> str:
        state = self._forward_graph(nodes, edges)
        target_id = query['target']
        if target_id not in state['id_to_idx']:
            return json.dumps({'error': f'target {target_id} not in graph'})
        target_idx = state['id_to_idx'][target_id]
        query_node = torch.tensor([target_idx], dtype=torch.long, device=self.device)
        g = state['graph']
        with torch.no_grad():
            logits = self.head.root_cause(
                state['h_out'], query_node, state['node_mask'])
            probs = F.softmax(logits, dim=-1)[0]

        # Enumerate candidates that are REACHABLE to the target (actual roots),
        # then pick the head's highest-probability reachable root.
        reachable_roots = []
        for candidate_id in state['order']:
            if candidate_id == target_id:
                continue
            try:
                nx.shortest_path(g, source=candidate_id, target=target_id)
                reachable_roots.append(candidate_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
        if not reachable_roots:
            # Fall back to head's raw top choice
            top_idx = int(probs.argmax().item())
            root_id = state['order'][top_idx]
            confidence = float(probs[top_idx].item())
            path_ids = [root_id]
        else:
            # Among reachable candidates, rank by head probability; tie-break
            # by path length (longer path = more upstream = more "root-like").
            ranked = []
            for cid in reachable_roots:
                idx = state['id_to_idx'][cid]
                prob = float(probs[idx].item())
                try:
                    path = nx.shortest_path(g, source=cid, target=target_id)
                    depth = len(path)
                except Exception:
                    depth = 0
                ranked.append((prob, depth, cid, path))
            # Sort by: highest prob, then deepest chain (prefer longer ancestry)
            ranked.sort(key=lambda x: (-x[0], -x[1]))
            root_id = ranked[0][2]
            confidence = ranked[0][0]
            path_ids = ranked[0][3]
        return json.dumps({
            'root_cause': root_id,
            'confidence': confidence,
            'path': path_ids,
            'hops': max(0, len(path_ids) - 1),
        })

    def _connection_check(self, nodes, edges, query) -> str:
        state = self._forward_graph(nodes, edges)
        src_id = query['src']
        dst_id = query['dst']
        if src_id not in state['id_to_idx'] or dst_id not in state['id_to_idx']:
            return json.dumps({'error': 'src or dst not in graph'})
        src_idx = state['id_to_idx'][src_id]
        dst_idx = state['id_to_idx'][dst_id]
        src_t = torch.tensor([src_idx], dtype=torch.long, device=self.device)
        dst_t = torch.tensor([dst_idx], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logit = self.head.connection(state['h_out'], src_t, dst_t)
            prob = float(torch.sigmoid(logit).item())
        # Fallback: use ground-truth BFS on the graph (more reliable than head
        # for this partial task).
        g = state['graph']
        try:
            connected_gt = nx.has_path(g.to_undirected(), src_id, dst_id)
        except Exception:
            connected_gt = False
        return json.dumps({
            'connected_head_prob': prob,
            'connected': bool(connected_gt),   # authoritative (from graph)
            'head_says_connected': prob > 0.5,
        })

    # ─────────────────────────────────────────────────────────────
    # Online correction mechanism
    # ─────────────────────────────────────────────────────────────

    def correct_answer(self, graph_json, query_json, correct_answer_json) -> str:
        """Apply a gradient step on the head using a user-supplied correction.

        Safeguards (per dev recommendations):
          (1) Cap — reject more than N corrections on the same (graph, query).
          (2) LR — default 1e-5 (smaller than typical fine-tune), single Adam step.
          (3) Replay — after the main step, replay K diverse prior examples
              from the buffer to prevent overfitting to the latest correction.
          (4) Health check — after the step (+replay), measure head output
              variance across a probe set; if collapsed below threshold,
              roll head back to pristine-checkpoint state.

        Returns: JSON with before/after loss, current prediction, corrections_applied,
                 diagnostics including cap/replay/health info.
        """
        graph = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
        query = json.loads(query_json) if isinstance(query_json, str) else query_json
        target = json.loads(correct_answer_json) if isinstance(correct_answer_json, str) else correct_answer_json
        nodes = _normalize_nodes(graph['nodes'])
        edges = _normalize_edges(graph['edges'])

        # (1) Cap check
        key = self._example_key(graph, query)
        count_now = self.example_correction_count.get(key, 0)
        if count_now >= self.max_corrections_per_example:
            return json.dumps({
                'learned': False,
                'error': 'correction_cap_reached',
                'cap': self.max_corrections_per_example,
                'corrections_on_this_example': count_now,
                'corrections_applied_total': self.n_corrections_applied,
                'message': (
                    f"This exact (graph, query) has been corrected "
                    f"{count_now} times (cap = {self.max_corrections_per_example}). "
                    f"Provide a different example to continue teaching — otherwise "
                    f"the head will overfit to a single case."
                ),
            })

        # Before-loss + prediction
        before = self._head_loss_and_prediction(nodes, edges, query, target, train=False)
        if before.get('error'):
            return json.dumps(before)
        before_loss = before['loss']

        # (2) One Adam step on this example
        self.head.train()
        self.optimizer.zero_grad()
        main = self._head_loss_and_prediction(nodes, edges, query, target, train=True)
        loss_t = main.get('_loss_tensor')
        if loss_t is None:
            self.head.eval()
            return json.dumps({'error': 'could not compute gradient for this query type'})
        loss_t.backward()
        torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=1.0)
        self.optimizer.step()

        # (3) Replay diverse prior examples
        replay_stats = self._replay_diverse(key)

        self.head.eval()

        # Recompute prediction + loss after (main step + replay)
        after_eval = self._head_loss_and_prediction(nodes, edges, query, target, train=False)
        after_loss = after_eval['loss']

        # Record correction BEFORE health check so it's in the buffer
        self.n_corrections_applied += 1
        self.example_correction_count[key] = count_now + 1
        record = {
            'graph': graph, 'query': query, 'target': target,
            'before_loss': before_loss, 'after_loss': after_loss,
        }
        self.replay_buffer.append(record)
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)
        if self.corrections_log:
            self._append_to_log(record)

        # (4) Health check — did the head collapse to a constant output?
        health = self._head_health_check()
        rolled_back = False
        if health.get('collapsed', False):
            self._rollback_head()
            rolled_back = True

        return json.dumps({
            'learned': not rolled_back,
            'before_loss': before_loss,
            'after_loss': after_loss,
            'current_prediction': after_eval['prediction'],
            'corrections_applied': self.n_corrections_applied,
            'corrections_on_this_example': count_now + 1,
            'replay': replay_stats,
            'health': health,
            'rolled_back': rolled_back,
            'message': (
                'Head collapsed after this step — rolled back to pristine '
                'checkpoint state. Consider lowering --correction_lr further '
                'or teaching with more diverse examples.'
            ) if rolled_back else None,
        })

    # ─────────────────────────────────────────────────────────────
    # Correction safeguard helpers
    # ─────────────────────────────────────────────────────────────

    def _example_key(self, graph: Dict, query: Dict) -> str:
        """Stable hash-key over (graph, query) for the cap counter."""
        return json.dumps({'g': graph, 'q': query}, sort_keys=True)

    def _replay_diverse(self, exclude_key: str) -> Dict:
        """Replay up to `replay_per_correction` prior examples to maintain
        discrimination across the example distribution.

        Each replay item: one gradient step on that (graph, query, target).
        Excludes the item we just corrected so it doesn't double-count.
        """
        k = self.replay_per_correction
        if k <= 0 or not self.replay_buffer:
            return {'n_replayed': 0, 'mean_loss': None}

        candidates = [
            r for r in self.replay_buffer
            if self._example_key(r['graph'], r['query']) != exclude_key
        ]
        if not candidates:
            return {'n_replayed': 0, 'mean_loss': None}

        import random as _random
        sample = _random.sample(candidates, min(k, len(candidates)))

        losses = []
        for r in sample:
            try:
                self.optimizer.zero_grad()
                res = self._head_loss_and_prediction(
                    _normalize_nodes(r['graph']['nodes']),
                    _normalize_edges(r['graph']['edges']),
                    r['query'], r['target'], train=True)
                lt = res.get('_loss_tensor')
                if lt is None:
                    continue
                lt.backward()
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses.append(res.get('loss', 0.0))
            except Exception:
                continue

        return {
            'n_replayed': len(losses),
            'mean_loss': float(sum(losses) / max(1, len(losses))) if losses else None,
        }

    def _head_health_check(self) -> Dict:
        """Probe head discrimination across the replay buffer.

        Collapse = head outputs the same value regardless of input.

        Low output variance alone is NOT sufficient evidence of collapse:
        if all probe examples happen to have the same label (e.g. all
        want_valid=True), a well-functioning head will correctly output
        near-identical high values — low variance, but healthy.

        So we condition on *label diversity*. Only flag collapse when:
          - Labels in the probe set are diverse (at least 2 distinct values)
          - AND predictions are near-constant (variance < threshold)
        If labels are uniform, there's no evidence either way and we skip.
        """
        if len(self.replay_buffer) < 6:
            return {'checked': False, 'reason': 'insufficient_buffer_size'}

        probes = self.replay_buffer[-16:]
        outputs = []
        labels = []
        with torch.no_grad():
            for r in probes:
                try:
                    res = self._head_loss_and_prediction(
                        _normalize_nodes(r['graph']['nodes']),
                        _normalize_edges(r['graph']['edges']),
                        r['query'], r['target'], train=False)
                    pred = res.get('prediction', {})
                    if 'valid_prob' in pred:
                        outputs.append(float(pred['valid_prob']))
                        labels.append(1.0 if r['target'].get('valid', False) else 0.0)
                    elif 'prob' in pred:
                        outputs.append(float(pred['prob']))
                        labels.append(1.0 if r['target'].get('connected', False) else 0.0)
                    elif 'root_cause' in pred:
                        outputs.append(float(hash(pred['root_cause']) % 1000) / 1000.0)
                        labels.append(float(hash(r['target'].get('root_cause', '')) % 1000) / 1000.0)
                except Exception:
                    continue

        if len(outputs) < 4:
            return {'checked': False, 'reason': 'too_few_probe_outputs'}

        out_t = torch.tensor(outputs)
        lab_t = torch.tensor(labels)
        output_variance = float(out_t.var().item())
        label_variance = float(lab_t.var().item())

        # If all labels are the same, there's no discrimination test to run.
        if label_variance < 1e-6:
            return {
                'checked': True,
                'n_probes': len(outputs),
                'output_variance': output_variance,
                'label_variance': label_variance,
                'collapsed': False,
                'reason': 'uniform_labels_no_discrimination_test',
            }

        # Labels vary → check if predictions also vary enough to track them.
        collapsed = output_variance < self.health_check_variance_threshold
        # Additional signal: correlation between preds and labels.
        if out_t.std() > 1e-8 and lab_t.std() > 1e-8:
            corr = float(((out_t - out_t.mean()) * (lab_t - lab_t.mean())).mean()
                         / (out_t.std() * lab_t.std()))
        else:
            corr = 0.0
        return {
            'checked': True,
            'n_probes': len(outputs),
            'output_variance': output_variance,
            'label_variance': label_variance,
            'pred_label_correlation': corr,
            'threshold': self.health_check_variance_threshold,
            'collapsed': collapsed,
        }

    def _rollback_head(self):
        """Restore the head to its pristine checkpoint state; re-create the
        optimizer so its internal moment buffers are also fresh."""
        self.head.load_state_dict(self._head_pristine_state)
        self.optimizer = torch.optim.Adam(
            self.head.parameters(), lr=self.correction_lr)
        print("[GraphEngine] head rolled back to pristine checkpoint state.",
              flush=True)

    def _head_loss_and_prediction(self, nodes, edges, query, target, train=False):
        """Shared helper: forward → compute loss vs target → return prediction.

        When train=True, leaves the gradient graph attached so backward() can be
        called on the returned loss tensor. When train=False, detaches.
        """
        qtype = query.get('type')
        ids_forward_scope = query.get('context_scope') if qtype == 'implication_check' else None
        state = self._forward_graph(nodes, edges, active_scope=ids_forward_scope)
        ids = state['id_to_idx']

        if qtype == 'root_cause':
            target_node = target.get('root_cause')
            query_tgt = query.get('target')
            if query_tgt not in ids or target_node not in ids:
                return {'error': 'target or root_cause not in graph'}
            query_node_t = torch.tensor([ids[query_tgt]], dtype=torch.long, device=self.device)
            label = torch.tensor([ids[target_node]], dtype=torch.long, device=self.device)
            logits = self.head.root_cause(state['h_out'], query_node_t, state['node_mask'])
            loss = F.cross_entropy(logits, label)
            with torch.no_grad():
                pred_idx = int(logits.argmax(dim=-1).item())
            return {
                'loss': float(loss.item()),
                '_loss_tensor': loss if train else None,
                'prediction': {'root_cause': state['order'][pred_idx]},
            }

        if qtype == 'connection_check':
            src = query['src']; dst = query['dst']
            if src not in ids or dst not in ids:
                return {'error': 'src or dst not in graph'}
            src_t = torch.tensor([ids[src]], dtype=torch.long, device=self.device)
            dst_t = torch.tensor([ids[dst]], dtype=torch.long, device=self.device)
            label = torch.tensor([1.0 if target.get('connected', False) else 0.0],
                                 device=self.device)
            logit = self.head.connection(state['h_out'], src_t, dst_t)
            loss = F.binary_cross_entropy_with_logits(logit, label)
            with torch.no_grad():
                p = float(torch.sigmoid(logit).item())
            return {
                'loss': float(loss.item()),
                '_loss_tensor': loss if train else None,
                'prediction': {'connected': p > 0.5, 'prob': p},
            }

        if qtype == 'implication_check':
            scope = query['context_scope']
            premise = query['premise']
            concl = query['conclusion']
            if scope not in ids or premise not in ids or concl not in ids:
                return {'error': 'scope/premise/conclusion not in graph'}
            scope_t = torch.tensor([ids[scope]], dtype=torch.long, device=self.device)
            premise_t = torch.tensor([ids[premise]], dtype=torch.long, device=self.device)
            concl_t = torch.tensor([ids[concl]], dtype=torch.long, device=self.device)
            label = torch.tensor([1 if target.get('valid', False) else 0],
                                 dtype=torch.long, device=self.device)
            logits = self.head.implication(state['h_out'], scope_t, premise_t,
                                           concl_t, state['node_mask'])
            loss = F.cross_entropy(logits, label)
            with torch.no_grad():
                probs = F.softmax(logits, dim=-1)[0]
                valid_prob = float(probs[1].item())
            return {
                'loss': float(loss.item()),
                '_loss_tensor': loss if train else None,
                'prediction': {'valid': valid_prob > 0.5, 'valid_prob': valid_prob},
            }

        return {'error': f'correction not supported for query type: {qtype}'}

    def _append_to_log(self, record):
        try:
            with open(self.corrections_log, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception:
            pass

    def _replay_from_log(self, path):
        """Replay corrections from disk log on startup — allows persistent
        learning across MCP restarts."""
        try:
            with open(path) as f:
                records = [json.loads(line) for line in f if line.strip()]
        except Exception:
            return
        if not records:
            return
        print(f"[GraphEngine] replaying {len(records)} corrections from {path}", flush=True)
        for r in records:
            try:
                self.head.train()
                self.optimizer.zero_grad()
                res = self._head_loss_and_prediction(
                    _normalize_nodes(r['graph']['nodes']),
                    _normalize_edges(r['graph']['edges']),
                    r['query'], r['target'], train=True)
                lt = res.get('_loss_tensor')
                if lt is not None:
                    lt.backward()
                    torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=1.0)
                    self.optimizer.step()
                self.head.eval()
                self.replay_buffer.append(r)
                if len(self.replay_buffer) > self.replay_buffer_size:
                    self.replay_buffer.pop(0)
                self.n_corrections_applied += 1
            except Exception as e:
                print(f"[GraphEngine] replay skipped (err: {e})", flush=True)

    # ─────────────────────────────────────────────────────────────
    # Tool: analyze_graph — implementations
    # ─────────────────────────────────────────────────────────────

    def _implication_check(self, nodes, edges, query) -> str:
        premise_id = query['premise']
        conclusion_id = query['conclusion']
        scope_id = query['context_scope']
        state = self._forward_graph(nodes, edges, active_scope=scope_id)
        ids = state['id_to_idx']
        if premise_id not in ids or conclusion_id not in ids or scope_id not in ids:
            return json.dumps({'error': 'premise/conclusion/scope not in graph'})
        scope_t = torch.tensor([ids[scope_id]], dtype=torch.long, device=self.device)
        premise_t = torch.tensor([ids[premise_id]], dtype=torch.long, device=self.device)
        concl_t = torch.tensor([ids[conclusion_id]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.head.implication(
                state['h_out'], scope_t, premise_t, concl_t, state['node_mask'])
            probs = F.softmax(logits, dim=-1)[0]
            head_valid_prob = float(probs[1].item())

        # AUTHORITATIVE reachability under scope-filtered graph.
        # The trained head has known weaknesses: premise node is not in its
        # input (training limitation), and training used a fixed topology
        # that doesn't generalize. For correctness, compute the implication
        # deterministically from the graph structure.
        #
        # An implication (premise ⊨ conclusion) is VALID under scope S iff,
        # after dropping edges whose `scope` attribute is set and != S, the
        # conclusion is reachable from the premise via a directed path.
        g_full = state['graph']
        g_filt = nx.DiGraph()
        g_filt.add_nodes_from(g_full.nodes)
        for u, v, data in g_full.edges(data=True):
            edge_scope = data.get('scope', None) if isinstance(data, dict) else None
            if edge_scope is None or edge_scope == scope_id:
                g_filt.add_edge(u, v)
        try:
            valid_auth = nx.has_path(g_filt, premise_id, conclusion_id)
        except Exception:
            valid_auth = False

        return json.dumps({
            'valid': bool(valid_auth),             # authoritative (scope-filtered reachability)
            'head_valid_prob': head_valid_prob,    # LiquidARC head estimate
            'head_says_valid': head_valid_prob > 0.5,
            'confidence': 1.0,                     # for authoritative answer
            # For diagnostics: how many edges the scope filter dropped
            'n_edges_full': g_full.number_of_edges(),
            'n_edges_after_scope_filter': g_filt.number_of_edges(),
        })

    def _shortest_path(self, nodes, edges, query) -> str:
        src_id = query['src']
        dst_id = query['dst']
        g = _records_to_graph(nodes, edges)
        try:
            path = nx.shortest_path(g.to_undirected(), source=src_id, target=dst_id)
            return json.dumps({'path': path, 'hops': len(path) - 1})
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            return json.dumps({'error': str(e), 'path': None})

    # ─────────────────────────────────────────────────────────────
    # Tool: compare_graphs
    # ─────────────────────────────────────────────────────────────

    def compare_graphs(self, graph_a_json: str, graph_b_json: str) -> str:
        a = json.loads(graph_a_json) if isinstance(graph_a_json, str) else graph_a_json
        b = json.loads(graph_b_json) if isinstance(graph_b_json, str) else graph_b_json
        a_nodes = _normalize_nodes(a['nodes'])
        a_edges = _normalize_edges(a['edges'])
        b_nodes = _normalize_nodes(b['nodes'])
        b_edges = _normalize_edges(b['edges'])
        state_a = self._forward_graph(a_nodes, a_edges)
        state_b = self._forward_graph(b_nodes, b_edges)
        with torch.no_grad():
            sig_a = self.head.signature(
                state_a['h_out'], state_a['g'], state_a['tau'],
                state_a['node_mask'], struct_features=state_a['struct'])
            sig_b = self.head.signature(
                state_b['h_out'], state_b['g'], state_b['tau'],
                state_b['node_mask'], struct_features=state_b['struct'])
            cos_sim = float(F.cosine_similarity(sig_a, sig_b, dim=-1).item())
        # Also run a deterministic structural isomorphism check
        g_a = state_a['graph']
        g_b = state_b['graph']
        iso = nx.is_isomorphic(g_a, g_b)
        return json.dumps({
            'isomorphic': bool(iso),                 # authoritative
            'signature_cosine': cos_sim,             # LiquidARC estimate
            'n_a': state_a['N'], 'n_b': state_b['N'],
        })

    # ─────────────────────────────────────────────────────────────
    # Tool: get_graph_diagnostics
    # ─────────────────────────────────────────────────────────────

    def get_graph_diagnostics(self, graph_json: str) -> str:
        """Return the full set of diagnostics required by the spec (line 428):
        CV, D²/4τ, τ distribution, metric clusters, per-node centrality
        in metric space."""
        graph = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
        nodes = _normalize_nodes(graph['nodes'])
        edges = _normalize_edges(graph['edges'])
        state = self._forward_graph(nodes, edges)
        g = state['g'][0]        # [N, d_metric]
        tau = state['tau'][0].squeeze(-1)  # [N]
        h = state['h_out'][0]    # [N, d]
        order = state['order']
        N = state['N']

        cv = (g.std() / g.mean().clamp(min=1e-8)).item()
        tau_mean = tau.mean().item()
        tau_log_spread = (tau.clamp(min=1e-8)).log().std().item()

        # Full pairwise D² matrix under the learned metric
        d_metric = g.shape[-1]
        h_slice = h[:, :d_metric]
        diff_all = h_slice.unsqueeze(1) - h_slice.unsqueeze(0)         # [N, N, d_m]
        g_avg_all = 0.5 * (g.unsqueeze(1) + g.unsqueeze(0))            # [N, N, d_m]
        D2 = (diff_all * diff_all * g_avg_all).sum(dim=-1)             # [N, N]

        if N >= 2:
            iu, ju = torch.triu_indices(N, N, offset=1, device=self.device)
            d_sq_offdiag = D2[iu, ju]
            d_sq_median = float(d_sq_offdiag.median().item())
            crit_ratio = d_sq_median / (4.0 * tau_mean + 1e-8)
        else:
            d_sq_median = 0.0
            crit_ratio = 0.0

        # ── Metric clusters: agglomerative clustering on pairwise D² ──
        # Use a simple single-link / threshold approach: two nodes are in the
        # same cluster if D²(i,j) < 0.5 * median pairwise D². Produces a list
        # of node-id groupings.
        metric_clusters: list = []
        assigned = [-1] * N
        if N >= 1:
            threshold = 0.5 * d_sq_median if d_sq_median > 0 else 1.0
            cluster_id = 0
            for i in range(N):
                if assigned[i] != -1:
                    continue
                # Seed a new cluster with node i, BFS to collect all reachable
                # nodes under the metric-distance threshold.
                stack = [i]
                members = []
                while stack:
                    u = stack.pop()
                    if assigned[u] != -1:
                        continue
                    assigned[u] = cluster_id
                    members.append(order[u])
                    for v in range(N):
                        if assigned[v] == -1 and float(D2[u, v].item()) < threshold:
                            stack.append(v)
                metric_clusters.append({
                    'cluster_id': cluster_id,
                    'members': members,
                    'size': len(members),
                })
                cluster_id += 1

        # ── Per-node centrality in metric space ──
        # Closeness centrality under the learned metric:
        #   C(i) = (N - 1) / sum_{j != i} D²(i, j)
        # Higher = more central (low total distance to everyone else).
        per_node_centrality: dict = {}
        if N >= 2:
            for i in range(N):
                mask_not_i = torch.ones(N, dtype=torch.bool, device=self.device)
                mask_not_i[i] = False
                total_dist = float(D2[i, mask_not_i].sum().item())
                centrality = (N - 1) / total_dist if total_dist > 1e-8 else 0.0
                per_node_centrality[order[i]] = centrality
        else:
            per_node_centrality = {order[0]: 0.0} if N == 1 else {}

        # Normalize centrality to [0, 1] for interpretability
        if per_node_centrality:
            max_c = max(per_node_centrality.values())
            if max_c > 0:
                per_node_centrality_norm = {k: v / max_c
                                            for k, v in per_node_centrality.items()}
            else:
                per_node_centrality_norm = per_node_centrality
        else:
            per_node_centrality_norm = {}

        # τ distribution per-node (also required by spec's "tau distribution")
        tau_per_node = {order[i]: float(tau[i].item()) for i in range(N)}

        return json.dumps({
            'cv_g': cv,
            'd_sq_median': d_sq_median,
            'criticality_ratio': crit_ratio,
            'tau_mean': tau_mean,
            'tau_log_spread': tau_log_spread,
            'tau_distribution': tau_per_node,
            'metric_clusters': metric_clusters,
            'per_node_centrality_metric_space': per_node_centrality_norm,
            'n_nodes': state['N'],
            'n_edges': state['graph'].number_of_edges(),
        })
