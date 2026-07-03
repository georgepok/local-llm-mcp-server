"""Graph output heads for LiquidARC multi-task training.

Spec: GRAPH_REASONING_ENGINE_SPEC.md lines 349-361.

Four output heads consume the ODE-evolved hidden states h_out [B, N, d]:
  - root_cause(h, query_node)      → logits over nodes [B, N]
  - connection(h, src, dst)        → binary score [B]
  - signature(h)                   → graph-level metric signature [B, sig_dim]
  - implication(h, scope_node)     → {valid, invalid} logits [B, 2]

Each head is a thin projection — the real reasoning has already happened
in the ODE integration. These heads translate the evolved geometry into
task-specific predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphOutputHead(nn.Module):
    """Multi-task readout for graph reasoning.

    All four subheads share the same input (h_out from the ODE) and produce
    task-specific outputs. Total params at d_model=768: ~2M.
    """

    def __init__(self, d_model: int, n_tasks: int = 4,
                 sig_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_tasks = n_tasks
        self.sig_dim = sig_dim

        # Shared input norm so every head sees normalized h_out
        self.input_norm = nn.LayerNorm(d_model)

        # ── Head 1: root_cause — query attends over all nodes ─────────
        # Given a query node's embedding, produce node-level logits over the
        # whole graph. Implemented as a scalar projection on element-wise
        # interaction between query and each candidate node.
        self.rc_query = nn.Linear(d_model, d_model)
        self.rc_candidate = nn.Linear(d_model, d_model)
        self.rc_score = nn.Linear(d_model, 1)

        # ── Head 2: connection — dual-branch classifier ──
        # Branch 1 (MLP): concat(h_s, h_d, h_s - h_d, |h_s - h_d|) → logit_mlp
        # Branch 2 (log-distance): logit_dist = α*log(||h_s-h_d||² + ε) + β
        # Output: logit_mlp + logit_dist.
        # The log-distance branch gives a direct, architecturally-correct
        # signal that requires no learning to be informative — "close →
        # connected, far → disconnected" is encoded in the functional form.
        self.conn_pair = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        # Zero init MLP output so log-distance branch dominates at start
        nn.init.zeros_(self.conn_pair[-1].weight)
        nn.init.zeros_(self.conn_pair[-1].bias)
        # Log-distance scalars
        self.conn_log_scale = nn.Parameter(torch.tensor(-1.0))   # α (negative: close → high logit)
        self.conn_log_bias = nn.Parameter(torch.tensor(0.0))     # β

        # ── Head 3: signature — topology-invariant, fully differentiable ──
        # Stats 0-19: h/g/τ-derived (still dependent on type labels via h).
        # Stats 20-51: moments (mean, std) of the 16 structural features across
        #   valid nodes. These are PURELY TOPOLOGICAL — identical for
        #   isomorphic graphs regardless of type labels. This is what finally
        #   makes two same-topology-different-labels graphs produce matching
        #   signatures.
        self.sig_stat_dim = 20 + 32  # 20 geometric + 32 structural moments
        self.sig_out = nn.Sequential(
            nn.Linear(self.sig_stat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, sig_dim),
        )

        # ── Head 4: implication — premise + scope + conclusion + pooled ──
        # Input: pooled(h) ⊕ h[scope_node] ⊕ h[premise_node] ⊕ h[conclusion_node]
        # Including premise_node in the input is LOAD-BEARING: without it, the
        # head cannot express "from THIS premise reach THAT conclusion under
        # scope S". Previous head architecture omitted premise; the network
        # collapsed to "is conclusion valid under scope" which only worked
        # when training data held premise fixed.
        self.impl_head = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    # ─────────────────────────────────────────────────────────────
    # Head 1 — root cause
    # ─────────────────────────────────────────────────────────────

    def root_cause(self, h_out: torch.Tensor,
                   query_node: torch.Tensor,
                   node_mask: torch.Tensor = None) -> torch.Tensor:
        """Predict the root cause node for a given query (terminal) node.

        Args:
            h_out:      [B, N, d] ODE-evolved hidden states
            query_node: [B] long — index of the query node in each graph
            node_mask:  [B, N] bool, True where valid nodes; masked positions
                        receive -inf in logits.

        Returns:
            logits: [B, N] — unnormalized log-probabilities over candidate root causes
        """
        h = self.input_norm(h_out)
        B, N, d = h.shape
        # Gather query embedding per batch
        idx = query_node.view(B, 1, 1).expand(B, 1, d)
        query = h.gather(1, idx)                    # [B, 1, d]
        q_proj = self.rc_query(query)               # [B, 1, d]
        cand = self.rc_candidate(h)                 # [B, N, d]
        interact = F.gelu(q_proj + cand)            # broadcast to [B, N, d]
        logits = self.rc_score(interact).squeeze(-1)  # [B, N]
        if node_mask is not None:
            logits = logits.masked_fill(~node_mask, float('-inf'))
        return logits

    # ─────────────────────────────────────────────────────────────
    # Head 2 — connection check
    # ─────────────────────────────────────────────────────────────

    def connection(self, h_out: torch.Tensor,
                   src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Predict whether src and dst are connected.

        Dual branch: MLP over concat+diff + log-distance scalar head.
        The log-distance branch is architecturally correct from initialization
        (close → high connected-logit), so the head doesn't need to learn
        the basic relationship; the MLP branch fine-tunes residual signal.

        Args:
            h_out: [B, N, d]
            src:   [B] long
            dst:   [B] long
        Returns:
            logit: [B]
        """
        h = self.input_norm(h_out)
        B, N, d = h.shape
        hs = h.gather(1, src.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
        hd = h.gather(1, dst.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
        diff = hs - hd
        abs_diff = diff.abs()
        pair = torch.cat([hs, hd, diff, abs_diff], dim=-1)
        logit_mlp = self.conn_pair(pair).squeeze(-1)
        # Log-distance branch: logit = α * log(||diff||² + ε) + β
        d_sq = (diff * diff).sum(dim=-1).clamp(min=1e-8)
        logit_dist = self.conn_log_scale * d_sq.log() + self.conn_log_bias
        return logit_mlp + logit_dist

    # ─────────────────────────────────────────────────────────────
    # Head 3 — signature (analogy): topology-invariant metric statistics
    # ─────────────────────────────────────────────────────────────

    def signature(self, h_out: torch.Tensor,
                  g: torch.Tensor,
                  tau: torch.Tensor,
                  node_mask: torch.Tensor = None,
                  struct_features: torch.Tensor = None) -> torch.Tensor:
        """Compute a topology-invariant, fully-differentiable metric signature.

        Features 0-19: geometric (CV(g), D² moments + quantiles, τ, g, crit).
        Features 20-51: moments (mean, std) of the 16 structural features —
          PURELY TOPOLOGICAL, label-invariant. For isomorphic graphs with
          different type labels these 32 dims are identical, making the
          signature topology-matching rather than type-matching.

        Args:
            h_out:           [B, N, d] ODE-evolved states
            g:               [B, N, d_metric] diagonal metric
            tau:             [B, N, 1] per-position timescales
            node_mask:       [B, N] bool, True where valid
            struct_features: [B, N, 16] topology features (required when
                             sig_stat_dim includes the structural moments)

        Returns:
            sig: [B, sig_dim] graph-level signature
        """
        B, N, d = h_out.shape
        device = h_out.device

        h_normed = self.input_norm(h_out)

        stats = []
        for b in range(B):
            valid = node_mask[b] if node_mask is not None else torch.ones(
                N, dtype=torch.bool, device=device)
            N_b = int(valid.sum().item())
            if N_b < 2:
                stats.append(torch.zeros(self.sig_stat_dim, device=device,
                                         dtype=h_out.dtype))
                continue
            h_b = h_normed[b][valid]                          # [N_b, d]
            g_b = g[b][valid]                                 # [N_b, d_metric]
            tau_b = tau[b][valid].squeeze(-1)                 # [N_b]

            # CV(g)
            cv_g = (g_b.std() / (g_b.mean().clamp(min=1e-8))).clamp(max=100.0)

            # Pairwise D² under the learned metric
            d_m = min(g_b.shape[-1], h_b.shape[-1])
            diff = h_b.unsqueeze(1)[..., :d_m] - h_b.unsqueeze(0)[..., :d_m]
            g_avg = 0.5 * (g_b.unsqueeze(1)[..., :d_m] + g_b.unsqueeze(0)[..., :d_m])
            d2 = (diff ** 2 * g_avg).sum(dim=-1)              # [N_b, N_b]
            iu, ju = torch.triu_indices(N_b, N_b, offset=1, device=device)
            d2_flat = d2[iu, ju].clamp(min=1e-8)              # [P] pairs

            # D² distribution moments (differentiable)
            d2_mean = d2_flat.mean()
            d2_std = d2_flat.std().clamp(min=1e-8)
            zn = (d2_flat - d2_mean) / d2_std
            d2_skew = (zn ** 3).mean().clamp(min=-20.0, max=20.0)
            d2_kurt = (zn ** 4).mean().clamp(min=0.0, max=50.0)
            d2_min = d2_flat.min()
            d2_max = d2_flat.max()

            # Order statistics via sort (differentiable through gather)
            d2_sorted, _ = d2_flat.sort()
            P = d2_sorted.shape[0]
            # Quantiles at p = 0.1 .. 0.8 (8 values)
            q_positions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            q_indices = torch.tensor(
                [min(max(int(p * (P - 1)), 0), P - 1) for p in q_positions],
                device=device, dtype=torch.long)
            d2_quantiles = d2_sorted[q_indices]               # [8]
            # Log-scale quantiles for numerical range compression
            d2_quantiles_log = d2_quantiles.log()

            # τ statistics
            tau_mean = tau_b.mean()
            tau_log_spread = tau_b.clamp(min=1e-8).log().std().nan_to_num(0.0)

            # g statistics
            g_mean = g_b.mean()
            g_ratio_log = (g_b.max().clamp(min=1e-8) /
                           g_b.min().clamp(min=1e-8)).clamp(max=1e6).log()

            # Criticality ratio
            crit_ratio = d2_mean / (4.0 * tau_mean.clamp(min=1e-8))

            row_geom = torch.stack([
                cv_g,
                d2_mean.log(),
                d2_std.log(),
                d2_skew,
                d2_kurt,
                d2_min.log(),
                d2_max.log(),
                *d2_quantiles_log.unbind(0),
                tau_mean,
                tau_log_spread,
                g_mean,
                g_ratio_log,
                crit_ratio.log().clamp(min=-10.0, max=10.0),
            ])  # 20 elements

            # Structural-feature moments (topology-only, label-invariant)
            if struct_features is not None:
                sf = struct_features[b][valid]                    # [N_b, 16]
                sf_mean = sf.mean(dim=0)                          # [16]
                sf_std = sf.std(dim=0).nan_to_num(0.0)            # [16]
                row_struct = torch.cat([sf_mean, sf_std])         # [32]
            else:
                row_struct = torch.zeros(32, device=device, dtype=h_out.dtype)

            row = torch.cat([row_geom, row_struct])               # [52]
            stats.append(row)

        stats_t = torch.stack(stats, dim=0)                    # [B, sig_stat_dim]
        return self.sig_out(stats_t)

    # ─────────────────────────────────────────────────────────────
    # Head 4 — scoped implication
    # ─────────────────────────────────────────────────────────────

    def implication(self, h_out: torch.Tensor,
                    scope_node: torch.Tensor,
                    premise_node: torch.Tensor,
                    conclusion_node: torch.Tensor,
                    node_mask: torch.Tensor = None) -> torch.Tensor:
        """Predict validity of (premise ⊨ conclusion) under scope.

        Args:
            h_out:           [B, N, d]
            scope_node:      [B] long — scope-defining node
            premise_node:    [B] long — premise node (from which to reach)
            conclusion_node: [B] long — conclusion node (to reach)
            node_mask:       [B, N] bool
        Returns:
            logits: [B, 2] over {invalid, valid}
        """
        h = self.input_norm(h_out)
        B, N, d = h.shape
        if node_mask is not None:
            mask_f = node_mask.float().unsqueeze(-1)
            pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            pooled = h.mean(dim=1)
        h_scope = h.gather(1, scope_node.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
        h_premise = h.gather(1, premise_node.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
        h_concl = h.gather(1, conclusion_node.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
        comb = torch.cat([pooled, h_scope, h_premise, h_concl], dim=-1)
        return self.impl_head(comb)


if __name__ == "__main__":
    print("Testing GraphOutputHead...")
    head = GraphOutputHead(d_model=128)
    B, N, d = 2, 6, 128
    h = torch.randn(B, N, d)
    node_mask = torch.ones(B, N, dtype=torch.bool)

    query = torch.tensor([5, 3], dtype=torch.long)
    rc_logits = head.root_cause(h, query, node_mask)
    assert rc_logits.shape == (B, N), f"rc: {rc_logits.shape}"

    src = torch.tensor([0, 2], dtype=torch.long)
    dst = torch.tensor([5, 3], dtype=torch.long)
    conn_logit = head.connection(h, src, dst)
    assert conn_logit.shape == (B,), f"conn: {conn_logit.shape}"

    g = torch.rand(B, N, d) + 0.1     # positive metric
    tau = torch.rand(B, N, 1) + 0.1
    sf = torch.rand(B, N, 16)
    sig = head.signature(h, g, tau, node_mask, struct_features=sf)
    assert sig.shape == (B, 64), f"sig: {sig.shape}"

    scope = torch.tensor([0, 1], dtype=torch.long)
    premise = torch.tensor([2, 0], dtype=torch.long)
    concl = torch.tensor([4, 2], dtype=torch.long)
    impl = head.implication(h, scope, premise, concl, node_mask)
    assert impl.shape == (B, 2), f"impl: {impl.shape}"

    # Gradient flow
    (rc_logits.sum() + conn_logit.sum() + sig.sum() + impl.sum()).backward()
    n_params = sum(p.numel() for p in head.parameters())
    print(f"  rc_logits: {rc_logits.shape}")
    print(f"  conn_logit: {conn_logit.shape}")
    print(f"  signature: {sig.shape}")
    print(f"  implication: {impl.shape}")
    print(f"  params: {n_params:,}")
    print("GraphOutputHead OK")
