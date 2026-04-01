"""FluidNetModel — pure geometric computation language model.

No attention anywhere. Information routing is entirely through diffusion
kernels derived from learned Riemannian geometry. Each layer has its own
MetricNet and TimeNet; all share the same ContextPool output.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig
from .context_pool import ContextPool
from .fluid_layer import FluidLayer
from .losses import CurvatureRegularization
from .structural_energy import StructuralEnergy


class DistancePredictor(nn.Module):
    """Auxiliary head: predict shortest-path hop count from room embedding pairs.

    Takes two room embeddings from h, outputs logits over hop count classes.
    Forces the model to encode spatial relationships in hidden states.
    """
    def __init__(self, d_model: int, max_hops: int = 10):
        super().__init__()
        self.max_hops = max_hops
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, max_hops),
        )

    def forward(self, h: torch.Tensor, room_token_positions: torch.Tensor,
                room_distances: torch.Tensor, n_rooms: torch.Tensor):
        """Predict hop counts for sampled room pairs.

        Args:
            h: [B, N, d] hidden states
            room_token_positions: [B, R_max] token positions
            room_distances: [B, R_max, R_max] normalized distances (0-1)
            n_rooms: [B] actual room count per episode

        Returns:
            aux_loss: scalar cross-entropy loss for hop prediction
        """
        B = h.shape[0]
        device = h.device
        total_loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for b in range(B):
            nr = n_rooms[b].item()
            if nr < 2:
                continue

            # Get room embeddings
            positions = room_token_positions[b, :nr].long()
            h_rooms = h[b, positions]  # [nr, d]

            # Get distance matrix for this episode, unnormalize to hop counts
            dist_mat = room_distances[b, :nr, :nr]  # [nr, nr] normalized 0-1

            # Sample upper-triangle pairs (exclude self-pairs)
            rows, cols = torch.triu_indices(nr, nr, offset=1, device=device)
            if rows.shape[0] == 0:
                continue

            # Cap at 32 pairs per episode to keep cost bounded
            if rows.shape[0] > 32:
                perm = torch.randperm(rows.shape[0], device=device)[:32]
                rows, cols = rows[perm], cols[perm]

            # Concatenate pair embeddings
            h_pairs = torch.cat([h_rooms[rows], h_rooms[cols]], dim=-1)  # [P, 2d]

            # Predict hop class
            logits = self.net(h_pairs)  # [P, max_hops]

            # Convert normalized distances to hop counts (integer classes)
            # dist_mat stores normalized distances; max distance = nr-1 hops for connected graph
            raw_dists = dist_mat[rows, cols]
            # Unnormalize: multiply by max possible distance and round
            max_dist = raw_dists.max()
            if max_dist > 0:
                hop_counts = (raw_dists / max_dist * (nr - 1)).round().long()
            else:
                hop_counts = torch.zeros_like(rows)
            hop_counts = hop_counts.clamp(0, self.max_hops - 1)

            total_loss = total_loss + F.cross_entropy(logits, hop_counts)
            n_pairs += 1

        if n_pairs > 0:
            total_loss = total_loss / n_pairs

        return total_loss


class FluidNetModel(nn.Module):
    def __init__(self, config: FGNConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Context pooling (shared, computed once from embeddings)
        self.context_pool = ContextPool(config)

        # FluidLayers
        self.layers = nn.ModuleList([
            FluidLayer(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Curvature regularization (compatible, typically disabled via lambda=0)
        self.curv_reg = CurvatureRegularization(
            n_layers=config.n_layers,
            curvature_lambda=config.curvature_lambda,
            correlation_length_init=config.correlation_length_init,
            curvature_reward_mu=config.curvature_reward_mu,
        )

        # Structural energy (resonance) — graph-distance mode
        self.structural_energy = StructuralEnergy(
            max_context_pairs=config.structural_energy_max_pairs,
            mode="graph",
            d_model=config.d_model,
            d_proj=config.structural_energy_d_proj,
            proj_mlp=config.structural_energy_proj_mlp,
        )
        self.lambda_struct = config.structural_energy_lambda

        # Auxiliary distance prediction head
        if config.aux_distance_max_hops > 0:
            self.distance_predictor = DistancePredictor(
                d_model=config.d_model,
                max_hops=config.aux_distance_max_hops,
            )
        else:
            self.distance_predictor = None

        # Initialize weights (generic init, then re-apply special FluidLayer inits)
        self.apply(self._init_weights)
        for layer in self.layers:
            layer.reinit_special()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        room_distances: Optional[torch.Tensor] = None,
        room_token_positions: Optional[torch.Tensor] = None,
        n_rooms: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: [B, N] token indices
            labels: [B, N] target tokens (optional)
            context_mask: [B, N] boolean mask for context positions (optional)
            room_distances: [B, R_max, R_max] normalized graph distances (optional)
            room_token_positions: [B, R_max] token positions per room (optional)
            n_rooms: [B] room count per episode (optional)

        Returns:
            Dict with logits, loss, metrics, and compatibility fields.
        """
        _, N = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(N, device=device).unsqueeze(0)
        h = self.embed(input_ids) + self.pos_embed(pos)

        # Compute shared context once from initial embeddings
        context = self.context_pool(h, context_mask)  # [B, d]

        # Causal mask
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)

        # Forward through layers
        curvatures = []
        metric_cvs = []
        t_avgs = []

        for layer in self.layers:
            h, kappa, m_cv, t_avg = layer(h, context, mask=mask)
            curvatures.append(kappa)
            metric_cvs.append(m_cv)
            t_avgs.append(t_avg)

        # Structural energy from LAST layer's processed hidden states.
        if self.lambda_struct > 0 and room_distances is not None:
            if self.structural_energy.proj is not None:
                # Projection mode: Euclidean distance in projected space, no metric needed
                e_struct = self.structural_energy(
                    h, None,
                    room_distances=room_distances,
                    room_token_positions=room_token_positions,
                    n_rooms=n_rooms,
                )
            else:
                # Metric mode: geodesic distance using diagonal metric
                g_final = self.layers[-1].get_current_metric(h, context)
                e_struct = self.structural_energy(
                    h, g_final,
                    room_distances=room_distances,
                    room_token_positions=room_token_positions,
                    n_rooms=n_rooms,
                )
        else:
            e_struct = torch.tensor(0.0, device=device)

        aux_dist_loss = torch.tensor(0.0, device=device)

        # Save pre-norm h for aux distance prediction (computed outside torch.compile)
        h_pre_norm = h

        # LM head
        h = self.norm(h)
        logits = self.lm_head(h)

        result = {"logits": logits, "aux_dist_loss": aux_dist_loss,
                  "h_pre_norm": h_pre_norm}

        # Monitoring stats
        result["metric_cv"] = sum(metric_cvs) / len(metric_cvs)
        result["avg_kappa"] = sum(k.abs().mean() for k in curvatures) / len(curvatures)

        # FluidNet-specific: average timescales across layers
        t_stack = torch.stack(t_avgs)  # [n_layers, B, n_scales]
        t_mean = t_stack.mean(dim=(0, 1))  # [n_scales]
        if t_mean.shape[0] >= 3:
            result["avg_t_local"] = t_mean[0]
            result["avg_t_medium"] = t_mean[1]
            result["avg_t_global"] = t_mean[2]
        else:
            result["avg_t_local"] = t_mean[0]
            result["avg_t_medium"] = t_mean[-1]
            result["avg_t_global"] = t_mean[-1]

        # Compatibility with v6/v7 training scripts
        result["scale_loss"] = torch.tensor(0.0, device=device)
        result["avg_gate"] = torch.tensor(0.0, device=device)
        result["escalation_rate"] = torch.tensor(0.0, device=device)
        result["avg_entropy"] = torch.tensor(0.0, device=device)
        result["esc_rates_per_layer"] = [torch.tensor(0.0, device=device)] * len(self.layers)
        result["entropies_per_layer"] = [torch.tensor(0.0, device=device)] * len(self.layers)

        result["structural_energy"] = e_struct

        if labels is not None:
            # Use reduction='sum' and divide manually to avoid NaN when all labels are -100
            flat_labels = labels.reshape(-1)
            n_valid = (flat_labels != -100).sum().clamp(min=1)
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                flat_labels,
                ignore_index=-100,
                reduction='sum',
            ) / n_valid
            result["ce_loss"] = ce_loss

            curv_loss = self.curv_reg(curvatures)
            result["curv_loss"] = curv_loss
            result["esc_penalty"] = torch.tensor(0.0, device=device)

            # Curvature floor: penalize |κ| below target to keep geometry active
            kappa_floor_loss = torch.tensor(0.0, device=device)
            if self.config.kappa_floor > 0 and self.config.kappa_floor_mu > 0:
                avg_kappa = result["avg_kappa"]
                kappa_floor_loss = self.config.kappa_floor_mu * F.relu(
                    self.config.kappa_floor - avg_kappa)
            result["kappa_floor_loss"] = kappa_floor_loss

            result["loss"] = (ce_loss + curv_loss + self.lambda_struct * e_struct
                              + kappa_floor_loss)

        return result

    def geo_parameters(self) -> List[nn.Parameter]:
        """MetricNet + TimeNet + CurvatureReg + ContextPool + StructuralEnergy + DistancePredictor parameters."""
        params = list(self.context_pool.parameters())
        params.extend(self.curv_reg.parameters())
        params.extend(self.structural_energy.parameters())
        if self.distance_predictor is not None:
            params.extend(self.distance_predictor.parameters())
        for layer in self.layers:
            params.extend(layer.metric_net_linear1.parameters())
            params.extend(layer.metric_net_linear2.parameters())
            params.extend(layer.time_net_linear1.parameters())
            params.extend(layer.time_net_linear2.parameters())
            params.extend(layer.curvature_engine.parameters())
        return params

    def other_parameters(self) -> List[nn.Parameter]:
        """Everything not in geo_parameters."""
        geo_ids = {id(p) for p in self.geo_parameters()}
        return [p for p in self.parameters() if id(p) not in geo_ids]


if __name__ == "__main__":
    print("Testing FluidNetModel...")

    cfg = FGNConfig(
        d_model=64, n_heads=4, d_ff=256, n_layers=4,
        vocab_size=100, max_seq_len=32,
        d_metric=16, d_ffn_fluid=128, n_scales=3,
        architecture_version="fluid",
        geo_metric_type="learned",
        curvature_lambda=0.0,
        curvature_reward_mu=0.0,
    )
    model = FluidNetModel(cfg)

    B, N = 2, 16
    input_ids = torch.randint(0, 100, (B, N))
    labels = torch.randint(0, 100, (B, N))
    context_mask = torch.zeros(B, N, dtype=torch.bool)
    context_mask[:, :4] = True

    # Forward with labels
    result = model(input_ids, labels=labels, context_mask=context_mask)

    expected_keys = {
        "logits", "loss", "ce_loss", "curv_loss", "esc_penalty",
        "metric_cv", "avg_kappa", "structural_energy",
        "avg_t_local", "avg_t_medium", "avg_t_global",
        "scale_loss", "avg_gate", "escalation_rate", "avg_entropy",
        "esc_rates_per_layer", "entropies_per_layer",
        "aux_dist_loss", "kappa_floor_loss", "h_pre_norm",
    }
    assert expected_keys.issubset(set(result.keys())), \
        f"Missing keys: {expected_keys - set(result.keys())}"

    assert result["logits"].shape == (B, N, 100)
    assert result["loss"].ndim == 0
    print(f"  Loss: {result['loss'].item():.4f}")
    print(f"  CE: {result['ce_loss'].item():.4f}")
    print(f"  CV: {result['metric_cv']:.4f}")
    print(f"  |k|: {result['avg_kappa'].item():.4f}")
    print(f"  t_local: {result['avg_t_local'].item():.4f}")
    print(f"  t_medium: {result['avg_t_medium'].item():.4f}")
    print(f"  t_global: {result['avg_t_global'].item():.4f}")

    # Gradient flow
    result["loss"].backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for _ in model.parameters())
    print(f"  Gradients: {has_grad}/{total}")
    assert has_grad > 0

    # Parameter groups
    geo_ids = {id(p) for p in model.geo_parameters()}
    other_ids = {id(p) for p in model.other_parameters()}
    all_ids = {id(p) for p in model.parameters()}
    assert len(geo_ids & other_ids) == 0, "Param groups overlap"
    assert geo_ids | other_ids == all_ids, "Param groups don't cover all"
    print(f"  Param groups: geo={len(geo_ids)}, other={len(other_ids)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    print("FluidNetModel OK")
