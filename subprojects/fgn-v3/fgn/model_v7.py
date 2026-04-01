"""FGNv7Model — Sandwich architecture: bottom geo → middle attn → top geo.

Structural division of labor:
  - Bottom geo layers: perceptual geometry (read and organize world state)
  - Middle attn layers: reasoning (multi-hop planning over organized representations)
  - Top geo layers: action geometry (map reasoning to output actions)

No gates, no budgets, no escalation, no thresholds.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FGNConfig
from .context_pool import ContextPool
from .layer_v7_geo import GeoOnlyLayer
from .layer_v7_attn import AttnOnlyLayer
from .losses import CurvatureRegularization


class FGNv7Model(nn.Module):
    def __init__(self, config: FGNConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)

        # Shared ContextPool (computed once from embeddings)
        if config.geo_metric_type == "learned":
            self.context_pool = ContextPool(config)

        # Bottom geometric layers
        self.bottom_geo = nn.ModuleList([
            GeoOnlyLayer(config, layer_idx=i)
            for i in range(config.sandwich_bottom_geo_layers)
        ])

        # Middle attention layers
        mid_start = config.sandwich_bottom_geo_layers
        self.middle_attn = nn.ModuleList([
            AttnOnlyLayer(config, layer_idx=mid_start + i)
            for i in range(config.sandwich_middle_attn_layers)
        ])

        # Top geometric layers (separate MetricNetwork per layer — different weights)
        top_start = mid_start + config.sandwich_middle_attn_layers
        self.top_geo = nn.ModuleList([
            GeoOnlyLayer(config, layer_idx=top_start + i)
            for i in range(config.sandwich_top_geo_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Curvature regularization (only geo layers have curvature)
        n_geo_layers = config.sandwich_bottom_geo_layers + config.sandwich_top_geo_layers
        self.curv_reg = CurvatureRegularization(
            n_layers=n_geo_layers,
            curvature_lambda=config.curvature_lambda,
            correlation_length_init=config.correlation_length_init,
            curvature_reward_mu=config.curvature_reward_mu,
        )

        # Initialize weights
        self.apply(self._init_weights)

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
    ) -> Dict[str, torch.Tensor]:
        _, N = input_ids.shape
        device = input_ids.device

        # 1. Embeddings
        pos = torch.arange(N, device=device).unsqueeze(0)
        h = self.embed(input_ids) + self.pos_embed(pos)

        # 2. Shared context (once, from embeddings)
        context = None
        if hasattr(self, 'context_pool'):
            context = self.context_pool(h, context_mask)

        # 3. Causal mask
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)

        # 4. Bottom geometric layers
        curvatures = []
        metric_cvs = []
        geo_entropies = []

        for layer in self.bottom_geo:
            h, kappa, m_cv, avg_ent = layer(h, mask=mask, context=context)
            curvatures.append(kappa)
            metric_cvs.append(m_cv)
            geo_entropies.append(avg_ent)

        # 5. Middle attention layers
        for layer in self.middle_attn:
            h = layer(h, mask=mask)

        # 6. Top geometric layers (same context, separate metrics)
        for layer in self.top_geo:
            h, kappa, m_cv, avg_ent = layer(h, mask=mask, context=context)
            curvatures.append(kappa)
            metric_cvs.append(m_cv)
            geo_entropies.append(avg_ent)

        # 7. LM head
        h = self.norm(h)
        logits = self.lm_head(h)

        # 8. Build result dict
        result = {"logits": logits}

        # Monitoring stats
        result["metric_cv"] = sum(metric_cvs) / len(metric_cvs)
        result["avg_kappa"] = sum(k.abs().mean() for k in curvatures) / len(curvatures)
        result["avg_entropy"] = sum(geo_entropies) / len(geo_entropies)

        # Per-stage stats (v7 specific)
        n_bot = self.config.sandwich_bottom_geo_layers
        result["bottom_metric_cv"] = sum(metric_cvs[:n_bot]) / max(n_bot, 1)
        result["top_metric_cv"] = sum(metric_cvs[n_bot:]) / max(len(metric_cvs) - n_bot, 1)
        result["bottom_avg_kappa"] = sum(
            k.abs().mean() for k in curvatures[:n_bot]) / max(n_bot, 1)
        result["top_avg_kappa"] = sum(
            k.abs().mean() for k in curvatures[n_bot:]) / max(len(curvatures) - n_bot, 1)

        # v6 compatibility fields
        result["escalation_rate"] = torch.tensor(0.0, device=device)
        result["esc_rates_per_layer"] = []
        result["entropies_per_layer"] = list(geo_entropies)
        result["scale_loss"] = torch.tensor(0.0, device=device)
        result["avg_gate"] = torch.tensor(0.0, device=device)

        if labels is not None:
            ce_loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
            result["ce_loss"] = ce_loss

            curv_loss = self.curv_reg(curvatures)
            result["curv_loss"] = curv_loss
            result["esc_penalty"] = torch.tensor(0.0, device=device)
            result["loss"] = ce_loss + curv_loss

        return result

    def geo_parameters(self) -> List[nn.Parameter]:
        """All geometric pathway params."""
        params = []
        if hasattr(self, 'context_pool'):
            params.extend(self.context_pool.parameters())
        for layer in self.bottom_geo:
            if hasattr(layer, 'metric'):
                params.extend(layer.metric.parameters())
            params.extend(layer.geo_route.parameters())
        for layer in self.top_geo:
            if hasattr(layer, 'metric'):
                params.extend(layer.metric.parameters())
            params.extend(layer.geo_route.parameters())
        params.extend(self.curv_reg.parameters())
        return params

    def attn_parameters(self) -> List[nn.Parameter]:
        """All attention pathway params."""
        params = []
        for layer in self.middle_attn:
            params.extend(layer.attention.parameters())
        return params

    def other_parameters(self) -> List[nn.Parameter]:
        """Embeddings, FFNs, LayerNorms, lm_head."""
        geo_ids = {id(p) for p in self.geo_parameters()}
        attn_ids = {id(p) for p in self.attn_parameters()}
        return [p for p in self.parameters()
                if id(p) not in geo_ids and id(p) not in attn_ids]


if __name__ == "__main__":
    print("=== FGNv7Model Self-Test ===\n")

    cfg = FGNConfig(
        d_model=64, n_heads=4, d_ff=256, vocab_size=100, max_seq_len=32,
        geo_heads=4, architecture_version="v7", geo_metric_type="learned",
        curvature_lambda=0.0, curvature_reward_mu=0.0,
        sandwich_mode=True, sandwich_bottom_geo_layers=2,
        sandwich_middle_attn_layers=4, sandwich_top_geo_layers=2,
    )

    # Test 1: Full forward/backward
    print("--- Test 1: Forward/backward ---")
    model = FGNv7Model(cfg)
    B, N = 2, 16
    input_ids = torch.randint(0, 100, (B, N))
    labels = torch.randint(0, 100, (B, N))
    ctx_mask = torch.zeros(B, N, dtype=torch.bool)
    ctx_mask[:, :4] = True

    result = model(input_ids, labels=labels, context_mask=ctx_mask)

    expected_keys = {
        "logits", "loss", "ce_loss", "curv_loss", "esc_penalty",
        "metric_cv", "avg_kappa", "avg_entropy",
        "bottom_metric_cv", "top_metric_cv",
        "bottom_avg_kappa", "top_avg_kappa",
        "escalation_rate", "esc_rates_per_layer", "entropies_per_layer",
        "scale_loss", "avg_gate",
    }
    assert expected_keys.issubset(set(result.keys())), \
        f"Missing keys: {expected_keys - set(result.keys())}"
    assert result["logits"].shape == (B, N, 100)
    assert result["loss"].ndim == 0

    result["loss"].backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for _ in model.parameters())
    print(f"Gradients: {has_grad}/{total}")
    assert has_grad > 0
    print("PASS\n")

    # Test 2: Layer counts
    print("--- Test 2: Layer counts ---")
    assert len(model.bottom_geo) == 2
    assert len(model.middle_attn) == 4
    assert len(model.top_geo) == 2
    print("PASS\n")

    # Test 3: Separate metric weights
    print("--- Test 3: Separate metrics ---")
    assert model.bottom_geo[0].metric is not model.top_geo[0].metric
    bot_w = model.bottom_geo[0].metric.h_proj.weight
    top_w = model.top_geo[0].metric.h_proj.weight
    assert bot_w is not top_w
    print("PASS\n")

    # Test 4: Disjoint parameter groups
    print("--- Test 4: Parameter groups ---")
    geo_ids = {id(p) for p in model.geo_parameters()}
    attn_ids = {id(p) for p in model.attn_parameters()}
    other_ids = {id(p) for p in model.other_parameters()}
    all_ids = {id(p) for p in model.parameters()}

    assert len(geo_ids & attn_ids) == 0, "geo/attn overlap"
    assert len(geo_ids & other_ids) == 0, "geo/other overlap"
    assert len(attn_ids & other_ids) == 0, "attn/other overlap"
    assert geo_ids | attn_ids | other_ids == all_ids, "groups don't cover all params"
    print(f"geo={len(geo_ids)}, attn={len(attn_ids)}, other={len(other_ids)}")
    print("PASS\n")

    # Test 5: v7-specific stats populated
    print("--- Test 5: v7 stats ---")
    assert result["bottom_metric_cv"].item() > -1  # just check it's a number
    assert result["top_metric_cv"].item() > -1
    assert result["bottom_avg_kappa"].item() >= 0
    assert result["top_avg_kappa"].item() >= 0
    print("PASS\n")

    # Test 6: v6 compatibility fields
    print("--- Test 6: v6 compat ---")
    assert result["escalation_rate"].item() == 0.0
    assert result["esc_penalty"].item() == 0.0
    assert result["scale_loss"].item() == 0.0
    assert result["avg_gate"].item() == 0.0
    print("PASS\n")

    # Test 7: Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    n_geo = sum(p.numel() for p in model.geo_parameters())
    n_attn = sum(p.numel() for p in model.attn_parameters())
    n_other = sum(p.numel() for p in model.other_parameters())
    print(f"--- Test 7: Params ---")
    print(f"Total: {n_params:,}  geo: {n_geo:,}  attn: {n_attn:,}  other: {n_other:,}")
    assert n_geo + n_attn + n_other == n_params
    print("PASS\n")

    print("=== All FGNv7Model tests PASSED ===")
