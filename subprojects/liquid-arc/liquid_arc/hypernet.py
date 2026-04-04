"""HyperNetwork for amortized test-time training (TTT V3 Experiment C).

Instead of running gradient-based TTT (100 steps), predict weight deltas
in a single forward pass from demo pair embeddings.

Architecture:
    TaskEncoder: pool demo pairs → task embedding [2d]
    LowRankHead (per melt module): task_embed → ΔW via low-rank factorization
    HyperNetwork: orchestrates TaskEncoder + per-module LowRankHeads

Training: distillation from gradient-based TTT target deltas.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LiquidARCConfig
from .model import LiquidARCModel


class LowRankHead(nn.Module):
    """Predict a weight delta ΔW = U @ diag(c) @ V * scale.

    Learned basis matrices U, V define the subspace.
    Coefficient predictor maps task embedding → mixing coefficients c.
    """

    def __init__(self, out_features: int, in_features: int, task_dim: int,
                 rank: int = 8, scale_init: float = 0.01):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank

        # Low-rank basis
        self.U = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        self.V = nn.Parameter(torch.randn(rank, in_features) * 0.02)

        # Coefficient predictor: task_embed → r coefficients
        self.coeff_net = nn.Sequential(
            nn.Linear(task_dim, 64),
            nn.GELU(),
            nn.Linear(64, rank),
        )

        # Learnable scale (log-space for stability, clamped to avoid NaN at 0)
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(scale_init, 1e-8))))

    def forward(self, task_embed: torch.Tensor) -> torch.Tensor:
        """Predict weight delta from task embedding.

        Args:
            task_embed: [task_dim] task representation

        Returns:
            delta_W: [out_features, in_features] weight delta
        """
        c = self.coeff_net(task_embed)  # [rank]
        scale = self.log_scale.exp()
        delta_W = (self.U * c.unsqueeze(0)) @ self.V * scale  # [out, in]
        return delta_W


class LowRankBiasHead(nn.Module):
    """Predict a bias delta Δb from task embedding."""

    def __init__(self, out_features: int, task_dim: int,
                 scale_init: float = 0.01):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(task_dim, 64),
            nn.GELU(),
            nn.Linear(64, out_features),
        )
        self.log_scale = nn.Parameter(torch.tensor(math.log(max(scale_init, 1e-8))))

    def forward(self, task_embed: torch.Tensor) -> torch.Tensor:
        """Predict bias delta.

        Args:
            task_embed: [task_dim]

        Returns:
            delta_b: [out_features]
        """
        return self.predictor(task_embed) * self.log_scale.exp()


class TaskEncoder(nn.Module):
    """Encode demo pairs into a fixed-size task representation.

    For each demo pair: mean-pool input cells → [d], mean-pool output cells → [d],
    concat → [2d]. Average across demo pairs → [2d]. Project via small MLP.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        task_dim = 2 * d_model
        self.proj = nn.Sequential(
            nn.LayerNorm(task_dim),
            nn.Linear(task_dim, task_dim),
            nn.GELU(),
            nn.Linear(task_dim, task_dim),
        )

    def forward(
        self,
        h0: torch.Tensor,
        roles: torch.Tensor,
        grid_ids: torch.Tensor,
        role_input_demo: int,
        role_output_demo: int,
    ) -> torch.Tensor:
        """Encode demos from embedded hidden states.

        Args:
            h0: [1, N, d] initial embeddings from base model
            roles: [1, N] role indices
            grid_ids: [1, N] grid IDs
            role_input_demo: int constant for input demo role
            role_output_demo: int constant for output demo role

        Returns:
            task_embed: [2d] task representation
        """
        h = h0.squeeze(0)  # [N, d]
        r = roles.squeeze(0)  # [N]

        # Pool input demo cells and output demo cells
        input_mask = (r == role_input_demo)
        output_mask = (r == role_output_demo)

        if input_mask.sum() > 0:
            input_pool = h[input_mask].mean(dim=0)  # [d]
        else:
            input_pool = torch.zeros(self.d_model, device=h.device, dtype=h.dtype)

        if output_mask.sum() > 0:
            output_pool = h[output_mask].mean(dim=0)  # [d]
        else:
            output_pool = torch.zeros(self.d_model, device=h.device, dtype=h.dtype)

        # Concat and project
        task_raw = torch.cat([input_pool, output_pool], dim=-1)  # [2d]
        return self.proj(task_raw)


def _get_melt_module_specs(
    model: LiquidARCModel,
    include_ffn: bool = False,
) -> List[Tuple[str, nn.Module]]:
    """Get the list of (name, module) pairs that TTT melts.

    Matches the melt_modules list in test_time_adapt().
    """
    specs = [
        ("metric_net_linear1", model.dynamics.metric_net_linear1),
        ("metric_net_linear2_diag", model.dynamics.metric_net_linear2_diag),
        ("tau_net_linear1", model.dynamics.tau_net_linear1),
        ("tau_net_linear2", model.dynamics.tau_net_linear2),
        ("W_o", model.dynamics.W_o),
    ]
    if include_ffn:
        specs.append(("ffn_out", model.dynamics.ffn[-1]))
    return specs


class HyperNetwork(nn.Module):
    """Amortized TTT: predict weight deltas from demo pairs in one forward pass.

    Contains a TaskEncoder and per-module LowRankHeads.
    """

    def __init__(self, config: LiquidARCConfig, base_model: LiquidARCModel):
        super().__init__()
        d = config.d_model
        task_dim = 2 * d
        rank = config.hypernet_rank
        scale_init = config.hypernet_scale_init

        self.task_encoder = TaskEncoder(d)

        # Create per-module heads
        self.weight_heads = nn.ModuleDict()
        self.bias_heads = nn.ModuleDict()

        specs = _get_melt_module_specs(base_model, include_ffn=config.hypernet_include_ffn)
        for name, module in specs:
            if hasattr(module, 'weight'):
                out_f, in_f = module.weight.shape
                self.weight_heads[name] = LowRankHead(
                    out_f, in_f, task_dim, rank=rank, scale_init=scale_init,
                )
            if hasattr(module, 'bias') and module.bias is not None:
                self.bias_heads[name] = LowRankBiasHead(
                    module.bias.shape[0], task_dim, scale_init=scale_init,
                )

        self._module_names = [name for name, _ in specs]

    def forward(self, task_embed: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Predict weight deltas for all melt modules.

        Args:
            task_embed: [2d] from TaskEncoder

        Returns:
            Dict mapping module_name → {"weight": ΔW, "bias": Δb (if exists)}
        """
        deltas = {}
        for name in self._module_names:
            d = {}
            if name in self.weight_heads:
                d["weight"] = self.weight_heads[name](task_embed)
            if name in self.bias_heads:
                d["bias"] = self.bias_heads[name](task_embed)
            deltas[name] = d
        return deltas

    def apply_deltas(
        self,
        adapted_model: LiquidARCModel,
        deltas: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        """Apply predicted deltas to an adapted model in-place.

        Args:
            adapted_model: deepcopy of base model to modify
            deltas: output of forward()
        """
        specs = _get_melt_module_specs(adapted_model,
                                        include_ffn="ffn_out" in deltas)
        name_to_module = dict(specs)

        for name, delta_dict in deltas.items():
            module = name_to_module[name]
            if "weight" in delta_dict:
                module.weight.data.add_(delta_dict["weight"])
            if "bias" in delta_dict and module.bias is not None:
                module.bias.data.add_(delta_dict["bias"])

    def encode_task(
        self,
        base_model: LiquidARCModel,
        meta: Dict[str, torch.Tensor],
        device: torch.device,
        role_input_demo: int,
        role_output_demo: int,
    ) -> torch.Tensor:
        """Run base model embedding + task encoding.

        Args:
            base_model: pre-trained model (not modified)
            meta: padded batch metadata from pad_single_to_batch()
            device: torch device
            role_input_demo: ROLE_INPUT_DEMO constant
            role_output_demo: ROLE_OUTPUT_DEMO constant

        Returns:
            task_embed: [2d] task representation
        """
        with torch.no_grad():
            # Get h0 from base model embedding
            colors_masked = meta["colors"].clone()
            target_input_colors = meta.get("target_input_colors")
            if target_input_colors is not None:
                colors_masked[meta["target_mask"]] = target_input_colors[meta["target_mask"]]
            else:
                from .model import PAD_COLOR as _PAD
                colors_masked[meta["target_mask"]] = _PAD

            h0 = base_model.embedding(
                colors_masked,
                meta["xs"], meta["ys"], meta["roles"],
                meta["sep_mask"], meta["sep_types"],
                grid_ids=meta.get("grid_ids"),
            )

        # Task encoder runs with gradients (for training the hypernet)
        return self.task_encoder(
            h0, meta["roles"], meta.get("grid_ids", torch.zeros_like(meta["roles"])),
            role_input_demo, role_output_demo,
        )


if __name__ == "__main__":
    print("Testing HyperNetwork...")

    config = LiquidARCConfig(
        d_model=64, d_metric=16, d_ffn=128, max_seq_len=128,
        n_ode_steps=4, hypernet_rank=4, hypernet_scale_init=0.01,
        hypernet_include_ffn=True,
    )
    model = LiquidARCModel(config)

    hypernet = HyperNetwork(config, model)
    n_hyper_params = sum(p.numel() for p in hypernet.parameters())
    print(f"  HyperNetwork params: {n_hyper_params:,}")

    # Test forward
    task_embed = torch.randn(2 * 64)
    deltas = hypernet(task_embed)
    print(f"  Modules with deltas: {list(deltas.keys())}")
    for name, d in deltas.items():
        shapes = {k: v.shape for k, v in d.items()}
        print(f"    {name}: {shapes}")

    # Test apply
    import copy
    adapted = copy.deepcopy(model)
    hypernet.apply_deltas(adapted, deltas)
    print("  apply_deltas: OK")

    # Verify weights changed
    for name in deltas:
        specs = _get_melt_module_specs(model, include_ffn=True)
        name_to_mod_orig = dict(specs)
        specs_adapted = _get_melt_module_specs(adapted, include_ffn=True)
        name_to_mod_adapted = dict(specs_adapted)
        diff = (name_to_mod_adapted[name].weight.data - name_to_mod_orig[name].weight.data).abs().sum()
        print(f"    {name} weight diff: {diff.item():.6f}")

    print("HyperNetwork OK")
