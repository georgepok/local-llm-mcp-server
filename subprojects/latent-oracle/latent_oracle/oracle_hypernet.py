"""Oracle HyperNet — predicts task-specific W_o deltas from oracle embeddings.

Amortizes what TTT does in 100 gradient steps into a single forward pass.
Oracle [B, oracle_dim] -> ProjectionHead -> z_context [B, d_model]
-> mean-pool -> OracleHyperNet -> delta_W_o [d_model, d_model]

The delta is applied functionally in dynamics.py:
    update = F.linear(routed_v, W_o.weight + delta_W_o, None)

Total new params: ~226K (adapter ~197K + LowRankHead ~29K) at d=768, task_dim=256, rank=8.
No auxiliary loss — CE backprops through delta_W_o into the hypernet.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Import LowRankHead from liquid-arc (sibling subproject)
_LIQUID_ARC_ROOT = str(Path(__file__).resolve().parent.parent.parent / "liquid-arc")
if _LIQUID_ARC_ROOT not in sys.path:
    sys.path.insert(0, _LIQUID_ARC_ROOT)

from liquid_arc.hypernet import LowRankHead


class OracleHyperNet(nn.Module):
    """Lightweight module predicting low-rank W_o weight deltas from oracle context.

    Architecture:
        z_context [B, d_model] -> mean(dim=0) -> [d_model]
        -> adapter: Linear(d_model, task_dim) + GELU  (~197K params at d=768, task_dim=256)
        -> LowRankHead(rank=8): U[d,8] @ diag(c) @ V[8,d]  (~29K params)
        -> delta_W_o [d_model, d_model]

    Args:
        d_model: model hidden dimension (768 for 5M model)
        task_dim: bottleneck dimension for task embedding (256)
        rank: rank of low-rank W_o delta (8)
        scale_init: initial scale for delta magnitude (0.01)
    """

    def __init__(self, d_model: int = 768, task_dim: int = 256,
                 rank: int = 8, scale_init: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.task_dim = task_dim

        # Adapter: compress d_model -> task_dim
        self.adapter = nn.Sequential(
            nn.Linear(d_model, task_dim),
            nn.GELU(),
        )

        # Low-rank head: task_dim -> delta_W_o [d_model, d_model]
        self.w_o_head = LowRankHead(
            out_features=d_model,
            in_features=d_model,
            task_dim=task_dim,
            rank=rank,
            scale_init=scale_init,
        )

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        """Predict task-specific W_o delta from oracle-projected context.

        Args:
            z_context: [B, d_model] oracle-projected context vectors

        Returns:
            delta_W_o: [d_model, d_model] weight delta for dynamics W_o
        """
        # Mean-pool across batch to get single task descriptor
        task_embed = self.adapter(z_context.mean(dim=0))  # [task_dim]
        return self.w_o_head(task_embed)  # [d_model, d_model]


if __name__ == "__main__":
    print("Testing OracleHyperNet...")

    d_model = 768
    task_dim = 256
    rank = 8
    scale_init = 0.01

    hypernet = OracleHyperNet(d_model, task_dim, rank, scale_init)
    n_params = sum(p.numel() for p in hypernet.parameters())
    print(f"  Parameters: {n_params:,}")

    # Shape test
    B = 4
    z_context = torch.randn(B, d_model)
    delta = hypernet(z_context)
    assert delta.shape == (d_model, d_model), f"Expected ({d_model}, {d_model}), got {delta.shape}"
    print(f"  delta_W_o shape: {delta.shape}")

    # Delta scale at init should be small
    delta_norm = delta.norm().item()
    delta_max = delta.abs().max().item()
    print(f"  delta norm: {delta_norm:.6f}, max: {delta_max:.6f}")
    assert delta_max < 1.0, f"Initial delta too large: max={delta_max}"

    # Gradient flow
    z_ctx_g = torch.randn(B, d_model, requires_grad=True)
    delta_g = hypernet(z_ctx_g)
    delta_g.sum().backward()
    assert z_ctx_g.grad is not None, "No gradient to z_context"
    print(f"  Gradient flows to z_context: OK")

    # Gradient to all hypernet params
    for name, p in hypernet.named_parameters():
        assert p.grad is not None, f"No gradient to {name}"
    print(f"  Gradient flows to all params: OK")

    # Different inputs should produce different deltas
    z1 = torch.randn(B, d_model)
    z2 = torch.randn(B, d_model) + 5.0
    d1 = hypernet(z1)
    d2 = hypernet(z2)
    diff = (d1 - d2).abs().sum().item()
    assert diff > 0, "Different inputs should produce different deltas"
    print(f"  Different inputs → different deltas: OK (diff={diff:.6f})")

    # Small model test (for CI)
    small_net = OracleHyperNet(d_model=64, task_dim=32, rank=4, scale_init=0.01)
    z_small = torch.randn(2, 64)
    d_small = small_net(z_small)
    assert d_small.shape == (64, 64)
    print(f"  Small model (d=64): OK")

    print("OracleHyperNet OK")
