"""End-to-end smoke: plug LiquidARCPredictor into the real JEPA wrapper
(upstream `jepa.py`) and verify a forward + backward pass works with the
LeWM training signature — without needing the full Lightning/Hydra harness
or PushT data.

Run:
    cd subprojects/lewm-integration
    PYTHONPATH=.:../liquid-arc:./le-wm python tests/test_jepa_integration.py
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'le-wm'))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), 'liquid-arc'))

from jepa import JEPA                        # noqa: E402
from module import Embedder, MLP             # noqa: E402
from liquid_arc.config import LiquidARCConfig  # noqa: E402
from liquid_arc_lewm import LiquidARCPredictor  # noqa: E402


class DummyEncoder(torch.nn.Module):
    """Stands in for ViT-tiny in the smoke path (avoids HF download)."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, hidden_dim, kernel_size=16, stride=16)
        self.hidden_dim = hidden_dim

        class _Cfg:
            pass
        self.config = _Cfg()
        self.config.hidden_size = hidden_dim

    def forward(self, x, **_):
        # x: (BT, 3, H, W). Emit a namespace with last_hidden_state [BT, N, D].
        feats = self.conv(x)                       # (BT, D, h', w')
        feats = feats.flatten(2).transpose(1, 2)   # (BT, N, D)
        cls = feats.mean(dim=1, keepdim=True)      # fake CLS
        feats = torch.cat([cls, feats], dim=1)

        class _Out:
            pass
        out = _Out()
        out.last_hidden_state = feats
        return out


def test_jepa_forward_backward_with_liquid_predictor():
    torch.manual_seed(0)
    B, T = 2, 4                       # batch, frames
    action_dim = 5
    frameskip = 3
    hidden_dim = 64
    embed_dim = 32
    H = W = 64                        # dummy image size

    encoder = DummyEncoder(hidden_dim=hidden_dim)
    ode_cfg = LiquidARCConfig(
        d_model=embed_dim, d_metric=16, d_ffn=64,
        n_ode_steps=4, ode_steps_min=4, ode_steps_max=4,
        integration_time=1.0, use_torch_compile=False,
    )
    predictor = LiquidARCPredictor(
        input_dim=embed_dim, action_emb_dim=embed_dim,
        ode_config=ode_cfg, output_dim=hidden_dim,
    )
    action_encoder = Embedder(input_dim=frameskip * action_dim, emb_dim=embed_dim)
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=128, norm_fn=torch.nn.BatchNorm1d)
    pred_proj = MLP(input_dim=hidden_dim, output_dim=embed_dim,
                    hidden_dim=128, norm_fn=torch.nn.BatchNorm1d)
    jepa = JEPA(encoder=encoder, predictor=predictor,
                action_encoder=action_encoder,
                projector=projector, pred_proj=pred_proj)

    pixels = torch.randn(B, T, 3, H, W)
    action = torch.randn(B, T, frameskip * action_dim)

    info = {"pixels": pixels, "action": action}
    info = jepa.encode(info)
    ctx_len = 3
    ctx_emb = info["emb"][:, :ctx_len]
    ctx_act = info["act_emb"][:, :ctx_len]
    pred = jepa.predict(ctx_emb, ctx_act)
    tgt = info["emb"][:, 1:]                                # upstream n_preds=1 shift
    assert pred.shape == tgt.shape == (B, ctx_len, embed_dim), (pred.shape, tgt.shape)

    loss = (pred - tgt).pow(2).mean()
    loss.backward()
    n_with_grad = sum(1 for p in jepa.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_params = sum(1 for _ in jepa.parameters())
    assert n_with_grad > n_params * 0.7, f"only {n_with_grad}/{n_params} params got grad"
    print(f"JEPA end-to-end OK: {n_with_grad}/{n_params} params received gradient")


if __name__ == "__main__":
    test_jepa_forward_backward_with_liquid_predictor()
    print("jepa-integration smoke passed")
