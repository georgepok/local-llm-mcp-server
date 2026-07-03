"""Smoke tests for LiquidARCPredictor.

Run locally (CPU OK):
    cd subprojects/lewm-integration
    PYTHONPATH=..:../liquid-arc python -m pytest tests/test_predictor.py -s
or
    PYTHONPATH=..:../liquid-arc python tests/test_predictor.py
"""

import os
import sys

import torch

# Allow running as a script from subprojects/lewm-integration/.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), 'liquid-arc'))

from liquid_arc.config import LiquidARCConfig  # noqa: E402
from liquid_arc_lewm import LiquidARCPredictor  # noqa: E402


def _cfg(d_model=64):
    return LiquidARCConfig(
        d_model=d_model, d_metric=16, d_ffn=128,
        n_ode_steps=4, ode_steps_min=4, ode_steps_max=4,
        integration_time=1.0, use_torch_compile=False,
    )


def test_shape_matches_arpredictor():
    torch.manual_seed(0)
    B, T, D, A = 2, 5, 32, 8
    pred = LiquidARCPredictor(input_dim=D, action_emb_dim=A, ode_config=_cfg())
    emb = torch.randn(B, T, D)
    act = torch.randn(B, T, A)
    out = pred(emb, act)
    assert out.shape == (B, T, D), out.shape

    # Asymmetric: in=192, out=384 (matches LeWM embed_dim→hidden_dim)
    pred2 = LiquidARCPredictor(input_dim=D, action_emb_dim=A,
                               ode_config=_cfg(), output_dim=2 * D)
    out2 = pred2(emb, act)
    assert out2.shape == (B, T, 2 * D), out2.shape


def test_backward_and_grads_flow():
    torch.manual_seed(0)
    B, T, D, A = 2, 4, 16, 4
    pred = LiquidARCPredictor(latent_dim=D, action_emb_dim=A, ode_config=_cfg(d_model=32))
    emb = torch.randn(B, T, D, requires_grad=True)
    act = torch.randn(B, T, A, requires_grad=True)
    out = pred(emb, act)
    loss = out.pow(2).mean()
    loss.backward()
    assert emb.grad is not None and emb.grad.abs().sum().item() > 0
    assert act.grad is not None and act.grad.abs().sum().item() > 0
    # MetricNet must receive gradient (geometry is learned)
    mw = pred.dynamics.metric_net_linear1.weight.grad
    assert mw is not None and mw.abs().sum().item() > 0


def test_causality_future_does_not_leak():
    """Position t<T-1 output must not depend on emb[:, T-1] (last position)."""
    torch.manual_seed(0)
    B, T, D, A = 1, 6, 16, 4
    pred = LiquidARCPredictor(latent_dim=D, action_emb_dim=A, ode_config=_cfg(d_model=32))
    pred.eval()
    emb = torch.randn(B, T, D)
    act = torch.randn(B, T, A)
    with torch.no_grad():
        out_a = pred(emb, act)
        emb_perturb = emb.clone()
        emb_perturb[:, -1] += 1000.0   # huge change at the last position
        out_b = pred(emb_perturb, act)
    diff = (out_a[:, :-1] - out_b[:, :-1]).abs().max().item()
    # Action context is mean-pooled over T → action path leaks globally.
    # To isolate spatial causality we pass identical actions and only
    # perturb the *embedding* at the last position; that perturbation must
    # NOT affect positions [0..T-2].
    assert diff < 1e-4, f"causality violated: max diff on past positions = {diff}"


if __name__ == "__main__":
    test_shape_matches_arpredictor()
    print("shape OK")
    test_backward_and_grads_flow()
    print("grads OK")
    test_causality_future_does_not_leak()
    print("causality OK")
    print("all smoke tests passed")
