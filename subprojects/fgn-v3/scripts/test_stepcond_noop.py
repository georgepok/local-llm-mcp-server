"""Sanity test: Tier 1 step-conditional operator at init is a no-op.

Builds two LiquidSequenceModel instances from the same seed:
  A) step_conditional_operator=False (baseline)
  B) step_conditional_operator=True (Tier 1)

Both use identical RNG for weight init. With γ=1, β=0, t_diff[s]=global init,
forward outputs should agree to within floating-point noise (~1e-5).

If diff > 1e-4, something in the FiLM wiring isn't actually no-op — either
γ or β is wrong, or t_diff_per_step init value is off, or an extra op is
being performed.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fgn.config import FGNConfig
from fgn.liquid_model import LiquidSequenceModel


def build(step_cond: bool, seed: int = 0):
    torch.manual_seed(seed)
    cfg = FGNConfig(
        d_model=256, n_heads=8, n_layers=6, d_ff=1024,
        vocab_size=50304, max_seq_len=512,
        model_type="liquid", use_torch_compile=False,
        liquid_routing="attention", metric_rank=0,
        n_ode_steps=16,
        liquid_tau_min=0.5, liquid_tau_max=1.0,
    )
    # Inject the step-conditional flag into la_cfg via getattr path
    cfg_dict = cfg.__dict__.copy()
    cfg_dict['step_conditional_operator'] = step_cond
    cfg_dict['step_conditional_n_max'] = 32
    # Monkey-assign; FGNConfig accepts arbitrary dict for getattr lookups
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)
    m = LiquidSequenceModel(cfg).cuda().eval()
    return m


def main():
    torch.manual_seed(42)
    ids = torch.randint(0, 50304, (2, 128), device='cuda')

    m_base = build(False, seed=0)
    m_cond = build(True, seed=0)

    # Seed replay doesn't match because step-cond model allocates extra
    # Embeddings that consume RNG state. Instead, copy baseline weights
    # into the shared subset of cond model — guarantees matched shared weights.
    base_dict = dict(m_base.named_parameters())
    cond_dict = dict(m_cond.named_parameters())
    copied = 0
    with torch.no_grad():
        for name, p in cond_dict.items():
            if name in base_dict and base_dict[name].shape == p.shape:
                p.copy_(base_dict[name])
                copied += 1
    print(f"Copied {copied}/{len(cond_dict)} shared params from baseline")

    with torch.no_grad():
        out_base = m_base(ids)
        out_cond = m_cond(ids)

    diff = (out_base['logits'] - out_cond['logits']).abs().max().item()
    rel = diff / (out_base['logits'].abs().max().item() + 1e-12)
    print(f"Max abs diff (logits): {diff:.3e}")
    print(f"Max rel diff: {rel:.3e}")

    # Count extra params in step-cond model
    base_n = sum(p.numel() for p in m_base.parameters())
    cond_n = sum(p.numel() for p in m_cond.parameters())
    print(f"Baseline params:       {base_n:,}")
    print(f"Step-cond params:      {cond_n:,}")
    print(f"New params:            {cond_n - base_n:,} "
          f"({(cond_n-base_n)/base_n*100:.3f}% increase)")

    if diff < 1e-4:
        print("PASS: step-conditional operator is no-op at init")
    else:
        print("FAIL: non-trivial difference — FiLM wiring is wrong")


if __name__ == "__main__":
    main()
