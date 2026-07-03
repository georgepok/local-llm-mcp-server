"""Sanity check for Tier 3 halting: at init p_halt~0 → still_active~1 → no-op."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fgn.config import FGNConfig
from fgn.liquid_model import LiquidSequenceModel


def main():
    cfg = FGNConfig.from_yaml("configs/tr_liquid_stepcond_halt.yaml")
    cfg.vocab_size = 50304
    cfg.max_seq_len = 512  # smaller for quick test

    torch.manual_seed(42)
    m = LiquidSequenceModel(cfg).cuda().eval()
    n = sum(p.numel() for p in m.parameters())
    dyn = m.dynamics._orig_mod if hasattr(m.dynamics, '_orig_mod') else m.dynamics

    print(f"Total params: {n:,}")
    print(f"halting_enabled:          {dyn.halting_enabled}")
    print(f"step_conditional_enabled: {dyn.step_conditional_enabled}")
    print(f"n_ode_steps (MAX budget): {m.n_ode_steps}")
    print(f"halting_min_steps:        {m.la_cfg.halting_min_steps}")
    print(f"halt_head bias:           {dyn.halt_head.bias.item():.3f}")
    print(f"halt_head weight max:     {dyn.halt_head.weight.abs().max().item():.3e}")

    torch.manual_seed(42)
    ids = torch.randint(0, 50304, (2, 128), device="cuda")
    labels = torch.full((2, 128), -100, dtype=torch.long, device="cuda")
    labels[:, -1] = 5

    with torch.no_grad():
        out = m(ids, labels=labels)

    print(f"\nOutput keys: {list(out.keys())}")
    print(f"ce_loss:   {out['ce_loss'].item():.4f}")
    print(f"loss:      {out['loss'].item():.4f}")
    if 'ponder_cost' in out:
        pc = out['ponder_cost'].item()
        pl = out['ponder_loss'].item()
        print(f"ponder_cost: {pc:.4f}  (should be ~0.87+ at init: first 4 "
              f"steps active=1, last 28 steps ~ decay by (1-4e-5)^i ≈ 1)")
        print(f"ponder_loss: {pl:.6f}")
        # Roughly: ponder_cost = mean(still_active across steps)
        # For min_steps=4, still_active stays 1 for those. After step 4,
        # it decays by (1 - sigmoid(-10))^i ≈ (1 - 4.5e-5)^i. Essentially 1.
        # Expected ponder ≈ 1.0 at init.
        if 0.95 < pc <= 1.01:
            print("PASS: ponder cost near 1.0 at init — halting is no-op (correct)")
        else:
            print(f"UNEXPECTED: ponder cost {pc:.4f} — halt_head may not be no-op init")


if __name__ == "__main__":
    main()
