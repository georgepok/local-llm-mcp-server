"""Quick smoke test for FGN v4 architecture."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from fgn.config import FGNConfig
from fgn.model_v4 import FGNv4Model

cfg = FGNConfig(d_model=64, n_heads=4, d_ff=256, n_layers=2,
                vocab_size=100, max_seq_len=32, geo_heads=1,
                architecture_version="v4", gate_init=3.0,
                curvature_lambda=0.0, curvature_reward_mu=0.0)
model = FGNv4Model(cfg)

B, N = 2, 16
ids = torch.randint(0, 100, (B, N))
labels = torch.randint(0, 100, (B, N))

# Forward + backward
result = model(ids, labels=labels)
print(f"loss={result['loss'].item():.4f}, ce={result['ce_loss'].item():.4f}")
print(f"gate={result['avg_gate'].item():.4f}, cv={result['metric_cv'].item():.4f}")
print(f"|kappa|={result['avg_kappa'].item():.4f}")

result["loss"].backward()
print("Gradient flow: OK")

# Phase 0 mode
model.freeze_attention()
model.force_gate(10.0)
model.zero_grad()
r2 = model(ids, labels=labels)
r2["loss"].backward()
print(f"Phase 0: loss={r2['loss'].item():.4f}, gate={r2['avg_gate'].item():.4f}")

# Phase 1 mode
model.unfreeze_attention()
model.init_gate(3.0)
model.zero_grad()
r3 = model(ids, labels=labels)
r3["loss"].backward()
print(f"Phase 1: loss={r3['loss'].item():.4f}, gate={r3['avg_gate'].item():.4f}")

# Per-layer gate grads
for i, layer in enumerate(model.layers):
    g = layer.gate_geo_raw.grad
    print(f"  Layer {i}: gate_grad={g.item():.6f}")

# Parameter count
n = sum(p.numel() for p in model.parameters())
geo = sum(p.numel() for p in model.geo_parameters())
attn = sum(p.numel() for p in model.attn_parameters())
other = sum(p.numel() for p in model.other_parameters())
print(f"Parameters: {n:,} (geo={geo:,}, attn={attn:,}, other={other:,})")
assert geo + attn + other == n, f"Split mismatch: {geo}+{attn}+{other} != {n}"

print("\nFGNv4 smoke test: PASS")
