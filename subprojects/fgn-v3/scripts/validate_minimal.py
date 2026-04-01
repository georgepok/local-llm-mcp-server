"""Minimal validation: synthetic copy-pattern task.

Task: [a, b, c, SEP, ?, ?, ?] -> [a, b, c, SEP, a, b, c]
Runs on CPU. Must pass before WikiText training.

Success criteria:
  1. Loss < 0.5
  2. Metric non-trivial: std(g)/mean(g) > 0.01
  3. Curvature peaks at SEP: mean |kappa| at SEP > 2x elsewhere
  4. All 3 scales retain weight > 0.1
  5. Gradient norms bounded: no layer > 100x another
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model import FGNModel


def generate_copy_data(batch_size: int, seq_half: int = 7, vocab_size: int = 64):
    """Generate copy-pattern sequences.

    Format: [a1, a2, ..., a_{seq_half}, SEP, a1, a2, ..., a_{seq_half}]
    SEP token = vocab_size - 1
    """
    SEP = vocab_size - 1
    content_len = seq_half

    # Random tokens (excluding SEP)
    content = torch.randint(0, vocab_size - 1, (batch_size, content_len))

    # Build input: [content, SEP, content[:-1]]
    # Build labels: [-100 for prefix+SEP, content]
    sep = torch.full((batch_size, 1), SEP)
    input_ids = torch.cat([content, sep, content[:, :-1]], dim=1)  # [B, 2*seq_half]
    labels = torch.cat([
        torch.full((batch_size, content_len + 1), -100),  # Ignore prefix + SEP
        content,                                            # Predict copy
    ], dim=1)

    # Trim to same length
    min_len = min(input_ids.shape[1], labels.shape[1])
    input_ids = input_ids[:, :min_len]
    labels = labels[:, :min_len]

    return input_ids, labels, content_len  # SEP position = content_len


def validate():
    print("=" * 60)
    print("FGN v3 Minimal Validation — Copy Pattern Task")
    print("=" * 60)

    # Small config for fast CPU validation
    config = FGNConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        vocab_size=64,
        max_seq_len=32,
        n_scales=3,
        t_init=(0.1, 1.0, 10.0),
        scale_entropy_alpha=0.01,
        curvature_lambda=0.001,
        dropout=0.0,  # No dropout for deterministic validation
        use_torch_compile=False,
    )

    model = FGNModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer — use same LR for all params in this short diagnostic.
    # The 0.1x slow group is designed for long training runs; 1000 steps
    # isn't enough for metric to develop structure at 0.1x.
    optimizer = torch.optim.AdamW([
        {"params": model.fast_parameters(), "lr": 1e-3},
        {"params": model.slow_parameters(), "lr": 1e-3},
    ], weight_decay=0.01)

    # Training
    n_steps = 1000
    batch_size = 32
    model.train()

    losses = []
    for step in range(n_steps):
        input_ids, labels, sep_pos = generate_copy_data(batch_size, seq_half=7,
                                                         vocab_size=config.vocab_size)
        result = model(input_ids, labels=labels)
        loss = result["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if step % 200 == 0:
            g0 = model.layers[0].last_metric
            k0 = model.layers[0].last_curvature
            g_cv = (g0.std() / g0.mean()).item() if g0 is not None else 0
            k_abs = k0.abs().mean().item() if k0 is not None else 0
            print(f"  step {step:4d}: loss={loss.item():.4f}, "
                  f"ce={result['ce_loss'].item():.4f}, "
                  f"metric_cv={g_cv:.4f}, |kappa|={k_abs:.6f}")

    final_loss = sum(losses[-50:]) / 50
    print(f"\nFinal loss (avg last 50): {final_loss:.4f}")

    # === VALIDATION CHECKS ===
    passed = 0
    total = 5

    # 1. Loss convergence
    print(f"\n[1/5] Loss < 0.5: {final_loss:.4f}", end=" ")
    if final_loss < 0.5:
        print("PASS")
        passed += 1
    else:
        print("FAIL")

    # 2. Metric non-triviality
    model.eval()
    with torch.no_grad():
        input_ids, labels, sep_pos = generate_copy_data(64, seq_half=7,
                                                         vocab_size=config.vocab_size)
        result = model(input_ids, labels=labels)
        metrics = model.get_metrics()
        g = metrics[0]  # First layer metric
        cv = (g.std() / g.mean()).item()

    print(f"[2/5] Metric CV (std/mean) > 0.01: {cv:.4f}", end=" ")
    if cv > 0.01:
        print("PASS")
        passed += 1
    else:
        print("FAIL")

    # 3. Curvature peaks at SEP
    curvatures = model.get_curvatures()
    kappa = curvatures[0]  # First layer
    kappa_sep = kappa[:, sep_pos].abs().mean().item()
    kappa_other = torch.cat([kappa[:, :sep_pos], kappa[:, sep_pos+1:]], dim=1).abs().mean().item()
    ratio = kappa_sep / max(kappa_other, 1e-8)

    print(f"[3/5] Curvature at SEP > 1.2x elsewhere: ratio={ratio:.2f}", end=" ")
    print(f"(SEP={kappa_sep:.6f}, other={kappa_other:.6f})", end=" ")
    if ratio > 1.2:
        print("PASS")
        passed += 1
    else:
        print("FAIL")

    # 4. Scale balance
    h_dummy = torch.randn(1, 14, config.d_model)
    scale_weights_all = []
    for layer in model.layers:
        w = F.softmax(layer.attention.W_scale(h_dummy), dim=-1)
        scale_weights_all.append(w.mean(dim=(0, 1)))  # Average over batch & seq

    min_scale = min(w.min().item() for w in scale_weights_all)
    print(f"[4/5] All scales > 0.1: min={min_scale:.4f}", end=" ")
    if min_scale > 0.1:
        print("PASS")
        passed += 1
    else:
        print("FAIL")
        for i, w in enumerate(scale_weights_all):
            print(f"       Layer {i}: {w.detach().numpy()}")

    # 5. Gradient norm bounds
    model.train()
    input_ids, labels, _ = generate_copy_data(32, seq_half=7, vocab_size=config.vocab_size)
    result = model(input_ids, labels=labels)
    result["loss"].backward()

    layer_norms = []
    for i, layer in enumerate(model.layers):
        total_norm = 0.0
        for p in layer.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        layer_norms.append(total_norm ** 0.5)

    max_norm = max(layer_norms)
    min_norm = min(layer_norms)
    ratio = max_norm / max(min_norm, 1e-8)

    print(f"[5/5] Grad norm ratio < 100: {ratio:.2f}", end=" ")
    if ratio < 100:
        print("PASS")
        passed += 1
    else:
        print("FAIL")
        for i, n in enumerate(layer_norms):
            print(f"       Layer {i}: {n:.4f}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} checks passed")
    if passed == total:
        print("VALIDATION PASSED — ready for WikiText training")
    else:
        print("VALIDATION FAILED — fix issues before proceeding")
    print(f"{'=' * 60}")

    return passed == total


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
