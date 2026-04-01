"""Holonomy noise floor test.

Phase 1a: transport is identity, so holonomy must be EXACTLY zero.
Phase 2: establishes epsilon noise floor for meaningful holonomy.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model import FGNModel
from fgn.transport import ParallelTransport, HolonomyDetector


def sample_triplets(seq_len: int, n_triplets: int) -> torch.Tensor:
    """Sample random non-degenerate triplets (i != j != k)."""
    triplets = []
    while len(triplets) < n_triplets:
        idx = torch.randperm(seq_len)[:3]
        if idx[0] != idx[1] and idx[1] != idx[2] and idx[0] != idx[2]:
            triplets.append(idx)
    return torch.stack(triplets)


def main():
    parser = argparse.ArgumentParser(description="Holonomy Noise Floor Test")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint (optional, uses random init if not provided)")
    parser.add_argument("--n_triplets", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        config = ckpt["config"]
        model = FGNModel(config).to(device)
        # Strip _orig_mod. prefix from torch.compile state dicts
        state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
        model.load_state_dict(state)
        print(f"Loaded model from {args.checkpoint}")
    else:
        config = FGNConfig(d_model=64, n_heads=4, n_layers=2, d_ff=256,
                           vocab_size=64, max_seq_len=32, use_torch_compile=False)
        model = FGNModel(config).to(device)
        print("Using randomly initialized model")

    # Get metric from a forward pass
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)
    with torch.no_grad():
        _ = model(input_ids)

    # Extract metric from first layer
    g = model.layers[0].last_metric  # [1, N, d]
    print(f"Metric shape: {g.shape}, mean={g.mean():.4f}, std={g.std():.4f}")

    # Create transport and holonomy detector
    transport = ParallelTransport(config).to(device)
    detector = HolonomyDetector().to(device)

    # Sample triplets
    triplets = sample_triplets(g.shape[1], args.n_triplets).to(device)

    # Measure holonomy
    with torch.no_grad():
        h_norms = detector(transport, g, triplets)

    print(f"\nHolonomy norms (n={args.n_triplets}):")
    print(f"  Mean:  {h_norms.mean().item():.6e}")
    print(f"  Std:   {h_norms.std().item():.6e}")
    print(f"  Max:   {h_norms.max().item():.6e}")
    print(f"  Min:   {h_norms.min().item():.6e}")

    # Phase 1a check: must be exactly zero
    if not transport.enabled:
        is_zero = torch.allclose(h_norms, torch.zeros_like(h_norms))
        print(f"\nPhase 1a identity check: {'PASS' if is_zero else 'FAIL'}")
        if not is_zero:
            print(f"  ERROR: Non-zero holonomy detected in identity transport!")
            sys.exit(1)
    else:
        print(f"\nPhase 2 epsilon noise floor: {h_norms.mean().item():.6e}")


if __name__ == "__main__":
    main()
