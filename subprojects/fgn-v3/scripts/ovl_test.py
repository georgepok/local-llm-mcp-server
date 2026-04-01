"""OVL separability test — measures curvature distribution overlap between tasks.

Computes Overlap Coefficient: OVL = integral min(f1(x), f2(x)) dx
Target: OVL(kappa_factual, kappa_narrative) < 0.3

Phase 1a.1: Uses NATURAL LANGUAGE data (factual vs narrative WikiText passages)
instead of synthetic recall/tracking sequences. The model was trained on natural
language, so curvature distributions should reflect learned geometry for actual
text, not synthetic patterns the model has never seen.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.model import FGNModel


def extract_curvatures(model: FGNModel, data: torch.Tensor,
                       device: torch.device, batch_size: int = 4) -> list:
    """Extract per-layer curvature distributions from data."""
    model.eval()
    all_curvatures = [[] for _ in range(len(model.layers))]

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device)
            _ = model(batch)
            for layer_idx, layer in enumerate(model.layers):
                if layer.last_curvature is not None:
                    all_curvatures[layer_idx].append(
                        layer.last_curvature.cpu().flatten()
                    )

    return [torch.cat(c) for c in all_curvatures]


def compute_ovl(dist1: torch.Tensor, dist2: torch.Tensor, n_bins: int = 200) -> float:
    """Compute overlap coefficient between two distributions."""
    all_vals = torch.cat([dist1, dist2])
    lo, hi = all_vals.quantile(0.01).item(), all_vals.quantile(0.99).item()
    if hi - lo < 1e-10:
        return 1.0  # Identical distributions
    bins = np.linspace(lo, hi, n_bins + 1)
    bin_width = (hi - lo) / n_bins

    h1, _ = np.histogram(dist1.numpy(), bins=bins, density=True)
    h2, _ = np.histogram(dist2.numpy(), bins=bins, density=True)

    ovl = np.sum(np.minimum(h1, h2)) * bin_width
    return float(ovl)


def get_natural_language_data(tokenizer, seq_len: int, n_sequences: int):
    """Extract factual and narrative passages from WikiText-103 validation set.

    Factual: passages containing dates, numbers, named entities (encyclopedic)
    Narrative: passages with dialogue markers or story-like flow
    """
    from datasets import load_dataset
    import re

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

    factual_ids = []
    narrative_ids = []

    # Concatenate all text and classify chunks
    all_texts = []
    for example in ds:
        text = example["text"].strip()
        if len(text) > 50:
            all_texts.append(text)

    # Concatenate into long strings and tokenize
    factual_text = []
    narrative_text = []

    for text in all_texts:
        # Factual: contains years, numbers, section headers
        has_dates = bool(re.search(r'\b(19|20)\d{2}\b', text))
        has_numbers = len(re.findall(r'\d+', text)) >= 5
        has_headers = text.startswith(' = ')

        # Narrative: contains pronouns, dialogue markers
        has_pronouns = len(re.findall(r'\b(he|she|they|him|her|his)\b', text, re.I)) >= 2
        has_dialogue = '"' in text

        if has_dates or has_numbers or has_headers:
            factual_text.append(text)
        elif has_pronouns or has_dialogue:
            narrative_text.append(text)

    # Tokenize and chunk
    factual_str = " ".join(factual_text)
    narrative_str = " ".join(narrative_text)

    factual_ids = tokenizer.encode(factual_str)
    narrative_ids = tokenizer.encode(narrative_str)

    # Chunk into sequences
    n_factual = min(len(factual_ids) // seq_len, n_sequences)
    n_narrative = min(len(narrative_ids) // seq_len, n_sequences)

    if n_factual < 10 or n_narrative < 10:
        print(f"WARNING: Insufficient data (factual={n_factual}, narrative={n_narrative})")
        print("Falling back to synthetic data")
        return None, None

    factual = torch.tensor(factual_ids[:n_factual * seq_len]).view(n_factual, seq_len)
    narrative = torch.tensor(narrative_ids[:n_narrative * seq_len]).view(n_narrative, seq_len)

    return factual, narrative


def generate_recall_data(vocab_size: int, seq_len: int, n_sequences: int) -> torch.Tensor:
    """Generate recall-style sequences (fallback synthetic)."""
    data = torch.randint(0, vocab_size - 1, (n_sequences, seq_len))
    data[:, seq_len // 2] = data[:, 0]
    return data


def generate_tracking_data(vocab_size: int, seq_len: int, n_sequences: int) -> torch.Tensor:
    """Generate tracking-style sequences (fallback synthetic)."""
    data = torch.zeros(n_sequences, seq_len, dtype=torch.long)
    for i in range(n_sequences):
        base = torch.randint(0, vocab_size // 4, (1,)).item()
        step = torch.randint(1, 4, (1,)).item()
        for j in range(seq_len):
            data[i, j] = (base + j * step) % (vocab_size - 1)
    return data


def main():
    parser = argparse.ArgumentParser(description="OVL Separability Test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_sequences", type=int, default=256)
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic data instead of natural language")
    parser.add_argument("--output", type=str, default=None, help="Save histogram plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = FGNModel(config).to(device)
    # Strip _orig_mod. prefix from torch.compile state dicts
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    print(f"Loaded model from {args.checkpoint}")

    if args.use_synthetic:
        print("\nUsing synthetic recall/tracking data")
        data_a = generate_recall_data(config.vocab_size, config.max_seq_len, args.n_sequences)
        data_b = generate_tracking_data(config.vocab_size, config.max_seq_len, args.n_sequences)
        label_a, label_b = "Recall", "Tracking"
    else:
        print("\nUsing natural language factual/narrative data from WikiText-103 validation")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        data_a, data_b = get_natural_language_data(
            tokenizer, config.max_seq_len, args.n_sequences)

        if data_a is None:
            print("Falling back to synthetic data")
            data_a = generate_recall_data(config.vocab_size, config.max_seq_len, args.n_sequences)
            data_b = generate_tracking_data(config.vocab_size, config.max_seq_len, args.n_sequences)
            label_a, label_b = "Recall", "Tracking"
        else:
            label_a, label_b = "Factual", "Narrative"
            print(f"  {label_a}: {len(data_a)} sequences")
            print(f"  {label_b}: {len(data_b)} sequences")

    # Extract curvatures
    print("Extracting curvatures...")
    curv_a = extract_curvatures(model, data_a, device)
    curv_b = extract_curvatures(model, data_b, device)

    # Compute OVL per layer
    print(f"\nOVL per layer ({label_a} vs {label_b}):")
    for layer_idx in range(len(model.layers)):
        ovl = compute_ovl(curv_a[layer_idx], curv_b[layer_idx])
        status = "PASS" if ovl < 0.3 else "FAIL"
        print(f"  Layer {layer_idx}: OVL = {ovl:.4f}  [{status}]")

    # Print curvature statistics for comparison
    print(f"\nCurvature statistics:")
    for layer_idx in range(len(model.layers)):
        a_mean = curv_a[layer_idx].abs().mean().item()
        b_mean = curv_b[layer_idx].abs().mean().item()
        a_std = curv_a[layer_idx].std().item()
        b_std = curv_b[layer_idx].std().item()
        print(f"  Layer {layer_idx}: {label_a} |κ|={a_mean:.6f} σ={a_std:.6f} | "
              f"{label_b} |κ|={b_mean:.6f} σ={b_std:.6f}")

    # Optional: save histogram plots
    if args.output:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, len(model.layers), figsize=(5 * len(model.layers), 4))
            if len(model.layers) == 1:
                axes = [axes]

            for i, ax in enumerate(axes):
                ax.hist(curv_a[i].numpy(), bins=100, alpha=0.5, label=label_a, density=True)
                ax.hist(curv_b[i].numpy(), bins=100, alpha=0.5, label=label_b, density=True)
                ovl = compute_ovl(curv_a[i], curv_b[i])
                ax.set_title(f"Layer {i} (OVL={ovl:.3f})")
                ax.legend()

            plt.tight_layout()
            plt.savefig(args.output, dpi=150)
            print(f"\nHistogram saved to {args.output}")
        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
