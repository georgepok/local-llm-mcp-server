"""Position-resolved curvature diagnostic — tests the Resolution Thesis.

Does curvature correlate with linguistic structure WITHIN sequences?

The OVL test measures distributional overlap (aggregating across positions),
which misses positional structure. The Resolution Thesis predicts that
curvature should peak at structural boundaries — clause breaks, sentence
boundaries, punctuation, word onsets.

This script computes per-position curvature and correlates it with multiple
structural boundary signals:
  1. Punctuation (., , ; : ! ? — )
  2. Word boundaries (GPT-2 BPE tokens starting with space)
  3. Sentence boundaries (period/!/?  followed by capitalized token)
  4. Model prediction entropy (high entropy = structural uncertainty)

Reports Spearman rank correlation per layer with bootstrap confidence
intervals and permutation-based significance testing.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.model import FGNModel


# --- Boundary detection heuristics ---

PUNCT_CHARS = set('.,;:!?—–-()[]{}"\'/…')


def compute_boundary_scores(token_ids, tokenizer):
    """Compute multiple structural boundary signals for a token sequence.

    Args:
        token_ids: 1D array/list of token IDs
        tokenizer: GPT-2 tokenizer

    Returns:
        dict of {signal_name: np.array of shape [N]} with binary or
        continuous scores per position.
    """
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    N = len(tokens)

    # 1. Punctuation: is this token punctuation?
    punct = np.zeros(N, dtype=np.float32)
    for i, tok in enumerate(tokens):
        stripped = tok.strip()
        if stripped and all(c in PUNCT_CHARS for c in stripped):
            punct[i] = 1.0

    # 2. Word boundary: GPT-2 BPE encodes word-initial tokens with leading
    #    space (Ġ in the vocab). Detecting these marks word onsets.
    word_boundary = np.zeros(N, dtype=np.float32)
    for i, tid in enumerate(token_ids):
        tok_str = tokenizer.convert_ids_to_tokens(int(tid))
        if tok_str and tok_str.startswith('Ġ'):
            word_boundary[i] = 1.0

    # 3. Sentence boundary: punctuation followed by space+capital letter
    sent_boundary = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        prev = tokens[i - 1].strip()
        curr = tokens[i]
        if prev and prev[-1] in '.!?' and curr and curr.lstrip().islower() is False:
            # Check for actual capital letter
            curr_stripped = curr.lstrip()
            if curr_stripped and curr_stripped[0].isupper():
                sent_boundary[i] = 1.0

    # 4. Combined structural score: weighted sum
    #    (used for a single aggregate correlation)
    combined = 0.5 * punct + 0.3 * word_boundary + 1.0 * sent_boundary

    return {
        'punctuation': punct,
        'word_boundary': word_boundary,
        'sentence_boundary': sent_boundary,
        'combined': combined,
    }


def compute_prediction_entropy(model, input_ids, device):
    """Compute model prediction entropy at each position.

    High entropy = model is uncertain = likely structural transition.

    Args:
        model: FGNModel
        input_ids: [B, N] tensor

    Returns:
        [B, N] numpy array of per-position entropy
    """
    with torch.no_grad():
        result = model(input_ids.to(device))
        logits = result['logits']  # [B, N, V]
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [B, N]
    return entropy.cpu().numpy()


def extract_position_curvatures(model, data, device, batch_size=2):
    """Extract per-position curvatures preserving sequence structure.

    Unlike OVL's extract_curvatures which flattens everything, this
    preserves the [B, N] shape so we can correlate with position.

    Returns:
        list of n_layers arrays, each [total_sequences, seq_len]
    """
    model.eval()
    n_layers = len(model.layers)
    all_curvatures = [[] for _ in range(n_layers)]
    all_entropies = []

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size].to(device)
            result = model(batch)

            # Collect per-position curvature from each layer
            for layer_idx, layer in enumerate(model.layers):
                if layer.last_curvature is not None:
                    # Shape: [B, N] — keep both dims
                    all_curvatures[layer_idx].append(
                        layer.last_curvature.cpu().numpy()
                    )

            # Prediction entropy
            logits = result['logits']
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            all_entropies.append(entropy.cpu().numpy())

    curvatures = [np.concatenate(c, axis=0) for c in all_curvatures]
    entropies = np.concatenate(all_entropies, axis=0)
    return curvatures, entropies


def correlate_with_permutation_test(curvature, boundary, n_permutations=1000):
    """Spearman correlation with permutation-based p-value.

    The permutation test is more robust than the asymptotic p-value
    for non-independent (positionally structured) data.

    Returns:
        (rho, p_value, ci_low, ci_high)
    """
    rho, _ = stats.spearmanr(curvature, boundary)

    # Permutation test: shuffle curvature, recompute correlation
    null_rhos = np.zeros(n_permutations)
    for j in range(n_permutations):
        perm = np.random.permutation(len(curvature))
        null_rhos[j], _ = stats.spearmanr(curvature[perm], boundary)

    p_value = np.mean(np.abs(null_rhos) >= np.abs(rho))

    # Bootstrap 95% CI on the real correlation
    n_boot = 1000
    boot_rhos = np.zeros(n_boot)
    n = len(curvature)
    for j in range(n_boot):
        idx = np.random.randint(0, n, n)
        boot_rhos[j], _ = stats.spearmanr(curvature[idx], boundary[idx])
    ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])

    return rho, p_value, ci_low, ci_high


def main():
    parser = argparse.ArgumentParser(
        description="Position-resolved curvature diagnostic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_sequences", type=int, default=100,
                        help="Number of WikiText sequences to analyze")
    parser.add_argument("--n_permutations", type=int, default=1000,
                        help="Permutations for significance test")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output", type=str, default=None,
                        help="Save position-curvature plot")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt["config"]
    model = FGNModel(config).to(device)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {args.checkpoint}")
    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"seq_len={config.max_seq_len}")

    # Load WikiText data with GPT-2 tokenizer
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

    # Concatenate, tokenize, and chunk into sequences
    all_text = " ".join([ex["text"] for ex in ds if len(ex["text"].strip()) > 50])
    all_ids = tokenizer.encode(all_text)
    seq_len = config.max_seq_len
    n_available = len(all_ids) // seq_len
    n_seq = min(args.n_sequences, n_available)
    print(f"Using {n_seq} sequences of {seq_len} tokens from WikiText-103 validation")

    data = torch.tensor(all_ids[:n_seq * seq_len]).view(n_seq, seq_len)

    # Extract per-position curvatures and prediction entropy
    print("Extracting per-position curvatures...")
    curvatures, entropies = extract_position_curvatures(
        model, data, device, batch_size=args.batch_size)

    # Compute boundary scores for each sequence
    print("Computing structural boundary scores...")
    all_boundaries = {
        'punctuation': [],
        'word_boundary': [],
        'sentence_boundary': [],
        'combined': [],
        'prediction_entropy': [],
    }

    for seq_idx in range(n_seq):
        token_ids = data[seq_idx].numpy()
        scores = compute_boundary_scores(token_ids, tokenizer)
        for key in scores:
            all_boundaries[key].append(scores[key])
        all_boundaries['prediction_entropy'].append(entropies[seq_idx])

    # Stack into arrays: [n_seq, seq_len]
    for key in all_boundaries:
        all_boundaries[key] = np.stack(all_boundaries[key])

    # Report boundary statistics
    print(f"\nBoundary signal statistics (fraction of positions):")
    for key in ['punctuation', 'word_boundary', 'sentence_boundary']:
        frac = all_boundaries[key].mean()
        print(f"  {key}: {frac:.4f}")
    print(f"  prediction_entropy: mean={all_boundaries['prediction_entropy'].mean():.3f}, "
          f"std={all_boundaries['prediction_entropy'].std():.3f}")

    # Compute correlations per layer, per signal
    print(f"\n{'='*75}")
    print("POSITION-CURVATURE CORRELATIONS (Spearman rho)")
    print(f"{'='*75}")

    signals = ['punctuation', 'word_boundary', 'sentence_boundary',
               'combined', 'prediction_entropy']

    results = {}  # (layer, signal) -> (rho, p, ci_lo, ci_hi)

    for layer_idx in range(len(model.layers)):
        print(f"\n--- Layer {layer_idx} ---")
        print(f"  Curvature: mean |κ|={np.abs(curvatures[layer_idx]).mean():.6f}, "
              f"std={curvatures[layer_idx].std():.6f}")

        for signal in signals:
            # Flatten across all sequences for aggregate correlation
            curv_flat = curvatures[layer_idx].flatten()
            bound_flat = all_boundaries[signal].flatten()

            rho, p_val, ci_lo, ci_hi = correlate_with_permutation_test(
                np.abs(curv_flat), bound_flat,
                n_permutations=args.n_permutations)

            results[(layer_idx, signal)] = (rho, p_val, ci_lo, ci_hi)

            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {signal:25s}: rho={rho:+.4f}  p={p_val:.4f}  "
                  f"CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  {sig}")

    # Per-sequence correlation distribution (not just aggregate)
    print(f"\n{'='*75}")
    print("PER-SEQUENCE CORRELATION DISTRIBUTION")
    print(f"{'='*75}")
    print("(Spearman rho computed within each sequence, then summarized)")

    for layer_idx in range(len(model.layers)):
        print(f"\n--- Layer {layer_idx} ---")
        for signal in ['combined', 'prediction_entropy']:
            seq_rhos = []
            for seq_idx in range(n_seq):
                curv = np.abs(curvatures[layer_idx][seq_idx])
                bound = all_boundaries[signal][seq_idx]
                if bound.std() > 0:  # skip constant sequences
                    r, _ = stats.spearmanr(curv, bound)
                    if not np.isnan(r):
                        seq_rhos.append(r)

            if seq_rhos:
                seq_rhos = np.array(seq_rhos)
                frac_pos = np.mean(seq_rhos > 0)
                print(f"  {signal:25s}: median={np.median(seq_rhos):+.4f}, "
                      f"mean={seq_rhos.mean():+.4f}, "
                      f"std={seq_rhos.std():.4f}, "
                      f"frac_positive={frac_pos:.2%} ({len(seq_rhos)} seqs)")

    # Summary table
    print(f"\n{'='*75}")
    print("SUMMARY")
    print(f"{'='*75}")
    print(f"{'Layer':>6} {'Signal':>25} {'Spearman rho':>13} {'p-value':>10} {'Significant':>12}")
    print("-" * 75)

    any_significant = False
    for layer_idx in range(len(model.layers)):
        for signal in signals:
            rho, p_val, ci_lo, ci_hi = results[(layer_idx, signal)]
            sig = "YES" if p_val < 0.05 else "no"
            if p_val < 0.05:
                any_significant = True
            print(f"{layer_idx:>6} {signal:>25} {rho:>+13.4f} {p_val:>10.4f} {sig:>12}")

    print()
    if any_significant:
        print("RESULT: Significant position-curvature correlations detected.")
        print("The geometry carries structural information that OVL cannot detect.")
    else:
        print("RESULT: No significant position-curvature correlations.")
        print("The geometry appears content-blind at the position level.")

    # Optional: save visualization
    if args.output:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Plot: curvature vs position for a few example sequences,
            # with structural boundaries overlaid
            n_examples = min(4, n_seq)
            fig, axes = plt.subplots(n_examples, 1,
                                     figsize=(14, 3 * n_examples),
                                     sharex=True)
            if n_examples == 1:
                axes = [axes]

            # Use the deepest layer
            deep_layer = len(model.layers) - 1

            for idx, ax in enumerate(axes):
                positions = np.arange(seq_len)
                curv = np.abs(curvatures[deep_layer][idx])
                punct = all_boundaries['punctuation'][idx]
                sent = all_boundaries['sentence_boundary'][idx]
                entropy = all_boundaries['prediction_entropy'][idx]

                # Normalize for visual overlay
                curv_norm = (curv - curv.mean()) / (curv.std() + 1e-8)
                ent_norm = (entropy - entropy.mean()) / (entropy.std() + 1e-8)

                ax.plot(positions, curv_norm, alpha=0.7, linewidth=0.8,
                        label=f'|κ| (Layer {deep_layer})', color='blue')
                ax.plot(positions, ent_norm, alpha=0.5, linewidth=0.8,
                        label='Prediction entropy', color='orange')

                # Mark punctuation and sentence boundaries
                punct_pos = np.where(punct > 0)[0]
                sent_pos = np.where(sent > 0)[0]
                for p in punct_pos:
                    ax.axvline(p, color='gray', alpha=0.15, linewidth=0.5)
                for s in sent_pos:
                    ax.axvline(s, color='red', alpha=0.4, linewidth=1.0)

                ax.set_ylabel(f'Seq {idx}')
                if idx == 0:
                    ax.legend(loc='upper right', fontsize=8)
                    ax.set_title(f'Position-Resolved Curvature (Layer {deep_layer}) '
                                 f'with Structural Boundaries')

            axes[-1].set_xlabel('Token position')
            plt.tight_layout()
            plt.savefig(args.output, dpi=150)
            print(f"\nVisualization saved to {args.output}")
        except ImportError:
            print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()
