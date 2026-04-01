"""FGN v3 Phase 1b v2 — Compound Reasoning Task evaluation.

Evaluates per-section accuracy (SORT, SCAN, CHAIN) and compound accuracy.
For FGN models, also reports geometric diagnostics per section.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model import FGNModel
from fgn.flat_model import FlatTransformerModel
from fgn.tasks import get_task


def load_model(config, checkpoint_path, device):
    """Load model from checkpoint."""
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    else:
        model = FGNModel(config).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    # Truncate pos_embed if needed
    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > config.max_seq_len:
        state[pe_key] = state[pe_key][:config.max_seq_len]

    model.load_state_dict(state)
    return model


def parse_answer_sections(text):
    """Parse a compound answer into (sort_str, count_str, chain_str).

    Expected format: "Answer: name1 name2 ... | count | city"
    Returns None for sections that can't be parsed.
    """
    # Strip "Answer:" prefix if present
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1]

    parts = text.split("|")
    if len(parts) < 3:
        return None, None, None

    sort_str = parts[0].strip()
    count_str = parts[1].strip()
    chain_str = parts[2].strip()

    return sort_str, count_str, chain_str


def evaluate_crt(model, task, tokenizer, _config, device, n_batches=100, batch_size=8):
    """Evaluate CRT with section-level accuracy."""
    model.eval()

    total_ce = 0.0
    sort_exact = 0
    sort_position_correct = 0
    sort_position_total = 0
    scan_correct = 0
    chain_correct = 0
    compound_correct = 0
    total_samples = 0

    with torch.no_grad():
        for _ in range(n_batches):
            input_ids, labels, _ = task.generate_batch(batch_size, device=device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                result = model(input_ids, labels=labels)

            total_ce += result["ce_loss"].item()

            # Get predictions
            preds = result["logits"].argmax(dim=-1)

            for b in range(batch_size):
                mask = labels[b] != -100
                if not mask.any():
                    continue

                pred_tokens = preds[b][mask]
                true_tokens = labels[b][mask]

                pred_text = tokenizer.decode(pred_tokens.tolist())
                true_text = tokenizer.decode(true_tokens.tolist())

                pred_sort, pred_count, pred_chain = parse_answer_sections(pred_text)
                true_sort, true_count, true_chain = parse_answer_sections(true_text)

                if true_sort is None:
                    continue

                total_samples += 1

                # SORT accuracy
                s_correct = (pred_sort == true_sort)
                sort_exact += int(s_correct)

                # Per-position sort accuracy
                pred_names = (pred_sort or "").split()
                true_names = true_sort.split()
                for i in range(min(len(pred_names), len(true_names))):
                    sort_position_total += 1
                    if pred_names[i] == true_names[i]:
                        sort_position_correct += 1
                # Count missing positions as wrong
                sort_position_total += abs(len(pred_names) - len(true_names))

                # SCAN accuracy
                c_correct = (pred_count == true_count)
                scan_correct += int(c_correct)

                # CHAIN accuracy
                ch_correct = (pred_chain == true_chain)
                chain_correct += int(ch_correct)

                # Compound: all three correct
                compound_correct += int(s_correct and c_correct and ch_correct)

    n = max(total_samples, 1)
    results = {
        "ce_loss": total_ce / max(n_batches, 1),
        "sort_exact_acc": sort_exact / n,
        "sort_position_acc": sort_position_correct / max(sort_position_total, 1),
        "scan_acc": scan_correct / n,
        "chain_acc": chain_correct / n,
        "compound_acc": compound_correct / n,
        "n_samples": total_samples,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="CRT Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_batches", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    config = FGNConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Model: {config.model_type}")
    print(f"Checkpoint: {args.checkpoint}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    model = load_model(config, args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    task = get_task("E", tokenizer, seq_len=config.max_seq_len)

    print(f"\nEvaluating on {args.n_batches} batches (bs={args.batch_size})...")
    results = evaluate_crt(model, task, tokenizer, None, device,
                          n_batches=args.n_batches, batch_size=args.batch_size)

    print(f"\n{'='*50}")
    print(f"CRT Evaluation Results ({config.model_type})")
    print(f"{'='*50}")
    print(f"  CE Loss:           {results['ce_loss']:.4f}")
    print(f"  SORT exact:        {results['sort_exact_acc']:.4f}")
    print(f"  SORT position:     {results['sort_position_acc']:.4f}")
    print(f"  SCAN count:        {results['scan_acc']:.4f}")
    print(f"  CHAIN hop:         {results['chain_acc']:.4f}")
    print(f"  COMPOUND (all 3):  {results['compound_acc']:.4f}")
    print(f"  Samples evaluated: {results['n_samples']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
