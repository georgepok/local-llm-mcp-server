"""Evaluate a temporal reachability checkpoint at multiple chain lengths.

Usage:
    python scripts/eval_temporal_reach.py \
        --config configs/affine_flat.yaml \
        --checkpoint output_tr_flat/stage1_taskTR/checkpoints/final.pt \
        --n_batches 30 --batch_size 16
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model import FGNModel
from fgn.flat_model import FlatTransformerModel
from fgn.tasks import get_task


def _get_tok():
    from transformers import GPT2Tokenizer
    t = GPT2Tokenizer.from_pretrained("gpt2")
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


def load_model(config, ckpt_path, device):
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    else:
        model = FGNModel(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    # Trim pos_embed if the checkpoint was saved at a longer max_seq_len
    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > config.max_seq_len:
        state[pe_key] = state[pe_key][:config.max_seq_len]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def run_cell(model, task, n_batches: int, batch_size: int,
             device) -> dict:
    correct = 0
    total = 0
    yes_correct = 0
    yes_total = 0
    no_correct = 0
    no_total = 0
    for _ in range(n_batches):
        ids, labels, meta = task.generate_batch(batch_size, device)
        out = model(ids)
        if isinstance(out, dict):
            logits = out["logits"]
        elif hasattr(out, "logits"):
            logits = out.logits
        else:
            logits = out
        # label is at the last non-ignored position per row
        # labels[-100] for ignored positions; pick the single supervised idx
        ans_pos = (labels != -100).int().argmax(dim=1)                 # [B]
        row_idx = torch.arange(ids.shape[0], device=device)
        true_answer = labels[row_idx, ans_pos]                          # [B]
        pred_logits = logits[row_idx, ans_pos]                          # [B, V]
        pred = pred_logits.argmax(dim=-1)                               # [B]

        correct += (pred == true_answer).sum().item()
        total += ids.shape[0]
        # Class-conditional accuracy
        yes_mask = (true_answer == task.yes_token)
        no_mask = (true_answer == task.no_token)
        yes_total += yes_mask.sum().item()
        no_total += no_mask.sum().item()
        yes_correct += ((pred == true_answer) & yes_mask).sum().item()
        no_correct += ((pred == true_answer) & no_mask).sum().item()
    return {
        "acc": correct / max(1, total),
        "yes_acc": yes_correct / max(1, yes_total),
        "no_acc": no_correct / max(1, no_total),
        "n": total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_batches", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_nodes", type=int, default=16)
    p.add_argument("--min_hops_yes", type=int, default=2,
                   help="Must match training to avoid distribution mismatch")
    p.add_argument("--n_decoy_chains", type=int, default=2,
                   help="Must match training n_decoy_chains")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    config = FGNConfig.from_yaml(args.config)
    device = torch.device(args.device)
    tok = _get_tok()
    model = load_model(config, args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.model_type}  params: {n_params:,}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    print(f"{'edges':>14}  {'acc':>6}  {'yes_acc':>8}  {'no_acc':>8}  {'n':>5}")
    print(f"{'-'*14}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*5}")

    cells = [
        ("50-100 (short)", 50, 100),
        ("200-400 (train)", 200, 400),
        ("500", 500, 500),
        ("600", 600, 600),
    ]
    for desc, mn, mx in cells:
        task = get_task("TR", tok, seq_len=config.max_seq_len,
                         n_nodes=args.n_nodes, min_edges=mn, max_edges=mx,
                         min_hops_yes=args.min_hops_yes,
                         n_decoy_chains=args.n_decoy_chains)
        r = run_cell(model, task, args.n_batches, args.batch_size, device)
        print(f"{desc:>14}  {r['acc']:>6.3f}  {r['yes_acc']:>8.3f}  "
              f"{r['no_acc']:>8.3f}  {r['n']:>5d}")


if __name__ == "__main__":
    main()
