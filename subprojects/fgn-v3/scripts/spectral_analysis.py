"""Spectral analysis of substrate representations on TSP inputs.

For each substrate (Flat, FGN, Liquid), capture hidden states at every
layer/ODE-step boundary on a fixed batch of TSP examples. Compute:
  - effective_rank = exp(Shannon entropy of normalized squared singular values)
  - stable_rank = sum(sigma^2) / sigma_1^2
  - participation_ratio = (sum sigma^2)^2 / sum sigma^4
  - anisotropy = mean pairwise cos similarity across token representations

Writes JSONL, one record per (substrate, checkpoint, depth).

Tests the rank-collapse hypothesis: does Liquid compress representations
from rank ~100 at step 0 to rank ~10 at step 16, while FGN maintains rank
across its 6 layers?

Usage:
    python scripts/spectral_analysis.py \\
        --config configs/tr_liquid_n512.yaml \\
        --checkpoint output_prod_liquid/stage1_taskTSP/checkpoints/final.pt \\
        --run_label liquid_final --json_out /tmp/spectral.jsonl
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.flat_model import FlatTransformerModel
from fgn.model import FGNModel
try:
    from fgn.liquid_model import LiquidSequenceModel
except Exception:
    LiquidSequenceModel = None
from fgn.tasks import get_task


def _tokenizer():
    from transformers import GPT2Tokenizer
    t = GPT2Tokenizer.from_pretrained("gpt2")
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


def spectral_stats(h: torch.Tensor, mask: torch.Tensor = None,
                   sample_k: int = 256) -> dict:
    """h: [B, N, d]. mask [B, N] True=keep. Stats over kept tokens only.

    Without a mask, pad tokens (all same embedding) dominate the statistics.
    Passing a non-pad mask isolates content-relevant rank/anisotropy.
    """
    B, N, d = h.shape
    if mask is not None:
        X = h[mask].float()  # [K, d] where K = sum(mask)
    else:
        X = h.reshape(B * N, d).float()
    # Center (row-wise anisotropy is separate; we center for rank)
    Xc = X - X.mean(dim=0, keepdim=True)
    # Compute singular values via eigvalsh of d×d Gram matrix (CUDA-friendly,
    # avoids cusolver SVD which errors on GB10). sigma_i = sqrt(lambda_i(X^T X)).
    # d=256 so Gram matrix is tiny.
    gram = Xc.T @ Xc  # [d, d]
    # Move to CPU as a belt-and-suspenders (some GB10 cusolver paths still error)
    eigvals = torch.linalg.eigvalsh(gram.cpu()).clamp_min(0.0)
    sigma = eigvals.sqrt().sort(descending=True).values.to(X.device)
    sigma_sq = sigma ** 2
    total = sigma_sq.sum()
    p = sigma_sq / (total + 1e-12)
    # Effective rank (exp of Shannon entropy)
    entropy = -(p * (p + 1e-12).log()).sum()
    eff_rank = math.exp(entropy.item())
    # Stable rank
    stable_rank = (total / (sigma[0] ** 2 + 1e-12)).item()
    # Participation ratio
    pr = (sigma_sq.sum() ** 2) / ((sigma_sq ** 2).sum() + 1e-12)
    # Anisotropy: mean pairwise cos similarity on RAW (uncentered) tokens.
    # Ethayarajh: BERT later layers have high mean_cos → collapsed.
    K = X.shape[0]  # actual (post-mask) count
    M = min(K, sample_k)
    idx = torch.randperm(K, device=X.device)[:M]
    Xs = F.normalize(X[idx], dim=-1)
    cos_mat = Xs @ Xs.T
    eye = torch.eye(M, device=X.device, dtype=torch.bool)
    mean_cos = cos_mat[~eye].mean().item()

    # Top-3 normalized singular values (shape peek)
    top3_ratio = (sigma[:3] / (sigma[0] + 1e-12)).tolist()
    n_tokens = int(X.shape[0])
    return {
        "n_tokens": n_tokens,
        "d": d,
        "eff_rank": eff_rank,
        "stable_rank": stable_rank,
        "participation_ratio": pr.item(),
        "anisotropy_mean_cos": mean_cos,
        "top_sigma": sigma[0].item(),
        "top3_ratio": top3_ratio,
    }


def analyze_flat(model, input_ids, records, meta, nonpad_mask):
    device = input_ids.device
    N = input_ids.shape[1]
    pos = torch.arange(N, device=device).unsqueeze(0)
    h = model.embed(input_ids) + model.pos_embed(pos)
    records.append({**meta, "depth_type": "layer", "depth": 0,
                    **spectral_stats(h.detach(), mask=nonpad_mask)})
    mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool),
                      diagonal=1)
    for i, layer in enumerate(model.layers, start=1):
        h = layer(h, mask=mask)
        records.append({**meta, "depth_type": "layer", "depth": i,
                        **spectral_stats(h.detach(), mask=nonpad_mask)})


def analyze_fgn(model, input_ids, records, meta, nonpad_mask):
    device = input_ids.device
    N = input_ids.shape[1]
    pos = torch.arange(N, device=device).unsqueeze(0)
    h = model.embed(input_ids) + model.pos_embed(pos)
    records.append({**meta, "depth_type": "layer", "depth": 0,
                    **spectral_stats(h.detach(), mask=nonpad_mask)})
    mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool),
                      diagonal=1)
    for i, layer in enumerate(model.layers, start=1):
        result = layer(h, mask=mask)
        h = result[0] if isinstance(result, tuple) else result
        records.append({**meta, "depth_type": "layer", "depth": i,
                        **spectral_stats(h.detach(), mask=nonpad_mask)})


def analyze_liquid(model, input_ids, records, meta, nonpad_mask,
                   integration_time_override=None,
                   n_steps_override=None,
                   tau_min_override=None,
                   tau_max_override=None):
    """Manually step the Euler integration to capture h at each ODE step.

    Optional overrides let us probe the dynamics at test time:
      - integration_time: total ODE integration time (default: trained value 2.0)
      - n_steps: number of Euler steps (default: trained 16)
      - tau_min/tau_max: clamp bounds on per-position tau (default: trained [0.5,1.0])

    Lowering tau_min forces smaller tau → faster LTC contraction → more rank
    compression, testing whether tau's learned floor was the bottleneck.
    """
    device = input_ids.device
    N = input_ids.shape[1]
    dyn = model.dynamics
    dyn_raw = dyn._orig_mod if hasattr(dyn, '_orig_mod') else dyn

    # Apply tau overrides (simple, direct attribute write)
    orig_tau_min = getattr(dyn_raw, 'tau_min', None)
    orig_tau_max = getattr(dyn_raw, 'tau_max', None)
    if tau_min_override is not None:
        dyn_raw.tau_min = tau_min_override
    if tau_max_override is not None:
        dyn_raw.tau_max = tau_max_override

    integration_time = (integration_time_override if integration_time_override
                        is not None else model.la_cfg.integration_time)
    n_ode_steps = n_steps_override if n_steps_override is not None else model.n_ode_steps

    pos = torch.arange(N, device=device).unsqueeze(0)
    h = model.embed(input_ids) + model.pos_embed(pos)
    records.append({**meta, "depth_type": "ode_step", "depth": 0,
                    **spectral_stats(h.detach(), mask=nonpad_mask)})
    mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool),
                      diagonal=1)
    context = model._pool_context(h)
    dyn_raw.set_context(context, mask=mask)
    if hasattr(dyn_raw, 'set_n_steps'):
        dyn_raw.set_n_steps(n_ode_steps)
    t_start, t_end = 0.0, integration_time
    dt = (t_end - t_start) / n_ode_steps
    t = t_start
    for i in range(n_ode_steps):
        if hasattr(dyn_raw, 'set_step_embed'):
            dyn_raw.set_step_embed(i, min(n_ode_steps, 20))
        if hasattr(dyn_raw, 'set_step_index'):
            dyn_raw.set_step_index(i, min(n_ode_steps, 20))
        dh = dyn_raw(t, h)
        h = h + dt * dh
        t = t + dt
        records.append({**meta, "depth_type": "ode_step", "depth": i + 1,
                        **spectral_stats(h.detach(), mask=nonpad_mask)})

    # Compute accuracy on this batch with the modified dynamics
    h_out = model.norm(h)
    logits_out = model.lm_head(h_out)

    # Restore
    if orig_tau_min is not None:
        dyn_raw.tau_min = orig_tau_min
    if orig_tau_max is not None:
        dyn_raw.tau_max = orig_tau_max
    return logits_out


def load_model(config, ckpt_path, device):
    # Force compile off for clean step-by-step hooking
    config.use_torch_compile = False
    if config.model_type == "flat":
        model = FlatTransformerModel(config).to(device)
    elif config.model_type == "liquid":
        assert LiquidSequenceModel is not None
        model = LiquidSequenceModel(config).to(device)
    else:
        model = FGNModel(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Strip compile prefixes (both top-level and nested on dynamics.)
    state = {}
    for k, v in ckpt["model"].items():
        nk = k.removeprefix("_orig_mod.")
        nk = nk.replace("dynamics._orig_mod.", "dynamics.")
        state[nk] = v
    pe_key = "pos_embed.weight"
    if pe_key in state and state[pe_key].shape[0] > config.max_seq_len:
        state[pe_key] = state[pe_key][:config.max_seq_len]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--run_label", required=True)
    ap.add_argument("--batch_size", type=int, default=14)
    ap.add_argument("--n_nodes", type=int, default=1024)
    ap.add_argument("--min_edges", type=int, default=50)
    ap.add_argument("--max_edges", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json_out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--integration_time", type=float, default=None,
                    help="Override ODE integration time (Liquid only)")
    ap.add_argument("--n_ode_steps", type=int, default=None,
                    help="Override ODE step count (Liquid only)")
    ap.add_argument("--tau_min", type=float, default=None,
                    help="Override tau_min clamp (Liquid only)")
    ap.add_argument("--tau_max", type=float, default=None,
                    help="Override tau_max clamp (Liquid only)")
    ap.add_argument("--compute_accuracy", action="store_true",
                    help="Also compute non-unreach accuracy on the batch")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = FGNConfig.from_yaml(args.config)
    device = torch.device(args.device)
    tok = _tokenizer()
    model = load_model(config, args.checkpoint, device)
    task = get_task("TSP", tok, seq_len=config.max_seq_len,
                     n_nodes=args.n_nodes,
                     min_edges=args.min_edges, max_edges=args.max_edges)
    input_ids, labels, _ = task.generate_batch(args.batch_size, device)
    # Build non-pad mask: any position with pad token excluded. This isolates
    # the spectral analysis from padding artifact (1750 of 2048 positions are
    # pad, would swamp statistics with the single pad embedding direction).
    pad_id = tok.pad_token_id
    nonpad_mask = input_ids != pad_id  # [B, N] bool

    meta = {
        "run_label": args.run_label,
        "checkpoint": args.checkpoint,
        "model_type": config.model_type,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "seq_len": input_ids.shape[1],
    }
    records = []
    print(f"Model: {config.model_type}  Label: {args.run_label}")
    with torch.no_grad():
        logits_out = None
        if config.model_type == "flat":
            analyze_flat(model, input_ids, records, meta, nonpad_mask)
        elif config.model_type == "fgn":
            analyze_fgn(model, input_ids, records, meta, nonpad_mask)
        elif config.model_type == "liquid":
            logits_out = analyze_liquid(
                model, input_ids, records, meta, nonpad_mask,
                integration_time_override=args.integration_time,
                n_steps_override=args.n_ode_steps,
                tau_min_override=args.tau_min,
                tau_max_override=args.tau_max)

    # Optional accuracy measurement on same batch (non-unreach bucket acc)
    if args.compute_accuracy and logits_out is not None:
        task = get_task("TSP", tok, seq_len=config.max_seq_len,
                        n_nodes=args.n_nodes, min_edges=args.min_edges,
                        max_edges=args.max_edges)
        # Need the SAME generated batch — regenerate with same seed
        random.seed(args.seed); torch.manual_seed(args.seed)
        _, labels2, _ = task.generate_batch(args.batch_size, device)
        pos = (labels2 != -100).int().argmax(dim=1)
        row = torch.arange(input_ids.shape[0], device=device)
        truth = labels2[row, pos]
        label_tokens = task.answer_tokens
        answer_logits = torch.stack(
            [logits_out[row, pos, t] for t in label_tokens], dim=-1)
        pred_bucket = answer_logits.argmax(dim=-1)
        truth_bucket = torch.full_like(truth, -1)
        for i, tid in enumerate(label_tokens):
            truth_bucket[truth == tid] = i
        unreach_idx = len(label_tokens) - 1
        n_non_u = 0; correct_non_u = 0; correct_total = 0; total = 0
        pred_counts = [0] * len(label_tokens)
        for tb, pb in zip(truth_bucket.tolist(), pred_bucket.tolist()):
            if tb < 0: continue
            total += 1; pred_counts[pb] += 1
            if tb == pb: correct_total += 1
            if tb != unreach_idx:
                n_non_u += 1
                if tb == pb: correct_non_u += 1
        # Overwrite the last record with accuracy fields
        acc_info = {
            "acc_total": correct_total / max(1, total),
            "acc_non_unreach": correct_non_u / max(1, n_non_u),
            "n_non_unreach": n_non_u,
            "degeneracy": max(pred_counts) / max(1, total),
        }
        print(f"  Accuracy (batch): total={acc_info['acc_total']:.3f}  "
              f"non_unreach={acc_info['acc_non_unreach']:.3f}  "
              f"degen={acc_info['degeneracy']:.3f}")
        # Also append to records for aggregation
        records.append({**meta, "depth_type": "summary", "depth": -1,
                        **acc_info})

    with open(args.json_out, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"{'Depth':>10}  {'EffRank':>8}  {'StableR':>8}  {'Anisotropy':>10}")
    for r in records:
        if "eff_rank" not in r:
            continue
        depth_type = r["depth_type"]
        d = r["depth"]
        print(f"  {depth_type[:4]:>4} {d:>3}  {r['eff_rank']:>8.1f}  "
              f"{r['stable_rank']:>8.2f}  {r['anisotropy_mean_cos']:>+10.3f}")
    print(f"Appended {len(records)} records to {args.json_out}")


if __name__ == "__main__":
    main()
