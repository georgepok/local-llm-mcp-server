"""Train CrossAttnHiddenLiquid for true cross-category generalization.

Architecture adds cross-attention between hidden_state and multi-token goal embeddings —
the content-comparison primitive missing from prior architectures. Goal-content-agnostic
operation should let model generalize to unseen goal categories.

Pipeline:
  At training time: encode each turn's goal text once with bge → multi-token embeddings
  Cache per-record goal_tokens (so we don't re-encode every step)
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from cross_attn_liquid import CrossAttnHiddenLiquid, forward_trajectory


def roc_auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rs = scores[labels == 1]
    rn = scores[labels == 0]
    wins = (rs[:, None] > rn[None, :]).sum()
    ties = (rs[:, None] == rn[None, :]).sum()
    return (wins + 0.5 * ties) / (n_pos * n_neg)


def compute_metric_cv(model, sample_records, device, n_sample=10):
    """Phase-transition diagnostic."""
    model.eval()
    all_diag = []
    with torch.no_grad():
        for r in sample_records[:n_sample]:
            T = r["T"]
            if T < 2:
                continue
            hs_traj = r["hidden_state_traj"].to(device)
            gt_traj = r["goal_tokens_traj"].to(device)
            z_g = r["z_goal_traj"].to(device)
            h_fast_traj, _, _ = forward_trajectory(
                model, hs_traj, gt_traj, z_g, device, target_t=None, training=False)
            # Replicate fast_step internals to set context, then read metric
            cross_ctx = model.cross_attention(hs_traj[-1:], gt_traj[-1:])
            hs_proj = model.hidden_proj(model.hidden_layernorm(hs_traj[-1:]))
            combined = torch.cat([cross_ctx, hs_proj], dim=-1)
            e = model.in_combined(combined)
            e = model.evidence_layernorm(e)
            evidence = e.unsqueeze(1) * model.evidence_mix.unsqueeze(0)
            h_input = h_fast_traj[-2].unsqueeze(0) + evidence
            h_input = model._soft_clamp(h_input)
            context = model.context_pool(h_input, None)
            model.dynamics.set_context(context, mask=None)
            metric_diag = model.dynamics.compute_metric_diag(h_input)
            all_diag.append(metric_diag.flatten().cpu().numpy())
    model.train()
    if not all_diag:
        return float("nan")
    arr = np.concatenate(all_diag)
    return float(arr.std() / max(1e-8, arr.mean()))


def encode_goal_tokens(text, tok, model, device, max_tokens=32):
    """Return per-token embeddings for goal text [G, d_enc]."""
    toks = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(device)
    out = model(**toks)
    hidden = out.last_hidden_state[0]  # [seq_len, d_enc]
    # Skip CLS and SEP if bge-style
    return torch.nn.functional.normalize(hidden, dim=-1).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_traj", required=True)
    p.add_argument("--test_traj", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--enc_model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--max_goal_tokens", type=int, default=32)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--d_attn", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--value_hidden", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--substrate_lr_ratio", type=float, default=0.3)
    p.add_argument("--lambda_value", type=float, default=5.0)
    p.add_argument("--lambda_jepa", type=float, default=0.3)
    p.add_argument("--lambda_var", type=float, default=0.1)
    p.add_argument("--jepa_window", type=int, default=2)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--early_stop_patience", type=int, default=40)
    p.add_argument("--min_steps_before_stop", type=int, default=3000)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[cal] device={device}, output={args.output}", flush=True)

    # Encoder for goal token embeddings
    print(f"[cal] loading encoder {args.enc_model}...", flush=True)
    enc_tok = AutoTokenizer.from_pretrained(args.enc_model)
    enc_model = AutoModel.from_pretrained(args.enc_model).to(device).eval()
    for pp in enc_model.parameters():
        pp.requires_grad = False
    d_enc = enc_model.config.hidden_size

    # Load data
    pack = torch.load(args.train_traj, map_location="cpu", weights_only=False)
    z_goal_dim = int(pack["enc_dim"])
    d_hidden = int(pack["d_hidden"])
    raw_records = pack["records"]
    print(f"[cal] train pack: d_hidden={d_hidden}, d_enc={d_enc}, "
          f"records={len(raw_records)}", flush=True)

    def augment(records_list):
        """Augment with per-chunk turn labels AND multi-token goal embeddings per chunk."""
        out = []
        for r in records_list:
            T = int(r["T"])
            if T < 4:
                continue
            hs_traj = r["hidden_state_traj"].float()
            z_goal_traj = r["z_goal_traj"].float()
            turn_starts = list(r["turn_chunk_starts"])
            turn_followed = list(r["turn_followed"])
            turn_instructions = list(r["turn_instructions"])
            # Encode multi-token goal embeddings PER TURN, then map chunks to turns
            per_turn_goal_tokens = []
            with torch.no_grad():
                for instr in turn_instructions:
                    gt = encode_goal_tokens(instr, enc_tok, enc_model, device,
                                              args.max_goal_tokens)
                    per_turn_goal_tokens.append(gt)
            # Build chunk-aligned goal_tokens trajectory
            chunk_to_turn = []
            cur_turn = 0
            for t in range(T):
                while (cur_turn + 1 < len(turn_starts) and
                       t >= turn_starts[cur_turn + 1]):
                    cur_turn += 1
                chunk_to_turn.append(cur_turn)
            # Pad to fixed max_goal_tokens for batching consistency
            G_max = args.max_goal_tokens
            goal_tokens_traj = torch.zeros(T, G_max, d_enc)
            goal_mask = torch.zeros(T, G_max, dtype=torch.bool)
            for t in range(T):
                gt = per_turn_goal_tokens[chunk_to_turn[t]]
                G = min(gt.shape[0], G_max)
                goal_tokens_traj[t, :G] = gt[:G]
                goal_mask[t, :G] = True
            chunk_labels = []
            cur_turn = 0
            for t in range(T):
                while (cur_turn + 1 < len(turn_starts) and
                       t >= turn_starts[cur_turn + 1]):
                    cur_turn += 1
                chunk_labels.append(int(turn_followed[cur_turn]))
            out.append({
                "hidden_state_traj": hs_traj,
                "goal_tokens_traj": goal_tokens_traj,
                "goal_mask": goal_mask,
                "z_goal_traj": z_goal_traj,
                "labels": torch.tensor(chunk_labels, dtype=torch.float32),
                "T": T,
                "sub_id": int(r["sub_id"]),
            })
        return out

    print(f"[cal] encoding goal tokens for train records...", flush=True)
    train_records = augment(raw_records)
    rng = np.random.default_rng(42)
    all_sub_ids = sorted({r["sub_id"] for r in train_records})
    rng.shuffle(all_sub_ids)
    n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
    val_ids = set(all_sub_ids[:n_val_ids])
    train_recs = [r for r in train_records if r["sub_id"] not in val_ids]
    val_recs = [r for r in train_records if r["sub_id"] in val_ids]
    print(f"[cal] train={len(train_recs)}  val_in_dist={len(val_recs)}", flush=True)

    test_recs = []
    if args.test_traj:
        print(f"[cal] encoding goal tokens for test records...", flush=True)
        test_pack = torch.load(args.test_traj, map_location="cpu", weights_only=False)
        test_recs = augment(test_pack["records"])
        print(f"[cal] held-out test: {len(test_recs)}", flush=True)

    # Free encoder GPU mem
    del enc_model

    online = CrossAttnHiddenLiquid(
        d_hidden=d_hidden, d_goal=d_enc, d=args.d, K=args.K,
        d_attn=args.d_attn, n_heads=args.n_heads, value_hidden=args.value_hidden,
    ).to(device)
    target = copy.deepcopy(online)
    for pp in target.parameters():
        pp.requires_grad = False
    n_params = sum(p.numel() for p in online.parameters() if p.requires_grad)
    print(f"[cal] CrossAttnHiddenLiquid d={args.d}, d_attn={args.d_attn}, "
          f"{n_params:,} params", flush=True)

    pred_params = (list(online.head_value.parameters()) +
                    list(online.head_jepa.parameters()))
    ca_params = (list(online.W_q.parameters()) + list(online.W_k.parameters()) +
                 list(online.W_v.parameters()) + list(online.out_proj.parameters()) +
                 list(online.hidden_layernorm.parameters()) +
                 list(online.goal_layernorm.parameters()) +
                 list(online.hidden_proj.parameters()) +
                 list(online.in_combined.parameters()))
    geom_param_ids = set()
    for mod in (online.dynamics, online.context_pool):
        for pp in mod.parameters():
            geom_param_ids.add(id(pp))
    geom_params = [p for p in online.parameters() if id(p) in geom_param_ids]
    excluded_ids = set()
    for plist in (pred_params, ca_params, geom_params):
        for p in plist:
            excluded_ids.add(id(p))
    body_params = [p for p in online.parameters() if id(p) not in excluded_ids]
    opt = torch.optim.AdamW([
        {"params": pred_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": ca_params, "lr": args.lr, "weight_decay": 0.005},
        {"params": body_params, "lr": args.lr, "weight_decay": 0.005},
        {"params": geom_params, "lr": args.lr * args.substrate_lr_ratio,
         "weight_decay": 0.0},
    ])
    bce = nn.BCEWithLogitsLoss()

    def loss_for_batch(recs, training=True):
        value_losses, jepa_losses, var_feats = [], [], []
        for r in recs:
            T = r["T"]
            hs_traj = r["hidden_state_traj"].to(device)
            gt_traj = r["goal_tokens_traj"].to(device)
            z_g = r["z_goal_traj"].to(device)
            labels = r["labels"].to(device)
            t = int(rng.integers(0, max(1, T - args.jepa_window)))
            h_fast_traj, _, cross_attn_traj = forward_trajectory(
                online, hs_traj, gt_traj, z_g, device, target_t=t, training=training)
            h_fast_now = h_fast_traj[t].unsqueeze(0)
            cross_attn_now = cross_attn_traj[t].unsqueeze(0)
            logit = online.value(h_fast_now, cross_attn_now)
            value_losses.append(bce(logit, labels[t].unsqueeze(0)))
            pred = online.jepa_predict(h_fast_now)
            with torch.no_grad():
                target_h_fast_traj, _, _ = forward_trajectory(
                    target, hs_traj, gt_traj, z_g, device,
                    target_t=t + args.jepa_window, training=False)
                tgt = target_h_fast_traj[t + args.jepa_window].detach().unsqueeze(0)
            jepa_losses.append(((pred - tgt) ** 2).mean())
            var_feats.append(h_fast_now[0].flatten().detach())
        if not value_losses:
            return None, None, None
        value_loss = torch.stack(value_losses).mean()
        jepa_loss = torch.stack(jepa_losses).mean()
        var = torch.stack(var_feats, dim=0).std(dim=0).mean()
        var_loss = torch.relu(args.lambda_var - var)
        return value_loss, jepa_loss, var_loss

    def ema_update(tau):
        with torch.no_grad():
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)

    @torch.no_grad()
    def auc_on_set(recs):
        if not recs:
            return float("nan"), 0, 0
        online.eval(); target.eval()
        all_logits, all_labels = [], []
        for r in recs:
            T = r["T"]
            hs_traj = r["hidden_state_traj"].to(device)
            gt_traj = r["goal_tokens_traj"].to(device)
            z_g = r["z_goal_traj"].to(device)
            labels = r["labels"].to(device)
            h_fast_traj, _, cross_attn_traj = forward_trajectory(
                online, hs_traj, gt_traj, z_g, device, target_t=None, training=False)
            for t in range(T):
                logit = online.value(h_fast_traj[t].unsqueeze(0),
                                       cross_attn_traj[t].unsqueeze(0))
                all_logits.append(float(logit.item()))
                all_labels.append(int(labels[t].item()))
        logits_np = np.array(all_logits)
        labels_np = np.array(all_labels)
        n_drift = int((labels_np == 0).sum())
        n_follow = int((labels_np == 1).sum())
        online.train(); target.train()
        if n_drift == 0 or n_follow == 0:
            return float("nan"), n_drift, n_follow
        return roc_auc(-logits_np, 1 - labels_np), n_drift, n_follow

    online.train(); target.train()
    best_test_auc = -1.0
    best_state = None
    last_improvement = 0
    n_nan = 0
    t_start = time.time()

    for step in range(args.max_steps + 1):
        batch = rng.choice(len(train_recs), args.batch_size, replace=True)
        recs = [train_recs[i] for i in batch]
        out = loss_for_batch(recs, training=True)
        if out[0] is None:
            continue
        value_l, jepa_l, var_l = out
        total = args.lambda_value * value_l + args.lambda_jepa * jepa_l + var_l
        if not torch.isfinite(total):
            n_nan += 1
            if n_nan >= 5:
                print(f"[cal] ABORT: 5 NaN losses at step {step}", flush=True)
                break
            continue
        n_nan = 0
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), args.max_grad_norm)
        opt.step()
        ema_update(args.ema_tau)

        if step % args.log_every == 0:
            v_auc, _, _ = auc_on_set(val_recs)
            t_auc, t_d, t_f = auc_on_set(test_recs) if test_recs else (float("nan"), 0, 0)
            cv = compute_metric_cv(online, val_recs, device)
            selection_auc = t_auc if test_recs else v_auc
            stale = (step - last_improvement) // args.log_every
            if not np.isnan(selection_auc) and selection_auc > best_test_auc:
                best_test_auc = selection_auc
                best_state = {
                    "online": copy.deepcopy(online.state_dict()),
                    "target": copy.deepcopy(target.state_dict()),
                }
                last_improvement = step
                stale = 0
            print(f"step {step:>5}  L_v={float(value_l.detach()):.3f}  "
                  f"v_auc={v_auc:.3f}  t_auc={t_auc:.3f}  "
                  f"(best {best_test_auc:.3f})  CV={cv:.2f}  stale={stale}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)
            if step >= args.min_steps_before_stop and stale >= args.early_stop_patience:
                print(f"[cal] early stop at step {step}", flush=True)
                break

    if best_state is not None:
        online.load_state_dict(best_state["online"])
        target.load_state_dict(best_state["target"])
    torch.save({
        "model_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "best_test_auc": best_test_auc,
        "args": vars(args),
        "d_hidden": d_hidden,
        "d_enc": d_enc,
    }, args.output)
    print(f"[cal] saved → {args.output}  best_test_auc={best_test_auc:.3f}", flush=True)
    print("[cal] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
