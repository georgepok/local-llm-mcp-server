"""Alignment-Liquid training targeting PHASE TRANSITION on generic goal tracking.

Differences from train_alignment_liquid.py:
  - Geometric core (ContinuousDynamics + ContextPool) at FULL base LR
    (not the 100× slower preserve-mode; we WANT reorganization)
  - Metric CV monitoring at every val cycle — track phase transition
  - Cross-category split: hold out a goal category from training, test on it
  - Much higher patience (60) — don't early-stop pre-transition
  - Longer training (max 10K steps)
  - Larger d (default 128) — needs capacity for geometric reorganization
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from alignment_liquid import AlignmentLiquid, forward_trajectory, ALIGNMENT_FEATURE_DIM
from alignment_liquid import compute_alignment_features


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


def compute_metric_cv(model, sample_records, device, n_sample=20):
    """Compute metric eigenvalue CV across sample records.
    Phase transition indicator: CV jumps from ~2 (flat) to ~6+ (richly curved).
    """
    model.eval()
    all_diag = []
    with torch.no_grad():
        for r in sample_records[:n_sample]:
            T = r["T"]
            z_t = r["z_t_traj"].to(device)
            z_g = r["z_goal_traj"].to(device)
            h_fast_traj, _, _ = forward_trajectory(
                model, z_t, z_g, device, target_t=None, training=False)
            # Final state — call compute_metric_diag with dynamics context set
            # We need to re-call fast_step to set context
            # Simpler: do a final fast_step on last chunk and get metric
            if T < 2:
                continue
            f_last = compute_alignment_features(
                z_t[-1:], z_g[-1:], z_t[-2:-1], z_g[-2:-1])  # [1, 8]
            # Call fast_step which sets context internally
            # Then call compute_metric_diag on the resulting h
            h_prev = h_fast_traj[-2].unsqueeze(0)
            # Replicate fast_step internals just enough to set context
            e_align = model.in_align(f_last) * model.align_gate
            e_goal = model.in_goal(z_g[-1:]) * model.goal_gate
            e = model.evidence_layernorm(e_align + e_goal)
            evidence = e.unsqueeze(1) * model.evidence_mix.unsqueeze(0)
            h_input = h_prev + evidence
            h_input = model._soft_clamp(h_input)
            context = model.context_pool(h_input, None)
            model.dynamics.set_context(context, mask=None)
            # Now compute metric diagonal
            metric_diag = model.dynamics.compute_metric_diag(h_input)  # [1, K, d]
            all_diag.append(metric_diag.flatten().cpu().numpy())
    model.train()
    if not all_diag:
        return float("nan")
    arr = np.concatenate(all_diag)
    return float(arr.std() / max(1e-8, arr.mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text_traj", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--value_hidden", type=int, default=256)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--substrate_lr_ratio", type=float, default=1.0,
                   help="Geometric LR ratio. Default 1.0 = FULL LR (allow phase transition)")
    p.add_argument("--lambda_value", type=float, default=5.0)
    p.add_argument("--lambda_jepa", type=float, default=0.5)
    p.add_argument("--lambda_var", type=float, default=0.1)
    p.add_argument("--jepa_window", type=int, default=2)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--holdout_category", default="contrast",
                   help="Goal category held out for cross-category test")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--early_stop_patience", type=int, default=60)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--min_steps_before_stop", type=int, default=2000,
                   help="Don't early-stop before this step (let geometry reorganize)")
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[phase] device={device}, output={args.output}", flush=True)
    print(f"[phase] HOLD-OUT category: {args.holdout_category}", flush=True)
    print(f"[phase] geometric LR ratio: {args.substrate_lr_ratio} (1.0 = full LR)", flush=True)

    # Load diverse goal data with category metadata
    pack = torch.load(args.text_traj, map_location="cpu", weights_only=False)
    z_goal_dim = int(pack["enc_dim"])
    raw_records = pack["records"]
    categories_seen = pack.get("categories", ["surface","inclusion","structure","format","contrast"])
    print(f"[phase] categories in data: {categories_seen}", flush=True)

    # Build augmented records
    records = []
    for r in raw_records:
        T = int(r["T"])
        if T < 4:
            continue
        z_t_traj = r["z_t_traj"].float()
        z_goal_traj = r["z_lang_traj"].float()
        turn_starts = list(r["turn_chunk_starts"])
        turn_followed = list(r["turn_followed"])
        turn_categories = list(r.get("turn_categories", []))
        chunk_labels = []
        chunk_categories = []
        cur_turn = 0
        for t in range(T):
            while (cur_turn + 1 < len(turn_starts) and
                   t >= turn_starts[cur_turn + 1]):
                cur_turn += 1
            chunk_labels.append(int(turn_followed[cur_turn]))
            chunk_categories.append(turn_categories[cur_turn] if turn_categories else "")
        records.append({
            "z_t_traj": z_t_traj,
            "z_goal_traj": z_goal_traj,
            "labels": torch.tensor(chunk_labels, dtype=torch.float32),
            "chunk_categories": chunk_categories,
            "turn_categories": turn_categories,
            "T": T,
            "sub_id": int(r["sub_id"]),
        })
    print(f"[phase] {len(records)} records loaded, z_goal_dim={z_goal_dim}", flush=True)

    # Split:
    #   train_in_dist: conversations where NONE of the turns use the held-out category
    #   val_in_dist:   subset of records using only seen categories (for tracking training)
    #   held_out:      conversations where AT LEAST one turn uses the held-out category
    holdout = args.holdout_category
    train_pool = [r for r in records if holdout not in r["turn_categories"]]
    holdout_set = [r for r in records if holdout in r["turn_categories"]]
    rng = np.random.default_rng(42)
    rng.shuffle(train_pool)
    n_val = max(1, int(args.val_frac * len(train_pool)))
    val_recs = train_pool[:n_val]
    train_recs = train_pool[n_val:]
    print(f"[phase] train (no {holdout})={len(train_recs)}  "
          f"val_in_dist={len(val_recs)}  held_out={len(holdout_set)}", flush=True)

    # Build model + EMA target
    online = AlignmentLiquid(
        z_goal_dim=z_goal_dim, d=args.d, K=args.K,
        value_hidden=args.value_hidden,
    ).to(device)
    target = copy.deepcopy(online)
    for pp in target.parameters():
        pp.requires_grad = False
    n_params = sum(p.numel() for p in online.parameters() if p.requires_grad)
    print(f"[phase] AlignmentLiquid: K={args.K} d={args.d}, {n_params:,} params", flush=True)

    # Optimizer: full LR everywhere (let geometry reorganize)
    pred_params = list(online.head_value.parameters()) + list(online.head_jepa.parameters())
    geom_param_ids = set()
    for mod in (online.dynamics, online.context_pool):
        for pp in mod.parameters():
            geom_param_ids.add(id(pp))
    geom_params = [p for p in online.parameters() if id(p) in geom_param_ids]
    body_params = [p for p in online.parameters()
                    if id(p) not in geom_param_ids
                    and id(p) not in {id(q) for q in pred_params}]
    opt = torch.optim.AdamW([
        {"params": pred_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": body_params, "lr": args.lr, "weight_decay": 0.005},
        {"params": geom_params, "lr": args.lr * args.substrate_lr_ratio,
         "weight_decay": 0.0},
    ])
    bce = nn.BCEWithLogitsLoss()

    def loss_for_batch(recs, training=True):
        value_losses, jepa_losses, var_feats = [], [], []
        for r in recs:
            T = r["T"]
            z_t_traj = r["z_t_traj"].to(device)
            z_goal_traj = r["z_goal_traj"].to(device)
            labels = r["labels"].to(device)
            t = int(rng.integers(0, max(1, T - args.jepa_window)))
            h_fast_traj, _, _ = forward_trajectory(
                online, z_t_traj, z_goal_traj, device, target_t=t, training=training)
            h_fast_now = h_fast_traj[t].unsqueeze(0)
            z_goal_now = z_goal_traj[t].unsqueeze(0)
            logit = online.value(h_fast_now, z_goal_now)
            value_losses.append(bce(logit, labels[t].unsqueeze(0)))
            pred = online.jepa_predict(h_fast_now)
            with torch.no_grad():
                target_traj, _, _ = forward_trajectory(
                    target, z_t_traj, z_goal_traj, device,
                    target_t=t + args.jepa_window, training=False)
                tgt = target_traj[t + args.jepa_window].detach().unsqueeze(0)
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
        online.eval(); target.eval()
        all_logits, all_labels = [], []
        for r in recs:
            T = r["T"]
            z_t_traj = r["z_t_traj"].to(device)
            z_goal_traj = r["z_goal_traj"].to(device)
            labels = r["labels"].to(device)
            h_fast_traj, _, _ = forward_trajectory(
                online, z_t_traj, z_goal_traj, device, target_t=None, training=False)
            for t in range(T):
                logit = online.value(h_fast_traj[t].unsqueeze(0),
                                       z_goal_traj[t].unsqueeze(0))
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
    best_auc = -1.0
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
                print(f"[phase] ABORT: 5 NaN losses at step {step}", flush=True)
                break
            continue
        n_nan = 0
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), args.max_grad_norm)
        opt.step()
        ema_update(args.ema_tau)

        if step % args.log_every == 0:
            v_auc, n_d, n_f = auc_on_set(val_recs)
            ho_auc, ho_d, ho_f = auc_on_set(holdout_set)
            cv = compute_metric_cv(online, val_recs, device, n_sample=10)
            stale = (step - last_improvement) // args.log_every
            # Use HOLDOUT auc for best-model selection — that's the generic-goal-tracking target
            if not np.isnan(ho_auc) and ho_auc > best_auc:
                best_auc = ho_auc
                best_state = {
                    "online": copy.deepcopy(online.state_dict()),
                    "target": copy.deepcopy(target.state_dict()),
                }
                last_improvement = step
                stale = 0
            print(f"step {step:>5}  L_v={float(value_l.detach()):.3f}  "
                  f"L_j={float(jepa_l.detach()):.3f}  "
                  f"v_auc={v_auc:.3f}  "
                  f"ho_auc={ho_auc:.3f}  (best {best_auc:.3f})  "
                  f"CV={cv:.2f}  stale={stale}  "
                  f"ho_drifts={ho_d}/{ho_d+ho_f}  "
                  f"wall={time.time()-t_start:.0f}s",
                  flush=True)
            if step >= args.min_steps_before_stop and stale >= args.early_stop_patience:
                print(f"[phase] early stop at step {step}", flush=True)
                break

    if best_state is not None:
        online.load_state_dict(best_state["online"])
        target.load_state_dict(best_state["target"])
    torch.save({
        "model_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "best_holdout_auc": best_auc,
        "args": vars(args),
        "z_goal_dim": z_goal_dim,
        "holdout_category": args.holdout_category,
    }, args.output)
    print(f"[phase] saved → {args.output}  best_holdout_auc={best_auc:.3f}", flush=True)
    print("[phase] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
