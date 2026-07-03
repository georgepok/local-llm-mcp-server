"""Train AlignmentLiquid on multi-turn conversations with per-turn drift labels.

Loss:
  L_value (BCE) per chunk on turn_followed — primary supervision
  L_jepa (MSE) per chunk on next h_fast — trajectory smoothness regularizer
  L_var — variance regularizer (prevents collapse)

EMA target substrate stabilizes JEPA loss.

Target: beat cosine baseline (AUC 0.611) — substrate must add value beyond raw cos.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text_traj", required=True,
                   help="raw multigoal trajectory file with metadata")
    p.add_argument("--output", required=True)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--value_hidden", type=int, default=128)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--substrate_lr_ratio", type=float, default=0.1)
    p.add_argument("--lambda_value", type=float, default=5.0)
    p.add_argument("--lambda_jepa", type=float, default=0.5)
    p.add_argument("--lambda_var", type=float, default=0.1)
    p.add_argument("--jepa_window", type=int, default=2)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--early_stop_patience", type=int, default=18)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[align] device={device}, output={args.output}", flush=True)

    # Load raw multigoal data (has z_t_traj, z_lang_traj which serves as z_goal, turn metadata)
    pack = torch.load(args.text_traj, map_location="cpu", weights_only=False)
    z_goal_dim = int(pack["enc_dim"])
    raw_records = pack["records"]

    # Build augmented records: per-chunk turn labels + tensors on CPU
    records = []
    for r in raw_records:
        T = int(r["T"])
        if T < 4:
            continue
        z_t_traj = r["z_t_traj"].float()       # [T, dim]
        # In multi-turn data, z_lang_traj IS the per-chunk goal (jumps at turns)
        z_goal_traj = r["z_lang_traj"].float()  # [T, dim]
        turn_starts = list(r["turn_chunk_starts"])
        turn_followed = list(r["turn_followed"])
        chunk_labels = []
        cur_turn = 0
        for t in range(T):
            while (cur_turn + 1 < len(turn_starts) and
                   t >= turn_starts[cur_turn + 1]):
                cur_turn += 1
            chunk_labels.append(int(turn_followed[cur_turn]))
        records.append({
            "z_t_traj": z_t_traj,
            "z_goal_traj": z_goal_traj,
            "labels": torch.tensor(chunk_labels, dtype=torch.float32),
            "T": T,
            "sub_id": int(r["sub_id"]),
        })
    print(f"[align] {len(records)} records loaded, z_goal_dim={z_goal_dim}", flush=True)

    # Train/val split
    rng = np.random.default_rng(42)
    all_sub_ids = sorted({r["sub_id"] for r in records})
    rng.shuffle(all_sub_ids)
    n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
    val_ids = set(all_sub_ids[:n_val_ids])
    train_recs = [r for r in records if r["sub_id"] not in val_ids]
    val_recs = [r for r in records if r["sub_id"] in val_ids]
    print(f"[align] train={len(train_recs)} / val={len(val_recs)}", flush=True)

    # Build model + EMA target
    online = AlignmentLiquid(
        z_goal_dim=z_goal_dim, d=args.d, K=args.K,
        value_hidden=args.value_hidden,
    ).to(device)
    target = copy.deepcopy(online)
    for pp in target.parameters():
        pp.requires_grad = False
    n_params = sum(p.numel() for p in online.parameters() if p.requires_grad)
    print(f"[align] AlignmentLiquid: K={args.K} d={args.d}, {n_params:,} params", flush=True)

    # Optimizer groups: dynamics+context_pool slower, rest at base LR
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
            # Pick random t with room for JEPA window
            t = int(rng.integers(0, max(1, T - args.jepa_window)))
            # Forward to t with gradient
            h_fast_traj, _, _ = forward_trajectory(
                online, z_t_traj, z_goal_traj, device, target_t=t, training=training)
            h_fast_now = h_fast_traj[t].unsqueeze(0)
            z_goal_now = z_goal_traj[t].unsqueeze(0)
            # Value prediction
            logit = online.value(h_fast_now, z_goal_now)
            value_losses.append(bce(logit, labels[t].unsqueeze(0)))
            # JEPA prediction
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
    def val_eval():
        online.eval(); target.eval()
        all_logits, all_labels = [], []
        for r in val_recs:
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
        # AUC predicting DRIFT (label==0): use -logit
        if n_drift == 0 or n_follow == 0:
            online.train(); target.train()
            return float("nan"), n_drift, n_follow
        scores = -logits_np
        rs = scores[labels_np == 0]
        rn = scores[labels_np == 1]
        wins = (rs[:, None] > rn[None, :]).sum()
        ties = (rs[:, None] == rn[None, :]).sum()
        auc = (wins + 0.5 * ties) / (n_drift * n_follow)
        online.train(); target.train()
        return auc, n_drift, n_follow

    online.train()
    target.train()
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
                print(f"[align] ABORT: 5 NaN losses at step {step}", flush=True)
                break
            continue
        n_nan = 0
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), args.max_grad_norm)
        opt.step()
        ema_update(args.ema_tau)

        if step % args.log_every == 0:
            v_auc, n_d, n_f = val_eval()
            stale = (step - last_improvement) // args.log_every
            if not np.isnan(v_auc) and v_auc > best_auc:
                best_auc = v_auc
                best_state = {
                    "online": copy.deepcopy(online.state_dict()),
                    "target": copy.deepcopy(target.state_dict()),
                }
                last_improvement = step
                stale = 0
            print(f"step {step:>4}  L_value={float(value_l.detach()):.3f}  "
                  f"L_jepa={float(jepa_l.detach()):.3f}  v_auc_drift={v_auc:.3f}  "
                  f"(best {best_auc:.3f})  stale={stale}  "
                  f"val drifts={n_d}/{n_d+n_f}  wall={time.time()-t_start:.0f}s",
                  flush=True)
            if stale >= args.early_stop_patience:
                print(f"[align] early stop at step {step}", flush=True)
                break

    if best_state is not None:
        online.load_state_dict(best_state["online"])
        target.load_state_dict(best_state["target"])
    torch.save({
        "model_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "best_auc_drift": best_auc,
        "args": vars(args),
        "z_goal_dim": z_goal_dim,
    }, args.output)
    print(f"[align] saved → {args.output}  best_auc_drift={best_auc:.3f}", flush=True)
    print("[align] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
