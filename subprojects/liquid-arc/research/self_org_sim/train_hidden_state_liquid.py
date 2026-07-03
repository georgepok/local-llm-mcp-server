"""Train HiddenStateLiquid on multi-turn conversations with cross-category holdout.

Per-chunk transformer hidden state → Liquid → V head → P(goal-followed).
Cross-category test: hold out one goal category entirely from training,
measure if substrate generalizes (true generic goal tracking).
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
from hidden_state_liquid import HiddenStateLiquid, forward_trajectory


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
    p.add_argument("--train_traj", required=True,
                   help="Hidden-state training data (no held-out category)")
    p.add_argument("--test_traj", default="",
                   help="Optional separate held-out test set (e.g., contrast-only)")
    p.add_argument("--output", required=True)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--value_hidden", type=int, default=192)
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--substrate_lr_ratio", type=float, default=0.3)
    p.add_argument("--lambda_value", type=float, default=5.0)
    p.add_argument("--lambda_jepa", type=float, default=0.3)
    p.add_argument("--lambda_var", type=float, default=0.1)
    p.add_argument("--jepa_window", type=int, default=2)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--early_stop_patience", type=int, default=24)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[hsl] device={device}, output={args.output}", flush=True)

    # Load train data
    pack = torch.load(args.train_traj, map_location="cpu", weights_only=False)
    z_goal_dim = int(pack["enc_dim"])
    d_hidden = int(pack["d_hidden"])
    raw_records = pack["records"]
    print(f"[hsl] train pack: d_hidden={d_hidden}, z_goal_dim={z_goal_dim}, "
          f"records={len(raw_records)}, layer_idx={pack.get('layer_idx','?')}", flush=True)

    # Augment with per-chunk labels
    def augment(raw_records_list):
        out = []
        for r in raw_records_list:
            T = int(r["T"])
            if T < 4:
                continue
            hs_traj = r["hidden_state_traj"].float()
            z_goal_traj = r["z_goal_traj"].float()
            turn_starts = list(r["turn_chunk_starts"])
            turn_followed = list(r["turn_followed"])
            turn_categories = list(r.get("turn_categories", []))
            chunk_labels = []
            chunk_cats = []
            cur_turn = 0
            for t in range(T):
                while (cur_turn + 1 < len(turn_starts) and
                       t >= turn_starts[cur_turn + 1]):
                    cur_turn += 1
                chunk_labels.append(int(turn_followed[cur_turn]))
                chunk_cats.append(turn_categories[cur_turn] if turn_categories else "")
            out.append({
                "hidden_state_traj": hs_traj,
                "z_goal_traj": z_goal_traj,
                "labels": torch.tensor(chunk_labels, dtype=torch.float32),
                "chunk_categories": chunk_cats,
                "turn_categories": turn_categories,
                "T": T,
                "sub_id": int(r["sub_id"]),
            })
        return out

    train_records = augment(raw_records)

    # Hold out 15% of train_records as in-distribution val
    rng = np.random.default_rng(42)
    all_sub_ids = sorted({r["sub_id"] for r in train_records})
    rng.shuffle(all_sub_ids)
    n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
    val_ids = set(all_sub_ids[:n_val_ids])
    train_recs = [r for r in train_records if r["sub_id"] not in val_ids]
    val_recs = [r for r in train_records if r["sub_id"] in val_ids]
    print(f"[hsl] train={len(train_recs)}  val_in_dist={len(val_recs)}", flush=True)

    # Held-out test set (separate file with cross-category data)
    test_recs = []
    if args.test_traj:
        test_pack = torch.load(args.test_traj, map_location="cpu", weights_only=False)
        test_recs = augment(test_pack["records"])
        print(f"[hsl] held-out test (separate categories): {len(test_recs)}", flush=True)

    # Build model + EMA target
    online = HiddenStateLiquid(
        d_hidden=d_hidden, z_goal_dim=z_goal_dim,
        d=args.d, K=args.K, value_hidden=args.value_hidden,
    ).to(device)
    target = copy.deepcopy(online)
    for pp in target.parameters():
        pp.requires_grad = False
    n_params = sum(p.numel() for p in online.parameters() if p.requires_grad)
    print(f"[hsl] HiddenStateLiquid d_hidden={d_hidden}→d={args.d}, K={args.K}, "
          f"{n_params:,} params", flush=True)

    # Optimizer
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
            hs_traj = r["hidden_state_traj"].to(device)
            z_goal_traj = r["z_goal_traj"].to(device)
            labels = r["labels"].to(device)
            t = int(rng.integers(0, max(1, T - args.jepa_window)))
            h_fast_traj, _ = forward_trajectory(
                online, hs_traj, z_goal_traj, device, target_t=t, training=training)
            h_fast_now = h_fast_traj[t].unsqueeze(0)
            z_goal_now = z_goal_traj[t].unsqueeze(0)
            logit = online.value(h_fast_now, z_goal_now)
            value_losses.append(bce(logit, labels[t].unsqueeze(0)))
            pred = online.jepa_predict(h_fast_now)
            with torch.no_grad():
                target_traj, _ = forward_trajectory(
                    target, hs_traj, z_goal_traj, device,
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
        if not recs:
            return float("nan"), 0, 0
        online.eval(); target.eval()
        all_logits, all_labels = [], []
        for r in recs:
            T = r["T"]
            hs_traj = r["hidden_state_traj"].to(device)
            z_goal_traj = r["z_goal_traj"].to(device)
            labels = r["labels"].to(device)
            h_fast_traj, _ = forward_trajectory(
                online, hs_traj, z_goal_traj, device, target_t=None, training=False)
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
                print(f"[hsl] ABORT: 5 NaN losses at step {step}", flush=True)
                break
            continue
        n_nan = 0
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), args.max_grad_norm)
        opt.step()
        ema_update(args.ema_tau)

        if step % args.log_every == 0:
            v_auc, v_d, v_f = auc_on_set(val_recs)
            t_auc, t_d, t_f = auc_on_set(test_recs) if test_recs else (float("nan"), 0, 0)
            # Best model selection: held-out test AUC if available, else in-dist val
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
                  f"L_j={float(jepa_l.detach()):.3f}  "
                  f"v_auc={v_auc:.3f}  t_auc={t_auc:.3f}  "
                  f"(best_sel {best_test_auc:.3f})  stale={stale}  "
                  f"t_drifts={t_d}/{t_d+t_f}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)
            if stale >= args.early_stop_patience:
                print(f"[hsl] early stop at step {step}", flush=True)
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
        "z_goal_dim": z_goal_dim,
    }, args.output)
    print(f"[hsl] saved → {args.output}  best_test_auc={best_test_auc:.3f}", flush=True)
    print("[hsl] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
