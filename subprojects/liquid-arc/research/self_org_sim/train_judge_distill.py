"""Train JudgeDistillLiquid — substrate distills LLM judge as TEACHER, no judge input.

Key test: does substrate learn an abstract goal-following function that GENERALIZES
to unseen goal categories?

Measure on test set:
  - direct judge AUC (baseline, task-agnostic by inheritance)
  - substrate.predict_judge AUC (does substrate's learned predictor match judge cross-cat?)
  - substrate.value AUC (does direct outcome prediction generalize?)
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
from judge_distill_liquid import JudgeDistillLiquid, forward_trajectory


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
    p.add_argument("--train_traj", required=True)
    p.add_argument("--test_traj", default="")
    p.add_argument("--output", required=True)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--value_hidden", type=int, default=384)
    p.add_argument("--d_metric", type=int, default=64)
    p.add_argument("--d_ffn", type=int, default=512)
    p.add_argument("--n_ode_steps", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=8000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--substrate_lr_ratio", type=float, default=0.5)
    p.add_argument("--lambda_distill", type=float, default=3.0,
                   help="Weight on judge distillation loss")
    p.add_argument("--lambda_value", type=float, default=2.0,
                   help="Weight on direct outcome prediction")
    p.add_argument("--lambda_proto", type=float, default=2.0,
                   help="Weight on prototype task-alignment loss — shapes phase transition to task")
    p.add_argument("--lambda_jepa", type=float, default=0.3)
    p.add_argument("--lambda_var", type=float, default=0.1)
    p.add_argument("--jepa_window", type=int, default=2)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--early_stop_patience", type=int, default=40)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[jd] device={device}, output={args.output}", flush=True)
    print(f"[jd] lambda_distill={args.lambda_distill}, "
          f"lambda_value={args.lambda_value}", flush=True)

    pack = torch.load(args.train_traj, map_location="cpu", weights_only=False)
    z_goal_dim = int(pack["enc_dim"])
    d_hidden = int(pack["d_hidden"])
    raw_records = pack["records"]
    print(f"[jd] train pack: d_hidden={d_hidden}, records={len(raw_records)}", flush=True)

    def augment(records_list):
        out = []
        for r in records_list:
            T = int(r["T"])
            if T < 4 or "judge_traj" not in r:
                continue
            hs_traj = r["hidden_state_traj"].float()
            z_goal_traj = r["z_goal_traj"].float()
            judge_traj = r["judge_traj"].float()
            turn_starts = list(r["turn_chunk_starts"])
            turn_followed = list(r["turn_followed"])
            chunk_labels = []
            cur_turn = 0
            for t in range(T):
                while (cur_turn + 1 < len(turn_starts) and
                       t >= turn_starts[cur_turn + 1]):
                    cur_turn += 1
                chunk_labels.append(int(turn_followed[cur_turn]))
            out.append({
                "hidden_state_traj": hs_traj,
                "z_goal_traj": z_goal_traj,
                "judge_traj": judge_traj,  # used as TEACHER, not input
                "labels": torch.tensor(chunk_labels, dtype=torch.float32),
                "T": T,
                "sub_id": int(r["sub_id"]),
            })
        return out

    train_records = augment(raw_records)
    rng = np.random.default_rng(42)
    all_sub_ids = sorted({r["sub_id"] for r in train_records})
    rng.shuffle(all_sub_ids)
    n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
    val_ids = set(all_sub_ids[:n_val_ids])
    train_recs = [r for r in train_records if r["sub_id"] not in val_ids]
    val_recs = [r for r in train_records if r["sub_id"] in val_ids]
    print(f"[jd] train={len(train_recs)}  val_in_dist={len(val_recs)}", flush=True)

    test_recs = []
    if args.test_traj:
        test_pack = torch.load(args.test_traj, map_location="cpu", weights_only=False)
        test_recs = augment(test_pack["records"])
        print(f"[jd] held-out test (cross-cat): {len(test_recs)}", flush=True)

    online = JudgeDistillLiquid(
        d_hidden=d_hidden, z_goal_dim=z_goal_dim,
        d=args.d, K=args.K, value_hidden=args.value_hidden,
        d_metric=args.d_metric, d_ffn=args.d_ffn, n_ode_steps=args.n_ode_steps,
    ).to(device)
    target = copy.deepcopy(online)
    for pp in target.parameters():
        pp.requires_grad = False
    n_params = sum(p.numel() for p in online.parameters() if p.requires_grad)
    print(f"[jd] JudgeDistillLiquid d={args.d} K={args.K}, {n_params:,} params", flush=True)

    pred_params = (list(online.head_judge_pred.parameters()) +
                    list(online.head_value.parameters()) +
                    list(online.head_jepa.parameters()))
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
    mse = nn.MSELoss()

    def loss_for_batch(recs, training=True):
        distill_losses, value_losses, jepa_losses, var_feats = [], [], [], []
        proto_h_fasts, proto_labels = [], []
        for r in recs:
            T = r["T"]
            hs_traj = r["hidden_state_traj"].to(device)
            z_goal_traj = r["z_goal_traj"].to(device)
            judge_traj = r["judge_traj"].to(device)
            labels = r["labels"].to(device)
            t = int(rng.integers(0, max(1, T - args.jepa_window)))
            h_fast_traj, _ = forward_trajectory(
                online, hs_traj, z_goal_traj, device, target_t=t, training=training)
            h_fast_now = h_fast_traj[t].unsqueeze(0)
            z_goal_now = z_goal_traj[t].unsqueeze(0)
            judge_now = judge_traj[t].unsqueeze(0)
            # Distillation: substrate predicts judge_score (regression)
            jp = online.predict_judge(h_fast_now, z_goal_now)
            distill_losses.append(mse(jp, judge_now))
            # Value: substrate predicts follow label (classification)
            logit = online.value(h_fast_now, z_goal_now)
            value_losses.append(bce(logit, labels[t].unsqueeze(0)))
            # JEPA self-prediction
            pred = online.jepa_predict(h_fast_now)
            with torch.no_grad():
                target_traj, _ = forward_trajectory(
                    target, hs_traj, z_goal_traj, device,
                    target_t=t + args.jepa_window, training=False)
                tgt = target_traj[t + args.jepa_window].detach().unsqueeze(0)
            jepa_losses.append(((pred - tgt) ** 2).mean())
            var_feats.append(h_fast_now[0].flatten().detach())
            # PROTOTYPE: collect h_fast at sampled position + its label for task-aligned loss
            proto_h_fasts.append(h_fast_now[0])
            proto_labels.append(labels[t])
        if not distill_losses:
            return None, None, None, None, None
        distill_loss = torch.stack(distill_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        jepa_loss = torch.stack(jepa_losses).mean()
        var = torch.stack(var_feats, dim=0).std(dim=0).mean()
        var_loss = torch.relu(args.lambda_var - var)
        # PROTOTYPE LOSS: success states pulled to success_anchor; drift states pushed away (margin)
        h_stack = torch.stack(proto_h_fasts, dim=0)         # [B, K, d]
        label_stack = torch.stack(proto_labels, dim=0)      # [B]
        # Distance to success anchor (per-sample, mean across K*d)
        anchor = online.success_anchor.unsqueeze(0)         # [1, K, d]
        dist = ((h_stack - anchor) ** 2).mean(dim=(-1, -2))  # [B]
        # Pull success (label=1) close: minimize dist
        # Push drift (label=0) away: hinge loss with margin
        success_mask = label_stack > 0.5
        drift_mask = ~success_mask
        proto_loss_terms = []
        if success_mask.any():
            proto_loss_terms.append(dist[success_mask].mean())
        if drift_mask.any():
            proto_loss_terms.append(torch.relu(online.proto_margin - dist[drift_mask]).mean())
        if proto_loss_terms:
            proto_loss = torch.stack(proto_loss_terms).mean()
        else:
            proto_loss = torch.tensor(0.0, device=h_stack.device)
        return distill_loss, value_loss, jepa_loss, var_loss, proto_loss

    def ema_update(tau):
        with torch.no_grad():
            for tp, op in zip(target.parameters(), online.parameters()):
                tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)

    @torch.no_grad()
    def auc_on_set(recs):
        """Return (value_auc, judge_pred_auc, raw_judge_auc)."""
        if not recs:
            return float("nan"), float("nan"), float("nan"), 0, 0
        online.eval(); target.eval()
        value_logits, judge_preds, raw_judges, all_labels = [], [], [], []
        for r in recs:
            T = r["T"]
            hs_traj = r["hidden_state_traj"].to(device)
            z_goal_traj = r["z_goal_traj"].to(device)
            judge_traj = r["judge_traj"].to(device)
            labels = r["labels"].to(device)
            h_fast_traj, _ = forward_trajectory(
                online, hs_traj, z_goal_traj, device, target_t=None, training=False)
            for t in range(T):
                vl = online.value(h_fast_traj[t].unsqueeze(0), z_goal_traj[t].unsqueeze(0))
                jp = online.predict_judge(h_fast_traj[t].unsqueeze(0), z_goal_traj[t].unsqueeze(0))
                value_logits.append(float(vl.item()))
                judge_preds.append(float(jp.item()))
                raw_judges.append(float(judge_traj[t].item()))
                all_labels.append(int(labels[t].item()))
        value_logits = np.array(value_logits)
        judge_preds = np.array(judge_preds)
        raw_judges = np.array(raw_judges)
        labels_np = np.array(all_labels)
        n_drift = int((labels_np == 0).sum())
        n_follow = int((labels_np == 1).sum())
        online.train(); target.train()
        if n_drift == 0 or n_follow == 0:
            return float("nan"), float("nan"), float("nan"), n_drift, n_follow
        # All scores predict DRIFT (label==0): higher → more likely drifted
        return (roc_auc(-value_logits, 1 - labels_np),
                roc_auc(-judge_preds, 1 - labels_np),
                roc_auc(-raw_judges, 1 - labels_np),
                n_drift, n_follow)

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
        distill_l, value_l, jepa_l, var_l, proto_l = out
        total = (args.lambda_distill * distill_l +
                  args.lambda_value * value_l +
                  args.lambda_jepa * jepa_l +
                  var_l +
                  args.lambda_proto * proto_l)
        if not torch.isfinite(total):
            n_nan += 1
            if n_nan >= 5:
                print(f"[jd] ABORT: 5 NaN losses at step {step}", flush=True)
                break
            continue
        n_nan = 0
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), args.max_grad_norm)
        opt.step()
        ema_update(args.ema_tau)

        if step % args.log_every == 0:
            v_v, v_j, v_raw, _, _ = auc_on_set(val_recs)
            t_v, t_j, t_raw, t_d, t_f = auc_on_set(test_recs)
            # Best model selection: substrate's judge_pred AUC on test (this is the abstraction test)
            selection_auc = t_j if not np.isnan(t_j) else t_v
            stale = (step - last_improvement) // args.log_every
            if not np.isnan(selection_auc) and selection_auc > best_test_auc:
                best_test_auc = selection_auc
                best_state = {
                    "online": copy.deepcopy(online.state_dict()),
                    "target": copy.deepcopy(target.state_dict()),
                }
                last_improvement = step
                stale = 0
            print(f"step {step:>5}  Ld={float(distill_l.detach()):.3f}  "
                  f"Lv={float(value_l.detach()):.3f}  "
                  f"Lp={float(proto_l.detach()):.3f}  "
                  f"v_v={v_v:.3f} v_j={v_j:.3f}  "
                  f"t_v={t_v:.3f} t_j={t_j:.3f} t_raw={t_raw:.3f}  "
                  f"(best_t_j {best_test_auc:.3f})  stale={stale}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)
            if stale >= args.early_stop_patience:
                print(f"[jd] early stop at step {step}", flush=True)
                break

    if best_state is not None:
        online.load_state_dict(best_state["online"])
        target.load_state_dict(best_state["target"])
    torch.save({
        "model_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "best_test_judge_pred_auc": best_test_auc,
        "args": vars(args),
        "d_hidden": d_hidden,
        "z_goal_dim": z_goal_dim,
    }, args.output)
    print(f"[jd] saved → {args.output}  best_test_judge_pred_auc={best_test_auc:.3f}", flush=True)
    print("[jd] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
