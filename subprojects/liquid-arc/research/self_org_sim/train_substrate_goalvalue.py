"""Joint training of substrate + goal-conditional value head.

Unlike train_substrate_twoflow.py which trains for JEPA self-prediction,
this trainer makes the substrate body learn to produce h_fast that's
predictive of GOAL-FOLLOWING under the current goal.

Loss = lambda_pred * L_pred         (JEPA: predict own future state — kept for stability)
     + lambda_value * L_value       (NEW: per-chunk BCE on per-turn goal-followed)
     + lambda_var * L_var           (variance regularizer)

L_value forces in_z, in_lang, in_goal projections + slow/fast dynamics to
encode "am I on goal" — the missing signal in JEPA-only training.

Per-chunk label = turn_followed for that chunk's turn (from raw record metadata).
Loaded from raw_traj alongside rb_traj.

If this works (val AUC > 0.6 with correct orientation on per-turn drift),
the architecture CAN do goal-tracking; it just needs the right supervision.
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
from liquid_goal_tracker_proprio import JEPA_LGT_Proprio


def forward_substrate(substrate, record, device, target_t):
    """Forward substrate through trajectory to target_t. Returns h_fast_traj [T, K, d].
    Gradient flows at target_t; earlier steps detached.
    """
    z_vl = record["z_vl_traj"].float().to(device)
    z_lang = record["z_lang_traj"].float().to(device)
    state8 = record["state8_traj"].float().to(device)
    chunks = record["chunk_traj"].float().to(device)
    z_goal = record["z_goal"].float().to(device).unsqueeze(0)
    T = z_vl.shape[0]
    h_fast = substrate.init_state(1, device)
    h_fast_traj = []
    end_t = min(target_t + 1, T)
    for t in range(end_t):
        z_t = z_vl[t].unsqueeze(0)
        z_l = z_lang[t].unsqueeze(0)
        s_t = state8[t].unsqueeze(0)
        ch_t = chunks[t].unsqueeze(0)
        if t == target_t:
            h_fast, _, _, _ = substrate.step(h_fast, z_t, z_goal, ch_t, s_t, z_lang_t=z_l)
        else:
            with torch.no_grad():
                h_fast, _, _, _ = substrate.step(h_fast, z_t, z_goal, ch_t, s_t, z_lang_t=z_l)
            h_fast = h_fast.detach()
        h_fast_traj.append(h_fast[0])
    return torch.stack(h_fast_traj, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate_starter", required=True)
    p.add_argument("--rb_traj", required=True)
    p.add_argument("--raw_traj", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--substrate_lr_ratio", type=float, default=0.05)
    p.add_argument("--lambda_pred", type=float, default=1.0)
    p.add_argument("--lambda_value", type=float, default=5.0)
    p.add_argument("--lambda_var", type=float, default=0.1)
    p.add_argument("--value_hidden", type=int, default=128)
    p.add_argument("--ema_tau", type=float, default=0.996)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--early_stop_patience", type=int, default=18)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[goalv] device={device}, output={args.output}", flush=True)

    # Build substrate (online + EMA target)
    ck = torch.load(args.substrate_starter, map_location=device, weights_only=False)
    init_belief_shape = ck["substrate_state_dict"]["init_belief"].shape
    K_bel = int(init_belief_shape[0])
    d_sub = int(init_belief_shape[1])
    print(f"[goalv] starter dims: K={K_bel} d={d_sub} z_vl={ck['z_vl_dim']}", flush=True)

    def build():
        s = JEPA_LGT_Proprio(
            z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
            horizon=ck["horizon"], state_dim=ck["state_dim"],
            d=d_sub, K=K_bel, n_tok_per_k=1,
        ).to(device)
        s.load_state_dict(ck["substrate_state_dict"], strict=False)
        s.use_evidence_layernorm = True
        s.h_input_clamp = 50.0
        return s

    online = build()
    target = copy.deepcopy(online)
    for pp in target.parameters():
        pp.requires_grad = False

    # Goal-conditional value head (separate module)
    h_dim = K_bel * d_sub
    g_dim = ck["z_vl_dim"]
    value_head = nn.Sequential(
        nn.Linear(h_dim + g_dim, args.value_hidden),
        nn.SiLU(), nn.LayerNorm(args.value_hidden),
        nn.Linear(args.value_hidden, args.value_hidden),
        nn.SiLU(), nn.LayerNorm(args.value_hidden),
        nn.Linear(args.value_hidden, 1),
    ).to(device)

    # Freeze: dynamics, context_pool (preserve learned dynamics from starter)
    for pp in online.parameters():
        pp.requires_grad = False
    safe_modules = (
        online.in_z, online.in_goal, online.in_delta,
        online.in_action, online.in_state, online.in_lang,
        online.head_jepa_predictor, online.evidence_layernorm,
    )
    for mod in safe_modules:
        for pp in mod.parameters():
            pp.requires_grad = True
    for name in ("action_gate", "goal_gate", "delta_gate", "state_gate",
                  "lang_gate", "init_belief", "evidence_mix"):
        getattr(online, name).requires_grad = True
    n_substrate_trainable = sum(p.numel() for p in online.parameters() if p.requires_grad)
    n_value_trainable = sum(p.numel() for p in value_head.parameters())
    print(f"[goalv] substrate trainable: {n_substrate_trainable:,}  "
          f"value head: {n_value_trainable:,}", flush=True)

    # Data loading
    from train_substrate_twoflow import load_records_with_inputs
    rb_records = load_records_with_inputs([args.rb_traj])
    raw_pack = torch.load(args.raw_traj, map_location="cpu", weights_only=False)
    raw_by_sub = {int(r["sub_id"]): r for r in raw_pack["records"]}
    rb_records = [r for r in rb_records if int(r["sub_id"]) in raw_by_sub]

    # Augment records with per-chunk labels
    for r in rb_records:
        raw = raw_by_sub[int(r["sub_id"])]
        T = r["z_vl_traj"].shape[0]
        turn_starts = list(raw["turn_chunk_starts"])
        turn_followed = list(raw["turn_followed"])
        chunk_labels = []
        cur_turn = 0
        for t in range(T):
            while (cur_turn + 1 < len(turn_starts) and
                   t >= turn_starts[cur_turn + 1]):
                cur_turn += 1
            chunk_labels.append(int(turn_followed[cur_turn]))
        r["chunk_labels"] = torch.tensor(chunk_labels, dtype=torch.float32)

    # Split
    rng = np.random.default_rng(42)
    all_sub_ids = sorted({int(r["sub_id"]) for r in rb_records})
    rng.shuffle(all_sub_ids)
    n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
    val_ids = set(all_sub_ids[:n_val_ids])
    train_records = [r for r in rb_records if int(r["sub_id"]) not in val_ids]
    val_records = [r for r in rb_records if int(r["sub_id"]) in val_ids]
    print(f"[goalv] train={len(train_records)} / val={len(val_records)}", flush=True)

    # Optimizer
    pred_params = list(online.head_jepa_predictor.parameters())
    body_params = [p for n, p in online.named_parameters()
                    if not n.startswith("head_jepa_predictor") and p.requires_grad]
    value_params = list(value_head.parameters())
    opt = torch.optim.AdamW([
        {"params": pred_params, "lr": args.lr},
        {"params": value_params, "lr": args.lr},
        {"params": body_params, "lr": args.lr * args.substrate_lr_ratio},
    ], weight_decay=0.0)

    bce = nn.BCEWithLogitsLoss()

    def loss_for_records(recs, training=True):
        pred_losses, value_losses, var_targets = [], [], []
        for r in recs:
            T = r["z_vl_traj"].shape[0]
            if T < 4:
                continue
            window = max(1, min(3, T - 2))
            t = int(rng.integers(0, T - window))
            if training:
                h_fast_traj = forward_substrate(online, r, device, t)
            else:
                with torch.no_grad():
                    h_fast_traj = forward_substrate(online, r, device, t)
            h_fast_now = h_fast_traj[t].unsqueeze(0)
            # Target h_fast_future from EMA target
            with torch.no_grad():
                h_fast_target_traj = forward_substrate(target, r, device, t + window)
                h_fast_future_target = h_fast_target_traj[t + window].detach().unsqueeze(0)
            chunks_at_t = r["chunk_traj"][t].float().to(device).flatten().unsqueeze(0)
            # JEPA prediction
            pred = online.jepa_predict_future_h_goal(h_fast_now, chunks_at_t)
            pred_losses.append(((pred - h_fast_future_target) ** 2).mean())
            # Goal-conditional value
            z_goal_now = r["z_lang_traj"][t].to(device).unsqueeze(0)  # current turn's goal
            h_flat = h_fast_now.flatten(1)
            x = torch.cat([h_flat, z_goal_now], dim=-1)
            logit = value_head(x).squeeze(-1)
            label = r["chunk_labels"][t].to(device).unsqueeze(0)
            value_losses.append(bce(logit, label))
            var_targets.append(h_fast_now[0].flatten().detach())

        if not pred_losses:
            return None, None, None
        pred_loss = torch.stack(pred_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        # Variance regularizer
        var_all = torch.stack(var_targets, dim=0)
        var_loss = torch.relu(args.lambda_var - var_all.std(dim=0).mean())
        return pred_loss, value_loss, var_loss

    def ema_update(target_sub, online_sub, tau):
        with torch.no_grad():
            for tp, op in zip(target_sub.parameters(), online_sub.parameters()):
                tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)

    def val_eval():
        online.eval()
        value_head.eval()
        # Sample-level AUC: for each chunk in each val record, compute V logit, label
        all_logits, all_labels = [], []
        with torch.no_grad():
            for r in val_records:
                T = r["z_vl_traj"].shape[0]
                h_fast_traj = forward_substrate(online, r, device, T - 1)
                z_lang = r["z_lang_traj"].to(device)
                labels = r["chunk_labels"].to(device)
                h_flat = h_fast_traj.flatten(1)
                x = torch.cat([h_flat, z_lang], dim=-1)
                logits = value_head(x).squeeze(-1)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        logits_np = np.concatenate(all_logits)
        labels_np = np.concatenate(all_labels).astype(int)
        n_drift = int((labels_np == 0).sum())
        n_follow = int((labels_np == 1).sum())
        if n_drift == 0 or n_follow == 0:
            online.train(); value_head.train()
            return float("nan"), n_drift, n_follow
        # AUC predicting DRIFT (label == 0): use -logit as failure-score
        scores = -logits_np
        rs = scores[labels_np == 0]
        rn = scores[labels_np == 1]
        wins = (rs[:, None] > rn[None, :]).sum()
        ties = (rs[:, None] == rn[None, :]).sum()
        auc = (wins + 0.5 * ties) / (n_drift * n_follow)
        online.train(); value_head.train()
        return auc, n_drift, n_follow

    online.train()
    value_head.train()
    best_auc = -1.0
    best_states = None
    last_improvement = 0
    n_nan = 0
    t_start = time.time()

    for step in range(args.max_steps + 1):
        batch = rng.choice(len(train_records), args.batch_size, replace=True)
        recs = [train_records[i] for i in batch]
        out = loss_for_records(recs, training=True)
        if out[0] is None:
            continue
        pred_l, value_l, var_l = out
        total = args.lambda_pred * pred_l + args.lambda_value * value_l + var_l
        if not torch.isfinite(total):
            n_nan += 1
            if n_nan >= 5:
                print(f"[goalv] ABORT: 5 NaN losses at step {step}", flush=True)
                break
            continue
        n_nan = 0
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in online.parameters() if p.requires_grad] + list(value_head.parameters()),
            args.max_grad_norm)
        opt.step()
        ema_update(target, online, args.ema_tau)

        if step % args.log_every == 0:
            v_auc, n_drift, n_follow = val_eval()
            stale = (step - last_improvement) // args.log_every
            if not np.isnan(v_auc) and v_auc > best_auc:
                best_auc = v_auc
                best_states = {
                    "online": copy.deepcopy(online.state_dict()),
                    "value_head": copy.deepcopy(value_head.state_dict()),
                }
                last_improvement = step
                stale = 0
            print(f"step {step:>4}  L_pred={float(pred_l.detach()):.3f}  "
                  f"L_value={float(value_l.detach()):.3f}  v_auc_drift={v_auc:.3f}  "
                  f"(best {best_auc:.3f})  stale={stale}  "
                  f"val drifts={n_drift}/{n_drift+n_follow}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)
            if stale >= args.early_stop_patience:
                print(f"[goalv] early stop at step {step}", flush=True)
                break

    if best_states is not None:
        online.load_state_dict(best_states["online"])
        value_head.load_state_dict(best_states["value_head"])
    torch.save({
        "substrate_state_dict": online.state_dict(),
        "value_head_state_dict": value_head.state_dict(),
        "best_auc_drift": best_auc,
        "args": vars(args),
        "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
        "horizon": ck["horizon"], "state_dim": ck["state_dim"],
        "dist_mean": ck.get("dist_mean", 0.0), "dist_std": ck.get("dist_std", 1.0),
        "sd_mean": ck.get("sd_mean", 0.0), "sd_std": ck.get("sd_std", 1.0),
        "h_dim": h_dim, "g_dim": g_dim, "value_hidden": args.value_hidden,
    }, args.output)
    print(f"[goalv] saved → {args.output}  best_auc_drift={best_auc:.3f}", flush=True)
    print("[goalv] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
