"""Train a goal-conditional value head on top of FROZEN existing substrate.

V(h_fast[t], z_goal[t]) → P(goal-followed at this turn)

Goal-conditional supervision:
  - For each chunk t in trajectory, determine which turn it belongs to
  - Target = that turn's `followed` label (0 or 1)
  - z_goal[t] = goal embedding for that turn (z_lang_traj[t] which jumps at turns)
  - L = BCE(V(h_fast[t], z_goal[t]), target)

If this works (AUC > 0.5 on per-turn drift detection with correct orientation),
the inversion problem was about missing goal-conditioning, not the substrate's
inability to learn drift.

If it doesn't work, h_fast itself needs to be retrained with goal-awareness
(Phase 2 joint training).
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
from train_substrate_twoflow import load_records_with_inputs, forward_two_flow_no_grad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate_ckpt", required=True)
    p.add_argument("--rb_traj", required=True)
    p.add_argument("--raw_traj", required=True,
                   help="multigoal raw with turn_chunk_starts, turn_followed metadata")
    p.add_argument("--output", required=True)
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--early_stop_patience", type=int, default=15)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"[val] device={device}", flush=True)

    # Load substrate (frozen)
    ck = torch.load(args.substrate_ckpt, map_location="cpu", weights_only=False)
    init_belief_shape = ck["substrate_state_dict"]["init_belief"].shape
    K_bel = int(init_belief_shape[0])
    d_sub = int(init_belief_shape[1])
    substrate = JEPA_LGT_Proprio(
        z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
        horizon=ck["horizon"], state_dim=ck["state_dim"],
        d=d_sub, K=K_bel, n_tok_per_k=1,
    )
    substrate.load_state_dict(ck["substrate_state_dict"], strict=False)
    substrate.use_evidence_layernorm = True
    substrate.h_input_clamp = 50.0
    substrate = substrate.to(device).eval()
    for pp in substrate.parameters():
        pp.requires_grad = False
    print(f"[val] frozen substrate loaded: K={K_bel} d={d_sub}, z_vl_dim={ck['z_vl_dim']}",
          flush=True)

    # Goal-conditional value head
    h_dim = K_bel * d_sub
    g_dim = ck["z_vl_dim"]
    value_head = nn.Sequential(
        nn.Linear(h_dim + g_dim, args.hidden),
        nn.SiLU(), nn.LayerNorm(args.hidden),
        nn.Linear(args.hidden, args.hidden),
        nn.SiLU(), nn.LayerNorm(args.hidden),
        nn.Linear(args.hidden, 1),
    ).to(device)
    n_params = sum(p.numel() for p in value_head.parameters())
    print(f"[val] value head: ({h_dim} + {g_dim} → {args.hidden} → 1), {n_params:,} params",
          flush=True)

    # Load data + per-turn metadata
    rb_records = load_records_with_inputs([args.rb_traj])
    raw_pack = torch.load(args.raw_traj, map_location="cpu", weights_only=False)
    raw_by_sub = {int(r["sub_id"]): r for r in raw_pack["records"]}

    # Filter records that have raw metadata
    rb_records = [r for r in rb_records if int(r["sub_id"]) in raw_by_sub]
    print(f"[val] {len(rb_records)} records with metadata", flush=True)

    # Train/val split by sub_id
    rng = np.random.default_rng(42)
    all_sub_ids = sorted({int(r["sub_id"]) for r in rb_records})
    rng.shuffle(all_sub_ids)
    n_val_ids = max(1, int(args.val_frac * len(all_sub_ids)))
    val_ids = set(all_sub_ids[:n_val_ids])
    train_ids = set(all_sub_ids[n_val_ids:])
    train_records = [r for r in rb_records if int(r["sub_id"]) in train_ids]
    val_records = [r for r in rb_records if int(r["sub_id"]) in val_ids]
    print(f"[val] train={len(train_records)} / val={len(val_records)}", flush=True)

    # Pre-compute h_fast trajectories (frozen substrate) for all records
    def precompute_h_fast(records):
        out = []
        for r in records:
            T = r["z_vl_traj"].shape[0]
            with torch.no_grad():
                _, h_fast_traj = forward_two_flow_no_grad(substrate, r, device, T - 1)
            raw = raw_by_sub[int(r["sub_id"])]
            turn_starts = list(raw["turn_chunk_starts"])
            turn_followed = list(raw["turn_followed"])
            # Per-chunk labels
            chunk_labels = []
            cur_turn = 0
            for t in range(T):
                while (cur_turn + 1 < len(turn_starts) and
                       t >= turn_starts[cur_turn + 1]):
                    cur_turn += 1
                chunk_labels.append(int(turn_followed[cur_turn]))
            out.append({
                "h_fast": h_fast_traj.detach(),                # [T, K, d]
                "z_goal": r["z_lang_traj"].to(device),         # [T, z_vl_dim] (jumps at turns)
                "labels": torch.tensor(chunk_labels, device=device, dtype=torch.float32),
                "T": T,
            })
        return out

    print(f"[val] precomputing h_fast for train+val (frozen substrate)...", flush=True)
    train_data = precompute_h_fast(train_records)
    val_data = precompute_h_fast(val_records)
    n_train_chunks = sum(d["T"] for d in train_data)
    n_val_chunks = sum(d["T"] for d in val_data)
    print(f"[val] train chunks: {n_train_chunks}, val chunks: {n_val_chunks}", flush=True)

    pos_w = sum(int((d["labels"] == 0).sum().item()) for d in train_data) / max(1,
                sum(int((d["labels"] == 1).sum().item()) for d in train_data))
    print(f"[val] pos_weight for BCE (follow:drift ratio inverted): {pos_w:.2f}", flush=True)
    # We're predicting "followed" (1) vs "not" (0); pos class = followed (majority)
    # For balance: weight the minority (drift) higher → invert pos_weight to weight=0 class
    # Use BCEWithLogitsLoss with pos_weight applied to the rarer class
    # Easier: just use unweighted BCE and look at AUC

    opt = torch.optim.AdamW(value_head.parameters(), lr=args.lr, weight_decay=0.01)
    bce = nn.BCEWithLogitsLoss()

    def val_eval():
        value_head.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for d in val_data:
                h_flat = d["h_fast"].flatten(1)  # [T, K*d]
                x = torch.cat([h_flat, d["z_goal"]], dim=-1)
                logits = value_head(x).squeeze(-1)
                all_preds.append(logits.cpu().numpy())
                all_labels.append(d["labels"].cpu().numpy())
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels).astype(int)
        loss = float(bce(torch.tensor(preds), torch.tensor(labels, dtype=torch.float32)))
        # Use score = -prediction as failure-score (low logit = predicts not-followed = drift)
        # For AUC predicting DRIFT (failure):
        n_pos = int((labels == 0).sum())  # drift count
        n_neg = int((labels == 1).sum())
        if n_pos == 0 or n_neg == 0:
            return loss, float("nan"), n_pos, n_neg
        rs = (-preds)[labels == 0]   # drift scores
        rn = (-preds)[labels == 1]   # follow scores
        wins = (rs[:, None] > rn[None, :]).sum()
        ties = (rs[:, None] == rn[None, :]).sum()
        auc_drift = (wins + 0.5 * ties) / (n_pos * n_neg)
        value_head.train()
        return loss, auc_drift, n_pos, n_neg

    value_head.train()
    best_auc = -1.0
    best_state = None
    last_improvement = 0
    t_start = time.time()
    rng_train = np.random.default_rng(43)

    for step in range(args.max_steps + 1):
        # Sample batch of (record, chunk_t)
        batch_h, batch_g, batch_y = [], [], []
        for _ in range(args.batch_size):
            d = train_data[rng_train.integers(len(train_data))]
            t = int(rng_train.integers(d["T"]))
            batch_h.append(d["h_fast"][t].flatten())  # [K*d]
            batch_g.append(d["z_goal"][t])            # [z_vl_dim]
            batch_y.append(d["labels"][t])
        h = torch.stack(batch_h, dim=0)
        g = torch.stack(batch_g, dim=0)
        y = torch.stack(batch_y, dim=0)
        x = torch.cat([h, g], dim=-1)
        logits = value_head(x).squeeze(-1)
        loss = bce(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % args.log_every == 0:
            v_loss, v_auc, n_drift, n_follow = val_eval()
            stale = (step - last_improvement) // args.log_every
            if v_auc > best_auc:
                best_auc = v_auc
                best_state = copy.deepcopy(value_head.state_dict())
                last_improvement = step
                stale = 0
            print(f"step {step:>4}  L_train={float(loss):.3f}  "
                  f"vL={v_loss:.3f}  v_auc_drift={v_auc:.3f}  "
                  f"(best {best_auc:.3f})  stale={stale}  "
                  f"val_drifts={n_drift}/{n_drift+n_follow}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)
            if stale >= args.early_stop_patience:
                print(f"[val] early stop at step {step}", flush=True)
                break

    if best_state is not None:
        value_head.load_state_dict(best_state)
    torch.save({
        "value_head_state_dict": value_head.state_dict(),
        "best_auc_drift": best_auc,
        "args": vars(args),
        "h_dim": h_dim,
        "g_dim": g_dim,
        "hidden": args.hidden,
    }, args.output)
    print(f"[val] saved → {args.output}  best_auc_drift={best_auc:.3f}", flush=True)
    print("[val] === ALL_DONE ===", flush=True)


if __name__ == "__main__":
    main()
