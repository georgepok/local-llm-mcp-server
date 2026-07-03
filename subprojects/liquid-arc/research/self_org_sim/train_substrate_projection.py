"""Train substrate's head_projection + head_zvl_residual on (state_t, state_{t+H})
pairs from libero_10 expert demos.

Substrate's role per user spec: "track goal while gr00t is following its target...
extract only goal related state signal and train on it with ability to project
future state and then feed it back to gr00t".

Substrate body frozen. Trains:
- head_projection (h_goal[B,K,d] -> state8 at t+H)
- head_zvl_residual (state8_proj -> z_vl_residual[B, 2048], bounded ±projection_zvl_bound)

The z_vl_residual is what gets fed back to GR00T via get_action_with_zvl_override.
Initial training: supervised on state-prediction only. Residual encoder learns to
map state-projection into z_vl space; weight is small to start.

Optional aux: residual L2 reg (keep small, GR00T should treat as gentle hint).

Usage on Spark:
  python train_substrate_projection.py \
    --substrate_ckpt /tmp/substrate_dynamics_corr.pt \
    --output /tmp/substrate_projection.pt
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_goal_tracker_proprio import JEPA_LGT_Proprio  # type: ignore

torch.set_float32_matmul_precision("high")

DATASET_ROOT = Path("/home/pokazge/datasets")


def collect_projection_data(suite_name, max_episodes, horizon=16, action_horizon=16):
    """Per expert episode, collect (state_t, action_chunk_t, state_{t+horizon}) for
    all valid t where t+horizon is within the same episode.

    The action_chunk gives substrate context about what GR00T is currently planning;
    we want substrate's projection to learn from "given current chunk, where will we be".
    """
    suite_short = suite_name.replace("libero_", "")
    sd = DATASET_ROOT / f"libero-{suite_short}-expert-v1"
    if not sd.exists():
        return None
    idx = np.load(sd / "index.npz")
    starts = idx["episode_starts"]; lengths = idx["episode_lengths"]
    success = idx.get("success_per_episode", np.ones(len(lengths), dtype=bool))
    n_total = int(idx["n_total"])
    states = np.memmap(sd / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))
    labels = np.load(sd / "labels_index.npz")
    sample_idx = labels["sample_idx"]
    n_samples = int(labels["n_samples"])
    chunks_mm = np.memmap(sd / "teacher_chunks.dat", dtype=np.float32, mode="r",
                          shape=(n_samples, action_horizon, 7))
    lookup = {(int(s[0]), int(s[1])): i for i, s in enumerate(sample_idx)}
    succ_eps = [i for i in range(len(lengths)) if bool(success[i])][:max_episodes]

    print(f"  collecting from {len(succ_eps)} episodes (horizon={horizon})...")
    s_curr, c_curr, s_future = [], [], []
    for ep in succ_eps:
        ep_start = int(starts[ep]); ep_len = int(lengths[ep])
        for t in range(0, ep_len - horizon):
            key = (ep, t)
            if key not in lookup:
                continue
            s_t = states[ep_start + t]
            c_t = chunks_mm[lookup[key]]
            s_tH = states[ep_start + t + horizon]
            s_curr.append(np.array(s_t, dtype=np.float32))
            c_curr.append(np.array(c_t, dtype=np.float32))
            s_future.append(np.array(s_tH, dtype=np.float32))
    return (np.stack(s_curr), np.stack(c_curr), np.stack(s_future))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate_ckpt", required=True)
    p.add_argument("--output", default="/tmp/substrate_projection.pt")
    p.add_argument("--suites", default="libero_10")
    p.add_argument("--max_eps_per_suite", type=int, default=50)
    p.add_argument("--horizon", type=int, default=16,
                   help="Env steps ahead to predict. Matches GR00T action_horizon.")
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--zvl_reg_weight", type=float, default=0.001,
                   help="L2 penalty on z_vl_residual to keep it a gentle hint")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[proj] device={device}, output={args.output}, horizon={args.horizon}",
          flush=True)

    # Load substrate, freeze body, unfreeze projection heads only
    ck = torch.load(args.substrate_ckpt, map_location=device, weights_only=False)
    sa = ck["args"]
    substrate = JEPA_LGT_Proprio(
        z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
        horizon=ck["horizon"], state_dim=ck["state_dim"],
        d=sa.get("d_substrate", 64), K=sa.get("K_belief", 4),
        n_tok_per_k=sa.get("n_tok_per_k", 1),
    ).to(device)
    substrate.load_state_dict(ck["substrate_state_dict"], strict=False)
    for pp in substrate.parameters():
        pp.requires_grad = False
    for mod in (substrate.head_projection, substrate.head_zvl_residual):
        for pp in mod.parameters():
            pp.requires_grad = True
    n_trainable = sum(p.numel() for p in substrate.parameters() if p.requires_grad)
    print(f"[proj] training projection heads ({n_trainable:,} params)", flush=True)
    substrate.projection_horizon = args.horizon  # ensure consistency

    # Collect (state_t, chunk_t, state_{t+H}) triples
    print(f"[proj] collecting per-frame triples...", flush=True)
    all_s, all_c, all_sf = [], [], []
    for s in [x.strip() for x in args.suites.split(",") if x.strip()]:
        result = collect_projection_data(s, args.max_eps_per_suite, args.horizon)
        if result is None:
            print(f"  {s}: NO DATA")
            continue
        sc, cc, sf = result
        all_s.append(sc); all_c.append(cc); all_sf.append(sf)
        print(f"  {s}: {sc.shape[0]} triples", flush=True)
    state_curr = np.concatenate(all_s, axis=0)
    chunk_curr = np.concatenate(all_c, axis=0)
    state_future = np.concatenate(all_sf, axis=0)
    n_total = state_curr.shape[0]
    print(f"[proj] total {n_total} triples", flush=True)

    # Train/val split
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_total)
    n_val = int(args.val_frac * n_total)
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    print(f"[proj] train={train_idx.shape[0]} val={val_idx.shape[0]}", flush=True)

    sc_t = torch.from_numpy(state_curr).to(device)
    cc_t = torch.from_numpy(chunk_curr).to(device)
    sf_t = torch.from_numpy(state_future).to(device)

    # Identity baseline for context: predicting "no change" (state stays same)
    iden_val = torch.nn.functional.mse_loss(sc_t[val_idx], sf_t[val_idx]).item()
    print(f"[proj] identity baseline val_MSE: {iden_val:.6f}", flush=True)

    # We don't have real (z_vl, z_goal, language) trajectories paired with state8
    # in the expert data. So at training, use a perturbed init_belief as h_goal.
    # head_projection is anchored on state8 (real, always observable). h_goal is a
    # noisy modulator — model learns to use state8 primarily, h_goal as a corrective.
    # This avoids distribution mismatch between train h_goal and inference h_goal.
    base_h = substrate.init_state(1, device).expand(args.batch_size, -1, -1)
    h_perturb_std = 0.1  # small noise to break degenerate h_goal at training

    opt_params = [pp for pp in substrate.parameters() if pp.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)

    t_start = time.time()
    rolling = {"loss_proj": [], "loss_res": [], "loss_total": []}
    best_val = float("inf")
    for step in range(args.max_steps):
        batch_ids = train_idx[rng.choice(len(train_idx), args.batch_size, replace=False)]
        sc_b = sc_t[batch_ids]
        sf_b = sf_t[batch_ids]

        # h_goal: perturbed init_belief (substrate body frozen; no real h_goal here)
        h_goal_train = base_h + torch.randn_like(base_h) * h_perturb_std

        # Project future state from (real state8, perturbed h_goal)
        state_pred = substrate.project_future_state(sc_b, h_goal_train)
        loss_proj = torch.nn.functional.mse_loss(state_pred, sf_b)

        # Also train z_vl_residual encoder (decoded from projection back to z_vl space).
        # We don't have z_vl ground truth here, so just regularize its norm to stay
        # small (it'll be learned at inference via the inference loop providing real
        # z_vl context). The encoder shape is fixed by training the projection head.
        zvl_res = substrate.encode_projection_residual(state_pred.detach())
        loss_res = (zvl_res ** 2).mean()  # L2 reg only

        loss = loss_proj + args.zvl_reg_weight * loss_res
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, args.max_grad_norm)
        opt.step()

        rolling["loss_proj"].append(float(loss_proj))
        rolling["loss_res"].append(float(loss_res))
        rolling["loss_total"].append(float(loss))
        for k in rolling:
            rolling[k] = rolling[k][-50:]

        if step % args.log_every == 0:
            with torch.no_grad():
                val_bs = min(1024, len(val_idx))
                vids = val_idx[:val_bs]
                vsc = sc_t[vids]; vsf = sf_t[vids]
                # Deterministic h_goal at val: use init_belief (no noise) for repro
                vh_goal = substrate.init_state(1, device).expand(val_bs, -1, -1)
                vstate_pred = substrate.project_future_state(vsc, vh_goal)
                vloss = torch.nn.functional.mse_loss(vstate_pred, vsf).item()
                vzvl = substrate.encode_projection_residual(vstate_pred)
                vzvl_norm = float(vzvl.norm(dim=-1).mean())
            if vloss < best_val:
                best_val = vloss
            print(f"step {step:>4}  L_proj={np.mean(rolling['loss_proj']):.6f}  "
                  f"vL_proj={vloss:.6f} (ident {iden_val:.6f})  "
                  f"best_val={best_val:.6f}  "
                  f"vzvl_norm={vzvl_norm:.3f}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "args": vars(args), "step": args.max_steps,
        "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
        "horizon": ck["horizon"], "state_dim": ck["state_dim"],
        "dist_mean": ck["dist_mean"], "dist_std": ck["dist_std"],
        "sd_mean": ck["sd_mean"], "sd_std": ck["sd_std"],
        "projection_horizon": args.horizon,
        "best_val": best_val,
    }, args.output)
    print(f"\n[proj] saved → {args.output}  best_val={best_val:.6f}", flush=True)


if __name__ == "__main__":
    main()
