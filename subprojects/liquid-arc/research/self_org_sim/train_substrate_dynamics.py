"""Train substrate's forward dynamics head on per-frame (state8_t, action_t, state8_{t+1})
triples extracted from libero expert demos.

The dynamics head predicts state8_{t+1} given (state8_t, action7_t, h_intent_pool).
h_intent comes from substrate.step() with the chunk emitted at the start of the gap.

Training strategy:
- For each expert episode, iterate consecutive frames t.
- state8_t = states_mm[ep_start + t]
- action_t = chunks_mm[lookup[(ep,t)]][0]  (first action of chunk emitted at frame t)
- state8_{t+1} = states_mm[ep_start + t + 1]
- h_intent = substrate.step(... chunk_t, state8_t) → use the K=4 belief

Loss: MSE(predicted_state8_next, actual_state8_next)

Trains ONLY dyn_in_state/action/intent + head_dynamics; substrate core stays frozen.
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


def collect_dynamics_triples(suite_name, max_episodes, action_horizon=16):
    """Extract per-frame quintuples (s_t, a_t, s_{t+1}, a_{t+1}, s_{t+2}) per episode.

    s_{t+2} and a_{t+1} are needed to train head_correction via rollout-through-dynamics:
    deviation = s_{t+1} - dynamics(s_t, a_t); correction = head_correction(deviation, h_intent);
    target = MSE(dynamics(s_{t+1}, a_{t+1} + correction), s_{t+2}).
    """
    suite_short = suite_name.replace("libero_", "")
    sd = DATASET_ROOT / f"libero-{suite_short}-expert-v1"
    if not sd.exists():
        return None
    idx = np.load(sd / "index.npz")
    starts = idx["episode_starts"]; lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
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
    succ_eps = [i for i in range(len(lengths)) if bool(success[i])]
    succ_eps = succ_eps[:max_episodes]

    print(f"  collecting from {len(succ_eps)} episodes...")
    s_curr, a_curr, s_next, a_next, s_next2 = [], [], [], [], []
    for ep in succ_eps:
        ep_start = int(starts[ep]); ep_len = int(lengths[ep])
        for t in range(0, ep_len - 2):  # need t, t+1, t+2 to exist
            key_t = (ep, t); key_t1 = (ep, t + 1)
            if key_t not in lookup or key_t1 not in lookup:
                continue
            s_t = states[ep_start + t]
            a_t = chunks_mm[lookup[key_t]][0]   # chunk[0] @ frame t
            s_t1 = states[ep_start + t + 1]
            a_t1 = chunks_mm[lookup[key_t1]][0]  # chunk[0] @ frame t+1
            s_t2 = states[ep_start + t + 2]
            s_curr.append(np.array(s_t, dtype=np.float32))
            a_curr.append(np.array(a_t, dtype=np.float32))
            s_next.append(np.array(s_t1, dtype=np.float32))
            a_next.append(np.array(a_t1, dtype=np.float32))
            s_next2.append(np.array(s_t2, dtype=np.float32))
    return (np.stack(s_curr), np.stack(a_curr), np.stack(s_next),
            np.stack(a_next), np.stack(s_next2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate_ckpt", required=True,
                   help="Proprio substrate ckpt (dynamics+correction heads trained, rest frozen)")
    p.add_argument("--output", default="/tmp/substrate_dynamics.pt")
    p.add_argument("--suites", default="libero_10")
    p.add_argument("--max_eps_per_suite", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--corr_loss_weight", type=float, default=1.0,
                   help="Weight on correction-head rollout-through-dynamics loss")
    p.add_argument("--corr_reg_weight", type=float, default=0.01,
                   help="L2 penalty on correction magnitude (minimum-intervention prior)")
    p.add_argument("--dyn_warmup_steps", type=int, default=500,
                   help="Train dynamics only for first N steps before correction loss kicks in")
    p.add_argument("--perturb_prob", type=float, default=0.5,
                   help="Per-batch probability of injecting perturbation to amplify deviation")
    p.add_argument("--perturb_sigma", type=float, default=0.03,
                   help="Std of synthetic state perturbation on s_t (in raw state units)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dyn] device={device}, output={args.output}", flush=True)

    # Load substrate (just for h_intent generation — we use its existing weights)
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
    # Unfreeze ONLY dynamics + correction heads
    for mod in (substrate.dyn_in_state, substrate.dyn_in_action,
                  substrate.dyn_in_intent, substrate.head_dynamics,
                  substrate.head_correction):
        for pp in mod.parameters():
            pp.requires_grad = True
    n_trainable = sum(pp.numel() for pp in substrate.parameters() if pp.requires_grad)
    print(f"[dyn] training dynamics+correction heads ({n_trainable:,} params)", flush=True)

    # Collect quintuples (s_t, a_t, s_{t+1}, a_{t+1}, s_{t+2})
    print(f"[dyn] collecting per-frame quintuples...", flush=True)
    all_s_curr = []; all_a_curr = []; all_s_next = []
    all_a_next = []; all_s_next2 = []
    for s in [x.strip() for x in args.suites.split(",") if x.strip()]:
        result = collect_dynamics_triples(s, args.max_eps_per_suite)
        if result is None:
            print(f"  {s}: NO DATA")
            continue
        sc, ac, sn, an, sn2 = result
        all_s_curr.append(sc); all_a_curr.append(ac); all_s_next.append(sn)
        all_a_next.append(an); all_s_next2.append(sn2)
        print(f"  {s}: {sc.shape[0]} quintuples", flush=True)
    state_curr = np.concatenate(all_s_curr, axis=0)
    action_curr = np.concatenate(all_a_curr, axis=0)
    state_next = np.concatenate(all_s_next, axis=0)
    action_next = np.concatenate(all_a_next, axis=0)
    state_next2 = np.concatenate(all_s_next2, axis=0)
    n_total = state_curr.shape[0]
    print(f"[dyn] total {n_total} quintuples", flush=True)

    # Shuffle + split train/val
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_total)
    n_val = int(args.val_frac * n_total)
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    print(f"[dyn] train={train_idx.shape[0]} val={val_idx.shape[0]}", flush=True)

    # Convert to tensors
    sc_t = torch.from_numpy(state_curr).to(device)
    ac_t = torch.from_numpy(action_curr).to(device)
    sn_t = torch.from_numpy(state_next).to(device)
    an_t = torch.from_numpy(action_next).to(device)
    sn2_t = torch.from_numpy(state_next2).to(device)

    opt_params = [pp for pp in substrate.parameters() if pp.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)

    # h_intent uses substrate's frozen init_belief (constant generic intent). The
    # dynamics+correction heads learn from (state, action, deviation) primarily.
    h_intent_train = substrate.init_state(1, device).expand(args.batch_size, -1, -1)

    t_start = time.time()
    rolling = {"loss_dyn": [], "loss_corr": [], "loss_total": [],
               "corr_norm": [], "dev_norm": []}
    best_val = float("inf")
    for step in range(args.max_steps):
        batch_ids = train_idx[rng.choice(len(train_idx), args.batch_size, replace=False)]
        bc = sc_t[batch_ids]
        ba = ac_t[batch_ids]
        bn = sn_t[batch_ids]
        ban = an_t[batch_ids]
        bn2 = sn2_t[batch_ids]

        # --- Dynamics loss (single-step prediction) ---
        pred = substrate.forward_dynamics(bc, ba, h_intent_train)
        loss_dyn = torch.nn.functional.mse_loss(pred, bn)

        # --- Correction loss (drift-compensation via rollout-through-dynamics) ---
        # Scenario: at gap step k+1, actual state has drifted by δ from the trajectory
        # GR00T's chunk assumed. The correction head should output an action delta that
        # when added to chunk[k+1] steers us back toward the expert's next state s_{t+2}.
        # We simulate this with synthetic drift δ on s_{t+1}.
        loss_corr = torch.tensor(0.0, device=device)
        loss_reg = torch.tensor(0.0, device=device)
        if step >= args.dyn_warmup_steps and args.corr_loss_weight > 0:
            # Half the batch gets drift; the other half stays clean so correction learns
            # to output ~0 when no drift is observed.
            drift_mask = (torch.rand(bn.shape[0], device=device) <
                          args.perturb_prob).float().unsqueeze(-1)
            drift = torch.randn_like(bn) * args.perturb_sigma * drift_mask  # [B, 8]
            s_drift = bn + drift  # drifted s_{t+1}

            # Deviation as seen at runtime: actual_observed - predicted_from_previous.
            # With well-trained dynamics, deviation ≈ drift (the dynamics prediction
            # from the un-drifted previous step would land on bn, but observation is bn+drift).
            deviation = drift  # [B, 8] — pure drift signal, exactly what's observable

            # Correction conditioned on deviation
            correction = substrate.compute_correction(deviation, h_intent_train)  # [B, 7] bounded ±0.05

            # Rollout from DRIFTED state, with corrected vs uncorrected action.
            # Loss pulls toward expert s_{t+2} → correction must learn to compensate drift.
            corrected_action = ban + correction
            pred_t2 = substrate.forward_dynamics(s_drift, corrected_action, h_intent_train)
            loss_corr = torch.nn.functional.mse_loss(pred_t2, bn2)
            loss_reg = (correction ** 2).mean()
            rolling["corr_norm"].append(float(correction.norm(dim=-1).mean()))
            rolling["dev_norm"].append(float(deviation.norm(dim=-1).mean()))

        loss = loss_dyn + args.corr_loss_weight * loss_corr + args.corr_reg_weight * loss_reg

        opt.zero_grad()
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(opt_params, args.max_grad_norm)
        opt.step()
        rolling["loss_dyn"].append(float(loss_dyn))
        rolling["loss_corr"].append(float(loss_corr))
        rolling["loss_total"].append(float(loss))
        for k in rolling:
            rolling[k] = rolling[k][-50:]

        if step % args.log_every == 0:
            # Val (both dynamics and correction-rollout, no perturbation, deterministic)
            with torch.no_grad():
                val_bs = min(1024, len(val_idx))
                val_ids = val_idx[:val_bs]
                vc = sc_t[val_ids]; va = ac_t[val_ids]; vn = sn_t[val_ids]
                van = an_t[val_ids]; vn2 = sn2_t[val_ids]
                h_intent_val = substrate.init_state(1, device).expand(val_bs, -1, -1)

                vpred = substrate.forward_dynamics(vc, va, h_intent_val)
                vloss_dyn = torch.nn.functional.mse_loss(vpred, vn).item()
                v_ident = torch.nn.functional.mse_loss(vc, vn).item()

                # Correction val: apply fixed drift pattern at s_{t+1}, measure how well
                # correction steers back to expert s_{t+2}.
                rng_val = torch.Generator(device=device).manual_seed(123)
                vdrift = torch.randn(vn.shape, device=device,
                                     generator=rng_val) * args.perturb_sigma
                vs_drift = vn + vdrift
                vdev = vdrift  # deviation as seen at runtime
                vcorr = substrate.compute_correction(vdev, h_intent_val)
                vpred_t2 = substrate.forward_dynamics(vs_drift, van + vcorr, h_intent_val)
                vpred_t2_nc = substrate.forward_dynamics(vs_drift, van, h_intent_val)
                vloss_corr = torch.nn.functional.mse_loss(vpred_t2, vn2).item()
                vloss_nocorr = torch.nn.functional.mse_loss(vpred_t2_nc, vn2).item()
                v_corr_norm = float(vcorr.norm(dim=-1).mean())
                v_dev_norm = float(vdev.norm(dim=-1).mean())

            score = vloss_dyn + (vloss_corr if step >= args.dyn_warmup_steps else 0.0)
            if score < best_val:
                best_val = score
            cn = (np.mean(rolling["corr_norm"]) if rolling["corr_norm"] else 0.0)
            dn = (np.mean(rolling["dev_norm"]) if rolling["dev_norm"] else 0.0)
            print(f"step {step:>4}  "
                  f"L_dyn={np.mean(rolling['loss_dyn']):.6f} "
                  f"L_corr={np.mean(rolling['loss_corr']):.6f}  "
                  f"vL_dyn={vloss_dyn:.6f} (ident={v_ident:.6f})  "
                  f"vL_corr={vloss_corr:.6f} (no_corr={vloss_nocorr:.6f})  "
                  f"corr_norm={cn:.4f} (val={v_corr_norm:.4f})  "
                  f"dev_norm={dn:.4f} (val={v_dev_norm:.4f})  "
                  f"wall={time.time()-t_start:.0f}s",
                  flush=True)

        if args.ckpt_every > 0 and step > 0 and step % args.ckpt_every == 0:
            torch.save({
                "substrate_state_dict": substrate.state_dict(),
                "args": vars(args), "step": step,
                "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
                "horizon": ck["horizon"], "state_dim": ck["state_dim"],
                "dist_mean": ck["dist_mean"], "dist_std": ck["dist_std"],
                "sd_mean": ck["sd_mean"], "sd_std": ck["sd_std"],
            }, args.output.replace(".pt", f"_step{step}.pt"))

    torch.save({
        "substrate_state_dict": substrate.state_dict(),
        "args": vars(args), "step": args.max_steps,
        "z_vl_dim": ck["z_vl_dim"], "action_dim": ck["action_dim"],
        "horizon": ck["horizon"], "state_dim": ck["state_dim"],
        "dist_mean": ck["dist_mean"], "dist_std": ck["dist_std"],
        "sd_mean": ck["sd_mean"], "sd_std": ck["sd_std"],
    }, args.output)
    print(f"\n[dyn] saved → {args.output}  best_val={best_val:.6f}", flush=True)


if __name__ == "__main__":
    main()
