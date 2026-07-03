"""Train action-modulator using proprio substrate as feature provider.

Architecture (modulator is small MLP, ~10-30K params):
  input:
    state8                         [B, 8]
    chunk_groot                    [B, 16, 7]    flattened
    substrate_preds: pred_dist, p_gripper_moving, pred_state_delta [B, 3]
  output:
    chunk_modulation               [B, 16, 7]    added to chunk_groot

Loss:
  target = chunk_expert
  prediction = chunk_groot + modulation
  L = smooth_L1(prediction, target)

Substrate is FROZEN during modulator training — it provides input features only.
Modulator learns when/how to correct GR00T's chunk based on substrate's signals.

Held-out validation + early stop on val L.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from liquid_goal_tracker_proprio import JEPA_LGT_Proprio  # type: ignore

torch.set_float32_matmul_precision("high")


def load_triples(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def build_episode_index(ep_id, suite):
    keys = [(str(s), int(e)) for s, e in zip(suite, ep_id)]
    eps = {}
    for i, k in enumerate(keys):
        eps.setdefault(k, []).append(i)
    return [sorted(idxs) for idxs in eps.values()]


class Modulator(nn.Module):
    """Small MLP: (state8, chunk_groot_flat, substrate_preds) → chunk_modulation."""

    def __init__(self, state_dim=8, horizon=16, action_dim=7, n_subs=3, d=64,
                 mod_scale=0.3):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.mod_scale = mod_scale
        in_dim = state_dim + horizon * action_dim + n_subs
        out_dim = horizon * action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, d * 2), nn.SiLU(), nn.LayerNorm(d * 2),
            nn.Linear(d * 2, d * 2), nn.SiLU(), nn.LayerNorm(d * 2),
            nn.Linear(d * 2, out_dim),
        )
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, state8, chunk_groot, subs):
        B = state8.shape[0]
        chunk_flat = chunk_groot.reshape(B, -1)
        x = torch.cat([state8, chunk_flat, subs], dim=-1)
        mod_flat = torch.tanh(self.net(x)) * self.mod_scale
        return mod_flat.reshape(B, self.horizon, self.action_dim)


@torch.no_grad()
def substrate_predict(substrate, data, idx, device, dist_mean, dist_std,
                       sd_mean, sd_std, h_carry):
    """Run substrate.step once for triple at index idx; return [pred_dist, p_gripper, pred_state_delta]."""
    z_t = torch.from_numpy(data["z_t"][idx]).to(device).unsqueeze(0)
    chunk_t = torch.from_numpy(data["chunks"][idx]).to(device).unsqueeze(0)
    z_goal = torch.from_numpy(data["z_goal"][idx]).to(device).unsqueeze(0)
    state8 = torch.from_numpy(data["state8_t"][idx]).to(device).unsqueeze(0)
    h_new, p_d, aux, _ = substrate.step(h_carry, z_t, z_goal, chunk_t, state8)
    p_dist = float(p_d) * dist_std + dist_mean
    p_grip = float(torch.sigmoid(aux["pred_gripper_moving_logit"]))
    p_sd = float(aux["pred_state_delta"]) * sd_std + sd_mean
    return h_new, torch.tensor([p_dist, p_grip, p_sd], device=device).float()


def precompute_substrate_features(substrate, data, episodes, device,
                                     dist_mean, dist_std, sd_mean, sd_std):
    """Walk each episode with substrate, cache 3-scalar preds per triple."""
    n = len(data["z_t"])
    feats = np.zeros((n, 3), dtype=np.float32)
    with torch.no_grad():
        for ep_idxs in episodes:
            h = substrate.init_state(1, device)
            for i in ep_idxs:
                h, f = substrate_predict(substrate, data, i, device,
                                           dist_mean, dist_std,
                                           sd_mean, sd_std, h)
                feats[i] = f.cpu().numpy()
    return feats


@torch.no_grad()
def val_loss(modulator, data, episodes, sub_feats, device):
    """Held-out val loss = smooth_L1((chunk_groot + mod), chunk_expert)."""
    losses_mod = []
    losses_identity = []  # chunk_groot vs chunk_expert (modulator does nothing)
    for ep_idxs in episodes:
        for i in ep_idxs:
            state8 = torch.from_numpy(data["state8_t"][i]).to(device).unsqueeze(0)
            ch_groot = torch.from_numpy(data["chunk_groot"][i]).to(device).unsqueeze(0)
            ch_expert = torch.from_numpy(data["chunks"][i]).to(device).unsqueeze(0)
            subs = torch.from_numpy(sub_feats[i]).to(device).unsqueeze(0)
            mod = modulator(state8, ch_groot, subs)
            pred = ch_groot + mod
            losses_mod.append(float(F.smooth_l1_loss(pred, ch_expert, beta=1.0)))
            losses_identity.append(float(F.smooth_l1_loss(ch_groot, ch_expert, beta=1.0)))
    return float(np.mean(losses_mod)), float(np.mean(losses_identity))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_data", required=True)
    p.add_argument("--val_data", required=True)
    p.add_argument("--substrate_ckpt", required=True)
    p.add_argument("--output", default="/tmp/modulator.pt")
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--mod_scale", type=float, default=0.3)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--validate_every", type=int, default=200)
    p.add_argument("--early_stop_patience", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mod] device={device}", flush=True)

    train_data = load_triples(args.train_data)
    val_data = load_triples(args.val_data)
    train_episodes = build_episode_index(train_data["ep_id"], train_data["suite"])
    val_episodes = build_episode_index(val_data["ep_id"], val_data["suite"])

    # Verify chunk_groot exists
    if "chunk_groot" not in train_data:
        print("[mod] FATAL: chunk_groot not in training data — recollect first")
        return

    # Load proprio substrate (frozen)
    ck = torch.load(args.substrate_ckpt, map_location=device, weights_only=False)
    sa = ck["args"]
    substrate = JEPA_LGT_Proprio(
        z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
        horizon=ck["horizon"], state_dim=ck["state_dim"],
        d=sa["d_substrate"], K=sa["K_belief"],
    ).to(device)
    substrate.load_state_dict(ck["substrate_state_dict"])
    substrate.eval()
    for pp in substrate.parameters(): pp.requires_grad = False
    dist_mean = ck["dist_mean"]; dist_std = ck["dist_std"]
    sd_mean = ck["sd_mean"]; sd_std = ck["sd_std"]
    print(f"[mod] substrate loaded (frozen); dist_mean={dist_mean:.2f} "
          f"sd_mean={sd_mean:.4f}", flush=True)

    # Precompute substrate features once
    print(f"[mod] precomputing substrate features on train + val...", flush=True)
    train_feats = precompute_substrate_features(
        substrate, train_data, train_episodes, device,
        dist_mean, dist_std, sd_mean, sd_std)
    val_feats = precompute_substrate_features(
        substrate, val_data, val_episodes, device,
        dist_mean, dist_std, sd_mean, sd_std)
    print(f"[mod] train feats: {train_feats.shape}, val feats: {val_feats.shape}",
          flush=True)

    # Build modulator
    horizon = train_data["chunks"].shape[1]
    action_dim = train_data["chunks"].shape[2]
    state_dim = train_data["state8_t"].shape[1]
    modulator = Modulator(state_dim=state_dim, horizon=horizon,
                            action_dim=action_dim, n_subs=3, d=args.d,
                            mod_scale=args.mod_scale).to(device)
    n_params = sum(pp.numel() for pp in modulator.parameters())
    print(f"[mod] modulator params: {n_params:,}, mod_scale={args.mod_scale}",
          flush=True)
    opt = torch.optim.AdamW(modulator.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    # Initial validation
    init_mod, init_iden = val_loss(modulator, val_data, val_episodes,
                                      val_feats, device)
    print(f"[mod] INIT val: mod_loss={init_mod:.5f}  "
          f"identity_loss={init_iden:.5f}", flush=True)

    best_val = init_mod; best_step = 0; consec_worse = 0
    n_train = len(train_data["z_t"])
    rng = np.random.default_rng(0)
    t_start = time.time()
    train_losses = []
    for step in range(args.max_steps):
        idx = rng.choice(n_train, args.batch_size, replace=False)
        state8 = torch.from_numpy(train_data["state8_t"][idx]).to(device)
        ch_groot = torch.from_numpy(train_data["chunk_groot"][idx]).to(device)
        ch_expert = torch.from_numpy(train_data["chunks"][idx]).to(device)
        subs = torch.from_numpy(train_feats[idx]).to(device)
        mod = modulator(state8, ch_groot, subs)
        pred = ch_groot + mod
        loss = F.smooth_l1_loss(pred, ch_expert, beta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()

        train_losses.append(float(loss))
        train_losses = train_losses[-50:]
        if step % args.log_every == 0:
            print(f"step {step:>5}  train={np.mean(train_losses):.5f}  "
                  f"wall={time.time()-t_start:.0f}s", flush=True)

        if step > 0 and step % args.validate_every == 0:
            v_mod, v_iden = val_loss(modulator, val_data, val_episodes,
                                       val_feats, device)
            print(f"  [VAL step {step}] mod={v_mod:.5f}  iden={v_iden:.5f}  "
                  f"delta={(v_iden-v_mod):+.5f}  (best={best_val:.5f})",
                  flush=True)
            if v_mod < best_val:
                best_val = v_mod; best_step = step
                consec_worse = 0
                torch.save({
                    "modulator_state_dict": modulator.state_dict(),
                    "args": vars(args), "horizon": horizon,
                    "action_dim": action_dim, "state_dim": state_dim,
                    "step": step, "val_mod": v_mod, "val_iden": v_iden,
                    "substrate_ckpt": args.substrate_ckpt,
                }, args.output.replace(".pt", "_best.pt"))
            else:
                consec_worse += 1
                if consec_worse >= args.early_stop_patience:
                    print(f"  [EARLY STOP] best val {best_val:.5f} @step{best_step}",
                          flush=True)
                    break

    torch.save({
        "modulator_state_dict": modulator.state_dict(),
        "args": vars(args), "horizon": horizon,
        "action_dim": action_dim, "state_dim": state_dim,
        "step": args.max_steps, "best_val": best_val, "best_step": best_step,
        "substrate_ckpt": args.substrate_ckpt,
    }, args.output)
    print(f"\n[mod] saved → {args.output}", flush=True)
    print(f"[mod] best val: {best_val:.5f} @step{best_step}", flush=True)


if __name__ == "__main__":
    main()
