"""Record a single trajectory of the trained shadow_hand vision policy.

Loads `/tmp/sh_dr_multi/dr_final.pt`, runs 1-2 episodes with num_envs=1, captures:
- Camera RGB frames per step → PNG sequence + mp4 (if imageio available)
- State log (step, reward, ep_len, cube_pos, action_norm, n_used) → CSV

Run on Spark host (needs Isaac Sim):
  cd /home/pokazge/IsaacLab
  ./isaaclab.sh -p /home/pokazge/liquid-arc/research/self_org_sim/record_shadow.py \\
      --task Isaac-Repose-Cube-Shadow-Vision-Direct-v0 \\
      --resume /tmp/sh_dr_multi/dr_final.pt \\
      --out_dir /tmp/sh_dr_multi/playback --max_steps 300

Then scp /tmp/sh_dr_multi/playback to view.
"""

from __future__ import annotations

import argparse
import csv
import functools
import os
import sys
from pathlib import Path

print = functools.partial(print, flush=True)

os.environ.setdefault("WARP_CACHE_PATH", os.path.expanduser("~/.cache/warp"))
os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Reuse the policy classes from cartpole_isaac.py (copy here to avoid import
# issues; kept minimal — only what's needed to load + forward-pass)
# ---------------------------------------------------------------------------

class FlatGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, d=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        self.mean_head = nn.Linear(d, action_dim)
        self.value_head = nn.Linear(d, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, obs):
        h = self.encoder(obs)
        return self.mean_head(h), self.value_head(h).squeeze(-1), {
            "steps_mean": torch.tensor(0.0, device=h.device)}


class LiquidGaussianPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, d=64, k_max=16, halt_mode="learned",
                 min_steps=4, dt=0.5, conv_eps=0.01, conv_eps_scale=0.005):
        super().__init__()
        self.k_max = k_max
        self.halt_mode = halt_mode
        self.min_steps = min_steps
        self.dt = dt
        self.conv_eps = conv_eps
        self.conv_eps_scale = conv_eps_scale
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU(),
        )
        self.drift = nn.Sequential(
            nn.Linear(d, d * 2), nn.SiLU(),
            nn.Linear(d * 2, d),
        )
        nn.init.zeros_(self.drift[-1].weight)
        nn.init.zeros_(self.drift[-1].bias)
        self.tau_raw = nn.Parameter(torch.zeros(d))
        if halt_mode == "learned":
            self.halt_head = nn.Linear(d, 1)
            with torch.no_grad():
                self.halt_head.bias.fill_(-3.0)
        self.mean_head = nn.Linear(d, action_dim)
        self.value_head = nn.Linear(d, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, obs):
        h = self.encoder(obs)
        B = h.shape[0]
        tau = F.softplus(self.tau_raw) + 0.1
        steps_used = torch.zeros(B, 1, device=h.device)
        still_active = torch.ones(B, 1, device=h.device)
        for k in range(self.k_max):
            dh = self.drift(h) / tau
            if self.halt_mode == "learned":
                h_new = h + self.dt * dh
                h = still_active * h_new + (1.0 - still_active) * h
                p_halt = torch.sigmoid(self.halt_head(h))
                steps_used = steps_used + still_active
                if k >= self.min_steps:
                    still_active = still_active * (1.0 - p_halt)
            else:
                h = h + self.dt * dh
                steps_used = steps_used + 1.0
        return self.mean_head(h), self.value_head(h).squeeze(-1), {
            "steps_mean": steps_used.mean().detach(),
            "steps_min": steps_used.min().detach(),
            "steps_max": steps_used.max().detach(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="Isaac-Repose-Cube-Shadow-Vision-Direct-v0")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="/tmp/playback")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--policy", choices=["flat", "liquid_fixed", "liquid_halt"],
                        default="liquid_halt")
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--halting_min_steps", type=int, default=4)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--enable_cameras", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=args.headless,
                                enable_cameras=args.enable_cameras)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device="cuda", num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)
    device = torch.device("cuda")

    obs_space = env.unwrapped.single_observation_space["policy"]
    action_space = env.unwrapped.single_action_space
    obs_dim = obs_space.shape[0]
    action_dim = action_space.shape[0]
    print(f"obs_dim={obs_dim} action_dim={action_dim}")

    torch.manual_seed(args.seed)
    if args.policy == "flat":
        policy = FlatGaussianPolicy(obs_dim, action_dim, d=args.d).to(device)
    else:
        halt_mode = "learned" if args.policy == "liquid_halt" else "none"
        policy = LiquidGaussianPolicy(obs_dim, action_dim, d=args.d, k_max=args.k,
                                       halt_mode=halt_mode,
                                       min_steps=args.halting_min_steps).to(device)

    # Load checkpoint
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own_sd = policy.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in own_sd and own_sd[k].shape == v.shape:
            own_sd[k].copy_(v)
            loaded += 1
    print(f"Loaded {loaded} tensors from {args.resume}")
    policy.eval()

    # Try to import imageio for mp4
    try:
        import imageio.v3 as imageio
        has_imageio = True
    except Exception:
        has_imageio = False
        print("imageio not available — saving PNG sequence only")

    # Try to grab tiled camera handle
    tiled_camera = None
    try:
        tiled_camera = env.unwrapped._tiled_camera
    except Exception:
        try:
            tiled_camera = env.unwrapped.scene.sensors.get("tiled_camera", None)
        except Exception:
            pass
    if tiled_camera is None:
        print("WARN: no tiled_camera handle found; will skip frame capture")

    # Open CSV log
    csv_path = out_dir / "trace.csv"
    csv_f = open(csv_path, "w")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["step", "reward", "ep_len", "n_used", "n_used_max", "action_norm",
                     "cube_x", "cube_y", "cube_z"])

    obs, _ = env.reset()
    obs_t = obs["policy"]
    ep_reward = 0.0
    ep_len = 0
    frames_saved = 0
    log_rows = 0

    print(f"Recording up to {args.max_steps} steps to {out_dir}/")
    for step in range(args.max_steps):
        with torch.no_grad():
            mean, value, info = policy(obs_t)
            # Use mean (deterministic) for visualization
            action = mean
        next_obs, reward, terminated, truncated, _ = env.step(action)
        ep_reward += reward[0].item() if reward.numel() > 0 else 0.0
        ep_len += 1

        # Capture camera frame
        if tiled_camera is not None and step % 1 == 0:
            try:
                rgb = tiled_camera.data.output["rgb"][0].detach().cpu().numpy()
                # rgb is HxWx3 uint8
                from PIL import Image
                if rgb.dtype != "uint8":
                    rgb_save = (rgb * 255).clip(0, 255).astype("uint8")
                else:
                    rgb_save = rgb
                Image.fromarray(rgb_save).save(out_dir / "frames" / f"frame_{step:04d}.png")
                frames_saved += 1
            except Exception as e:
                if step == 0:
                    print(f"frame save failed: {e}")

        # Log state
        try:
            cube_pos = env.unwrapped.object_pos[0].detach().cpu().tolist()
        except Exception:
            cube_pos = [0.0, 0.0, 0.0]
        csv_w.writerow([step, float(reward[0].item()), ep_len,
                        float(info.get("steps_mean", torch.tensor(0.0)).item()),
                        float(info.get("steps_max", torch.tensor(0.0)).item()),
                        float(action.norm().item()),
                        cube_pos[0], cube_pos[1], cube_pos[2]])
        log_rows += 1

        if (terminated[0] or truncated[0]).item():
            print(f"  step {step}: episode ended (reward={ep_reward:.2f}, ep_len={ep_len})")
            ep_reward = 0.0
            ep_len = 0
            obs, _ = env.reset()
            obs_t = obs["policy"]
        else:
            obs_t = next_obs["policy"]

        if step % 30 == 0:
            print(f"  step {step}/{args.max_steps}  ep_len={ep_len}  cube_z={cube_pos[2]:.3f}")

    csv_f.close()
    print(f"Saved {frames_saved} frames to {out_dir}/frames/")
    print(f"Saved trace log to {csv_path} ({log_rows} rows)")

    # Encode mp4 from frames
    if has_imageio and frames_saved > 0:
        from PIL import Image
        import numpy as np
        frame_files = sorted((out_dir / "frames").glob("frame_*.png"))
        frames = [np.array(Image.open(f)) for f in frame_files]
        if frames:
            mp4_path = out_dir / "playback.mp4"
            try:
                imageio.imwrite(str(mp4_path), frames, fps=30)
                print(f"Saved mp4: {mp4_path} ({len(frames)} frames @ 30fps)")
            except Exception as e:
                print(f"mp4 encoding failed: {e}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
