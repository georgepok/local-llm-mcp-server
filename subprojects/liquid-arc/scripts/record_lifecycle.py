"""Visualize trained lifecycle Anymal on DGX Spark monitor.

Usage (run from Spark terminal):
    cd /home/pokazge/IsaacLab
    export WARP_CACHE_PATH=/home/pokazge/.cache/warp
    export PYTHONPATH=/home/pokazge/liquid-arc
    export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
    ./isaaclab.sh -p /home/pokazge/liquid-arc/scripts/record_lifecycle.py \
        --checkpoint /home/pokazge/liquid-arc/output_isaac/lifecycle_anymal_final.pt \
        --num_envs 4 --n_steps 2000
"""

import argparse
import os
import sys
from pathlib import Path

os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
os.environ["WARP_CACHE_PATH"] = os.path.expanduser("~/.cache/warp")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--n_steps", type=int, default=2000)
parser.add_argument("--push_force", type=float, default=300.0)
parser.add_argument("--push_interval", type=int, default=300)

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.anymal_c.anymal_c_env_cfg import AnymalCFlatEnvCfg

from liquid_arc.config import LiquidARCConfig
from liquid_arc.lifecycle import ContinuousLifecycleRunner
from liquid_arc.isaac_wrapper import AnymalTokenizer


def main():
    env_cfg = AnymalCFlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make("Isaac-Velocity-Flat-Anymal-C-Direct-v0", cfg=env_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = LiquidARCConfig(
        d_model=768, d_metric=192, d_ffn=1536,
        n_ode_steps=16, tau_min=0.5, tau_max=1.0,
        t_diffusion_init=1.0, dropout=0.0,
        n_colors=10, n_roles=8, n_sep_types=4,
        max_grid_size=30, max_grids=16, max_seq_len=64,
    )
    config.integration_time = 2.0
    config.persist_alpha = 1.0

    # Create lifecycle model
    model = ContinuousLifecycleRunner.from_pretrained(
        checkpoint_path="/home/pokazge/liquid-arc/output_30m/checkpoints/step_10000.pt",
        config=config,
        n_entities=13, n_actuated=12, action_dim=12, state_dim=16,
        internal_steps=16, autonomous_steps=0,
        freeze_dynamics=True, device=str(device),
    )

    # Load trained lifecycle weights
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt['model']
    cleaned = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.eval()
    print(f"Loaded lifecycle checkpoint: {args.checkpoint}")
    print(f"Beta body: {model.forcing.beta[0].item():.3f}")
    print(f"Beta feet (avg): {model.forcing.beta[1:13].mean().item():.3f}")

    tokenizer = AnymalTokenizer(obs_dim=48)

    # Get robot reference for perturbations
    robot = env.unwrapped._robot
    n_bodies = robot.num_bodies

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
    model.reset(args.num_envs, device)

    for step in range(args.n_steps):
        # Periodic push
        if step > 0 and step % args.push_interval == 0:
            forces = torch.zeros(args.num_envs, n_bodies, 3, device=device)
            angle = torch.rand(args.num_envs, 1, device=device) * 6.28
            forces[:, 0, 0] = torch.cos(angle.squeeze()) * args.push_force
            forces[:, 0, 1] = torch.sin(angle.squeeze()) * args.push_force
            torques = torch.zeros_like(forces)
            robot.set_external_force_and_torque(forces, torques)
            robot.write_data_to_sim()
            print(f"  >> PUSH at step {step}! ({args.push_force}N)")
        elif step > 0 and step % args.push_interval == 5:
            forces = torch.zeros(args.num_envs, n_bodies, 3, device=device)
            robot.set_external_force_and_torque(forces, torch.zeros_like(forces))
            robot.write_data_to_sim()

        with torch.no_grad():
            tokens = tokenizer.tokenize(obs)
            actuated_indices = torch.arange(1, 13, device=device)
            result = model.step(tokens, actuated_indices)
            actions = result['actions'].clamp(-10, 10)

        obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        dones = terminated | truncated

        if dones.any():
            model.handle_resets(dones)

        if step % 100 == 0:
            print(f"  Step {step}/{args.n_steps}, reward={rewards.mean():.2f}, "
                  f"tau={result['tau_mean']:.3f}, cv={result['metric_cv']:.3f}, "
                  f"pred_err={result['prediction_error'].mean():.2f}")

    env.close()
    print(f"\nDone — {args.n_steps} steps rendered.")
    simulation_app.close()


if __name__ == "__main__":
    main()
