"""Record Anymal policy execution as video.

Usage:
    cd /home/pokazge/IsaacLab
    WARP_CACHE_PATH=/home/pokazge/.cache/warp_kernels \
    PYTHONPATH=/home/pokazge/liquid-arc \
    ./isaaclab.sh -p /home/pokazge/liquid-arc/scripts/record_anymal.py \
        --checkpoint /home/pokazge/liquid-arc/output_isaac/anymal_run1.pt \
        --num_envs 4 --n_steps 500
"""

import argparse
import os
import sys
from pathlib import Path

# Set Warp cache BEFORE any imports that trigger Warp init
os.environ["WARP_CACHE_PATH"] = os.path.expanduser("~/.cache/warp")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Parse our args BEFORE AppLauncher touches them
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--n_steps", type=int, default=2000)
parser.add_argument("--output", type=str, default="/home/pokazge/anymal_video")

# Add Isaac Lab's standard args (--headless, --device, etc.)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app (handles headless/display automatically)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now safe to import Isaac Lab modules
import torch
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.direct.anymal_c.anymal_c_env_cfg import AnymalCFlatEnvCfg

from liquid_arc.config import LiquidARCConfig
from liquid_arc.robotics_model import LiquidARCRoboticsModel
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

    # Load base model
    model = LiquidARCRoboticsModel.from_pretrained(
        checkpoint_path="/home/pokazge/liquid-arc/output_30m/checkpoints/step_10000.pt",
        config=config,
        action_dim=12, n_entities=13, n_actuated=12,
        state_dim_per_entity=16, freeze_dynamics=True,
        device=str(device),
    )

    # Load trained weights on top
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    tokenizer = AnymalTokenizer(obs_dim=48)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

    # Get reference to the robot for applying external forces
    robot = env.unwrapped._robot
    n_bodies = robot.num_bodies
    push_interval = 300  # steps between pushes (~2.5s apart)
    push_force = 300.0   # Newtons (strong enough to tilt significantly)
    push_duration = 5    # brief shove

    for step in range(args.n_steps):
        # Apply random destabilizing push
        steps_since_push = step % push_interval
        if steps_since_push == 0 and step > 0:
            # New push — random horizontal direction
            angle = torch.rand(args.num_envs, 1, device=device) * 6.28
            push_dir_x = torch.cos(angle.squeeze())
            push_dir_y = torch.sin(angle.squeeze())
            print(f"  >> PUSH at step {step}! ({push_force}N for {push_duration} steps)")

        if 0 < steps_since_push <= push_duration and step > push_interval:
            # Apply force to base body for push_duration steps
            forces = torch.zeros(args.num_envs, n_bodies, 3, device=device)
            forces[:, 0, 0] = push_dir_x * push_force
            forces[:, 0, 1] = push_dir_y * push_force
            torques = torch.zeros_like(forces)
            robot.set_external_force_and_torque(forces, torques)
            robot.write_data_to_sim()
        elif steps_since_push == push_duration + 1:
            # Clear forces
            forces = torch.zeros(args.num_envs, n_bodies, 3, device=device)
            robot.set_external_force_and_torque(forces, torch.zeros_like(forces))
            robot.write_data_to_sim()

        with torch.no_grad():
            tokens = tokenizer.tokenize(obs)
            result = model(**tokens)
            actions = result['actions']

        obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if step % 100 == 0:
            print(f"  Step {step}/{args.n_steps}, reward={rewards.mean():.2f}")

    env.close()
    print(f"\nDone — {args.n_steps} steps rendered.")
    simulation_app.close()


if __name__ == "__main__":
    main()
