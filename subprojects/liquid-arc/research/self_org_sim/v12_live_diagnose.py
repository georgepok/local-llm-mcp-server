"""Diagnose v12 by running it on a LIVE sim's first frame and comparing
chunk-for-chunk against what v10-DEMO predicts in the SAME scenario.

This bypasses the question of whether training-data distributions match
sim distributions — we feed the SAME sim observation to both models and
look at what they actually output. If v12's chunk is systematically off
from v10's on identical input, the bug is in v12's model itself. If v12's
chunk matches v10's but sim still fails, the bug is elsewhere.

Run on Spark in main venv (CUDA):
  python v12_live_diagnose.py
"""
import os, sys
import numpy as np
import torch
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_v12(ckpt_path, device):
    from distill_groot_v12 import V12Policy
    from liquid_arc_substrate_libero import make_v12_config
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = make_v12_config()
    for k, v in ckpt["config"].items():
        if hasattr(config, k):
            setattr(config, k, v)
    model = V12Policy(config, action_horizon=16, action_dim=7, state_dim=8,
                     head_d=256, head_layers=4, head_heads=4, use_goal_img=False).to(device)
    sd = ckpt["policy"]
    own = model.state_dict()
    for k, v in sd.items():
        kk = k.replace("_orig_mod.", "")
        if kk in own and own[kk].shape == v.shape:
            own[kk].copy_(v)
    model.eval()
    return model


def load_v10(ckpt_path, device):
    from distill_groot_flow import LiquidFlowPolicy
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ckpt["args"]
    halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
    model = LiquidFlowPolicy(
        state_dim=8, action_dim=7, action_horizon=sa["action_horizon"],
        d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"], k_max=sa["k"],
        halt_mode=halt_mode, min_steps=sa["halting_min_steps"],
        n_tasks=sa["n_tasks"], d_task=sa["d_task"],
        head_d=sa["head_d"], head_layers=sa["head_layers"], head_heads=sa["head_heads"],
        n_task_heads=sa.get("n_task_heads", 0),
        z_groot_dim=sa.get("z_groot_dim", 0),
        gated_mixture=sa.get("gated_mixture", False),
        z_channel_dims=sa.get("z_channel_dims", None),
        query_bank=sa.get("use_query_bank", False),
        query_dim=sa.get("query_dim", 8),
        gripper_head=sa.get("gripper_head", False),
        pretrained_vision=sa.get("pretrained_vision", ""),
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
    model.eval()
    return model


def get_sim_first_frame(suite_name, sim_id):
    """Return the first observation from a fresh LIBERO sim of given task."""
    import math
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(sim_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    init_states = suite.get_task_init_states(sim_id)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.reset()
    env.set_init_state(init_states[0])
    obs = None
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
    img = obs["agentview_image"].copy()
    wrist = obs["robot0_eye_in_hand_image"].copy()
    xyz = obs["robot0_eef_pos"]
    quat = obs["robot0_eef_quat"]
    if quat[3] > 1: quat[3] = 1
    elif quat[3] < -1: quat[3] = -1
    den = math.sqrt(1.0 - quat[3]**2)
    if math.isclose(den, 0.0):
        rpy = np.zeros(3, dtype=np.float32)
    else:
        rpy = (quat[:3] * 2.0 * math.acos(quat[3])) / den
    grip = obs["robot0_gripper_qpos"]
    state = np.concatenate([xyz, rpy, grip]).astype(np.float32)
    env.close()
    return img, wrist, state, task.language


@torch.no_grad()
def predict_v12_chunk(model, img, wrist, state, device, seed=42):
    torch.manual_seed(seed)
    from PIL import Image
    img224 = np.array(Image.fromarray(img).resize((224, 224)), dtype=np.uint8)
    wri224 = np.array(Image.fromarray(wrist).resize((224, 224)), dtype=np.uint8)
    img_t = torch.from_numpy(img224).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wri224).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(state).to(device).float().unsqueeze(0)
    out = model.encode(img_t, wri_t, st_t)
    cond = out["cond"]
    x = torch.randn(1, 16, 7, device=device)
    n_steps = 10
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = torch.full((1,), i * dt, device=device)
        v = model.velocity(x, t_val, cond)
        x = x + dt * v
    return x[0].cpu().numpy()


@torch.no_grad()
def predict_v10_chunk(model, img, wrist, state, device, seed=42):
    torch.manual_seed(seed)
    from PIL import Image
    img224 = np.array(Image.fromarray(img).resize((224, 224)), dtype=np.uint8)
    wri224 = np.array(Image.fromarray(wrist).resize((224, 224)), dtype=np.uint8)
    img_t = torch.from_numpy(img224).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wri224).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(state).to(device).float().unsqueeze(0)
    chunk = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=10)
    return chunk[0].cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading models...")
    v12 = load_v12("/tmp/distill_v12/step_008000.pt", device)
    v10 = load_v10("/tmp/distill_v10_goal/step_008000.pt", device)

    print("\nGetting fresh sim observations...")
    for sim_id in [0, 3]:  # alphabet soup + drawer task
        img, wrist, state, task_lang = get_sim_first_frame("libero_10", sim_id)
        print(f"\n=== libero_10 sim{sim_id}: {task_lang[:80]} ===")
        print(f"state8: {state}")
        print(f"img shape: {img.shape}, dtype: {img.dtype}, mean: {img.mean():.1f}")

        v12_chunk = predict_v12_chunk(v12, img, wrist, state, device)
        v10_chunk = predict_v10_chunk(v10, img, wrist, state, device)
        print(f"v12 chunk[0]: {np.round(v12_chunk[0], 3)}")
        print(f"v10 chunk[0]: {np.round(v10_chunk[0], 3)}")
        print(f"v12 chunk[5]: {np.round(v12_chunk[5], 3)}")
        print(f"v10 chunk[5]: {np.round(v10_chunk[5], 3)}")
        print(f"v12 chunk[15]: {np.round(v12_chunk[-1], 3)}")
        print(f"v10 chunk[15]: {np.round(v10_chunk[-1], 3)}")
        print(f"v12 grip seq: {np.round(v12_chunk[:, -1], 2).tolist()}")
        print(f"v10 grip seq: {np.round(v10_chunk[:, -1], 2).tolist()}")
        print(f"v12 ||action|| per step: {[round(float(np.linalg.norm(v12_chunk[k,:6])), 3) for k in range(16)]}")
        print(f"v10 ||action|| per step: {[round(float(np.linalg.norm(v10_chunk[k,:6])), 3) for k in range(16)]}")


if __name__ == "__main__":
    main()
