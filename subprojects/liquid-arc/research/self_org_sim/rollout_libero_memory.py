"""Memory-augmented rollout: Liquid student + episodic kNN retrieval blend.

At each chunk-prediction step:
  1. Encode obs via Liquid encoder → query feature
  2. Retrieve top-k nearest from memory bank by cosine similarity
  3. Blend retrieved-mean action_chunk with Liquid's own predicted chunk
  4. Execute first exec_horizon steps, repeat

Run inside the LIBERO sim venv (CPU torch ok):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python rollout_libero_memory.py \\
    --student_ckpt /tmp/distill_flow_notext_v1/step_030000.pt \\
    --memory_bank /home/pokazge/datasets/memory_bank_v1.npz \\
    --task_suite libero_10 --rollouts_per_task 5 --max_steps 720 \\
    --top_k 3 --alpha 0.5
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot_flow import LiquidFlowPolicy

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
except Exception:
    pass


def quat2axisangle(quat):
    import math
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def build_state8(obs_raw):
    xyz = obs_raw["robot0_eef_pos"]
    rpy = quat2axisangle(obs_raw["robot0_eef_quat"])
    grip = obs_raw["robot0_gripper_qpos"]
    return np.concatenate([xyz, rpy, grip], axis=0).astype(np.float32)


def get_raw_imgs(obs_raw):
    return (
        obs_raw["agentview_image"][::-1, ::-1].copy(),
        obs_raw["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
    )


def load_flow_policy(ckpt_path: Path, device):
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
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
    model.eval()
    return model, sa


@torch.no_grad()
def query_chunk(model, img_raw, wrist_raw, state8, mem_features, mem_actions,
                device, target_size, n_steps, top_k, alpha,
                adaptive_alpha=False, sim_thresh=0.0):
    """Sample Liquid action chunk + retrieve memory chunk and blend.

    Returns final blended chunk [K, 7], plus diagnostics.
    """
    img = np.array(Image.fromarray(img_raw).resize((target_size, target_size)), dtype=np.uint8)
    wri = np.array(Image.fromarray(wrist_raw).resize((target_size, target_size)), dtype=np.uint8)
    img_t = torch.from_numpy(img).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wri).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(state8).to(device).float().unsqueeze(0)

    # 1) Liquid encoder feature for retrieval
    cond, _ = model.encoder(img_t, wri_t, st_t, task_id=None)
    query = F.normalize(cond, dim=-1).cpu().numpy()[0]   # [d]

    # 2) Top-k nearest in memory by cosine sim (mem already normalized)
    sims = mem_features @ query                          # [N]
    topk_idx = np.argpartition(-sims, top_k)[:top_k]
    topk_idx = topk_idx[np.argsort(-sims[topk_idx])]
    topk_sims = sims[topk_idx]
    # Weighted mean (similarity-softmax)
    w = np.exp(topk_sims * 8.0)                          # temperature
    w = w / w.sum()
    mem_chunk = (w[:, None, None] * mem_actions[topk_idx]).sum(axis=0)  # [K, 7]

    # 3) Liquid sample
    liquid_chunk = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=n_steps)[0].cpu().numpy()

    # 4) Blend
    if adaptive_alpha:
        # Trust memory more when retrieval is similar (sim ~1) and less when far (~0)
        avg_sim = float(topk_sims.mean())
        if avg_sim < sim_thresh:
            a = 0.0
        else:
            a = float(np.clip(avg_sim, 0.0, 1.0)) * alpha
    else:
        a = alpha
    final = a * mem_chunk + (1.0 - a) * liquid_chunk
    return final, {"top_sims": topk_sims, "alpha_used": a}


def run_rollout(env_underlying, model, init_state, args, device, target_size,
                mem_features, mem_actions):
    env_underlying.reset()
    env_underlying.set_init_state(init_state)
    obs = None
    for _ in range(5):
        obs, _, _, _ = env_underlying.step(np.zeros(7, dtype=np.float32))
    chunk = None
    chunk_idx = 0
    success = False
    n_steps = 0
    diag_alpha = []
    diag_sim = []
    for step in range(args.max_steps):
        n_steps = step + 1
        if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
            img_raw, wrist_raw = get_raw_imgs(obs)
            state8 = build_state8(obs)
            chunk, info = query_chunk(
                model, img_raw, wrist_raw, state8, mem_features, mem_actions,
                device, target_size, n_steps=args.infer_steps, top_k=args.top_k,
                alpha=args.alpha, adaptive_alpha=args.adaptive_alpha, sim_thresh=args.sim_thresh,
            )
            chunk_idx = 0
            diag_alpha.append(info["alpha_used"])
            diag_sim.append(float(info["top_sims"].mean()))
        action7 = chunk[chunk_idx].copy()
        g = action7[-1]
        action7[-1] = args.gripper_sign * np.sign(g) if abs(g) > 0.1 else 0.0
        obs, _, done, _ = env_underlying.step(action7.astype(np.float32))
        chunk_idx += 1
        if env_underlying.check_success():
            success = True; break
        if done: break
    return success, n_steps, float(np.mean(diag_alpha) if diag_alpha else 0), float(np.mean(diag_sim) if diag_sim else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True, type=str)
    p.add_argument("--memory_bank", required=True, type=str)
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--rollouts_per_task", type=int, default=5)
    p.add_argument("--task_indices", type=str, default="")
    p.add_argument("--max_steps", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--infer_steps", type=int, default=10)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Memory mixing weight (0=Liquid only, 1=memory only)")
    p.add_argument("--adaptive_alpha", action="store_true",
                   help="Scale alpha by retrieval similarity (less trust on OOD)")
    p.add_argument("--sim_thresh", type=float, default=0.0,
                   help="Below this similarity, use Liquid only (alpha=0)")
    p.add_argument("--out_json", default="", type=str)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sargs = load_flow_policy(Path(args.student_ckpt), device)
    target_size = sargs["img_size"]

    print(f"Loading memory bank from {args.memory_bank}...")
    bank = np.load(args.memory_bank, allow_pickle=True)
    mem_features = bank["features"]   # [N, d] already normalized
    mem_actions = bank["actions"]     # [N, K, 7]
    print(f"  bank: {len(mem_features)} entries, d={mem_features.shape[1]}")
    print(f"  alpha={args.alpha} top_k={args.top_k} adaptive={args.adaptive_alpha} thresh={args.sim_thresh}")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    if args.task_indices:
        task_ids = [int(t) for t in args.task_indices.split(",")]
    else:
        task_ids = list(range(n_tasks))

    summary = {"task_suite": args.task_suite, "tasks": []}
    overall_succ, overall_total = 0, 0
    for sim_id in task_ids:
        task = suite.get_task(sim_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(sim_id)
        n_rollouts = min(args.rollouts_per_task, len(init_states))
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        rollouts = []
        print(f"\n=== sim{sim_id}: {task.language[:80]} ===")
        for r in range(n_rollouts):
            t0 = time.time()
            succ, n_steps, mean_alpha, mean_sim = run_rollout(
                env, model, init_states[r], args, device, target_size,
                mem_features, mem_actions,
            )
            wall = time.time() - t0
            overall_succ += int(succ); overall_total += 1
            rollouts.append({"r": r, "success": bool(succ), "n_steps": n_steps,
                             "mean_alpha": mean_alpha, "mean_top_sim": mean_sim, "wall_s": wall})
            print(f"  r{r}: {'SUCCESS' if succ else 'fail'}  n_steps={n_steps:3d}  "
                  f"mean_sim={mean_sim:.3f}  alpha_used={mean_alpha:.2f}  wall={wall:.1f}s")
        env.close()
        rate = sum(r["success"] for r in rollouts) / max(n_rollouts, 1)
        print(f"  sim{sim_id} success: {rate:.0%}")
        summary["tasks"].append({"sim_id": sim_id, "task": task.language,
                                  "rollouts": rollouts, "success_rate": rate})

    print("\n" + "=" * 80)
    print(f"OVERALL: {overall_succ}/{overall_total} = {overall_succ/max(overall_total,1):.0%}")
    print("=" * 80)
    summary["overall_successes"] = overall_succ
    summary["overall_total"] = overall_total
    summary["overall_success_rate"] = overall_succ / max(overall_total, 1)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
