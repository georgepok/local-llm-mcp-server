"""Chained-LIBERO eval — substrate as CLOSED-LOOP CONTROLLER in gap between GR00T chunks.

Per chunk emission cycle:
  1. Call GR00T → get chunk[16,7], z_t, state8_at_emission
  2. Substrate.step(...) → encode intent h_intent (substrate's per-chunk state)
  3. Initialize predicted_state = state8_at_emission

Per gap step k (0..exec_horizon-1):
  4. Get actual proprio state8_actual
  5. If k > 0:
        deviation = actual - predicted (from previous step's forward_dynamics)
        if ||deviation|| > threshold:
            correction = substrate.compute_correction(deviation, h_intent)
            chunk[k] += correction  (bounded)
  6. action = chunk[k]
  7. Predicted next state = substrate.forward_dynamics(actual, action, h_intent)
  8. Execute action in env

Compare to baseline (no correction, --no_correction flag).

Substrate uses UNIQUE info GR00T can't see: intermediate gap-step proprio states.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pickle
import torch
import zmq

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF_DIR))

from liquid_goal_tracker_proprio import JEPA_LGT_Proprio  # type: ignore
from rollout_libero_v11_client import (  # type: ignore
    build_state8, get_raw_imgs, build_groot_obs,
)

DATASET_ROOT = Path("/home/pokazge/datasets")


def extract_z_goal_via_groot(sock, suite_name, task_id):
    suite_short = suite_name.replace("libero_", "")
    sd = DATASET_ROOT / f"libero-{suite_short}-expert-v1"
    if not sd.exists():
        return None
    idx = np.load(sd / "index.npz")
    starts = idx["episode_starts"]; lengths = idx["episode_lengths"]
    task_indices = idx["task_indices"]
    success = idx.get("success_per_episode", np.ones(len(lengths), dtype=bool))
    n_total = int(idx["n_total"]); img_size = int(idx["img_size"])
    imgs = np.memmap(sd / "imgs.dat", dtype=np.uint8, mode="r",
                     shape=(n_total, img_size, img_size, 3))
    wrists = np.memmap(sd / "wrists.dat", dtype=np.uint8, mode="r",
                       shape=(n_total, img_size, img_size, 3))
    states = np.memmap(sd / "states.dat", dtype=np.float32, mode="r",
                       shape=(n_total, 8))
    task_lang = {}
    lp = sd / "task_languages.json"
    if lp.exists():
        task_lang = json.loads(lp.read_text())
    for ep in range(len(lengths)):
        if int(task_indices[ep]) != task_id or not bool(success[ep]):
            continue
        ep_start = int(starts[ep]); ep_len = int(lengths[ep])
        last_gi = ep_start + ep_len - 1
        img = np.array(imgs[last_gi])[::-1, ::-1].copy()
        wri = np.array(wrists[last_gi])[::-1, ::-1].copy()
        st = np.array(states[last_gi])
        lang = (task_lang.get(str(task_id)) or task_lang.get(task_id)
                or "do the task")
        obs = build_groot_obs(img, wri, st, lang)
        sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs}))
        resp = pickle.loads(sock.recv())
        return resp["z_vl"].astype(np.float32)
    return None


class _StuckHead(torch.nn.Module):
    """Mirror of train_stuck_head.py StuckHead — small MLP head."""
    def __init__(self, K, d, hidden=64):
        super().__init__()
        in_dim = 2 * K * d
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.SiLU(),
            torch.nn.LayerNorm(hidden),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, h_goal, h_goal_delta):
        x = torch.cat([h_goal.flatten(1), h_goal_delta.flatten(1)], dim=-1)
        return self.net(x).squeeze(-1)


class DynamicsClient:
    def __init__(self, port, substrate_ckpt, suite_name, device=None,
                 deviation_threshold=0.05, correction_enabled=True,
                 correction_scale=1.0, correction_dev_ceiling=0.0,
                 early_bail_enabled=False, bail_window=10,
                 bail_dist_thresh=0.05, bail_min_steps=200,
                 collect_trajectories=False,
                 stuck_head_ckpt=None, stuck_prob_thresh=0.5,
                 projection_residual=False, projection_scale=1.0):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.suite_name = suite_name
        self.correction_scale = correction_scale
        self.correction_dev_ceiling = correction_dev_ceiling
        self.early_bail_enabled = early_bail_enabled
        self.bail_window = bail_window
        self.bail_dist_thresh = bail_dist_thresh
        self.bail_min_steps = bail_min_steps
        self.goaldist_history = []  # per-chunk pred_goaldist for current sub-task
        self.n_bails = 0
        # Trajectory collection (for training a stuck-head classifier later)
        self.collect_trajectories = collect_trajectories
        self.h_goal_history = []   # per-chunk h_goal tensor [K, d] for current sub-task
        self.trajectory_records = []  # list of dicts, appended on sub-task end
        # JEPA-extended collection: substrate INPUTS per chunk
        self.z_vl_history = []        # per-chunk [2048] pooled VL
        self.z_lang_history = []      # per-chunk [2048] pooled language
        self.state8_history = []      # per-chunk [8] proprio at chunk start
        self.chunk_history = []       # per-chunk [horizon, action_dim] emitted action plan
        # Projection-residual mode (V9-like: substrate's projection -> z_vl override).
        # When enabled, emit_chunk uses get_action_with_zvl_override op with substrate's
        # projection_residual computed from PREVIOUS h_goal.
        self.projection_residual = projection_residual
        self.projection_scale = projection_scale
        # Optional learned stuck-head — when set, drives bail decision instead
        # of pred_goaldist spread.
        self.stuck_head = None
        self.stuck_prob_thresh = stuck_prob_thresh
        self.stuck_head_window = bail_window
        if stuck_head_ckpt is not None:
            sh_ck = torch.load(stuck_head_ckpt, map_location=device, weights_only=False)
            cfg = sh_ck["stuck_head_config"]
            self.stuck_head = _StuckHead(cfg["K"], cfg["d"], cfg["hidden"]).to(device)
            self.stuck_head.load_state_dict(sh_ck["stuck_head_state_dict"])
            self.stuck_head.eval()
            self.stuck_head_window = cfg["window"]
            for pp in self.stuck_head.parameters():
                pp.requires_grad = False
            print(f"[dyn-eval] loaded stuck_head from {stuck_head_ckpt}: "
                  f"K={cfg['K']} d={cfg['d']} window={cfg['window']} "
                  f"val_AUC={sh_ck.get('val_AUC', '?'):.3f}", flush=True)
        self.deviation_threshold = deviation_threshold
        self.correction_enabled = correction_enabled
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, 60000)
        self.sock.connect(f"tcp://localhost:{port}")
        ck = torch.load(substrate_ckpt, map_location=device, weights_only=False)
        sa = ck["args"]
        d_sub = sa.get("d_substrate", 64)
        K_bel = sa.get("K_belief", 4)
        n_tok = sa.get("n_tok_per_k", 1)
        self.substrate = JEPA_LGT_Proprio(
            z_vl_dim=ck["z_vl_dim"], action_dim=ck["action_dim"],
            horizon=ck["horizon"], state_dim=ck["state_dim"],
            d=d_sub, K=K_bel, n_tok_per_k=n_tok,
        ).to(device)
        self.substrate.load_state_dict(ck["substrate_state_dict"], strict=False)
        self.substrate.eval()
        self.action_dim = ck["action_dim"]
        self.horizon = ck["horizon"]
        self.z_dim = ck["z_vl_dim"]
        self.K = K_bel
        self.h_goal = None
        self.h_intent = None  # set at each chunk emission
        self.z_goal_cache = {}
        self.current_z_goal = None
        # Diagnostics
        self.n_chunks = 0
        self.n_gap_steps = 0
        self.n_corrections = 0
        self.total_deviation = 0.0
        self.total_correction_norm = 0.0
        print(f"[dyn-eval] substrate {substrate_ckpt}, K={self.K}, "
              f"deviation_thr={deviation_threshold}, "
              f"correction={'ON' if correction_enabled else 'OFF'}", flush=True)

    def set_subtask(self, task_id):
        self.h_goal = None
        self.h_intent = None
        self.goaldist_history = []  # reset stuck-detector window per sub-task
        self.h_goal_history = []    # reset trajectory window per sub-task
        self.z_vl_history = []
        self.z_lang_history = []
        self.state8_history = []
        self.chunk_history = []
        self._current_task_id = task_id
        if task_id not in self.z_goal_cache:
            z_g = extract_z_goal_via_groot(self.sock, self.suite_name, task_id)
            if z_g is None:
                z_g = np.zeros(self.z_dim, dtype=np.float32)
            self.z_goal_cache[task_id] = z_g
        self.current_z_goal = self.z_goal_cache[task_id]

    def record_subtask_end(self, sub_steps, succ, bailed):
        """Called after each sub-task to log h_goal trajectory + (optionally) inputs
        for offline training.
        """
        if not self.collect_trajectories or len(self.h_goal_history) == 0:
            return
        traj = torch.stack(self.h_goal_history, dim=0).cpu().to(torch.float16)
        record = {
            "h_goal_traj": traj,
            "goaldist_traj": list(self.goaldist_history),
            "sub_steps": int(sub_steps),
            "succ": bool(succ),
            "bailed": bool(bailed),
            "sub_id": int(getattr(self, "_current_task_id", -1)),
            "suite": self.suite_name,
        }
        # JEPA-extended: save substrate inputs per chunk (only if collected)
        if len(self.z_vl_history) == len(self.h_goal_history):
            record["z_vl_traj"] = torch.stack(
                [torch.from_numpy(np.asarray(x, dtype=np.float32))
                 for x in self.z_vl_history], dim=0).to(torch.float16)
            record["z_lang_traj"] = torch.stack(
                [torch.from_numpy(np.asarray(x, dtype=np.float32))
                 for x in self.z_lang_history], dim=0).to(torch.float16)
            record["state8_traj"] = torch.stack(
                [torch.from_numpy(np.asarray(x, dtype=np.float32))
                 for x in self.state8_history], dim=0).to(torch.float32)
            record["chunk_traj"] = torch.stack(
                [torch.from_numpy(np.asarray(x, dtype=np.float32))
                 for x in self.chunk_history], dim=0).to(torch.float16)
            # z_goal is constant per sub-task; save once
            if self.current_z_goal is not None:
                record["z_goal"] = torch.from_numpy(
                    np.asarray(self.current_z_goal, dtype=np.float32)).to(torch.float16)
        self.trajectory_records.append(record)

    def emit_chunk(self, obs_dict, state8_now):
        """Called at chunk-emission time. Queries GR00T, encodes h_intent.
        Returns the chunk; gap-step adaptation happens in gap_correct().

        If projection_residual is enabled, computes substrate's future-state
        projection from PREVIOUS h_goal, encodes to z_vl residual, and uses
        get_action_with_zvl_override so GR00T sees a goal-direction hint.
        """
        if self.projection_residual:
            # Compute residual from previous h_goal (init_state for first chunk)
            with torch.no_grad():
                h_prev = self.h_goal if self.h_goal is not None \
                    else self.substrate.init_state(1, self.device)
                state8_t = torch.from_numpy(state8_now.astype(np.float32)).to(
                    self.device).unsqueeze(0)
                state_proj = self.substrate.project_future_state(state8_t, h_prev)
                residual = self.substrate.encode_projection_residual(state_proj)
                residual = residual[0] * self.projection_scale  # [2048]
            residual_np = residual.cpu().numpy().astype(np.float32)
            self.sock.send(pickle.dumps({
                "op": "get_action_with_zvl_override",
                "obs": obs_dict,
                "zvl_residual": residual_np,
            }))
            resp = pickle.loads(self.sock.recv())
            chunk = np.asarray(resp["chunk"], dtype=np.float32)
            z_t = resp.get("z_vl_post_override", np.zeros(self.z_dim, dtype=np.float32))
            z_lang = z_t  # override op doesn't return z_lang separately
        else:
            self.sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
            resp = pickle.loads(self.sock.recv())
            chunk = np.asarray(resp["chunk"], dtype=np.float32)
            z_t = resp["z_vl"]
            z_lang = resp.get("z_lang", z_t)

        # Substrate.step: encode intent from observed state + chunk
        with torch.no_grad():
            z_t_t = torch.from_numpy(z_t).to(self.device).unsqueeze(0)
            z_lang_t = torch.from_numpy(np.asarray(z_lang, dtype=np.float32)).to(
                self.device).unsqueeze(0)
            z_g_t = torch.from_numpy(self.current_z_goal).to(self.device).unsqueeze(0)
            state8_t = torch.from_numpy(state8_now.astype(np.float32)).to(
                self.device).unsqueeze(0)
            chunk_t = torch.from_numpy(chunk).to(self.device).unsqueeze(0).float()
            if self.h_goal is None:
                self.h_goal = self.substrate.init_state(1, self.device)
            self.h_goal, pred_goaldist, _, _ = self.substrate.step(
                self.h_goal, z_t_t, z_g_t, chunk_t, state8_t, z_lang_t=z_lang_t)
            self.h_intent = self.h_goal  # use h_goal as intent for dynamics/correction
            self.goaldist_history.append(float(pred_goaldist.mean().item()))
            if self.collect_trajectories:
                # Save snapshot of h_goal [K, d] (squeeze batch dim)
                self.h_goal_history.append(self.h_goal[0].detach().cpu())
                # JEPA-extended: also save substrate INPUTS this chunk
                self.z_vl_history.append(z_t.copy() if hasattr(z_t, 'copy')
                                          else np.asarray(z_t).copy())
                self.z_lang_history.append(z_lang.copy() if hasattr(z_lang, 'copy')
                                            else np.asarray(z_lang).copy())
                self.state8_history.append(state8_now.astype(np.float32).copy())
                self.chunk_history.append(chunk.astype(np.float32).copy())

        self.n_chunks += 1
        return chunk

    def should_bail(self, sub_steps_done):
        """Return True if substrate signals sub-task is stuck.

        Two modes:
        - With a trained stuck_head: P(stuck) > stuck_prob_thresh.
        - Otherwise: pred_goaldist spread over window < bail_dist_thresh.

        Both refuse to bail before bail_min_steps (warmup floor).
        """
        if not self.early_bail_enabled:
            return False
        if sub_steps_done < self.bail_min_steps:
            return False
        # Stuck-head signal path
        if self.stuck_head is not None:
            W = self.stuck_head_window
            if len(self.h_goal_history) < W + 1:
                return False
            with torch.no_grad():
                h_t = self.h_goal_history[-1].float().to(self.device).unsqueeze(0)
                h_tw = self.h_goal_history[-1 - W].float().to(self.device).unsqueeze(0)
                logit = self.stuck_head(h_t, h_t - h_tw)
                p_stuck = float(torch.sigmoid(logit).item())
            if p_stuck > self.stuck_prob_thresh:
                self.n_bails += 1
                return True
            return False
        # Default pred_goaldist signal path
        if len(self.goaldist_history) < self.bail_window:
            return False
        window = self.goaldist_history[-self.bail_window:]
        spread = max(window) - min(window)
        if spread < self.bail_dist_thresh:
            self.n_bails += 1
            return True
        return False

    def gap_correct(self, chunk, k, state8_actual, predicted_state8):
        """Called at each gap step k (0..exec_horizon-1).
        Compares actual proprio to predicted; if deviation > threshold,
        applies bounded correction to chunk[k].
        Returns: (adjusted_action [7], next_predicted_state8 [8])
        """
        with torch.no_grad():
            actual_t = torch.from_numpy(state8_actual.astype(np.float32)).to(
                self.device).unsqueeze(0)
            action_orig = torch.from_numpy(chunk[k].astype(np.float32)).to(
                self.device).unsqueeze(0)
            # Compute deviation if k > 0
            corrected_action = chunk[k].copy()
            if k > 0 and predicted_state8 is not None and self.correction_enabled:
                predicted_t = torch.from_numpy(predicted_state8.astype(np.float32)).to(
                    self.device).unsqueeze(0)
                deviation = actual_t - predicted_t  # [1, 8]
                dev_norm = float(deviation.norm(dim=-1).mean())
                self.total_deviation += dev_norm
                # Suppress correction when deviation is too large to be trusted
                # (model is OOD; correction's prediction is unreliable, cascades).
                under_ceiling = (self.correction_dev_ceiling <= 0.0
                                  or dev_norm < self.correction_dev_ceiling)
                if dev_norm > self.deviation_threshold and under_ceiling:
                    delta_action = self.substrate.compute_correction(
                        deviation, self.h_intent) * self.correction_scale
                    corrected_action = (action_orig + delta_action)[0].cpu().numpy()
                    self.n_corrections += 1
                    self.total_correction_norm += float(delta_action.norm().item())
            self.n_gap_steps += 1
            # Predict next state under (corrected) action
            action_used = torch.from_numpy(corrected_action.astype(np.float32)).to(
                self.device).unsqueeze(0)
            next_pred = self.substrate.forward_dynamics(
                actual_t, action_used, self.h_intent)
            next_pred_np = next_pred[0].cpu().numpy().astype(np.float32)
        return corrected_action.astype(np.float32), next_pred_np


def step_subtask(env, sub_task, obs, args, client, max_steps=None):
    sub_lang = sub_task.language
    chunk = None
    chunk_idx = 0
    sub_steps = 0
    n_calls = 0
    predicted_state8 = None
    bailed = False
    cap = max_steps if max_steps is not None else args.max_steps_per_task
    while sub_steps < cap:
        if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
            img_raw, wrist_raw = get_raw_imgs(obs)
            state8 = build_state8(obs)
            groot_img = img_raw[::-1, ::-1].copy()
            groot_wrist = wrist_raw[::-1, ::-1].copy()
            groot_obs = build_groot_obs(groot_img, groot_wrist, state8, sub_lang)
            chunk = client.emit_chunk(groot_obs, state8)
            chunk_idx = 0
            n_calls += 1
            predicted_state8 = state8.copy()  # init prediction to current state
            # Substrate-driven early bail: pred_goaldist hasn't progressed → cut sub-task
            if client.should_bail(sub_steps):
                bailed = True
                return False, sub_steps, obs, n_calls, bailed
        # GAP-STEP closed-loop: actual state, deviation, correction, predict next
        state8_actual = build_state8(obs)
        action_used, predicted_next = client.gap_correct(
            chunk, chunk_idx, state8_actual, predicted_state8)
        predicted_state8 = predicted_next
        # Gripper discretization (as in other evals)
        g = action_used[-1]
        action_used[-1] = np.sign(g) if abs(g) > 0.1 else 0.0
        obs, _, done, _ = env.step(action_used)
        chunk_idx += 1
        sub_steps += 1
        if env.check_success():
            return True, sub_steps, obs, n_calls, bailed
        if done:
            return False, sub_steps, obs, n_calls, bailed
    return False, sub_steps, obs, n_calls, bailed


def transfer_state(src_env, dst_env, src_obs):
    try:
        dst_env.reset()
        src_joints = None
        if isinstance(src_obs, dict) and "robot0_joint_pos" in src_obs:
            src_joints = np.asarray(src_obs["robot0_joint_pos"], dtype=np.float64)
        elif hasattr(src_env, "robots") and src_env.robots:
            src_joints = np.asarray(src_env.robots[0]._joint_positions, dtype=np.float64)
        if src_joints is None or len(src_joints) == 0:
            return False
        if hasattr(dst_env, "robots") and dst_env.robots:
            dst_env.robots[0].set_robot_joint_positions(src_joints)
            if hasattr(dst_env, "sim"):
                dst_env.sim.forward()
            return True
    except Exception:
        pass
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--substrate_ckpt", required=True, type=str)
    p.add_argument("--groot_port", type=int, default=5555)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--n_chains", type=int, default=6)
    p.add_argument("--rollouts_per_chain", type=int, default=4)
    p.add_argument("--chain_length", type=int, default=5)
    p.add_argument("--task_suite", default="libero_10")
    p.add_argument("--max_steps_per_task", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--retract_steps", type=int, default=30)
    p.add_argument("--retract_grip_thr", type=float, default=0.030)
    p.add_argument("--shared_budget", type=int, default=1100)
    p.add_argument("--deviation_threshold", type=float, default=0.05,
                   help="Min deviation norm to trigger correction")
    p.add_argument("--no_correction", action="store_true",
                   help="Disable correction (baseline: still runs dynamics for metrics)")
    p.add_argument("--correction_scale", type=float, default=1.0,
                   help="Scale factor for correction output (1.0 = raw bounded ±0.05/dim)")
    p.add_argument("--correction_dev_ceiling", type=float, default=0.0,
                   help="Above this deviation norm, suppress correction (0=disabled). "
                        "Use to avoid correction cascade when env state is OOD wrt trained dynamics.")
    p.add_argument("--early_bail", action="store_true",
                   help="Enable substrate-driven early bail when pred_goaldist plateaus")
    p.add_argument("--bail_window", type=int, default=10,
                   help="Number of recent chunks to check for goaldist progress")
    p.add_argument("--bail_dist_thresh", type=float, default=0.05,
                   help="Max spread (max-min) of pred_goaldist across window to declare stuck")
    p.add_argument("--bail_min_steps", type=int, default=200,
                   help="Min sub-task env steps before bail is allowed (warmup floor)")
    p.add_argument("--bail_advances_chain", action="store_true",
                   help="When bail fires on a sub-task, advance to next sub-task "
                        "instead of ending the chain. Lets substrate's bail signal "
                        "reallocate budget across remaining sub-tasks.")
    p.add_argument("--collect_trajectories", type=str, default=None,
                   help="If set, save (h_goal_traj, sub_steps, succ, sub_id) per "
                        "sub-task to this .pt path for offline stuck-classifier training.")
    p.add_argument("--stuck_head_ckpt", type=str, default=None,
                   help="If set, load trained stuck-head from this .pt path and use "
                        "P(stuck)>stuck_prob_thresh as the bail signal (overrides "
                        "pred_goaldist-spread signal).")
    p.add_argument("--stuck_prob_thresh", type=float, default=0.5,
                   help="P(stuck) threshold above which to bail. Higher = more "
                        "conservative (fewer bails, higher precision).")
    p.add_argument("--projection_residual", action="store_true",
                   help="Use substrate's head_projection + head_zvl_residual to inject "
                        "a future-state-projection residual into GR00T's vl_embeds via "
                        "get_action_with_zvl_override op. Requires substrate ckpt with "
                        "trained projection + zvl_residual heads.")
    p.add_argument("--projection_scale", type=float, default=1.0,
                   help="Scale on projection residual at inference (1.0 = trained mag)")
    p.add_argument("--out_json", default="/tmp/chained_dynamics.json")
    args = p.parse_args()

    client = DynamicsClient(
        args.groot_port, args.substrate_ckpt, args.task_suite,
        deviation_threshold=args.deviation_threshold,
        correction_enabled=not args.no_correction,
        correction_scale=args.correction_scale,
        correction_dev_ceiling=args.correction_dev_ceiling,
        early_bail_enabled=args.early_bail,
        bail_window=args.bail_window,
        bail_dist_thresh=args.bail_dist_thresh,
        bail_min_steps=args.bail_min_steps,
        collect_trajectories=bool(args.collect_trajectories),
        stuck_head_ckpt=args.stuck_head_ckpt,
        stuck_prob_thresh=args.stuck_prob_thresh,
        projection_residual=args.projection_residual,
        projection_scale=args.projection_scale)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    rng = np.random.default_rng(args.seed)
    chains = [sorted(rng.choice(n_tasks, size=args.chain_length, replace=False).tolist())
               for _ in range(args.n_chains)]
    print(f"[dyn-eval] suite={args.task_suite} chains={len(chains)} "
          f"L={args.chain_length} budget={args.shared_budget}", flush=True)

    envs = {}
    def get_env(task_id):
        if task_id in envs: return envs[task_id]
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        e = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        envs[task_id] = e
        return e

    total_succ = 0; total_attempts = 0
    chain_completions = []
    summary = {"chains": []}

    for chain_idx, chain in enumerate(chains):
        print(f"\n=== chain{chain_idx} {chain} ===", flush=True)
        first_env = get_env(chain[0])
        init_states = suite.get_task_init_states(chain[0])
        n_rollouts = min(args.rollouts_per_chain, len(init_states))
        for r in range(n_rollouts):
            t0 = time.time()
            n_complete = 0
            first_env.reset()
            first_env.set_init_state(init_states[r])
            obs = None
            for _ in range(5):
                obs, _, _, _ = first_env.step(np.zeros(7, dtype=np.float32))
            current_env = first_env
            budget_remaining = args.shared_budget if args.shared_budget > 0 else None
            sub_details = []
            for sub_idx, sub_id in enumerate(chain):
                sub_task = suite.get_task(sub_id)
                client.set_subtask(sub_id)
                if sub_idx > 0:
                    next_env = get_env(sub_id)
                    next_env.reset()
                    next_init_states = suite.get_task_init_states(sub_id)
                    next_env.set_init_state(next_init_states[r % len(next_init_states)])
                    for _ in range(3):
                        obs, _, _, _ = next_env.step(np.zeros(7, dtype=np.float32))
                    transfer_state(current_env, next_env, obs)
                    current_env = next_env
                    for _ in range(5):
                        obs, _, _, _ = current_env.step(np.zeros(7, dtype=np.float32))
                if budget_remaining is not None and budget_remaining <= 0:
                    sub_details.append({"sub_id": sub_id, "succ": False, "steps": 0})
                    break
                step_cap = (min(args.max_steps_per_task, budget_remaining)
                            if budget_remaining is not None
                            else args.max_steps_per_task)
                succ, steps, obs, n_calls, bailed = step_subtask(
                    current_env, sub_task, obs, args, client, max_steps=step_cap)
                client.record_subtask_end(steps, succ, bailed)
                if budget_remaining is not None:
                    budget_remaining -= steps
                sub_details.append({"sub_id": sub_id, "succ": succ, "steps": steps,
                                     "n_calls": n_calls, "bailed": bailed})
                # Retract after success OR after bail (to release any held object
                # before advancing to next sub-task), but not on final sub-task.
                if (succ or (bailed and args.bail_advances_chain)) \
                        and args.retract_steps > 0 and sub_idx < len(chain) - 1:
                    open_lift = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, -1.0],
                                          dtype=np.float32)
                    for _ in range(args.retract_steps):
                        if budget_remaining is not None and budget_remaining <= 0: break
                        if isinstance(obs, dict):
                            gq = obs.get("robot0_gripper_qpos")
                            if gq is not None and float(np.asarray(gq).mean()) > args.retract_grip_thr:
                                break
                        obs, _, _, _ = current_env.step(open_lift)
                        if budget_remaining is not None:
                            budget_remaining -= 1
                if succ:
                    n_complete += 1
                elif bailed and args.bail_advances_chain:
                    continue  # skip this sub-task, advance to next with remaining budget
                else:
                    break
            wall = time.time() - t0
            chain_completions.append(n_complete)
            total_succ += n_complete
            total_attempts += len(chain)
            dev_avg = client.total_deviation / max(client.n_gap_steps, 1)
            corr_frac = client.n_corrections / max(client.n_gap_steps, 1)
            corr_norm = client.total_correction_norm / max(client.n_corrections, 1)
            steps_str = ",".join(str(s["steps"]) for s in sub_details if "steps" in s)
            print(f"  r{r}: {n_complete}/{len(chain)} sub-tasks  wall={wall:.0f}s  "
                  f"steps=[{steps_str}]  dev_avg={dev_avg:.4f}  "
                  f"corr_frac={corr_frac:.2f}  corr_norm={corr_norm:.4f}", flush=True)
            summary["chains"].append({
                "chain_idx": chain_idx, "rollout": r, "chain": chain,
                "n_complete": n_complete, "wall_s": wall, "details": sub_details,
                "deviation_avg": dev_avg, "correction_frac": corr_frac,
                "correction_norm_mean": corr_norm,
            })

    for e in envs.values():
        try: e.close()
        except: pass

    summary["total_succ"] = total_succ
    summary["total_attempts"] = total_attempts
    summary["completion_rate"] = total_succ / max(total_attempts, 1)
    summary["mean_chain_completion"] = (
        float(np.mean(chain_completions)) if chain_completions else 0)
    summary["correction_enabled"] = not args.no_correction
    summary["deviation_threshold"] = args.deviation_threshold
    summary["total_gap_steps"] = client.n_gap_steps
    summary["total_corrections"] = client.n_corrections
    summary["correction_fire_rate"] = client.n_corrections / max(client.n_gap_steps, 1)
    summary["early_bail_enabled"] = args.early_bail
    summary["n_bails"] = client.n_bails
    print(f"\n{'='*60}", flush=True)
    print(f"OVERALL: {total_succ}/{total_attempts} sub-tasks = "
          f"{100*summary['completion_rate']:.0f}%   "
          f"corrections fired {client.n_corrections}/{client.n_gap_steps} = "
          f"{100*summary['correction_fire_rate']:.0f}%   "
          f"early_bails={client.n_bails}", flush=True)
    print(f"{'='*60}", flush=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))
    if args.collect_trajectories and client.trajectory_records:
        n = len(client.trajectory_records)
        n_succ = sum(1 for r in client.trajectory_records if r["succ"])
        torch.save({
            "records": client.trajectory_records,
            "seed": args.seed, "task_suite": args.task_suite,
            "substrate_ckpt": args.substrate_ckpt,
        }, args.collect_trajectories)
        print(f"[traj] saved {n} sub-task trajectories ({n_succ} succ, "
              f"{n-n_succ} fail) → {args.collect_trajectories}", flush=True)


if __name__ == "__main__":
    main()
