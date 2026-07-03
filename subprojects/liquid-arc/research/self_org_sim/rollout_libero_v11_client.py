"""v11 thin client rollout: LIBERO sim + GR00T server + Liquid server.

All Liquid inference + DINOv2 retrieval + adaptive SGD happen on the GPU
via liquid_server.py over ZMQ. This script runs inside the CPU LIBERO sim
venv and just orchestrates: query GR00T (every groot_freq chunks), call
liquid_server for predict+retrieve, optionally adapt.

Run on Spark in libero_uv venv:
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python rollout_libero_v11_client.py \\
    --liquid_addr tcp://localhost:7777 \\
    --groot_port 5555 \\
    --task_suite libero_10 \\
    --rollouts_per_task 3 \\
    --max_steps 720 \\
    --exec_horizon 8 \\
    --infer_steps 10 \\
    --groot_freq 2 \\
    --adaptive --demo_replay_n 4 \\
    --depth_indices 0,1,2,3

Assumes:
  - liquid_server.py running at --liquid_addr with all options set there.
  - groot_server.py running at --groot_port for the suite under test.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import zmq

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

print = functools.partial(print, flush=True)


# -------------------- helpers --------------------------------------------- #


def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def build_state8(obs_raw):
    xyz = obs_raw["robot0_eef_pos"]
    rpy = quat2axisangle(obs_raw["robot0_eef_quat"])
    grip = obs_raw["robot0_gripper_qpos"]
    return np.concatenate([xyz, rpy, grip], axis=0).astype(np.float32)


def get_raw_imgs(obs_raw):
    # NOTE: NO image flip here — matches v9b+ scaffolding fix
    return (
        obs_raw["agentview_image"].copy(),
        obs_raw["robot0_eye_in_hand_image"].copy(),
    )


def build_groot_obs(img_256, wrist_256, state8, task_lang: str):
    """Match groot_server's expected obs structure exactly (per existing rollout)."""
    state_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    video_keys = ["image", "wrist_image"]
    language_keys = ["annotation.human.action.task_description"]
    state_slots = {
        "x": (0, 1), "y": (1, 2), "z": (2, 3),
        "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6),
        "gripper": (6, 8),
    }
    obs = {"video": {}, "state": {}, "language": {}}
    for k in video_keys:
        arr = img_256 if k == "image" else wrist_256
        obs["video"][k] = arr[None, None, ...]
    for k in state_keys:
        lo, hi = state_slots[k]
        obs["state"][k] = state8[lo:hi].astype(np.float32)[None, None, :]
    for lk in language_keys:
        obs["language"][lk] = [[task_lang]]
    return obs


# -------------------- RPC ------------------------------------------------- #


class RpcClient:
    """Thin pickle-over-ZMQ REQ/REP client with timeout + reconnect."""

    def __init__(self, addr, name="rpc", timeout_ms=120000):
        self.addr = addr
        self.name = name
        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context.instance()
        self.sock = None
        self._connect()

    def _connect(self):
        if self.sock is not None:
            try: self.sock.close(linger=0)
            except Exception: pass
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def call(self, cmd, **kwargs):
        req = {"cmd": cmd, **kwargs}
        try:
            self.sock.send(pickle.dumps(req))
            resp = pickle.loads(self.sock.recv())
        except zmq.ZMQError as e:
            print(f"[{self.name}] ZMQ error: {e}; reconnecting...")
            self._connect()
            self.sock.send(pickle.dumps(req))
            resp = pickle.loads(self.sock.recv())
        if not resp.get("ok", False):
            raise RuntimeError(f"[{self.name}] cmd={cmd} failed: {resp.get('error', resp)}")
        return resp


# -------------------- GR00T queries --------------------------------------- #


class GrootClient:
    """Direct ZMQ client matching the existing groot_server protocol.

    groot_server uses: pickle wire format, request is the obs dict (with
    optional `op` key — `op=get_action_with_state` returns chunk + z_vl
    + z_state + z_motor + traj_model_output).
    """

    def __init__(self, port, name="groot", timeout_ms=60000):
        self.port = port
        self.name = name
        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context.instance()
        self.sock = None
        self._connect()

    def _connect(self):
        if self.sock is not None:
            try: self.sock.close(linger=0)
            except Exception: pass
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://localhost:{self.port}")

    def query_full(self, obs_dict, depth_indices):
        """Returns {z_vl, z_vl_bank [len(depth_idx), hidden], delta_bank, action_chunk}."""
        try:
            self.sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
            resp = pickle.loads(self.sock.recv())
        except zmq.ZMQError as e:
            print(f"[{self.name}] ZMQ error: {e}; reconnecting...")
            self._connect()
            self.sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
            resp = pickle.loads(self.sock.recv())
        if "error" in resp:
            raise RuntimeError(f"[{self.name}] server error: {resp['error']}")
        full_traj = resp["traj_model_output"]  # [N_steps, hidden_dim]
        z_vl_bank = np.stack([full_traj[d] for d in depth_indices]).astype(np.float32)
        delta_bank = np.eye(len(depth_indices), dtype=np.float32)
        return {
            "z_vl": resp["z_vl"].astype(np.float32),
            "z_state": resp["z_state"].astype(np.float32),
            "z_vl_bank": z_vl_bank,
            "delta_bank": delta_bank,
            "action_chunk": resp["chunk"].astype(np.float32),
        }


# -------------------- main rollout loop ----------------------------------- #


def run_rollout(env, init_state, task_lang, args, liquid_rpc, groot_rpc,
                goal_img_resized=None, drift_log=None,
                sim_id=-1, rollout_idx=-1):
    env.reset()
    env.set_init_state(init_state)
    obs = None
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

    # Episode reset: restore adaptive params if adapter is active
    liquid_rpc.call("episode_reset")

    chunk = None
    chunk_idx = 0
    cached_z = None
    cached_bank = None
    cached_z_state = None
    cached_delta = None
    chunks_since_groot = 0
    n_groot_calls = 0
    n_chunk_calls = 0
    n_adapt_calls = 0
    adapt_loss_sum = 0.0
    retrieval_sims = []
    retrieval_alphas = []
    t_groot_total = 0.0
    t_liquid_total = 0.0
    t_adapt_total = 0.0
    success = False
    n_steps = 0
    depth_idxs = args.depth_indices_list

    for step in range(args.max_steps):
        n_steps = step + 1
        if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
            img_raw, wrist_raw = get_raw_imgs(obs)
            state8 = build_state8(obs)

            # DAgger drift logging: capture (img, wrist, state, lang) at every chunk
            # decision so we can later query GR00T live and use its prediction as the
            # DAgger label at Liquid-drifted states.
            if drift_log is not None:
                drift_log.append({
                    "img": img_raw.astype(np.uint8),
                    "wrist": wrist_raw.astype(np.uint8),
                    "state8": state8.astype(np.float32),
                    "task_lang": task_lang,
                    "suite": args.task_suite,
                    "step": step,
                    "sim_id": int(sim_id),
                    "rollout_idx": int(rollout_idx),
                })

            # GR00T cadence: refresh every groot_freq chunks (0=episode-start only).
            # For --use_groot_chunk, ALWAYS query GR00T every chunk decision so
            # the chunk is fresh for the current obs (avoids reusing stale chunk[0:8]
            # for env steps 8-15 — robot has moved since then).
            need_groot = (cached_z is None) or args.use_groot_chunk
            if not need_groot and args.groot_freq > 0 and chunks_since_groot >= args.groot_freq:
                need_groot = True
            if need_groot:
                # GR00T was trained on FLIPPED images (per official libero_env.py wrapper:
                # obs["agentview_image"][::-1, ::-1]). Our rollout's get_raw_imgs returns
                # un-flipped images (matches v10's training). Must flip before sending to
                # GR00T — otherwise GR00T sees the world upside-down and outputs garbage.
                groot_img = img_raw[::-1, ::-1].copy()
                groot_wrist = wrist_raw[::-1, ::-1].copy()
                obs_dict = build_groot_obs(groot_img, groot_wrist, state8, task_lang)
                t0 = time.perf_counter()
                gr = groot_rpc.query_full(obs_dict, depth_idxs)
                t_groot_total += time.perf_counter() - t0
                cached_z = gr["z_vl"]
                cached_bank = gr["z_vl_bank"]
                cached_z_state = gr["z_state"]
                cached_delta = gr["delta_bank"]
                groot_chunk_target = gr["action_chunk"]
                n_groot_calls += 1
                chunks_since_groot = 0

                # V8 adaptive: SGD on GR00T's chunk as flow-matching target
                if args.adaptive and groot_chunk_target is not None:
                    t0 = time.perf_counter()
                    resp = liquid_rpc.call(
                        "adapt_v8",
                        img_raw=img_raw, wrist_raw=wrist_raw, state8=state8,
                        target_chunk=groot_chunk_target,
                        z_groot=cached_z, z_bank=cached_bank, z_state=cached_z_state,
                        delta_bank=cached_delta,
                        goal_img_resized=goal_img_resized,
                    )
                    t_adapt_total += time.perf_counter() - t0
                    adapt_loss_sum += resp["loss"]
                    n_adapt_calls += 1
            else:
                chunks_since_groot += 1

            # Demo replay adapt: server-side
            if args.adaptive and args.demo_replay_n > 0:
                t0 = time.perf_counter()
                resp = liquid_rpc.call(
                    "adapt_demo", suite=args.task_suite, task_language=task_lang,
                    goal_img_resized=goal_img_resized,
                )
                t_adapt_total += time.perf_counter() - t0
                if resp.get("ok"):
                    adapt_loss_sum += resp["loss"]
                    n_adapt_calls += 1

            # Predict chunk + retrieve+blend on server
            t0 = time.perf_counter()
            resp = liquid_rpc.call(
                "predict_and_retrieve",
                img_raw=img_raw, wrist_raw=wrist_raw, state8=state8,
                z_groot=cached_z, z_bank=cached_bank, z_state=cached_z_state,
                delta_bank=cached_delta,
                groot_chunk=groot_chunk_target,
                chunks_since_groot=chunks_since_groot,
                goal_img_resized=goal_img_resized,
                n_steps=args.infer_steps,
            )
            t_liquid_total += time.perf_counter() - t0
            chunk = resp["chunk"]
            # GR00T-alone baseline: execute GR00T's chunk directly (skip Liquid).
            # groot_server applies the same `1.0 - 2.0 * grip` conversion that
            # gen_groot_labels.py uses for teacher_chunks.dat — so the gripper
            # is already in env convention (-1=open at approach, +1=close on grasp).
            # No additional flip needed. Use the chunk as-is.
            if args.use_groot_chunk and groot_chunk_target is not None:
                chunk = np.asarray(groot_chunk_target, dtype=np.float32)
            chunk_idx = 0
            n_chunk_calls += 1
            if resp.get("blended"):
                retrieval_sims.append(resp["retrieval_mean_sim"])
                retrieval_alphas.append(resp["retrieval_alpha"])
            # Attach predicted chunk + groot chunk to the most-recent drift log entry
            if drift_log is not None and len(drift_log) > 0:
                drift_log[-1]["pred_chunk"] = np.asarray(chunk, dtype=np.float32)
                if need_groot and groot_chunk_target is not None:
                    drift_log[-1]["groot_chunk"] = np.asarray(groot_chunk_target, dtype=np.float32)

        # Execute one step from current chunk
        action7 = chunk[chunk_idx].copy()
        if args.gripper_clamp_steps > 0 and step < args.gripper_clamp_steps:
            if args.gripper_clamp_conditional:
                # GR00T-guided correction: override to -1 only when model wants
                # to close (grip>0) AND teacher's latest chunk says still approach
                # (early positions grip<0). Preserves correct early-grasp cases.
                model_says_close = action7[-1] > 0.1
                groot_says_open = (
                    groot_chunk_target is not None
                    and groot_chunk_target.shape[0] >= 1
                    and groot_chunk_target[0, -1] < -0.1
                )
                if model_says_close and groot_says_open:
                    action7[-1] = -1.0
                else:
                    g = action7[-1]
                    action7[-1] = args.gripper_sign * np.sign(g) if abs(g) > 0.1 else 0.0
            else:
                # Unconditional clamp (Stage-2 baseline)
                action7[-1] = -1.0
        else:
            g = action7[-1]
            action7[-1] = args.gripper_sign * np.sign(g) if abs(g) > 0.1 else 0.0
        obs, _, done, _ = env.step(action7.astype(np.float32))
        chunk_idx += 1
        if env.check_success():
            success = True
            break
        if done:
            break

    rec = {
        "success": success, "n_steps": n_steps,
        "n_groot_calls": n_groot_calls, "n_chunk_calls": n_chunk_calls,
        "groot_ms_per_call": (t_groot_total * 1000 / max(n_groot_calls, 1)),
        "liquid_ms_per_call": (t_liquid_total * 1000 / max(n_chunk_calls, 1)),
        "system2_load": n_groot_calls / max(n_chunk_calls, 1),
        "n_adapt_calls": n_adapt_calls,
        "mean_adapt_loss": adapt_loss_sum / max(n_adapt_calls, 1),
        "adapt_ms_per_call": (t_adapt_total * 1000 / max(n_adapt_calls, 1)),
    }
    if len(retrieval_sims) > 0:
        rec["retrieval_mean_sim"] = float(np.mean(retrieval_sims))
        rec["retrieval_mean_alpha"] = float(np.mean(retrieval_alphas))
    return rec


# -------------------- main ----------------------------------------------- #


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--liquid_addr", default="tcp://localhost:7777", type=str)
    p.add_argument("--groot_port", type=int, default=5555)
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--rollouts_per_task", type=int, default=3)
    p.add_argument("--task_indices", default="", type=str)
    p.add_argument("--max_steps", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--infer_steps", type=int, default=10)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--groot_freq", type=int, default=2)
    p.add_argument("--depth_indices", default="0,1,2,3", type=str)
    p.add_argument("--adaptive", action="store_true",
                   help="Enable V8 + demo replay adaptive SGD on the server")
    p.add_argument("--demo_replay_n", type=int, default=0,
                   help="Number of demo frames per chunk for demo replay (server-side)")
    p.add_argument("--use_goal_img", action="store_true",
                   help="Load per-suite canonical goal image and pass to server")
    p.add_argument("--out_json", default="", type=str)
    p.add_argument("--save_drift_obs", default="", type=str,
                   help="If set, save (img, wrist, state, lang) at every chunk decision "
                        "to this .npz for DAgger relabel.")
    p.add_argument("--gripper_clamp_steps", type=int, default=0,
                   help="Stage-2 diagnostic: force gripper=-1 (open) for the first N "
                        "env steps regardless of model prediction. Tests whether fixing "
                        "the diagnosed gripper-state lock-up on libero_object unlocks success.")
    p.add_argument("--gripper_clamp_conditional", action="store_true",
                   help="If set with --gripper_clamp_steps, only override gripper when "
                        "model predicts close (grip>0) AND GR00T's latest chunk says "
                        "still approach (grip<0). Preserves correct early-grasp predictions.")
    p.add_argument("--use_groot_chunk", action="store_true",
                   help="GR00T-alone baseline: execute GR00T's predicted chunk directly, "
                        "skip Liquid student. Forces groot_freq=1 (GR00T queried every chunk).")
    p.add_argument("--use_expert_chunk", action="store_true",
                   help="Sanity check: execute expert teacher_chunks.dat directly from current "
                        "episode step. If this works, format is correct; problem is GR00T-specific.")
    args = p.parse_args()
    args.depth_indices_list = [int(x) for x in args.depth_indices.split(",") if x.strip()]

    # Connect to servers (liquid uses my new RpcClient, groot uses its own protocol)
    liquid_rpc = RpcClient(args.liquid_addr, name="liquid", timeout_ms=120000)
    groot_rpc = GrootClient(args.groot_port, name="groot", timeout_ms=60000)

    # Server init (snapshot adaptive params)
    info = liquid_rpc.call("init")
    print(f"[client] liquid_server info: {info['info']}")

    # Tell server which suite to filter retrieval to
    liquid_rpc.call("set_retrieval_filter", suite=args.task_suite)

    # Optionally load canonical goal image (matches existing v10-DEMO behavior)
    target_size = int(info["info"]["img_size"])
    goal_img_resized = None
    if args.use_goal_img:
        from PIL import Image
        goal_dir = Path("/home/pokazge/datasets/task_goals") / args.task_suite
        # We pick the per-task goal inside the per-task loop below
        print(f"[client] use_goal_img=True; will load per-task from {goal_dir}")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    task_ids = [int(t) for t in args.task_indices.split(",")] if args.task_indices else list(range(n_tasks))

    summary = {"task_suite": args.task_suite, "tasks": []}
    overall_succ = 0
    overall_total = 0
    drift_log = [] if args.save_drift_obs else None

    for sim_id in task_ids:
        task = suite.get_task(sim_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(sim_id)
        n_rollouts = min(args.rollouts_per_task, len(init_states))
        task_lang = task.language
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)

        per_task_goal = None
        if args.use_goal_img:
            from PIL import Image
            goal_path = Path("/home/pokazge/datasets/task_goals") / args.task_suite / f"task_{sim_id}.png"
            if goal_path.exists():
                gi = np.array(Image.open(goal_path).resize((target_size, target_size)), dtype=np.uint8)
                per_task_goal = gi

        rollouts = []
        s2_loads = []
        print(f"\n=== sim{sim_id}: {task_lang[:80]} ===")
        for r in range(n_rollouts):
            t0 = time.time()
            drift_log_len_before = len(drift_log) if drift_log is not None else 0
            rec = run_rollout(env, init_states[r], task_lang, args,
                              liquid_rpc, groot_rpc,
                              goal_img_resized=per_task_goal,
                              drift_log=drift_log,
                              sim_id=sim_id, rollout_idx=r)
            if drift_log is not None:
                # Tag every drift entry from this rollout with its outcome
                for k in range(drift_log_len_before, len(drift_log)):
                    drift_log[k]["rollout_success"] = bool(rec["success"])
                    drift_log[k]["rollout_n_steps"] = int(rec["n_steps"])
            rec["wall_s"] = time.time() - t0
            overall_succ += int(rec["success"])
            overall_total += 1
            s2_loads.append(rec["system2_load"])
            rollouts.append(rec)
            ret_str = (f"  ret_sim={rec.get('retrieval_mean_sim', 0):.3f} "
                       f"alpha={rec.get('retrieval_mean_alpha', 0):.2f}"
                       if 'retrieval_mean_sim' in rec else "")
            adapt_str = (f"  adapt_loss={rec['mean_adapt_loss']:.4f} ({rec['n_adapt_calls']} calls)"
                          if args.adaptive else "")
            print(f"  r{r}: {'SUCCESS' if rec['success'] else 'fail'}  "
                  f"steps={rec['n_steps']:3d}  "
                  f"S2={rec['n_groot_calls']}/S1={rec['n_chunk_calls']}  "
                  f"S2_load={rec['system2_load']:.2f}  "
                  f"S2_ms={rec['groot_ms_per_call']:.0f}  S1_ms={rec['liquid_ms_per_call']:.1f}  "
                  f"wall={rec['wall_s']:.1f}s{adapt_str}{ret_str}")
        env.close()
        rate = sum(int(r["success"]) for r in rollouts) / max(n_rollouts, 1)
        print(f"  sim{sim_id} success: {rate:.0%}  mean_S2_load={float(np.mean(s2_loads)):.2f}")
        summary["tasks"].append({
            "sim_id": sim_id, "task": task_lang,
            "rollouts": rollouts, "success_rate": rate,
        })

    print("\n" + "=" * 80)
    print(f"OVERALL: {overall_succ}/{overall_total} = {overall_succ/max(overall_total,1):.0%}")
    print("=" * 80)
    summary["overall_successes"] = overall_succ
    summary["overall_total"] = overall_total
    summary["overall_success_rate"] = overall_succ / max(overall_total, 1)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))

    if args.save_drift_obs and drift_log is not None and len(drift_log) > 0:
        out_path = Path(args.save_drift_obs)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imgs = np.stack([d["img"] for d in drift_log])
        wrists = np.stack([d["wrist"] for d in drift_log])
        states = np.stack([d["state8"] for d in drift_log])
        langs = np.array([d["task_lang"] for d in drift_log])
        suites = np.array([d["suite"] for d in drift_log])
        steps = np.array([d["step"] for d in drift_log], dtype=np.int32)
        sim_ids = np.array([d.get("sim_id", -1) for d in drift_log], dtype=np.int32)
        rollout_idxs = np.array([d.get("rollout_idx", -1) for d in drift_log], dtype=np.int32)
        rollout_success = np.array([int(d.get("rollout_success", False)) for d in drift_log], dtype=np.int8)
        rollout_n_steps = np.array([int(d.get("rollout_n_steps", 0)) for d in drift_log], dtype=np.int32)
        # Predicted/GR00T chunks may be missing for early entries — pad with NaN
        K = 16; A = 7
        pred_chunks = np.full((len(drift_log), K, A), np.nan, dtype=np.float32)
        groot_chunks = np.full((len(drift_log), K, A), np.nan, dtype=np.float32)
        for i, d in enumerate(drift_log):
            if "pred_chunk" in d:
                pc = d["pred_chunk"]
                if pc.shape == (K, A):
                    pred_chunks[i] = pc
            if "groot_chunk" in d:
                gc = d["groot_chunk"]
                if gc.shape == (K, A):
                    groot_chunks[i] = gc
        np.savez(out_path, imgs=imgs, wrists=wrists, states=states,
                 langs=langs, suites=suites, steps=steps,
                 sim_ids=sim_ids, rollout_idxs=rollout_idxs,
                 rollout_success=rollout_success, rollout_n_steps=rollout_n_steps,
                 pred_chunks=pred_chunks, groot_chunks=groot_chunks)
        print(f"[client] saved {len(drift_log)} drift obs to {out_path} "
              f"({imgs.nbytes / 1e6:.0f}MB), incl. pred_chunks + outcome tags")


if __name__ == "__main__":
    main()
