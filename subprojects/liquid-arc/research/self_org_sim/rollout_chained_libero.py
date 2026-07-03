"""Chained-LIBERO long-horizon eval.

Strategy: chain N libero_10 tasks by RESETTING into the next task's env using
the robot state from the previous task's success. Tasks share the same scene
(LIBERO_LIVING_ROOM or LIBERO_KITCHEN) so object positions transfer naturally.

Per chain:
  1. Reset task-1 env, run GR00T with task-1 language until success or max_steps
  2. If task-1 succeeds, get robot's current joint state + object positions
  3. Set task-2 env to that state, continue with task-2 language
  4. Score = number of consecutive sub-tasks completed

This stresses GR00T's long-horizon coherence: maintaining task-relevant
behavior across sub-task transitions where language changes but scene continues.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SELF_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SELF_DIR))
from rollout_libero_v11_client import (  # type: ignore
    build_state8, get_raw_imgs, build_groot_obs, GrootClient,
)


class SubstrateOverlay:
    """v10 encoder + SubstrateGoalTracker for gripper override on GR00T's chunk.

    Mirrors liquid_server's GT override path, but with GR00T's chunk as input
    (instead of v10's self-prediction). h_goal persists across chunks within a
    sub-task; reset between sub-tasks (or set --gt_keep_across_subtasks).
    """
    def __init__(self, v10_ckpt_path, gt_ckpt_path, device, threshold=0.5,
                 apply_positions=8):
        import torch
        from distill_groot_flow import LiquidFlowPolicy  # type: ignore
        from substrate_goal_tracker import SubstrateGoalTracker  # type: ignore
        self.torch = torch
        self.device = device
        self.threshold = threshold
        self.apply_positions = apply_positions

        v10_ck = torch.load(v10_ckpt_path, map_location=device, weights_only=False)
        sa = v10_ck["args"]
        halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
        v10 = LiquidFlowPolicy(
            state_dim=8, action_dim=7, action_horizon=sa["action_horizon"],
            d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"],
            k_max=sa["k"], halt_mode=halt_mode,
            min_steps=sa["halting_min_steps"],
            n_tasks=sa["n_tasks"], d_task=sa["d_task"],
            head_d=sa["head_d"], head_layers=sa["head_layers"],
            head_heads=sa["head_heads"],
            n_task_heads=sa.get("n_task_heads", 0),
            z_groot_dim=sa.get("z_groot_dim", 0),
            gated_mixture=sa.get("gated_mixture", False),
            z_channel_dims=sa.get("z_channel_dims", None),
            query_bank=sa.get("use_query_bank", False),
        ).to(device)
        sd = v10_ck.get("policy", v10_ck.get("model", v10_ck))
        own = v10.state_dict()
        for k, v in sd.items():
            kk = k.replace("_orig_mod.", "")
            if kk in own and own[kk].shape == v.shape:
                own[kk].copy_(v)
        v10.eval()
        for pp in v10.parameters():
            pp.requires_grad = False
        self.v10 = v10
        self.img_size = sa["img_size"]
        self.d_obs = sa["d"]

        gt_ck = torch.load(gt_ckpt_path, map_location=device, weights_only=False)
        self.gt = SubstrateGoalTracker(
            d_obs=gt_ck["d_obs"], d_state=8, d_chunk=16 * 7,
            d=gt_ck["d_substrate"], K=gt_ck["K_belief"], action_horizon=16,
        ).to(device)
        self.gt.load_state_dict(gt_ck["state_dict"])
        self.gt.eval()
        self.h_goal = None
        self.prev_chunk = None
        self.total_overridden = 0
        self.n_chunks = 0
        print(f"[subs] v10 + GT loaded: d_obs={gt_ck['d_obs']} K={gt_ck['K_belief']} "
              f"d_sub={gt_ck['d_substrate']} threshold={threshold} apply={apply_positions}")

    def reset(self):
        self.h_goal = None
        self.prev_chunk = None

    def override_chunk(self, chunk_np, img_raw, wrist_raw, state8):
        """Apply per-position gripper override to GR00T's chunk."""
        import torch
        import torch.nn.functional as F
        # Resize images to v10's expected size, [0,1] floats, BCHW
        def prep(arr):
            t = torch.from_numpy(arr).to(self.device).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0)
            if t.shape[-1] != self.img_size:
                t = F.interpolate(t, size=(self.img_size, self.img_size),
                                   mode="bilinear", align_corners=False)
            return t
        img_t = prep(img_raw)
        wri_t = prep(wrist_raw)
        st_t = torch.from_numpy(np.asarray(state8, dtype=np.float32)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            cond_v10, _ = self.v10.encoder(img_t, wri_t, st_t)
            chunk_t = torch.from_numpy(np.asarray(chunk_np, dtype=np.float32)).to(self.device).unsqueeze(0)
            if self.h_goal is None:
                self.h_goal = self.gt.init_state(1, self.device)
            h_new, gripper_logits, info = self.gt.step(
                self.h_goal, cond_v10, st_t, chunk_t, prev_groot_chunk=self.prev_chunk,
            )
            self.h_goal = h_new
            self.prev_chunk = chunk_t
            gt_p_open = torch.sigmoid(gripper_logits[0]).cpu().numpy()
        chunk_out = chunk_np.copy()
        overridden = 0
        n_apply = min(self.apply_positions, chunk_out.shape[0])
        for k in range(n_apply):
            if gt_p_open[k] > self.threshold and chunk_out[k, -1] > 0.1:
                chunk_out[k, -1] = -1.0
                overridden += 1
        self.total_overridden += overridden
        self.n_chunks += 1
        return chunk_out, {"overridden": overridden, "p_open_mean": float(gt_p_open.mean())}


class GoalImageOverlay:
    """v10 encoder + GoalImageSubstrate with explicit goal-image conditioning.

    Substrate predicts progress + goal_reached given (obs, state, chunk, goal_img).
    Intervention: when progress STALLS (Δprogress < eps over K chunks) AND
    goal_reached < threshold, force exec_horizon=1 for next force_steps env steps
    — makes GR00T re-query every step instead of executing stale chunk.

    No gripper override. Goal carrier = DINOv2 average of expert end-frames per task.
    """
    def __init__(self, v10_ckpt_path, gi_ckpt_path, goal_features_npz, device,
                  stall_eps=0.02, stall_K=3, goal_reached_threshold=0.3,
                  force_steps=32):
        import torch
        from distill_groot_flow import LiquidFlowPolicy  # type: ignore
        from goal_image_substrate import GoalImageSubstrate  # type: ignore
        self.torch = torch
        self.device = device
        self.stall_eps = stall_eps
        self.stall_K = stall_K
        self.goal_reached_threshold = goal_reached_threshold
        self.force_steps = force_steps

        v10_ck = torch.load(v10_ckpt_path, map_location=device, weights_only=False)
        sa = v10_ck["args"]
        halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
        v10 = LiquidFlowPolicy(
            state_dim=8, action_dim=7, action_horizon=sa["action_horizon"],
            d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"],
            k_max=sa["k"], halt_mode=halt_mode,
            min_steps=sa["halting_min_steps"],
            n_tasks=sa["n_tasks"], d_task=sa["d_task"],
            head_d=sa["head_d"], head_layers=sa["head_layers"],
            head_heads=sa["head_heads"],
            n_task_heads=sa.get("n_task_heads", 0),
            z_groot_dim=sa.get("z_groot_dim", 0),
            gated_mixture=sa.get("gated_mixture", False),
            z_channel_dims=sa.get("z_channel_dims", None),
            query_bank=sa.get("use_query_bank", False),
        ).to(device)
        sd = v10_ck.get("policy", v10_ck.get("model", v10_ck))
        own = v10.state_dict()
        for k, v in sd.items():
            kk = k.replace("_orig_mod.", "")
            if kk in own and own[kk].shape == v.shape:
                own[kk].copy_(v)
        v10.eval()
        for pp in v10.parameters():
            pp.requires_grad = False
        self.v10 = v10
        self.img_size = sa["img_size"]
        self.d_obs = sa["d"]

        gi_ck = torch.load(gi_ckpt_path, map_location=device, weights_only=False)
        self.gi = GoalImageSubstrate(
            d_obs=gi_ck["d_obs"], d_state=8, d_chunk=16 * 7,
            d_goal=gi_ck["d_goal"],
            d=gi_ck["d_substrate"], K=gi_ck["K_belief"], action_horizon=16,
            use_chunk=gi_ck.get("use_chunk", True),
        ).to(device)
        self.gi.load_state_dict(gi_ck["state_dict"])
        self.gi.eval()
        self.gi_use_chunk = gi_ck.get("use_chunk", True)

        # Load goal features
        gf = np.load(goal_features_npz)
        self.goal_features = {
            suite: gf[f"{suite}_features"]
            for suite in ["libero_10", "libero_object", "libero_goal", "libero_spatial"]
            if f"{suite}_features" in gf.files
        }

        self.reset()
        self.total_stalls = 0
        self.total_forces = 0
        print(f"[gi_overlay] loaded: d_obs={gi_ck['d_obs']} d_goal={gi_ck['d_goal']} "
              f"K={gi_ck['K_belief']} d_sub={gi_ck['d_substrate']} "
              f"stall_eps={stall_eps} stall_K={stall_K} thr={goal_reached_threshold}")

    def reset(self):
        self.h_goal = None
        self.prev_chunk = None
        self.progress_history = []
        self.last_progress = None
        self.last_goal_reached = None
        self.force_remaining = 0
        self.progress_peak = 0.0
        self.high_progress_count = 0  # consecutive chunks with progress >= 0.95

    def set_goal(self, suite_name, task_id):
        """Set the goal-image features for the current sub-task. Call at sub-task boundary."""
        if suite_name not in self.goal_features:
            raise KeyError(f"goal_features missing suite {suite_name}")
        import torch
        feats = self.goal_features[suite_name][task_id]
        self.current_goal_feats = torch.from_numpy(feats).float().to(self.device).unsqueeze(0)

    def query(self, chunk_np, img_raw, wrist_raw, state8):
        """Run substrate step on current (obs, chunk, goal_feats); update belief.
        Returns dict: progress, goal_reached, stalled, force_active.
        """
        import torch
        import torch.nn.functional as F
        def prep(arr):
            t = torch.from_numpy(arr).to(self.device).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0)
            if t.shape[-1] != self.img_size:
                t = F.interpolate(t, size=(self.img_size, self.img_size),
                                    mode="bilinear", align_corners=False)
            return t
        img_t = prep(img_raw)
        wri_t = prep(wrist_raw)
        st_t = torch.from_numpy(np.asarray(state8, dtype=np.float32)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            cond_v10, _ = self.v10.encoder(img_t, wri_t, st_t)
            chunk_t = torch.from_numpy(np.asarray(chunk_np, dtype=np.float32)).to(self.device).unsqueeze(0)
            if self.h_goal is None:
                self.h_goal = self.gi.init_state(1, self.device)
            h_new, progress, goal_reached_logit, info = self.gi.step(
                self.h_goal, cond_v10, st_t, chunk_t, self.current_goal_feats,
                prev_groot_chunk=self.prev_chunk,
            )
            self.h_goal = h_new
            self.prev_chunk = chunk_t
            p_val = float(progress.cpu().item())
            gr_val = float(torch.sigmoid(goal_reached_logit).cpu().item())
        self.progress_history.append(p_val)
        self.last_progress = p_val
        self.last_goal_reached = gr_val
        self.progress_peak = max(self.progress_peak, p_val)
        if p_val >= 0.95:
            self.high_progress_count += 1
        else:
            self.high_progress_count = 0

        # Stall detection (force_replan path)
        stalled = False
        if len(self.progress_history) >= self.stall_K + 1:
            recent = self.progress_history[-(self.stall_K + 1):]
            delta = max(recent) - min(recent)
            if delta < self.stall_eps and gr_val < self.goal_reached_threshold:
                stalled = True

        # Signal-derived advance: progress saturated at high value for K consecutive chunks
        # → substrate confident sub-task done. End early in chained eval (saves budget).
        signal_advance = self.high_progress_count >= 5

        # Signal-derived bail: progress peaked, now dropping → sub-task regressing
        # Peak must be substantial (>0.4) AND current must be < 0.5×peak
        signal_bail = (self.progress_peak > 0.4
                        and p_val < 0.5 * self.progress_peak
                        and len(self.progress_history) >= 10)

        if stalled:
            self.total_stalls += 1
            self.force_remaining = self.force_steps
        force_active = self.force_remaining > 0
        if force_active:
            self.total_forces += 1
        return {"progress": p_val, "goal_reached": gr_val,
                 "stalled": stalled, "force_active": force_active,
                 "force_remaining": self.force_remaining,
                 "signal_advance": signal_advance,
                 "signal_bail": signal_bail,
                 "progress_peak": self.progress_peak,
                 "high_progress_count": self.high_progress_count}

    def consume_force_step(self):
        if self.force_remaining > 0:
            self.force_remaining -= 1


class ResidualOverlay:
    """Substrate #6: residual on GR00T's chunk. Output IS the action, no
    inference-time controller. Avoids asymmetric-cost pathology that broke
    substrate variants 2-5.
    """
    def __init__(self, v10_ckpt_path, res_ckpt_path, goal_features_npz, device,
                  delta_scale=1.0):
        import torch
        from distill_groot_flow import LiquidFlowPolicy  # type: ignore
        from goal_image_residual_substrate import GoalImageResidualSubstrate  # type: ignore
        self.torch = torch
        self.device = device

        v10_ck = torch.load(v10_ckpt_path, map_location=device, weights_only=False)
        sa = v10_ck["args"]
        halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
        v10 = LiquidFlowPolicy(
            state_dim=8, action_dim=7, action_horizon=sa["action_horizon"],
            d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"],
            k_max=sa["k"], halt_mode=halt_mode,
            min_steps=sa["halting_min_steps"],
            n_tasks=sa["n_tasks"], d_task=sa["d_task"],
            head_d=sa["head_d"], head_layers=sa["head_layers"],
            head_heads=sa["head_heads"],
            n_task_heads=sa.get("n_task_heads", 0),
            z_groot_dim=sa.get("z_groot_dim", 0),
            gated_mixture=sa.get("gated_mixture", False),
            z_channel_dims=sa.get("z_channel_dims", None),
            query_bank=sa.get("use_query_bank", False),
        ).to(device)
        sd = v10_ck.get("policy", v10_ck.get("model", v10_ck))
        own = v10.state_dict()
        for k, v in sd.items():
            kk = k.replace("_orig_mod.", "")
            if kk in own and own[kk].shape == v.shape:
                own[kk].copy_(v)
        v10.eval()
        for pp in v10.parameters():
            pp.requires_grad = False
        self.v10 = v10
        self.img_size = sa["img_size"]

        res_ck = torch.load(res_ckpt_path, map_location=device, weights_only=False)
        self.res = GoalImageResidualSubstrate(
            d_obs=res_ck["d_obs"], d_state=8, d_chunk=16 * 7,
            d_goal=res_ck["d_goal"],
            d=res_ck["d_substrate"], K=res_ck["K_belief"], action_horizon=16,
            action_dim_residual=6, max_delta=res_ck.get("max_delta", 0.05),
        ).to(device)
        self.res.load_state_dict(res_ck["state_dict"])
        self.res.eval()

        gf = np.load(goal_features_npz)
        self.goal_features = {
            suite: gf[f"{suite}_features"]
            for suite in ["libero_10", "libero_object", "libero_goal", "libero_spatial"]
            if f"{suite}_features" in gf.files
        }
        self.h_goal = None
        self.total_corrections = 0
        self.delta_norm_sum = 0.0
        self.n_chunks = 0
        self.delta_scale = delta_scale
        print(f"[residual] loaded: d_obs={res_ck['d_obs']} d_goal={res_ck['d_goal']} "
              f"K={res_ck['K_belief']} d={res_ck['d_substrate']} "
              f"max_delta={res_ck.get('max_delta', 0.05)} delta_scale={delta_scale}")

    def reset(self):
        self.h_goal = None

    def set_goal(self, suite_name, task_id):
        import torch
        if suite_name not in self.goal_features:
            raise KeyError(f"goal_features missing suite {suite_name}")
        feats = self.goal_features[suite_name][task_id]
        self.current_goal_feats = torch.from_numpy(feats).float().to(self.device).unsqueeze(0)

    def correct_chunk(self, chunk_np, img_raw, wrist_raw, state8):
        """Apply substrate's residual to GR00T's chunk on xyz/rpy only."""
        import torch
        import torch.nn.functional as F
        def prep(arr):
            t = torch.from_numpy(arr).to(self.device).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0)
            if t.shape[-1] != self.img_size:
                t = F.interpolate(t, size=(self.img_size, self.img_size),
                                    mode="bilinear", align_corners=False)
            return t
        img_t = prep(img_raw)
        wri_t = prep(wrist_raw)
        st_t = torch.from_numpy(np.asarray(state8, dtype=np.float32)).to(self.device).unsqueeze(0)
        chunk_t = torch.from_numpy(np.asarray(chunk_np, dtype=np.float32)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            cond, _ = self.v10.encoder(img_t, wri_t, st_t)
            if self.h_goal is None:
                self.h_goal = self.res.init_state(1, self.device)
            h_new, delta, info = self.res.step(
                self.h_goal, cond, st_t, chunk_t, self.current_goal_feats,
            )
            self.h_goal = h_new
            delta_scaled = delta * self.delta_scale
            corrected = self.res.apply_to_chunk(chunk_t, delta_scaled)
            corrected_np = corrected[0].cpu().numpy()
            delta_norm = float(delta_scaled.abs().mean().item())
        self.delta_norm_sum += delta_norm
        self.n_chunks += 1
        return corrected_np, {"delta_norm": delta_norm,
                                "metric_cv": float(info["metric_cv"])}


def step_subtask(env, sub_task, obs, args, groot_rpc, subs=None, gi_overlay=None,
                  diag_record=None, res_overlay=None):
    """Step one sub-task until success or max_steps_per_task. Returns:
    (succeeded, steps_used, final_obs, info_dict)

    If diag_record is not None (a list), append per-step dicts with
    {step, gripper, xyz, action_xyz, action_grip} for offline analysis.
    """
    sub_lang = sub_task.language
    chunk = None
    chunk_idx = 0
    cached_z = None
    groot_chunk_target = None
    chunks_since_groot = 0
    sub_steps = 0
    depth_idxs = args.depth_indices_list
    n_groot_calls = 0
    n_subs_chunks = 0
    n_subs_overrides = 0
    p_open_sum = 0.0
    n_gi_queries = 0
    n_gi_stalls = 0
    n_gi_force_steps = 0
    progress_first = None
    progress_last = None
    goal_reached_max = 0.0
    succeeded = False
    # Signal_derived intervention state:
    advance_confirm_remaining = 0  # When >0, give env this many steps to confirm success
    advance_confirm_window = 30  # env steps after signal_advance to wait for env success
    while sub_steps < args.max_steps_per_task:
        # Effective exec horizon: 1 when goal-image overlay is forcing replan
        eff_exec_horizon = args.exec_horizon
        if gi_overlay is not None and gi_overlay.force_remaining > 0:
            eff_exec_horizon = 1
        if chunk is None or chunk_idx >= eff_exec_horizon or chunk_idx >= len(chunk):
            img_raw, wrist_raw = get_raw_imgs(obs)
            state8 = build_state8(obs)
            need_groot = (cached_z is None) or args.use_groot_chunk
            if not need_groot and args.groot_freq > 0 and chunks_since_groot >= args.groot_freq:
                need_groot = True
            if need_groot:
                groot_img = img_raw[::-1, ::-1].copy()
                groot_wrist = wrist_raw[::-1, ::-1].copy()
                obs_dict = build_groot_obs(groot_img, groot_wrist, state8, sub_lang)
                gr = groot_rpc.query_full(obs_dict, depth_idxs)
                cached_z = gr["z_vl"]
                groot_chunk_target = gr["action_chunk"]
                n_groot_calls += 1
                chunks_since_groot = 0
            else:
                chunks_since_groot += 1
            chunk = np.asarray(groot_chunk_target, dtype=np.float32)
            # Apply residual substrate (substrate variant #6: trained correction
            # on GR00T's xyz/rpy chunk; gripper unchanged). Output IS action,
            # no controller — avoids asymmetric-cost pathology of variants 2-5.
            if res_overlay is not None:
                chunk, _ = res_overlay.correct_chunk(chunk, img_raw, wrist_raw, state8)
            # Apply substrate gripper override BEFORE execution
            if subs is not None:
                chunk, ov_info = subs.override_chunk(chunk, img_raw, wrist_raw, state8)
                n_subs_chunks += 1
                n_subs_overrides += ov_info["overridden"]
                p_open_sum += ov_info["p_open_mean"]
            # Query goal-image overlay (updates belief, may trigger force or bail)
            if gi_overlay is not None:
                gi_info = gi_overlay.query(chunk, img_raw, wrist_raw, state8)
                n_gi_queries += 1
                if gi_info["stalled"]:
                    n_gi_stalls += 1
                if gi_info["force_active"]:
                    n_gi_force_steps += 1
                if progress_first is None:
                    progress_first = gi_info["progress"]
                progress_last = gi_info["progress"]
                goal_reached_max = max(goal_reached_max, gi_info["goal_reached"])
                if diag_record is not None:
                    diag_record.append({
                        "step": sub_steps,
                        "gi_progress": gi_info["progress"],
                        "gi_goal_reached": gi_info["goal_reached"],
                        "gi_stalled": gi_info["stalled"],
                    })
                # Interventions:
                # - bail: legacy stall-based exit (kept for compatibility)
                # - signal_derived: act on observed substrate signal patterns
                #   - signal_advance: substrate predicts done → arm confirmation window;
                #     exit with success ONLY if env confirms within the window
                #   - signal_bail: substrate detected regression → exit immediately (failed)
                intervention = getattr(args, "gi_intervention", "force_replan")
                bail_trigger = False
                if intervention == "bail":
                    bail_trigger = (gi_info["stalled"]
                                     and sub_steps >= getattr(args, "gi_bail_min_steps", 80))
                elif intervention == "signal_derived":
                    min_steps = getattr(args, "gi_bail_min_steps", 80)
                    if sub_steps >= min_steps:
                        # Arm confirmation window on advance signal (don't exit yet)
                        if gi_info.get("signal_advance", False) and advance_confirm_remaining <= 0:
                            advance_confirm_remaining = advance_confirm_window
                        # Bail immediately on regression signal
                        if gi_info.get("signal_bail", False):
                            bail_trigger = True
                if bail_trigger:
                    info_exit = {"n_groot_calls": n_groot_calls, "lang": sub_lang,
                                  "n_subs_chunks": n_subs_chunks,
                                  "n_subs_overrides": n_subs_overrides,
                                  "avg_overrides_per_chunk": n_subs_overrides / max(n_subs_chunks, 1),
                                  "mean_p_open": p_open_sum / max(n_subs_chunks, 1),
                                  "n_gi_queries": n_gi_queries,
                                  "n_gi_stalls": n_gi_stalls,
                                  "n_gi_force_steps": n_gi_force_steps,
                                  "progress_first": progress_first,
                                  "progress_last": progress_last,
                                  "progress_delta": (progress_last - progress_first)
                                                    if progress_first is not None else None,
                                  "goal_reached_max": goal_reached_max,
                                  "exit_reason": "bail"}
                    return False, sub_steps, obs, info_exit
            chunk_idx = 0
        action7 = chunk[chunk_idx].copy()
        g = action7[-1]
        action7[-1] = np.sign(g) if abs(g) > 0.1 else 0.0
        obs, _, done, _ = env.step(action7.astype(np.float32))
        if diag_record is not None and sub_steps % 10 == 0:
            try:
                eef_pos = obs.get("robot0_eef_pos") if isinstance(obs, dict) else None
                grip_qpos = obs.get("robot0_gripper_qpos") if isinstance(obs, dict) else None
                diag_record.append({
                    "step": sub_steps,
                    "eef_xyz": eef_pos.tolist() if eef_pos is not None else None,
                    "grip_qpos": grip_qpos.tolist() if grip_qpos is not None else None,
                    "action_xyz": action7[:3].tolist(),
                    "action_grip": float(action7[-1]),
                })
            except Exception:
                pass
        chunk_idx += 1
        sub_steps += 1
        if gi_overlay is not None:
            gi_overlay.consume_force_step()
        if env.check_success():
            succeeded = True
            break
        if done:
            break
        # Signal_derived advance confirmation: decrement window; if substrate
        # said done but env hasn't confirmed within window, give up early
        # (saves the rest of the budget for next sub-task)
        if advance_confirm_remaining > 0:
            advance_confirm_remaining -= 1
            if advance_confirm_remaining == 0:
                # Confirmation window expired without env success — exit as failed,
                # save remaining budget
                info_exit = {"n_groot_calls": n_groot_calls, "lang": sub_lang,
                              "n_subs_chunks": n_subs_chunks,
                              "n_subs_overrides": n_subs_overrides,
                              "avg_overrides_per_chunk": n_subs_overrides / max(n_subs_chunks, 1),
                              "mean_p_open": p_open_sum / max(n_subs_chunks, 1),
                              "n_gi_queries": n_gi_queries,
                              "n_gi_stalls": n_gi_stalls,
                              "n_gi_force_steps": n_gi_force_steps,
                              "progress_first": progress_first,
                              "progress_last": progress_last,
                              "progress_delta": (progress_last - progress_first)
                                                if progress_first is not None else None,
                              "goal_reached_max": goal_reached_max,
                              "exit_reason": "advance_window_expired"}
                return False, sub_steps, obs, info_exit
    info = {"n_groot_calls": n_groot_calls, "lang": sub_lang}
    if subs is not None:
        info["n_subs_chunks"] = n_subs_chunks
        info["n_subs_overrides"] = n_subs_overrides
        info["avg_overrides_per_chunk"] = n_subs_overrides / max(n_subs_chunks, 1)
        info["mean_p_open"] = p_open_sum / max(n_subs_chunks, 1)
    if gi_overlay is not None:
        info["n_gi_queries"] = n_gi_queries
        info["n_gi_stalls"] = n_gi_stalls
        info["n_gi_force_steps"] = n_gi_force_steps
        info["progress_first"] = progress_first
        info["progress_last"] = progress_last
        info["progress_delta"] = (
            (progress_last - progress_first) if progress_first is not None else None
        )
        info["goal_reached_max"] = goal_reached_max
    return succeeded, sub_steps, obs, info


def transfer_state(src_env, dst_env, src_obs):
    """Transfer ONLY the robot's joint state from src_env to dst_env.

    Different LIBERO tasks have different mujoco scenes (object count differs)
    so full sim.get_state()/set_state has incompatible qpos shapes. We transfer
    just the Franka joint positions — leaves dst_env's objects in their initial
    positions, which IS the long-horizon test: "from where robot is now, can it
    do the next sub-task?".
    """
    try:
        # Reset dst_env to baseline
        dst_env.reset()
        # Pick a known init_state from dst_env's suite (not required if we
        # immediately overwrite joints; reset gives a clean object layout)

        # Get src robot joint state — try obs first, fall back to env.robots
        src_joints = None
        if isinstance(src_obs, dict) and "robot0_joint_pos" in src_obs:
            src_joints = np.asarray(src_obs["robot0_joint_pos"], dtype=np.float64)
        elif hasattr(src_env, "robots") and src_env.robots:
            src_joints = np.asarray(src_env.robots[0]._joint_positions, dtype=np.float64)

        if src_joints is None or len(src_joints) == 0:
            return False

        # Set dst robot joints
        if hasattr(dst_env, "robots") and dst_env.robots:
            dst_env.robots[0].set_robot_joint_positions(src_joints)
            if hasattr(dst_env, "sim"):
                dst_env.sim.forward()
            # Also try to set gripper to match src
            src_gripper_qpos = None
            if hasattr(src_env, "sim") and hasattr(src_env, "robots") and src_env.robots:
                rob = src_env.robots[0]
                if hasattr(rob, "gripper") and hasattr(rob.gripper, "init_qpos"):
                    try:
                        gripper_qpos_addrs = [src_env.sim.model.get_joint_qpos_addr(j)
                                              for j in rob.gripper.joints]
                        src_gripper_qpos = np.array([src_env.sim.data.qpos[a]
                                                       for a in gripper_qpos_addrs])
                    except Exception:
                        src_gripper_qpos = None
            if src_gripper_qpos is not None and hasattr(dst_env, "robots") and dst_env.robots:
                drob = dst_env.robots[0]
                try:
                    dst_addrs = [dst_env.sim.model.get_joint_qpos_addr(j)
                                  for j in drob.gripper.joints]
                    for a, v in zip(dst_addrs, src_gripper_qpos):
                        dst_env.sim.data.qpos[a] = v
                    dst_env.sim.forward()
                except Exception:
                    pass
            return True
    except Exception as e:
        print(f"[transfer_state] failed: {type(e).__name__}: {e}")
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groot_port", type=int, default=5555)
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--chain_length", type=int, default=2)
    p.add_argument("--n_chains", type=int, default=5, help="Number of chains to evaluate")
    p.add_argument("--rollouts_per_chain", type=int, default=3)
    p.add_argument("--max_steps_per_task", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--groot_freq", type=int, default=1)
    p.add_argument("--depth_indices", default="0,1,2,3", type=str)
    p.add_argument("--use_groot_chunk", action="store_true", default=True,
                   help="Always True for chained eval — we execute GR00T's chunk directly")
    p.add_argument("--out_json", default="/tmp/chained_libero.json", type=str)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for chain selection")
    p.add_argument("--substrate_gt_ckpt", default="", type=str,
                   help="Path to trained SubstrateGoalTracker .pt. Adds gripper override.")
    p.add_argument("--substrate_v10_ckpt", default="/tmp/distill_v10_goal/step_008000.pt",
                   type=str, help="v10 ckpt for substrate encoder (must match GT training)")
    p.add_argument("--gt_threshold", type=float, default=0.5,
                   help="P(open) threshold for gripper override")
    p.add_argument("--gt_apply_positions", type=int, default=8,
                   help="Number of chunk positions to apply override on")
    p.add_argument("--gt_keep_across_subtasks", action="store_true",
                   help="Don't reset h_goal between sub-tasks (true long-horizon test)")
    p.add_argument("--gi_ckpt", default="", type=str,
                   help="Path to trained GoalImageSubstrate. Adds progress-stall replanning intervention.")
    p.add_argument("--gi_goal_features", default="/tmp/goal_features.npz", type=str,
                   help="DINOv2 goal-features per task")
    p.add_argument("--gi_stall_eps", type=float, default=0.02,
                   help="Progress-delta threshold below which sub-task is 'stalled'")
    p.add_argument("--gi_stall_K", type=int, default=3,
                   help="Number of chunks of flat progress before stall triggers")
    p.add_argument("--gi_goal_reached_thr", type=float, default=0.3,
                   help="Only consider stall if goal_reached < this")
    p.add_argument("--gi_force_steps", type=int, default=32,
                   help="Env steps to run with exec_horizon=1 after stall trigger")
    p.add_argument("--gi_intervention",
                   choices=["force_replan", "bail", "signal_derived"],
                   default="force_replan",
                   help="What substrate does: 'force_replan' (exec_horizon=1 on stall), "
                        "'bail' (end sub-task on stall), 'signal_derived' (early advance when "
                        "progress saturates high; bail when progress regresses from peak)")
    p.add_argument("--gi_bail_min_steps", type=int, default=80,
                   help="Don't bail before this many env steps (avoid premature bail at episode start)")
    p.add_argument("--res_ckpt", default="", type=str,
                   help="Substrate variant #6: residual on GR00T's chunk")
    p.add_argument("--res_goal_features", default="/tmp/goal_features.npz", type=str)
    p.add_argument("--res_delta_scale", type=float, default=1.0,
                   help="Scale residual delta at inference (0.5 = half magnitude, 0.0 = passive)")
    p.add_argument("--save_diag", default="", type=str,
                   help="If set, save per-step eef/gripper/action records for diagnostics")
    p.add_argument("--total_chain_steps", type=int, default=0,
                   help="If >0, SHARED step budget across all sub-tasks in a chain. "
                        "Forces per-sub-task efficiency to matter for long-horizon test.")
    p.add_argument("--continue_past_failures", action="store_true",
                   help="Attempt subsequent sub-tasks even after a failure (don't break)")
    p.add_argument("--retract_between_subtasks", type=int, default=0,
                   help="If >0, max retract env steps after each successful sub-task")
    p.add_argument("--retract_conditional", action="store_true",
                   help="Only retract while gripper_qpos < threshold (don't unstick what isn't stuck)")
    p.add_argument("--retract_grip_thr", type=float, default=0.030,
                   help="grip_qpos below this = gripper closed (stuck)")
    p.add_argument("--retract_hard_tasks", default="", type=str,
                   help="Comma-separated task IDs that get --retract_hard_steps instead of default")
    p.add_argument("--retract_hard_steps", type=int, default=45,
                   help="Retract steps when NEXT sub-task is in retract_hard_tasks")
    p.add_argument("--retract_easy_steps", type=int, default=20,
                   help="Retract steps when NEXT sub-task is NOT in retract_hard_tasks")
    args = p.parse_args()
    args.depth_indices_list = [int(x) for x in args.depth_indices.split(",") if x.strip()]

    groot_rpc = GrootClient(args.groot_port, name="groot", timeout_ms=60000)

    subs = None
    gi_overlay = None
    res_overlay = None
    if args.substrate_gt_ckpt or args.gi_ckpt or args.res_ckpt:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.substrate_gt_ckpt:
            subs = SubstrateOverlay(args.substrate_v10_ckpt, args.substrate_gt_ckpt,
                                     device, threshold=args.gt_threshold,
                                     apply_positions=args.gt_apply_positions)
        if args.gi_ckpt:
            gi_overlay = GoalImageOverlay(
                args.substrate_v10_ckpt, args.gi_ckpt, args.gi_goal_features,
                device, stall_eps=args.gi_stall_eps, stall_K=args.gi_stall_K,
                goal_reached_threshold=args.gi_goal_reached_thr,
                force_steps=args.gi_force_steps,
            )
        if args.res_ckpt:
            res_overlay = ResidualOverlay(
                args.substrate_v10_ckpt, args.res_ckpt, args.res_goal_features,
                device, delta_scale=args.res_delta_scale,
            )

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()

    # Build chains: sliding windows over task indices, only tasks in same SCENE
    # (LIBERO tasks have scene prefixes in BDDL filename). For simplicity, just
    # use sliding windows; tasks at the suite-level may already share scenes.
    rng = np.random.default_rng(args.seed)
    chains = []
    for c in range(args.n_chains):
        # Random distinct task ids
        ids = sorted(rng.choice(n_tasks, size=args.chain_length, replace=False).tolist())
        chains.append(ids)
    print(f"[chain] suite={args.task_suite} chains={len(chains)} chain_len={args.chain_length}")

    summary = {"suite": args.task_suite, "chain_length": args.chain_length,
                "chains": []}
    total_subtasks_completed = 0
    total_subtasks_attempted = 0
    chain_completions = []

    # Cache envs per task (avoid re-building)
    envs = {}
    def get_env(task_id):
        if task_id in envs:
            return envs[task_id]
        task = suite.get_task(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"),
                            task.problem_folder, task.bddl_file)
        e = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        envs[task_id] = e
        return e

    for chain_idx, chain in enumerate(chains):
        print(f"\n=== chain{chain_idx} {chain}: '{suite.get_task(chain[0]).language[:60]}...' ===")
        first_task = suite.get_task(chain[0])
        init_states = suite.get_task_init_states(chain[0])
        n_rollouts = min(args.rollouts_per_chain, len(init_states))
        for r in range(n_rollouts):
            t0 = time.time()
            results = []
            n_complete = 0
            chain_total_steps_used = 0
            first_env = get_env(chain[0])
            first_env.reset()
            first_env.set_init_state(init_states[r])
            obs = None
            for _ in range(5):
                obs, _, _, _ = first_env.step(np.zeros(7, dtype=np.float32))

            current_env = first_env
            # Reset substrate h_goal at start of each chain rollout
            if subs is not None and not args.gt_keep_across_subtasks:
                subs.reset()
            elif subs is not None:
                subs.reset()
            if gi_overlay is not None:
                gi_overlay.reset()
                gi_overlay.set_goal(args.task_suite, chain[0])
            if res_overlay is not None:
                res_overlay.reset()
                res_overlay.set_goal(args.task_suite, chain[0])
            for sub_idx, sub_id in enumerate(chain):
                sub_task = suite.get_task(sub_id)
                # Optionally reset substrate between sub-tasks
                if subs is not None and sub_idx > 0 and not args.gt_keep_across_subtasks:
                    subs.reset()
                if gi_overlay is not None and sub_idx > 0:
                    gi_overlay.reset()
                    gi_overlay.set_goal(args.task_suite, sub_id)
                if res_overlay is not None and sub_idx > 0:
                    res_overlay.reset()
                    res_overlay.set_goal(args.task_suite, sub_id)
                # If not the first sub-task, transfer state from previous env
                if sub_idx > 0:
                    next_env = get_env(sub_id)
                    # Set dst env to a deterministic init first (clean objects),
                    # then overwrite robot joints from src.
                    next_env.reset()
                    next_init_states = suite.get_task_init_states(sub_id)
                    next_env.set_init_state(next_init_states[r % len(next_init_states)])
                    for _ in range(3):
                        obs, _, _, _ = next_env.step(np.zeros(7, dtype=np.float32))
                    ok = transfer_state(current_env, next_env, obs)
                    if not ok:
                        results.append({"sub_idx": sub_idx, "sub_id": sub_id, "succeeded": False,
                                         "error": "state transfer failed"})
                        break
                    current_env = next_env
                    # Let sim settle after joint override
                    for _ in range(5):
                        obs, _, _, _ = current_env.step(np.zeros(7, dtype=np.float32))
                diag_buf = [] if args.save_diag else None
                # Compute effective per-sub-task step cap when total_chain_steps is set
                orig_max_per_task = args.max_steps_per_task
                if args.total_chain_steps > 0:
                    remaining = args.total_chain_steps - chain_total_steps_used
                    args.max_steps_per_task = min(orig_max_per_task, max(remaining, 0))
                    if args.max_steps_per_task <= 0:
                        # Out of total budget — record as failed and stop chain
                        results.append({"sub_idx": sub_idx, "sub_id": sub_id,
                                         "succeeded": False,
                                         "error": "out_of_total_chain_budget",
                                         "steps": 0})
                        args.max_steps_per_task = orig_max_per_task
                        break
                succ, steps, obs, info = step_subtask(
                    current_env, sub_task, obs, args, groot_rpc,
                    subs=subs, gi_overlay=gi_overlay, diag_record=diag_buf,
                    res_overlay=res_overlay,
                )
                # Restore original cap and tally
                if args.total_chain_steps > 0:
                    args.max_steps_per_task = orig_max_per_task
                chain_total_steps_used += steps
                # Retract between sub-tasks: open gripper + lift to release held object.
                # Variant #7: task-conditional retract — modulate retract steps based on
                # NEXT sub-task's difficulty (precursor to substrate-conditioned version).
                if succ and args.retract_between_subtasks > 0 and sub_idx < len(chain) - 1:
                    next_task_id = chain[sub_idx + 1]
                    hard_tasks = set()
                    if args.retract_hard_tasks:
                        hard_tasks = {int(x) for x in args.retract_hard_tasks.split(",")
                                       if x.strip()}
                    if next_task_id in hard_tasks:
                        retract_budget = args.retract_hard_steps
                    elif args.retract_hard_tasks:  # explicit hard/easy split mode
                        retract_budget = args.retract_easy_steps
                    else:
                        retract_budget = args.retract_between_subtasks
                    open_lift = np.array([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, -1.0],
                                          dtype=np.float32)
                    retract_used = 0
                    for _ in range(retract_budget):
                        if args.retract_conditional and isinstance(obs, dict):
                            gq = obs.get("robot0_gripper_qpos")
                            if gq is not None and float(np.asarray(gq).mean()) > args.retract_grip_thr:
                                break  # Gripper already open, stop retract
                        obs, _, _, _ = current_env.step(open_lift)
                        retract_used += 1
                    r_entry_extra_retract_used = retract_used
                r_entry = {"sub_idx": sub_idx, "sub_id": sub_id, "language": info["lang"],
                            "succeeded": succ, "steps": steps,
                            "n_groot_calls": info["n_groot_calls"]}
                if "n_subs_chunks" in info:
                    r_entry["n_subs_chunks"] = info["n_subs_chunks"]
                    r_entry["n_subs_overrides"] = info["n_subs_overrides"]
                    r_entry["avg_overrides_per_chunk"] = info["avg_overrides_per_chunk"]
                    r_entry["mean_p_open"] = info["mean_p_open"]
                if "n_gi_queries" in info:
                    r_entry["n_gi_queries"] = info["n_gi_queries"]
                    r_entry["n_gi_stalls"] = info["n_gi_stalls"]
                    r_entry["n_gi_force_steps"] = info["n_gi_force_steps"]
                    r_entry["progress_first"] = info["progress_first"]
                    r_entry["progress_last"] = info["progress_last"]
                    r_entry["progress_delta"] = info["progress_delta"]
                    r_entry["goal_reached_max"] = info["goal_reached_max"]
                if diag_buf:
                    r_entry["diag"] = diag_buf
                results.append(r_entry)
                if succ:
                    n_complete += 1
                elif not args.continue_past_failures:
                    break
                # Else: continue to next sub-task even though this one failed
            wall = time.time() - t0
            chain_completions.append(n_complete)
            total_subtasks_completed += n_complete
            total_subtasks_attempted += len(chain)
            extra = ""
            if subs is not None and results and "n_subs_chunks" in results[0]:
                tot_chunks = sum(r.get("n_subs_chunks", 0) for r in results)
                tot_ov = sum(r.get("n_subs_overrides", 0) for r in results)
                mean_p = sum(r.get("mean_p_open", 0) * r.get("n_subs_chunks", 0)
                              for r in results) / max(tot_chunks, 1)
                extra = (f"  subs: chunks={tot_chunks} overrides={tot_ov} "
                          f"({100*tot_ov/max(tot_chunks*8,1):.1f}%)  p_open={mean_p:.3f}")
            if gi_overlay is not None and results and "n_gi_queries" in results[0]:
                tot_q = sum(r.get("n_gi_queries", 0) for r in results)
                tot_s = sum(r.get("n_gi_stalls", 0) for r in results)
                tot_f = sum(r.get("n_gi_force_steps", 0) for r in results)
                last_progs = [r.get("progress_last") for r in results
                               if r.get("progress_last") is not None]
                gr_max = max((r.get("goal_reached_max", 0) for r in results), default=0)
                extra += (f"  gi: q={tot_q} stalls={tot_s} force_chunks={tot_f} "
                           f"prog_last={','.join(f'{p:.2f}' for p in last_progs)} "
                           f"gr_max={gr_max:.2f}")
            print(f"  r{r}: {n_complete}/{len(chain)} sub-tasks  wall={wall:.0f}s  "
                  f"steps={[r['steps'] for r in results if 'steps' in r]}{extra}")
            summary["chains"].append({
                "chain_idx": chain_idx, "rollout": r, "chain": chain,
                "n_complete": n_complete, "wall_s": wall, "results": results,
            })

    # Close envs
    for e in envs.values():
        try: e.close()
        except: pass

    summary["total_subtasks_completed"] = total_subtasks_completed
    summary["total_subtasks_attempted"] = total_subtasks_attempted
    summary["completion_rate"] = total_subtasks_completed / max(total_subtasks_attempted, 1)
    summary["mean_chain_completion"] = float(np.mean(chain_completions)) if chain_completions else 0
    print("\n" + "=" * 60)
    print(f"OVERALL: {total_subtasks_completed}/{total_subtasks_attempted} sub-tasks = {summary['completion_rate']:.0%}")
    print(f"Mean chain completion: {summary['mean_chain_completion']:.2f} / {args.chain_length}")
    print("=" * 60)
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
