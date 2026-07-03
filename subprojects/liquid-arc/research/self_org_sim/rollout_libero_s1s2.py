"""System 1 / System 2 closed-loop rollout: Liquid + GR00T-as-intent-source.

Liquid (System 1) runs every chunk-prediction step, conditioned on:
  vision + state + z_groot (System-2 intent vector, cached & periodically refreshed)

GR00T (System 2) is queried every `groot_freq` chunks for a fresh z_groot. Between
queries, Liquid reuses the stale z_groot — that's the whole point: a small fast
reflex driven by an occasional, expensive intent computation.

Modes via --groot_freq:
  1   : query GR00T on every chunk (dense System-2 baseline; same as having no caching)
  K>1 : System-2 runs once per K Liquid chunk decisions
  0   : query only at episode start (extreme — test if intent generalizes across full episode)

Run inside the LIBERO sim venv (CPU torch ok). Make sure groot_server.py is
already running in main venv on the same port.

  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/activate
  python rollout_libero_s1s2.py \\
    --student_ckpt /tmp/distill_s1s2_v1/step_030000.pt \\
    --task_suite libero_10 --rollouts_per_task 5 \\
    --max_steps 720 --exec_horizon 8 --infer_steps 10 \\
    --groot_freq 4 --port 5555
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zmq
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot_flow import LiquidFlowPolicy

print = functools.partial(print, flush=True)
torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
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


def preprocess_for_liquid(img_raw, wrist_raw, target_size: int):
    img_r = np.array(Image.fromarray(img_raw).resize((target_size, target_size)), dtype=np.uint8)
    wrist_r = np.array(Image.fromarray(wrist_raw).resize((target_size, target_size)), dtype=np.uint8)
    return img_r, wrist_r


def apply_attn_to_img(img: np.ndarray, attn_grid: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """V6c: modulate uint8 image by attention map. Mirrors data-collection logic."""
    target = img.shape[0]
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    attn_pil = Image.fromarray((attn_grid * 255.0).astype(np.uint8))
    attn_up = np.array(attn_pil.resize((target, target), bilinear)).astype(np.float32) / 255.0
    scale = 1.0 + alpha * (attn_up - 0.5)
    modulated = img.astype(np.float32) * scale[:, :, None]
    return np.clip(modulated, 0.0, 255.0).astype(np.uint8)


def build_groot_obs(img_256, wrist_256, state8, task_lang: str):
    state_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    video_keys = ["image", "wrist_image"]
    language_keys = ["annotation.human.action.task_description"]
    state_slots = {
        "x": (0, 1), "y": (1, 2), "z": (2, 3),
        "roll": (3, 4), "pitch": (4, 5), "yaw": (5, 6),
        "gripper": (6, 8),
    }
    new_obs = {"video": {}, "state": {}, "language": {}}
    for k in video_keys:
        arr = img_256 if k == "image" else wrist_256
        new_obs["video"][k] = arr[None, None, ...]
    for k in state_keys:
        lo, hi = state_slots[k]
        new_obs["state"][k] = state8[lo:hi].astype(np.float32)[None, None, :]
    for lk in language_keys:
        new_obs["language"][lk] = [[task_lang]]
    return new_obs


def query_groot_state(sock, obs_dict, z_kind: str) -> np.ndarray:
    """Query GR00T server with op=get_action_with_state, return the requested z.

    `z_kind` may be a single channel name ('z_vl', 'z_state', 'z_motor') or
    a comma-separated list ('z_vl,z_state') for composite mode — channels are
    concatenated in the listed order to match the dataset's __getitem__.
    """
    sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
    resp = pickle.loads(sock.recv())
    if "error" in resp:
        raise RuntimeError(f"server error: {resp['error']}")
    channels = [c.strip() for c in z_kind.split(",") if c.strip()]
    if len(channels) == 1:
        return resp[channels[0]]
    return np.concatenate([resp[c] for c in channels]).astype(np.float32)


def query_groot_depth_bank(sock, obs_dict, depth_indices: list) -> np.ndarray:
    """V7a: query GR00T, return depth-subsampled bank from traj_model_output.

    Returns [K, hidden_dim] bank where each row is the DiT mean-pooled hidden
    state at one denoising step.
    """
    sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
    resp = pickle.loads(sock.recv())
    if "error" in resp:
        raise RuntimeError(f"server error: {resp['error']}")
    full_traj = resp["traj_model_output"]  # [N_steps, hidden_dim]
    return np.stack([full_traj[d] for d in depth_indices]).astype(np.float32)


def query_groot_full(sock, obs_dict) -> dict:
    """V8: query GR00T, return full response dict (chunk + z_vl + bank + ...)."""
    sock.send(pickle.dumps({"op": "get_action_with_state", "obs": obs_dict}))
    resp = pickle.loads(sock.recv())
    if "error" in resp:
        raise RuntimeError(f"server error: {resp['error']}")
    return resp


def setup_adaptive_optimizer(model, lr: float, include_forward_model: bool = False):
    """V8/V9: deployment-time adaptation. Freeze most params, expose only the
    small "fast geometry" components for online SGD updates: drift MLP + tau_raw
    + z_groot_proj (+ forward_pred_head for V9). The rest of Liquid stays fixed.

    These are Liquid's small learnable dynamics — the analog of cerebellar
    Purkinje synapses. ~1.4M params (V8) or ~3M params (V9 with fwd model).
    """
    # Freeze everything by default.
    for p in model.parameters():
        p.requires_grad = False

    adaptive_params = []
    enc = model.encoder
    # drift MLP (~600K)
    for p in enc.drift.parameters():
        p.requires_grad = True
        adaptive_params.append(p)
    # tau_raw (768)
    enc.tau_raw.requires_grad = True
    adaptive_params.append(enc.tau_raw)
    # z_groot_proj (~790K) — how to read System-2 response
    if enc.z_groot_proj is not None:
        for p in enc.z_groot_proj.parameters():
            p.requires_grad = True
            adaptive_params.append(p)
    # forward_pred_head (~1.6M for d=768) — V9 prediction head for self-supervised loop
    if include_forward_model and hasattr(enc, "forward_pred_head"):
        for p in enc.forward_pred_head.parameters():
            p.requires_grad = True
            adaptive_params.append(p)

    n_adaptive = sum(p.numel() for p in adaptive_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[adaptive] {n_adaptive:,} adaptable params / {n_total:,} total "
          f"({100 * n_adaptive / n_total:.1f}%) lr={lr}")
    return torch.optim.SGD(adaptive_params, lr=lr), adaptive_params


def adaptive_forward_step(model, img_t, wri_t, st_t, z_t, bank_t, delta_t,
                           predicted_next_cond, optimizer, grad_clip=1.0):
    """V9: SGD update on prediction error of next cond. Self-supervised, fires
    every chunk regardless of System-2 cadence.

    predicted_next_cond was computed at chunk t-1 (cached in caller). Here we
    compute the actual cond at chunk t and update so future predictions match.
    """
    cond_actual, _ = model.forward_encoder(
        img_t, wri_t, st_t, task_id=None,
        z_groot=z_t, z_bank=bank_t, delta_bank=delta_t,
    )
    # Detach actual to avoid backprop through encoder being pulled toward
    # the (potentially noisy) predicted target — predict to match actual.
    loss = F.mse_loss(predicted_next_cond, cond_actual.detach())
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in optimizer.param_groups[0]["params"]], grad_clip
    )
    optimizer.step()
    return loss.item()


def snapshot_adaptive(adaptive_params):
    """Return CPU clones of adaptive params for per-episode reset."""
    return [p.detach().clone() for p in adaptive_params]


def restore_adaptive(adaptive_params, snapshot):
    """Restore adaptive params from snapshot (per-episode reset)."""
    with torch.no_grad():
        for p, s in zip(adaptive_params, snapshot):
            p.copy_(s)


def adaptive_step(model, img_t, wri_t, st_t, z_t, bank_t, delta_t,
                  groot_chunk_t, optimizer):
    """V8: one SGD update on adaptive params using GR00T's chunk as flow-
    matching target. Liquid's velocity must match GR00T's straight-line flow.
    """
    cond, _ = model.forward_encoder(
        img_t, wri_t, st_t, task_id=None,
        z_groot=z_t, z_bank=bank_t, delta_bank=delta_t,
    )
    B = groot_chunk_t.shape[0]
    t = torch.rand(B, device=groot_chunk_t.device)
    noise = torch.randn_like(groot_chunk_t)
    t_b = t.view(-1, 1, 1)
    noisy = (1.0 - t_b) * noise + t_b * groot_chunk_t
    v_target = groot_chunk_t - noise
    v_pred = model.velocity(noisy, t, cond, task_id=None)
    loss = F.mse_loss(v_pred, v_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def load_flow_policy(ckpt_path: Path, device: torch.device,
                     enable_forward_model: bool = False,
                     enable_cadence_head: bool = False):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sa = ckpt["args"]
    halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"
    z_groot_dim = sa.get("z_groot_dim", 0)
    model = LiquidFlowPolicy(
        state_dim=8, action_dim=7, action_horizon=sa["action_horizon"],
        d=sa["d"], d_vis=sa["d"], img_size=sa["img_size"], k_max=sa["k"],
        halt_mode=halt_mode, min_steps=sa["halting_min_steps"],
        n_tasks=sa["n_tasks"], d_task=sa["d_task"],
        head_d=sa["head_d"], head_layers=sa["head_layers"], head_heads=sa["head_heads"],
        n_task_heads=sa.get("n_task_heads", 0),
        z_groot_dim=z_groot_dim,
        gated_mixture=sa.get("gated_mixture", False),
        z_channel_dims=sa.get("z_channel_dims", None),
        query_bank=sa.get("use_query_bank", False),
        query_dim=sa.get("query_dim", 8),
        forward_model=enable_forward_model,  # V9: forward_pred_head added if True
        cadence_head=enable_cadence_head,    # Stage 2: cadence_head added if True
    ).to(device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
    own = model.state_dict()
    loaded = 0
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v); loaded += 1
    print(f"[s1s2] loaded {loaded}/{len(own)} tensors (step={ckpt.get('step')})  "
          f"z_groot={sa.get('use_z_groot', '')} dim={z_groot_dim}")
    model.eval()
    return model, sa


@torch.no_grad()
def liquid_predict_chunk(model, img_resized, wrist_resized, state8, z_groot_np,
                         device, n_steps: int, z_bank=None, delta_bank=None):
    img_t = torch.from_numpy(img_resized).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    wri_t = torch.from_numpy(wrist_resized).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    st_t = torch.from_numpy(state8).to(device).float().unsqueeze(0)
    z_t = (torch.from_numpy(z_groot_np).to(device).float().unsqueeze(0)
           if z_groot_np is not None else None)
    bank_t = (torch.from_numpy(z_bank).to(device).float().unsqueeze(0)
              if z_bank is not None else None)
    delta_t = (torch.from_numpy(delta_bank).to(device).float().unsqueeze(0)
               if delta_bank is not None else None)
    chunk = model.sample(img_t, wri_t, st_t, task_id=None, n_steps=n_steps,
                         z_groot=z_t, z_bank=bank_t, delta_bank=delta_t)
    return chunk[0].cpu().numpy()


def run_rollout(env, model, init_state, task_lang, sock, args, device,
                target_size, z_kind, adaptive_optimizer=None,
                adaptive_snapshot=None, adaptive_params=None):
    env.reset()
    env.set_init_state(init_state)
    obs = None
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

    # V8: per-episode reset of adaptive params (safe mode). Persistent across
    # episodes if --no_reset_adapter set (more aggressive, can drift).
    if adaptive_optimizer is not None and adaptive_snapshot is not None and not args.no_reset_adapter:
        restore_adaptive(adaptive_params, adaptive_snapshot)

    chunk = None
    chunk_idx = 0
    cached_z = None       # V3/V6a/V6c: cached z_vl
    cached_bank = None    # V7a: cached depth bank [K, hidden_dim]
    cached_delta_bank = None  # V7a: identity matrix [K, K]
    z_ema = None
    # V9 forward-model state: predicted next cond from previous chunk
    pending_predicted_cond = None
    # V10/V11 adaptive-cadence state
    chunks_since_groot = 0
    v10_predicted_cond = None     # cached prediction (no grad) for next chunk's struggle check
    v10_struggle_history = []     # all per-chunk struggle scores
    v10_trigger_reasons = {"init": 0, "max_cadence": 0, "struggle": 0, "groot_hint": 0}
    v11_groot_K_advised = args.cadence_max  # GR00T's last-advised refresh horizon
    v11_groot_hint_history = []   # all GR00T-emitted horizons
    # Physics-cadence state (no learned head, no tuned threshold).
    # Algorithm: cumulate per-step ||cond_t - cond_{t-1}||_2 since last fire;
    # fire when cumulative drift exceeds median of past triggering drifts.
    # prev_cond resets to None at each fire to skip the bank-refresh discontinuity.
    # Self-calibrating per episode; no magic-number threshold.
    physics_prev_cond = None
    physics_cum_drift = 0.0
    physics_drifts_at_fires: list[float] = []
    physics_chunks_since_fire = 0
    n_groot_calls = 0
    n_chunk_calls = 0
    n_adapt_steps = 0
    adapt_loss_sum = 0.0
    success = False
    n_steps = 0
    t_groot_total = 0.0
    t_liquid_total = 0.0
    t_adapt_total = 0.0

    for step in range(args.max_steps):
        n_steps = step + 1
        if chunk is None or chunk_idx >= args.exec_horizon or chunk_idx >= len(chunk):
            img_raw, wrist_raw = get_raw_imgs(obs)
            state8 = build_state8(obs)

            # Decide whether to refresh System-2 (GR00T)
            uninitialized = (cached_z is None and cached_bank is None)
            if args.use_cadence_head:
                # Stage 2: model decides cadence via cadence_head. NO thresholds
                # set by the experimenter, NO bounds, NO if-elif controller.
                # Model emits P(fire); we either threshold or sample.
                if uninitialized:
                    need_groot = True
                    v10_trigger_reasons["init"] += 1
                else:
                    img_r_c, wrist_r_c = preprocess_for_liquid(img_raw, wrist_raw, target_size)
                    img_t_c = torch.from_numpy(img_r_c).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    wri_t_c = torch.from_numpy(wrist_r_c).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    st_t_c = torch.from_numpy(state8).to(device).float().unsqueeze(0)
                    z_t_c = (torch.from_numpy(cached_z).to(device).float().unsqueeze(0)
                             if cached_z is not None else None)
                    bank_t_c = (torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                                if cached_bank is not None else None)
                    delta_t_c = (torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                                 if cached_delta_bank is not None else None)
                    fire_p = model.emit_cadence(
                        img_t_c, wri_t_c, st_t_c,
                        z_groot=z_t_c, z_bank=bank_t_c, delta_bank=delta_t_c,
                    )
                    v10_struggle_history.append(fire_p)  # repurposed: stores fire-prob history
                    if args.cadence_stochastic:
                        need_groot = (np.random.random() < fire_p)
                    else:
                        need_groot = (fire_p > 0.5)
                    if need_groot:
                        v10_trigger_reasons["struggle"] += 1  # repurposed: model-decided fires
                if need_groot:
                    chunks_since_groot = 0
                else:
                    chunks_since_groot += 1
            elif args.physics_cadence:
                # Physics cadence: cumulate per-step ||cond_t - cond_{t-1}|| since
                # last fire; fire when cumulative drift exceeds median of past
                # triggering drifts. Bank stable between fires, so within-period
                # deltas reflect pure scene change; reset prev_cond on fire to
                # skip the post-fire bank discontinuity.
                if uninitialized:
                    need_groot = True
                else:
                    img_r_p, wrist_r_p = preprocess_for_liquid(img_raw, wrist_raw, target_size)
                    img_t_p = torch.from_numpy(img_r_p).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    wri_t_p = torch.from_numpy(wrist_r_p).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    st_t_p = torch.from_numpy(state8).to(device).float().unsqueeze(0)
                    bank_t_p = (torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                                if cached_bank is not None else None)
                    delta_t_p = (torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                                 if cached_delta_bank is not None else None)
                    with torch.no_grad():
                        cond_p, _ = model.forward_encoder(
                            img_t_p, wri_t_p, st_t_p, task_id=None,
                            z_bank=bank_t_p, delta_bank=delta_t_p,
                        )
                        cond_p_np = cond_p.squeeze(0).cpu().numpy()
                    physics_chunks_since_fire += 1
                    if physics_prev_cond is not None:
                        delta = float(np.linalg.norm(cond_p_np - physics_prev_cond))
                        physics_cum_drift += delta
                    physics_prev_cond = cond_p_np
                    # Self-calibrating fire decision.
                    if len(physics_drifts_at_fires) > 0:
                        threshold = float(np.median(physics_drifts_at_fires))
                        need_groot = (physics_cum_drift > threshold
                                       and physics_chunks_since_fire >= 2)
                    else:
                        # Bootstrap before any fire history exists: wait at least 4
                        # chunks then fire to seed the median.
                        need_groot = (physics_chunks_since_fire >= 4)
                    # Structural upper bound — never wait more than 32 chunks.
                    if physics_chunks_since_fire >= 32:
                        need_groot = True
                    if need_groot:
                        physics_drifts_at_fires.append(physics_cum_drift)
                        physics_cum_drift = 0.0
                        physics_chunks_since_fire = 0
                        physics_prev_cond = None  # skip post-fire bank discontinuity
                        v10_trigger_reasons["struggle"] += 1
                if need_groot:
                    chunks_since_groot = 0
                else:
                    chunks_since_groot += 1
            elif args.adaptive_cadence:
                # V10: struggle-gated adaptive cadence within [cadence_min, cadence_max].
                # Compute prediction error against last chunk's prediction (if any).
                cur_struggle = 0.0
                if v10_predicted_cond is not None:
                    img_r_v, wrist_r_v = preprocess_for_liquid(img_raw, wrist_raw, target_size)
                    img_t_v = torch.from_numpy(img_r_v).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    wri_t_v = torch.from_numpy(wrist_r_v).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    st_t_v = torch.from_numpy(state8).to(device).float().unsqueeze(0)
                    z_t_v = (torch.from_numpy(cached_z).to(device).float().unsqueeze(0)
                             if cached_z is not None else None)
                    bank_t_v = (torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                                if cached_bank is not None else None)
                    delta_t_v = (torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                                 if cached_delta_bank is not None else None)
                    with torch.no_grad():
                        cond_actual_v, _ = model.forward_encoder(
                            img_t_v, wri_t_v, st_t_v, task_id=None,
                            z_groot=z_t_v, z_bank=bank_t_v, delta_bank=delta_t_v,
                        )
                        cur_struggle = ((v10_predicted_cond - cond_actual_v) ** 2).mean().item()
                    v10_struggle_history.append(cur_struggle)
                    v10_predicted_cond = None  # consumed; will be set after sampling

                # V11: also respect GR00T's advised horizon if enabled
                effective_cadence_max = args.cadence_max
                if args.use_groot_hint:
                    effective_cadence_max = min(args.cadence_max, max(1, v11_groot_K_advised))

                if uninitialized:
                    need_groot = True
                    v10_trigger_reasons["init"] += 1
                elif chunks_since_groot >= effective_cadence_max:
                    need_groot = True
                    if args.use_groot_hint and effective_cadence_max < args.cadence_max:
                        v10_trigger_reasons["groot_hint"] += 1
                    else:
                        v10_trigger_reasons["max_cadence"] += 1
                elif (chunks_since_groot >= args.cadence_min
                      and cur_struggle > args.struggle_threshold):
                    need_groot = True
                    v10_trigger_reasons["struggle"] += 1
                else:
                    need_groot = False
                if need_groot:
                    chunks_since_groot = 0
                else:
                    chunks_since_groot += 1
            else:
                need_groot = (
                    uninitialized
                    or (args.groot_freq > 0 and n_chunk_calls % args.groot_freq == 0)
                )
            if need_groot:
                t0 = time.perf_counter()
                # V6a/V6c: bidirectional handshake. Liquid emits a query first;
                # GR00T is queried with the input modified by that query.
                state8_query = state8
                img_query = img_raw
                if args.query_bank and args.query_channel != "depth":
                    img_r, wrist_r = preprocess_for_liquid(img_raw, wrist_raw, target_size)
                    img_t = torch.from_numpy(img_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    wri_t = torch.from_numpy(wrist_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    st_t = torch.from_numpy(state8).to(device).float().unsqueeze(0)
                    query_emit = model.emit_query(img_t, wri_t, st_t)[0].cpu().numpy()
                    if args.query_channel == "image":
                        # V6c: emitted vector reshapes to G×G logit map; sigmoid;
                        # bilinear up-sampled and applied multiplicatively to image.
                        G = int(np.sqrt(query_emit.shape[0]))
                        attn_grid = 1.0 / (1.0 + np.exp(-query_emit.reshape(G, G).astype(np.float32)))
                        img_query = apply_attn_to_img(img_raw, attn_grid, alpha=args.attn_alpha)
                    else:
                        # V6a: Δs added to state.
                        state8_query = (state8 + query_emit).astype(np.float32)
                groot_obs = build_groot_obs(img_query, wrist_raw, state8_query, task_lang)
                # V8: query full response so we have GR00T's chunk for adaptive loss.
                # Always use full query when adaptive is on (overhead is negligible).
                groot_chunk_for_adapt = None
                if args.query_channel == "depth":
                    use_full_resp = (adaptive_optimizer is not None) or args.use_groot_hint
                    if use_full_resp:
                        resp = query_groot_full(sock, groot_obs)
                        full_traj = resp["traj_model_output"]
                        cached_bank = np.stack([full_traj[d] for d in args.depth_indices_list]).astype(np.float32)
                        groot_chunk_for_adapt = resp["chunk"]
                        # V11: capture GR00T's complexity advice
                        if args.use_groot_hint and resp.get("expected_refresh_horizon", -1) > 0:
                            v11_groot_K_advised = int(resp["expected_refresh_horizon"])
                            v11_groot_hint_history.append(v11_groot_K_advised)
                    else:
                        cached_bank = query_groot_depth_bank(sock, groot_obs, args.depth_indices_list)
                    K = cached_bank.shape[0]
                    cached_delta_bank = np.eye(K, dtype=np.float32)
                    z_new = None
                else:
                    if adaptive_optimizer is not None:
                        resp = query_groot_full(sock, groot_obs)
                        z_new = resp[z_kind] if "," not in z_kind else np.concatenate(
                            [resp[c.strip()] for c in z_kind.split(",")]
                        ).astype(np.float32)
                        groot_chunk_for_adapt = resp["chunk"]
                    else:
                        z_new = query_groot_state(sock, groot_obs, z_kind)
                t_groot_total += time.perf_counter() - t0
                n_groot_calls += 1
                # EMA smoothing test (only meaningful for non-depth channels).
                if args.query_channel != "depth":
                    if z_ema is None:
                        z_ema = z_new
                    else:
                        z_ema = args.z_ema_alpha * z_new + (1.0 - args.z_ema_alpha) * z_ema
                    cached_z = z_ema

                # V8: adaptive SGD step using GR00T's chunk as flow-matching target.
                # Skip if V9 forward-model is active (V9 uses per-chunk loop instead).
                if (adaptive_optimizer is not None
                        and groot_chunk_for_adapt is not None
                        and not args.forward_model):
                    t_adapt = time.perf_counter()
                    img_r_a, wrist_r_a = preprocess_for_liquid(img_raw, wrist_raw, target_size)
                    img_t_a = torch.from_numpy(img_r_a).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    wri_t_a = torch.from_numpy(wrist_r_a).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    st_t_a = torch.from_numpy(state8).to(device).float().unsqueeze(0)
                    z_t_a = (torch.from_numpy(cached_z).to(device).float().unsqueeze(0)
                             if cached_z is not None else None)
                    bank_t_a = (torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                                if cached_bank is not None else None)
                    delta_t_a = (torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                                 if cached_delta_bank is not None else None)
                    groot_chunk_t_a = torch.from_numpy(groot_chunk_for_adapt).to(device).float().unsqueeze(0)
                    adapt_loss = adaptive_step(
                        model, img_t_a, wri_t_a, st_t_a, z_t_a, bank_t_a, delta_t_a,
                        groot_chunk_t_a, adaptive_optimizer,
                    )
                    adapt_loss_sum += adapt_loss
                    n_adapt_steps += 1
                    t_adapt_total += time.perf_counter() - t_adapt

            # V9: per-chunk forward-model adaptation. Fires every chunk decision
            # regardless of System-2 cadence — enables K=0 adaptive learning.
            # If we have a pending prediction from the previous chunk, take SGD
            # step now (using current obs to compute actual cond, comparing
            # against the prediction made at the previous chunk).
            if (adaptive_optimizer is not None and args.forward_model
                    and pending_predicted_cond is not None):
                t_adapt = time.perf_counter()
                img_r_a, wrist_r_a = preprocess_for_liquid(img_raw, wrist_raw, target_size)
                img_t_a = torch.from_numpy(img_r_a).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                wri_t_a = torch.from_numpy(wrist_r_a).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                st_t_a = torch.from_numpy(state8).to(device).float().unsqueeze(0)
                z_t_a = (torch.from_numpy(cached_z).to(device).float().unsqueeze(0)
                         if cached_z is not None else None)
                bank_t_a = (torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                            if cached_bank is not None else None)
                delta_t_a = (torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                             if cached_delta_bank is not None else None)
                adapt_loss = adaptive_forward_step(
                    model, img_t_a, wri_t_a, st_t_a, z_t_a, bank_t_a, delta_t_a,
                    pending_predicted_cond, adaptive_optimizer,
                )
                adapt_loss_sum += adapt_loss * args.forward_loss_weight
                n_adapt_steps += 1
                pending_predicted_cond = None  # consumed
                t_adapt_total += time.perf_counter() - t_adapt

            img_r, wrist_r = preprocess_for_liquid(img_raw, wrist_raw, target_size)
            t0 = time.perf_counter()
            chunk = liquid_predict_chunk(
                model, img_r, wrist_r, state8, cached_z,
                device, n_steps=args.infer_steps,
                z_bank=cached_bank, delta_bank=cached_delta_bank,
            )
            t_liquid_total += time.perf_counter() - t0
            n_chunk_calls += 1
            chunk_idx = 0

            # V9: compute and cache predicted next cond for the next chunk's
            # SGD step. WITH grad — graph held until next chunk's backward.
            if adaptive_optimizer is not None and args.forward_model:
                img_t_p = torch.from_numpy(img_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                wri_t_p = torch.from_numpy(wrist_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                st_t_p = torch.from_numpy(state8).to(device).float().unsqueeze(0)
                z_t_p = (torch.from_numpy(cached_z).to(device).float().unsqueeze(0)
                         if cached_z is not None else None)
                bank_t_p = (torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                            if cached_bank is not None else None)
                delta_t_p = (torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                             if cached_delta_bank is not None else None)
                cond_for_pred, _ = model.forward_encoder(
                    img_t_p, wri_t_p, st_t_p, task_id=None,
                    z_groot=z_t_p, z_bank=bank_t_p, delta_bank=delta_t_p,
                )
                chunk_t_p = torch.from_numpy(chunk).to(device).float().unsqueeze(0)
                pending_predicted_cond = model.predict_next_cond(cond_for_pred, chunk_t_p)

            # V10: compute and cache predicted next cond as struggle-monitor for
            # the next chunk decision. NO grad — this is just a forecast; if it
            # diverges from reality at the next step, that's the struggle signal.
            if args.adaptive_cadence:
                with torch.no_grad():
                    img_t_v = torch.from_numpy(img_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    wri_t_v = torch.from_numpy(wrist_r).to(device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    st_t_v = torch.from_numpy(state8).to(device).float().unsqueeze(0)
                    z_t_v = (torch.from_numpy(cached_z).to(device).float().unsqueeze(0)
                             if cached_z is not None else None)
                    bank_t_v = (torch.from_numpy(cached_bank).to(device).float().unsqueeze(0)
                                if cached_bank is not None else None)
                    delta_t_v = (torch.from_numpy(cached_delta_bank).to(device).float().unsqueeze(0)
                                 if cached_delta_bank is not None else None)
                    cond_for_pred_v, _ = model.forward_encoder(
                        img_t_v, wri_t_v, st_t_v, task_id=None,
                        z_groot=z_t_v, z_bank=bank_t_v, delta_bank=delta_t_v,
                    )
                    chunk_t_v = torch.from_numpy(chunk).to(device).float().unsqueeze(0)
                    v10_predicted_cond = model.predict_next_cond(cond_for_pred_v, chunk_t_v)

        action7 = chunk[chunk_idx].copy()
        g = action7[-1]
        action7[-1] = args.gripper_sign * np.sign(g) if abs(g) > 0.1 else 0.0
        obs, _, done, _ = env.step(action7.astype(np.float32))
        chunk_idx += 1

        if env.check_success():
            success = True; break
        if done: break

    rec = {
        "success": success, "n_steps": n_steps,
        "n_groot_calls": n_groot_calls, "n_chunk_calls": n_chunk_calls,
        "groot_ms_per_call": (t_groot_total * 1000 / max(n_groot_calls, 1)),
        "liquid_ms_per_call": (t_liquid_total * 1000 / max(n_chunk_calls, 1)),
        "system2_load": n_groot_calls / max(n_chunk_calls, 1),
        "n_adapt_steps": n_adapt_steps,
        "mean_adapt_loss": adapt_loss_sum / max(n_adapt_steps, 1),
        "adapt_ms_per_step": (t_adapt_total * 1000 / max(n_adapt_steps, 1)),
    }
    if args.adaptive_cadence or args.use_cadence_head:
        rec.update({
            "mean_K": n_chunk_calls / max(n_groot_calls, 1),
            "trigger_init": v10_trigger_reasons["init"],
            "trigger_max_cadence": v10_trigger_reasons["max_cadence"],
            "trigger_struggle": v10_trigger_reasons["struggle"],  # Stage 2: reuses for fires
            "trigger_groot_hint": v10_trigger_reasons["groot_hint"],
            "mean_fire_prob": (sum(v10_struggle_history) / len(v10_struggle_history)
                                if v10_struggle_history else 0.0),  # Stage 2 fire-prob
            "max_fire_prob": (max(v10_struggle_history) if v10_struggle_history else 0.0),
            "mean_groot_hint": (sum(v11_groot_hint_history) / len(v11_groot_hint_history)
                                if v11_groot_hint_history else 0.0),
        })
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True, type=str)
    p.add_argument("--task_suite", default="libero_10", type=str)
    p.add_argument("--rollouts_per_task", type=int, default=5)
    p.add_argument("--task_indices", type=str, default="")
    p.add_argument("--max_steps", type=int, default=720)
    p.add_argument("--exec_horizon", type=int, default=8)
    p.add_argument("--infer_steps", type=int, default=10)
    p.add_argument("--gripper_sign", type=float, default=1.0)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--groot_freq", type=int, default=4,
                   help="Refresh System-2 every K Liquid chunk calls. "
                        "1=every chunk, 0=once at episode start.")
    p.add_argument("--z_ema_alpha", type=float, default=1.0,
                   help="EMA blend on z_groot updates: z = alpha*new + (1-alpha)*prev. "
                        "1.0 = no smoothing (default). Lower = heavier low-pass. "
                        "Used to test whether K=16's plateau reflects temporal "
                        "filtering vs trained-distribution effects.")
    p.add_argument("--query_bank", action="store_true",
                   help="V6a/V6c: bidirectional handshake. Liquid emits a query "
                        "and GR00T is queried with input modified by that query. "
                        "Requires checkpoint trained with --use_query_bank.")
    p.add_argument("--query_channel", default="state",
                   choices=["state", "image", "depth"],
                   help="V6a 'state' = Δs; V6c 'image' = attention; V7a 'depth' = "
                        "select over GR00T's denoising-step trajectory.")
    p.add_argument("--attn_alpha", type=float, default=0.4,
                   help="V6c image attention strength.")
    p.add_argument("--depth_indices", default="0,5,10,15", type=str,
                   help="V7a: comma-separated indices into GR00T's denoising trajectory.")
    p.add_argument("--adaptive", action="store_true",
                   help="V8: enable deployment-time adaptation. SGD update on "
                        "drift+tau+z_groot_proj using GR00T's chunk as flow target.")
    p.add_argument("--adaptive_lr", type=float, default=1e-4,
                   help="Learning rate for adaptive SGD step.")
    p.add_argument("--no_reset_adapter", action="store_true",
                   help="Persistent adaptation across episodes (default resets per episode).")
    p.add_argument("--forward_model", action="store_true",
                   help="V9: enable forward-model self-supervised adaptation. "
                        "Liquid predicts next cond at every chunk; gradient comes "
                        "from prediction error vs actual next cond. Fires every "
                        "chunk regardless of K — enables K=0 adaptive learning.")
    p.add_argument("--forward_loss_weight", type=float, default=1.0,
                   help="Weight on forward-model prediction loss (V9).")
    p.add_argument("--adaptive_cadence", action="store_true",
                   help="V10: cadence gated by Liquid's forward-model prediction "
                        "error. GR00T fires when struggle signal exceeds threshold "
                        "(within K_min..K_max bounds). Frozen weights, no SGD.")
    p.add_argument("--cadence_min", type=int, default=2,
                   help="V10: never refresh GR00T more often than every K_min chunks.")
    p.add_argument("--cadence_max", type=int, default=16,
                   help="V10: always refresh GR00T at least every K_max chunks.")
    p.add_argument("--struggle_threshold", type=float, default=0.05,
                   help="V10: prediction-error threshold (mean squared) above "
                        "which Liquid signals struggle and triggers GR00T.")
    p.add_argument("--use_groot_hint", action="store_true",
                   help="V11: use GR00T's expected_refresh_horizon as a per-call "
                        "upper bound for K (alongside V10 struggle trigger). "
                        "Requires groot_server launched with --complexity_head_ckpt.")
    p.add_argument("--physics_cadence", action="store_true",
                   help="Cadence from cond drift, no learned head. Fires when "
                        "cumulative ||cond - cond_at_last_fire||_2 exceeds median of "
                        "past triggering drifts. Self-calibrating; no tuned threshold.")
    p.add_argument("--use_cadence_head", action="store_true",
                   help="Stage 2: cadence decided by Liquid's cadence_head per "
                        "chunk. NO --groot_freq, NO thresholds, NO controllers. "
                        "Model emits P(fire); we sample/threshold ≥0.5. The "
                        "cognitive-system way of cadence — model decides K.")
    p.add_argument("--cadence_stochastic", action="store_true",
                   help="Stage 2: sample fire decision from Bernoulli(P) instead "
                        "of thresholding at 0.5. Useful for RL exploration.")
    p.add_argument("--out_json", default="", type=str)
    args = p.parse_args()
    args.depth_indices_list = [int(x) for x in args.depth_indices.split(",") if x.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # V10 also needs forward_pred_head loaded (it's a checkpoint-resident component).
    enable_fwd = args.forward_model or args.adaptive_cadence
    model, sargs = load_flow_policy(Path(args.student_ckpt), device,
                                     enable_forward_model=enable_fwd,
                                     enable_cadence_head=args.use_cadence_head)
    target_size = sargs["img_size"]
    z_kind = sargs.get("use_z_groot", "")
    if not z_kind:
        raise RuntimeError(
            "checkpoint was trained without --use_z_groot — use rollout_libero_flow.py instead"
        )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f"tcp://127.0.0.1:{args.port}")
    sock.setsockopt(zmq.RCVTIMEO, 60000)
    print(f"Connected to GR00T server on tcp://127.0.0.1:{args.port}  z_kind={z_kind}")
    print(f"groot_freq={args.groot_freq} (1=dense, 0=episode-start only)")

    # V8/V9 adaptive setup
    adaptive_optimizer = None
    adaptive_snapshot = None
    adaptive_params = None
    if args.adaptive:
        adaptive_optimizer, adaptive_params = setup_adaptive_optimizer(
            model, args.adaptive_lr, include_forward_model=args.forward_model,
        )
        adaptive_snapshot = snapshot_adaptive(adaptive_params)
        reset_str = "persistent" if args.no_reset_adapter else "per-episode reset"
        mode_tag = "V9 (forward-model)" if args.forward_model else "V8 (groot-chunk)"
        print(f"[adaptive] mode={mode_tag}, reset={reset_str}")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.get_num_tasks()
    if args.task_indices:
        task_ids = [int(t) for t in args.task_indices.split(",")]
    else:
        task_ids = list(range(n_tasks))

    summary = {
        "task_suite": args.task_suite, "z_kind": z_kind,
        "groot_freq": args.groot_freq, "tasks": [],
    }
    overall_succ, overall_total = 0, 0

    for sim_id in task_ids:
        task = suite.get_task(sim_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        init_states = suite.get_task_init_states(sim_id)
        n_rollouts = min(args.rollouts_per_task, len(init_states))
        task_lang = suite.get_task(sim_id).language
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        rollouts = []
        s2_loads = []
        print(f"\n=== sim{sim_id}: {task_lang[:80]} ===")
        for r in range(n_rollouts):
            t0 = time.time()
            rec = run_rollout(env, model, init_states[r], task_lang, sock,
                              args, device, target_size, z_kind,
                              adaptive_optimizer=adaptive_optimizer,
                              adaptive_snapshot=adaptive_snapshot,
                              adaptive_params=adaptive_params)
            rec["wall_s"] = time.time() - t0
            overall_succ += int(rec["success"]); overall_total += 1
            s2_loads.append(rec["system2_load"])
            rollouts.append(rec)
            adapt_str = (f"  adapt_loss={rec['mean_adapt_loss']:.4f} ({rec['n_adapt_steps']} steps)"
                          if args.adaptive else "")
            v10_str = (f"  meanK={rec['mean_K']:.1f} (init={rec['trigger_init']}, "
                       f"max={rec['trigger_max_cadence']}, fires={rec['trigger_struggle']}) "
                       f"fire_p_avg={rec['mean_fire_prob']:.3f}"
                       if (args.adaptive_cadence or args.use_cadence_head) else "")
            print(f"  r{r}: {'SUCCESS' if rec['success'] else 'fail'}  "
                  f"steps={rec['n_steps']:3d}  "
                  f"S2_calls={rec['n_groot_calls']}  S1_calls={rec['n_chunk_calls']}  "
                  f"S2_load={rec['system2_load']:.2f}  "
                  f"S2_ms={rec['groot_ms_per_call']:.0f}  S1_ms={rec['liquid_ms_per_call']:.1f}  "
                  f"wall={rec['wall_s']:.1f}s{adapt_str}{v10_str}")
        env.close()
        rate = sum(int(r["success"]) for r in rollouts) / max(n_rollouts, 1)
        print(f"  sim{sim_id} success: {rate:.0%}  mean_S2_load={float(np.mean(s2_loads)):.2f}")
        summary["tasks"].append({
            "sim_id": sim_id, "task": task_lang,
            "rollouts": rollouts, "success_rate": rate,
        })

    print("\n" + "=" * 80)
    print(f"OVERALL: {overall_succ}/{overall_total} = "
          f"{overall_succ/max(overall_total,1):.0%}  z_kind={z_kind}  "
          f"groot_freq={args.groot_freq}")
    print("=" * 80)
    summary["overall_successes"] = overall_succ
    summary["overall_total"] = overall_total
    summary["overall_success_rate"] = overall_succ / max(overall_total, 1)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
