"""GR00T inference server — listens on ZMQ, returns action chunks for obs queries.

Runs in main venv (CUDA torch + gr00t). Lets the LIBERO sim venv (CPU torch +
libero) query GR00T's policy in real time without any cross-package conflicts.

Wire format: pickle. Request is a dict {"video": {...}, "state": {...},
"language": [...]} matching GR00T's batch-size-1 observation. Response is
np.ndarray [K, 7] action chunk in DEMO gripper convention (-1=close, +1=open).

Run on Spark (background):
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  HF_HOME=/home/pokazge/hf_cache HF_TOKEN=hf_... python groot_server.py \\
    --teacher_path /home/pokazge/Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_10 \\
    --port 5555 --action_horizon 16
"""

from __future__ import annotations

import argparse
import functools
import pickle
import time

import numpy as np
import torch
import zmq

print = functools.partial(print, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_path", required=True, type=str)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--action_horizon", type=int, default=16)
    p.add_argument("--complexity_head_ckpt", default="", type=str,
                   help="V11: path to trained complexity head; if set, "
                        "responses include 'expected_refresh_horizon' (int).")
    args = p.parse_args()

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    from transformers.feature_extraction_utils import BatchFeature

    print(f"Loading GR00T from {args.teacher_path}...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.LIBERO_PANDA,
        model_path=str(args.teacher_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # V11: optional complexity head — predicts expected_refresh_horizon from z_vl.
    complexity_head = None
    complexity_k_max = 16
    if args.complexity_head_ckpt:
        import torch.nn as nn
        ckpt_c = torch.load(args.complexity_head_ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
        zv_dim = int(ckpt_c["z_vl_dim"])
        complexity_k_max = int(ckpt_c["k_max"])
        complexity_head = nn.Sequential(
            nn.Linear(zv_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 1),
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        complexity_head.load_state_dict(ckpt_c["head_state_dict"])
        complexity_head.eval()
        print(f"[v11] complexity head loaded from {args.complexity_head_ckpt}, "
              f"k_max={complexity_k_max}")
    modality_configs = policy.get_modality_config()
    action_keys = modality_configs["action"].modality_keys
    print(f"action_keys={action_keys}, horizon={args.action_horizon}")

    # Monkey-patch the action head's denoising loop to capture intermediate
    # (x_t, t, pred_velocity) tuples on each call. The captured trajectory is
    # stashed on the action_head as `_last_trajectory` for the server to read.
    ah = policy.model.action_head
    orig_get = ah.get_action_with_features

    @torch.no_grad()
    def get_action_with_trajectory_capture(
        self, backbone_features, state_features, embodiment_id,
        backbone_output, action_input, options=None,
    ):
        """Replacement for Gr00tN1d7ActionHead.get_action_with_features that also
        records the denoising trajectory. Mirrors the original except for the
        added capture lines. Does not support RTC inpainting (no 'action' input).

        v9: if self._pending_zvl_residual is set (shape [B, 2048] or [2048]),
        add it (broadcast across seq_len) to vl_embeds before action head runs.
        This lets the Liquid substrate inject a goal-tracking residual into
        GR00T's vl-embed representation. Residual is consumed after one use.
        """
        vl_embeds = backbone_features
        # v9: z_vl override (consumed once per call)
        # Supports: [2048] (broadcast), [B, 2048] (broadcast), [B, seq_len, 2048] (per-token)
        residual = getattr(self, "_pending_zvl_residual", None)
        if residual is not None:
            residual = residual.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
            if residual.dim() == 1:                     # [2048]
                residual = residual.unsqueeze(0).unsqueeze(0)   # [1, 1, 2048]
                vl_embeds = vl_embeds + residual          # broadcast over [B, seq, 2048]
            elif residual.dim() == 2:                    # [B, 2048]
                vl_embeds = vl_embeds + residual.unsqueeze(1)  # [B, 1, 2048]
            elif residual.dim() == 3:                    # [B, seq, 2048] — per-token
                vl_embeds = vl_embeds + residual
            else:
                raise ValueError(f"_pending_zvl_residual must be 1/2/3-D, got {residual.shape}")
            self._pending_zvl_residual = None

        # APPEND substrate's virtual tokens to vl_embeds sequence
        # _pending_vl_appended: [N, 2048] or [B, N, 2048]. Extended to vl_embeds
        # AND backbone_output.backbone_attention_mask (all-1s for these tokens)
        # AND backbone_output.image_mask (all-0s for these tokens — not image).
        appended = getattr(self, "_pending_vl_appended", None)
        if appended is not None:
            appended = appended.to(device=vl_embeds.device, dtype=vl_embeds.dtype)
            if appended.dim() == 2:                     # [N, 2048]
                appended = appended.unsqueeze(0)        # [1, N, 2048]
            if appended.dim() != 3:
                raise ValueError(f"_pending_vl_appended must be 2/3-D, got {appended.shape}")
            _, N_app, _ = appended.shape
            vl_embeds = torch.cat([vl_embeds, appended], dim=1)  # [B, seq+N, 2048]
            # Extend the attention mask in backbone_output (all 1s for new tokens)
            old_mask = backbone_output.backbone_attention_mask  # [B, seq]
            extra_mask = torch.ones((old_mask.shape[0], N_app),
                                       dtype=old_mask.dtype, device=old_mask.device)
            backbone_output.backbone_attention_mask = torch.cat(
                [old_mask, extra_mask], dim=1)
            # Extend image_mask if present (all 0s — new tokens aren't image)
            old_img = getattr(backbone_output, "image_mask", None)
            if old_img is not None:
                extra_img = torch.zeros((old_img.shape[0], N_app),
                                           dtype=old_img.dtype, device=old_img.device)
                backbone_output["image_mask"] = torch.cat(
                    [old_img, extra_img], dim=1)
            self._pending_vl_appended = None
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            (batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype, device=device,
        )
        N = self.num_inference_timesteps
        dt = 1.0 / N
        traj_xt, traj_t, traj_v = [], [], []
        traj_model_output = []  # V7a: mean-pooled DiT hidden state per depth
        for t_idx in range(N):
            t_cont = t_idx / float(N)
            t_disc = int(t_cont * self.num_timestep_buckets)
            t_tensor = torch.full((batch_size,), t_disc, device=device)
            action_features = self.action_encoder(actions, t_tensor, embodiment_id)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
            sa_embs = torch.cat((state_features, action_features), dim=1)
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs, encoder_hidden_states=vl_embeds,
                    timestep=t_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs, encoder_hidden_states=vl_embeds,
                    timestep=t_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]
            # Capture (x_t, t, v_pred) for batch item 0
            traj_xt.append(actions[0].float().cpu().numpy().copy())
            traj_t.append(float(t_cont))
            traj_v.append(pred_velocity[0].float().cpu().numpy().copy())
            # V7a: also capture the DiT hidden state (mean-pooled across seq_len).
            # Different denoising depths represent different levels of motor commitment:
            # early steps = scene+language fusion abstract intent,
            # late steps = concrete motor plan. Liquid will learn to pick the
            # right level via depth-attention.
            traj_model_output.append(
                model_output[0].float().mean(dim=0).cpu().numpy().copy()
            )
            actions = actions + dt * pred_velocity
        # Stash trajectory for the server to fetch
        self._last_trajectory = {
            "traj_xt": np.stack(traj_xt).astype(np.float32),
            "traj_t": np.array(traj_t, dtype=np.float32),
            "traj_v": np.stack(traj_v).astype(np.float32),
            "traj_model_output": np.stack(traj_model_output).astype(np.float32),
        }
        # Also stash the multimodal internal state — z_groot.
        # vl_embeds: [B, seq_len, 2048] post-Qwen3-VL backbone fused (image+lang)
        # state_features: [B, 1, hidden] proprioceptive features
        # Mean-pool vl_embeds over sequence -> [B, 2048] = "what's in scene + what to do"
        z_vl_pooled = vl_embeds[0].float().mean(dim=0).cpu().numpy().copy()
        z_state_pooled = state_features[0].float().mean(dim=0).cpu().numpy().copy()
        # X-attn variant: also stash the FULL per-token vl_embeds [seq_len, 2048]
        bb_full = vl_embeds[0].float().cpu().numpy().copy()  # [seq_len, 2048]
        # JEPA-VL: extract LANGUAGE-ONLY pool via image_mask (lang tokens = image_mask==0
        # within active attention). Falls back to full-pool if mask unavailable.
        z_lang_pooled = z_vl_pooled.copy()  # default: same as z_vl if no mask info
        img_mask = getattr(backbone_output, "image_mask", None)
        attn_mask = getattr(backbone_output, "backbone_attention_mask", None)
        if img_mask is not None and attn_mask is not None:
            lang_mask = ((img_mask == 0) & (attn_mask == 1))[0].float()  # [seq_len]
            n_lang = float(lang_mask.sum().item())
            if n_lang > 0:
                z_lang_pooled = (vl_embeds[0].float() *
                                  lang_mask.unsqueeze(-1)).sum(dim=0) / n_lang
                z_lang_pooled = z_lang_pooled.cpu().numpy().copy()
        self._last_internal_state = {
            "z_vl": z_vl_pooled.astype(np.float32),         # [2048]
            "z_lang": z_lang_pooled.astype(np.float32),     # [2048] language tokens only
            "z_state": z_state_pooled.astype(np.float32),   # [hidden]
            "z_motor": np.zeros(7, dtype=np.float32),       # filled by server loop
            "bb_full": bb_full.astype(np.float32),          # [seq_len, 2048] (X-attn)
        }
        return BatchFeature(data={
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        })

    import types
    ah.get_action_with_features = types.MethodType(get_action_with_trajectory_capture, ah)
    print("Monkey-patched action_head.get_action_with_features for trajectory capture")

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"GR00T server listening on tcp://127.0.0.1:{args.port}")

    n_served = 0
    t_start = time.time()
    last_log = t_start
    while True:
        try:
            msg = sock.recv()
            obs = pickle.loads(msg)

            if obs.get("op") == "shutdown":
                sock.send(pickle.dumps({"ok": True}))
                print("shutdown received")
                break

            op = obs.get("op", "get_action")

            if op == "get_action_with_state":
                # System 1/System 2: return action chunk + GR00T's internal
                # intent representations (z_vl, z_state, z_motor) so the
                # Liquid student can condition on the multimodal latent
                # rather than only on labeled actions.
                inner_obs = obs["obs"]
                t0 = time.perf_counter()
                action_chunk_dict, _ = policy.get_action(inner_obs)
                dt = time.perf_counter() - t0
                chunk_rows = []
                for ak in action_keys:
                    v = action_chunk_dict[ak][0]
                    chunk_rows.append(np.atleast_2d(v.reshape(args.action_horizon, -1)))
                chunk = np.concatenate(chunk_rows, axis=1).astype(np.float32)
                chunk[:, -1] = 1.0 - 2.0 * chunk[:, -1]
                istate = ah._last_internal_state
                # z_motor: mean of final action chunk [16,7] -> [7].
                # (The monkey-patch captures pred_velocity before key-concat, so its
                # dim != 7; use the already-assembled chunk instead.)
                z_motor_7 = chunk.mean(axis=0).astype(np.float32)
                # V7a: include the DiT depth trajectory (mean-pooled hidden
                # state per denoising step). Shape [N_steps, hidden_dim].
                traj = ah._last_trajectory
                # V11: complexity head — predicted refresh horizon from z_vl.
                expected_refresh_horizon = -1
                if complexity_head is not None:
                    z_vl_t = torch.from_numpy(istate["z_vl"]).to("cuda" if torch.cuda.is_available() else "cpu").float().unsqueeze(0)
                    with torch.no_grad():
                        log_h = complexity_head(z_vl_t).squeeze().item()
                    expected_refresh_horizon = int(max(1, min(complexity_k_max, round(np.expm1(log_h)))))
                sock.send(pickle.dumps({
                    "chunk": chunk,
                    "z_vl": istate["z_vl"],
                    "z_lang": istate.get("z_lang", istate["z_vl"]),
                    "z_state": istate["z_state"],
                    "z_motor": z_motor_7,
                    "bb_full": istate["bb_full"],   # [seq_len, 2048] for cross-attn substrate
                    "traj_model_output": traj["traj_model_output"],
                    "expected_refresh_horizon": expected_refresh_horizon,
                    "infer_ms": dt * 1000,
                }))
            elif op == "get_action_with_zvl_override":
                # v9: Liquid-substrate goal-tracking residual injected into vl_embeds.
                # Request: {"op": ..., "obs": <inner_obs>, "zvl_residual": [2048] np.float32}
                # The residual is ADDED to vl_embeds (broadcast across seq_len).
                inner_obs = obs["obs"]
                residual_np = obs["zvl_residual"]  # [2048] or [B, 2048]
                device = "cuda" if torch.cuda.is_available() else "cpu"
                ah._pending_zvl_residual = torch.from_numpy(
                    np.asarray(residual_np, dtype=np.float32)
                ).to(device)
                t0 = time.perf_counter()
                action_chunk_dict, _ = policy.get_action(inner_obs)
                dt = time.perf_counter() - t0
                # Safety: make sure pending was consumed (action_head should null it)
                ah._pending_zvl_residual = None
                chunk_rows = []
                for ak in action_keys:
                    v = action_chunk_dict[ak][0]
                    chunk_rows.append(np.atleast_2d(v.reshape(args.action_horizon, -1)))
                chunk = np.concatenate(chunk_rows, axis=1).astype(np.float32)
                chunk[:, -1] = 1.0 - 2.0 * chunk[:, -1]
                istate = ah._last_internal_state
                z_motor_7 = chunk.mean(axis=0).astype(np.float32)
                sock.send(pickle.dumps({
                    "chunk": chunk,
                    "z_vl_post_override": istate["z_vl"],  # this is the MODIFIED z_vl pooled
                    "z_motor": z_motor_7,
                    "infer_ms": dt * 1000,
                }))
            elif op == "get_full_bb_features":
                # X-attn: return FULL per-token backbone_features [seq_len, 2048].
                # Used for JEPA-XAttn data collection + inference.
                inner_obs = obs["obs"]
                t0 = time.perf_counter()
                action_chunk_dict, _ = policy.get_action(inner_obs)
                dt = time.perf_counter() - t0
                chunk_rows = []
                for ak in action_keys:
                    v = action_chunk_dict[ak][0]
                    chunk_rows.append(np.atleast_2d(v.reshape(args.action_horizon, -1)))
                chunk = np.concatenate(chunk_rows, axis=1).astype(np.float32)
                chunk[:, -1] = 1.0 - 2.0 * chunk[:, -1]
                istate = ah._last_internal_state
                sock.send(pickle.dumps({
                    "chunk": chunk,
                    "z_vl": istate["z_vl"],          # [2048] pooled (kept for convenience)
                    "bb_full": istate["bb_full"],    # [seq_len, 2048] PER-TOKEN
                    "infer_ms": dt * 1000,
                }))
            elif op == "get_action_with_per_token_override":
                # X-attn variant of get_action_with_zvl_override: residual is
                # per-token [seq_len, 2048] (or [B, seq_len, 2048]).
                inner_obs = obs["obs"]
                residual_np = obs["zvl_residual"]  # [seq_len, 2048] or [B, seq_len, 2048]
                device = "cuda" if torch.cuda.is_available() else "cpu"
                ah._pending_zvl_residual = torch.from_numpy(
                    np.asarray(residual_np, dtype=np.float32)
                ).to(device)
                t0 = time.perf_counter()
                action_chunk_dict, _ = policy.get_action(inner_obs)
                dt = time.perf_counter() - t0
                ah._pending_zvl_residual = None
                chunk_rows = []
                for ak in action_keys:
                    v = action_chunk_dict[ak][0]
                    chunk_rows.append(np.atleast_2d(v.reshape(args.action_horizon, -1)))
                chunk = np.concatenate(chunk_rows, axis=1).astype(np.float32)
                chunk[:, -1] = 1.0 - 2.0 * chunk[:, -1]
                istate = ah._last_internal_state
                sock.send(pickle.dumps({
                    "chunk": chunk,
                    "z_vl_post_override": istate["z_vl"],
                    "bb_full_post": istate["bb_full"],
                    "infer_ms": dt * 1000,
                }))
            elif op == "get_action_with_vl_appended":
                # APPEND substrate's virtual tokens to vl_embeds before action head.
                # Request: {"op": ..., "obs": <inner_obs>, "vl_appended": [N, 2048] np.float32}
                # Each token is concatenated to the existing vl_embeds sequence;
                # attention mask is extended with all-1s for the new tokens.
                inner_obs = obs["obs"]
                vl_app_np = obs["vl_appended"]  # [N, 2048] or [B, N, 2048]
                device = "cuda" if torch.cuda.is_available() else "cpu"
                ah._pending_vl_appended = torch.from_numpy(
                    np.asarray(vl_app_np, dtype=np.float32)
                ).to(device)
                t0 = time.perf_counter()
                action_chunk_dict, _ = policy.get_action(inner_obs)
                dt = time.perf_counter() - t0
                ah._pending_vl_appended = None
                chunk_rows = []
                for ak in action_keys:
                    v = action_chunk_dict[ak][0]
                    chunk_rows.append(np.atleast_2d(v.reshape(args.action_horizon, -1)))
                chunk = np.concatenate(chunk_rows, axis=1).astype(np.float32)
                chunk[:, -1] = 1.0 - 2.0 * chunk[:, -1]
                istate = ah._last_internal_state
                sock.send(pickle.dumps({
                    "chunk": chunk,
                    "z_vl": istate["z_vl"],
                    "infer_ms": dt * 1000,
                }))
            elif op == "get_trajectory":
                # Dense distillation: run normal get_action (which now also
                # captures trajectory thanks to the monkey-patch above), then
                # read trajectory from action_head._last_trajectory.
                inner_obs = obs["obs"]
                t0 = time.perf_counter()
                action_chunk_dict, _ = policy.get_action(inner_obs)
                dt = time.perf_counter() - t0
                # Final chunk in DEMO gripper convention
                chunk_rows = []
                for ak in action_keys:
                    v = action_chunk_dict[ak][0]
                    chunk_rows.append(np.atleast_2d(v.reshape(args.action_horizon, -1)))
                final_chunk = np.concatenate(chunk_rows, axis=1).astype(np.float32)
                final_chunk[:, -1] = 1.0 - 2.0 * final_chunk[:, -1]
                # Trajectory (in raw GR00T gripper convention; convert here)
                traj = ah._last_trajectory
                xt = traj["traj_xt"].copy()
                vv = traj["traj_v"].copy()
                xt[:, :, -1] = 1.0 - 2.0 * xt[:, :, -1]
                vv[:, :, -1] = -2.0 * vv[:, :, -1]
                sock.send(pickle.dumps({
                    "final_chunk": final_chunk,
                    "traj_xt": xt.astype(np.float32),
                    "traj_t": traj["traj_t"].astype(np.float32),
                    "traj_v": vv.astype(np.float32),
                    "infer_ms": dt * 1000,
                }))
            else:
                # Standard get_action (backwards-compatible default)
                t0 = time.perf_counter()
                action_chunk_dict, _ = policy.get_action(obs)
                chunk_rows = []
                for ak in action_keys:
                    v = action_chunk_dict[ak][0]
                    chunk_rows.append(np.atleast_2d(v.reshape(args.action_horizon, -1)))
                chunk = np.concatenate(chunk_rows, axis=1).astype(np.float32)
                chunk[:, -1] = 1.0 - 2.0 * chunk[:, -1]
                dt = time.perf_counter() - t0
                sock.send(pickle.dumps({"chunk": chunk, "infer_ms": dt * 1000}))

            n_served += 1
            now = time.time()
            if now - last_log > 30 or n_served % 50 == 0:
                rate = n_served / (now - t_start)
                print(f"  served {n_served} (rate {rate:.1f}/s, last infer {dt*1000:.1f}ms)")
                last_log = now
        except KeyboardInterrupt:
            print("interrupted")
            break
        except Exception as e:
            print(f"ERR: {e}")
            try:
                sock.send(pickle.dumps({"error": str(e)}))
            except Exception:
                pass

    sock.close()
    ctx.term()


if __name__ == "__main__":
    main()
