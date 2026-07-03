"""GPU-side Liquid student server.

Mirrors groot_server.py pattern: listens on ZMQ REQ/REP, runs the Liquid
student + DINOv2 retrieval + adaptive SGD on the GPU. Frees the LIBERO
sim venv (CPU torch) from doing forward passes — eval was bottlenecked
on CPU at ~330ms per chunk.

Commands (pickle wire format):

  {"cmd": "init"} → {"ok": True, "info": {...}}
      Snapshot current adaptive params. Server stores snapshot internally.

  {"cmd": "episode_reset"} → {"ok": True}
      Restore adaptive params to snapshot (per-episode reset).

  {"cmd": "predict_chunk", "img_raw": <H,W,3 uint8>, "wrist_raw": ..., "state8": ...,
    "z_groot": ..., "z_bank": ..., "delta_bank": ..., "goal_img_resized": ...,
    "n_steps": int}
  → {"ok": True, "chunk": <K,7 float32>, "cond": <d float32>}

  {"cmd": "encode_retrieval", "img_raw": ..., "wrist_raw": ..., "state8": ...}
  → {"ok": True, "feat": <776 float32 L2-norm>}

  {"cmd": "predict_and_retrieve", ...}
      Combined call. Returns both chunk and retrieval feat.

  {"cmd": "adapt_v8", "img_raw": ..., "wrist_raw": ..., "state8": ...,
    "target_chunk": ..., "z_groot": ..., "z_bank": ..., "delta_bank": ...,
    "goal_img_resized": ...}
  → {"ok": True, "loss": float}

  {"cmd": "adapt_demo", "k": int, "task_language": str|None, "goal_img_resized": ...}
  → {"ok": True, "loss": float}

  {"cmd": "set_retrieval_filter", "suite": str|None}
  → {"ok": True}

Run on Spark in main venv:
  cd /home/pokazge/liquid-arc/research/self_org_sim
  source /home/pokazge/Isaac-GR00T/.venv/bin/activate
  source /home/pokazge/Isaac-GR00T/scripts/activate_spark.sh
  python liquid_server.py \\
    --student_ckpt /tmp/distill_v10_goal/step_008000.pt \\
    --memory_bank /home/pokazge/datasets/memory_bank_v11.npz \\
    --port 7777 \\
    --adaptive --adaptive_lr 1e-4 \\
    --retrieve_top_k 3 --retrieve_alpha 0.5 --retrieve_adaptive
"""
from __future__ import annotations

import argparse
import functools
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zmq

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot_flow import LiquidFlowPolicy
from retrieval import RetrievalBank


def to_tensor(arr, device, dtype=torch.float32):
    if arr is None:
        return None
    t = torch.from_numpy(np.asarray(arr)).to(device).to(dtype)
    return t


def preprocess_image_for_liquid(img_raw_uint8, target_size):
    """[H, W, 3] uint8 → [target_size, target_size, 3] uint8 via PIL resize."""
    from PIL import Image
    if img_raw_uint8.shape[0] == target_size and img_raw_uint8.shape[1] == target_size:
        return img_raw_uint8
    return np.array(
        Image.fromarray(img_raw_uint8).resize((target_size, target_size)),
        dtype=np.uint8,
    )


class LiquidServer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[liquid_server] device={self.device}")

        # SDPA backend hints (per existing rollout)
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception:
            pass
        torch.set_float32_matmul_precision("high")

        # Load model
        print(f"[liquid_server] loading {args.student_ckpt}...")
        ckpt = torch.load(args.student_ckpt, map_location=self.device, weights_only=False)
        sa = ckpt["args"]
        self.sa = sa
        self.target_size = sa["img_size"]
        halt_mode = "learned" if sa["policy"] == "liquid_halt" else "none"

        self.model = LiquidFlowPolicy(
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
            query_dim=sa.get("query_dim", 8),
            forward_model=False,
            cadence_head=False,
            gripper_head=sa.get("gripper_head", False),
            pretrained_vision=sa.get("pretrained_vision", ""),
        ).to(self.device)

        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["policy"].items()}
        own = self.model.state_dict()
        loaded = 0
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v); loaded += 1
        print(f"[liquid_server] loaded {loaded}/{len(own)} tensors")

        # Adaptive SGD setup (matches rollout_libero_s1s2.setup_adaptive_optimizer)
        self.adaptive_enabled = args.adaptive
        self.adapt_optimizer = None
        self.adapt_params = []
        self.adapt_snapshot = None
        if self.adaptive_enabled:
            self._setup_adaptive(args.adaptive_lr)

        # Demo replay (server-side, GPU)
        self.demo_replay = None
        self.demo_k = args.demo_replay_n
        if args.demo_replay_n > 0 and args.demo_replay_suites:
            # Use server-side DemoReplay matching the client's expectations.
            # We re-import the existing class to avoid drift.
            from rollout_libero_s1s2 import DemoReplay  # type: ignore
            self.DemoReplay = DemoReplay
            # We'll lazy-init per current suite via set_demo_replay_suite()
            self._demo_replay_class_ready = True
            self._demo_replay_suite_loaded = None
        else:
            self._demo_replay_class_ready = False

        # Retrieval bank
        self.retrieval_bank = None
        if args.memory_bank:
            filter_suite = args.retrieve_filter_suite_default or None
            self.retrieval_bank = RetrievalBank(
                bank_path=args.memory_bank,
                device=self.device,
                top_k=args.retrieve_top_k,
                alpha_base=args.retrieve_alpha,
                adaptive_alpha=args.retrieve_adaptive,
                filter_suite=filter_suite,
                success_only=args.retrieve_success_only,
                softmax_temperature=args.retrieve_temp,
            )

        # Phase classifier: learned P(approach_phase) from DINOv2 features.
        # Used to gate gripper override (replaces fixed step<N clamp).
        # If --substrate_phase_head is set, the classifier is the substrate-based
        # head (v18.0). Otherwise it's the MLP head (original phase classifier).
        self.phase_classifier = None
        self.phase_classifier_kind = "none"
        self.phase_threshold = args.phase_threshold
        self.phase_override_positions = args.phase_override_positions
        # === SubstrateGoalTracker (v19): persistent goal-belief observer ===
        # Sits on top of v10's full pipeline (preserves demo_replay + adapt_v8).
        # Maintains h_goal across chunk decisions, outputs per-position gripper
        # override based on goal-tracked belief.
        self.gt = None
        self.gt_h_goal = None
        self.gt_prev_chunk = None
        self.gt_threshold = args.gt_threshold
        self.gt_apply_positions = args.gt_apply_positions
        if args.goal_tracker:
            print(f"[liquid_server] loading SubstrateGoalTracker from {args.goal_tracker}")
            from substrate_goal_tracker import SubstrateGoalTracker
            gt_ckpt = torch.load(args.goal_tracker, map_location=self.device, weights_only=False)
            d_obs = gt_ckpt.get("d_obs", sa["d"])
            K_belief = gt_ckpt.get("K_belief", 8)
            d_substrate = gt_ckpt.get("d_substrate", 128)
            self.gt = SubstrateGoalTracker(
                d_obs=d_obs, d_state=8, d_chunk=16 * 7,
                d=d_substrate, K=K_belief, action_horizon=16,
            ).to(self.device)
            self.gt.load_state_dict(gt_ckpt["state_dict"])
            self.gt.eval()
            for pp in self.gt.parameters():
                pp.requires_grad = False
            print(f"[liquid_server] gt ready: K={K_belief}, d={d_substrate}, "
                  f"threshold={self.gt_threshold}, apply_positions={self.gt_apply_positions}")

        self.pm_v18 = None
        if args.phase_manager_v18:
            print(f"[liquid_server] loading PHASE MANAGER v18 from {args.phase_manager_v18}")
            from train_phase_manager_v18 import PhaseManagerV18
            pm_ckpt = torch.load(args.phase_manager_v18, map_location=self.device, weights_only=False)
            cond_dim = pm_ckpt.get("cond_dim", 768)
            d_substrate = pm_ckpt.get("d_substrate", 128)
            self.pm_v18 = PhaseManagerV18(
                cond_dim=cond_dim, state_dim=8,
                action_horizon=16, action_dim=7,
                d_substrate=d_substrate,
            ).to(self.device)
            self.pm_v18.load_state_dict(pm_ckpt["state_dict"])
            self.pm_v18.eval()
            self.phase_classifier_kind = "pm_v18"
            print(f"[liquid_server] pm_v18 ready: cond_dim={cond_dim}, d_substrate={d_substrate}")
        elif args.substrate_phase_head:
            print(f"[liquid_server] loading SUBSTRATE phase head from {args.substrate_phase_head}")
            from train_gripper_head_probe import SubstrateGripperHead
            sh_ckpt = torch.load(args.substrate_phase_head, map_location=self.device, weights_only=False)
            d_in = sh_ckpt.get("d_in", 776)
            d = sh_ckpt.get("d", 128)
            K = sh_ckpt.get("K", 4)
            self.phase_classifier = SubstrateGripperHead(d_in=d_in, d=d, K=K).to(self.device)
            self.phase_classifier.load_state_dict(sh_ckpt["state_dict"])
            self.phase_classifier.eval()
            self.phase_classifier_kind = "substrate"
            print(f"[liquid_server] substrate phase head ready: val_acc={sh_ckpt.get('val_acc', 0):.4f}")
        elif args.phase_classifier:
            print(f"[liquid_server] loading phase classifier from {args.phase_classifier}")
            pc_ckpt = torch.load(args.phase_classifier, map_location=self.device, weights_only=False)
            d_in = pc_ckpt.get("d_in", 776)
            d_hidden = pc_ckpt.get("d_hidden", 128)
            self.phase_classifier = torch.nn.Sequential(
                torch.nn.Linear(d_in, d_hidden), torch.nn.SiLU(),
                torch.nn.Linear(d_hidden, d_hidden), torch.nn.SiLU(),
                torch.nn.Linear(d_hidden, 1),
            ).to(self.device)
            # Map state dict to our Sequential format (model.net.0/2/4 → 0/2/4)
            sd = pc_ckpt["model_state"]
            sd_remap = {k.replace("net.", ""): v for k, v in sd.items()}
            self.phase_classifier.load_state_dict(sd_remap)
            self.phase_classifier.eval()
            self.phase_classifier_kind = "mlp"
            print(f"[liquid_server] phase classifier ready: kind=mlp threshold={self.phase_threshold}, "
                  f"override_positions={self.phase_override_positions}")

        self.model.eval()
        print(f"[liquid_server] ready (d={sa['d']}, img={self.target_size}, "
              f"adaptive={self.adaptive_enabled}, demo_k={self.demo_k}, "
              f"retrieval={self.retrieval_bank is not None}, "
              f"phase_classifier={self.phase_classifier is not None})")

    # -------- Adaptive helpers ----------------------------------------------

    def _setup_adaptive(self, lr):
        m = self.model
        # Freeze all first (matches setup_adaptive_optimizer pattern)
        for p in m.parameters():
            p.requires_grad = False
        enc = m.encoder
        params = []
        seen_ids = set()
        def _add(p):
            if id(p) not in seen_ids:
                p.requires_grad = True
                params.append(p)
                seen_ids.add(id(p))
        # drift MLP
        if hasattr(enc, "drift"):
            for p in enc.drift.parameters():
                _add(p)
        # tau_raw — global per-channel timescale
        if hasattr(enc, "tau_raw"):
            _add(enc.tau_raw)
        # z_groot_proj if present
        if hasattr(enc, "z_groot_proj") and enc.z_groot_proj is not None:
            for p in enc.z_groot_proj.parameters():
                _add(p)
        self.adapt_params = params
        self.adapt_optimizer = torch.optim.SGD(params, lr=lr)
        n_adapt = sum(p.numel() for p in params)
        n_total = sum(p.numel() for p in m.parameters())
        print(f"[liquid_server.adapt] {n_adapt:,}/{n_total:,} adaptable params lr={lr}")
        self.adapt_snapshot = [p.detach().clone() for p in params]

    def _restore_snapshot(self):
        with torch.no_grad():
            for p, s in zip(self.adapt_params, self.adapt_snapshot):
                p.copy_(s)

    # -------- Encoding helpers ----------------------------------------------

    def _make_image_tensors(self, img_raw, wrist_raw, state8, goal_img_resized=None):
        """Resize, normalize, push to GPU."""
        img_r = preprocess_image_for_liquid(img_raw, self.target_size)
        wri_r = preprocess_image_for_liquid(wrist_raw, self.target_size)
        img_t = torch.from_numpy(img_r).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        wri_t = torch.from_numpy(wri_r).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        st_t = torch.from_numpy(np.asarray(state8, dtype=np.float32)).to(self.device).unsqueeze(0)
        goal_t = None
        if goal_img_resized is not None:
            goal_t = torch.from_numpy(goal_img_resized).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return img_t, wri_t, st_t, goal_t

    def _wrap_z(self, arr):
        if arr is None:
            return None
        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(self.device).unsqueeze(0)

    # -------- Commands ------------------------------------------------------

    def cmd_init(self, kwargs):
        # Re-snapshot adaptive params (in case the caller adapts across episodes
        # outside per-episode reset)
        if self.adaptive_enabled:
            self.adapt_snapshot = [p.detach().clone() for p in self.adapt_params]
        return {"ok": True, "info": {"d": int(self.sa["d"]),
                                       "img_size": int(self.target_size),
                                       "action_horizon": int(self.sa["action_horizon"])}}

    def cmd_episode_reset(self, kwargs):
        if self.adaptive_enabled and self.adapt_snapshot is not None:
            self._restore_snapshot()
        # Reset goal-tracker belief state at episode boundary
        if self.gt is not None:
            self.gt_h_goal = self.gt.init_state(1, self.device)
            self.gt_prev_chunk = None
        return {"ok": True}

    @torch.no_grad()
    def cmd_predict_chunk(self, kwargs):
        img_t, wri_t, st_t, goal_t = self._make_image_tensors(
            kwargs["img_raw"], kwargs["wrist_raw"], kwargs["state8"],
            kwargs.get("goal_img_resized"),
        )
        z_t = self._wrap_z(kwargs.get("z_groot"))
        bank_t = self._wrap_z(kwargs.get("z_bank"))
        delta_t = self._wrap_z(kwargs.get("delta_bank"))
        n_steps = int(kwargs.get("n_steps", 10))
        chunk = self.model.sample(
            img_t, wri_t, st_t, task_id=None, n_steps=n_steps,
            z_groot=z_t, z_bank=bank_t, delta_bank=delta_t, goal_img=goal_t,
        )
        chunk_np = chunk[0].cpu().numpy().astype(np.float32)
        return {"ok": True, "chunk": chunk_np}

    @torch.no_grad()
    def cmd_encode_retrieval(self, kwargs):
        if self.retrieval_bank is None:
            return {"ok": False, "error": "no retrieval bank loaded"}
        img_raw = kwargs["img_raw"]
        wrist_raw = kwargs["wrist_raw"]
        state8 = kwargs["state8"]
        q = self.retrieval_bank.encode_query(img_raw, wrist_raw, state8)
        return {"ok": True, "feat": q.cpu().numpy().astype(np.float32)}

    @torch.no_grad()
    def cmd_predict_and_retrieve(self, kwargs):
        # Predict chunk
        out_pred = self.cmd_predict_chunk(kwargs)
        if not out_pred["ok"]:
            return out_pred
        chunk_np = out_pred["chunk"]
        # Retrieve and blend
        if self.retrieval_bank is not None:
            img_raw = kwargs["img_raw"]
            wrist_raw = kwargs["wrist_raw"]
            state8 = kwargs["state8"]
            final, diag = self.retrieval_bank.query_and_blend(
                img_raw, wrist_raw, state8, chunk_np,
            )
            chunk_np = final
            blend_info = {"blended": True,
                          "retrieval_mean_sim": diag["mean_sim"],
                          "retrieval_alpha": diag["alpha_used"]}
        else:
            blend_info = {"blended": False}

        # === Goal Tracker override (v19): persistent belief, overrides gripper ===
        gt_info = {}
        if self.gt is not None:
            img_raw = kwargs["img_raw"]
            wrist_raw = kwargs["wrist_raw"]
            state8 = kwargs["state8"]
            img_t, wri_t, st_t, _ = self._make_image_tensors(img_raw, wrist_raw, state8, None)
            with torch.no_grad():
                cond_v10, _ = self.model.encoder(img_t, wri_t, st_t)
                chunk_for_gt = torch.from_numpy(chunk_np).to(self.device).unsqueeze(0)
                if self.gt_h_goal is None:
                    self.gt_h_goal = self.gt.init_state(1, self.device)
                h_goal_new, gripper_logits, info = self.gt.step(
                    self.gt_h_goal, cond_v10, st_t, chunk_for_gt,
                    prev_groot_chunk=self.gt_prev_chunk,
                )
                self.gt_h_goal = h_goal_new
                self.gt_prev_chunk = chunk_for_gt
                gt_p_open = torch.sigmoid(gripper_logits[0]).cpu().numpy()  # [16]
            chunk_out = chunk_np.copy()
            overridden = 0
            n_apply = min(self.gt_apply_positions, chunk_out.shape[0])
            for k in range(n_apply):
                if gt_p_open[k] > self.gt_threshold and chunk_out[k, -1] > 0.1:
                    chunk_out[k, -1] = -1.0
                    overridden += 1
            chunk_np = chunk_out
            gt_info["gt_overridden"] = overridden
            gt_info["gt_p_open_mean"] = float(gt_p_open.mean())
            gt_info["gt_metric_cv"] = float(info["metric_cv"])

        # Phase-gated gripper override.
        phase_info = {}
        # v18 phase manager path: uses GR00T chunk + v10 cond + state as input
        if self.pm_v18 is not None and "groot_chunk" in kwargs and kwargs["groot_chunk"] is not None:
            img_raw = kwargs["img_raw"]
            wrist_raw = kwargs["wrist_raw"]
            state8 = kwargs["state8"]
            # Compute v10's cond from the encoder (need image tensors)
            img_t, wri_t, st_t, _ = self._make_image_tensors(img_raw, wrist_raw, state8, None)
            with torch.no_grad():
                cond_v10, _ = self.model.encoder(img_t, wri_t, st_t)
                groot_chunk_t = torch.from_numpy(
                    np.asarray(kwargs["groot_chunk"], dtype=np.float32)
                ).to(self.device).unsqueeze(0)
                stale = torch.tensor([float(kwargs.get("chunks_since_groot", 0))], device=self.device)
                pm_out = self.pm_v18(groot_chunk_t, cond_v10, st_t, staleness=stale)
                pm_logits = pm_out["logits"][0].cpu().numpy()  # [16]
                pm_p_open = 1.0 / (1.0 + np.exp(-pm_logits))
            # Override gripper per-chunk-position
            chunk_out = chunk_np.copy()
            overridden = 0
            for k in range(chunk_out.shape[0]):
                if pm_p_open[k] > self.phase_threshold and chunk_out[k, -1] > 0.1:
                    chunk_out[k, -1] = -1.0
                    overridden += 1
            chunk_np = chunk_out
            phase_info["pm_v18_overridden"] = overridden
            phase_info["pm_v18_p_open_mean"] = float(pm_p_open.mean())
        elif self.phase_classifier is not None and self.retrieval_bank is not None:
            img_raw = kwargs["img_raw"]
            wrist_raw = kwargs["wrist_raw"]
            state8 = kwargs["state8"]
            q = self.retrieval_bank.encode_query(img_raw, wrist_raw, state8)
            logit = self.phase_classifier(q.unsqueeze(0)).squeeze()
            p_approach = float(torch.sigmoid(logit).item())
            phase_info["p_approach"] = p_approach
            if p_approach > self.phase_threshold:
                n_override = min(self.phase_override_positions, chunk_np.shape[0])
                overridden = 0
                chunk_out = chunk_np.copy()
                for k in range(n_override):
                    if chunk_out[k, -1] > 0.1:
                        chunk_out[k, -1] = -1.0
                        overridden += 1
                chunk_np = chunk_out
                phase_info["phase_overridden_positions"] = overridden

        return {
            "ok": True,
            "chunk": chunk_np,
            **blend_info,
            **phase_info,
            **gt_info,
        }

    def cmd_adapt_v8(self, kwargs):
        if not self.adaptive_enabled:
            return {"ok": False, "error": "adaptive not enabled"}
        img_t, wri_t, st_t, goal_t = self._make_image_tensors(
            kwargs["img_raw"], kwargs["wrist_raw"], kwargs["state8"],
            kwargs.get("goal_img_resized"),
        )
        z_t = self._wrap_z(kwargs.get("z_groot"))
        bank_t = self._wrap_z(kwargs.get("z_bank"))
        delta_t = self._wrap_z(kwargs.get("delta_bank"))
        target_chunk = torch.from_numpy(
            np.asarray(kwargs["target_chunk"], dtype=np.float32)
        ).to(self.device).unsqueeze(0)

        self.adapt_optimizer.zero_grad()
        # Sample a t and noise; do flow matching loss with target as target velocity
        # Same as adaptive_step in rollout_libero_s1s2.py: predict velocity, MSE with target chunk derivative.
        # The simpler form used: MSE between model.sample's chunk and target chunk.
        # We reuse model.flow_loss if it exists, else compute MSE on velocity.
        # Re-create the V8 adaptive step semantics here using model.flow_loss API.
        cond, _ = self.model.forward_encoder(
            img_t, wri_t, st_t, task_id=None,
            z_groot=z_t, z_bank=bank_t, delta_bank=delta_t, goal_img=goal_t,
        )
        # Flow matching target velocity loss
        bs = target_chunk.shape[0]
        t_rand = torch.rand(bs, device=self.device)
        noise = torch.randn_like(target_chunk)
        v_target = target_chunk - noise  # rectified flow target velocity
        x_t = t_rand.view(-1, 1, 1) * target_chunk + (1.0 - t_rand.view(-1, 1, 1)) * noise
        v_pred = self.model.velocity(x_t, t_rand, cond=cond)
        loss = F.mse_loss(v_pred, v_target)
        loss.backward()
        self.adapt_optimizer.step()
        return {"ok": True, "loss": float(loss.detach().cpu())}

    def cmd_adapt_demo(self, kwargs):
        if not self.adaptive_enabled:
            return {"ok": False, "error": "adaptive not enabled"}
        if not self._demo_replay_class_ready:
            return {"ok": False, "error": "demo replay not configured"}
        # Lazy-init demo replay for current suite
        suite = kwargs.get("suite", self.args.demo_replay_suites)
        if self._demo_replay_suite_loaded != suite:
            self.demo_replay = self.DemoReplay(suite, self.target_size, self.device)
            self._demo_replay_suite_loaded = suite
        task_lang = kwargs.get("task_language", None)
        if task_lang is not None and hasattr(self.demo_replay, "set_task_filter"):
            self.demo_replay.set_task_filter(task_lang)
        # Optional: per-iter goal_img tensor (uses suite canonical goal already in batch).
        from rollout_libero_s1s2 import adaptive_demo_step  # type: ignore
        goal_img_resized = kwargs.get("goal_img_resized", None)
        goal_t = None
        if goal_img_resized is not None:
            goal_t = torch.from_numpy(goal_img_resized).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        loss = adaptive_demo_step(
            self.model, self.demo_replay, self.adapt_optimizer,
            k=self.demo_k, goal_img=goal_t,
        )
        return {"ok": True, "loss": float(loss)}

    def cmd_set_retrieval_filter(self, kwargs):
        """Switch retrieval bank's per-suite filter. Rebuilds the GPU index."""
        if self.retrieval_bank is None:
            return {"ok": False, "error": "no retrieval bank"}
        # Re-load with new filter — keeps DINOv2 backbone, just remasks features
        new_suite = kwargs.get("suite")
        # Easiest approach: re-instantiate the bank with new filter.
        # Capture current settings and rebuild.
        rb = self.retrieval_bank
        self.retrieval_bank = RetrievalBank(
            bank_path=self.args.memory_bank,
            device=self.device,
            top_k=rb.top_k,
            alpha_base=rb.alpha_base,
            adaptive_alpha=rb.adaptive_alpha,
            filter_suite=new_suite,
            success_only=self.args.retrieve_success_only,
            softmax_temperature=rb.softmax_temperature,
        )
        return {"ok": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True, type=str)
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--memory_bank", default="", type=str)
    # Adaptive flags
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--adaptive_lr", type=float, default=1e-4)
    p.add_argument("--demo_replay_n", type=int, default=0)
    p.add_argument("--demo_replay_suites", default="libero_10", type=str,
                   help="Default suite for demo replay; client can override via cmd_adapt_demo")
    # Retrieval flags
    p.add_argument("--retrieve_top_k", type=int, default=3)
    p.add_argument("--retrieve_alpha", type=float, default=0.5)
    p.add_argument("--retrieve_adaptive", action="store_true")
    p.add_argument("--retrieve_success_only", action="store_true")
    p.add_argument("--retrieve_temp", type=float, default=8.0)
    p.add_argument("--retrieve_filter_suite_default", default="", type=str)
    p.add_argument("--phase_classifier", default="", type=str,
                   help="Path to phase_classifier.pt (learned P(approach) head, MLP). "
                        "If set, replaces fixed-step gripper clamp with learned phase gating.")
    p.add_argument("--substrate_phase_head", default="", type=str,
                   help="v18.0: path to substrate_head.pt (4-position substrate gripper head). "
                        "Takes precedence over --phase_classifier when both are set.")
    p.add_argument("--phase_manager_v18", default="", type=str,
                   help="v18 path: substrate phase manager that uses GR00T chunk + v10 cond + state. "
                        "Takes precedence over --substrate_phase_head and --phase_classifier.")
    p.add_argument("--goal_tracker", default="", type=str,
                   help="v19: path to trained SubstrateGoalTracker. Adds per-episode goal-belief "
                        "overlay ON TOP of v10 pipeline (demo_replay + adapt_v8 still active).")
    p.add_argument("--gt_threshold", type=float, default=0.5,
                   help="P(open) threshold for GT gripper override.")
    p.add_argument("--gt_apply_positions", type=int, default=8,
                   help="Number of chunk positions to apply GT override to.")
    p.add_argument("--phase_threshold", type=float, default=0.7,
                   help="If P(approach)>threshold, override gripper-close predictions to -1.")
    p.add_argument("--phase_override_positions", type=int, default=4,
                   help="Number of leading chunk positions whose gripper to override "
                        "when phase=approach AND model says close.")
    args = p.parse_args()

    server = LiquidServer(args)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[liquid_server] listening on tcp://*:{args.port}")

    HANDLERS = {
        "init": server.cmd_init,
        "episode_reset": server.cmd_episode_reset,
        "predict_chunk": server.cmd_predict_chunk,
        "encode_retrieval": server.cmd_encode_retrieval,
        "predict_and_retrieve": server.cmd_predict_and_retrieve,
        "adapt_v8": server.cmd_adapt_v8,
        "adapt_demo": server.cmd_adapt_demo,
        "set_retrieval_filter": server.cmd_set_retrieval_filter,
    }

    while True:
        try:
            raw = sock.recv()
        except KeyboardInterrupt:
            print("\n[liquid_server] interrupt; shutting down.")
            break
        try:
            req = pickle.loads(raw)
            cmd = req["cmd"]
            kwargs = {k: v for k, v in req.items() if k != "cmd"}
            handler = HANDLERS.get(cmd)
            if handler is None:
                resp = {"ok": False, "error": f"unknown cmd {cmd}"}
            else:
                t0 = time.perf_counter()
                resp = handler(kwargs)
                if isinstance(resp, dict):
                    resp.setdefault("server_ms", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            import traceback
            resp = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
        sock.send(pickle.dumps(resp))


if __name__ == "__main__":
    main()
