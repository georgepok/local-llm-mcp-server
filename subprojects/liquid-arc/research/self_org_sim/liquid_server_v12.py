"""v12 LiquidARC student server (GPU-side ZMQ inference).

Adapts liquid_server.py to the V12Policy + LiquidARCVisionEncoder shape.
Same wire protocol as v11 server — the LIBERO sim client doesn't need to
know which version it's talking to, only the ckpt is different.

Differences from v11 server:
  - Model: V12Policy (LiquidARCVisionEncoder + FlowMatchingHead + gripper_head)
  - No z_groot / z_bank / delta_bank inputs at inference (v12 substrate doesn't use them)
  - goal_img conditioning is OFF (v12 trained without it per catalog 14.9)
  - Separate sigmoid gripper head sampled alongside continuous chunk

Commands (same as v11 server):
  init, episode_reset, predict_chunk, encode_retrieval,
  predict_and_retrieve, adapt_v8, adapt_demo, set_retrieval_filter

Run on Spark in main venv:
  python liquid_server_v12.py --student_ckpt /tmp/distill_v12/step_007000.pt --port 7777 \\
    --memory_bank /home/pokazge/datasets/memory_bank_v11.npz \\
    --adaptive --adaptive_lr 1e-4 --demo_replay_n 4 \\
    --retrieve_top_k 3 --retrieve_alpha 0.3 --retrieve_adaptive
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
from distill_groot_v12 import V12Policy
from liquid_arc_substrate_libero import make_v12_config
from retrieval import RetrievalBank


def preprocess_image(img_raw_uint8, target_size=224):
    from PIL import Image
    if img_raw_uint8.shape[0] == target_size and img_raw_uint8.shape[1] == target_size:
        return img_raw_uint8
    return np.array(
        Image.fromarray(img_raw_uint8).resize((target_size, target_size)),
        dtype=np.uint8,
    )


class LiquidServerV12:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[v12_server] device={self.device}")
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception:
            pass
        torch.set_float32_matmul_precision("high")

        # === Load v12 checkpoint ===
        print(f"[v12_server] loading {args.student_ckpt}...")
        ckpt = torch.load(args.student_ckpt, map_location=self.device, weights_only=False)
        config_dict = ckpt["config"]
        # Reconstruct config from saved dict
        config = make_v12_config()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        self.config = config

        train_args = ckpt["args"]
        use_goal_img = train_args.get("use_goal_img", False)
        self.target_size = 224  # DINOv2 native
        self.use_goal_img = use_goal_img

        # v12.2: probe ckpt for z_vl_dim from saved state_dict
        z_vl_dim = 0
        for k, v in ckpt["policy"].items():
            if "z_groot_proj.weight" in k:
                z_vl_dim = int(v.shape[1])
                break
        self.z_vl_dim = z_vl_dim
        if z_vl_dim > 0:
            print(f"[v12_server] z_vl_dim={z_vl_dim} detected — substrate uses z_vl")

        self.model = V12Policy(
            config, action_horizon=16, action_dim=7, state_dim=8,
            head_d=256, head_layers=4, head_heads=4,
            use_goal_img=use_goal_img,
            z_vl_dim=z_vl_dim,
        ).to(self.device)

        sd = ckpt["policy"]
        own = self.model.state_dict()
        loaded = 0
        for k, v in sd.items():
            kk = k.replace("_orig_mod.", "")
            if kk in own and own[kk].shape == v.shape:
                own[kk].copy_(v); loaded += 1
        print(f"[v12_server] loaded {loaded}/{len(own)} tensors (step={ckpt.get('step', '?')})")

        # === Adaptive setup ===
        self.adaptive_enabled = args.adaptive
        self.adapt_optimizer = None
        self.adapt_params = []
        self.adapt_snapshot = None
        if self.adaptive_enabled:
            self._setup_adaptive(args.adaptive_lr)

        # === Demo replay ===
        self.demo_replay = None
        self.demo_k = args.demo_replay_n
        if args.demo_replay_n > 0:
            from rollout_libero_s1s2 import DemoReplay  # type: ignore
            self.DemoReplay = DemoReplay
            self._demo_replay_loaded_suite = None
        else:
            self.DemoReplay = None

        # === Retrieval bank ===
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

        self.model.eval()
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[v12_server] ready (d={config.d_model}, halting={config.halting_enabled}, "
              f"params={n_total:,}, adaptive={self.adaptive_enabled}, "
              f"demo_k={self.demo_k}, retrieval={self.retrieval_bank is not None}, "
              f"goal_img={self.use_goal_img})")

    def _setup_adaptive(self, lr):
        # v12 adaptive: small fast-geometry components only (analog of v10's drift+tau+z_groot_proj)
        # For v12: dynamics.drift would be the analog, but ContinuousDynamics has different structure.
        # We adapt: MetricNet linear2 diag bias + tau_net (small bottleneck) + structural_tau
        for p in self.model.parameters():
            p.requires_grad = False
        enc = self.model.encoder
        dyn = enc.dynamics
        params = []
        seen = set()
        def _add(p):
            if id(p) not in seen:
                p.requires_grad = True
                params.append(p)
                seen.add(id(p))
        # Tau path (small)
        if hasattr(dyn, "tau_net_linear1"):
            for p in dyn.tau_net_linear1.parameters(): _add(p)
            for p in dyn.tau_net_linear2.parameters(): _add(p)
        # Structural tau
        if hasattr(dyn, "structural_tau") and dyn.structural_tau is not None:
            _add(dyn.structural_tau)
        # Metric net (the "fast geometry")
        if hasattr(dyn, "metric_net_linear1"):
            for p in dyn.metric_net_linear1.parameters(): _add(p)
            for p in dyn.metric_net_linear2_diag.parameters(): _add(p)
        # ReZero gate
        if hasattr(dyn, "rezero_gate_logit"):
            _add(dyn.rezero_gate_logit)
        self.adapt_params = params
        self.adapt_optimizer = torch.optim.SGD(params, lr=lr)
        n_adapt = sum(p.numel() for p in params)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[v12_server.adapt] {n_adapt:,}/{n_total:,} adaptable params lr={lr}")
        self.adapt_snapshot = [p.detach().clone() for p in params]

    def _restore_snapshot(self):
        with torch.no_grad():
            for p, s in zip(self.adapt_params, self.adapt_snapshot):
                p.copy_(s)

    # -------- v12-specific helpers ----------------------------------------

    def _make_image_tensors(self, img_raw, wrist_raw, state8, goal_img_resized=None):
        img_r = preprocess_image(img_raw, self.target_size)
        wri_r = preprocess_image(wrist_raw, self.target_size)
        img_t = torch.from_numpy(img_r).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        wri_t = torch.from_numpy(wri_r).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        st_t = torch.from_numpy(np.asarray(state8, dtype=np.float32)).to(self.device).unsqueeze(0)
        goal_t = None
        if goal_img_resized is not None and self.use_goal_img:
            goal_t = torch.from_numpy(goal_img_resized).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        return img_t, wri_t, st_t, goal_t

    def _wrap_z(self, z):
        """Wrap z numpy → [1, dim] GPU tensor. For raw z (1D) only."""
        if z is None:
            return None
        arr = np.asarray(z, dtype=np.float32)
        return torch.from_numpy(arr).to(self.device).unsqueeze(0)

    def _z_vl_from_bank(self, z_bank):
        """v12.3: compute z_vl signal as mean over depth-bank K samples.

        Training used z_vl_bank.dat (depth-subsampled traj_model_output, K×1024).
        Inference receives z_bank from client (current GR00T's traj_model_output[depth_indices], K×1024).
        Mean over K matches the training-time aggregation.
        """
        if z_bank is None:
            return None
        arr = np.asarray(z_bank, dtype=np.float32)
        if arr.ndim == 2:  # [K, dim]
            arr = arr.mean(axis=0)  # → [dim]
        return torch.from_numpy(arr).to(self.device).unsqueeze(0)

    @torch.no_grad()
    def _v12_sample_chunk(self, img_t, wri_t, st_t, goal_t, n_steps, z_vl_t=None):
        """Rectified flow sampling.

        Gripper override DISABLED — the per-timestep gripper head
        (single Linear → 16-d) can't learn temporal grasp transitions
        from a static cond vector. Verified offline: head predicts
        constant gripper across all 16 timesteps, killing all sims
        requiring open/close transitions (libero_object, libero_goal).

        Fall back to the flow-head's continuous gripper output; the
        LIBERO client sign-snaps to ±1 at execution.
        """
        out = self.model.encode(img_t, wri_t, st_t, goal_img=goal_t, z_vl=z_vl_t)
        cond = out["cond"]
        B = cond.shape[0]
        K = self.model.action_horizon
        A = self.model.action_dim
        x = torch.randn(B, K, A, device=self.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = torch.full((B,), i * dt, device=self.device)
            v = self.model.velocity(x, t_val, cond)
            x = x + dt * v
        return x[0].cpu().numpy().astype(np.float32)

    # -------- Commands -----------------------------------------------------

    def cmd_init(self, kwargs):
        if self.adaptive_enabled:
            self.adapt_snapshot = [p.detach().clone() for p in self.adapt_params]
        return {"ok": True, "info": {
            "d": int(self.config.d_model),
            "img_size": int(self.target_size),
            "action_horizon": 16,
            "version": "v12",
        }}

    def cmd_episode_reset(self, kwargs):
        if self.adaptive_enabled and self.adapt_snapshot is not None:
            self._restore_snapshot()
        return {"ok": True}

    @torch.no_grad()
    def cmd_predict_chunk(self, kwargs):
        img_t, wri_t, st_t, goal_t = self._make_image_tensors(
            kwargs["img_raw"], kwargs["wrist_raw"], kwargs["state8"],
            kwargs.get("goal_img_resized"),
        )
        n_steps = int(kwargs.get("n_steps", 10))
        # v12.3: derive z_vl from z_bank mean (dim-compatible with training)
        z_vl_t = self._z_vl_from_bank(kwargs.get("z_bank")) if self.z_vl_dim > 0 else None
        chunk = self._v12_sample_chunk(img_t, wri_t, st_t, goal_t, n_steps, z_vl_t=z_vl_t)
        return {"ok": True, "chunk": chunk}

    @torch.no_grad()
    def cmd_encode_retrieval(self, kwargs):
        if self.retrieval_bank is None:
            return {"ok": False, "error": "no retrieval bank loaded"}
        q = self.retrieval_bank.encode_query(
            kwargs["img_raw"], kwargs["wrist_raw"], kwargs["state8"],
        )
        return {"ok": True, "feat": q.cpu().numpy().astype(np.float32)}

    @torch.no_grad()
    def cmd_predict_and_retrieve(self, kwargs):
        # Sample v12 chunk
        out = self.cmd_predict_chunk(kwargs)
        if not out["ok"]:
            return out
        chunk_np = out["chunk"]
        if self.retrieval_bank is None:
            return {"ok": True, "chunk": chunk_np, "blended": False}
        final, diag = self.retrieval_bank.query_and_blend(
            kwargs["img_raw"], kwargs["wrist_raw"], kwargs["state8"], chunk_np,
        )
        return {
            "ok": True, "chunk": final, "blended": True,
            "retrieval_mean_sim": diag["mean_sim"],
            "retrieval_alpha": diag["alpha_used"],
        }

    def cmd_adapt_v8(self, kwargs):
        """V8 adaptive: flow matching + gripper BCE on GR00T target."""
        if not self.adaptive_enabled:
            return {"ok": False, "error": "adaptive not enabled"}
        img_t, wri_t, st_t, goal_t = self._make_image_tensors(
            kwargs["img_raw"], kwargs["wrist_raw"], kwargs["state8"],
            kwargs.get("goal_img_resized"),
        )
        target_chunk = torch.from_numpy(
            np.asarray(kwargs["target_chunk"], dtype=np.float32)
        ).to(self.device).unsqueeze(0)

        # v12.3: derive z_vl from z_bank mean (dim-compatible with training)
        z_vl_t = self._z_vl_from_bank(kwargs.get("z_bank")) if self.z_vl_dim > 0 else None
        self.adapt_optimizer.zero_grad()
        out = self.model.encode(img_t, wri_t, st_t, goal_img=goal_t, z_vl=z_vl_t)
        cond = out["cond"]
        B, K, A = target_chunk.shape
        t_rand = torch.rand(B, device=self.device)
        noise = torch.randn_like(target_chunk)
        v_target = target_chunk - noise
        x_t = t_rand.view(-1, 1, 1) * target_chunk + (1 - t_rand.view(-1, 1, 1)) * noise
        v_pred = self.model.velocity(x_t, t_rand, cond)
        flow_loss = F.mse_loss(v_pred, v_target)
        # Gripper BCE
        gripper_target = (target_chunk[..., -1] > 0).float()
        gripper_logits = self.model.gripper_logits(cond)
        gripper_loss = F.binary_cross_entropy_with_logits(gripper_logits, gripper_target)
        loss = flow_loss + gripper_loss
        loss.backward()
        self.adapt_optimizer.step()
        return {"ok": True, "loss": float(loss.detach())}

    def cmd_adapt_demo(self, kwargs):
        """v12 demo replay: sample K frames from suite, flow + gripper loss."""
        if not self.adaptive_enabled or self.DemoReplay is None:
            return {"ok": False, "error": "demo replay not configured"}
        suite = kwargs.get("suite", "libero_10")
        if self._demo_replay_loaded_suite != suite:
            self.demo_replay = self.DemoReplay(suite, self.target_size, self.device)
            self._demo_replay_loaded_suite = suite
        task_lang = kwargs.get("task_language", None)
        if task_lang is not None and hasattr(self.demo_replay, "set_task_filter"):
            self.demo_replay.set_task_filter(task_lang)
        # Sample batch
        img_t, wri_t, st_t, ch_t, _bank, _delta = self.demo_replay.sample_batch(self.demo_k)
        self.adapt_optimizer.zero_grad()
        out = self.model.encode(img_t, wri_t, st_t, goal_img=None)
        cond = out["cond"]
        B, K, A = ch_t.shape
        t_rand = torch.rand(B, device=self.device)
        noise = torch.randn_like(ch_t)
        v_target = ch_t - noise
        x_t = t_rand.view(-1, 1, 1) * ch_t + (1 - t_rand.view(-1, 1, 1)) * noise
        v_pred = self.model.velocity(x_t, t_rand, cond)
        flow_loss = F.mse_loss(v_pred, v_target)
        gripper_target = (ch_t[..., -1] > 0).float()
        gripper_logits = self.model.gripper_logits(cond)
        gripper_loss = F.binary_cross_entropy_with_logits(gripper_logits, gripper_target)
        loss = flow_loss + gripper_loss
        loss.backward()
        self.adapt_optimizer.step()
        return {"ok": True, "loss": float(loss.detach())}

    def cmd_set_retrieval_filter(self, kwargs):
        if self.retrieval_bank is None:
            return {"ok": False, "error": "no retrieval bank"}
        new_suite = kwargs.get("suite")
        rb = self.retrieval_bank
        self.retrieval_bank = RetrievalBank(
            bank_path=self.args.memory_bank, device=self.device,
            top_k=rb.top_k, alpha_base=rb.alpha_base,
            adaptive_alpha=rb.adaptive_alpha, filter_suite=new_suite,
            success_only=self.args.retrieve_success_only,
            softmax_temperature=rb.softmax_temperature,
        )
        return {"ok": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True, type=str)
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--memory_bank", default="", type=str)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--adaptive_lr", type=float, default=1e-4)
    p.add_argument("--demo_replay_n", type=int, default=0)
    p.add_argument("--retrieve_top_k", type=int, default=3)
    p.add_argument("--retrieve_alpha", type=float, default=0.3)  # default per v11 sweep best
    p.add_argument("--retrieve_adaptive", action="store_true")
    p.add_argument("--retrieve_success_only", action="store_true")
    p.add_argument("--retrieve_temp", type=float, default=8.0)
    p.add_argument("--retrieve_filter_suite_default", default="libero_10", type=str)
    args = p.parse_args()

    server = LiquidServerV12(args)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[v12_server] listening on tcp://*:{args.port}")

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
            print("\n[v12_server] interrupt; shutting down.")
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
