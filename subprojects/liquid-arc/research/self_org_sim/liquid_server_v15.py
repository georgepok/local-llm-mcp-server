"""v15 server: Liquid as decoder over GR00T features.

Input contract differs from v12/v13 servers:
  - NO images processed by Liquid (no img_raw, no wrist_raw — Liquid doesn't see images)
  - Required from client per call: z_bank [K_bank, 1024], z_state [1536], state8 [8]
  - Client extracts these from GR00T's response (traj_model_output, z_state)

Commands (pickle wire format):
  init, episode_reset, predict_chunk, encode_retrieval (passthrough), predict_and_retrieve,
  adapt_v8, adapt_demo, set_retrieval_filter

predict_chunk:
  Inputs: state8, z_bank [K, 1024], z_state [1536], n_steps
  Returns: chunk [16, 7]

predict_and_retrieve:
  Same predict_chunk inputs PLUS img_raw, wrist_raw for retrieval feature extraction.
  Returns blended chunk + retrieval diagnostics.
"""
from __future__ import annotations

import argparse
import functools
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zmq

print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_groot_v15 import V15Policy
from liquid_arc_substrate_v15 import make_v15_config
from retrieval import RetrievalBank


class LiquidServerV15:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception:
            pass
        torch.set_float32_matmul_precision("high")

        print(f"[v15_server] loading {args.student_ckpt}...")
        ckpt = torch.load(args.student_ckpt, map_location=self.device, weights_only=False)
        config_dict = ckpt["config"]
        config = make_v15_config()
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        self.config = config
        train_args = ckpt["args"]
        self.K_latent = train_args.get("K_latent", 2)

        self.model = V15Policy(
            config, action_horizon=16, action_dim=7, state_dim=8,
            head_d=256, head_layers=4, head_heads=4,
            z_vl_dim=1024, z_state_dim=1536,
            K_bank=4, K_latent=self.K_latent,
        ).to(self.device)

        sd = ckpt["policy"]
        own = self.model.state_dict()
        loaded = 0
        for k, v in sd.items():
            kk = k.replace("_orig_mod.", "")
            if kk in own and own[kk].shape == v.shape:
                own[kk].copy_(v); loaded += 1
        print(f"[v15_server] loaded {loaded}/{len(own)} tensors (step={ckpt.get('step', '?')})")

        self.adaptive_enabled = args.adaptive
        self.ablate_substrate = args.ablate_substrate
        self.adapt_optimizer = None
        self.adapt_params = []
        self.adapt_snapshot = None
        if self.adaptive_enabled:
            self._setup_adaptive(args.adaptive_lr)

        # Retrieval bank (optional — uses DINOv2 separately for the bank's query feature)
        self.zbank_retrieve_alpha = float(args.zbank_retrieve_alpha)
        self.retrieval_bank = None
        if args.memory_bank:
            self.retrieval_bank = RetrievalBank(
                bank_path=args.memory_bank, device=self.device,
                top_k=args.retrieve_top_k, alpha_base=args.retrieve_alpha,
                adaptive_alpha=args.retrieve_adaptive,
                filter_suite=args.retrieve_filter_suite_default or None,
                success_only=args.retrieve_success_only,
                softmax_temperature=args.retrieve_temp,
                load_z_bank_z_state=(self.zbank_retrieve_alpha > 0.0),
            )

        self.model.eval()
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[v15_server] ready (d={config.d_model}, params={n_total:,}, "
              f"K_latent={self.K_latent}, adaptive={self.adaptive_enabled}, "
              f"ablate={self.ablate_substrate}, retrieval={self.retrieval_bank is not None})")

    def _setup_adaptive(self, lr):
        for p in self.model.parameters():
            p.requires_grad = False
        enc = self.model.encoder
        dyn = enc.dynamics
        params = []; seen = set()
        def _add(p):
            if id(p) not in seen:
                p.requires_grad = True
                params.append(p); seen.add(id(p))
        if hasattr(dyn, "tau_net_linear1"):
            for p in dyn.tau_net_linear1.parameters(): _add(p)
            for p in dyn.tau_net_linear2.parameters(): _add(p)
        if hasattr(dyn, "structural_tau") and dyn.structural_tau is not None:
            _add(dyn.structural_tau)
        if hasattr(dyn, "metric_net_linear1"):
            for p in dyn.metric_net_linear1.parameters(): _add(p)
            for p in dyn.metric_net_linear2_diag.parameters(): _add(p)
        if hasattr(dyn, "rezero_gate_logit"):
            _add(dyn.rezero_gate_logit)
        self.adapt_params = params
        self.adapt_optimizer = torch.optim.SGD(params, lr=lr)
        n_adapt = sum(p.numel() for p in params)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[v15_server.adapt] {n_adapt:,}/{n_total:,} adaptable lr={lr}")
        self.adapt_snapshot = [p.detach().clone() for p in params]

    def _restore_snapshot(self):
        with torch.no_grad():
            for p, s in zip(self.adapt_params, self.adapt_snapshot):
                p.copy_(s)

    def _to_dev(self, arr, dtype=torch.float32):
        if arr is None:
            return None
        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(self.device).to(dtype)

    @torch.no_grad()
    def _v15_sample_chunk(self, z_bank_t, z_state_t, state_t, n_steps):
        out = self.model.encode(z_bank_t, z_state_t, state_t,
                                ablate_substrate=self.ablate_substrate)
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

    # === Commands ===

    def cmd_init(self, kwargs):
        if self.adaptive_enabled:
            self.adapt_snapshot = [p.detach().clone() for p in self.adapt_params]
        return {"ok": True, "info": {
            "d": int(self.config.d_model),
            "version": "v15",
            "action_horizon": 16,
            "img_size": 224,
            "needs_z_bank": True,
            "needs_z_state": True,
        }}

    def cmd_episode_reset(self, kwargs):
        if self.adaptive_enabled and self.adapt_snapshot is not None:
            self._restore_snapshot()
        return {"ok": True}

    @torch.no_grad()
    def cmd_predict_chunk(self, kwargs):
        z_bank = self._to_dev(kwargs.get("z_bank"))   # [K, 1024]
        z_state = self._to_dev(kwargs.get("z_state"))  # [1536]
        state = self._to_dev(kwargs.get("state8"))     # [8]
        if z_bank is None or z_state is None or state is None:
            return {"ok": False, "error": "v15 requires z_bank, z_state, state8"}

        # v15-zretrieve: optionally project live z_bank onto training manifold
        # via DINOv2-feature kNN retrieval. Fixes compound-error distribution
        # shift on z_bank (live GR00T on Liquid-drifted obs → OOD z_bank).
        zret_diag = None
        if (self.zbank_retrieve_alpha > 0.0 and self.retrieval_bank is not None
                and self.retrieval_bank.zbank_memmaps is not None
                and "img_raw" in kwargs and "wrist_raw" in kwargs):
            q = self.retrieval_bank.encode_query(
                kwargs["img_raw"], kwargs["wrist_raw"], kwargs["state8"],
            )
            zb_ret, zs_ret, zret_diag = self.retrieval_bank.retrieve_zbank_zstate(q)
            zb_ret_t = torch.from_numpy(zb_ret).to(self.device).float()
            zs_ret_t = torch.from_numpy(zs_ret).to(self.device).float()
            a = self.zbank_retrieve_alpha
            z_bank = a * zb_ret_t + (1.0 - a) * z_bank
            z_state = a * zs_ret_t + (1.0 - a) * z_state

        # Add batch dim
        z_bank = z_bank.unsqueeze(0)
        z_state = z_state.unsqueeze(0)
        state = state.unsqueeze(0)
        n_steps = int(kwargs.get("n_steps", 10))
        chunk = self._v15_sample_chunk(z_bank, z_state, state, n_steps)
        out = {"ok": True, "chunk": chunk}
        if zret_diag is not None:
            out["zret_mean_sim"] = zret_diag["mean_sim"]
            out["zret_alpha"] = self.zbank_retrieve_alpha
        return out

    @torch.no_grad()
    def cmd_encode_retrieval(self, kwargs):
        if self.retrieval_bank is None:
            return {"ok": False, "error": "no retrieval bank"}
        q = self.retrieval_bank.encode_query(
            kwargs["img_raw"], kwargs["wrist_raw"], kwargs["state8"],
        )
        return {"ok": True, "feat": q.cpu().numpy().astype(np.float32)}

    @torch.no_grad()
    def cmd_predict_and_retrieve(self, kwargs):
        # Predict v15 chunk
        pred_out = self.cmd_predict_chunk(kwargs)
        if not pred_out["ok"]:
            return pred_out
        chunk_np = pred_out["chunk"]
        if self.retrieval_bank is None:
            return {"ok": True, "chunk": chunk_np, "blended": False}
        # Retrieval uses raw images for DINOv2-based query
        if "img_raw" not in kwargs or "wrist_raw" not in kwargs:
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
        if not self.adaptive_enabled:
            return {"ok": False, "error": "adaptive not enabled"}
        if self.ablate_substrate:
            return {"ok": True, "loss": 0.0, "note": "ablate — skipping adapt"}
        z_bank = self._to_dev(kwargs.get("z_bank"))
        z_state = self._to_dev(kwargs.get("z_state"))
        state = self._to_dev(kwargs.get("state8"))
        if z_bank is None or z_state is None or state is None:
            return {"ok": False, "error": "adapt_v8 needs z_bank, z_state, state8"}
        z_bank = z_bank.unsqueeze(0); z_state = z_state.unsqueeze(0); state = state.unsqueeze(0)
        target_chunk = torch.from_numpy(
            np.asarray(kwargs["target_chunk"], dtype=np.float32)
        ).to(self.device).unsqueeze(0)

        self.adapt_optimizer.zero_grad()
        out = self.model.encode(z_bank, z_state, state)
        cond = out["cond"]
        B, K, A = target_chunk.shape
        t_rand = torch.rand(B, device=self.device)
        noise = torch.randn_like(target_chunk)
        v_target = target_chunk - noise
        x_t = t_rand.view(-1, 1, 1) * target_chunk + (1 - t_rand.view(-1, 1, 1)) * noise
        v_pred = self.model.velocity(x_t, t_rand, cond)
        flow_loss = F.mse_loss(v_pred, v_target)
        gripper_target = (target_chunk[..., -1] > 0).float()
        gripper_logits = self.model.gripper_logits(cond)
        gripper_loss = F.binary_cross_entropy_with_logits(gripper_logits, gripper_target)
        loss = flow_loss + gripper_loss
        loss.backward()
        self.adapt_optimizer.step()
        return {"ok": True, "loss": float(loss.detach())}

    def cmd_adapt_demo(self, kwargs):
        # Demo replay would need z_bank+z_state per demo sample → would require
        # custom dataset loading. For v15 we skip demo replay at adapt time.
        return {"ok": True, "loss": 0.0, "note": "v15 demo replay not implemented"}

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
            load_z_bank_z_state=(self.zbank_retrieve_alpha > 0.0),
        )
        return {"ok": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True, type=str)
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--memory_bank", default="", type=str)
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--adaptive_lr", type=float, default=1e-4)
    p.add_argument("--ablate_substrate", action="store_true")
    p.add_argument("--retrieve_top_k", type=int, default=3)
    p.add_argument("--retrieve_alpha", type=float, default=0.3)
    p.add_argument("--retrieve_adaptive", action="store_true")
    p.add_argument("--retrieve_success_only", action="store_true")
    p.add_argument("--retrieve_temp", type=float, default=8.0)
    p.add_argument("--retrieve_filter_suite_default", default="libero_10", type=str)
    p.add_argument("--zbank_retrieve_alpha", type=float, default=0.0,
                   help="v15-zretrieve: blend retrieved training-manifold z_bank/z_state into substrate input. "
                        "0.0=pure live (default), 1.0=pure retrieved. Probe value: 1.0")
    args = p.parse_args()

    server = LiquidServerV15(args)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[v15_server] listening on tcp://*:{args.port}")

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
            resp = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
        sock.send(pickle.dumps(resp))


if __name__ == "__main__":
    main()
