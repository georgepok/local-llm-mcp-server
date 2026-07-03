"""v17 inference server.

Loads V17Policy (v10 encoder + flow head + substrate gripper head). At chunk
sampling, runs standard rectified flow for xyz/rpy then overrides chunk's
gripper dim with the substrate head's prediction.

Input contract: img_raw, wrist_raw, state8 — same as v10 server. No GR00T
features needed (encoder is v10's pure-vision pipeline).
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zmq

_LA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_LA_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from distill_groot_v17 import V17Policy  # type: ignore
from retrieval import RetrievalBank  # type: ignore

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

torch.set_float32_matmul_precision("high")


class V17Server:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[v17_server] device={self.device}")
        print(f"[v17_server] loading {args.student_ckpt}...")
        ckpt = torch.load(args.student_ckpt, map_location=self.device, weights_only=False)
        sa = ckpt["args"]
        self.target_size = sa["target_img_size"]
        d = sa["d"]

        encoder_kwargs = dict(
            state_dim=8, d=d, d_vis=d, img_size=self.target_size,
            k_max=16, halt_mode="learned", min_steps=4, dt=0.5,
            n_tasks=0, d_task=32, z_groot_dim=0,
        )
        head_kwargs = dict(d_model=256, n_layers=4, n_heads=4, d_t=64)
        self.model = V17Policy(
            encoder_kwargs=encoder_kwargs, head_kwargs=head_kwargs,
            gripper_d=sa["gripper_d"], gripper_K=sa["gripper_K"],
            action_horizon=16, action_dim=7, d=d,
        ).to(self.device)

        sd = ckpt["model"]
        own = self.model.state_dict()
        loaded = 0
        for k, v in sd.items():
            kk = k.replace("_orig_mod.", "")
            if kk in own and own[kk].shape == v.shape:
                own[kk].copy_(v); loaded += 1
        print(f"[v17_server] loaded {loaded}/{len(own)} tensors (step={ckpt.get('step', '?')})")

        # Retrieval bank (optional, mirrors v10 server for compat with eval wrappers)
        self.retrieval_bank = None
        if args.memory_bank:
            self.retrieval_bank = RetrievalBank(
                bank_path=args.memory_bank, device=self.device,
                top_k=args.retrieve_top_k, alpha_base=args.retrieve_alpha,
                adaptive_alpha=args.retrieve_adaptive,
                filter_suite=args.retrieve_filter_suite_default or None,
                success_only=args.retrieve_success_only,
                softmax_temperature=args.retrieve_temp,
            )

        self.model.eval()
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[v17_server] ready (d={d}, params={n_total:,}, "
              f"img={self.target_size}, retrieval={self.retrieval_bank is not None})")

        # Image preprocessing tensors
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(self.device)

    def _img_to_tensor(self, img_uint8):
        x = torch.from_numpy(img_uint8).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        return x

    @torch.no_grad()
    def _sample(self, img_raw, wrist_raw, state8, n_steps=10):
        img_t = self._img_to_tensor(img_raw)
        wri_t = self._img_to_tensor(wrist_raw)
        st_t = torch.from_numpy(np.asarray(state8, dtype=np.float32)).to(self.device).unsqueeze(0)

        cond, _ = self.model.encode(img_t, wri_t, st_t)

        # Flow integration for xyz/rpy (all 7 dims; gripper dim is overridden after)
        B = cond.shape[0]
        K = self.model.action_horizon
        A = self.model.action_dim
        x = torch.randn(B, K, A, device=self.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = torch.full((B,), i * dt, device=self.device)
            v = self.model.velocity(x, t_val, cond)
            x = x + dt * v
        chunk = x[0].cpu().numpy().astype(np.float32)  # [K, A]

        # Override gripper from substrate head
        sub_out = self.model.gripper_logits(cond)
        grip_logits = sub_out["logits"][0].cpu().numpy()  # [K]
        # logit > 0 → P(open) > 0.5 → grip = -1 (open). logit < 0 → grip = +1 (close).
        grip_signs = np.where(grip_logits > 0, -1.0, +1.0).astype(np.float32)
        chunk[:, -1] = grip_signs
        return chunk

    def cmd_init(self, kwargs):
        return {"ok": True, "info": {
            "d": int(768), "version": "v17",
            "action_horizon": 16, "img_size": int(self.target_size),
        }}

    def cmd_episode_reset(self, kwargs):
        return {"ok": True}

    @torch.no_grad()
    def cmd_predict_chunk(self, kwargs):
        chunk = self._sample(kwargs["img_raw"], kwargs["wrist_raw"], kwargs["state8"],
                             n_steps=int(kwargs.get("n_steps", 10)))
        return {"ok": True, "chunk": chunk}

    @torch.no_grad()
    def cmd_predict_and_retrieve(self, kwargs):
        out = self.cmd_predict_chunk(kwargs)
        if not out["ok"]:
            return out
        chunk_np = out["chunk"]
        if self.retrieval_bank is None:
            return {"ok": True, "chunk": chunk_np, "blended": False}
        if "img_raw" not in kwargs:
            return {"ok": True, "chunk": chunk_np, "blended": False}
        final, diag = self.retrieval_bank.query_and_blend(
            kwargs["img_raw"], kwargs["wrist_raw"], kwargs["state8"], chunk_np,
        )
        return {"ok": True, "chunk": final, "blended": True,
                "retrieval_mean_sim": diag["mean_sim"],
                "retrieval_alpha": diag["alpha_used"]}

    def cmd_adapt_v8(self, kwargs):
        return {"ok": True, "loss": 0.0, "note": "v17 adapt not implemented"}

    def cmd_adapt_demo(self, kwargs):
        return {"ok": True, "loss": 0.0, "note": "v17 demo replay not implemented"}

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
    p.add_argument("--retrieve_top_k", type=int, default=3)
    p.add_argument("--retrieve_alpha", type=float, default=0.0)
    p.add_argument("--retrieve_adaptive", action="store_true")
    p.add_argument("--retrieve_success_only", action="store_true")
    p.add_argument("--retrieve_temp", type=float, default=8.0)
    p.add_argument("--retrieve_filter_suite_default", default="libero_object", type=str)
    args = p.parse_args()

    server = V17Server(args)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[v17_server] listening on tcp://*:{args.port}")

    cmd_table = {
        "init": server.cmd_init,
        "episode_reset": server.cmd_episode_reset,
        "predict_chunk": server.cmd_predict_chunk,
        "predict_and_retrieve": server.cmd_predict_and_retrieve,
        "adapt_v8": server.cmd_adapt_v8,
        "adapt_demo": server.cmd_adapt_demo,
        "set_retrieval_filter": server.cmd_set_retrieval_filter,
    }

    while True:
        try:
            msg = pickle.loads(sock.recv())
        except Exception as e:
            print(f"[v17_server] recv error: {e}"); continue
        cmd = msg.get("cmd")
        kwargs = {k: v for k, v in msg.items() if k != "cmd"}
        t0 = time.perf_counter()
        if cmd in cmd_table:
            try:
                resp = cmd_table[cmd](kwargs)
            except Exception as e:
                import traceback
                resp = {"ok": False, "error": f"{e}\n{traceback.format_exc()}"}
        else:
            resp = {"ok": False, "error": f"unknown cmd: {cmd}"}
        resp["server_ms"] = (time.perf_counter() - t0) * 1000
        sock.send(pickle.dumps(resp))


if __name__ == "__main__":
    main()
