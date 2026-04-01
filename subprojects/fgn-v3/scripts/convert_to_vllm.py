"""Convert trained FGN model to vLLM-compatible HuggingFace format.

Exports:
  - config.json (HuggingFace model config)
  - model.safetensors (weights)
  - modeling_fgn.py (custom model class for --trust-remote-code)
  - tokenizer files (copies from GPT-2)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fgn.config import FGNConfig
from fgn.model import FGNModel


MODELING_FGN_TEMPLATE = '''\
"""FGN model implementation for vLLM --trust-remote-code loading."""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FGNConfig:
    model_type = "fgn"

    def __init__(self, **kwargs):
        self.d_model = kwargs.get("d_model", 256)
        self.n_heads = kwargs.get("n_heads", 8)
        self.n_layers = kwargs.get("n_layers", 6)
        self.d_ff = kwargs.get("d_ff", 1024)
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.max_seq_len = kwargs.get("max_seq_len", 2048)
        self.n_scales = kwargs.get("n_scales", 3)
        self.t_init = tuple(kwargs.get("t_init", [0.1, 1.0, 10.0]))

    @property
    def d_head(self):
        return self.d_model // self.n_heads


class MetricNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.net = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.GELU(),
            nn.Linear(d // 4, d),
        )

    def forward(self, h):
        return F.softplus(self.net(h))


class CurvatureEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kernel_first", torch.tensor([[[1.0, 0.0, -1.0]]]))
        self.register_buffer("kernel_second", torch.tensor([[[1.0, -2.0, 1.0]]]))

    def forward(self, g):
        B, N, d = g.shape
        g_conv = g.permute(0, 2, 1).reshape(B * d, 1, N)
        g_padded = F.pad(g_conv, (1, 1), mode="reflect")
        dg = F.conv1d(g_padded, self.kernel_first)
        d2g = F.conv1d(g_padded, self.kernel_second)
        dg = dg.reshape(B, d, N).permute(0, 2, 1)
        d2g = d2g.reshape(B, d, N).permute(0, 2, 1)
        g_sq = g * g + 1e-6
        term = (g * d2g - 0.5 * dg * dg) / g_sq
        return term.mean(dim=-1)


class HeatKernelAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.n_scales = config.n_scales

        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = nn.Linear(config.d_model, config.d_model, bias=False)
        self.log_t = nn.Parameter(torch.zeros(config.n_scales))
        self.W_scale = nn.Linear(config.d_model, config.n_scales)

    def forward(self, h, g, mask=None):
        B, N, _ = h.shape
        Q = self.W_q(h).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = self.W_k(h).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = self.W_v(h).view(B, N, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        # Shared metric: average across head groups -> [B, 1, N, d_head]
        g_shared = g.view(B, N, self.n_heads, self.d_head).mean(dim=2).unsqueeze(1)

        diff = Q.unsqueeze(3) - K.unsqueeze(2)
        g_avg = (g_shared.unsqueeze(3) + g_shared.unsqueeze(2)) / 2.0
        d_sq = (diff * diff * g_avg).sum(-1)

        t = self.log_t.exp()
        scale_weights = F.softmax(self.W_scale(h), dim=-1)
        attn_out = torch.zeros_like(V)

        for s in range(self.n_scales):
            log_K = -d_sq / (4.0 * t[s])
            if mask is not None:
                log_K = log_K.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            K_s = F.softmax(log_K, dim=-1)
            w_s = scale_weights[:, :, s].unsqueeze(1).unsqueeze(-1)
            attn_out = attn_out + w_s * (K_s @ V)

        return self.W_o(attn_out.permute(0, 2, 1, 3).reshape(B, N, self.d_model))


class FGNTransformerLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm_metric = nn.LayerNorm(config.d_model)
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.norm_ff = nn.LayerNorm(config.d_model)
        self.metric = MetricNetwork(config)
        self.curvature = CurvatureEngine()
        self.attention = HeatKernelAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, h, mask=None):
        g = self.metric(self.norm_metric(h))
        h = h + self.attention(self.norm_attn(h), g, mask=mask)
        h = h + self.ffn(self.norm_ff(h))
        return h


class FGNForCausalLM(nn.Module):
    """vLLM-compatible FGN model."""

    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = FGNConfig(**config)
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.layers = nn.ModuleList([
            FGNTransformerLayer(config, i) for i in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, **kwargs):
        B, N = input_ids.shape
        pos = torch.arange(N, device=input_ids.device).unsqueeze(0)
        h = self.embed(input_ids) + self.pos_embed(pos)
        mask = torch.triu(
            torch.ones(N, N, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            h = layer(h, mask=mask)
        return self.lm_head(self.norm(h))
'''


def convert(checkpoint_path: str, output_dir: str):
    """Convert FGN checkpoint to HuggingFace format."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    # Strip _orig_mod. prefix from torch.compile state dicts
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}

    os.makedirs(output_dir, exist_ok=True)

    # 1. Save config.json
    config_dict = {
        "model_type": "fgn",
        "architectures": ["FGNForCausalLM"],
        "auto_map": {"AutoModelForCausalLM": "modeling_fgn.FGNForCausalLM"},
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "d_ff": config.d_ff,
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "n_scales": config.n_scales,
        "t_init": list(config.t_init),
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"  Saved config.json")

    # 2. Save model weights
    # Remove curvature regularization params (inference-only)
    inference_state = {}
    for k, v in state_dict.items():
        if not k.startswith("curv_reg."):
            inference_state[k] = v

    try:
        from safetensors.torch import save_file
        weights_path = os.path.join(output_dir, "model.safetensors")
        save_file(inference_state, weights_path)
        print(f"  Saved model.safetensors ({len(inference_state)} tensors)")
    except ImportError:
        weights_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(inference_state, weights_path)
        print(f"  Saved pytorch_model.bin ({len(inference_state)} tensors)")

    # 3. Save modeling_fgn.py
    modeling_path = os.path.join(output_dir, "modeling_fgn.py")
    with open(modeling_path, "w") as f:
        f.write(MODELING_FGN_TEMPLATE)
    print(f"  Saved modeling_fgn.py")

    # 4. Copy tokenizer files from GPT-2
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(output_dir)
        print(f"  Saved tokenizer files")
    except ImportError:
        print("  WARNING: transformers not available, skipping tokenizer export")

    print(f"\nExported to {output_dir}")
    print(f"Load with vLLM:")
    print(f"  vllm serve {output_dir} --trust-remote-code")


def main():
    parser = argparse.ArgumentParser(description="Convert FGN to vLLM format")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    convert(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
