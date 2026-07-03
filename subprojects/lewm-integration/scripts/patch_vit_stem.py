"""Replace HuggingFace ViT's Conv2d patch embedding with an equivalent
Unfold + Linear implementation.

Why: on NGC vLLM 26.01 + GB10 (sm_121a), cuDNN's algorithm registry cannot
find a backward kernel for Conv2d(3, hidden, kernel=patch, stride=patch) at
batch sizes >~32. The failure cascades into "GET was unable to find an
engine to execute this computation".

Unfold+Linear is the MATHEMATICAL IDENTITY of Conv2d with stride=kernel, so
numerics are unchanged. It runs via cuBLAS GEMM kernels which are fully
supported on sm_121a, keeping cuDNN enabled for all other ops in the model.

Usage:
    from patch_vit_stem import replace_vit_patch_embeddings
    replace_vit_patch_embeddings(encoder)  # in-place

Call this BEFORE any forward pass. Weights from the original Conv2d are
preserved (same param tensor, reshaped view).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class UnfoldLinearPatchEmbed(nn.Module):
    """Drop-in for HF ViT's internal Conv2d patch embedding.

    Output shape matches the Conv2d: [B, hidden, H/patch, W/patch].
    The caller (`ViTPatchEmbeddings.forward`) then flattens + transposes.
    """

    def __init__(self, num_channels: int, hidden_size: int,
                 patch_size: Tuple[int, int] | int, bias: bool = True):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        # Back the parameters in an nn.Linear so device moves (.to / .cuda)
        # work reliably through the standard nn.Module path.
        self.proj = nn.Linear(
            num_channels * patch_size[0] * patch_size[1],
            hidden_size, bias=bias)

    # HF ViT's forward reads `.weight.dtype` on the patch projection to decide
    # autocast dtype — expose the underlying nn.Linear params at the top level.
    @property
    def weight(self) -> torch.Tensor:
        return self.proj.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.proj.bias

    @classmethod
    def from_conv(cls, conv: nn.Conv2d) -> "UnfoldLinearPatchEmbed":
        assert conv.stride == conv.kernel_size, \
            "patch-stem conv must have stride == kernel_size"
        out_ch, in_ch, kH, kW = conv.weight.shape
        mod = cls(num_channels=in_ch, hidden_size=out_ch,
                  patch_size=(kH, kW), bias=conv.bias is not None)
        with torch.no_grad():
            mod.proj.weight.copy_(conv.weight.view(out_ch, in_ch * kH * kW))
            if conv.bias is not None:
                assert mod.proj.bias is not None
                mod.proj.bias.copy_(conv.bias)
        return mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        pH, pW = self.patch_size
        assert C == self.num_channels
        assert H % pH == 0 and W % pW == 0
        # Unfold into non-overlapping patches → [B, C*pH*pW, nH*nW]
        patches = nn.functional.unfold(x, kernel_size=(pH, pW), stride=(pH, pW))
        # Linear takes [..., in_features]; patches is [B, K, N] → transpose → [B, N, K]
        out = self.proj(patches.transpose(1, 2))       # [B, N, hidden]
        nH, nW = H // pH, W // pW
        return out.transpose(1, 2).contiguous().view(B, self.hidden_size, nH, nW)


class LinearConv1d(nn.Module):
    """Drop-in for Conv1d(in, out, kernel_size=1) — routes through Linear
    to avoid cuDNN Conv1d backward failures on GB10/sm_121a."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    @classmethod
    def from_conv1d(cls, conv: nn.Conv1d) -> "LinearConv1d":
        assert conv.kernel_size == (1,) and conv.stride == (1,)
        mod = cls(conv.in_channels, conv.out_channels, bias=conv.bias is not None)
        with torch.no_grad():
            mod.linear.weight.copy_(conv.weight.squeeze(-1))
            if conv.bias is not None:
                mod.linear.bias.copy_(conv.bias)
        return mod

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d: [B, C_in, T] → [B, C_out, T]
        return self.linear(x.transpose(1, 2)).transpose(1, 2)


def replace_vit_patch_embeddings(module: nn.Module) -> int:
    """Walk `module` and replace:
      - Conv2d whose stride==kernel → UnfoldLinearPatchEmbed
      - Conv1d with kernel_size=1 → LinearConv1d
    Returns the number of replacements.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and child.stride == child.kernel_size:
            setattr(module, name, UnfoldLinearPatchEmbed.from_conv(child))
            n += 1
        elif isinstance(child, nn.Conv1d) and child.kernel_size == (1,) and child.stride == (1,):
            setattr(module, name, LinearConv1d.from_conv1d(child))
            n += 1
        else:
            n += replace_vit_patch_embeddings(child)
    return n
