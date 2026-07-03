"""Verify UnfoldLinearPatchEmbed is numerically equivalent to Conv2d with
stride==kernel (non-overlapping patches)."""

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from patch_vit_stem import (                         # noqa: E402
    UnfoldLinearPatchEmbed, replace_vit_patch_embeddings,
)


def test_unfold_equals_conv():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 192, kernel_size=14, stride=14, bias=True)
    mod = UnfoldLinearPatchEmbed.from_conv(conv)
    x = torch.randn(4, 3, 224, 224)
    y_conv = conv(x)
    y_mod = mod(x)
    assert y_conv.shape == y_mod.shape == (4, 192, 16, 16), (y_conv.shape, y_mod.shape)
    max_err = (y_conv - y_mod).abs().max().item()
    assert max_err < 1e-4, f"max err = {max_err}"
    print(f"conv vs unfold+linear max err = {max_err:.2e}")


def test_replace_walks_tree():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, 16, kernel_size=4, stride=4)
            self.extra = nn.Conv2d(16, 16, kernel_size=3, padding=1)  # NOT a patch stem

    m = Toy()
    n = replace_vit_patch_embeddings(m)
    assert n == 1, n
    assert isinstance(m.stem, UnfoldLinearPatchEmbed)
    assert isinstance(m.extra, nn.Conv2d)  # untouched
    print(f"replace_vit_patch_embeddings swapped {n} conv (expected 1)")


if __name__ == "__main__":
    test_unfold_equals_conv()
    test_replace_walks_tree()
    print("patch_vit_stem tests passed")
