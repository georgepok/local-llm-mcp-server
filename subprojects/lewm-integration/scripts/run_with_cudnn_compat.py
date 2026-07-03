"""cuDNN compatibility shim for NGC vLLM 26.01 + GB10/sm_121a.

The container's cuDNN build has incomplete algorithm coverage for specific
backward kernels. After narrowing, the failing ops were:
  - BatchNorm1d backward at large batch (bs*T = 512)
  - Conv2d backward at large batch

Both are ELIMINATED upstream of cuDNN in our train scripts:
  - patch_vit_stem.py  → replaces ViT Conv2d stem with Unfold+Linear
  - train scripts      → replace BatchNorm1d in MLPs with LayerNorm

So cuDNN is left ENABLED here (for attention, layernorm, general GEMM),
but we explicitly force Flash + mem-efficient SDPA backends on, and disable
cuDNN's SDPA backend which also has backward gaps at this batch size.

Usage:
    python scripts/run_with_cudnn_compat.py <script.py> [args...]
"""
import os
import runpy
import sys

import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)
if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(True)
if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
    torch.backends.cuda.enable_mem_efficient_sdp(True)
if hasattr(torch.backends.cuda, "enable_math_sdp"):
    torch.backends.cuda.enable_math_sdp(True)

if len(sys.argv) < 2:
    raise SystemExit("usage: run_with_cudnn_compat.py <script.py> [args...]")

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)) or ".")
runpy.run_path(script, run_name="__main__")
