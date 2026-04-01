# Phase 1 — Addendum: Context for Task 4

**From:** Claude Desktop  
**Date:** 2026-03-10  
**Re:** The FluidGeometryLogitsProcessor and reasoning parser

---

## Important Context

The `fluid_geometry.py` and `nano_v3_reasoning_parser.py` files mounted in the vllm container are **George's own prior work** from the Fluid Geometry Networks (FGN) research project. They are NOT standard NVIDIA components.

### What They Implement

The FluidGeometryLogitsProcessor is an inference-time implementation of FGN concepts from the LiquidARC research paper (available in the project knowledge). Key ideas:

1. **Heat kernel diffusion on a learned Riemannian metric**: Information routing is determined by a metric field over token positions. The metric defines distances on a curved manifold; a heat kernel computed from these distances determines information flow. Positions that the metric places "close together" exchange information readily.

2. **SDPA-factored heat kernel**: The Riemannian heat kernel `K = softmax(-D²/(4t))` is algebraically reformulated as scaled dot-product attention, enabling FlashAttention acceleration.

3. **WHERE/WHAT decomposition**: Geometric routing (WHERE information flows) is separated from content transformation (WHAT operation is applied).

4. **The GeometricEngine log** (`think_start=12, think_end=13, warmup=15, tau=15.0`) indicates the processor is active during "thinking" token generation, modulating logits based on geometric principles.

### Why This Matters for the Neuroplastic Project

This existing infrastructure represents a **behavioral modification layer** that already operates at inference time without touching model weights. It modifies HOW the model generates tokens by reshaping the logit distribution based on geometric computations.

For the neuroplastic self-modification project, this is potentially the first modification pathway — the model could learn to adjust its own FluidGeometryLogitsProcessor parameters (tau, warmup, think boundaries, and potentially the metric computation itself) as a form of behavioral self-modification that doesn't require weight changes or vllm restarts.

### What to Do With This Information

When reading `fluid_geometry.py` for Task 4:
- Document the actual parameters and how they're used
- Identify which parameters could be modified at runtime without restart
- Assess whether the logits processor could be extended to accept self-modification commands from the model itself (e.g., the model outputs special tokens that adjust its own geometric parameters)
- Look for any existing hooks that allow dynamic parameter adjustment

This is the lowest-friction path to Phase 2 self-modification experiments.

---

*End of addendum.*
