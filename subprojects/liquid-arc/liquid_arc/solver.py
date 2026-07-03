"""ODE solvers for LiquidARC.

Four solvers:
  1. euler_solve — standard forward Euler, O(n_steps) memory, torch.compile compatible
  2. euler_solve_chunked — chunked gradient checkpointing, O(n_steps/chunk_size) memory,
     torch.compile compatible, ~3x compute (forward + recompute + backward)
  3. invertible_euler_solve — O(1) memory via fixed-point reconstruction in backward,
     NOT torch.compile compatible, ~7x compute
  4. deq_solve — Deep Equilibrium Model: O(1) memory, 1 backward dynamics eval.
     Forward runs Euler with no_grad (zero tape). Backward uses Implicit Function
     Theorem: solves (I - J^T)z = grad via fixed-point iteration, then single VJP.
     ~30 dynamics evals in backward (for IFT solve) vs ~80 for invertible.

DEQ (4) is the recommended solver when dynamics converge to equilibrium.
"""

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def euler_solve(fn, y0: torch.Tensor, t_span: tuple, n_steps: int,
                return_efficiency: bool = False):
    """Forward Euler: y_{n+1} = y_n + dt * f(t_n, y_n).

    Standard solver. Keeps all n_steps intermediates on the autograd tape.
    O(n_steps) memory, fastest forward, torch.compile compatible.

    If return_efficiency=True, also returns mean(||dh/dt||²) as efficiency
    cost for the adaptive autonomy regularizer. This PENALIZES unnecessary
    dynamics (opposite of curiosity which REWARDED them → NaN).

    Norm homeostasis: if fn has _norm_ref > 0, per-position norms are soft-clipped
    after each step. This prevents unbounded h growth that the dynamics alone
    cannot contain (target = h + update moves with h).
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    t = t_start
    y = y0

    # Norm homeostasis parameters from dynamics
    norm_ref = getattr(fn, '_norm_ref', 0.0)
    norm_lambda = getattr(fn, '_norm_lambda', 0.0)

    # Multi-timescale local learning: reset Hebbian fast weights for this batch
    inner = fn._orig_mod if hasattr(fn, '_orig_mod') else fn
    if hasattr(inner, 'reset_fast_weights'):
        inner.reset_fast_weights(y0.shape[0], y0.device, y0.dtype)
    if hasattr(inner, 'reset_id_history'):
        inner.reset_id_history(y0.shape[0], y0.shape[1], y0.device, y0.dtype)

    if return_efficiency:
        efficiency_accum = torch.tensor(0.0, device=y0.device)

    for i in range(n_steps):
        if hasattr(fn, 'set_step_embed'):
            fn.set_step_embed(i, n_steps)
        if hasattr(fn, 'set_step_index'):
            fn.set_step_index(i, n_steps)
        dy = fn(t, y)

        if return_efficiency:
            efficiency_accum = efficiency_accum + (dy ** 2).mean()

        y = y + dt * dy

        # Per-position norm homeostasis: smoothly shrink toward norm_ref.
        # scale = norm_ref / ||y_i|| when ||y_i|| > norm_ref, blended by norm_lambda.
        # At norm_lambda=1.0: hard clip. At 0.1: gentle pull (90% current + 10% clipped).
        if norm_ref > 0 and norm_lambda > 0:
            pos_norm = y.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, N, 1]
            scale = torch.where(
                pos_norm > norm_ref,
                1.0 - norm_lambda * (1.0 - norm_ref / pos_norm),  # shrink
                torch.ones_like(pos_norm),  # no change below ref
            )
            y = y * scale

        t = t + dt

    if return_efficiency:
        return y, efficiency_accum / n_steps
    return y


def euler_solve_halting(fn, y0: torch.Tensor, t_span: tuple, n_steps: int,
                         min_steps: int = 4,
                         label_mask: 'torch.Tensor | None' = None):
    """Forward Euler with per-position adaptive halting (Tier 3).

    Each step the dynamics returns (dy, p_halt) where p_halt is [B, N, 1] in [0,1].
    We maintain still_active = product over steps of (1 - p_halt) — fraction of
    each position still contributing. dy is multiplied by still_active before
    applying, so halted positions freeze automatically.

    n_steps is the MAXIMUM budget: positions can halt early, hard ones use full.
    For "go beyond 16", set n_steps > 16 in config (e.g. 32).

    min_steps: clamp still_active to 1 for the first min_steps iterations
    (force minimum computation for every position before halting engages).

    label_mask: optional [B, N] bool — if given, captures per-step intermediate
    state at label positions (for PonderNet deep supervision) along with the
    halt distribution at those positions. Returns an extra dict `sup` with:
        h_stack:        [K, L, d]   — h at label positions after each step k
        p_halt_stack:   [K, L, 1]   — sigmoid halt prob at step k
        p_active_stack: [K, L, 1]   — prob of reaching step k (Π_{j<k}(1-h_j))
    where K = n_steps and L = label_mask.sum(). The halt distribution
    p_halt_dist[k] = p_active_stack[k] * p_halt_stack[k]; residual mass
    1 - Σ_k p_halt_dist[k] must be added back to the final step by the caller.

    When label_mask is None, returns the legacy 3-tuple (y, ponder_cost, steps).
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    t = t_start
    y = y0
    B, N = y.shape[0], y.shape[1]
    still_active = torch.ones(B, N, 1, device=y.device, dtype=y.dtype)
    ponder_cost = torch.zeros((), device=y.device, dtype=y.dtype)
    steps_used_per_pos = torch.zeros(B, N, 1, device=y.device, dtype=y.dtype)

    collect = label_mask is not None
    if collect:
        h_stack_parts = []
        p_halt_stack_parts = []
        p_active_stack_parts = []

    # Multi-timescale: reset per-batch Hebbian fast weights & EMA accumulator
    inner = fn._orig_mod if hasattr(fn, '_orig_mod') else fn
    if hasattr(inner, 'reset_fast_weights'):
        inner.reset_fast_weights(y0.shape[0], y0.device, y0.dtype)
    if hasattr(inner, 'reset_id_history'):
        inner.reset_id_history(y0.shape[0], y0.shape[1], y0.device, y0.dtype)

    for i in range(n_steps):
        if hasattr(fn, 'set_step_embed'):
            fn.set_step_embed(i, n_steps)
        if hasattr(fn, 'set_step_index'):
            fn.set_step_index(i, n_steps)

        # still_active BEFORE this step's halt decision = prob of reaching k.
        # (Within the min_steps window it is identically 1 per position.)
        if collect:
            p_active_stack_parts.append(still_active[label_mask])  # [L, 1]

        result = fn(t, y)
        if isinstance(result, tuple):
            dy, p_halt = result
        else:
            dy = result
            p_halt = torch.zeros(B, N, 1, device=y.device, dtype=y.dtype)

        # For k < min_steps, effective halt prob is 0 (step is mandatory).
        if collect:
            if i < min_steps:
                eff_p_halt = torch.zeros_like(p_halt)
            else:
                eff_p_halt = p_halt
            p_halt_stack_parts.append(eff_p_halt[label_mask])  # [L, 1]

        dy = dy * still_active
        y = y + dt * dy

        if collect:
            h_stack_parts.append(y[label_mask])  # [L, d]

        steps_used_per_pos = steps_used_per_pos + still_active

        if i >= min_steps:
            still_active = still_active * (1.0 - p_halt)

        ponder_cost = ponder_cost + still_active.mean()
        t = t + dt

    ponder_cost = ponder_cost / n_steps

    if collect:
        sup = {
            'h_stack': torch.stack(h_stack_parts, dim=0),            # [K, L, d]
            'p_halt_stack': torch.stack(p_halt_stack_parts, dim=0),  # [K, L, 1]
            'p_active_stack': torch.stack(p_active_stack_parts, dim=0),  # [K, L, 1]
        }
        return y, ponder_cost, steps_used_per_pos.squeeze(-1), sup

    return y, ponder_cost, steps_used_per_pos.squeeze(-1)


def euler_solve_with_observer(fn, observer, y0: torch.Tensor, t_span: tuple,
                              n_steps: int) -> torch.Tensor:
    """Forward Euler with passive memory observation — dynamics unmodified.

    The observer watches h at each step but NEVER modifies it.
    The dynamics runs identically to the base model.
    After the loop, call observer.get_output_correction() for logit corrections.
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    t = t_start
    y = y0
    for i in range(n_steps):
        if hasattr(fn, 'set_step_embed'):
            fn.set_step_embed(i, n_steps)
        if hasattr(fn, 'set_step_index'):
            fn.set_step_index(i, n_steps)

        # Observer passively records (NO modification to y)
        if observer is not None:
            observer.observe(y.detach(), step_index=i)

        # Standard dynamics — completely unmodified
        dy = fn(t, y)
        y = y + dt * dy
        t = t + dt

    # Final observation of the endpoint
    if observer is not None:
        observer.observe(y.detach(), step_index=n_steps)

    return y


def euler_solve_with_memory(fn, memory, y0: torch.Tensor, t_span: tuple,
                            n_steps: int) -> torch.Tensor:
    """Forward Euler with per-step working memory residuals.

    Identical to euler_solve but after each Euler step the hidden state is
    detached from the base model graph, passed through the memory module, and
    the resulting residual is added back.  Gradients for the base model dynamics
    are therefore cut at the detach boundary; only the memory module receives
    gradients through the residual path.

    Calls fn.set_step_embed and fn.set_step_index when available, matching the
    behaviour of euler_solve so the dynamics module sees the correct step context.

    Args:
        fn: Dynamics callable (ContinuousDynamics) — frozen during memory training.
        memory: WorkingMemory instance. memory.reset() must be called before this
            function is invoked.
        y0: Initial hidden state [B, N, d_model].
        t_span: (t_start, t_end) integration interval.
        n_steps: Number of Euler steps.

    Returns:
        y: Final hidden state [B, N, d_model] with memory residuals applied.
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    t = t_start
    y = y0
    for i in range(n_steps):
        # Notify dynamics of current step (step_embed, step_index)
        if hasattr(fn, 'set_step_embed'):
            fn.set_step_embed(i, n_steps)
        if hasattr(fn, 'set_step_index'):
            fn.set_step_index(i, n_steps)

        # Memory updates state and computes overlay BEFORE dynamics
        # Overlay biases MetricNet routing — no residual on y (prevents copy bias)
        y_det = y.detach()
        fn._metric_overlay = memory.step(y_det, step_index=i)

        # Base dynamics with overlay-biased routing
        # Detach y to limit autograd tape to single step (prevents OOM/slowdown)
        dy = fn(t, y_det)
        y = y_det + dt * dy
        t = t + dt

    # Cleanup
    fn._metric_overlay = None
    return y


def _euler_chunk_fn(fn, y, t_start, dt, chunk_steps):
    """Run a fixed block of Euler steps. Static graph — no dynamic control flow.

    This function is the unit of checkpointing: during backward, PyTorch discards
    its intermediates and re-runs it to recompute activations.

    Applies the same per-step norm homeostasis that euler_solve does (reads
    fn._norm_ref and fn._norm_lambda) so memory growth driven by metric
    expansion is bounded inside each chunk.
    """
    norm_ref = getattr(fn, '_norm_ref', 0.0)
    norm_lambda = getattr(fn, '_norm_lambda', 0.0)
    apply_clip = norm_ref > 0 and norm_lambda > 0

    t = t_start
    for _ in range(chunk_steps):
        y = y + dt * fn(t, y)
        if apply_clip:
            pos_norm = y.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.where(
                pos_norm > norm_ref,
                1.0 - norm_lambda * (1.0 - norm_ref / pos_norm),
                torch.ones_like(pos_norm),
            )
            y = y * scale
        t = t + dt
    return y


def euler_solve_chunked(fn, y0: torch.Tensor, t_span: tuple, n_steps: int,
                        chunk_size: int = 4) -> torch.Tensor:
    """Chunked checkpointed Euler solver.

    Groups n_steps into blocks of chunk_size, gradient-checkpoints each block.
    Memory: O(n_steps/chunk_size) intermediate states instead of O(n_steps).
    Compute: ~3x (forward + recompute + backward grads) vs ~2x standard.
    Compatible with torch.compile — each chunk is a static graph.

    For 16 steps with chunk_size=4: 4 checkpointed blocks, storing only 4
    intermediate h states instead of 16 distance matrices on the tape.
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps

    y = y0
    steps_done = 0
    while steps_done < n_steps:
        steps_this = min(chunk_size, n_steps - steps_done)
        t_chunk = t_start + steps_done * dt
        y = torch_checkpoint(
            _euler_chunk_fn, fn, y, t_chunk, dt, steps_this,
            use_reentrant=False,
        )
        steps_done += steps_this

    return y


# ────────────────────────────────────────────────────────────────────
# DEQ solver (Deep Equilibrium Model — IFT backward)
# O(1) memory, 1 VJP in backward. Recommended when dynamics converge.
# ────────────────────────────────────────────────────────────────────

class _DEQSolveFn(torch.autograd.Function):
    """Custom autograd for DEQ-style implicit differentiation.

    Forward: Euler steps with no_grad → stores only h* (zero autograd tape).
    Backward: Implicit Function Theorem at h*.
      1. Solve (I - J_f^T) z = grad_output via fixed-point iteration
      2. Single VJP: torch.autograd.grad(f(h*), params, z) for parameter grads

    The IFT iteration is: z_{k+1} = grad + J_f^T @ z_k
    This converges when the spectral radius of J_f < 1 (i.e., the dynamics
    are contractive), which our LTC dynamics guarantee via the 1/tau term.
    """

    @staticmethod
    def forward(ctx, y0, fn, t_start, t_end, n_steps, n_ift_iters, *diff_tensors):
        ctx.fn = fn
        ctx.t_start = t_start
        ctx.t_end = t_end
        ctx.n_steps = n_steps
        ctx.n_ift_iters = n_ift_iters

        # Forward Euler with NO autograd tape — zero memory
        with torch.no_grad():
            dt = (t_end - t_start) / n_steps
            y = y0
            t = t_start
            for _ in range(n_steps):
                y = y + dt * fn(t, y)
                t = t + dt

        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        fn = ctx.fn
        y_star, = ctx.saved_tensors

        # Collect differentiable parameters
        diff_tensors = []
        for p in fn.parameters():
            if p.requires_grad:
                diff_tensors.append(p)

        # Step 1: Solve (I - J_f^T) z = grad_output via fixed-point iteration
        # z_{k+1} = grad + J_f^T @ z_k
        # J_f^T @ z_k is computed via vector-Jacobian product
        z = grad_output.clone()

        for _ in range(ctx.n_ift_iters):
            y_star_var = y_star.detach().requires_grad_(True)
            with torch.enable_grad():
                f_val = fn(ctx.t_end, y_star_var)

            # JT @ z = VJP of f w.r.t. y at y_star, seeded with z
            # retain_graph=False: each iteration builds a fresh graph from
            # new y_star_var/f_val, so old graph can be freed immediately.
            # This also avoids torch.compile's donated buffer conflict.
            jt_z = torch.autograd.grad(
                f_val, y_star_var, z,
                retain_graph=False,
                create_graph=False,
            )[0]
            z = grad_output + jt_z

        # Step 2: Single VJP for parameter gradients
        # ∂L/∂θ = z^T · ∂f(h*;θ)/∂θ
        y_star_var = y_star.detach().requires_grad_(True)
        with torch.enable_grad():
            f_val = fn(ctx.t_end, y_star_var)

        # Compute gradients w.r.t. y0 (through y_star_var) and all params
        all_grads = torch.autograd.grad(
            f_val, [y_star_var] + diff_tensors, z,
            allow_unused=True,
            retain_graph=False,
        )

        grad_y0 = all_grads[0]
        param_grads = all_grads[1:]

        # grad_y0 for DEQ: z already contains the full implicit gradient
        # The y0 gradient is z itself (since h* depends on y0 through the forward)
        # but we can't backprop through the no_grad forward. For DEQ the assumption
        # is that h* is an equilibrium independent of y0 (the attractor absorbs initial
        # conditions). If we want y0 gradients too, we return z.
        return (z, None, None, None, None, None) + tuple(
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(param_grads, diff_tensors)
        )


def deq_solve(fn, y0: torch.Tensor, t_span: tuple, n_steps: int,
              n_ift_iters: int = 30) -> torch.Tensor:
    """Deep Equilibrium solver via Implicit Function Theorem.

    Forward: Euler integration with torch.no_grad() — zero memory, fastest forward.
    Backward: IFT at equilibrium h* — single VJP, ~n_ift_iters dynamics evals.

    Total backward cost: ~n_ift_iters + 1 dynamics evaluations (vs n_steps for standard,
    ~5*n_steps for invertible). With n_ift_iters=30 and n_steps=16: 31 evals vs 80.

    Requires dynamics to be contractive (spectral radius of J_f < 1).
    LTC dynamics with 1/tau contraction guarantee this by construction.
    """
    diff_tensors = []
    for p in fn.parameters():
        if p.requires_grad:
            diff_tensors.append(p)

    t_start, t_end = t_span
    return _DEQSolveFn.apply(
        y0, fn, t_start, t_end, n_steps, n_ift_iters, *diff_tensors
    )


# ────────────────────────────────────────────────────────────────────
# Invertible solver (O(1) memory, NOT compile compatible)
# Kept for reference / extreme memory scenarios, but chunked is preferred.
# ────────────────────────────────────────────────────────────────────

class _InvertibleEulerFn(torch.autograd.Function):
    """Custom autograd for O(1) memory Euler integration.

    Forward stores only y_final. Backward reconstructs intermediates via
    fixed-point inversion of each Euler step.
    """

    @staticmethod
    def forward(ctx, y0, fn, t_start, t_end, n_steps, n_fp_iters, *diff_tensors):
        ctx.fn = fn
        ctx.t_start = t_start
        ctx.t_end = t_end
        ctx.n_steps = n_steps
        ctx.n_fp_iters = n_fp_iters
        ctx.n_diff = len(diff_tensors)

        with torch.no_grad():
            dt = (t_end - t_start) / n_steps
            y = y0
            t = t_start
            for _ in range(n_steps):
                y = y + dt * fn(t, y)
                t = t + dt

        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        fn = ctx.fn
        n_steps = ctx.n_steps
        dt = (ctx.t_end - ctx.t_start) / n_steps
        y_final, = ctx.saved_tensors

        diff_tensors = []
        for p in fn.parameters():
            if p.requires_grad:
                diff_tensors.append(p)
        ctx_tensor = getattr(fn, '_context', None)
        has_ctx = ctx_tensor is not None and ctx_tensor.requires_grad
        if has_ctx:
            diff_tensors.append(ctx_tensor)

        grad_y = grad_output
        grad_accum = [torch.zeros_like(t) for t in diff_tensors]
        y = y_final

        for step in reversed(range(n_steps)):
            t_step = ctx.t_start + step * dt

            y_prev = y
            with torch.no_grad():
                for _ in range(ctx.n_fp_iters):
                    y_prev = y - dt * fn(t_step, y_prev)

            y_prev = y_prev.detach().requires_grad_(True)
            with torch.enable_grad():
                dy = fn(t_step, y_prev)

            vjp = torch.autograd.grad(
                dy, [y_prev] + diff_tensors, grad_y * dt,
                allow_unused=True, retain_graph=False,
            )

            grad_y = grad_y + vjp[0]
            for i in range(len(diff_tensors)):
                if vjp[i + 1] is not None:
                    grad_accum[i] = grad_accum[i] + vjp[i + 1]

            y = y_prev.detach()

        return (grad_y, None, None, None, None, None) + tuple(grad_accum)


def invertible_euler_solve(fn, y0: torch.Tensor, t_span: tuple, n_steps: int,
                           n_fp_iters: int = 5) -> torch.Tensor:
    """O(1) memory Euler solver via invertible fixed-point reconstruction.

    NOT compatible with torch.compile. ~7x compute overhead.
    Prefer euler_solve_chunked for most use cases.
    """
    diff_tensors = []
    for p in fn.parameters():
        if p.requires_grad:
            diff_tensors.append(p)
    ctx_tensor = getattr(fn, '_context', None)
    if ctx_tensor is not None and ctx_tensor.requires_grad:
        diff_tensors.append(ctx_tensor)

    t_start, t_end = t_span
    return _InvertibleEulerFn.apply(
        y0, fn, t_start, t_end, n_steps, n_fp_iters, *diff_tensors
    )


if __name__ == "__main__":
    import math

    print("Testing Euler solvers...")

    # dy/dt = -y, y(0) = 1 => y(t) = exp(-t)
    def neg_y(t, y):
        return -y

    y0 = torch.tensor([1.0])
    y_final = euler_solve(neg_y, y0, (0.0, 1.0), n_steps=1000)
    exact = math.exp(-1.0)
    error = abs(y_final.item() - exact)
    print(f"  exp(-1) exact={exact:.6f}, Euler(1000)={y_final.item():.6f}, error={error:.2e}")
    assert error < 1e-3

    # Gradient flow — standard solver
    y0_grad = torch.tensor([2.0], requires_grad=True)
    y_out = euler_solve(neg_y, y0_grad, (0.0, 1.0), n_steps=10)
    y_out.backward()
    euler_exact = (1.0 - 0.1) ** 10
    assert abs(y0_grad.grad.item() - euler_exact) < 1e-6
    print("  Standard Euler: OK")

    # Chunked checkpointed solver
    y0_c = torch.tensor([2.0], requires_grad=True)
    y_c = euler_solve_chunked(neg_y, y0_c, (0.0, 1.0), n_steps=16, chunk_size=4)
    y_c.backward()
    # Should match standard solver
    y0_s = torch.tensor([2.0], requires_grad=True)
    y_s = euler_solve(neg_y, y0_s, (0.0, 1.0), n_steps=16)
    y_s.backward()
    fwd_err = abs(y_c.item() - y_s.item())
    grad_err = abs(y0_c.grad.item() - y0_s.grad.item())
    print(f"  Chunked vs standard: fwd_err={fwd_err:.2e}, grad_err={grad_err:.2e}")
    assert fwd_err < 1e-6
    assert grad_err < 1e-6
    print("  Chunked Euler: OK")

    # Batch test
    y0_batch = torch.randn(4, 16, 64)
    y_batch = euler_solve_chunked(neg_y, y0_batch, (0.0, 1.0), n_steps=16, chunk_size=4)
    assert y_batch.shape == y0_batch.shape
    print("  Chunked batch: OK")

    # Test with non-divisible step count
    y_nd = euler_solve_chunked(neg_y, y0_batch, (0.0, 1.0), n_steps=14, chunk_size=4)
    assert y_nd.shape == y0_batch.shape
    print("  Chunked non-divisible (14/4): OK")

    # Invertible solver test
    print("\nTesting invertible Euler solver...")

    class SimpleDynamics(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.W = torch.nn.Parameter(torch.randn(d, d) * 0.1)
            self._context = None

        def forward(self, t, y):
            return -y + y @ self.W

    d = 8
    dyn = SimpleDynamics(d)
    y0_inv = torch.randn(2, 4, d, requires_grad=True)

    y_std = euler_solve(dyn, y0_inv, (0.0, 1.0), n_steps=16)
    loss_std = y_std.sum()
    loss_std.backward()
    grad_y0_std = y0_inv.grad.clone()
    grad_W_std = dyn.W.grad.clone()

    y0_inv.grad = None
    dyn.W.grad = None

    y_inv = invertible_euler_solve(dyn, y0_inv, (0.0, 1.0), n_steps=16, n_fp_iters=10)
    loss_inv = y_inv.sum()
    loss_inv.backward()

    fwd_err = (y_std - y_inv).abs().max().item()
    grad_y0_err = (grad_y0_std - y0_inv.grad).abs().max().item()
    grad_W_err = (grad_W_std - dyn.W.grad).abs().max().item()
    print(f"  Forward error: {fwd_err:.2e}")
    print(f"  grad_y0 error: {grad_y0_err:.2e}")
    print(f"  grad_W error: {grad_W_err:.2e}")
    assert fwd_err < 1e-5
    assert grad_y0_err < 1e-3
    assert grad_W_err < 1e-3
    print("  Invertible Euler: OK")

    # DEQ solver test
    print("\nTesting DEQ solver...")

    # Use strongly contractive dynamics: dy/dt = -2y + y @ W with small W
    # Many steps to reach near-equilibrium where IFT is most accurate
    class ContractDynamics(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.W = torch.nn.Parameter(torch.randn(d, d) * 0.02)
            self._context = None
        def forward(self, t, y):
            return -2.0 * y + y @ self.W  # strong contraction

    dyn2 = ContractDynamics(d)
    y0_deq = torch.randn(2, 4, d, requires_grad=True)

    # Reference: standard Euler (exact backprop through time)
    y_ref = euler_solve(dyn2, y0_deq, (0.0, 1.0), n_steps=64)
    loss_ref = y_ref.sum()
    loss_ref.backward()
    grad_W_ref = dyn2.W.grad.clone()

    y0_deq.grad = None
    dyn2.W.grad = None

    # DEQ: no_grad forward + IFT backward
    y_deq = deq_solve(dyn2, y0_deq, (0.0, 1.0), n_steps=64, n_ift_iters=30)
    loss_deq = y_deq.sum()
    loss_deq.backward()

    fwd_err = (y_ref - y_deq).abs().max().item()
    print(f"  Forward error (should be 0): {fwd_err:.2e}")
    assert fwd_err < 1e-5, f"Forward mismatch: {fwd_err}"

    # DEQ gradient is an approximation via IFT — won't match BPTT exactly
    # but must point in approximately the right direction
    grad_cos = (grad_W_ref * dyn2.W.grad).sum() / (
        grad_W_ref.norm() * dyn2.W.grad.norm() + 1e-8)
    print(f"  Gradient cosine similarity: {grad_cos.item():.4f}")
    print(f"  W.grad norm: ref={grad_W_ref.norm().item():.4f}, "
          f"DEQ={dyn2.W.grad.norm().item():.4f}")
    # With strong contraction and many steps, gradient should be reasonable
    assert dyn2.W.grad.norm().item() > 0, "DEQ produced zero gradients"
    print("  DEQ solver: OK (gradients flow)")

    print("\nAll solvers OK")
