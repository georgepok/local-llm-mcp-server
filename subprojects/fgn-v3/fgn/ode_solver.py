"""Fixed-step ODE solvers — fully torch.compile compatible.

No external dependencies. Fixed step count means loops unroll at trace time.
Euler is the default: 1 fn eval/step, minimal memory, compile-friendly.

Why not torchdiffeq adjoint? The adjoint backward ODE re-evaluates dynamics
at every step. With O(N^2) geodesic distances, this is catastrophically slow
(~16 tok/s vs 500 tok/s with Euler). The memory savings aren't worth 30x slowdown.
"""

import torch


def euler_solve(fn, y0: torch.Tensor, t_span: tuple, n_steps: int) -> torch.Tensor:
    """Forward Euler: y_{n+1} = y_n + dt * f(t_n, y_n).

    1 dynamics evaluation per step. Memory = n_steps distance matrices on autograd tape.
    With n_steps=4, same memory footprint as FluidLayer's 3-scale diffusion.

    Args:
        fn: callable(t: float, y: Tensor) -> Tensor
        y0: initial condition [any shape]
        t_span: (t_start, t_end)
        n_steps: number of Euler steps (fixed — unrolls at compile time)

    Returns:
        y at t_end (same shape as y0)
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    t = t_start
    y = y0
    for _ in range(n_steps):
        y = y + dt * fn(t, y)
        t = t + dt
    return y


if __name__ == "__main__":
    import math

    print("Testing Euler solver...")

    # dy/dt = -y, y(0) = 1 => y(t) = exp(-t)
    def neg_y(t, y):
        return -y

    y0 = torch.tensor([1.0])
    y_final = euler_solve(neg_y, y0, (0.0, 1.0), n_steps=1000)
    exact = math.exp(-1.0)
    error = abs(y_final.item() - exact)
    print(f"  exp(-1) exact={exact:.6f}, Euler(1000)={y_final.item():.6f}, error={error:.2e}")
    assert error < 1e-3, f"Euler error too large: {error}"

    # 4 steps (what we'll actually use) — lower accuracy but learned dynamics compensate
    y4 = euler_solve(neg_y, y0, (0.0, 1.0), n_steps=4)
    error4 = abs(y4.item() - exact)
    print(f"  exp(-1) Euler(4)={y4.item():.6f}, error={error4:.2e}")

    # Gradient flow
    y0_grad = torch.tensor([2.0], requires_grad=True)
    y_out = euler_solve(neg_y, y0_grad, (0.0, 1.0), n_steps=10)
    y_out.backward()
    euler_exact = (1.0 - 0.1) ** 10
    print(f"  Gradient: dy/dy0 = {y0_grad.grad.item():.6f} (Euler exact={euler_exact:.6f})")
    assert abs(y0_grad.grad.item() - euler_exact) < 1e-6, "Gradient error"

    # Batch test
    y0_batch = torch.randn(4, 16, 64)
    y_batch = euler_solve(neg_y, y0_batch, (0.0, 1.0), n_steps=4)
    assert y_batch.shape == y0_batch.shape, f"Shape mismatch: {y_batch.shape}"
    print(f"  Batch test: {y0_batch.shape} -> {y_batch.shape}")

    print("Euler solver OK")
