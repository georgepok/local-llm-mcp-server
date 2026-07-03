"""Multi-substrate toy: do K identical-template substrates with shared
input but separate parameters develop DIFFERENT functional roles when the
task requires it?

The question this answers: is "self-organizing differentiation through
training pressure on coupled substrates" a real mechanism, or do
substrates collapse to redundant copies?

Task: minimal mode-switched computation. Input is 4 numbers (a, b, c, mode):
    if mode > 0:  target = a + b
    else:         target = b + c
This requires (1) reading the mode bit, (2) conditionally selecting the
right pair to sum. Two distinct cognitive operations.

Architecture:
    K parallel substrates, each is a small MLP (4 → 8 → 4 hidden + ReLU).
    At every "step" of an inner loop, each substrate's input is
    [original_input || mean(other substrates' hidden states)].
    Final output = linear(concat all substrates' final hidden states).

Two ablations:
    A) K=1            (single substrate baseline)
    B) K=2 isolated   (no lateral coupling)
    C) K=2 coupled    (lateral coupling — the proposed mechanism)

Measure:
    - Test MSE on held-out examples
    - Substrate functional divergence: cosine similarity between substrates'
      final hidden states across the test set. Low cos = specialized.
    - Per-substrate sensitivity: ablate (zero out) one substrate, measure
      MSE degradation. If different substrates' ablations cause different
      MSE patterns, they have different functional roles.

Pass criterion for the proposed mechanism (coupled K=2):
    1. MSE(coupled K=2) < MSE(K=1) by at least 30%
    2. Cos similarity between substrates < 0.5 (genuinely different)
    3. Different ablations cause different error patterns
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# Task
# -----------------------------------------------------------------------------

def make_task(n_examples: int, rng):
    """Multi-mode pair-selection task — much harder than 4-input.
    Input: 8 numbers + 2 mode bits = 10-dim.
    Mode (m1, m2) ∈ {(0,0), (0,1), (1,0), (1,1)} selects one of 4 ops:
        (0,0): x[0] + x[1]
        (0,1): x[2] * x[3]
        (1,0): max(x[4], x[5])
        (1,1): x[6] - x[7]
    A single small MLP must internally TIME-SHARE its capacity across 4
    distinct operations — limits achievable accuracy at constrained width.
    """
    X_num = rng.normal(size=(n_examples, 8)).astype(np.float32)
    m = rng.integers(0, 2, size=(n_examples, 2)).astype(np.float32)
    X = np.concatenate([X_num, m], axis=-1)
    op = (m[:, 0] * 2 + m[:, 1]).astype(int)
    targets = np.zeros(n_examples, dtype=np.float32)
    for i in range(n_examples):
        if op[i] == 0:   targets[i] = X_num[i, 0] + X_num[i, 1]
        elif op[i] == 1: targets[i] = X_num[i, 2] * X_num[i, 3]
        elif op[i] == 2: targets[i] = max(X_num[i, 4], X_num[i, 5])
        else:            targets[i] = X_num[i, 6] - X_num[i, 7]
    return X, targets


# -----------------------------------------------------------------------------
# Multi-substrate model (numpy, manual gradients via finite-diff free —
# we use autograd-style chain rule on a small MLP)
# -----------------------------------------------------------------------------

def relu(x): return np.maximum(0, x)
def relu_grad(x): return (x > 0).astype(x.dtype)

class Substrate:
    """One small MLP: in_dim → hidden → hidden_out."""
    def __init__(self, in_dim, hidden, out_dim, rng):
        s = 1.0 / np.sqrt(in_dim)
        self.W1 = rng.normal(size=(in_dim, hidden)).astype(np.float32) * s
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(size=(hidden, out_dim)).astype(np.float32) * (1.0 / np.sqrt(hidden))
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        # Caches for backward
        self.cache = None

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        self.cache = (x, z1, a1)
        return z2

    def backward(self, dout):
        x, z1, a1 = self.cache
        dW2 = a1.T @ dout
        db2 = dout.sum(axis=0)
        da1 = dout @ self.W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)
        dx = dz1 @ self.W1.T
        return dx, (dW1, db1, dW2, db2)

    def step(self, grads, lr):
        dW1, db1, dW2, db2 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2


class MultiSubstrate:
    """K substrates with optional lateral coupling.  Output via linear head."""
    def __init__(self, K, in_dim, hidden, out_dim_per_sub, n_inner_steps,
                  coupled, rng):
        self.K = K
        self.coupled = coupled
        self.n_inner_steps = n_inner_steps
        self.out_dim_per_sub = out_dim_per_sub
        coupled_extra = (out_dim_per_sub if coupled else 0)
        sub_in_dim = in_dim + coupled_extra
        self.subs = [Substrate(sub_in_dim, hidden, out_dim_per_sub, rng)
                      for _ in range(K)]
        # Output head: K * out_dim_per_sub → 1
        self.W_out = rng.normal(size=(K * out_dim_per_sub, 1)).astype(np.float32) * 0.1
        self.b_out = np.zeros(1, dtype=np.float32)
        self._cache = None

    def forward(self, X):
        B = X.shape[0]
        # Initialize each substrate's hidden state to zeros
        hs = [np.zeros((B, self.out_dim_per_sub), dtype=np.float32)
              for _ in range(self.K)]
        # Run n_inner_steps; at each step, every substrate sees its prior
        # output (not used) plus a summary of others (mean of others' h)
        # Note: this is a SINGLE outer pass per substrate, plus optional
        # iteration. For minimal toy, n_inner_steps=1 means each substrate
        # runs once with input features (and zero coupling).
        per_step_caches = []
        for t in range(self.n_inner_steps):
            # Compute the coupling summary BEFORE this step's updates
            if self.coupled and self.K > 1:
                stack = np.stack(hs, axis=0)  # [K, B, out]
                others_sum = stack.sum(axis=0, keepdims=True) - stack  # [K, B, out]
                others_mean = others_sum / max(self.K - 1, 1)
            else:
                others_mean = None
            new_hs = []
            step_cache = []
            for k, sub in enumerate(self.subs):
                if self.coupled and self.K > 1:
                    inp = np.concatenate([X, others_mean[k]], axis=-1)
                else:
                    inp = X
                out = sub.forward(inp)
                new_hs.append(out)
                step_cache.append(inp)
            hs = new_hs
            per_step_caches.append(step_cache)
        # Concatenate all substrate outputs and project to scalar
        h_concat = np.concatenate(hs, axis=-1)
        y_pred = (h_concat @ self.W_out + self.b_out).squeeze(-1)
        self._cache = (X, hs, h_concat, per_step_caches)
        return y_pred

    def backward(self, dy_pred):
        X, hs, h_concat, per_step_caches = self._cache
        # dy_pred: [B]
        dy = dy_pred[:, None]  # [B, 1]
        dW_out = h_concat.T @ dy
        db_out = dy.sum(axis=0)
        dh_concat = dy @ self.W_out.T  # [B, K * out_per_sub]
        # Split into per-substrate gradients
        dhs = [dh_concat[:, k * self.out_dim_per_sub:(k + 1) * self.out_dim_per_sub]
               for k in range(self.K)]
        # Backprop through n_inner_steps in reverse
        all_grads = [None] * self.K
        for t in reversed(range(self.n_inner_steps)):
            new_dhs = [np.zeros_like(d) for d in dhs]
            grads_this_step = []
            for k, sub in enumerate(self.subs):
                # Restore cache for this substrate at this step
                sub.cache = (per_step_caches[t][k],) + sub.cache[1:] \
                    if sub.cache else None
                # Re-do forward to set cache (simpler than restoring full cache)
                sub.forward(per_step_caches[t][k])
                dx, grads = sub.backward(dhs[k])
                grads_this_step.append(grads)
                # Split dx into "original X" gradient and "coupling" gradient
                if self.coupled and self.K > 1:
                    in_dim_orig = X.shape[1]
                    # dx_coupling: [B, out_per_sub] is gradient from k's perspective
                    dx_coupling = dx[:, in_dim_orig:]
                    # Coupling came from mean of OTHER substrates' h
                    # ∂(others_mean[k]) / ∂(h_j for j != k) = 1/(K-1)
                    for j in range(self.K):
                        if j != k:
                            new_dhs[j] = new_dhs[j] + dx_coupling / max(self.K - 1, 1)
            # Accumulate gradients into all_grads (sum across inner steps)
            for k in range(self.K):
                if all_grads[k] is None:
                    all_grads[k] = list(grads_this_step[k])
                else:
                    all_grads[k] = [a + b for a, b in zip(all_grads[k],
                                                            grads_this_step[k])]
            dhs = new_dhs
        return all_grads, dW_out, db_out

    def step(self, all_grads, dW_out, db_out, lr):
        for k, sub in enumerate(self.subs):
            sub.step(all_grads[k], lr)
        self.W_out -= lr * dW_out
        self.b_out -= lr * db_out

    def get_substrate_outputs(self, X):
        _ = self.forward(X)
        _, hs, _, _ = self._cache
        return hs  # list of [B, out_per_sub]


# -----------------------------------------------------------------------------
# Train + measure
# -----------------------------------------------------------------------------

def train_and_eval(K, coupled, n_inner_steps, n_train_steps=8000,
                    batch_size=64, hidden=8, out_per_sub=2, lr=0.005,
                    seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    model = MultiSubstrate(K=K, in_dim=10, hidden=hidden,
                            out_dim_per_sub=out_per_sub,
                            n_inner_steps=n_inner_steps,
                            coupled=coupled, rng=rng)
    X_test, y_test = make_task(500, rng)
    losses = []
    for step in range(n_train_steps):
        X, y = make_task(batch_size, rng)
        y_pred = model.forward(X)
        loss = ((y_pred - y) ** 2).mean()
        dy = 2 * (y_pred - y) / batch_size
        all_grads, dW_out, db_out = model.backward(dy)
        model.step(all_grads, dW_out, db_out, lr)
        losses.append(loss)
        if verbose and step % 500 == 0:
            test_pred = model.forward(X_test)
            test_mse = ((test_pred - y_test) ** 2).mean()
            print(f"  step {step}: train={loss:.4f} test_mse={test_mse:.4f}")
    test_pred = model.forward(X_test)
    test_mse = float(((test_pred - y_test) ** 2).mean())
    return model, test_mse


def measure_differentiation(model, X_test):
    """Return cosine similarity between substrates' final outputs across tests."""
    if model.K < 2:
        return None
    hs = model.get_substrate_outputs(X_test)
    sims = []
    for i in range(model.K):
        for j in range(i + 1, model.K):
            a = hs[i].reshape(-1)
            b = hs[j].reshape(-1)
            cos = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            sims.append(cos)
    return float(np.mean(sims))


def measure_ablation(model, X_test, y_test):
    """Zero out each substrate in turn, measure MSE degradation."""
    if model.K < 2:
        return None
    base_pred = model.forward(X_test)
    base_mse = float(((base_pred - y_test) ** 2).mean())
    abl_mses = []
    for k in range(model.K):
        # Save and zero substrate k's W2 (its output projection)
        W2_save = model.subs[k].W2.copy()
        b2_save = model.subs[k].b2.copy()
        model.subs[k].W2 = np.zeros_like(W2_save)
        model.subs[k].b2 = np.zeros_like(b2_save)
        pred = model.forward(X_test)
        mse = float(((pred - y_test) ** 2).mean())
        abl_mses.append(mse)
        model.subs[k].W2 = W2_save
        model.subs[k].b2 = b2_save
    return base_mse, abl_mses


def main():
    print("Multi-substrate differentiation toy")
    print("Task: y = a+b if mode > 0 else b+c, where (a,b,c,mode) ~ N(0,I)")
    print()
    rng = np.random.default_rng(99)
    X_test, y_test = make_task(500, rng)
    print(f"Random baseline (predict 0):  MSE={(y_test ** 2).mean():.3f}")
    print(f"Mean-target baseline:         MSE={((y_test - y_test.mean()) ** 2).mean():.3f}")
    print()

    # Three configurations
    configs = [
        ("K=1                       ", 1, False, 1),
        ("K=2 isolated (no coupling)", 2, False, 1),
        ("K=2 coupled, 1 inner step ", 2, True,  1),
        ("K=2 coupled, 3 inner steps", 2, True,  3),
        ("K=4 coupled, 3 inner steps", 4, True,  3),
    ]
    results = {}
    for name, K, coupled, n_inner in configs:
        # Average over 3 seeds
        mses = []
        sims = []
        ablation_spreads = []
        for seed in range(3):
            model, mse = train_and_eval(
                K=K, coupled=coupled, n_inner_steps=n_inner,
                n_train_steps=8000, seed=seed, verbose=False,
            )
            mses.append(mse)
            sim = measure_differentiation(model, X_test)
            if sim is not None:
                sims.append(sim)
            abl = measure_ablation(model, X_test, y_test)
            if abl is not None:
                _, abl_mses = abl
                ablation_spreads.append(np.std(abl_mses))
        mse_mean = np.mean(mses)
        sim_mean = np.mean(sims) if sims else float("nan")
        abl_mean = np.mean(ablation_spreads) if ablation_spreads else float("nan")
        results[name] = (mse_mean, sim_mean, abl_mean)
        print(f"{name} | test_mse={mse_mean:.4f}  cos_sim={sim_mean:.3f}  "
              f"ablation_spread={abl_mean:.3f}")

    print()
    print("Differentiation success criterion (coupled K>=2):")
    print("  - test_mse < K=1's mse * 0.7  (capacity gain from coupling)")
    print("  - cos_sim < 0.5               (substrates ARE different)")
    print("  - ablation_spread > 0.1       (different substrates do different jobs)")
    base_mse = results["K=1                       "][0]
    print()
    for name, (mse, sim, abl) in results.items():
        if "K=1" in name:
            continue
        passes = (mse < base_mse * 0.7 and sim < 0.5 and abl > 0.1)
        print(f"{name}: {'PASS' if passes else 'fail'} "
              f"(mse_ratio={mse/base_mse:.2f}, cos={sim:.3f}, abl={abl:.3f})")


if __name__ == "__main__":
    main()
