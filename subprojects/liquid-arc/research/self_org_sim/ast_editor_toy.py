"""AST editor — sequence-editing analog with (pointer, op, payload) heads.

Sibling to multi_substrate_toy.py. Same coupling mechanism, different head
structure. If K=2 coupled outperforms K=1 wide AND substrate ablation shows
asymmetric per-head dependency, the mechanism applies to AST editing and we
port it into LiquidARC for Phase 1.

Pure NumPy. Re-uses Substrate from multi_substrate_toy.

Run:
    python ast_editor_toy.py
"""

from __future__ import annotations

import numpy as np

# Local re-use of the prior toy's Substrate. Same directory, so vanilla import
# works whether you `python ast_editor_toy.py` or run from the repo root.
from multi_substrate_toy import Substrate


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

N = 12          # sequence length
V = 8           # token vocabulary
N_OPS = 3       # NOP, SET, SWAP
NOP, SET, SWAP = 0, 1, 2
K_CORRUPT = 2
K_FIX = 3
PAYLOAD_DIM = max(V, N - 1)


def apply_edit(seq: np.ndarray, p: int, o: int, y: int) -> np.ndarray:
    """Apply edit; payload is interpreted by op (token for SET, offset for SWAP).
    Out-of-range payloads are silently ignored (treated as NOP) so eval-time
    rollout can't corrupt state when the model emits an inconsistent triple."""
    seq = seq.copy()
    if o == NOP:
        return seq
    if o == SET:
        if 0 <= y < V:
            seq[p] = y
    elif o == SWAP:
        if 0 <= y < N - 1:
            q = (p + y + 1) % N
            seq[p], seq[q] = seq[q], seq[p]
    return seq


def gen_corruption(target: np.ndarray, k_corrupt: int, rng) -> np.ndarray:
    state = target.copy()
    for _ in range(k_corrupt):
        o = int(rng.choice([SET, SWAP]))
        p = int(rng.integers(0, N))
        if o == SET:
            choices = [v for v in range(V) if v != state[p]]
            y = int(rng.choice(choices))
        else:
            for _try in range(2 * N):
                y = int(rng.integers(0, N - 1))
                q = (p + y + 1) % N
                if state[p] != state[q]:
                    break
            else:
                p = int(rng.integers(0, N))
                y = int(rng.integers(0, N - 1))
        state = apply_edit(state, p, o, y)
    return state


def canonical_fix(state: np.ndarray, target: np.ndarray, max_steps: int):
    """Greedy leftmost-mismatch fix policy. Returns (script, recovered)."""
    state = state.copy()
    script = []
    for _ in range(max_steps):
        diff = [i for i in range(N) if state[i] != target[i]]
        if not diff:
            script.append((0, NOP, 0))
            continue
        p = diff[0]
        swap_q = None
        for q in diff[1:]:
            if state[q] == target[p] and state[p] == target[q]:
                swap_q = q
                break
        if swap_q is not None:
            y = (swap_q - p - 1) % N
            edit = (p, SWAP, y)
        else:
            edit = (p, SET, int(target[p]))
        state = apply_edit(state, *edit)
        script.append(edit)
    return script, bool((state == target).all())


def gen_example(rng, max_resamples: int = 5):
    """Returns (state_0, target, script) where canonical fix recovers target."""
    target = rng.integers(0, V, size=N)
    src = gen_corruption(target, K_CORRUPT, rng)
    script, recovered = canonical_fix(src, target, K_FIX)
    for _ in range(max_resamples - 1):
        if recovered:
            break
        target = rng.integers(0, V, size=N)
        src = gen_corruption(target, K_CORRUPT, rng)
        script, recovered = canonical_fix(src, target, K_FIX)
    return src, target, script


def encode_input(state: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Encode (state, target) with a per-position match bit so the substrate
    doesn't have to derive equality from raw one-hot comparison.
    Layout: [state_oh (N*V), target_oh (N*V), match_bits (N)] → 2*N*V + N.
    """
    s_oh = np.eye(V, dtype=np.float32)[state]
    t_oh = np.eye(V, dtype=np.float32)[target]
    match = (state == target).astype(np.float32)  # [N]
    return np.concatenate([s_oh.ravel(), t_oh.ravel(), match])


def gen_training_triples(rng, n_examples: int):
    """For each generated example, expand to K_FIX (state_k, target, gt_edit)
    triples using teacher-forced cumulative state."""
    triples = []
    for _ in range(n_examples):
        s0, t, script = gen_example(rng)
        state = s0
        for k, edit in enumerate(script):
            triples.append((state.copy(), t.copy(), edit))
            state = apply_edit(state, *edit)
    return triples


def batch_from_triples(triples, indices):
    B = len(indices)
    X = np.zeros((B, 2 * N * V + N), dtype=np.float32)
    gt_p = np.zeros(B, dtype=np.int64)
    gt_o = np.zeros(B, dtype=np.int64)
    gt_y = np.zeros(B, dtype=np.int64)
    for i, idx in enumerate(indices):
        s, t, (p, o, y) = triples[idx]
        X[i] = encode_input(s, t)
        gt_p[i], gt_o[i], gt_y[i] = p, o, y
    return X, gt_p, gt_o, gt_y


# ---------------------------------------------------------------------------
# Multi-substrate editor — three heads on top of the multi_substrate_toy core
# ---------------------------------------------------------------------------

class MultiSubstrateEditor:
    """K substrates with optional lateral coupling, three emit heads.

    Substrate, coupling, and inner-step backprop mirror MultiSubstrate from
    multi_substrate_toy.py. The only difference is the output side: instead of
    a scalar regression head, we have three independent linear heads
    (pointer, op, payload).
    """

    def __init__(self, K, in_dim, hidden, out_per_sub, coupled,
                 n_inner_steps, rng):
        self.K = K
        self.coupled = coupled
        self.n_inner_steps = n_inner_steps
        self.out_per_sub = out_per_sub
        coupled_extra = (out_per_sub if coupled else 0)
        sub_in_dim = in_dim + coupled_extra
        self.sub_in_dim = sub_in_dim
        self.in_dim = in_dim
        self.subs = [Substrate(sub_in_dim, hidden, out_per_sub, rng)
                     for _ in range(K)]
        concat_dim = K * out_per_sub
        s = 1.0 / np.sqrt(concat_dim)
        self.W_ptr = (rng.normal(size=(concat_dim, N)).astype(np.float32) * s)
        self.b_ptr = np.zeros(N, dtype=np.float32)
        self.W_op = (rng.normal(size=(concat_dim, N_OPS)).astype(np.float32) * s)
        self.b_op = np.zeros(N_OPS, dtype=np.float32)
        self.W_pay = (rng.normal(size=(concat_dim, PAYLOAD_DIM)).astype(np.float32) * s)
        self.b_pay = np.zeros(PAYLOAD_DIM, dtype=np.float32)
        self._cache = None

    def _run_substrates(self, X):
        B = X.shape[0]
        hs = [np.zeros((B, self.out_per_sub), dtype=np.float32)
              for _ in range(self.K)]
        per_step_caches = []
        for _t in range(self.n_inner_steps):
            others_mean = None
            if self.coupled and self.K > 1:
                stack = np.stack(hs, axis=0)
                others_sum = stack.sum(axis=0, keepdims=True) - stack
                others_mean = others_sum / max(self.K - 1, 1)
            new_hs = []
            step_cache = []
            for k, sub in enumerate(self.subs):
                if others_mean is not None:
                    inp = np.concatenate([X, others_mean[k]], axis=-1)
                else:
                    inp = X
                out = sub.forward(inp)
                new_hs.append(out)
                step_cache.append(inp)
            hs = new_hs
            per_step_caches.append(step_cache)
        return hs, per_step_caches

    def forward(self, X, head_mask=None):
        """head_mask: dict {(sub_idx, head_name): True} to zero those rows of
        the substrate-→-head projection (used for ablation probes). Does NOT
        affect substrate forward — only fusion."""
        hs, per_step_caches = self._run_substrates(X)
        h_concat = np.concatenate(hs, axis=-1)
        # Optional ablation: zero the slice of the per-head projection that
        # corresponds to this substrate.
        ptr = h_concat @ self._maybe_ablate(self.W_ptr, head_mask, "ptr") + self.b_ptr
        op = h_concat @ self._maybe_ablate(self.W_op, head_mask, "op") + self.b_op
        pay = h_concat @ self._maybe_ablate(self.W_pay, head_mask, "pay") + self.b_pay
        self._cache = (X, hs, h_concat, per_step_caches)
        return ptr, op, pay

    def _maybe_ablate(self, W, head_mask, head_name):
        if not head_mask:
            return W
        W = W.copy()
        for (sub_idx, h), val in head_mask.items():
            if h != head_name or not val:
                continue
            i0 = sub_idx * self.out_per_sub
            i1 = (sub_idx + 1) * self.out_per_sub
            W[i0:i1, :] = 0.0
        return W

    def backward(self, d_ptr, d_op, d_pay):
        X, hs, h_concat, per_step_caches = self._cache
        dW_ptr = h_concat.T @ d_ptr
        db_ptr = d_ptr.sum(axis=0)
        dW_op = h_concat.T @ d_op
        db_op = d_op.sum(axis=0)
        dW_pay = h_concat.T @ d_pay
        db_pay = d_pay.sum(axis=0)
        dh_concat = (d_ptr @ self.W_ptr.T) + (d_op @ self.W_op.T) + (d_pay @ self.W_pay.T)
        dhs = [dh_concat[:, k * self.out_per_sub:(k + 1) * self.out_per_sub]
               for k in range(self.K)]
        all_grads: list = [None] * self.K
        for t in reversed(range(self.n_inner_steps)):
            new_dhs = [np.zeros_like(d) for d in dhs]
            grads_this = []
            for k, sub in enumerate(self.subs):
                # Re-set substrate cache for this step's input
                sub.forward(per_step_caches[t][k])
                dx, grads = sub.backward(dhs[k])
                grads_this.append(grads)
                if self.coupled and self.K > 1:
                    dx_coupling = dx[:, self.in_dim:]
                    for j in range(self.K):
                        if j != k:
                            new_dhs[j] = new_dhs[j] + dx_coupling / max(self.K - 1, 1)
            for k in range(self.K):
                if all_grads[k] is None:
                    all_grads[k] = list(grads_this[k])
                else:
                    all_grads[k] = [a + b for a, b in zip(all_grads[k], grads_this[k])]
            dhs = new_dhs
        return all_grads, (dW_ptr, db_ptr, dW_op, db_op, dW_pay, db_pay)

    def step(self, all_grads, head_grads, lr):
        for k, sub in enumerate(self.subs):
            sub.step(all_grads[k], lr)
        dW_ptr, db_ptr, dW_op, db_op, dW_pay, db_pay = head_grads
        self.W_ptr -= lr * dW_ptr
        self.b_ptr -= lr * db_ptr
        self.W_op -= lr * dW_op
        self.b_op -= lr * db_op
        self.W_pay -= lr * dW_pay
        self.b_pay -= lr * db_pay

    def get_substrate_outputs(self, X):
        hs, _ = self._run_substrates(X)
        return hs


def n_params(model):
    total = 0
    for sub in model.subs:
        total += sub.W1.size + sub.b1.size + sub.W2.size + sub.b2.size
    total += model.W_ptr.size + model.b_ptr.size
    total += model.W_op.size + model.b_op.size
    total += model.W_pay.size + model.b_pay.size
    return total


# ---------------------------------------------------------------------------
# Loss + train + eval
# ---------------------------------------------------------------------------

def softmax_xe(logits, target):
    """Returns (loss, dlogits). logits [B, C], target [B] int."""
    B = logits.shape[0]
    z = logits - logits.max(axis=-1, keepdims=True)
    expz = np.exp(z)
    p = expz / expz.sum(axis=-1, keepdims=True)
    nll = -np.log(p[np.arange(B), target] + 1e-12).mean()
    dl = p.copy()
    dl[np.arange(B), target] -= 1.0
    dl = dl / B
    return float(nll), dl.astype(np.float32)


def train_and_eval(K, coupled, hidden, n_inner_steps,
                   n_train_steps, batch_size, lr, seed,
                   eval_examples=1024, verbose=False):
    rng = np.random.default_rng(seed)
    model = MultiSubstrateEditor(
        K=K, in_dim=2 * N * V + N, hidden=hidden, out_per_sub=16,
        coupled=coupled, n_inner_steps=n_inner_steps, rng=rng,
    )
    train_pool = gen_training_triples(rng, n_examples=2048)

    losses = []
    for step in range(n_train_steps):
        idx = rng.integers(0, len(train_pool), size=batch_size)
        X, gt_p, gt_o, gt_y = batch_from_triples(train_pool, idx)
        ptr, op, pay = model.forward(X)
        l_ptr, d_ptr = softmax_xe(ptr, gt_p)
        l_op, d_op = softmax_xe(op, gt_o)
        l_pay, d_pay = softmax_xe(pay, gt_y)
        loss = l_ptr + l_op + l_pay
        all_grads, head_grads = model.backward(d_ptr, d_op, d_pay)
        model.step(all_grads, head_grads, lr)
        losses.append(loss)
        if verbose and (step % 1000 == 0 or step == n_train_steps - 1):
            recent = np.mean(losses[-200:])
            print(f"  step {step:5d}  loss {loss:.3f}  avg-200 {recent:.3f}")
        # Periodic refresh of train pool to avoid overfitting fixed batches
        if (step + 1) % 1000 == 0:
            train_pool = gen_training_triples(rng, n_examples=2048)
    em = exact_match_eval(model, rng_seed=seed + 9999, n_examples=eval_examples)
    return model, em, losses


def exact_match_eval(model, rng_seed, n_examples=1024, head_mask=None,
                     batch_size=64):
    """Sequentially apply model edits; return fraction of examples where
    final state == target."""
    rng = np.random.default_rng(rng_seed)
    correct = 0
    total = 0
    while total < n_examples:
        # Generate a batch of fresh examples
        sources = []
        targets = []
        for _ in range(batch_size):
            s, t, _ = gen_example(rng)
            sources.append(s)
            targets.append(t)
        sources = np.stack(sources)
        targets = np.stack(targets)
        cur = sources.copy()
        for _step in range(K_FIX):
            X = np.stack([encode_input(cur[b], targets[b]) for b in range(batch_size)])
            ptr, op, pay = model.forward(X, head_mask=head_mask)
            p_pred = ptr.argmax(axis=-1)
            o_pred = op.argmax(axis=-1)
            y_pred = pay.argmax(axis=-1)
            new_cur = []
            for b in range(batch_size):
                new_cur.append(apply_edit(cur[b], int(p_pred[b]),
                                          int(o_pred[b]), int(y_pred[b])))
            cur = np.stack(new_cur)
        match = (cur == targets).all(axis=-1)
        correct += int(match.sum())
        total += batch_size
    return correct / total


def measure_differentiation(model, rng_seed, n_examples=512, batch_size=64):
    if model.K < 2:
        return None
    rng = np.random.default_rng(rng_seed)
    sims = []
    n = 0
    while n < n_examples:
        Xs = []
        for _ in range(batch_size):
            s, t, _ = gen_example(rng)
            Xs.append(encode_input(s, t))
        X = np.stack(Xs)
        hs = model.get_substrate_outputs(X)
        for i in range(model.K):
            for j in range(i + 1, model.K):
                a = hs[i].ravel()
                b = hs[j].ravel()
                cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                sims.append(float(cos))
        n += batch_size
    return float(np.mean(sims))


def measure_ablation(model, rng_seed):
    if model.K < 2:
        return None
    base = exact_match_eval(model, rng_seed=rng_seed, n_examples=512)
    out = {"base": base}
    for sub in range(model.K):
        for head in ("ptr", "op", "pay"):
            mask = {(sub, head): True}
            em = exact_match_eval(model, rng_seed=rng_seed,
                                  n_examples=512, head_mask=mask)
            out[(sub, head)] = em
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

CONFIGS = [
    # name, K, coupled, hidden, n_inner
    ("K1",          1, False, 64,  1),
    ("K1_wide",     1, False, 128, 1),
    ("K2_isolated", 2, False, 64,  3),
    ("K2_coupled",  2, True,  64,  3),
]


def main():
    n_train_steps = 12000
    batch_size = 64
    lr = 0.02
    n_seeds = 3
    eval_examples = 1024

    print(f"AST editor toy — sequence editing with (where, what) heads")
    print(f"Task: N={N}, V={V}, K_corrupt={K_CORRUPT}, K_fix={K_FIX}, ops={N_OPS}")
    print(f"Train: {n_train_steps} steps × {n_seeds} seeds, batch={batch_size}, lr={lr}")
    print()

    summary = {}
    sample_models = {}
    for name, K, coupled, hidden, n_inner in CONFIGS:
        ems = []
        sims = []
        params = None
        for seed in range(n_seeds):
            model, em, _ = train_and_eval(
                K=K, coupled=coupled, hidden=hidden,
                n_inner_steps=n_inner,
                n_train_steps=n_train_steps,
                batch_size=batch_size, lr=lr, seed=seed,
                eval_examples=eval_examples, verbose=(seed == 0),
            )
            params = n_params(model)
            ems.append(em)
            s = measure_differentiation(model, rng_seed=seed + 7777)
            if s is not None:
                sims.append(s)
            if seed == 0:
                sample_models[name] = model
            print(f"  [{name} seed {seed}] EM={em*100:.2f}%"
                  + (f" cos_sim={s:.3f}" if s is not None else ""))
        summary[name] = {
            "em_mean": float(np.mean(ems)),
            "em_std": float(np.std(ems)),
            "cos_sim_mean": float(np.mean(sims)) if sims else None,
            "params": params,
        }
        print(f"  → {name}: EM mean {summary[name]['em_mean']*100:.2f}% "
              f"± {summary[name]['em_std']*100:.2f}, params={params:,}")
        print()

    # Pass-criterion check
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for name, s in summary.items():
        cs = (f" cos_sim={s['cos_sim_mean']:.3f}"
              if s["cos_sim_mean"] is not None else "")
        print(f"  {name:14s} EM={s['em_mean']*100:6.2f}% ± {s['em_std']*100:.2f}"
              f"  params={s['params']:>7,}{cs}")

    em_k1w = summary["K1_wide"]["em_mean"]
    em_k2i = summary["K2_isolated"]["em_mean"]
    em_k2c = summary["K2_coupled"]["em_mean"]
    d_wide = em_k2c - em_k1w
    d_iso = em_k2c - em_k2i
    print(f"\nDeltas:")
    print(f"  K2_coupled − K1_wide      = {d_wide*100:+6.2f}pp")
    print(f"  K2_coupled − K2_isolated  = {d_iso*100:+6.2f}pp")
    coupling_pass = (d_wide >= 0.05) and (d_iso >= 0.05)
    print(f"\nMechanism (both ≥ 5pp): {'PASS' if coupling_pass else 'FAIL'}")

    cs_k2c = summary["K2_coupled"]["cos_sim_mean"]
    cs_k2i = summary["K2_isolated"]["cos_sim_mean"]
    print(f"\nCos-sim K2_coupled = {cs_k2c:.3f}, K2_isolated = {cs_k2i:.3f}")
    diff_pass = cs_k2c is not None and cs_k2c < 0.5
    print(f"Differentiation (K2_coupled cos_sim < 0.5): "
          f"{'PASS' if diff_pass else 'FAIL'}")

    print("\nAblation probe — K2_coupled (seed 0):")
    abl = measure_ablation(sample_models["K2_coupled"], rng_seed=12345)
    if abl is not None:
        base = abl["base"]
        print(f"  base EM = {base*100:.2f}%")
        head_spread = {}
        for head in ("ptr", "op", "pay"):
            d0 = base - abl[(0, head)]
            d1 = base - abl[(1, head)]
            head_spread[head] = abs(d0 - d1)
            print(f"    {head}: sub0 drop {d0*100:+5.2f}pp, "
                  f"sub1 drop {d1*100:+5.2f}pp, |Δ| {head_spread[head]*100:.2f}pp")
        spread_total = sum(head_spread.values())
        print(f"  sum |Δ| across heads = {spread_total*100:.2f}pp")
        spec_pass = spread_total >= 0.05
        print(f"  Asymmetric ablation (sum ≥ 5pp): "
              f"{'PASS' if spec_pass else 'FAIL'}")
    else:
        spec_pass = False

    print("\n" + "=" * 60)
    overall = "PASS" if (coupling_pass and diff_pass and spec_pass) else "FAIL"
    print(f"OVERALL Phase 0 (mechanism + differentiation + asymmetry): {overall}")
    print("=" * 60)


if __name__ == "__main__":
    main()
