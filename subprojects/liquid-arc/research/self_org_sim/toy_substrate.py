"""Self-organizing substrate — minimal simulation per math_spec.md.

Round 2: adds Rule E (similarity SOM), multi-exposure protocol, better
numerical stability, and a tree encoding where parent-child share a coordinate.

Run:
    python toy_substrate.py
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# Inputs (true structure G*)
# -----------------------------------------------------------------------------

def make_chain(N: int = 20, d: int = 6, rng=None):
    rng = rng or np.random.default_rng(0)
    h = rng.normal(size=(N, d)) * 0.05
    pos = np.arange(N) / N
    h[:, 0] = np.cos(2 * np.pi * pos)
    h[:, 1] = np.sin(2 * np.pi * pos)
    G_star = np.zeros((N, N))
    for i in range(N - 1):
        G_star[i, i + 1] = 1.0
        G_star[i + 1, i] = 1.0
    return h, G_star


def make_ring(N: int = 20, d: int = 6, rng=None):
    rng = rng or np.random.default_rng(1)
    h = rng.normal(size=(N, d)) * 0.05
    pos = np.arange(N) / N
    h[:, 0] = np.cos(2 * np.pi * pos)
    h[:, 1] = np.sin(2 * np.pi * pos)
    G_star = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        G_star[i, j] = 1.0
        G_star[j, i] = 1.0
    return h, G_star


def make_tree(depth: int = 4, d: int = 6, rng=None):
    """Random-walk additive encoding: child = parent + small per-edge ε.
    Parent-child distance = |ε|.  Sibling distance = sqrt(2)|ε|.
    Discriminable by sharp similarity rules.
    """
    rng = rng or np.random.default_rng(2)
    N = (2 ** depth) - 1
    parents = [-1] + [(i - 1) // 2 for i in range(1, N)]
    h = np.zeros((N, d))
    h[0] = rng.normal(size=d) * 0.5
    for i in range(1, N):
        eps = rng.normal(size=d) * 0.3
        h[i] = h[parents[i]] + eps
    G_star = np.zeros((N, N))
    for i in range(1, N):
        p = parents[i]
        G_star[i, p] = 1.0
        G_star[p, i] = 1.0
    return h, G_star


# -----------------------------------------------------------------------------
# Multi-exposure protocol: re-clamp h to input every reclamp_every steps,
# accumulate G across exposures.
# -----------------------------------------------------------------------------

def run_with_protocol(rule_fn, h0, T=4000, reclamp_every=20, **kwargs):
    """Run a rule that takes (h, G, **kwargs) and returns (h, G) per step.
    Re-clamps h to h0 every reclamp_every steps so the substrate sees the
    input multiple times — required for Hopfield/SOM-style learning.
    """
    N, d = h0.shape
    rng = np.random.default_rng(kwargs.get('seed', 0))
    h = h0.copy()
    G = rng.normal(size=(N, N)) * 0.02
    np.fill_diagonal(G, 0.0)
    state = {'h': h, 'G': G, 'aux': {}}
    for t in range(T):
        if t % reclamp_every == 0:
            state['h'] = h0.copy()
        rule_fn(state, **kwargs)
        np.fill_diagonal(state['G'], 0.0)
    return state['G'], state['h']


# -----------------------------------------------------------------------------
# Rules (each modifies state in place)
# -----------------------------------------------------------------------------

def rule_predictive(state, eta_g=0.02, lam_g=0.005, dt=0.02, seed=10, **_):
    h, G = state['h'], state['G']
    pred = G @ h
    eps = h - pred
    dh = -eps + G.T @ eps
    state['h'] = np.clip(h + dt * dh, -5.0, 5.0)
    outer = eps @ h.T
    G_new = G + dt * (eta_g * outer - lam_g * G)
    state['G'] = np.clip(G_new, -1.0, 1.0)


def rule_hopfield(state, eta_g=0.02, lam_g=0.005, dt=0.05, seed=11, **_):
    h, G = state['h'], state['G']
    h_new = np.tanh(G @ h) - h
    state['h'] = np.clip(h + dt * h_new, -3.0, 3.0)
    outer = h @ h.T
    G_new = G + dt * (eta_g * outer - lam_g * G)
    state['G'] = np.clip(G_new, -1.0, 1.0)


def rule_reaction_diffusion(state, eta_g=0.02, lam_g=0.01, gamma=0.05,
                              dt=0.05, seed=12, **_):
    h, G = state['h'], state['G']
    if 'm' not in state['aux']:
        state['aux']['m'] = np.tanh(np.linalg.norm(h, axis=1))
    m = state['aux']['m']
    dm = -gamma * m + G @ m
    m = np.clip(m + dt * dm, -1.0, 1.0)
    # Re-anchor to current h activity
    m = 0.9 * m + 0.1 * np.tanh(np.linalg.norm(h, axis=1))
    state['aux']['m'] = m
    outer = np.outer(m, m)
    G_new = G + dt * (eta_g * outer - lam_g * G)
    state['G'] = np.clip(G_new, -1.0, 1.0)


def rule_free_energy(state, eta_g=0.05, lam_g=0.01, dt=0.05, seed=13, **_):
    h, G = state['h'], state['G']
    N, d = h.shape
    if 'mu' not in state['aux']:
        state['aux']['mu'] = h[None, :, :].repeat(N, axis=0)
        state['aux']['pi'] = np.ones((N, N)) * 0.3
        np.fill_diagonal(state['aux']['pi'], 0.0)
    mu = state['aux']['mu']
    pi = state['aux']['pi']
    diff = h[None, :, :] - mu
    sq = (diff * diff).sum(axis=-1)
    dmu = -pi[..., None] * (mu - h[None, :, :])
    state['aux']['mu'] = mu + dt * dmu
    dpi = eta_g * (1.0 - pi * sq) - lam_g * pi
    state['aux']['pi'] = np.clip(pi + dt * dpi, 0.0, 5.0)
    np.fill_diagonal(state['aux']['pi'], 0.0)
    state['G'] = state['aux']['pi'].copy()


def rule_similarity(state, eta_g=0.05, lam_g=0.02, sigma=0.25, dt=0.05,
                       seed=14, **_):
    """Rule E. Strengthen connection by inverse-distance similarity:
        dg_ij/dt = eta * exp(-||h_i - h_j||^2 / sigma) - lam * g_ij
    The substrate doesn't 'know' that distance encodes structure — it just
    locally observes activity similarity. If structure is similarity-encoded
    in input, this should recover it.
    """
    h, G = state['h'], state['G']
    diff = h[:, None, :] - h[None, :, :]      # [N, N, d]
    dist_sq = (diff * diff).sum(axis=-1)
    sim = np.exp(-dist_sq / sigma)
    np.fill_diagonal(sim, 0.0)
    G_new = G + dt * (eta_g * sim - lam_g * G)
    state['G'] = np.clip(G_new, -1.0, 1.0)


def rule_competitive(state, eta_g=0.05, lam_g=0.02, sigma=0.25, k=3,
                       dt=0.05, seed=15, **_):
    """Rule F. Competitive Hebbian — only the top-k most similar neighbors
    get reinforced (winner-take-most). Closer to SOM / lateral inhibition.
    """
    h, G = state['h'], state['G']
    N = h.shape[0]
    diff = h[:, None, :] - h[None, :, :]
    dist_sq = (diff * diff).sum(axis=-1)
    np.fill_diagonal(dist_sq, np.inf)         # exclude self
    sim = np.exp(-dist_sq / sigma)
    # Top-k mask per row
    top_k_idx = np.argpartition(-sim, k, axis=1)[:, :k]
    mask = np.zeros_like(sim)
    rows = np.arange(N).repeat(k)
    cols = top_k_idx.flatten()
    mask[rows, cols] = 1.0
    target = sim * mask
    G_new = G + dt * (eta_g * target - lam_g * G)
    state['G'] = np.clip(G_new, -1.0, 1.0)


def rule_mutual_knn(state, eta_g=0.05, lam_g=0.02, sigma=0.3, k=3,
                       dt=0.05, seed=17, **_):
    """Rule H. Mutual k-NN: connect (i,j) only if i has j in its top-k AND
    j has i in its top-k. Sharper than competitive top-k because it requires
    reciprocity, killing one-sided "best friend" attachments.
    """
    h, G = state['h'], state['G']
    N = h.shape[0]
    diff = h[:, None, :] - h[None, :, :]
    dist_sq = (diff * diff).sum(axis=-1)
    np.fill_diagonal(dist_sq, np.inf)
    sim = np.exp(-dist_sq / sigma)
    top_k_idx = np.argpartition(-sim, k, axis=1)[:, :k]
    in_top_k = np.zeros((N, N), dtype=bool)
    rows = np.arange(N).repeat(k)
    in_top_k[rows, top_k_idx.flatten()] = True
    mutual = in_top_k & in_top_k.T
    target = sim * mutual.astype(sim.dtype)
    G_new = G + dt * (eta_g * target - lam_g * G)
    state['G'] = np.clip(G_new, -1.0, 1.0)


def rule_distance_attractor(state, eta_g=0.05, lam_g=0.02, dt=0.05,
                              alpha=0.1, seed=16, **_):
    """Rule G. Distance + activity-correlation hybrid.
    Combines similarity (rule E) with neighbor-mediated state pull
    (so units that are connected become MORE similar over time —
    closing the loop between connectivity and state).
    """
    h, G = state['h'], state['G']
    diff = h[:, None, :] - h[None, :, :]
    dist_sq = (diff * diff).sum(axis=-1)
    sim = np.exp(-dist_sq / 0.25)
    np.fill_diagonal(sim, 0.0)
    G_new = G + dt * (eta_g * sim - lam_g * G)
    G_new = np.clip(G_new, -1.0, 1.0)
    # Now h relaxes toward weighted-neighbor-mean (state co-evolves with G)
    in_w = G_new.sum(axis=1, keepdims=True) + 1e-6
    h_pull = (G_new @ h) / in_w
    state['h'] = h + dt * alpha * (h_pull - h)
    state['G'] = G_new


# -----------------------------------------------------------------------------
# AUROC
# -----------------------------------------------------------------------------

def auroc(scores, labels):
    """Standard Mann-Whitney U / AUROC with average-rank tie handling.
    Sort ASCENDING so rank 1 = lowest score; high-scored positives → high
    rank → high sum_ranks_pos → AUROC near 1.0.
    """
    s = np.asarray(scores, dtype=np.float64).flatten()
    y = np.asarray(labels, dtype=np.float64).flatten()
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind='stable')   # ascending
    s_sorted = s[order]
    y_sorted = y[order]
    ranks = np.empty_like(s_sorted)
    i = 0
    while i < len(s_sorted):
        j = i
        while j < len(s_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    sum_ranks_pos = ranks[y_sorted > 0].sum()
    return (sum_ranks_pos - pos * (pos + 1) / 2) / (pos * neg)


def evaluate(G_final, G_star):
    N = G_star.shape[0]
    mask = ~np.eye(N, dtype=bool)
    return auroc(np.abs(G_final[mask]), G_star[mask])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    inputs = {
        "chain": make_chain,
        "ring":  make_ring,
        "tree":  make_tree,
    }
    rules = {
        "A: pred-coding":         rule_predictive,
        "B: hopfield":            rule_hopfield,
        "C: rxn-diffusion":       rule_reaction_diffusion,
        "D: free-energy":         rule_free_energy,
        "E: similarity-SOM":      rule_similarity,
        "F: competitive top-k":   rule_competitive,
        "G: distance+attractor":  rule_distance_attractor,
        "H: mutual k-NN":         rule_mutual_knn,
    }

    print(f"{'rule':<28s} | {'chain':>7s} | {'ring':>7s} | {'tree':>7s} | mean")
    print("-" * 65)

    results = {}
    for rname, rfn in rules.items():
        row = {}
        for iname, ifn in inputs.items():
            h0, G_star = ifn()
            G_final, _ = run_with_protocol(rfn, h0, T=4000, reclamp_every=20)
            score = evaluate(G_final, G_star)
            row[iname] = score
        results[rname] = row
        mean = np.mean(list(row.values()))
        print(f"{rname:<28s} | {row['chain']:7.3f} | {row['ring']:7.3f} | "
              f"{row['tree']:7.3f} | {mean:.3f}")

    print()
    print("Pass criterion: all three inputs >= 0.85, single hyperparam set.")
    passed = [n for n, r in results.items() if min(r.values()) >= 0.85]
    if passed:
        print(f"PASS: {passed}")
    else:
        print("NO RULE PASSES")
        for iname in inputs:
            best = max(results.items(), key=lambda kv: kv[1][iname])
            print(f"  best on {iname:>5s}: {best[0]} = {best[1][iname]:.3f}")


if __name__ == "__main__":
    main()
