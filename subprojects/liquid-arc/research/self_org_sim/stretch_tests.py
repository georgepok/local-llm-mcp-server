"""Stretch tests for the self-organizing rule that passed the basic spec.

Three diagnostics:
  1. Re-organization: train on input A, then switch input to B without
     resetting G. Does G re-form to match B's structure?
  2. Noise robustness: add Gaussian noise to input. Does the score degrade
     gracefully?
  3. Hierarchy: input is two chains joined at a node. Does G recover both
     chains AND the join?

Uses Rule E (similarity-SOM) — the simpler of the two that passed.
"""

from __future__ import annotations

import numpy as np

from toy_substrate import (
    make_chain, make_ring, make_tree,
    rule_similarity, run_with_protocol, evaluate, auroc,
)


# -----------------------------------------------------------------------------
# Test 1: Re-organization (input switch mid-run)
# -----------------------------------------------------------------------------

def test_reorganization():
    """Train on chain (h_0..h_19 = position-encoded). Then SHUFFLE the input
    so position k now has h_{perm[k]}. The "true" G_star permutes accordingly.
    Substrate's old G points to OLD adjacencies — must re-form to NEW ones.
    """
    print("\n=== Test 1: Re-organization (input permutation) ===")
    N = 20
    h_chain, G_chain = make_chain(N=N)

    rng = np.random.default_rng(42)
    G = rng.normal(size=(N, N)) * 0.02
    np.fill_diagonal(G, 0.0)
    state = {'h': h_chain.copy(), 'G': G, 'aux': {}}
    T = 4000
    reclamp = 20
    for t in range(T):
        if t % reclamp == 0:
            state['h'] = h_chain.copy()
        rule_similarity(state)
        np.fill_diagonal(state['G'], 0.0)
    score_orig = evaluate(state['G'], G_chain)
    print(f"  Phase A (chain, T={T}): AUROC vs original chain = {score_orig:.3f}")

    # Permute node identities — node k now receives h_{perm[k]}
    perm = rng.permutation(N)
    h_permuted = h_chain[perm]
    # New true graph: edges (perm^-1[i], perm^-1[i+1])
    G_star_new = np.zeros((N, N))
    inv = np.argsort(perm)
    for i in range(N - 1):
        a, b = inv[i], inv[i + 1]
        G_star_new[a, b] = 1.0
        G_star_new[b, a] = 1.0

    score_immediate = evaluate(state['G'], G_star_new)
    print(f"  Immediately after permutation:    AUROC vs new chain = "
          f"{score_immediate:.3f}  (old G — should be near random)")

    # Phase B: train on permuted input, keep G state
    for t in range(T):
        if t % reclamp == 0:
            state['h'] = h_permuted.copy()
        rule_similarity(state)
        np.fill_diagonal(state['G'], 0.0)
    score_new = evaluate(state['G'], G_star_new)
    score_old = evaluate(state['G'], G_chain)
    print(f"  After {T} permuted steps: AUROC vs new chain = {score_new:.3f}, "
          f"AUROC vs old chain = {score_old:.3f}")
    if score_new >= 0.85 and score_new > score_immediate + 0.2:
        print("  PASS — substrate re-organized to new structure")
    else:
        print("  FAIL — substrate did not re-organize")


# -----------------------------------------------------------------------------
# Test 2: Noise robustness
# -----------------------------------------------------------------------------

def test_noise_robustness():
    """Add Gaussian noise to input each re-clamp, measure score degradation."""
    print("\n=== Test 2: Noise robustness ===")
    h0, G_star = make_ring(N=20)

    for noise in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        rng = np.random.default_rng(100)
        G = rng.normal(size=(20, 20)) * 0.02
        np.fill_diagonal(G, 0.0)
        state = {'h': h0.copy(), 'G': G, 'aux': {}}
        T = 4000
        reclamp = 20
        for t in range(T):
            if t % reclamp == 0:
                noisy = h0 + rng.normal(size=h0.shape) * noise
                state['h'] = noisy
            rule_similarity(state)
            np.fill_diagonal(state['G'], 0.0)
        score = evaluate(state['G'], G_star)
        print(f"  noise σ={noise:.2f}:  AUROC={score:.3f}")


# -----------------------------------------------------------------------------
# Test 3: Hierarchical (two chains joined at a node)
# -----------------------------------------------------------------------------

def make_hierarchical(N1: int = 10, N2: int = 10, d: int = 6, rng=None):
    """Two chains share node 0.  Total N = N1 + N2 - 1."""
    rng = rng or np.random.default_rng(7)
    N = N1 + N2 - 1
    # Build the path graph as one extended structure
    h = np.zeros((N, d))
    h[0] = rng.normal(size=d) * 0.3
    # Chain 1: nodes 0..N1-1
    for i in range(1, N1):
        eps = rng.normal(size=d) * 0.3
        h[i] = h[i - 1] + eps
    # Chain 2: nodes 0, N1, N1+1, ..., N1+N2-2
    prev = 0
    for k, i in enumerate(range(N1, N1 + N2 - 1)):
        eps = rng.normal(size=d) * 0.3
        h[i] = h[prev] + eps
        prev = i
    G_star = np.zeros((N, N))
    # Chain 1 edges
    for i in range(N1 - 1):
        G_star[i, i + 1] = 1.0
        G_star[i + 1, i] = 1.0
    # Chain 2 edges (0 — N1 — N1+1 — ... — N1+N2-2)
    G_star[0, N1] = 1.0
    G_star[N1, 0] = 1.0
    for i in range(N1, N1 + N2 - 2):
        G_star[i, i + 1] = 1.0
        G_star[i + 1, i] = 1.0
    return h, G_star


def test_hierarchy():
    print("\n=== Test 3: Hierarchical (two chains joined at node 0) ===")
    h0, G_star = make_hierarchical(N1=10, N2=10)
    rng = np.random.default_rng(200)
    G = rng.normal(size=(h0.shape[0], h0.shape[0])) * 0.02
    np.fill_diagonal(G, 0.0)
    state = {'h': h0, 'G': G, 'aux': {}}
    T = 4000
    reclamp = 20
    for t in range(T):
        if t % reclamp == 0:
            state['h'] = h0.copy()
        rule_similarity(state)
        np.fill_diagonal(state['G'], 0.0)
    score = evaluate(state['G'], G_star)
    print(f"  AUROC={score:.3f}  (need >= 0.85 to claim recovery)")
    # Specifically check the join node: did it recover both neighbors?
    join_neighbors_in_truth = np.where(G_star[0] > 0)[0]
    join_neighbor_strengths = state['G'][0, join_neighbors_in_truth]
    print(f"  Join-node truth neighbors: {join_neighbors_in_truth.tolist()}")
    print(f"  Their G strengths: {[f'{x:.3f}' for x in join_neighbor_strengths]}")


if __name__ == "__main__":
    test_reorganization()
    test_noise_robustness()
    test_hierarchy()
