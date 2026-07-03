# Self-Organizing Substrate — Mathematical Specification

## Goal

Find a *minimal* dynamical system on a generic substrate that, given a structured
input, **spontaneously develops a connectivity matrix that reflects the input's
true structure** — without any designer specifying which structure.

If we can show this on a 50-unit toy substrate for three different input
structure classes (chain, ring, branching tree), we have an existence proof.
If we cannot, no LiquidARC engineering will rescue the idea.

## Substrate

A field of N units indexed `i = 1..N`. Each unit has:

- **State** `h_i ∈ R^d` (small d, e.g. d = 4-8 for the toy)
- **Outgoing connections** `g_ij ∈ R` to every other unit (initially random small)
- **Local memory** `m_i` (per-unit running statistics — error history, activity history)

The connectivity matrix `G = (g_ij)` is the *structure*. We want G to come to
reflect the input's structure under repeated exposure.

## Locality

A rule is *local* if updating `(h_i, g_ij, m_i)` requires only:

- Unit i's own state `h_i`
- States of units j with `g_ij > τ` for some threshold τ (current "neighbors")
- Unit i's own memory `m_i`

Globally-pooled signals (e.g., total system energy) are *not* allowed.
Backprop through global loss is *not* allowed.

## Local Objective

Each unit minimizes its own **prediction error** of the input it receives.
This is the predictive-coding family of rules (Rao-Ballard, Friston).

For unit i, the predicted state is a function of its current connections:

    ĥ_i = Σ_j g_ij · h_j     (weighted sum of "neighbors" by current g)

The prediction error is

    ε_i = h_i − ĥ_i = h_i − Σ_j g_ij · h_j

Each unit's local objective is `||ε_i||²`, minimized over (h_i, g_ij).

## Candidate Rule Families

### A. Predictive Coding (Rao-Ballard)

State update (gradient flow on local error):

    dh_i/dt = − ε_i  +  Σ_j g_ji · ε_j     (own error + reflected from those it predicts)

Connection update (anti-Hebbian on errors, scaled by neighbor activity):

    dg_ij/dt = η · ε_i · h_j   −  λ · g_ij     (η > 0, λ > 0)

Reduces to: "strengthen connection j → i when h_j explains away ε_i; decay otherwise."

### B. Hopfield-style Hebbian

    dh_i/dt = tanh(Σ_j g_ij · h_j) − h_i             (settling toward attractor)
    dg_ij/dt = η · h_i · h_j − λ · g_ij              (Hebbian + decay)

Strengthens g where i and j co-activate. Pure correlation, no error signal.

### C. Reaction-Diffusion (Turing-style)

Each unit produces a "morphogen" m_i = σ(||h_i||²).
Morphogen diffuses through current connections, decays in time.

    dm_i/dt = − γ · m_i  +  Σ_j g_ij · m_j
    dh_i/dt = f(h_i) + ∇_g · h            (some local activity rule)
    dg_ij/dt = η · m_i · m_j − λ · g_ij    (connections strengthen where both have high morphogen)

Self-similar inputs → similar morphogens → cluster formation.

### D. Free Energy (Friston-style, simplified)

Each unit has a small Gaussian belief over neighbor states:

    p_i(h_j) = N(μ_ij, σ_ij²)

Free energy per unit:

    F_i = Σ_j (1/(2 σ_ij²)) · (h_j − μ_ij)²  +  log σ_ij

State, mean, variance, and connections all updated by gradient flow on F_i.
Connections g_ij ∝ 1/σ_ij² (precision = how confidently unit i predicts unit j).

## Inputs as Initial States

We feed input by clamping initial `h` and letting dynamics run. Three test inputs:

1. **Chain**:  N = 20.  h_i = onehot(i mod 4) for "tag" + onehot(i) for position.
   True G* = adjacency of path graph `0—1—2—...—19`.
2. **Ring**:   N = 20.  same encoding, but unit N-1 connected back to 0.
   True G* = adjacency of cycle.
3. **Tree**:   N = 15.  h_i encodes `parent(i)` and `depth(i)`.
   True G* = adjacency of binary tree.

For each input, the substrate is initialized with random small G ∈ [-0.1, 0.1].
Dynamics runs for T steps. Final G is compared to G*.

## Emergence Criterion

Let `G_final` be the substrate's connection matrix after T steps. Compare to
the input's true adjacency `G*`.

Define **structural recovery score**:

    S = AUROC(g_ij_final ∝ G*_ij)

i.e., treating G* as binary labels and g_ij_final as scores, how well does the
final connectivity rank true edges above non-edges? S = 1.0 = perfect, S = 0.5 =
random.

A rule passes if **S ≥ 0.85 across all 3 input types**, with no per-input
hyperparameter tuning. (If a rule needs different (η, λ) for chain vs tree, it's
not really self-organizing — the designer is encoding the structure.)

## Pass / Fail Decision

If at least one of A/B/C/D meets S ≥ 0.85 on all 3 inputs with a single set of
hyperparameters, **proceed to LiquidARC port**.

If none do, **stop**. Either:
- The rule families above are insufficient (try others, or accept it's harder)
- The locality constraint is too strict (allow global modulation)
- The substrate must include explicit memory dynamics (recurrence beyond state)

This decision criterion is the whole point of doing the simulation before
the implementation. We get a binary, fast answer to "is this idea worth
building."

## Stretch Tests (only if pass)

- **Re-organization**: train on chain, then switch input to ring without
  resetting G. Does it re-form? (tests plasticity)
- **Hierarchy**: input is two chains joined at a node. Does G recover both
  chains AND the join?
- **Noise robustness**: add 10-30% noise to input. Does S degrade gracefully?

## Compute Budget

All simulations: N ≤ 50, d ≤ 8, T ≤ 5000 steps. Runs in <60 seconds on CPU.
No GPU needed. Notebook-friendly. The whole point is fast iteration on the
math before any architectural commitment.
