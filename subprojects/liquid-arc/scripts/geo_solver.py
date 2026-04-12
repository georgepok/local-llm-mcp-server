"""Geodesic ODE solver — head-to-head with the flat-4B bench.

Demonstrates the architectural primitive (heat kernel on learned metric)
computing shortest paths *as its native operation*, on the exact same
graphs the LLM bench evaluated.

Two solvers, both single-forward-pass (no scratch space):

  heat_kernel:  Continuous diffusion. Initialize node states as one-hot
                at source; integrate dh/dt = -L_w · h for T steps using
                the weighted graph Laplacian L_w. Arrival time at each
                node ∝ geodesic distance from source. Reconstruct path
                via gradient descent on the arrival-time field.

  min_plus:     LiquidARC-style discrete ODE. Initialize D[i,j] = edge
                weight (or +inf). At each step: D[i,j] ← min_k(D[i,k] +
                D[k,j]). After N >= diameter steps, D is exact shortest
                path. Reconstruct path by greedy backtrace.

Both are SINGLE FORWARD-PASS (no scratch tokens, no autoregressive
generation). Compares directly to the flat-4B direct-mode bench.

Run:
  python3 scripts/geo_solver.py --solver min_plus --n_graphs 30 \
      --sizes 10,14,18,22 --cycle_densities 0.4,0.8

Same args/seeds as bench_geodesic.py — apples-to-apples comparison.
"""

import argparse, json, math, time
import numpy as np

from bench_geodesic import (
    make_graph, render_natural, render_adversarial, score, Graph,
)


# ─────────────────────────────────────────────────────────────────────
# Graph → matrices
# ─────────────────────────────────────────────────────────────────────

def build_matrices(g: Graph):
    """Returns:
        nodes:  list of node labels (index ↔ label)
        idx:    dict label → int
        W:      weighted adjacency [N,N], 0 on missing edges
        L:      graph Laplacian (D - W) where D = diag(row_sum(W))
        D_init: distance matrix [N,N], edge_weight on edges, +inf elsewhere
    """
    nodes = sorted(g.nodes)
    idx = {x: i for i, x in enumerate(nodes)}
    N = len(nodes)
    W = np.zeros((N, N), dtype=np.float64)
    D_init = np.full((N, N), np.inf, dtype=np.float64)
    for u, v, w in g.edges:
        i, j = idx[u], idx[v]
        W[i, j] = w
        W[j, i] = w
        D_init[i, j] = w
        D_init[j, i] = w
    np.fill_diagonal(D_init, 0.0)
    deg = W.sum(axis=1)
    L = np.diag(deg) - W
    return nodes, idx, W, L, D_init


# ─────────────────────────────────────────────────────────────────────
# Solver A — Heat kernel ODE
# ─────────────────────────────────────────────────────────────────────

def solve_heat_kernel(g: Graph, n_steps: int = 200, dt: float = None):
    """Continuous diffusion solver.

    Initialize source state as a unit impulse at node s. Integrate
    dh/dt = -L_w · h for n_steps using forward Euler. Each node's
    *time of first significant arrival* (heat reaching threshold)
    is proportional to geodesic distance from source.

    Reconstruct path: from t, walk to neighbor with smallest
    (arrival_time + edge_weight_to_t) — i.e. the predecessor along the
    shortest-path tree from s.
    """
    nodes, idx, W, L, D_init = build_matrices(g)
    N = len(nodes)
    if dt is None:
        # CFL stability: dt < 2 / max_eig(L). Spectral radius ≤ max(deg).
        dt = 0.5 / max(W.sum(axis=1).max(), 1.0)

    # Initial state: unit impulse at source
    h = np.zeros(N)
    h[idx[g.s]] = 1.0

    # Track first-arrival "time" at each node (when h crosses threshold)
    arrival = np.full(N, np.inf)
    arrival[idx[g.s]] = 0.0
    threshold = 1e-6

    for step in range(n_steps):
        h = h + dt * (-L @ h)  # forward Euler diffusion
        h = np.clip(h, 0.0, None)  # physical: no negative mass
        t_now = (step + 1) * dt
        newly = (arrival == np.inf) & (h > threshold)
        arrival[newly] = t_now

    # Path reconstruction: from t backwards using shortest-tree property
    # parent(v) = argmin over neighbors u of {arrival[u] + edge_weight(u,v)}
    t_idx = idx[g.t]
    s_idx = idx[g.s]
    if not np.isfinite(arrival[t_idx]):
        return None, None, None
    parent = np.full(N, -1, dtype=int)
    for v in range(N):
        if v == s_idx:
            continue
        best_u, best_score = -1, np.inf
        for u in range(N):
            if W[u, v] > 0 and np.isfinite(arrival[u]):
                s = arrival[u] + W[u, v] * dt  # surrogate for "u was on path"
                if s < best_score:
                    best_score, best_u = s, u
        parent[v] = best_u

    # Walk from t back to s
    path_idx = [t_idx]
    cur = t_idx
    seen = {t_idx}
    while cur != s_idx:
        nxt = parent[cur]
        if nxt < 0 or nxt in seen:
            return None, None, None
        path_idx.append(nxt)
        seen.add(nxt)
        cur = nxt
        if len(path_idx) > N:
            return None, None, None
    path_idx.reverse()
    path = [nodes[i] for i in path_idx]
    cost = sum(W[path_idx[i], path_idx[i + 1]] for i in range(len(path_idx) - 1))
    return path, int(cost), n_steps


# ─────────────────────────────────────────────────────────────────────
# Solver B — Min-plus ODE (LiquidARC-style discrete ODE)
# ─────────────────────────────────────────────────────────────────────

def solve_min_plus(g: Graph, n_steps: int = None):
    """Discrete ODE on the metric tensor D.

    Each step: D ← min(D, D[:, k:k+1] + D[k:k+1, :]) over all k
    (equivalent to one round of Floyd-Warshall as matrix update).

    After ⌈log₂(N)⌉ doubling steps OR N relaxation rounds, D = true
    all-pairs shortest path. This is one "Euler step" per ODE iteration,
    fully parallel — exactly the same shape as LiquidARC's ContinuousDynamics
    where each step refines a relational matrix via gather operations.
    """
    nodes, idx, W, L, D_init = build_matrices(g)
    N = len(nodes)
    if n_steps is None:
        n_steps = max(1, int(math.ceil(math.log2(max(2, N)))))  # repeated squaring is enough

    D = D_init.copy()
    # Repeated min-plus squaring: D ← D ⊗ D where ⊗ is (min, +)
    # Each squaring doubles the path length explored; ⌈log₂(N)⌉ steps suffice.
    for step in range(n_steps):
        # D_new[i,j] = min_k (D[i,k] + D[k,j])
        D_expanded = D[:, :, None] + D[None, :, :]   # [N,N,N]
        D = D_expanded.min(axis=1)

    # Path reconstruction by greedy: at each node v, pick neighbor u minimizing
    #   W[v,u] + D[u, t]
    s_idx = idx[g.s]
    t_idx = idx[g.t]
    if not np.isfinite(D[s_idx, t_idx]):
        return None, None, None

    path_idx = [s_idx]
    cur = s_idx
    visited = {s_idx}
    while cur != t_idx:
        best_u, best_score = -1, np.inf
        for u in range(N):
            if W[cur, u] > 0 and u not in visited:
                s = W[cur, u] + D[u, t_idx]
                if s < best_score:
                    best_score, best_u = s, u
        if best_u < 0:
            return None, None, None
        path_idx.append(best_u)
        visited.add(best_u)
        cur = best_u
        if len(path_idx) > N:
            return None, None, None

    path = [nodes[i] for i in path_idx]
    cost = sum(W[path_idx[i], path_idx[i + 1]] for i in range(len(path_idx) - 1))
    return path, int(cost), n_steps


SOLVERS = {
    'heat_kernel': solve_heat_kernel,
    'min_plus': solve_min_plus,
}


def render_response(path, cost):
    if path is None or cost is None:
        return ""
    return f"PATH: {' -> '.join(path)}  COST: {cost}"


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--solver', choices=list(SOLVERS.keys()), default='min_plus')
    p.add_argument('--n_graphs', type=int, default=30)
    p.add_argument('--sizes', default='10,14,18,22')
    p.add_argument('--cycle_densities', default='0.4,0.8')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='geo_solver.json')
    p.add_argument('--n_steps', type=int, default=None,
                   help='heat_kernel: integration steps (200); '
                        'min_plus: squaring rounds (auto = ceil(log2 N))')
    args = p.parse_args()
    sizes = [int(x) for x in args.sizes.split(',')]
    densities = [float(x) for x in args.cycle_densities.split(',')]
    solver_fn = SOLVERS[args.solver]

    print("=" * 70)
    print(f"GEO SOLVER ({args.solver}) — single forward pass on explicit graph")
    print("=" * 70)
    print(f"  sizes={sizes}  densities={densities}  n_graphs/cell={args.n_graphs}")

    cells = []
    seed_counter = args.seed
    grand_correct = 0
    grand_total = 0
    grand_time = 0.0

    for n in sizes:
        for density in densities:
            extra = max(1, int(round(density * n)))
            graphs = []
            attempts = 0
            while len(graphs) < args.n_graphs and attempts < args.n_graphs * 10:
                g = make_graph(n, extra, seed=seed_counter)
                seed_counter += 1
                attempts += 1
                if g is not None:
                    graphs.append(g)

            print(f"\n--- cell n={n} extra_edges={extra} ({len(graphs)} graphs) ---")

            results = []
            t_cell = 0.0
            for gi, g in enumerate(graphs):
                t0 = time.time()
                kwargs = {}
                if args.n_steps is not None:
                    kwargs['n_steps'] = args.n_steps
                path, cost, steps_used = solver_fn(g, **kwargs)
                t_cell += time.time() - t0

                response = render_response(path, cost)
                s = score(g, response)
                s['steps'] = steps_used
                results.append(s)

            from collections import Counter
            c = Counter(r['label'] for r in results)
            tot = max(1, len(results))
            tally = {k: c.get(k, 0) / tot for k in
                     ['CORRECT', 'DECOY', 'SUBOPTIMAL', 'HALLUCINATED', 'REVISIT', 'UNPARSED']}

            grand_correct += c.get('CORRECT', 0)
            grand_total += tot
            grand_time += t_cell

            cell = {'n': n, 'extra_edges': extra, 'n_graphs': len(graphs),
                    'tally': tally, 'mean_latency_ms': t_cell / tot * 1000}
            cells.append(cell)

            print(f"  CORRECT={tally['CORRECT']:>5.0%}  "
                  f"SUBOPTIMAL={tally['SUBOPTIMAL']:>5.0%}  "
                  f"HALLUC={tally['HALLUCINATED']:>4.0%}  "
                  f"UNPARSED={tally['UNPARSED']:>4.0%}  "
                  f"|  {t_cell / tot * 1000:>6.1f} ms/graph")

    with open(args.out, 'w') as f:
        json.dump({'solver': args.solver, 'cells': cells, 'config': vars(args)},
                  f, indent=2)
    print(f"\nSaved → {args.out}")

    print("\n" + "=" * 70)
    print(f"OVERALL ({args.solver}): {grand_correct}/{grand_total} = "
          f"{grand_correct / max(1, grand_total):.0%}  "
          f"|  mean {grand_time / max(1, grand_total) * 1000:.1f} ms/graph")
    print("=" * 70)


if __name__ == '__main__':
    main()
