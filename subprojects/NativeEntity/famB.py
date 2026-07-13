#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAMILY B -- MUTABLE CATALYTIC HYPERGRAPH
========================================
Part of ENDOGENOUS_CLOSURE_ORIGIN_PROGRAM_V1.

Purpose
-------
Test whether *reconstructive causal closure* can EMERGE (rather than being
pre-wired) through MULTI-COMPONENT interactions.  A reaction is a HYPEREDGE:
it consumes a multiset of substrate fragments, requires a catalyst fragment to
be present (not consumed), consumes local resource, and produces a multiset of
output fragments plus waste.  This is deliberately NOT a one-type -> one-type
producer map, so closure (each type reconstructible from the *others*) is a
genuine multi-component property that must be assembled from parts.

This file is ONLY the substrate + measurement harness.  It makes NO scientific
claim.  It builds the world, guarantees the scientific controls listed below,
runs a self-test (certificate + hidden positive control) and a bounded smoke
experiment, then prints a labeled report.

Scientific controls encoded here (see HARD REQUIREMENTS in the program spec)
---------------------------------------------------------------------------
* NO endogenous reconstructive cycle at t=0  -> `certificate_no_cycle` +
  seed resampling (M2).  Individual fragments may have WEAK PARTIAL functions
  (e.g. a composite producible from food alone) -- those are allowed; only
  reconstructive CYCLES among composites are forbidden at t=0.
* Mutation kernel is CLOSURE-BLIND and RECIPROCITY-BLIND: operators retarget
  recognition/catalysis uniformly at random, add/delete/redirect hyperedges,
  create/delete types.  No operator prefers to close a cycle or create a
  reciprocal pair.  There is no "closure"/"mutualism"/"reciprocal" primitive.
  (The C10 diagnostic condition additionally enables a compound edit that CAN
  create a cycle in one step -- clearly labeled diagnostic, not part of the
  unbiased kernel, and it is never rewarded.)
* Selection is CLOSURE-BLIND: fitness comes only from generic physics
  (resource throughput, material retained through a dilution bottleneck,
  bulk recovery after GENERIC random damage, propagule establishment in a
  fresh resource patch, minus execution cost).  Nothing in fitness references
  cycle count, SCC size, a reconstruction score, or any specific fragment
  identity.
* The RECONSTRUCTION MATRIX is the key measurement: leave-one-type-out
  deletion with a CLONED state + CLONED RNG across an intervention arm and a
  matched control arm; restoration gain G = S1 - S0; selective gain subtracts
  the control drift; R_ij attributes regeneration of deleted type j to the
  other types i that were consumed to remake it.  Full/sparse matrix reported,
  not only the max.

Runnable in a few minutes (pure Python; numpy not required).
"""

import sys
import math
import random
import itertools
from collections import defaultdict

# ---------------------------------------------------------------------------
# Global configuration (kept small + inspectable)
# ---------------------------------------------------------------------------
BASE_LETTERS   = list("abcdefghij")   # 10 base "food" fragment letters
N_COMPOSITES   = 6                    # composite (assembled) fragment types
COMPOSITE_LEVELS = [1, 1, 2, 2, 3, 3] # DAG levels used ONLY at seed construction
WASTE          = "W"                  # waste fragment (fast decay, not food, not "material")

COUNT_CAP      = 40                   # per-type instance cap (bounds dynamics)
INIT_RESOURCE  = 20.0
RESOURCE_INFLUX = 6.0                 # resource added to a region each life step
INFLUX_COST    = 0.2                  # cost of a food-influx reaction
FIRE_CAP       = 2                    # max firings of a given edge per step
DECAY_P        = 0.08                 # per-step decay probability for normal fragments
WASTE_DECAY_P  = 0.5                  # waste decays fast

# life / assay horizons (short -> smoke runs in minutes)
T_LIFE     = 10
T_RECOVER  = 6      # recovery horizon for damage assay AND for reconstruction test
T_PROP     = 5

# fitness weights (all generic physics; closure-blind)
W_TP, W_RET, W_REC, W_EST, W_COST = 1.0, 1.0, 1.5, 1.5, 0.2

# reconstruction analysis
MIN_ABUND       = 3      # a type must have >= this count to be a deletion candidate
CLOSURE_THR     = 0.02   # selective restoration gain threshold to call "closure"

# smoke experiment
N_SEEDS      = 8
N_GENS       = 20
N_IND        = 6         # individuals per population
BOOT_B       = 2000      # bootstrap resamples

CONDITIONS = ["C1", "C2", "C5", "C6", "C10"]
COND_DESC = {
    "C1":  "real-selection (fitness=generic physics, mutable topology)",
    "C2":  "shuffled-fitness control (fitness permuted among regions)",
    "C5":  "fixed-topology control (compositions evolve, hyperedges frozen)",
    "C6":  "mutable-topology-no-selection (random survival)",
    "C10": "one-step-cycle DIAGNOSTIC (compound edit may create a cycle in one edit)",
}
# deterministic per-condition offset (avoid hash randomization)
COND_OFFSET = {"C1": 11, "C2": 22, "C5": 55, "C6": 66, "C10": 100}


# ===========================================================================
# Data structures
# ===========================================================================
class State:
    """A region: a multiset (dict) of fragment instances + a resource pool."""
    __slots__ = ("frag", "resource")

    def __init__(self, frag=None, resource=0.0):
        self.frag = dict(frag) if frag else {}
        self.resource = float(resource)

    def clone(self):
        return State(self.frag, self.resource)

    def get(self, t):
        return self.frag.get(t, 0)


def make_edge(eid, inputs, catalyst, outputs, cost, kind):
    """A reaction HYPEREDGE.

    inputs   : tuple multiset of required substrate types (consumed)
    catalyst : a required-present type (NOT consumed), or None
    outputs  : tuple multiset of produced types
    cost     : resource consumed per firing
    kind     : 'influx' (food, FIXED physics) or 'assembly' (mutable structure)
    """
    return {"id": eid, "inputs": tuple(inputs), "catalyst": catalyst,
            "outputs": tuple(outputs), "cost": float(cost), "kind": kind}


def new_world():
    return {"types": set(), "base": set(BASE_LETTERS), "edges": [], "next_eid": 0}


def add_edge(world, inputs, catalyst, outputs, cost, kind):
    e = make_edge(world["next_eid"], inputs, catalyst, outputs, cost, kind)
    world["next_eid"] += 1
    world["edges"].append(e)
    return e


def clone_world(world):
    w = {"types": set(world["types"]), "base": set(world["base"]),
         "edges": [dict(e) for e in world["edges"]], "next_eid": world["next_eid"]}
    return w


def composites(world):
    """Composite (assembled) types = everything that is not base food or waste."""
    return [t for t in world["types"] if t not in world["base"] and t != WASTE]


def edge_signature(e):
    return (tuple(sorted(e["inputs"])), e["catalyst"], tuple(sorted(e["outputs"])))


def topology_signature(world):
    return frozenset(edge_signature(e) for e in world["edges"] if e["kind"] == "assembly")


# ===========================================================================
# World construction (guaranteed acyclic at t=0)
# ===========================================================================
def make_seed_world(seed):
    """Construct a world whose composite dependency graph is a DAG.

    Composites are placed on strictly increasing levels; an assembly edge for a
    composite may only draw substrate/catalyst from base letters or strictly
    LOWER-level composites.  That makes the composite dependency graph acyclic
    by construction (verified by the certificate; resampled on the rare miss).
    """
    rng = random.Random(seed * 2654435761 & 0x7FFFFFFF)
    for attempt in range(200):
        w = new_world()
        for L in BASE_LETTERS:
            w["types"].add(L)
        w["types"].add(WASTE)
        comp_names = ["X%d" % i for i in range(N_COMPOSITES)]
        levels = {comp_names[i]: COMPOSITE_LEVELS[i] for i in range(N_COMPOSITES)}
        for c in comp_names:
            w["types"].add(c)

        # fixed food-influx edges (resource -> a base letter). NOT mutable.
        for L in BASE_LETTERS:
            add_edge(w, inputs=(), catalyst=None, outputs=(L,),
                     cost=INFLUX_COST, kind="influx")

        # assembly edges (mutable). one or two producers per composite.
        for c in comp_names:
            L = levels[c]
            lower = [x for x in comp_names if levels[x] < L]
            allowed = BASE_LETTERS + lower  # strictly lower -> DAG
            n_prod = 1 + (1 if rng.random() < 0.4 else 0)
            for _ in range(n_prod):
                n_in = rng.choice([1, 2])
                ins = tuple(rng.choice(allowed) for _ in range(n_in))
                cat = rng.choice(allowed + [None, None])  # bias toward None a bit
                cost = round(rng.uniform(0.3, 0.9), 3)
                add_edge(w, inputs=ins, catalyst=cat,
                         outputs=(c, WASTE), cost=cost, kind="assembly")

        w["_levels"] = levels
        if certificate_no_cycle(w):
            return w, rng
    raise RuntimeError("could not construct acyclic seed world (should not happen)")


# ===========================================================================
# Certificate: NO reconstructive cycle among composites
# ===========================================================================
def dependency_edges(world):
    """Composite dependency graph.

    Edge  o -> d  means: some assembly edge outputs composite o, requires
    composite d (as input or catalyst), d != o, AND that edge does not itself
    require o.  Hence when o is deleted, d can be used to remake o.  A directed
    cycle in this graph is a candidate reconstructive cycle (mutual regeneration).
    """
    comps = set(composites(world))
    adj = {c: set() for c in comps}
    for e in world["edges"]:
        if e["kind"] != "assembly":
            continue
        ins = set(e["inputs"])
        deps = set(e["inputs"])
        if e["catalyst"] is not None:
            deps.add(e["catalyst"])
        for o in e["outputs"]:
            if o not in comps:
                continue
            if o in ins:
                continue  # edge requires o -> cannot run when o is deleted
            for d in deps:
                if d in comps and d != o:
                    adj[o].add(d)
    return adj


def _reachable(adj, start):
    seen = set()
    stack = [start]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def reconstructive_sccs(world):
    """Return list of node-sets forming non-trivial (size>=2) reconstructive
    cycles in the composite dependency graph."""
    adj = dependency_edges(world)
    nodes = list(adj.keys())
    reach = {u: _reachable(adj, u) for u in nodes}
    seen = set()
    sccs = []
    for u in nodes:
        if u in seen:
            continue
        grp = {v for v in nodes if v != u and v in reach[u] and u in reach[v]}
        if grp:
            grp = grp | {u}
            if not (grp & seen):
                sccs.append(grp)
                seen |= grp
    return sccs


def certificate_no_cycle(world):
    """True iff the world has NO reconstructive cycle among composites (M2)."""
    return len(reconstructive_sccs(world)) == 0


# ===========================================================================
# Physics dynamics
# ===========================================================================
def _remove_fraction(state, frac, rng):
    """Remove ~frac of each type's instances, stochastically (generic)."""
    for t in list(state.frag.keys()):
        c = state.frag[t]
        if c <= 0:
            continue
        d = int(c * frac)
        if rng.random() < (c * frac - d):
            d += 1
        nc = c - d
        if nc <= 0:
            del state.frag[t]
        else:
            state.frag[t] = nc


def apply_dilution(state, frac, rng):
    _remove_fraction(state, frac, rng)


def apply_generic_damage(state, frac, rng):
    """Delete a random fraction of instances across ALL types.  Type-blind."""
    _remove_fraction(state, frac, rng)


def make_propagule(state, frac, rng):
    """Sample a small propagule into a FRESH resource patch."""
    p = State(resource=INIT_RESOURCE)
    for t, c in state.frag.items():
        if t == WASTE:
            continue
        k = int(c * frac)
        if rng.random() < (c * frac - k):
            k += 1
        if k > 0:
            p.frag[t] = min(k, COUNT_CAP)
    return p


def _can_fire(e, state):
    if state.resource < e["cost"]:
        return False
    need = {}
    for t in e["inputs"]:
        need[t] = need.get(t, 0) + 1
    for t, c in need.items():
        if state.frag.get(t, 0) < c:
            return False
    if e["catalyst"] is not None and state.frag.get(e["catalyst"], 0) < 1:
        return False
    # per-type cap: do not fire if any (non-waste) output already at cap
    for o in e["outputs"]:
        if o != WASTE and state.frag.get(o, 0) >= COUNT_CAP:
            return False
    return True


def _fire(e, state, produced_log=None):
    """Fire e once.  Returns list of (output-type) produced.  Assumes _can_fire."""
    for t in e["inputs"]:
        state.frag[t] = state.frag.get(t, 0) - 1
        if state.frag[t] <= 0:
            del state.frag[t]
    state.resource -= e["cost"]
    outs = []
    for o in e["outputs"]:
        state.frag[o] = state.frag.get(o, 0) + 1
        outs.append(o)
    if produced_log is not None:
        produced_log.append(e)
    return outs


def _decay(state, rng):
    for t in list(state.frag.keys()):
        c = state.frag[t]
        p = WASTE_DECAY_P if t == WASTE else DECAY_P
        d = int(c * p)
        if rng.random() < (c * p - d):
            d += 1
        nc = c - d
        if nc <= 0:
            del state.frag[t]
        else:
            state.frag[t] = nc


def run_life(world, state, rng, steps, produce_log=None):
    """Advance region dynamics.  Returns (assembly_throughput, resource_spent)."""
    throughput = 0
    spent = 0.0
    edges = world["edges"]
    for _ in range(steps):
        state.resource += RESOURCE_INFLUX
        order = edges[:]
        rng.shuffle(order)
        for e in order:
            fires = 0
            while fires < FIRE_CAP and _can_fire(e, state):
                spent += e["cost"]
                _fire(e, state, produce_log)
                if e["kind"] == "assembly":
                    throughput += 1
                fires += 1
        _decay(state, rng)
    return throughput, spent


def total_material(state):
    """Bulk viable material (excludes waste).  Used by the generic physics."""
    return sum(c for t, c in state.frag.items() if t != WASTE)


# ===========================================================================
# CLOSURE-BLIND fitness (generic physics only)
# ===========================================================================
def physics_fitness(world, state, rng):
    """Return (fitness, post-life state).

    Fitness is a weighted sum of GENERIC physics quantities.  It never inspects
    cycle count, SCC size, reconstruction score, or a specific fragment
    identity.  It does not know closure exists.
    """
    s_life = state.clone()
    tp, spent = run_life(world, s_life, rng, T_LIFE)

    # bottleneck: material retained through a dilution event
    b = s_life.clone()
    apply_dilution(b, 0.5, rng)
    retained = total_material(b)

    # reconstruction after GENERIC (type-blind) damage: bulk recovery
    d = s_life.clone()
    apply_generic_damage(d, 0.4, rng)
    after_damage = total_material(d)
    run_life(world, d, rng, T_RECOVER)
    recovery = total_material(d) - after_damage

    # propagule establishment in a fresh resource patch
    p = make_propagule(s_life, 0.2, rng)
    run_life(world, p, rng, T_PROP)
    establishment = total_material(p)

    fit = (W_TP * tp
           + W_RET * retained
           + W_REC * max(0.0, recovery)
           + W_EST * establishment
           - W_COST * spent)
    return fit, s_life


# ===========================================================================
# CLOSURE-BLIND / RECIPROCITY-BLIND mutation kernel
# ===========================================================================
def _assembly_edges(world):
    return [e for e in world["edges"] if e["kind"] == "assembly"]


def mutate_topology(world, rng, allow_compound=False):
    """Apply ONE random topology mutation in place.

    Operators (all uniform; none prefers closure/reciprocity):
      * redirect  : retarget one input slot or the catalyst to a random type
      * add_edge  : add an assembly edge with random inputs/catalyst/output
      * del_edge  : delete a random assembly edge
      * new_type  : create a composite type + a random producer for it
      * del_type  : delete a composite type and edges referencing it
      * jitter    : perturb an edge's cost

    If allow_compound (C10 DIAGNOSTIC only): with small probability apply a
    compound edit that makes two composites require each other (a cycle in one
    edit).  This is NOT a reciprocity-seeking bias -- it is a reachability probe
    and is never rewarded by selection.
    """
    all_types = list(world["types"])
    comps = composites(world)

    if allow_compound and comps and rng.random() < 0.15:
        # DIAGNOSTIC compound edit: pick two composites, make each require the
        # other via one of its producing edges.  (One-step cycle creation.)
        if len(comps) >= 2:
            a, b = rng.sample(comps, 2)
            for (x, y) in ((a, b), (b, a)):
                prods = [e for e in _assembly_edges(world) if x in e["outputs"] and x not in e["inputs"]]
                if prods:
                    e = rng.choice(prods)
                    ins = list(e["inputs"])
                    if ins:
                        ins[rng.randrange(len(ins))] = y
                    else:
                        ins = [y]
                    e["inputs"] = tuple(ins)
            return "compound_diag"

    op = rng.random()
    aedges = _assembly_edges(world)

    if op < 0.42 and aedges:
        # redirect recognition / catalytic compatibility (uniform random target)
        e = rng.choice(aedges)
        target = rng.choice(all_types)
        if rng.random() < 0.6 and e["inputs"]:
            ins = list(e["inputs"])
            ins[rng.randrange(len(ins))] = target
            e["inputs"] = tuple(ins)
        else:
            e["catalyst"] = None if rng.random() < 0.2 else target
        return "redirect"

    if op < 0.60:
        # add an assembly edge, random inputs/catalyst, random composite output
        if comps:
            n_in = rng.choice([1, 2, 3])
            ins = tuple(rng.choice(all_types) for _ in range(n_in))
            cat = None if rng.random() < 0.3 else rng.choice(all_types)
            out = (rng.choice(comps), WASTE)
            add_edge(world, ins, cat, out, round(rng.uniform(0.3, 0.9), 3), "assembly")
        return "add_edge"

    if op < 0.74 and len(aedges) > 1:
        e = rng.choice(aedges)
        world["edges"].remove(e)
        return "del_edge"

    if op < 0.86 and len(comps) < 12:
        # create a new composite type + one random producer
        idx = 0
        while ("X%d" % idx) in world["types"]:
            idx += 1
        c = "X%d" % idx
        world["types"].add(c)
        n_in = rng.choice([1, 2])
        ins = tuple(rng.choice(all_types) for _ in range(n_in))
        cat = None if rng.random() < 0.4 else rng.choice(all_types)
        add_edge(world, ins, cat, (c, WASTE), round(rng.uniform(0.3, 0.9), 3), "assembly")
        return "new_type"

    if op < 0.94 and len(comps) > 2:
        c = rng.choice(comps)
        world["types"].discard(c)
        world["edges"] = [e for e in world["edges"]
                          if c not in e["outputs"] and c not in e["inputs"] and e["catalyst"] != c]
        return "del_type"

    if aedges:
        e = rng.choice(aedges)
        e["cost"] = round(max(0.1, e["cost"] + rng.uniform(-0.3, 0.3)), 3)
        return "jitter"
    return "noop"


# ===========================================================================
# RECONSTRUCTION MATRIX  (the key measurement)
# ===========================================================================
def _comp_vector(state, types):
    return [state.frag.get(t, 0) for t in types]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def reconstruction_analysis(world, state, base_rng):
    """Leave-one-type-out reconstruction with cloned state + cloned RNG.

    For every sufficiently-abundant type j:
      - CLONE the post-life state; delete all instances of j (intervention arm).
      - CLONE the SAME state without deletion (matched control arm).
      - Clone the RNG so both arms see an identical random stream.
      - S0  = cos(pre, immediate-post-deletion);  S1 = cos(pre, post-dynamics).
      - G   = S1 - S0  (restoration gain in the intervention arm).
      - drift = control-arm (S1-S0), i.e. change from dynamics alone.
      - selective_gain(j) = G - drift.
      - via_composite(j)  = j was regenerated by firings that consumed at
        least one OTHER composite (genuine closure route, not food influx).
      - R[i][j] += amount of j produced by firings that consumed/were catalyzed
        by type i (mechanistic causal attribution).

    Returns dict with per-type records, the sparse R matrix, and a closure_score
    (max selective gain over composites that are in a reconstructive cycle AND
    were regenerated via other composites).
    """
    types = sorted(world["types"])
    comps = set(composites(world))
    sccs = reconstructive_sccs(world)
    in_cycle = set().union(*sccs) if sccs else set()

    pre = _comp_vector(state, types)
    records = {}
    R = defaultdict(lambda: defaultdict(float))

    candidates = [j for j in types
                  if j != WASTE and state.frag.get(j, 0) >= MIN_ABUND]

    for j in candidates:
        rng_state = base_rng.getstate()

        # intervention arm: delete j
        inter = state.clone()
        inter.frag.pop(j, None)
        s0 = _cosine(pre, _comp_vector(inter, types))
        r_int = random.Random(); r_int.setstate(rng_state)
        log = []
        run_life(world, inter, r_int, T_RECOVER, produce_log=log)
        s1 = _cosine(pre, _comp_vector(inter, types))
        g_int = s1 - s0

        # matched control arm: identical clone, NO deletion, SAME rng stream
        ctrl = state.clone()
        s0c = _cosine(pre, _comp_vector(ctrl, types))
        r_ctrl = random.Random(); r_ctrl.setstate(rng_state)
        run_life(world, ctrl, r_ctrl, T_RECOVER)
        s1c = _cosine(pre, _comp_vector(ctrl, types))
        drift = s1c - s0c

        base_rng.setstate(rng_state)   # keep master deterministic
        for _ in range(1):             # advance master a fixed amount
            base_rng.random()

        selective = g_int - drift

        # attribution: which types helped regenerate j
        j_regen = 0
        via_comp = False
        for e in log:
            if j in e["outputs"]:
                j_regen += 1
                contribs = set(e["inputs"])
                if e["catalyst"] is not None:
                    contribs.add(e["catalyst"])
                if contribs & comps:
                    via_comp = True
                for i in contribs:
                    R[i][j] += 1.0

        records[j] = {
            "is_composite": j in comps,
            "in_cycle": j in in_cycle,
            "S0": round(s0, 4), "S1": round(s1, 4),
            "G": round(g_int, 4), "drift": round(drift, 4),
            "selective_gain": round(selective, 4),
            "regen_count": j_regen,
            "via_composite": via_comp,
        }

    # normalize R columns
    Rnorm = {}
    col_tot = defaultdict(float)
    for i in R:
        for j in R[i]:
            col_tot[j] += R[i][j]
    for i in R:
        for j in R[i]:
            if col_tot[j] > 0:
                Rnorm.setdefault(i, {})[j] = round(R[i][j] / col_tot[j], 3)

    # closure score: composite, in a reconstructive cycle, regenerated via other
    # composites, with positive selective restoration gain.
    closure_score = 0.0
    closure_type = None
    for j, rec in records.items():
        if (rec["is_composite"] and rec["in_cycle"] and rec["via_composite"]
                and rec["selective_gain"] > closure_score):
            closure_score = rec["selective_gain"]
            closure_type = j

    return {
        "records": records,
        "R": Rnorm,
        "sccs": [sorted(s) for s in sccs],
        "closure_score": round(closure_score, 4),
        "closure_type": closure_type,
        "closure": closure_score > CLOSURE_THR,
    }


def seed_region(world, rng):
    """Initial region: food letters, a few composites, a resource pool."""
    st = State(resource=INIT_RESOURCE)
    for L in BASE_LETTERS:
        st.frag[L] = 8
    for c in composites(world):
        st.frag[c] = 4
    return st


# ===========================================================================
# HIDDEN POSITIVE CONTROL
# ===========================================================================
def _pick_two_composites(world):
    # prefer two LEVEL-1 composites (X0, X1): both bootstrap from food, so a
    # clean 2-cycle can be injected with a food letter for resource bootstrap.
    comps = composites(world)
    pref = [c for c in ("X0", "X1") if c in comps]
    if len(pref) == 2:
        return pref[0], pref[1]
    return comps[0], comps[1]


def _pc_region(world, x, y, rng):
    """A region seeded so the injected-cycle types x,y stay clearly abundant."""
    st = State(resource=INIT_RESOURCE)
    for L in BASE_LETTERS:
        st.frag[L] = 6
    for c in composites(world):
        st.frag[c] = 3
    st.frag[x] = 20
    st.frag[y] = 20
    run_life(world, st, rng, 3)   # short settle -> x,y remain >= MIN_ABUND
    return st


def inject_two_cycle(world, x, y):
    """Inject the minimal hyperedge change that makes x and y mutually
    reconstructive: force x's producer to require y (and drop x's food-only
    route) and y's producer to require x.  Returns the ids of the two
    'complementary producer' edges so the collapse test can remove one."""
    comp_set = set(composites(world))
    prod_ids = {}
    for (a, b) in ((x, y), (y, x)):
        # remove pure-food producers of a so its only route requires b
        keep = []
        for e in world["edges"]:
            if e["kind"] == "assembly" and a in e["outputs"]:
                if not (set(e["inputs"]) & comp_set) and e["catalyst"] not in comp_set:
                    continue  # drop the food-only route for a
            keep.append(e)
        world["edges"] = keep
        prods = [e for e in world["edges"]
                 if e["kind"] == "assembly" and a in e["outputs"] and a not in e["inputs"]]
        if prods:
            e = prods[0]
        else:
            e = add_edge(world, (b,), None, (a, WASTE), 0.4, "assembly")
        ins = list(e["inputs"])
        # ensure b is a required input, keep one food letter for bootstrap resource
        if b not in ins:
            if ins:
                ins[0] = b
            else:
                ins = [b]
        if BASE_LETTERS[0] not in ins:
            ins.append(BASE_LETTERS[0])
        e["inputs"] = tuple(ins)
        e["catalyst"] = None
        prod_ids[(a, b)] = e["id"]
    return prod_ids


def run_positive_control(verbose=True):
    """Inject a known 2-type cycle; verify the harness DETECTS it (positive
    selective restoration + collapse when a complementary producer is removed);
    estimate mutation-reachability of the closing edit."""
    out = {}
    world, _ = make_seed_world(seed=9973)
    assert certificate_no_cycle(world), "seed world must be acyclic"
    x, y = _pick_two_composites(world)

    prod_ids = inject_two_cycle(world, x, y)
    sccs = reconstructive_sccs(world)
    cycle_detected = any({x, y} <= s for s in sccs)
    out["cycle_detected"] = cycle_detected
    out["x"], out["y"] = x, y

    rng = random.Random(4242)
    st = _pc_region(world, x, y, rng)

    ana = reconstruction_analysis(world, st, random.Random(777))
    out["sel_gain_x"] = ana["records"].get(x, {}).get("selective_gain")
    out["sel_gain_y"] = ana["records"].get(y, {}).get("selective_gain")
    out["via_comp_x"] = ana["records"].get(x, {}).get("via_composite")
    out["via_comp_y"] = ana["records"].get(y, {}).get("via_composite")
    out["closure_detected"] = ana["closure"]
    out["R"] = ana["R"]

    # collapse test: remove the complementary producer of x (the "x from y"
    # edge) -> deleting x can no longer be reconstructed via y.
    w2 = clone_world(world)
    rm_id = prod_ids[(x, y)]
    w2["edges"] = [e for e in w2["edges"] if e["id"] != rm_id]
    rng2 = random.Random(4242)
    st2 = _pc_region(w2, x, y, rng2)
    ana2 = reconstruction_analysis(w2, st2, random.Random(777))
    out["collapse_sel_gain_x"] = ana2["records"].get(x, {}).get("selective_gain")
    out["collapse_closure"] = ana2["closure"]

    # mutation-reachability: from a PRECURSOR where only x<-y exists (y made
    # from a food letter Z), estimate P(one random unbiased mutation closes the
    # cycle by retargeting that input to x).
    prec, _ = make_seed_world(seed=9973)
    comp_set = set(composites(prec))
    # keep x<-y (drop food-only route for x, add y requirement)
    keep = []
    for e in prec["edges"]:
        if e["kind"] == "assembly" and x in e["outputs"]:
            if not (set(e["inputs"]) & comp_set) and e["catalyst"] not in comp_set:
                continue
        keep.append(e)
    prec["edges"] = keep
    xprod = [e for e in prec["edges"] if e["kind"] == "assembly" and x in e["outputs"] and x not in e["inputs"]]
    if xprod:
        e = xprod[0]
        ins = list(e["inputs"]);  ins = [y] + [BASE_LETTERS[0]]
        e["inputs"] = tuple(ins); e["catalyst"] = None
    else:
        add_edge(prec, (y, BASE_LETTERS[0]), None, (x, WASTE), 0.4, "assembly")
    # make y's producer food-only (from letter Z) so the closing edit is a
    # single redirect of that input to x.
    yprod = [e for e in prec["edges"] if e["kind"] == "assembly" and y in e["outputs"] and y not in e["inputs"]]
    if not yprod:
        add_edge(prec, (BASE_LETTERS[1],), None, (y, WASTE), 0.4, "assembly")
    else:
        yprod[0]["inputs"] = (BASE_LETTERS[1],)
        yprod[0]["catalyst"] = None
    assert certificate_no_cycle(prec), "precursor must still be acyclic (only x<-y)"

    trials = 4000
    hits = 0
    rk = random.Random(20260713)
    for _ in range(trials):
        wtmp = clone_world(prec)
        mutate_topology(wtmp, rk, allow_compound=False)
        if any({x, y} <= s for s in reconstructive_sccs(wtmp)):
            hits += 1
    out["reachability_p"] = hits / trials
    out["reachability_trials"] = trials

    if verbose:
        print("  cycle injected on   : (%s, %s)" % (x, y))
        print("  certificate detects : %s" % cycle_detected)
        print("  selective gain x/y  : %s / %s  (via_composite %s/%s)" %
              (out["sel_gain_x"], out["sel_gain_y"], out["via_comp_x"], out["via_comp_y"]))
        print("  closure DETECTED    : %s" % out["closure_detected"])
        print("  collapse test (remove complementary producer of x):")
        print("     sel_gain_x -> %s ; closure -> %s (expected collapse)" %
              (out["collapse_sel_gain_x"], out["collapse_closure"]))
        print("  mutation-reachability of closing edit: p=%.4f over %d trials" %
              (out["reachability_p"], trials))
        _print_R(out["R"], indent="     ")
    return out


def _print_R(R, indent=""):
    if not R:
        print(indent + "R matrix: (empty)")
        return
    print(indent + "R_ij (regeneration attribution, column-normalized):")
    for i in sorted(R.keys()):
        row = ", ".join("%s<-%s:%.2f" % (j, i, R[i][j]) for j in sorted(R[i].keys()))
        print(indent + "  %s -> %s" % (i, row))


# ===========================================================================
# Population / GA under each condition
# ===========================================================================
def _roulette(weights, k, rng):
    lo = min(weights)
    shifted = [w - lo + 1e-6 for w in weights]
    tot = sum(shifted)
    picks = []
    for _ in range(k):
        r = rng.random() * tot
        acc = 0.0
        for idx, w in enumerate(shifted):
            acc += w
            if acc >= r:
                picks.append(idx)
                break
        else:
            picks.append(len(shifted) - 1)
    return picks


def run_population(seed, condition, gens=N_GENS, n_ind=N_IND):
    """Evolve one population (the experimental unit) under a condition."""
    rng = random.Random(seed * 100003 + COND_OFFSET[condition])
    world0, _ = make_seed_world(seed)
    t0_sig = topology_signature(world0)
    m2_ok = certificate_no_cycle(world0)

    fixed_topo = (condition == "C5")
    real_sel   = condition in ("C1", "C5", "C10")
    allow_compound = (condition == "C10")

    # individuals: independent world copies + region states
    pop = []
    for _ in range(n_ind):
        w = clone_world(world0)
        pop.append({"world": w, "state": seed_region(w, rng)})

    topology_changed = False
    for g in range(gens):
        fits = []
        for ind in pop:
            f, s_new = physics_fitness(ind["world"], ind["state"], rng)
            ind["state"] = s_new
            fits.append(f)

        if condition == "C2":
            # shuffled-fitness: permute fitness among regions (break trait->fit)
            perm = fits[:]
            rng.shuffle(perm)
            sel_w = perm
        elif condition == "C6":
            # no selection: random survival
            sel_w = [1.0 for _ in pop]
        else:
            sel_w = fits

        parents = _roulette(sel_w, n_ind, rng)

        newpop = []
        for pi in parents:
            par = pop[pi]
            w = clone_world(par["world"])
            st = par["state"].clone()
            apply_dilution(st, 0.5, rng)   # inheritance = founding propagule
            if not fixed_topo:
                mutate_topology(w, rng, allow_compound=allow_compound)
            if topology_signature(w) != t0_sig:
                topology_changed = True
            newpop.append({"world": w, "state": st})
        pop = newpop

    # final settle + reconstruction analysis on the fittest few individuals
    scored = []
    for ind in pop:
        f, s_new = physics_fitness(ind["world"], ind["state"], rng)
        ind["state"] = s_new
        scored.append((f, ind))
    scored.sort(key=lambda x: x[0], reverse=True)

    best_closure = 0.0
    best_ana = None
    n_analyze = min(2, len(scored))
    for _, ind in scored[:n_analyze]:
        ana = reconstruction_analysis(ind["world"], ind["state"], random.Random(seed * 13 + 7))
        if ana["closure_score"] >= best_closure:
            best_closure = ana["closure_score"]
            best_ana = ana

    return {
        "seed": seed, "condition": condition,
        "m2_ok": m2_ok,
        "topology_changed": topology_changed,
        "closure_score": best_closure,
        "closure": best_closure > CLOSURE_THR,
        "n_sccs_final": len(best_ana["sccs"]) if best_ana else 0,
        "ana": best_ana,
    }


# ===========================================================================
# Statistics: paired bootstrap 95% CI
# ===========================================================================
def paired_bootstrap(a, b, B=BOOT_B, seed=12345):
    """Paired bootstrap of mean(a-b).  Returns (mean_diff, ci_lo, ci_hi, sem)."""
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    md = sum(diffs) / n
    # SEM of the paired differences (analytic)
    if n > 1:
        var = sum((d - md) ** 2 for d in diffs) / (n - 1)
        sem = math.sqrt(var / n)
    else:
        sem = 0.0
    rng = random.Random(seed)
    boots = []
    for _ in range(B):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * B)]
    hi = boots[min(B - 1, int(0.975 * B))]
    return md, lo, hi, sem


def mean_sem(xs):
    n = len(xs)
    m = sum(xs) / n
    if n > 1:
        var = sum((x - m) ** 2 for x in xs) / (n - 1)
        sem = math.sqrt(var / n)
    else:
        sem = 0.0
    return m, sem


# ===========================================================================
# Smoke driver
# ===========================================================================
def run_smoke(n_seeds=N_SEEDS, gens=N_GENS, n_ind=N_IND):
    print("\n[SMOKE] %d seeds x %d generations x %d individuals, conditions=%s"
          % (n_seeds, gens, n_ind, CONDITIONS))
    seeds = list(range(1, n_seeds + 1))
    results = {c: {} for c in CONDITIONS}
    m1_any = False
    m2_all = True

    for c in CONDITIONS:
        print("  running condition %-4s : %s" % (c, COND_DESC[c]))
        for s in seeds:
            r = run_population(s, c, gens=gens, n_ind=n_ind)
            results[c][s] = r
            m1_any = m1_any or r["topology_changed"]
            m2_all = m2_all and r["m2_ok"]

    # per-condition closure vectors (paired by seed)
    closure_vec = {c: [results[c][s]["closure_score"] for s in seeds] for c in CONDITIONS}
    closure_hit = {c: [1.0 if results[c][s]["closure"] else 0.0 for s in seeds] for c in CONDITIONS}

    print("\n--- PER-SEED closure_score (selective leave-one-out restoration gain) ---")
    header = "seed  " + "".join("%8s" % c for c in CONDITIONS)
    print(header)
    for s in seeds:
        print("%-6d" % s + "".join("%8.3f" % results[c][s]["closure_score"] for c in CONDITIONS))

    print("\n--- PER-CONDITION mean closure_score (labeled SEM) ---")
    cond_mean = {}
    for c in CONDITIONS:
        m, sem = mean_sem(closure_vec[c])
        hit_m, _ = mean_sem(closure_hit[c])
        cond_mean[c] = m
        print("  %-4s  mean=%.4f  SEM=%.4f   closure_rate=%.2f   %s"
              % (c, m, sem, hit_m, COND_DESC[c]))

    print("\n--- PAIRED BOOTSTRAP 95%% CI : C1 (real selection) minus each control ---")
    m4_flags = {}
    for c in ["C2", "C5", "C6", "C10"]:
        md, lo, hi, sem = paired_bootstrap(closure_vec["C1"], closure_vec[c])
        excludes_zero = (lo > 0.0) or (hi < 0.0)
        m4_flags[c] = (md > 0.0)
        print("  C1 - %-4s : diff=%+.4f  SEM=%.4f  95%%CI=[%+.4f, %+.4f]  CI_excludes_0=%s"
              % (c, md, sem, lo, hi, excludes_zero))

    return {
        "results": results, "seeds": seeds,
        "closure_vec": closure_vec, "cond_mean": cond_mean,
        "m1_any": m1_any, "m2_all": m2_all, "m4_flags": m4_flags,
    }


# ===========================================================================
# Main: certificate check + positive control + smoke + milestone report
# ===========================================================================
def main():
    print("=" * 74)
    print("FAMILY B -- MUTABLE CATALYTIC HYPERGRAPH  (substrate + harness self-test)")
    print("=" * 74)

    # ---- Certificate check across seeds (M2) ---------------------------
    print("\n[1] CERTIFICATE: no endogenous reconstructive cycle at t=0")
    cert_ok = True
    for s in range(1, N_SEEDS + 1):
        w, _ = make_seed_world(s)
        ok = certificate_no_cycle(w)
        n_assembly = sum(1 for e in w["edges"] if e["kind"] == "assembly")
        cert_ok = cert_ok and ok
        print("  seed %2d : acyclic=%s  (types=%d, assembly_edges=%d, composites=%d)"
              % (s, ok, len(w["types"]), n_assembly, len(composites(w))))
    print("  --> all seeds acyclic at t=0: %s" % cert_ok)

    # ---- Hidden positive control ---------------------------------------
    print("\n[2] HIDDEN POSITIVE CONTROL: inject a known 2-type cycle, verify detection")
    pc = run_positive_control(verbose=True)

    # ---- Smoke experiment ----------------------------------------------
    print("\n[3] SMOKE EXPERIMENT")
    smoke = run_smoke()

    # ---- Milestone report ----------------------------------------------
    print("\n" + "=" * 74)
    print("MILESTONE REPORT")
    print("=" * 74)

    M1 = smoke["m1_any"]
    M2 = smoke["m2_all"] and cert_ok
    # M3: harness demonstrably detects operational closure (positive control is
    # the ground-truth injected cycle) AND/OR emergent closure in the smoke.
    emergent_closure = any(smoke["results"][c][s]["closure"]
                           for c in CONDITIONS for s in smoke["seeds"])
    M3 = bool(pc["closure_detected"]) or emergent_closure
    # M4: closure_score higher under real selection (C1) than each control.
    M4 = all(smoke["m4_flags"][c] for c in ["C2", "C6"]) and (
        smoke["cond_mean"]["C1"] > smoke["cond_mean"]["C5"])

    print("  M1 topology change occurs (mutable conditions)        : %s" % _mark(M1))
    print("  M2 endogenous reconstructive cycle ABSENT at t=0      : %s" % _mark(M2))
    print("  M3 operational closure detectable (pos. selective     : %s" % _mark(M3))
    print("     leave-one-out restoration; validated by pos ctrl)")
    print("  M4 closure_score(C1 real-sel) > shuffled/random/no-   : %s" % _mark(M4))
    print("     selection controls  [C1>%s, C1>%s, C1>%s(fixed-topo)]"
          % ("C2" if smoke["m4_flags"]["C2"] else "!C2",
             "C6" if smoke["m4_flags"]["C6"] else "!C6",
             "C5" if smoke["cond_mean"]["C1"] > smoke["cond_mean"]["C5"] else "!C5"))
    print("\n  NOTE: milestone flags report only what the harness detected in this")
    print("        bounded smoke run.  They are substrate/harness diagnostics, not")
    print("        a scientific result.  Positive-control detection validates the")
    print("        measurement; emergence under selection is the open question.")

    print("\n=== FAMB_SMOKE_DONE ===")


def _mark(b):
    return "REACHED" if b else "not-reached"


if __name__ == "__main__":
    main()
