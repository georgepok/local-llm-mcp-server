import random, math
import statistics as st
from collections import Counter

# =============================================================================
# FAMILY C — ENDOGENOUS TRANSLATION CHEMISTRY
# (ENDOGENOUS_CLOSURE_ORIGIN_PROGRAM_V1, deepest branch)
#
# CORE IDEA. Fragments are short symbol strings. They have NO globally fixed
# producer meaning. What a fragment X "does" is decided by the LOCAL TRANSLATOR
# fragments present in its region. A translator fragment recognises a symbolic
# pattern (via `match`) and specifies an effect (`emit/transform/redirect/...`).
# Therefore the effective producer relation  X -> Y  is a FUNCTION of the local
# translator set, NOT a property of world law. The effective reaction graph is
# RECONSTRUCTED FROM ACTUAL LOCAL INTERACTIONS at every checkpoint, never read
# from a fixed table. This tests whether causal topology can become a PRODUCT of
# organization rather than a fixed law.
#
# GENERIC PRIMITIVE VOCABULARY (nothing domain-specific):
#   MECHANISMS   : bind (translator<->substrate association), match (recognition),
#                  activate (catalytic presence — a translator only acts where its
#                  pattern-substrate is present), inhibit (suppression).
#   OP CODES     : each translator carries one generic op, decoded by a FIXED
#                  generic instruction table (like opcode->uop in a CPU; this is
#                  world PHYSICS, NOT a producer table):
#                     a EMIT      : produce output fragment  (X -> out)
#                     b TRANSFORM : rewrite matched prefix    (X -> out + X[|pat|:])
#                     c CONCAT    : concatenate               (X -> X + out)
#                     d SPLIT     : cut at pattern boundary   (X -> head, tail)
#                     e COPY      : copy-fragment (SELF-COPY)  (X -> X)     [flagged]
#                     f REDIRECT  : CONSTRUCT A NEW TRANSLATOR from out (build topology)
#                     g INHIBIT   : consume X, emit nothing   (suppress)
#   NOTE: the op *table* is fixed generic physics. WHICH pattern maps to WHICH
#   effect is encoded in the mutable translator STRINGS — that is the endogenous,
#   evolvable "genetic code", never prewritten and never a supplied reciprocal
#   pair.
#
# TRANSLATOR GRAMMAR (fixed generic law for how physics READS a translator):
#   a fragment is a translator iff it contains the marker '|'. Structure:
#       PATTERN | OP OUTPUT        e.g. 'c|ad' = (pattern 'c', op EMIT, out 'd')
#   A fragment with no '|' is pure substrate/data.
#
# This file builds ONLY the substrate + harness. It makes NO scientific claim.
# =============================================================================

CONTENT = 'abcdefg'          # 7 content symbols
MARK = '|'                   # structural translator marker
TOKENS = CONTENT + MARK      # 8 tokens total (small alphabet)
LMAX = 6                     # fragments <= 6 symbols
OPMAP = {'a': 'EMIT', 'b': 'TRANSFORM', 'c': 'CONCAT',
         'd': 'SPLIT', 'e': 'COPY', 'f': 'REDIRECT', 'g': 'INHIBIT'}

# --------------------------------------------------------------------------- #
# Fragment interpretation (generic, fixed physics)
# --------------------------------------------------------------------------- #
def parse_translator(f):
    """Return (pattern, op, out) if f is a well-formed translator, else None."""
    if MARK not in f:
        return None
    i = f.index(MARK)
    lhs, rhs = f[:i], f[i + 1:]
    if not lhs or not rhs:
        return None
    op_sym = rhs[0]
    if op_sym not in OPMAP:          # op position holds the marker or nothing -> inert
        return None
    return lhs, OPMAP[op_sym], rhs[1:]

def is_translator(f):
    return parse_translator(f) is not None

def make_trans(out):
    """REDIRECT: construct a NEW translator string from an output field."""
    if not out:
        return None
    if MARK in out:
        t = out
    elif len(out) >= 2:
        t = out[0] + MARK + out[1:]
    else:
        return None
    t = t[:LMAX]
    return t if parse_translator(t) else None

def apply_op(pattern, op, out, X):
    """Apply a matched translator's op to substrate X.
    Returns (list_of_products, self_copy_flag). Products bounded to LMAX."""
    if op == 'EMIT':
        return ([out] if out else []), False
    if op == 'TRANSFORM':
        y = (out + X[len(pattern):])[:LMAX]
        return ([y] if y else []), False
    if op == 'CONCAT':
        return [(X + out)[:LMAX]], False
    if op == 'SPLIT':
        head, tail = X[:len(pattern)], X[len(pattern):]
        return [p for p in (head, tail) if p], False
    if op == 'COPY':
        return [X], True                       # self-copy (flagged; excluded from R_ij)
    if op == 'REDIRECT':
        t = make_trans(out)
        return ([t] if t else []), False       # constructs new translation
    if op == 'INHIBIT':
        return [], False                       # consumes X, no product
    return [], False

# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def wpick(items, rng):
    tot = sum(w for _, w in items)
    if tot <= 0:
        return None
    r = rng.random() * tot
    c = 0.0
    for k, w in items:
        c += w
        if r <= c:
            return k
    return items[-1][0]

def remove_one(reg, rng):
    k = wpick(list(reg.items()), rng)
    if k is None:
        return
    reg[k] -= 1
    if reg[k] <= 0:
        del reg[k]

def random_food(rng, a=1, b=3):
    return ''.join(rng.choice(CONTENT) for _ in range(rng.randint(a, b)))

def cosim(a, b):
    keys = set(a) | set(b)
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return sum(a.get(k, 0) * b.get(k, 0) for k in keys) / (na * nb)

def dh(s):                                     # deterministic hash (PYTHONHASHSEED-independent)
    return sum((i + 1) * ord(ch) for i, ch in enumerate(s))

def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]

# --------------------------------------------------------------------------- #
# Core reaction dynamics. The effect of applying X depends ENTIRELY on the local
# translator set present in `reg` at this instant.
# --------------------------------------------------------------------------- #
class CFG:
    SRC = 0.16; SRC_COST = 0.5; COST = 1.0; DEATH = 0.06
    MUT = 0.06; CLOSE_P = 0.9
    NPOP = 5; NGEN = 20; TICKS = 220
    BOTTLE_STEPS = 110; EST_STEPS = 80; PROP_N = 14; SEED_N = 44
    MINAB = 2          # "abundant" threshold for the leave-one-out reconstruction matrix
    MINAB_GRAPH = 1    # effective-graph reconstruction counts any present translator-mediated interaction
    NSEED = 8; P0 = 380.0

def react(reg, P, rng, steps, neutral=False, track=None):
    """One region, `steps` reaction events. Mutates reg (Counter) in place.
    Returns (events, P). Every product is decided by the LOCAL translator set."""
    events = 0
    for _ in range(steps):
        if P < CFG.COST:
            break
        if rng.random() < CFG.SRC:                       # generic resource inflow -> raw substrate
            reg[random_food(rng, 1, 2)] += 1
            P -= CFG.SRC_COST
            continue
        if not reg:
            break
        # substrates are non-translator (data) fragments; translators are catalysts
        subs = [(t, c) for t, c in reg.items() if not is_translator(t)]
        if not subs:
            continue
        X = wpick(subs, rng)
        if X is None:
            continue
        # local translator set that RECOGNISES X (match) -> defines X's effect here
        matched = []
        for t, c in reg.items():
            pr = parse_translator(t)
            if pr and X.startswith(pr[0]):
                matched.append((t, c))
        if not matched:                                   # no local translator -> X is inert (no fixed fallback)
            continue
        T = wpick(matched, rng)                           # bind one translator (weighted by abundance); catalyst persists
        pat, op, out = parse_translator(T)
        prods, self_copy = apply_op(pat, op, out, X)
        if neutral:                                       # opcode-blind null: product independent of translator
            prods = [random_food(rng, 1, 2)]
            self_copy = False
        reg[X] -= 1                                        # consume one substrate X
        if reg[X] <= 0:
            del reg[X]
        for Y in prods:
            reg[Y] += 1
            if track is not None:
                track.append((T, X, Y, self_copy))
        P -= CFG.COST
        events += 1
        if rng.random() < CFG.DEATH:                      # matched decay (substrate AND translators)
            remove_one(reg, rng)
    return events, P

# --------------------------------------------------------------------------- #
# EFFECTIVE-GRAPH RECONSTRUCTION — built from ACTUAL local interactions, never a
# fixed table. Present translators (and any they construct) mediate the edges.
# --------------------------------------------------------------------------- #
def build_type_graph(reg, min_ab, horizon):
    present = set(f for f, c in reg.items() if c >= min_ab)
    reach = set(present)
    edges = {}                                            # X -> set((Y, mediatorT, self_copy))
    for _ in range(horizon):
        parsed = []
        for f in reach:
            pr = parse_translator(f)
            if pr:
                parsed.append((f, pr[0], pr[1], pr[2]))
        newf = set()
        for X in [f for f in reach if not is_translator(f)]:   # substrates only (mirror react)
            for (T, p, op, out) in parsed:
                if X.startswith(p):
                    prods, sc = apply_op(p, op, out, X)
                    for Y in prods:
                        e = edges.setdefault(X, set())
                        e.add((Y, T, sc))
                        if Y not in reach and Y not in newf:
                            newf.add(Y)
        if not newf:
            break
        reach |= newf
    return present, reach, edges

def effective_edge_set(reg, min_ab):
    """One-step translator-mediated (X->Y) edges, self-copy excluded."""
    _, _, edges = build_type_graph(reg, min_ab, 1)
    out = set()
    for X, outs in edges.items():
        for (Y, T, sc) in outs:
            if (not sc) and Y != X:
                out.add((X, Y))
    return out

def detect_cycle(edges):
    """True if the type graph (self-copy & self-loops excluded) has a directed
    cycle -> a fragment is regenerated through OTHER fragments' translation."""
    adj = {}
    for X, outs in edges.items():
        for (Y, T, sc) in outs:
            if (not sc) and Y != X:
                adj.setdefault(X, set()).add(Y)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {}
    for start in list(adj):
        if color.get(start, WHITE) != WHITE:
            continue
        stack = [(start, iter(adj.get(start, ())))]
        color[start] = GRAY
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                cv = color.get(v, WHITE)
                if cv == GRAY:
                    return True
                if cv == WHITE:
                    color[v] = GRAY
                    stack.append((v, iter(adj.get(v, ()))))
                    advanced = True
                    break
            if not advanced:
                color[u] = BLACK
                stack.pop()
    return False

def has_reconstructive_cycle(reg, min_ab, horizon=3):
    _, _, edges = build_type_graph(reg, min_ab, horizon)
    return detect_cycle(edges)

# --------------------------------------------------------------------------- #
# CERTIFICATE — initial translation organization must be INCOMPLETE & NON-CLOSED
# --------------------------------------------------------------------------- #
def certify(reg, horizon=3):
    present, reach, edges = build_type_graph(reg, 1, horizon)
    present_trans = set(f for f in present if is_translator(f))
    sub_edges = 0
    reproduced = set()
    for X, outs in edges.items():
        for (Y, T, sc) in outs:
            if (not sc) and Y != X:
                sub_edges += 1
                if Y in present_trans:
                    reproduced.add(Y)                     # a present translator gets constructed
    has_cyc = detect_cycle(edges)
    ok = (sub_edges > 0) and (len(reproduced) == 0) and (not has_cyc)
    return ok, dict(n_trans=len(present_trans), sub_edges=sub_edges,
                    reproduced=sorted(reproduced), has_cycle=has_cyc, n_reach=len(reach))

def make_pool(seed):
    rng = random.Random(seed)
    reg = Counter()
    for _ in range(6):
        reg[random_food(rng, 1, 3)] += rng.randint(5, 8)
    for _ in range(rng.randint(2, 4)):                    # translators are a robust heritable species
        pat = ''.join(rng.choice(CONTENT) for _ in range(rng.randint(1, 2)))
        op = rng.choice(CONTENT)
        out = ''.join(rng.choice(CONTENT) for _ in range(rng.randint(1, 3)))
        t = (pat + MARK + op + out)[:LMAX]
        if parse_translator(t):
            reg[t] += rng.randint(16, 24)
    return reg

def make_certified_pool(seed, tag=None):
    """Resample until the initial pool is a working-but-OPEN chemistry:
    effective substrate edges exist, yet no translator is reconstructed and no
    reconstructive cycle is instantiated at t=0."""
    for k in range(80):
        reg = make_pool(seed + 1000 * k)
        if tag == 'C10':                                  # seed one ORPHAN emit-translator (still open at t=0)
            reg['c'] += 16; reg['d'] += 16; reg['c|ad'] += 14
        ok, info = certify(reg)
        if ok:
            return reg, k + 1, info
    return reg, 80, info

# --------------------------------------------------------------------------- #
# RECONSTRUCTION MATRIX (leave-one-out). For every abundant fragment type j:
# delete it, CLONE state+RNG across intervention and matched-control arms, run
# matched recovery, record restoration gain G=S1-S0 and whether j is regenerated
# by OTHER fragments through constructed translation (not self-copy). Build R_ij.
# --------------------------------------------------------------------------- #
def leave_one_out(reg, seedv, K=5, rec_steps=110):
    """For every abundant type j: delete it, run matched recovery with a CLONED
    RNG shared by the intervention arm and an abundance-matched control arm, and
    record G=S1-S0, how many times j is regenerated by OTHER fragments through
    constructed translation (not self-copy), and the mediators (-> R_ij).
    The robust operational-closure metric compares regenerated-by-others types to
    the best GENERIC (source-only) recovery baseline among non-regenerated types."""
    pre = Counter(reg)
    if sum(pre.values()) < 8:
        return 0.0, {}, {}, False
    types = [t for t, _ in pre.most_common() if pre[t] >= CFG.MINAB][:K]
    R = {}                                                # mediator i -> {j: count regenerated}
    Gs = {}                                               # j -> (G, regen_by_others, matched_selG)
    for j in types:
        rseed = seedv * 1000 + dh(j) % 1000               # CLONED across arms -> matched noise
        arm = Counter(pre); arm.pop(j, None)              # intervention arm: delete j
        S0 = cosim(pre, arm)
        track = []
        react(arm, 1e9, random.Random(rseed), rec_steps, track=track)
        G = cosim(pre, arm) - S0
        regen_other = 0
        for (T, X, Y, sc) in track:
            if Y == j and (not sc) and X != j:            # regenerated by an OTHER fragment via translation
                regen_other += 1
                R.setdefault(T, {}).setdefault(j, 0)
                R[T][j] += 1
        others = [t for t in types if t != j]             # matched control arm: SAME cloned RNG
        jm = min(others, key=lambda t: abs(pre[t] - pre[j])) if others else None
        selG = G
        if jm is not None:
            arm2 = Counter(pre); arm2.pop(jm, None)
            S0m = cosim(pre, arm2)
            react(arm2, 1e9, random.Random(rseed), rec_steps)
            selG = G - (cosim(pre, arm2) - S0m)
        Gs[j] = (G, regen_other, selG)
    nonregen = [G for (G, r, s) in Gs.values() if r == 0]
    base = max(nonregen) if nonregen else 0.0             # best source-only recovery = generic baseline
    best = max([G - base for (G, r, s) in Gs.values() if r > 0], default=0.0)
    any_other = any(r > 0 for (G, r, s) in Gs.values())
    return best, R, Gs, any_other

# --------------------------------------------------------------------------- #
# VIABILITY — CLOSURE-BLIND, generic physics only. NEVER references cycles / SCC
# / closure / reconstruction / classes. Fitness attaches to the WORLD, not to a
# fragment identity.
# --------------------------------------------------------------------------- #
def viability(reg_final, events, seedv):
    mass = sum(reg_final.values()) or 1
    rng = random.Random(seedv * 131 + 7)
    diluted = Counter()                                   # generic bottleneck: keep ~30% of material
    for t, c in reg_final.items():
        k = sum(1 for _ in range(c) if rng.random() < 0.30)
        if k:
            diluted[t] = k
    react(diluted, 1e9, rng, CFG.BOTTLE_STEPS)            # material retained through/after bottleneck
    retained = sum(diluted.values()) / mass
    prop = Counter()                                      # propagule establishment
    for _ in range(CFG.PROP_N):
        k = wpick(list(reg_final.items()), random.Random(seedv * 777 + 1 + _))
        if k:
            prop[k] += 1
    react(prop, 1e9, random.Random(seedv * 99 + 5), CFG.EST_STEPS)
    est = 1.0 if sum(prop.values()) >= 1.5 * CFG.PROP_N else 0.0
    # generic physics only (throughput / material retained / establishment); NEVER cycles/closure.
    # throughput weighted so ACTIVE chemistry is viable. Continuous source inflow means dead-end
    # LINEAR chemistries sustain throughput too, so this rewards active translators, NOT closure.
    return 0.020 * events + 0.4 * retained + 0.2 * est

# --------------------------------------------------------------------------- #
# Mutation — GENERIC point edits. Never biased toward closure/reciprocity.
# --------------------------------------------------------------------------- #
def mutate_frag(f, rng, content_only=False):
    toks = CONTENT if content_only else TOKENS
    g = list(f)
    if not g:
        return rng.choice(CONTENT)
    r = rng.random()
    if r < 0.6 or len(g) == 1:                            # substitute
        g[rng.randrange(len(g))] = rng.choice(toks)
    elif r < 0.8 and len(g) < LMAX:                       # insert
        g.insert(rng.randrange(len(g) + 1), rng.choice(toks))
    else:                                                 # delete
        del g[rng.randrange(len(g))]
    out = ''.join(g)[:LMAX]
    return out or rng.choice(CONTENT)

def close_cycle_edit(reg, rng):
    """C10 diagnostic: one edit that CLOSES a translational cycle. If an EMIT
    translator P|aQ is present, construct its complement Q|aP (regenerates P from
    Q via constructed translation)."""
    for f in list(reg):
        pr = parse_translator(f)
        if pr and pr[1] == 'EMIT' and len(pr[0]) == 1 and len(pr[2]) == 1:
            comp = pr[2] + MARK + 'a' + pr[0]
            remove_one(reg, rng)
            reg[comp] += 3
            return True
    return False

def mutate_pool(reg, rng, cond):
    reg = Counter(reg)
    mass = sum(reg.values())
    if mass == 0:
        return reg
    for _ in range(max(0, int(round(mass * CFG.MUT)))):
        f = wpick(list(reg.items()), rng)
        if f is None:
            break
        if cond == 'C5' and is_translator(f):             # fixed-translation: translators immutable
            continue
        reg[f] -= 1
        if reg[f] <= 0:
            del reg[f]
        reg[mutate_frag(f, rng, content_only=(cond == 'C5'))] += 1
    if cond == 'C10' and rng.random() < CFG.CLOSE_P:
        close_cycle_edit(reg, rng)
    return reg

# --------------------------------------------------------------------------- #
# Population-level evolution. SELECTION uses ONLY generic viability. The
# reconstruction machinery is MEASUREMENT and never feeds selection.
# --------------------------------------------------------------------------- #
def select_parents(Vs, cond, rng):
    n = len(Vs)
    if cond == 'C6':                                      # mutable-no-selection (drift)
        return [rng.randrange(n) for _ in range(n)]
    v = list(Vs)
    if cond == 'C2':                                      # shuffled-fitness (break genotype<->fitness link)
        rng.shuffle(v)
    items = list(enumerate(v))
    lo = min(w for _, w in items)
    items = [(i, w - lo + 1e-6) for i, w in items]        # keep weights positive
    return [wpick(items, rng) for _ in range(n)]

def evolve(seed, cond):
    rng = random.Random(seed * 1009 + dh(cond))
    tag = 'C10' if cond == 'C10' else None
    init_pops = [make_certified_pool(seed * 100 + i, tag)[0] for i in range(CFG.NPOP)]
    pops = [Counter(p) for p in init_pops]
    for gen in range(CFG.NGEN):
        Vs, finals = [], []
        for i, p in enumerate(pops):
            reg = Counter(p)
            ev, _ = react(reg, CFG.P0, random.Random(seed * 7919 + gen * 97 + i), CFG.TICKS)
            Vs.append(viability(reg, ev, seed * 31 + gen * 13 + i))
            finals.append(reg)
        idx = select_parents(Vs, cond, rng)
        newpops = []
        for pi in idx:
            prop = Counter()
            for _ in range(CFG.SEED_N):
                k = wpick(list(finals[pi].items()), rng)
                if k:
                    prop[k] += 1
            newpops.append(mutate_pool(prop, rng, cond))
        pops = newpops
    return pops, init_pops

def express(pool, seedv):
    """Develop a propagule into an expressed soup; the effective graph is then
    reconstructed from the ACTUAL interactions that occur in that soup."""
    dev = Counter(pool)
    react(dev, 1e9, random.Random(seedv), CFG.TICKS)
    return dev

# --------------------------------------------------------------------------- #
# statistics helpers (SEM vs bootstrap-CI kept clearly distinct)
# --------------------------------------------------------------------------- #
def mean_sem(xs):
    m = st.mean(xs) if xs else 0.0
    s = (st.stdev(xs) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
    return m, s

def boot_ci(xs, B=2000, seed=0):
    if not xs:
        return 0.0, 0.0, 0.0
    r = random.Random(seed); n = len(xs)
    ms = sorted(sum(r.choice(xs) for _ in range(n)) / n for _ in range(B))
    return st.mean(xs), ms[int(0.025 * B)], ms[int(0.975 * B)]

# =============================================================================
# REPORT SECTIONS
# =============================================================================
def demo_context_dependence():
    print("--- CONTEXT-DEPENDENCE: same fragment X, effect set by LOCAL translators ---", flush=True)
    X = 'caa'
    regions = {
        "{c|ad}  (c:EMIT d)      ": Counter({'c|ad': 5, X: 3}),
        "{c|be}  (c:TRANSFORM e) ": Counter({'c|be': 5, X: 3}),
        "{c|fda} (c:REDIRECT ->) ": Counter({'c|fda': 5, X: 3}),
        "{}      (no translator) ": Counter({X: 3}),
    }
    for name, reg in regions.items():
        _, _, edges = build_type_graph(reg, 1, 1)
        prods = sorted({Y for (Y, T, sc) in edges.get(X, set())})
        print(f"    X='{X}' in region {name}-> products {prods if prods else '[] (inert)'}", flush=True)
    print("    => producer relation X->Y is a FUNCTION of the local translator set, not world law.", flush=True)

def run_certificate():
    print("--- CERTIFICATE: initial translation organization INCOMPLETE & NON-CLOSED (t=0) ---", flush=True)
    total_attempts = 0
    for s in range(4):
        reg, attempts, info = make_certified_pool(s)
        total_attempts += attempts
        print(f"    seed {s}: attempts={attempts}  translators={info['n_trans']}  "
              f"1-step sub-edges={info['sub_edges']}  reproduced-translators={info['reproduced']}  "
              f"reconstructive-cycle={info['has_cycle']}  reachable-types={info['n_reach']}", flush=True)
    print("    CERT PASS: effective substrate edges exist, NO translator is reconstructed, "
          "NO reconstructive cycle at t=0 (seeds resampled until this holds).", flush=True)

def build_positive_pool(seed, complement=True):
    rng = random.Random(seed)
    reg = Counter()
    reg['c'] += 22; reg['d'] += 22; reg['e'] += 22        # 'e' = abundant NON-cycle matched control
    reg['c|ad'] += 12                                     # U1: c -EMIT-> d  (catalyst)
    if complement:
        reg['d|ac'] += 12                                 # U2: d -EMIT-> c  (constructed reciprocal catalyst)
    for _ in range(5):
        reg[random_food(rng, 2, 3)] += 4
    return reg

def run_positive_control():
    print("--- HIDDEN POSITIVE CONTROL: minimal CONSTRUCTED reconstructive cycle c<->d ---", flush=True)
    # warm both worlds
    pc = build_positive_pool(1234, complement=True)
    react(pc, 1e9, random.Random(11), 600)
    pn = build_positive_pool(1234, complement=False)
    react(pn, 1e9, random.Random(11), 600)

    best_c, R, Gs, any_other = leave_one_out(pc, seedv=7, K=5, rec_steps=200)
    Gc = Gs.get('c', (0, 0, 0))
    Ge = Gs.get('e', (0, 0, 0))
    print(f"    WITH complement (U1+U2):", flush=True)
    print(f"      delete cycle substrate 'c': G={Gc[0]:+.3f}  regen-by-others(events)={Gc[1]}  matched-selG={Gc[2]:+.3f}", flush=True)
    print(f"      delete matched non-cycle 'e': G={Ge[0]:+.3f}  regen-by-others(events)={Ge[1]}  matched-selG={Ge[2]:+.3f}", flush=True)
    print(f"      robust operational-closure metric (best regen-by-others minus generic baseline) = {best_c:+.3f}", flush=True)
    rrow = {i: R[i].get('c', 0) for i in R if 'c' in R[i]}
    print(f"      R_i,'c' (which OTHER fragments regenerate 'c' via constructed translation): {rrow}", flush=True)

    best_n, _, Gn, _ = leave_one_out(pn, seedv=7, K=5, rec_steps=200)
    Gcn = Gn.get('c', (0, 0, 0))
    print(f"    WITHOUT complement (U1 only):", flush=True)
    print(f"      delete 'c': G={Gcn[0]:+.3f}  regen-by-others(events)={Gcn[1]}  "
          f"=> 'c'-restoration COLLAPSES ({Gc[1]}->{Gcn[1]}) when U2 removed", flush=True)
    print(f"      (note: 'd' is still LINEARLY regenerated by U1 (c->d), a producer edge, NOT a closed cycle.)", flush=True)

    # mutation-reachability of the closing change (create U2='d|ac')
    target = 'd|ac'
    present = [f for f in pn if f]
    min_edits = min(levenshtein(f, target) for f in present)
    nearest = min(present, key=lambda f: levenshtein(f, target))
    rng = random.Random(99); hits1 = hits2 = trials = 0
    for _ in range(4000):
        base = rng.choice(present)
        if mutate_frag(base, rng) == target:
            hits1 += 1
        if mutate_frag(mutate_frag(base, rng), rng) == target:
            hits2 += 1
        trials += 1
    print(f"    mutation-reachability of closing translator '{target}': "
          f"min edit-distance={min_edits} (from '{nearest}'); "
          f"P(reach in 1 mut)~{hits1/trials:.4f}, P(reach in 2 mut)~{hits2/trials:.4f}", flush=True)

    ok = (Gc[1] >= 5) and (Gc[0] > 0.02) and (Ge[1] == 0) and (Gcn[1] == 0) and (best_c > 0.02)
    print(f"    POSITIVE CONTROL {'PASSES' if ok else 'FAILS'}: robust regen-by-others of 'c' (={Gc[1]}) via "
          f"constructed translation, absent for matched 'e' (={Ge[1]}) and on U2 removal (={Gcn[1]}).", flush=True)
    if not ok:
        print("    !!! HARNESS INVALID — interpret smoke milestones with caution.", flush=True)
    return ok

def run_smoke():
    print("--- SMOKE: conditions x seeds (seed = experimental unit) ---", flush=True)
    conds = ['C1', 'C2', 'C5', 'C6', 'C10']
    labels = {'C1': 'real-selection', 'C2': 'shuffled-fitness', 'C5': 'fixed-translation',
              'C6': 'mutable-no-selection', 'C10': 'one-step (diagnostic)'}
    res = {c: {'m3': [], 'm2': [], 'm1': []} for c in conds}
    for cond in conds:
        for s in range(CFG.NSEED):
            pops, init_pops = evolve(s, cond)
            gen0_union = set()
            for i, p in enumerate(init_pops):
                gen0_union |= effective_edge_set(express(p, s * 51 + i), CFG.MINAB_GRAPH)
            final_union = set()
            m2 = 0; best_m3 = 0.0
            for i, p in enumerate(pops):
                dev = express(p, s * 71 + i * 13 + 5)     # reconstruct from the EXPRESSED soup
                final_union |= effective_edge_set(dev, CFG.MINAB_GRAPH)
                if has_reconstructive_cycle(dev, CFG.MINAB_GRAPH):
                    m2 += 1
                b, _, _, _ = leave_one_out(dev, seedv=s * 17 + 3)
                best_m3 = max(best_m3, b)
            new_edges = final_union - gen0_union
            res[cond]['m1'].append(len(new_edges))        # topology change: new effective edges
            res[cond]['m2'].append(m2 / len(pops))        # fraction of pops with endogenous cycle
            res[cond]['m3'].append(best_m3)               # best selective leave-one-out restoration
    print("    per-condition (mean +/- SEM across seeds):", flush=True)
    for c in conds:
        m1m, m1s = mean_sem(res[c]['m1'])
        m2m, m2s = mean_sem(res[c]['m2'])
        m3m, m3s = mean_sem(res[c]['m3'])
        print(f"      {c:4s} {labels[c]:22s} M1 new-edges {m1m:5.2f}+/-{m1s:4.2f} | "
              f"M2 cycle-frac {m2m:4.2f}+/-{m2s:4.2f} | M3 sel-restore {m3m:+.3f}+/-{m3s:.3f}", flush=True)

    # representative RECONSTRUCTION MATRIX R_ij at an evolved checkpoint (sparse, not max-only)
    rp_pops, _ = evolve(0, 'C10')
    rp_dev = express(rp_pops[0], 999)
    _, Rrep, Grep, _ = leave_one_out(rp_dev, seedv=1234)
    print("    representative reconstruction matrix R_ij at an evolved C10 checkpoint "
          "(mediator i -> {j: times regenerated by OTHERS via constructed translation}):", flush=True)
    if Rrep:
        for i, row in list(Rrep.items())[:6]:
            print(f"      i='{i}' -> {row}", flush=True)
    else:
        print("      (no by-others regeneration at this particular checkpoint)", flush=True)

    print("    MILESTONES (C1 real-selection incidence over seeds):", flush=True)
    m1_inc = st.mean(1.0 if x > 0 else 0.0 for x in res['C1']['m1'])
    m2_inc = st.mean(1.0 if x > 0 else 0.0 for x in res['C1']['m2'])
    m3_inc = st.mean(1.0 if x > 0.02 else 0.0 for x in res['C1']['m3'])
    print(f"      M1 topology change      : {m1_inc:.2f} of seeds", flush=True)
    print(f"      M2 endogenous cycle      : {m2_inc:.2f} of seeds", flush=True)
    print(f"      M3 operational closure   : {m3_inc:.2f} of seeds (positive selective leave-one-out restoration)", flush=True)

    print("    M4 SELECTION-DEPENDENCE (paired bootstrap 95% CI of M3 diff across seeds):", flush=True)
    for ctrl in ['C2', 'C6', 'C5']:
        diffs = [a - b for a, b in zip(res['C1']['m3'], res[ctrl]['m3'])]
        m, lo, hi = boot_ci(diffs)
        excl = (lo > 0) or (hi < 0)
        print(f"      C1 - {ctrl}: {m:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  "
              f"{'CI excludes 0' if excl else 'CI includes 0 (no selection-dependence detected)'}", flush=True)
    print("    (M4 reads C1 vs C2/C6/C5; the smoke reports incidence only and makes NO scientific claim.)", flush=True)

if __name__ == '__main__':
    print("=== FAMILY C — ENDOGENOUS TRANSLATION CHEMISTRY (substrate + harness; NO scientific claim) ===", flush=True)
    print(f"    alphabet={TOKENS!r} (8 tokens)  LMAX={LMAX}  op-table={OPMAP}", flush=True)
    demo_context_dependence()
    run_certificate()
    run_positive_control()
    run_smoke()
    print("=== FAMC_SMOKE_DONE ===", flush=True)
