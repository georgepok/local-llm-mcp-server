"""entity_substrate.py — ESV1 Pure characterization probe.
NO training, NO actions, NO worlds. Studies closed-loop dynamics of
persistent slot S coupled into frozen LLM. Gate + field are random-init.
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.set_float32_matmul_precision('high')

import habitat_evo as H
import slots as SL

dev   = H.dev
model = H.model
tok   = H.tok

# ── ENV CONFIG (copied verbatim from world_pop) ───────────────────────────────
SEED     = int(os.environ.get('SEED', '0'))
torch.manual_seed(SEED)
np.random.seed(SEED)

CKDIR    = os.environ.get('CKDIR', '/home/pokazge/checkpoints')
os.makedirs(CKDIR, exist_ok=True)

D_MODEL  = (model.config.hidden_size
            if getattr(model.config, 'hidden_size', None)
            else getattr(model.config.text_config, 'hidden_size', 5120))
N_LAYERS = (model.config.num_hidden_layers
            if getattr(model.config, 'num_hidden_layers', None)
            else getattr(model.config.text_config, 'num_hidden_layers', 64))

K           = int(os.environ.get('K',            '12'))
SLOW_K      = int(os.environ.get('SLOW_K',        '6'))
D_S         = int(os.environ.get('D_S',          '768'))
FIELD_LAYERS = [int(x) for x in os.environ.get('FIELD_LAYERS', '40,48,56').split(',')]
EPS         = float(os.environ.get('EPS',        '0.12'))
N_INITS     = int(os.environ.get('N_INITS',      '20'))
T_TICKS     = int(os.environ.get('T_TICKS',      '50'))
EPS_SWEEP   = os.environ.get('EPS_SWEEP', '')
CONTENT     = os.environ.get('CONTENT', '')          # '1' → content_characterize()
CONTENT_INITS = int(os.environ.get('CONTENT_INITS', '6'))
ESV2          = os.environ.get('ESV2', '')                 # '1' → esv2_train()
ESV2_EVAL     = os.environ.get('ESV2_EVAL', '')            # '1' → esv2_eval()
ESV2_CKPT     = os.environ.get('ESV2_CKPT', '')            # explicit ckpt path for eval
ESV2_N_CI     = int(os.environ.get('ESV2_N_CI', '6'))
ESV2_STEPS    = int(os.environ.get('ESV2_STEPS', '1500'))
ESV2_LR       = float(os.environ.get('ESV2_LR', '3e-4'))
ESV2_T_DIFF   = int(os.environ.get('ESV2_T_DIFF', '8'))
ESV2_REFRESH  = int(os.environ.get('ESV2_REFRESH', '150'))
ESV2_TAU      = float(os.environ.get('ESV2_TAU', '0.1'))
ESV2_LDIV     = float(os.environ.get('ESV2_LDIV', '0.5'))
ESV2_FIELDOFF = os.environ.get('ESV2_FIELDOFF', '')        # '1' → content probe with field OFF (anti-router control)
ESV2_SETTLE   = int(os.environ.get('ESV2_SETTLE', '50'))   # ticks to settle states in esv2_collect
ESV2_REWARM   = int(os.environ.get('ESV2_REWARM', '8'))    # warm-start re-settle ticks per refresh (tracks moving attractor)
ESV2_NOEVAL   = os.environ.get('ESV2_NOEVAL', '')          # '1' → skip auto-eval (fast probe)
ESV2_LOG      = int(os.environ.get('ESV2_LOG', '20'))      # metric log interval
ESV2_DIVFLOOR = float(os.environ.get('ESV2_DIVFLOOR', '0.2'))  # slot-diversity floor (hinge anti-collapse)
ESV3           = os.environ.get('ESV3', '')
ESV3_N_INITS   = int(os.environ.get('ESV3_N_INITS', '20'))
ESV3_T_SETTLE  = int(os.environ.get('ESV3_T_SETTLE', '40'))
ESV3_T_POST    = int(os.environ.get('ESV3_T_POST', '40'))
ESV3_N_CI      = int(os.environ.get('ESV3_N_CI', '3'))
ESV3_LATE_FRAC = float(os.environ.get('ESV3_LATE_FRAC', '0.25'))
ESV3_ALPHAS    = os.environ.get('ESV3_ALPHAS', '0.5,1.0,2.0,5.0,10.0,20.0')
ESV3_THETA     = os.environ.get('ESV3_THETA', '0.25,0.5,0.75,1.0')
ESV3_P1_RECOV  = int(os.environ.get('ESV3_P1_RECOV', '4'))

ESV4           = os.environ.get('ESV4', '')
ESV4_EVAL      = os.environ.get('ESV4_EVAL', '')
ESV4_CKPT      = os.environ.get('ESV4_CKPT', '')
ESV4_SURR_DATA = int(os.environ.get('ESV4_SURR_DATA', '350'))
ESV4_SURR_STEPS= int(os.environ.get('ESV4_SURR_STEPS', '2000'))
ESV4_STEPS     = int(os.environ.get('ESV4_STEPS', '1000'))
ESV4_LR        = float(os.environ.get('ESV4_LR', '1e-4'))
ESV4_T_ROLL    = int(os.environ.get('ESV4_T_ROLL', '8'))
ESV4_T_LATE    = int(os.environ.get('ESV4_T_LATE', '4'))
ESV4_K_REFRESH = int(os.environ.get('ESV4_K_REFRESH', '50'))
ESV4_K_EVAL    = int(os.environ.get('ESV4_K_EVAL', '200'))
ESV4_VEL_FLOOR = float(os.environ.get('ESV4_VEL_FLOOR', '0.01'))
ESV4_MARGIN    = float(os.environ.get('ESV4_MARGIN', '0.05'))
ESV4_L_VEL     = float(os.environ.get('ESV4_L_VEL', '2.0'))
ESV4_L_MARGIN  = float(os.environ.get('ESV4_L_MARGIN', '1.0'))
ESV4_L_MULTI   = float(os.environ.get('ESV4_L_MULTI', '0.5'))
ESV4_EVAL_NINITS = int(os.environ.get('ESV4_EVAL_NINITS', '8'))
ESV4_SURR_ONLY = os.environ.get('ESV4_SURR_ONLY', '')

ESV4D           = os.environ.get('ESV4D', '')
ESV4D_STEPS     = int(os.environ.get('ESV4D_STEPS', '300'))
ESV4D_POOL      = int(os.environ.get('ESV4D_POOL', '120'))
ESV4D_ALPHAS    = os.environ.get('ESV4D_ALPHAS', '0.0,0.5,1.0,2.0')

ESV4E           = os.environ.get('ESV4E', '')
ESV4E_STEPS     = int(os.environ.get('ESV4E_STEPS', '120'))
ESV4E_K         = int(os.environ.get('ESV4E_K', '3'))        # rollout horizon
ESV4E_B         = int(os.environ.get('ESV4E_B', '3'))        # batch (distinct cycles per step)
ESV4E_NCYC      = int(os.environ.get('ESV4E_NCYC', '8'))     # baseline cycles to target
ESV4E_CYCLEN    = int(os.environ.get('ESV4E_CYCLEN', '8'))   # points per cycle
ESV4E_ALPHA     = float(os.environ.get('ESV4E_ALPHA', '1.0'))# perturb magnitude
ESV4E_TAU       = float(os.environ.get('ESV4E_TAU', '0.05')) # softmin temperature for cycle distance

# ── CRITICAL: LOOP_READ_LAYER must be DOWNSTREAM of all field writes ──────────
# max(FIELD_LAYERS) = 56. If LOOP_READ_LAYER ≤ 56 the S→field→h→S loop
# closure is 0 BY CONSTRUCTION (read happens before the field has written).
# N_LAYERS - 4 = 60 > 56 = max(FIELD_LAYERS).  This is non-negotiable.
LOOP_READ_LAYER = N_LAYERS - 4  # e.g. 64-4=60 for a 64-layer model

print('ESV1 | D_MODEL=%d N_LAYERS=%d LOOP_READ=%d K=%d SLOW_K=%d D_S=%d '
      'FIELD_LAYERS=%s EPS=%.2f SEED=%d CKDIR=%s' % (
      D_MODEL, N_LAYERS, LOOP_READ_LAYER, K, SLOW_K, D_S,
      FIELD_LAYERS, EPS, SEED, CKDIR), flush=True)

assert LOOP_READ_LAYER > max(FIELD_LAYERS), (
    'LOOP_READ_LAYER=%d must exceed max(FIELD_LAYERS)=%d — closure is zero otherwise' % (
        LOOP_READ_LAYER, max(FIELD_LAYERS)))


# ── FROZEN SUBSTRATE — AdaptiveGateSlot (copied VERBATIM from world_pop) ──────
class AdaptiveGateSlot(nn.Module):  # FROZEN
    def __init__(s, d_model, d_s, K, slow_k, heads=4):
        super().__init__(); s.d_s, s.K, s.slow_k, s.heads, s.dh = d_s, K, slow_k, heads, d_s // heads
        s.read_in = nn.Linear(d_model, d_s)
        s.q, s.k, s.v = nn.Linear(d_s, d_s), nn.Linear(d_s, d_s), nn.Linear(d_s, d_s)
        s.gru = nn.GRUCell(d_s, d_s); s.ln = nn.LayerNorm(d_s)
        s.f_write = nn.Sequential(nn.Linear(2 * d_s, 128), nn.GELU(), nn.Linear(128, 1))
        s.S0 = nn.Parameter(torch.randn(K, d_s) * 0.02)
    def init(s): return s.S0.clone()
    def step(s, S, Hh):
        Hp  = s.read_in(Hh.float())
        Q   = s.q(S).view(s.K, s.heads, s.dh).transpose(0, 1)
        Kk  = s.k(Hp).view(-1, s.heads, s.dh).transpose(0, 1)
        Vv  = s.v(Hp).view(-1, s.heads, s.dh).transpose(0, 1)
        a   = torch.softmax((Q @ Kk.transpose(-1, -2)) / (s.dh ** 0.5), dim=-1)
        ctx = (a @ Vv).transpose(0, 1).reshape(s.K, s.d_s)
        C   = s.gru(ctx, S)
        gw  = torch.sigmoid(s.f_write(torch.cat([S, Hp.mean(0, keepdim=True).expand(s.K, -1)], -1)))
        return s.ln(S + gw * (C - S))
    @property
    def slow(s): return slice(0, s.slow_k)


# ── FIELD FEEDBACK DICT + HOOK INSTALL (same pattern as world_pop _install) ───
_fb = {'fields': None, 'S': None, 'on': False}
_ESV4_HOFF = None  # fp32 [T_tok, D_MODEL] baseline hidden; set by esv4_train()


def _install():
    hs = []
    for L in FIELD_LAYERS:
        def mk(L):
            def hook(mod, inp, out):
                if not _fb['on']:
                    return out
                h  = out[0] if isinstance(out, tuple) else out
                h2 = _fb['fields'][L](h, _fb['S'])
                return ((h2,) + tuple(out[1:])) if isinstance(out, tuple) else h2
            return hook
        hs.append(model.model.layers[L].register_forward_hook(mk(L)))
    return hs


# ── GATE + FIELDS — random-init, eval(), no grad ──────────────────────────────
g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
g.eval()
for p in g.parameters():
    p.requires_grad_(False)

_fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev)
                 for L in FIELD_LAYERS}
for L in FIELD_LAYERS:
    _fb['fields'][L].eval()
    for p in _fb['fields'][L].parameters():
        p.requires_grad_(False)

_HANDLES = _install()

# ── FIXED PROMPT — single invariant input; dynamics come from S alone ─────────
FIXED_PROMPT = "Status: nominal."
fixed_ids = tok(
    H.tmpl([{'role': 'user', 'content': FIXED_PROMPT}]),
    return_tensors='pt'
).input_ids.to(dev)


# ── HELPERS — all in DIRECTION space ──────────────────────────────────────────
# NOTE: LayerNorm inside AdaptiveGateSlot keeps ||S[k]|| ≈ const (~96).
#       NEVER use norm as a signal — use cosine / angular distance only.

def ang(a, b):
    """Angular distance (radians) between two tensors after flattening."""
    af = a.reshape(1, -1).float()
    bf = b.reshape(1, -1).float()
    cos = F.cosine_similarity(af, bf).clamp(-1.0, 1.0)
    return float(torch.acos(cos).item())


def diversity(S):
    """Mean pairwise (1 - cosine) over all slot pairs i < j. Range [0,1]."""
    Sn   = F.normalize(S.float(), dim=-1)       # [K, D_S]
    sims = Sn @ Sn.t()                           # [K, K]
    K_   = S.shape[0]
    mask = torch.triu(torch.ones(K_, K_, device=S.device), diagonal=1).bool()
    return float((1.0 - sims[mask]).mean().item())


def diversity_t(S):
    """Grad-friendly version of diversity: returns tensor (no .item())."""
    Sn   = F.normalize(S.float(), dim=-1)       # [K, D_S]
    sims = Sn @ Sn.t()                           # [K, K]
    K_   = S.shape[0]
    mask = torch.triu(torch.ones(K_, K_, device=S.device), diagonal=1).bool()
    return (1.0 - sims[mask]).mean()


def velocity(S_t, S_prev):
    """Angular displacement between successive slot-state matrices."""
    return ang(S_t, S_prev)


# ── TICK — one closed-loop step ───────────────────────────────────────────────
@torch.no_grad()
def TICK(S_t):
    """Returns (S_next_on, S_next_off).
    S_next_on : field ON — S_t is injected into LLM via field hooks; LLM output
                 is read at LOOP_READ_LAYER (downstream of all field writes) and
                 fed back to gate.  AUTONOMOUS LOOP advances via this branch.
    S_next_off: field OFF — same S_t, but field hooks are no-ops; H_off is the
                 baseline LLM response independent of S.  Used ONLY for the
                 closure metric (isolates LLM-mediated path).
    """
    _fb['S'] = S_t
    _fb['on'] = True
    ho_on = model(fixed_ids, output_hidden_states=True)
    H_on  = ho_on.hidden_states[LOOP_READ_LAYER][0].float()
    _fb['on'] = False

    ho_off = model(fixed_ids, output_hidden_states=True)
    H_off  = ho_off.hidden_states[LOOP_READ_LAYER][0].float()

    S_next_on  = g.step(S_t, H_on)
    S_next_off = g.step(S_t, H_off)
    return S_next_on, S_next_off


@torch.no_grad()
def _tick_on_only(S_t):
    """Field-ON tick; skips the off-pass (BLOCK B / C use this for efficiency)."""
    _fb['S'] = S_t
    _fb['on'] = True
    ho = model(fixed_ids, output_hidden_states=True)
    H  = ho.hidden_states[LOOP_READ_LAYER][0].float()
    _fb['on'] = False
    return g.step(S_t, H)


@torch.no_grad()
def _tick_on_p(S_t, prompt_ids):
    """Field-ON tick driven by an ARBITRARY prompt (not the global fixed one).
    Used by content_characterize() to vary the LLM's context while the loop runs.
    """
    _fb['S'] = S_t
    _fb['on'] = True
    ho = model(prompt_ids, output_hidden_states=True)
    H  = ho.hidden_states[LOOP_READ_LAYER][0].float()
    _fb['on'] = False
    return g.step(S_t, H)


@torch.no_grad()
def _tick_on_p_off(S_t, prompt_ids):
    """Field-OFF tick under an arbitrary prompt: hooks are no-ops; S still updates
    from the BARE prompt-response H (no S-injection). Used by the anti-router
    control — does g organize attractors by content WITHOUT the loop?"""
    _fb['S'] = S_t
    _fb['on'] = False
    ho = model(prompt_ids, output_hidden_states=True)
    H  = ho.hidden_states[LOOP_READ_LAYER][0].float()
    return g.step(S_t, H)


@torch.no_grad()
def _tick_with_hoff(S_t, H_off_pre):
    """Field-ON tick using a precomputed H_off (saves one LLM forward pass).
    Returns (S_next_on, S_next_off) both derived from S_t, mirroring TICK()
    but reusing the caller-supplied H_off_pre instead of re-running the LLM.
    """
    _fb['S'] = S_t
    _fb['on'] = True
    ho  = model(fixed_ids, output_hidden_states=True)
    H_on = ho.hidden_states[LOOP_READ_LAYER][0].float()
    _fb['on'] = False
    return g.step(S_t, H_on), g.step(S_t, H_off_pre)


# ── ANALYSIS UTILITIES ────────────────────────────────────────────────────────
def _kl_hist(p_vals, q_vals, n_bins=20):
    """KL(p||q) estimated via histograms with Laplace smoothing."""
    all_v = list(p_vals) + list(q_vals)
    lo, hi = min(all_v), max(all_v)
    if hi == lo:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    ph, _ = np.histogram(p_vals, bins=bins)
    qh, _ = np.histogram(q_vals, bins=bins)
    ph = ph.astype(float) + 1e-10;  ph /= ph.sum()
    qh = qh.astype(float) + 1e-10;  qh /= qh.sum()
    return float(np.sum(ph * np.log(ph / qh)))


def _top5_pca_frac(traj):
    """Cumulative variance fraction of top-5 PCs of trajectory S-sequence."""
    if len(traj) < 6:
        return float('nan')
    X = torch.stack([s.float().cpu().flatten() for s in traj])  # [T, K*D_S]
    X = X - X.mean(0, keepdim=True)
    try:
        _, sv, _ = torch.linalg.svd(X, full_matrices=False)
        var = sv ** 2
        return float((var[:5].sum() / (var.sum() + 1e-12)).item())
    except Exception:
        return float('nan')


def _detect_period(seq, max_period=10):
    """Returns dominant autocorr period (2-10) if corr > 0.5, else -1."""
    arr = np.array(seq, dtype=float)
    if len(arr) < max_period * 2:
        return -1
    arr = arr - arr.mean()
    best_lag, best_corr = -1, -1.0
    for lag in range(2, min(max_period + 1, len(arr) // 2)):
        corr = float(np.corrcoef(arr[:-lag], arr[lag:])[0, 1])
        if not math.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_lag  = lag
    return best_lag if best_corr > 0.5 else -1


def _classify(vel_late, div_late):
    """Classify trajectory type from late-phase velocity and diversity."""
    mv = float(np.mean(vel_late))
    vd = float(np.var(div_late))
    if mv < 0.01 and vd < 1e-3:
        return 'FIXED_POINT'
    if _detect_period(vel_late) >= 2:
        return 'LIMIT_CYCLE'
    return 'BOUNDED_WANDERING'


def _dist_stats(vals):
    arr = np.array(vals)
    return (float(arr.mean()), float(arr.std()),
            float(np.percentile(arr, 10)), float(np.percentile(arr, 90)))


# ═══════════════════════════════════════════════════════════════════════════════
def eps_sweep():
    """Sweep EPS_SWEEP values; H_off precomputed once (field-independent).

    Interpretation: rising closure with EPS but full_div ≈ frozen_div ⇒
    GRU+LN swamps the loop (fundamental bottleneck); full_div diverging
    from frozen_div with structure ⇒ closure was the bottleneck.
    """
    eps_vals   = [float(x) for x in EPS_SWEEP.split(',') if x.strip()]
    late_start = T_TICKS // 2 + 1

    # H_off is eps-independent (field OFF, fixed prompt) — precompute once
    with torch.no_grad():
        _fb['on'] = False
        H_off_pre = (model(fixed_ids, output_hidden_states=True)
                     .hidden_states[LOOP_READ_LAYER][0].float())

    print('\n=== ESV1_SWEEP: eps=%s N_INITS=%d T_TICKS=%d ===' % (
          EPS_SWEEP, N_INITS, T_TICKS), flush=True)

    for eps in eps_vals:
        # Rebuild fields with fixed seed so all eps share the same random directions
        torch.manual_seed(SEED)
        _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=eps).to(dev)
                         for L in FIELD_LAYERS}
        for L in FIELD_LAYERS:
            _fb['fields'][L].eval()
            for p in _fb['fields'][L].parameters():
                p.requires_grad_(False)

        # Inits: same pattern as BLOCK B (5 alphas × 4 repeats), trimmed to N_INITS
        alphas = [0.0, 0.5, 1.0, 2.0, 5.0]
        rng_b  = torch.Generator(); rng_b.manual_seed(42)
        inits  = []
        for alpha in alphas:
            for _ in range(4):
                noise = torch.randn(K, D_S, generator=rng_b).to(dev)
                inits.append(g.init() + alpha * noise)
        inits = inits[:N_INITS]

        closure_angles  = []
        div_full_all    = []
        vel_full_all    = []
        div_frozen_all  = []
        vel_frozen_all  = []

        for S0 in inits:
            S_f = S0.clone()
            S_z = S0.clone()   # independent frozen trajectory
            for t in range(T_TICKS):
                # Full (field-ON) tick + closure comparison, both from current S_f
                S_f_next, S_z_compare = _tick_with_hoff(S_f, H_off_pre)
                closure_angles.append(ang(S_f_next, S_z_compare))
                # Frozen trajectory advances independently from S_z
                with torch.no_grad():
                    S_z_next = g.step(S_z, H_off_pre)
                if t >= late_start:
                    div_full_all.append(diversity(S_f_next))
                    vel_full_all.append(velocity(S_f_next, S_f))
                    div_frozen_all.append(diversity(S_z_next))
                    vel_frozen_all.append(velocity(S_z_next, S_z))
                S_f = S_f_next
                S_z = S_z_next

        CLOSURE_SCORE   = float(np.mean(closure_angles))
        mean_full_div   = float(np.mean(div_full_all))   if div_full_all   else float('nan')
        mean_frozen_div = float(np.mean(div_frozen_all)) if div_frozen_all else float('nan')
        mean_full_vel   = float(np.mean(vel_full_all))   if vel_full_all   else float('nan')
        mean_frozen_vel = float(np.mean(vel_frozen_all)) if vel_frozen_all else float('nan')
        kl              = (_kl_hist(div_full_all, div_frozen_all)
                           if div_full_all and div_frozen_all else float('nan'))
        distinguishable = (kl > 0.1) if not math.isnan(kl) else False

        print('SWEEP eps=%.2f closure=%.4f full_div=%.4f frozen_div=%.4f '
              'full_vel=%.4f frozen_vel=%.4f kl=%.4f distinguishable=%s' % (
              eps, CLOSURE_SCORE, mean_full_div, mean_frozen_div,
              mean_full_vel, mean_frozen_vel, kl, distinguishable), flush=True)

    print('=== ESV1_SWEEP_DONE ===', flush=True)
    print('=== ESV1_END ===', flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
def characterize():
    T_WARM      = 50    # warmup ticks for BLOCK A
    LATE_START  = T_TICKS // 2 + 1   # t > T_TICKS//2 = "late phase"
    N_C_TRAJ    = 5     # trajectories for BLOCK C
    T_STAR      = 25    # perturbation point
    T_POST      = 25    # ticks after perturbation

    # ── Pre-compute H_off (constant: field OFF, fixed prompt → same every call) ─
    with torch.no_grad():
        _fb['on'] = False
        _H_off_pre = (model(fixed_ids, output_hidden_states=True)
                      .hidden_states[LOOP_READ_LAYER][0].float())

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK A — CLOSURE
    # ═══════════════════════════════════════════════════════════════════════════
    print('\n=== ESV1_BLOCK_A: CLOSURE ===', flush=True)
    S = g.init()
    closure_angles  = []
    per_slot_close  = [[] for _ in range(K)]

    for _ in range(T_WARM):
        S_on, S_off = TICK(S)
        closure_angles.append(ang(S_on, S_off))
        for k in range(K):
            per_slot_close[k].append(ang(S_on[k], S_off[k]))
        S = S_on  # advance FULL loop

    CLOSURE_SCORE = float(np.mean(closure_angles))
    print('CLOSURE_SCORE=%.4f rad  (null<0.05  random~0.05-0.15  structured>0.15)' %
          CLOSURE_SCORE, flush=True)
    for k in range(K):
        print('  slot[%2d] closure=%.4f' % (k, float(np.mean(per_slot_close[k]))), flush=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK B — SELF-MAINTENANCE
    # ═══════════════════════════════════════════════════════════════════════════
    print('\n=== ESV1_BLOCK_B: SELF-MAINTENANCE ===', flush=True)
    alphas   = [0.0, 0.5, 1.0, 2.0, 5.0]
    rng_b    = torch.Generator(); rng_b.manual_seed(42)
    inits    = []
    for alpha in alphas:
        for _ in range(4):
            noise = torch.randn(K, D_S, generator=rng_b).to(dev)
            inits.append(g.init() + alpha * noise)

    traj_full   = []   # 20 × (T+1) slot tensors
    div_full    = []   # 20 × T floats
    vel_full    = []   # 20 × T floats
    traj_frozen = []
    div_frozen  = []
    vel_frozen  = []

    for i, S0 in enumerate(inits):
        tf_list   = [S0.clone()]; tf_div  = []; tf_vel  = []
        tfz_list  = [S0.clone()]; tfz_div = []; tfz_vel = []
        S_f   = S0.clone()
        S_frz = S0.clone()

        for _ in range(T_TICKS):
            # FULL loop — field ON
            S_f_next = _tick_on_only(S_f)
            tf_div.append(diversity(S_f_next))
            tf_vel.append(velocity(S_f_next, S_f))
            tf_list.append(S_f_next.clone())
            S_f = S_f_next

            # FROZEN loop — field OFF (H_off_pre is constant, independent of S)
            with torch.no_grad():
                S_frz_next = g.step(S_frz, _H_off_pre)
            tfz_div.append(diversity(S_frz_next))
            tfz_vel.append(velocity(S_frz_next, S_frz))
            tfz_list.append(S_frz_next.clone())
            S_frz = S_frz_next

        traj_full.append(tf_list);   div_full.append(tf_div);   vel_full.append(tf_vel)
        traj_frozen.append(tfz_list); div_frozen.append(tfz_div); vel_frozen.append(tfz_vel)

        if (i + 1) % 5 == 0:
            print('  init %d/%d done' % (i + 1, N_INITS), flush=True)

    # Save trajectories
    traj_cache = os.path.join(CKDIR, 'esv1_traj_s%d.pt' % SEED)
    torch.save({'traj_full':   [[s.cpu() for s in tl] for tl in traj_full],
                'traj_frozen': [[s.cpu() for s in tl] for tl in traj_frozen],
                'div_full': div_full,   'vel_full': vel_full,
                'div_frozen': div_frozen, 'vel_frozen': vel_frozen,
                'SEED': SEED, 'K': K, 'SLOW_K': SLOW_K, 'D_S': D_S},
               traj_cache)
    print('cached → %s' % traj_cache, flush=True)

    # Classify trajectories
    counts = {'FIXED_POINT': 0, 'LIMIT_CYCLE': 0, 'BOUNDED_WANDERING': 0}
    for i in range(N_INITS):
        cl = _classify(vel_full[i][LATE_START:], div_full[i][LATE_START:])
        counts[cl] += 1

    # Late-phase statistics
    late_div_full   = [d for i in range(N_INITS) for d in div_full[i][LATE_START:]]
    late_vel_full   = [v for i in range(N_INITS) for v in vel_full[i][LATE_START:]]
    late_div_frozen = [d for i in range(N_INITS) for d in div_frozen[i][LATE_START:]]
    late_vel_frozen = [v for i in range(N_INITS) for v in vel_frozen[i][LATE_START:]]

    mean_late_div_full   = float(np.mean(late_div_full))
    mean_late_vel_full   = float(np.mean(late_vel_full))
    mean_late_div_frozen = float(np.mean(late_div_frozen))
    mean_late_vel_frozen = float(np.mean(late_vel_frozen))

    # Inter-init distance at t=50
    S_finals   = [traj_full[i][T_TICKS] for i in range(N_INITS)]
    inter_sum  = 0.0; n_pairs = 0
    for ii in range(N_INITS):
        for jj in range(ii + 1, N_INITS):
            inter_sum += ang(S_finals[ii], S_finals[jj])
            n_pairs   += 1
    inter_init_dist = inter_sum / max(1, n_pairs)

    # PCA top-5 variance fraction (mean over 20 trajectories)
    pca_fracs    = [_top5_pca_frac(traj_full[i][1:]) for i in range(N_INITS)]
    mean_pca5    = float(np.nanmean(pca_fracs))

    # KL between FULL and FROZEN late-diversity histograms
    kl_div_fb = _kl_hist(late_div_full, late_div_frozen)

    print('CLASSIFICATION (full loop, %d inits): %s' % (N_INITS, counts), flush=True)
    print('mean_late_div  FULL=%.4f  FROZEN=%.4f' % (mean_late_div_full, mean_late_div_frozen), flush=True)
    print('mean_late_vel  FULL=%.4f  FROZEN=%.4f' % (mean_late_vel_full, mean_late_vel_frozen), flush=True)
    print('inter_init_dist_at_t%d=%.4f' % (T_TICKS, inter_init_dist), flush=True)
    print('mean_top5_pca_frac=%.4f' % mean_pca5, flush=True)
    print('KL(full||frozen)=%.4f  (>0.1 → distinguishable)' % kl_div_fb, flush=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK C — PERTURBATION RECOVERY
    # ═══════════════════════════════════════════════════════════════════════════
    print('\n=== ESV1_BLOCK_C: PERTURBATION RECOVERY ===', flush=True)
    perturb_alphas     = [0.1, 0.5, 1.0, 2.0]
    recovery_by_alpha  = {pa: [] for pa in perturb_alphas}
    slow_recov_by_alpha = {pa: [] for pa in perturb_alphas}
    fast_recov_by_alpha = {pa: [] for pa in perturb_alphas}
    # ANTI-MASQUERADE arm: recovery under FIELD-OFF (bare GRU+LN) from the SAME
    # full-loop state S_star, SAME perturbation. If full ≈ frozen the contraction
    # is the GRU (not the loop); only full ≪ frozen is loop-induced self-maintenance.
    frozen_recov_by_alpha = {pa: [] for pa in perturb_alphas}
    rng_c = torch.Generator(); rng_c.manual_seed(99)

    for ti in range(N_C_TRAJ):
        # Advance FULL loop to t* = 25
        S_run = inits[ti].clone()
        for _ in range(T_STAR):
            S_run = _tick_on_only(S_run)
        S_star = S_run.clone()

        # Control: 25 more ticks from S_star (unperturbed) — FULL loop
        ctrl = []
        S_ctrl = S_star.clone()
        for _ in range(T_POST):
            S_ctrl = _tick_on_only(S_ctrl)
            ctrl.append(S_ctrl.clone())

        # Control: 25 more ticks from S_star — FROZEN (field OFF, bare GRU)
        ctrl_z = []
        S_ctrl_z = S_star.clone()
        for _ in range(T_POST):
            S_ctrl_z = g.step(S_ctrl_z, _H_off_pre)
            ctrl_z.append(S_ctrl_z.clone())

        for pa in perturb_alphas:
            noise     = torch.randn(K, D_S, generator=rng_c).to(dev)
            S_perturb = S_star.clone() + pa * noise
            D0        = ang(S_perturb, S_star)
            D0_slow   = ang(S_perturb[:SLOW_K], S_star[:SLOW_K])
            D0_fast   = ang(S_perturb[SLOW_K:], S_star[SLOW_K:])

            # FULL-loop recovery from the perturbed state
            perturb = []
            S_p = S_perturb.clone()
            for _ in range(T_POST):
                S_p = _tick_on_only(S_p)
                perturb.append(S_p.clone())

            # FROZEN recovery from the SAME perturbed state (field OFF)
            perturb_z = []
            S_pz = S_perturb.clone()
            for _ in range(T_POST):
                S_pz = g.step(S_pz, _H_off_pre)
                perturb_z.append(S_pz.clone())

            Dk      = [ang(perturb[t], ctrl[t]) for t in range(T_POST)]
            Dk_slow = [ang(perturb[t][:SLOW_K], ctrl[t][:SLOW_K]) for t in range(T_POST)]
            Dk_fast = [ang(perturb[t][SLOW_K:], ctrl[t][SLOW_K:]) for t in range(T_POST)]
            Dk_z    = [ang(perturb_z[t], ctrl_z[t]) for t in range(T_POST)]

            eps = 1e-12
            recovery_by_alpha[pa].append(min(Dk)      / (D0      + eps) if D0      > eps else float('nan'))
            slow_recov_by_alpha[pa].append(min(Dk_slow) / (D0_slow + eps) if D0_slow > eps else float('nan'))
            fast_recov_by_alpha[pa].append(min(Dk_fast) / (D0_fast + eps) if D0_fast > eps else float('nan'))
            frozen_recov_by_alpha[pa].append(min(Dk_z) / (D0 + eps) if D0 > eps else float('nan'))

        print('  traj %d/%d done' % (ti + 1, N_C_TRAJ), flush=True)

    loop_helps_count = 0
    for pa in perturb_alphas:
        ri   = float(np.nanmean(recovery_by_alpha[pa]))
        ri_s = float(np.nanmean(slow_recov_by_alpha[pa]))
        ri_f = float(np.nanmean(fast_recov_by_alpha[pa]))
        ri_z = float(np.nanmean(frozen_recov_by_alpha[pa]))
        loop_helps = ri < ri_z - 0.05   # loop recovers meaningfully better than bare GRU
        if loop_helps:
            loop_helps_count += 1
        print('  alpha=%.1f  RECOVERY_INDEX full=%.4f frozen=%.4f  loop_helps=%s '
              '(slow=%.4f fast=%.4f)  [<0.7=attractor  full<frozen=loop-induced]' % (
              pa, ri, ri_z, loop_helps, ri_s, ri_f), flush=True)
    print('  LOOP_SELF_MAINTAINS=%s  (loop recovery beats bare-GRU on %d/%d alphas)' % (
          loop_helps_count >= (len(perturb_alphas) // 2 + 1), loop_helps_count, len(perturb_alphas)),
          flush=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOCK D — VIABILITY
    # ═══════════════════════════════════════════════════════════════════════════
    print('\n=== ESV1_BLOCK_D: VIABILITY ===', flush=True)

    d_mean_f, d_std_f, d_p10_f, d_p90_f = _dist_stats(late_div_full)
    v_mean_f, v_std_f, v_p10_f, v_p90_f = _dist_stats(late_vel_full)
    d_mean_z, d_std_z, d_p10_z, d_p90_z = _dist_stats(late_div_frozen)
    v_mean_z, v_std_z, v_p10_z, v_p90_z = _dist_stats(late_vel_frozen)

    print('  FULL   late div: mean=%.4f std=%.4f p10=%.4f p90=%.4f' % (
          d_mean_f, d_std_f, d_p10_f, d_p90_f), flush=True)
    print('  FROZEN late div: mean=%.4f std=%.4f p10=%.4f p90=%.4f' % (
          d_mean_z, d_std_z, d_p10_z, d_p90_z), flush=True)
    print('  FULL   late vel: mean=%.4f std=%.4f p10=%.4f p90=%.4f' % (
          v_mean_f, v_std_f, v_p10_f, v_p90_f), flush=True)
    print('  FROZEN late vel: mean=%.4f std=%.4f p10=%.4f p90=%.4f' % (
          v_mean_z, v_std_z, v_p10_z, v_p90_z), flush=True)

    # VIABILITY_FRACTION: fraction of 20 inits whose t=50 diversity falls
    # inside the FULL late-phase [p10, p90] band
    d_t50 = [diversity(traj_full[i][T_TICKS]) for i in range(N_INITS)]
    viab_frac = float(sum(1 for d in d_t50 if d_p10_f <= d <= d_p90_f) / N_INITS)
    print('  VIABILITY_FRACTION=%.3f  (fraction of %d inits in [p10,p90] at t=%d)' % (
          viab_frac, N_INITS, T_TICKS), flush=True)

    # ANTI-MASQUERADE: if FULL ≈ FROZEN the dynamics are LN-bounded noise (NULL)
    distinguishable = kl_div_fb > 0.1
    print('  ANTI-MASQUERADE: KL(full||frozen)=%.4f  distinguishable=%s  '
          '(indistinguishable → LN-bounded noise → NULL verdict)' % (
          kl_div_fb, distinguishable), flush=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════════════
    majority_not_fp = counts['FIXED_POINT'] < (N_INITS // 2)
    some_recovery   = any(
        float(np.nanmean(recovery_by_alpha[pa])) < 0.7 for pa in perturb_alphas)

    if (CLOSURE_SCORE > 0.15
            and majority_not_fp
            and mean_late_div_full > 0.1
            and distinguishable
            and some_recovery):
        verdict = 'SELF_MAINTAINING'
    elif CLOSURE_SCORE < 0.05 or not distinguishable:
        verdict = 'NULL_TRIVIAL_OR_LN_NOISE'
    else:
        verdict = 'PARTIAL_STRUCTURE'

    print('\nESV1_VERDICT: %s  '
          'closure=%.4f  fp=%d lc=%d bw=%d  '
          'mean_late_div_full=%.4f  kl=%.4f  '
          'majority_not_fp=%s  some_recovery<0.7=%s' % (
          verdict,
          CLOSURE_SCORE,
          counts['FIXED_POINT'], counts['LIMIT_CYCLE'], counts['BOUNDED_WANDERING'],
          mean_late_div_full,
          kl_div_fb,
          majority_not_fp,
          some_recovery), flush=True)
    print('=== ESV1_DONE ===', flush=True)
    print('=== ESV1_END ===', flush=True)


def _centroid(seq):
    """Phase-averaged attractor descriptor: mean S over a tick sequence [K,D_S]."""
    return torch.stack(seq, 0).mean(0)


def _content_tick(S, pids):
    return _tick_on_p_off(S, pids) if ESV2_FIELDOFF else _tick_on_p(S, pids)


def content_characterize():
    """ESV1.6 — is the (validated) self-maintenance ORGANIZED BY THE LLM COUPLING,
    or a content-blind random-weight artifact?

    ESV1 showed attractors are INIT-determined (inter_init 1.31) at a fixed prompt.
    Discriminator: hold the init FIXED, vary the PROMPT. If the attractor moves
    (between-prompt > ~0.15) AND the loop MIGRATES toward a new prompt's attractor
    when the prompt is switched mid-trajectory → the self-maintained organization
    tracks the LLM's computational state (entity-relevant). If between-prompt ≈ 0
    and no migration → content-blind: 'a recurrent net has attractors', not the
    substrate self-organizing around its LLM coupling.
    """
    PROMPTS = [
        "Status: nominal.",
        "The reactor core temperature is rising rapidly toward meltdown.",
        "She walked along the quiet beach as the sun set over the water.",
        "Solve for x in the equation: three x plus seven equals twenty two.",
        "The board will reconvene next Thursday to review the quarterly budget.",
        "Red green blue yellow orange purple black white silver gold.",
    ]
    M       = len(PROMPTS)
    N_CI    = CONTENT_INITS
    T_SET   = T_TICKS
    L_CENT  = max(6, T_TICKS // 4)   # # late ticks averaged into the centroid

    print('CONTENT_FIELD=%s' % ('OFF' if ESV2_FIELDOFF else 'ON'), flush=True)
    print('\n=== ESV1.6 CONTENT-DEPENDENCE  M=%d prompts  N_INITS=%d  T_SETTLE=%d  EPS=%.2f ===' % (
          M, N_CI, T_SET, EPS), flush=True)

    prompt_ids = [tok(H.tmpl([{'role': 'user', 'content': p}]),
                      return_tensors='pt').input_ids.to(dev) for p in PROMPTS]

    # Shared init set (SAME inits across all prompts → isolates content from init)
    rng = torch.Generator(); rng.manual_seed(7)
    inits = [g.init() + 1.0 * torch.randn(K, D_S, generator=rng).to(dev)
             for _ in range(N_CI)]

    # Settle every (prompt, init) to its attractor; store phase-averaged centroid
    # and the raw final state (a point on the limit cycle, for migration).
    cent = [[None] * N_CI for _ in range(M)]
    last = [[None] * N_CI for _ in range(M)]
    for pi in range(M):
        for ii in range(N_CI):
            S = inits[ii].clone()
            tail = []
            for t in range(T_SET):
                S = _content_tick(S, prompt_ids[pi])
                if t >= T_SET - L_CENT:
                    tail.append(S.clone())
            cent[pi][ii] = _centroid(tail)
            last[pi][ii] = S.clone()
        print('  settled prompt %d/%d' % (pi + 1, M), flush=True)

    # within-prompt (across init, same prompt) = how much INIT determines attractor
    within_vals = []
    for pi in range(M):
        for a in range(N_CI):
            for b in range(a + 1, N_CI):
                within_vals.append(ang(cent[pi][a], cent[pi][b]))
    within_prompt = float(np.mean(within_vals))

    # between-prompt (same init, across prompt) = how much CONTENT shifts attractor
    between_vals = []
    for ii in range(N_CI):
        for pa in range(M):
            for pb in range(pa + 1, M):
                between_vals.append(ang(cent[pa][ii], cent[pb][ii]))
    between_prompt = float(np.mean(between_vals))

    print('  WITHIN_PROMPT  (across init, same prompt) = %.4f rad  [init effect]' % within_prompt, flush=True)
    print('  BETWEEN_PROMPT (same init, across prompt) = %.4f rad  [content effect]' % between_prompt, flush=True)
    print('  content/init ratio = %.3f  (>~0.5 → content matters as much as init)' % (
          between_prompt / (within_prompt + 1e-9)), flush=True)

    # MIGRATION: settle under p, then run under q; does it move toward q's attractor?
    pairs = [(0, 1), (0, 3), (2, 4), (1, 5)]
    print('  --- MIGRATION (settle under p, switch to q) ---', flush=True)
    mig_indices = []
    for (p, q) in pairs:
        per_pair = []
        for ii in range(N_CI):
            S = last[p][ii].clone()
            tail = []
            for t in range(T_SET):
                S = _content_tick(S, prompt_ids[q])
                if t >= T_SET - L_CENT:
                    tail.append(S.clone())
            mig = _centroid(tail)
            d_to_q = ang(mig, cent[q][ii])
            d_to_p = ang(mig, cent[p][ii])
            per_pair.append(d_to_q / (d_to_q + d_to_p + 1e-9))
        mi = float(np.mean(per_pair))
        mig_indices.append(mi)
        print('    p=%d→q=%d  migration_index=%.4f  [<0.5 → moved toward q (content drives)]' % (
              p, q, mi), flush=True)
    mig_mean = float(np.mean(mig_indices))

    # VERDICT
    content_structured = between_prompt > 0.15
    migrates           = mig_mean < 0.45
    if content_structured and migrates:
        verdict = 'CONTENT_ORGANIZED'        # self-maintenance tracks LLM coupling
    elif content_structured or migrates:
        verdict = 'PARTIAL_CONTENT'
    else:
        verdict = 'CONTENT_BLIND'             # recurrent-net artifact, prompt-invariant

    print('\nESV16_VERDICT: %s  between_prompt=%.4f within_prompt=%.4f '
          'mig_mean=%.4f  content_structured=%s migrates=%s' % (
          verdict, between_prompt, within_prompt, mig_mean,
          content_structured, migrates), flush=True)
    print('=== ESV16_DONE ===', flush=True)
    print('=== ESV16_END ===', flush=True)


PROMPTS_ESV2 = [
    "Status: nominal.",
    "The reactor core temperature is rising rapidly toward meltdown.",
    "She walked along the quiet beach as the sun set over the water.",
    "Solve for x in the equation: three x plus seven equals twenty two.",
    "The board will reconvene next Thursday to review the quarterly budget.",
    "Red green blue yellow orange purple black white silver gold.",
]


@torch.no_grad()
def esv2_collect(prompt_ids, inits, T_settle):
    """For each (prompt p, init i): run the REAL closed loop (field ON) T_settle
    ticks via _tick_on_p; cache the settled state S and the LLM hidden H_on at the
    settled state. Returns s_cache[p][i] (detached [K,D_S]) and h_cache[p][i]
    (detached [seq,D_MODEL] — the layer-LOOP_READ hidden at the settled S)."""
    M = len(prompt_ids)
    s_cache = [[None] * len(inits) for _ in range(M)]
    h_cache = [[None] * len(inits) for _ in range(M)]
    for p in range(M):
        for i in range(len(inits)):
            S = inits[i].clone()
            for _ in range(T_settle):
                S = _tick_on_p(S, prompt_ids[p])
            # one more field-ON forward to capture H_on AT the settled S
            _fb['S'] = S; _fb['on'] = True
            ho = model(prompt_ids[p], output_hidden_states=True)
            H  = ho.hidden_states[LOOP_READ_LAYER][0].float()
            _fb['on'] = False
            s_cache[p][i] = S.detach().clone()
            h_cache[p][i] = H.detach().clone()
    return s_cache, h_cache


@torch.no_grad()
def esv2_recollect(prompt_ids, s_prev, rewarm):
    """Warm-start refresh: re-settle each (p,i) from its PREVIOUS settled state for
    `rewarm` ticks with the CURRENT g, then capture H_on. Cheap — tracks the slowly
    moving real attractor as g trains, so the differentiable signal stays anchored to
    the REAL closed-loop fixed points (fixes the decorative-organization failure where
    a full re-settle from inits decoupled the proxy from the real attractors)."""
    M = len(prompt_ids); N = len(s_prev[0])
    s_cache = [[None] * N for _ in range(M)]
    h_cache = [[None] * N for _ in range(M)]
    for p in range(M):
        for i in range(N):
            S = s_prev[p][i].clone()
            for _ in range(rewarm):
                S = _tick_on_p(S, prompt_ids[p])
            _fb['S'] = S; _fb['on'] = True
            ho = model(prompt_ids[p], output_hidden_states=True)
            H  = ho.hidden_states[LOOP_READ_LAYER][0].float()
            _fb['on'] = False
            s_cache[p][i] = S.detach().clone()
            h_cache[p][i] = H.detach().clone()
    return s_cache, h_cache


def esv2_rollout(S0_detached, H_detached, t_diff):
    """Differentiable rollout WITHOUT LLM calls: from a detached settled state,
    apply g.step(S, H_detached) t_diff times. Gradient flows through g.params only
    (H is a constant). NO torch.no_grad here. Returns list of t_diff S tensors."""
    S = S0_detached
    traj = []
    for _ in range(t_diff):
        S = g.step(S, H_detached)
        traj.append(S)
    return traj


def esv2_loss_cac(C, tau, lambda_div):
    """C[p][i] = centroid [K,D_S] (with grad). InfoNCE: positive = same-prompt
    different-init; negatives = different-prompt same-init. label index 0.
    Anti-collapse = HINGE floor: penalty lambda_div * relu(ESV2_DIVFLOOR - mean_div)
    activates ONLY when slot diversity drops below the floor (does not fight the
    contrastive term once diversity is healthy, unlike an unbounded linear reward)."""
    M = len(C); N = len(C[0])
    losses = []
    for p in range(M):
        for i in range(N):
            anchor = C[p][i].flatten().unsqueeze(0)
            # positive: a different init, same prompt (deterministic pick i+1 mod N)
            j = (i + 1) % N
            pos = C[p][j].flatten().unsqueeze(0)
            logits = [F.cosine_similarity(anchor, pos)]
            for q in range(M):
                if q == p: continue
                neg = C[q][i].flatten().unsqueeze(0)
                logits.append(F.cosine_similarity(anchor, neg))
            logits = torch.cat(logits).unsqueeze(0) / tau     # [1, M]
            label = torch.zeros(1, dtype=torch.long, device=logits.device)
            losses.append(F.cross_entropy(logits, label))
    L_nce = torch.stack(losses).mean()
    L_div = torch.stack([diversity_t(C[p][i]) for p in range(M) for i in range(N)]).mean()
    div_pen = F.relu(ESV2_DIVFLOOR - L_div)
    return L_nce + lambda_div * div_pen, L_nce.detach(), L_div.detach()


def esv2_train():
    print('\n=== ESV2_TRAIN  SEED=%d  N_CI=%d  STEPS=%d  LR=%.2e  T_DIFF=%d  '
          'REFRESH=%d  TAU=%.3f  LDIV=%.3f ===' % (
          SEED, ESV2_N_CI, ESV2_STEPS, ESV2_LR, ESV2_T_DIFF,
          ESV2_REFRESH, ESV2_TAU, ESV2_LDIV), flush=True)

    g.train()
    for p in g.parameters():
        p.requires_grad_(True)

    prompt_ids = [tok(H.tmpl([{'role': 'user', 'content': pr}]),
                      return_tensors='pt').input_ids.to(dev) for pr in PROMPTS_ESV2]
    M = len(prompt_ids)

    rng = torch.Generator(); rng.manual_seed(7)
    inits = [g.init() + 1.0 * torch.randn(K, D_S, generator=rng).to(dev)
             for _ in range(ESV2_N_CI)]

    opt   = AdamW(g.parameters(), lr=ESV2_LR, weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=ESV2_STEPS, eta_min=ESV2_LR / 10)

    s_cache, h_cache = esv2_collect(prompt_ids, inits, ESV2_SETTLE)

    for step in range(1, ESV2_STEPS + 1):
        if step % ESV2_REFRESH == 0:
            s_cache, h_cache = esv2_recollect(prompt_ids, s_cache, ESV2_REWARM)

        C = [[_centroid(esv2_rollout(s_cache[p][i], h_cache[p][i], ESV2_T_DIFF))
              for i in range(ESV2_N_CI)] for p in range(M)]

        loss, l_nce, l_div = esv2_loss_cac(C, ESV2_TAU, ESV2_LDIV)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % ESV2_LOG == 0:
            refreshed = (step % ESV2_REFRESH == 0)
            with torch.no_grad():
                within_vals = []
                between_vals = []
                C_det = [[C[p][i].detach() for i in range(ESV2_N_CI)] for p in range(M)]
                for p in range(M):
                    for a in range(ESV2_N_CI):
                        for b in range(a + 1, ESV2_N_CI):
                            within_vals.append(ang(C_det[p][a], C_det[p][b]))
                for i in range(ESV2_N_CI):
                    for pa in range(M):
                        for pb in range(pa + 1, M):
                            between_vals.append(ang(C_det[pa][i], C_det[pb][i]))
                w = float(np.mean(within_vals)) if within_vals else float('nan')
                b = float(np.mean(between_vals)) if between_vals else float('nan')
            print('ESV2 step=%d%s  loss=%.4f  nce=%.4f  div=%.4f  within=%.4f  between=%.4f  ratio=%.3f' % (
                  step, ' [POST-REFRESH]' if refreshed else '',
                  loss.item(), l_nce.item(), l_div.item(), w, b,
                  b / (w + 1e-9)), flush=True)

        if step % 500 == 0:
            torch.save(g.state_dict(),
                       os.path.join(CKDIR, 'esv2_gate_s%d_step%d.pt' % (SEED, step)))

    torch.save(g.state_dict(), os.path.join(CKDIR, 'esv2_gate_s%d.pt' % SEED))
    print('ESV2_TRAIN_DONE', flush=True)

    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)

    if ESV2_NOEVAL:
        print('=== ESV2 NOEVAL (probe) — skipping auto-eval ===', flush=True)
    else:
        print('=== ESV2 AUTO-EVAL: content (field ON) then full characterization ===', flush=True)
        content_characterize()
        characterize()
    print('=== ESV2_DONE ===', flush=True)
    print('=== ESV2_END ===', flush=True)


def esv2_eval():
    path = ESV2_CKPT or os.path.join(CKDIR, 'esv2_gate_s%d.pt' % SEED)
    g.load_state_dict(torch.load(path, map_location=dev))
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    print('ESV2_EVAL loaded %s' % path, flush=True)
    content_characterize()
    characterize()
    print('=== ESV2_DONE ===', flush=True)
    print('=== ESV2_END ===', flush=True)


# ── ESV3: CHARACTERIZATION MODE (no training) ────────────────────────────────

def _settle_and_centroid(S0, tick_fn, T_settle, late_frac):
    """tick_fn(S)->S_next. Run T_settle ticks; centroid = _centroid of last int(T_settle*late_frac) states. Return (S_final, centroid)."""
    n_late = max(2, int(T_settle * late_frac)); S = S0.clone(); tail = []
    for t in range(T_settle):
        S = tick_fn(S)
        if t >= T_settle - n_late:
            tail.append(S.clone())
    return S, _centroid(tail)


def _cluster_states(cents, theta):
    """Union-find connected components: edge if ang(cents[i],cents[j])<theta. Return int n_clusters. Pure compute."""
    n = len(cents); parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if ang(cents[i], cents[j]) < theta:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return len({find(i) for i in range(n)})


def _direction_consistency(cents, eps_angle=0.05):
    """cents[M][N] of [K,D_S]. For each prompt pair (p,q): shift_i=normalize(flatten(cents[q][i]-cents[p][i])); DC_pq=mean cos(shift_i,shift_j) over i<j with ang(cents[q][i],cents[p][i])>eps_angle. Return mean over pairs (float)."""
    M = len(cents); N = len(cents[0]); dcs = []
    for p in range(M):
        for q in range(p + 1, M):
            shifts = []
            for i in range(N):
                if ang(cents[q][i], cents[p][i]) > eps_angle:
                    d = (cents[q][i] - cents[p][i]).flatten()
                    shifts.append(F.normalize(d.unsqueeze(0), dim=-1))
            for a in range(len(shifts)):
                for b in range(a + 1, len(shifts)):
                    dcs.append(float(F.cosine_similarity(shifts[a], shifts[b]).item()))
    return float(np.mean(dcs)) if dcs else float('nan')


def _dc_permuted(cents, n_perm=10):
    """Permute init index per prompt (deterministic permutations via index rotations, NOT RNG — vary by perm index k: i->(i+k)%N for prompt-rows>0), recompute _direction_consistency, return mean. Avoids Math.random/Date restrictions."""
    M = len(cents); N = len(cents[0]); vals = []
    for k in range(1, n_perm + 1):
        permuted = []
        for p in range(M):
            shift = (p * k) % N
            permuted.append([cents[p][(i + shift) % N] for i in range(N)])
        v = _direction_consistency(permuted)
        if not (v != v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float('nan')


@torch.no_grad()
def esv3_p1_basin(H_off_pre):
    print('\n=== ESV3_P1: ATTRACTOR INVENTORY + RETURN-TO-IDENTITY ===', flush=True)
    rng = torch.Generator(); rng.manual_seed(42); inits = []
    for a in [0.0, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(4):
            inits.append(g.init() + a * torch.randn(K, D_S, generator=rng).to(dev))
    inits = inits[:ESV3_N_INITS]
    cents_full = []; cents_frz = []; finals_full = []
    for S0 in inits:
        Sf, cf = _settle_and_centroid(S0, _tick_on_only, ESV3_T_SETTLE, ESV3_LATE_FRAC)
        cents_full.append(cf); finals_full.append(Sf)
        _, cz = _settle_and_centroid(S0, lambda S: g.step(S, H_off_pre), ESV3_T_SETTLE, ESV3_LATE_FRAC)
        cents_frz.append(cz)
    thetas = [float(x) for x in ESV3_THETA.split(',')]
    for th in thetas:
        nf = _cluster_states(cents_full, th); nz = _cluster_states(cents_frz, th)
        print('  theta=%.2f  N_ATTRACT_FULL=%d  N_ATTRACT_FROZEN=%d' % (th, nf, nz), flush=True)
    di = []; dc = []
    for i in range(len(inits)):
        for j in range(i + 1, len(inits)):
            di.append(ang(inits[i], inits[j])); dc.append(ang(cents_full[i], cents_full[j]))
    isc = float(np.corrcoef(di, dc)[0, 1]) if len(di) > 1 else float('nan')
    print('  ISC_FULL(init-vs-centroid corr)=%.4f  mean_pairwise_centroid_ang=%.4f' % (isc, float(np.mean(dc))), flush=True)
    n_av = len(finals_full)
    idxs = sorted(set(int(round(x)) for x in np.linspace(0, n_av - 1, ESV3_P1_RECOV + 2)[1:-1])) if n_av > ESV3_P1_RECOV else list(range(n_av))
    rng_r = torch.Generator(); rng_r.manual_seed(123); rtai_f = []; rtai_z = []
    for ii in idxs:
        Sf = finals_full[ii]
        _, Cctrl = _settle_and_centroid(Sf, _tick_on_only, ESV3_T_POST, ESV3_LATE_FRAC)
        noise = torch.randn(K, D_S, generator=rng_r).to(dev)
        _, Crec_f = _settle_and_centroid(Sf + 2.0 * noise, _tick_on_only, ESV3_T_POST, ESV3_LATE_FRAC)
        _, Crec_z = _settle_and_centroid(Sf + 2.0 * noise, lambda S: g.step(S, H_off_pre), ESV3_T_POST, ESV3_LATE_FRAC)
        dsame_f = ang(Crec_f, cents_full[ii])
        doth_f = min(ang(Crec_f, cents_full[j]) for j in range(len(cents_full)) if j != ii)
        dsame_z = ang(Crec_z, cents_full[ii])
        doth_z = min(ang(Crec_z, cents_full[j]) for j in range(len(cents_full)) if j != ii)
        rtai_f.append(dsame_f / (dsame_f + doth_f + 1e-9))
        rtai_z.append(dsame_z / (dsame_z + doth_z + 1e-9))
    mrf = float(np.mean(rtai_f)); mrz = float(np.mean(rtai_z))
    print('  RTAI_FULL=%.4f  RTAI_FROZEN=%.4f  (0=returns to own attractor, 1=goes elsewhere)' % (mrf, mrz), flush=True)
    nf1 = _cluster_states(cents_full, 1.0); nz1 = _cluster_states(cents_frz, 1.0)
    discrete = nf1 < 8 and (nz1 - nf1) >= 3
    rtai_li = mrf < 0.3 and (mrz - mrf) > 0.15
    print('  P1_VERDICT discrete=%s rtai_loop_induced=%s (Nfull@1.0=%d Nfrozen@1.0=%d)' % (discrete, rtai_li, nf1, nz1), flush=True)
    return {'finals_full': finals_full, 'cents_full': cents_full, 'idxs': idxs, 'discrete': discrete, 'rtai_li': rtai_li}


@torch.no_grad()
def esv3_p2_viability(p1, H_off_pre):
    print('\n=== ESV3_P2: VIABILITY BOUNDARY ===', flush=True)
    alphas = [float(x) for x in ESV3_ALPHAS.split(',')]; idxs = p1['idxs']; finals = p1['finals_full']
    rng = torch.Generator(); rng.manual_seed(77)
    curve_f = {}; curve_z = {}
    for a in alphas:
        rf = []; rz = []; nrec = []
        for ii in idxs:
            Sf = finals[ii]
            ctrl_f = []; Sc = Sf.clone()
            for _ in range(ESV3_T_POST):
                Sc = _tick_on_only(Sc); ctrl_f.append(Sc.clone())
            ctrl_z = []; Scz = Sf.clone()
            for _ in range(ESV3_T_POST):
                Scz = g.step(Scz, H_off_pre); ctrl_z.append(Scz.clone())
            noise = torch.randn(K, D_S, generator=rng).to(dev); Sp = Sf + a * noise; D0 = ang(Sp, Sf)
            pf = []; S = Sp.clone()
            for _ in range(ESV3_T_POST):
                S = _tick_on_only(S); pf.append(S.clone())
            rf.append(min(ang(pf[t], ctrl_f[t]) for t in range(ESV3_T_POST)) / (D0 + 1e-12))
            pz = []; S = Sp.clone()
            for _ in range(ESV3_T_POST):
                S = g.step(S, H_off_pre); pz.append(S.clone())
            rz.append(min(ang(pz[t], ctrl_z[t]) for t in range(ESV3_T_POST)) / (D0 + 1e-12))
            if a >= 10.0:
                nrec.append(float((pf[1].norm() / (Sp.norm() + 1e-9)).item()))
        curve_f[a] = float(np.mean(rf)); curve_z[a] = float(np.mean(rz))
        extra = (' norm_recov@t1=%.3f' % np.mean(nrec)) if nrec else ''
        print('  alpha=%.1f  recov_FULL=%.4f  recov_FROZEN=%.4f%s' % (a, curve_f[a], curve_z[a], extra), flush=True)
    af = sorted(alphas); jumps = [curve_f[af[i + 1]] - curve_f[af[i]] for i in range(len(af) - 1)]
    SI = max(jumps) if jumps else 0.0

    def crit(curve):
        for a in af:
            if curve[a] > 0.7:
                return a
        return af[-1]

    caf = crit(curve_f); caz = crit(curve_z); lve = caf / (caz + 1e-6)
    print('  P2_VERDICT SI_FULL=%.3f CRIT_ALPHA_FULL=%.1f CRIT_ALPHA_FROZEN=%.1f LVE=%.3f' % (SI, caf, caz, lve), flush=True)
    return {'SI': SI, 'lve': lve, 'sharp': SI > 0.4 and caf >= 5.0, 'extended': lve > 1.5}


@torch.no_grad()
def esv3_p3_content(H_off_pre):
    print('\n=== ESV3_P3: CONTENT-STRUCTURED BASINS ===', flush=True)
    pids = [tok(H.tmpl([{'role': 'user', 'content': p}]), return_tensors='pt').input_ids.to(dev) for p in PROMPTS_ESV2]
    M = len(pids); rng = torch.Generator(); rng.manual_seed(7)
    inits = [g.init() + 1.0 * torch.randn(K, D_S, generator=rng).to(dev) for _ in range(ESV3_N_CI)]
    cf = [[None] * ESV3_N_CI for _ in range(M)]; cz = [[None] * ESV3_N_CI for _ in range(M)]
    for p in range(M):
        for i in range(ESV3_N_CI):
            _, cf[p][i] = _settle_and_centroid(inits[i], lambda S: _tick_on_p(S, pids[p]), ESV3_T_SETTLE, ESV3_LATE_FRAC)
            _, cz[p][i] = _settle_and_centroid(inits[i], lambda S: _tick_on_p_off(S, pids[p]), ESV3_T_SETTLE, ESV3_LATE_FRAC)
    dc_f = _direction_consistency(cf); dc_z = _direction_consistency(cz); dc_r = _dc_permuted(cf, 10)
    print('  DC_FULL=%.4f  DC_FROZEN=%.4f  DC_RANDOM=%.4f  (delta_full_vs_random=%.4f)' % (dc_f, dc_z, dc_r, dc_f - dc_r), flush=True)
    neutral = g.init()
    bif_f = []; bif_z = []
    for p in range(M):
        _, bf = _settle_and_centroid(neutral, lambda S: _tick_on_p(S, pids[p]), ESV3_T_SETTLE, ESV3_LATE_FRAC)
        bif_f.append(bf)
        _, bz = _settle_and_centroid(neutral, lambda S: _tick_on_p_off(S, pids[p]), ESV3_T_SETTLE, ESV3_LATE_FRAC)
        bif_z.append(bz)

    def mean_pair(cs):
        v = [ang(cs[i], cs[j]) for i in range(len(cs)) for j in range(i + 1, len(cs))]
        return float(np.mean(v))

    bts_f = mean_pair(bif_f); bts_z = mean_pair(bif_z)
    wv = [ang(cf[p][a], cf[p][b]) for p in range(M) for a in range(ESV3_N_CI) for b in range(a + 1, ESV3_N_CI)]
    bv = [ang(cf[pa][i], cf[pb][i]) for i in range(ESV3_N_CI) for pa in range(M) for pb in range(pa + 1, M)]
    within = float(np.mean(wv)); between = float(np.mean(bv))
    print('  BT_SPREAD_FULL=%.4f  BT_SPREAD_FROZEN=%.4f  between=%.4f within=%.4f ratio=%.3f' % (bts_f, bts_z, between, within, between / (within + 1e-9)), flush=True)
    dc_li = (not (dc_f != dc_f)) and dc_f > 0.3 and (dc_f - dc_r) > 0.2 and dc_f > dc_z + 0.1
    bt_li = bts_f > 0.3 and bts_f > bts_z + 0.1
    print('  P3_VERDICT dc_loop_induced=%s bt_loop_induced=%s' % (dc_li, bt_li), flush=True)
    return {'dc_li': dc_li, 'bt_li': bt_li}


@torch.no_grad()
def esv3():
    print('ESV3 | N_INITS=%d T_SETTLE=%d T_POST=%d N_CI=%d EPS=%.2f' % (ESV3_N_INITS, ESV3_T_SETTLE, ESV3_T_POST, ESV3_N_CI, EPS), flush=True)
    _fb['on'] = False
    H_off_pre = model(fixed_ids, output_hidden_states=True).hidden_states[LOOP_READ_LAYER][0].float()
    p1 = esv3_p1_basin(H_off_pre)
    p2 = esv3_p2_viability(p1, H_off_pre)
    p3 = esv3_p3_content(H_off_pre)
    cnt = sum([p1['discrete'], p1['rtai_li'], p2['sharp'], p2['extended'], p3['dc_li'], p3['bt_li']])
    struct = 'NULL_ARTIFACT' if cnt == 0 else ('THIN' if cnt <= 1 else ('PARTIAL' if cnt <= 3 else 'RICH'))
    print('\nESV3_FINAL_VERDICT LOOP_INDUCED_COUNT=%d ENTITY_STRUCTURE=%s' % (cnt, struct), flush=True)
    print('=== ESV3_DONE ===', flush=True)
    print('=== ESV3_END ===', flush=True)


# ── ESV4: SURROGATE-GUIDED GATE TRAINING ─────────────────────────────────────

class SurrogateH(nn.Module):
    """fp32. Predicts the S-induced delta of layer-60 hidden (broadcast over tokens)."""

    def __init__(self, k, d_s, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k * d_s, 1024),
            nn.GELU(),
            nn.Linear(1024, d_model),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, S):
        """S [K, D_S] -> [d_model] predicted delta."""
        return self.net(S.reshape(-1).float())


@torch.no_grad()
def esv4_collect_surrogate_data(N):
    """Collect (S[K,D_S], delta[D_MODEL]) pairs.

    Half off-attractor (random perturbations of g.init()), half on-attractor
    (settled via _tick_on_only from random inits).  delta = mean over tokens of
    (H60_on - H60_off).
    """
    data = []
    n_half = N // 2

    # OFF-attractor half: random perturbations of g.init()
    rng_off = torch.Generator(); rng_off.manual_seed(11)
    alphas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    off_inits = []
    for a in alphas:
        for _ in range(max(1, n_half // len(alphas) + 1)):
            S0 = g.init() + a * torch.randn(K, D_S, generator=rng_off).to(dev)
            off_inits.append(S0)
    off_inits = off_inits[:n_half]

    for S in off_inits:
        _fb['S'] = S; _fb['on'] = True
        H_on = model(fixed_ids, output_hidden_states=True).hidden_states[LOOP_READ_LAYER][0].float()
        _fb['on'] = False
        H_off = model(fixed_ids, output_hidden_states=True).hidden_states[LOOP_READ_LAYER][0].float()
        delta = (H_on - H_off).mean(0)  # [D_MODEL]
        data.append((S.detach().clone(), delta.detach().clone()))

    # ON-attractor half: settled states
    rng_on = torch.Generator(); rng_on.manual_seed(12)
    n_on = N - n_half
    on_inits = []
    for a in [0.0, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(max(1, n_on // 5 + 1)):
            S0 = g.init() + a * torch.randn(K, D_S, generator=rng_on).to(dev)
            on_inits.append(S0)
    on_inits = on_inits[:n_on]

    for S0 in on_inits:
        Sf, _ = _settle_and_centroid(S0, _tick_on_only, 40, 0.25)
        S = Sf
        _fb['S'] = S; _fb['on'] = True
        H_on = model(fixed_ids, output_hidden_states=True).hidden_states[LOOP_READ_LAYER][0].float()
        _fb['on'] = False
        H_off = model(fixed_ids, output_hidden_states=True).hidden_states[LOOP_READ_LAYER][0].float()
        delta = (H_on - H_off).mean(0)
        data.append((S.detach().clone(), delta.detach().clone()))

    print('ESV4_SURR_DATA collected N=%d' % len(data), flush=True)
    return data


def esv4_train_surrogate(data, surr, n_steps):
    """Train fp32 surrogate to predict S-induced delta of layer-60 hidden.

    Returns (surr, gate_deg) where gate_deg is mean angular error (degrees)
    on heldout between predicted delta and true delta.
    """
    surr.train()
    opt = torch.optim.Adam(surr.parameters(), lr=3e-3)
    split = max(1, int(0.9 * len(data)))
    train_data = data[:split]
    heldout = data[split:]

    rng_t = torch.Generator(); rng_t.manual_seed(99)
    for step in range(1, n_steps + 1):
        idxs = torch.randperm(len(train_data), generator=rng_t)[:16].tolist()
        batch = [train_data[i] for i in idxs]
        pred = torch.stack([surr(S) for S, _ in batch])
        tgt  = torch.stack([d for _, d in batch]).to(dev)
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print('ESV4_SURR step=%d mse=%.6f' % (step, float(loss)), flush=True)

    # Compute gate: angular error between predicted delta and true delta on heldout
    surr.eval()
    with torch.no_grad():
        if heldout:
            angles = []
            for S, d_true in heldout:
                d_pred = surr(S)
                a = float(ang(d_pred.unsqueeze(0), d_true.unsqueeze(0)))
                angles.append(math.degrees(a))
            gate_deg = float(np.mean(angles))
        else:
            gate_deg = 0.0
    print('ESV4_SURR_GATE deg=%.2f' % gate_deg, flush=True)
    if gate_deg > 20:
        print('ESV4_SURR_GATE FAIL (>20deg)', flush=True)
    return surr, gate_deg


def esv4_surrogate_rollout(S0, surr, T_roll, T_late):
    """DIFFERENTIABLE rollout through surrogate + gate.

    Grad flows through g; surr params are frozen but surr.forward() is
    differentiable wrt S (not needed since surr input is detached S0, but
    the gate g.step receives H_hat which depends on surr(S) — however S fed
    to surr here is .detach()ed to prevent second-order through surr; only
    g params receive grad).

    Returns (S_rec[K,D_S], traj_vel list of scalar tensors).
    """
    S = S0.detach().clone().requires_grad_(False)
    # Re-attach via a leaf that grad can flow through for g params
    # Actually: grad must flow through g.step which takes S and H_hat.
    # surr(S) gives delta (fp32), H_hat = _ESV4_HOFF + delta.unsqueeze(0)
    # g.step(S, H_hat) — g params receive grad through this call.
    # We want S to evolve but NOT carry grad history (avoid exploding graph).
    traj_vel = []
    states = []
    S = S0.detach().clone()
    for t in range(T_roll):
        delta = surr(S.detach())                       # [D_MODEL], surrogate frozen
        H_hat = _ESV4_HOFF + delta.unsqueeze(0)        # [T_tok, D_MODEL]
        Snew = g.step(S.detach(), H_hat)               # grad flows through g params
        if t > 0:
            sim = F.cosine_similarity(
                Snew.reshape(1, -1), S.reshape(1, -1)
            ).clamp(-1.0, 1.0)
            traj_vel.append(1.0 - sim)
        states.append(Snew)
        S = Snew.detach()
    S_rec = torch.stack(states[-T_late:]).mean(0)      # [K, D_S]
    return S_rec, traj_vel


@torch.no_grad()
def esv4_frozen_rollout(S0, T_roll, T_late):
    """No-grad frozen rollout: H = _ESV4_HOFF constant (S-independent)."""
    S = S0.detach().clone()
    states = []
    for _ in range(T_roll):
        Snew = g.step(S, _ESV4_HOFF)
        states.append(Snew)
        S = Snew
    S_rec_frz = torch.stack(states[-T_late:]).mean(0)
    return S_rec_frz


def esv4_loss(S_recs, S_recs_frz, vels, cents, ci, vel_floor, margin, lams):
    """Compute ESV4 training loss.

    Args:
        S_recs:     list[B] of [K,D_S] tensors (differentiable)
        S_recs_frz: list[B] of [K,D_S] tensors (detached)
        vels:       list[B] of list of scalar tensors (velocity sequence)
        cents:      list of [K,D_S] centroid tensors (detached)
        ci:         list[B] ints, assigned centroid index per batch item
        vel_floor, margin: floats
        lams:       dict with keys 'vel', 'margin', 'multi'
    """
    beta = 5.0
    B = len(S_recs)
    T_late = lams.get('T_late', ESV4_T_LATE)

    rtai_list = []
    vel_list = []
    margin_list = []

    for b in range(B):
        Sr = S_recs[b].reshape(1, -1).float()       # [1, K*D_S]
        c_own = cents[ci[b]].reshape(1, -1).float()  # [1, K*D_S]

        d_own = 1.0 - F.cosine_similarity(Sr, c_own)  # scalar tensor

        # softmin over other centroids
        others = [cents[j].reshape(1, -1).float()
                  for j in range(len(cents)) if j != ci[b]]
        if others:
            d_others = torch.stack(
                [1.0 - F.cosine_similarity(Sr, co) for co in others]
            )  # [N_other]
            d_other = -1.0 / beta * torch.logsumexp(-beta * d_others, dim=0)
        else:
            d_other = d_own.detach() * 0.0 + 1.0

        rtai = d_own / (d_own + d_other + 1e-6)
        rtai_list.append(rtai)

        # velocity: mean of last T_late velocity values
        vel_seq = vels[b]
        if vel_seq:
            vel_late = torch.stack(vel_seq[-T_late:]).mean()
        else:
            vel_late = torch.zeros(1, device=dev)
        vel_list.append(F.relu(vel_floor - vel_late))

        # margin vs frozen baseline
        Sfz = S_recs_frz[b].reshape(1, -1).float()
        d_loop = 1.0 - F.cosine_similarity(Sr, c_own)
        d_frz  = 1.0 - F.cosine_similarity(Sfz, c_own)
        margin_list.append(F.relu(d_loop - d_frz + margin))

    L_basin  = torch.stack(rtai_list).mean()
    L_vel    = torch.stack(vel_list).mean()
    L_margin = torch.stack(margin_list).mean()

    # batch diversity: negative mean pairwise (1-cos) over flattened states
    flat = torch.stack([s.reshape(-1).float() for s in S_recs])  # [B, K*D_S]
    n = flat.shape[0]
    if n > 1:
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append(1.0 - F.cosine_similarity(
                    flat[i].unsqueeze(0), flat[j].unsqueeze(0)
                ))
        L_multi = -torch.stack(pairs).mean()
    else:
        L_multi = torch.zeros(1, device=dev).squeeze()

    L = L_basin + lams['vel'] * L_vel + lams['margin'] * L_margin + lams['multi'] * L_multi

    parts = {
        'basin':  float(L_basin.detach()),
        'vel':    float(L_vel.detach()),
        'margin': float(L_margin.detach()),
        'multi':  float(L_multi.detach()),
    }
    return L, parts


@torch.no_grad()
def esv4_refresh_centroids(n_inits, T_settle):
    """Settle n_inits states and return deduplicated centroid list."""
    rng = torch.Generator(); rng.manual_seed(42)
    inits = []
    for a in [0.0, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(max(1, n_inits // 5)):
            inits.append(g.init() + a * torch.randn(K, D_S, generator=rng).to(dev))
    cents = []
    for S0 in inits[:n_inits]:
        _, c = _settle_and_centroid(S0, _tick_on_only, T_settle, 0.25)
        cents.append(c)
    # Greedy dedup at theta=1.0 rad
    keep = []
    for c in cents:
        if all(ang(c, k) > 1.0 for k in keep):
            keep.append(c)
    return [c.detach() for c in keep] if keep else [cents[0].detach()]


@torch.no_grad()
def esv4_real_eval(H_off_pre):
    """Run ESV3 P1+P2 probes with reduced N_INITS for in-training checks."""
    print('--- ESV4 REAL-LOOP EVAL (WALL-P1 check) ---', flush=True)
    p1 = esv3_p1_basin(H_off_pre)
    p2 = esv3_p2_viability(p1, H_off_pre)
    return p1, p2


def esv4_train():
    """Train gate g (fp32) through differentiable surrogate of S→H delta."""
    global _ESV4_HOFF, ESV3_N_INITS, ESV3_T_SETTLE, ESV3_T_POST

    print('\n=== ESV4_TRAIN  SEED=%d  STEPS=%d  LR=%.2e  T_ROLL=%d  T_LATE=%d ===' % (
        SEED, ESV4_STEPS, ESV4_LR, ESV4_T_ROLL, ESV4_T_LATE), flush=True)

    # Reduce ESV3 probe cost for in-training evals
    ESV3_N_INITS  = ESV4_EVAL_NINITS
    ESV3_T_SETTLE = min(ESV3_T_SETTLE, 30)
    ESV3_T_POST   = min(ESV3_T_POST,   30)

    # Precompute baseline hidden (field OFF)
    _fb['on'] = False
    _ESV4_HOFF = model(
        fixed_ids, output_hidden_states=True
    ).hidden_states[LOOP_READ_LAYER][0].float().detach()
    print('ESV4 _ESV4_HOFF shape=%s' % str(tuple(_ESV4_HOFF.shape)), flush=True)

    # ── PHASE 0: Surrogate fit ────────────────────────────────────────────────
    print('ESV4 PHASE 0: collecting %d surrogate samples ...' % ESV4_SURR_DATA, flush=True)
    data = esv4_collect_surrogate_data(ESV4_SURR_DATA)
    surr = SurrogateH(K, D_S, D_MODEL).to(dev).float()
    surr, gate_deg = esv4_train_surrogate(data, surr, ESV4_SURR_STEPS)
    surr.eval()
    for p in surr.parameters():
        p.requires_grad_(False)

    if gate_deg > 20:
        print('ESV4_ABORT surrogate gate failed', flush=True)
        print('=== ESV4_DONE ===', flush=True)
        print('=== ESV4_END ===', flush=True)
        return

    if ESV4_SURR_ONLY:
        print('ESV4_SURR_ONLY: running baseline eval then exiting.', flush=True)
        esv4_real_eval(_ESV4_HOFF)
        print('=== ESV4_DONE ===', flush=True)
        print('=== ESV4_END ===', flush=True)
        return

    # ── PHASE 1: Gate training ────────────────────────────────────────────────
    g.float()
    g.train()
    for p in g.parameters():
        p.requires_grad_(True)

    cents = esv4_refresh_centroids(20, 30)
    print('ESV4 N_attractors=%d' % len(cents), flush=True)

    opt  = AdamW(g.parameters(), lr=ESV4_LR, weight_decay=0.01)
    rngp = torch.Generator(); rngp.manual_seed(77)

    # Baseline eval before training
    print('ESV4 BASELINE EVAL:', flush=True)
    esv4_real_eval(_ESV4_HOFF)

    lams = {'vel': ESV4_L_VEL, 'margin': ESV4_L_MARGIN, 'multi': ESV4_L_MULTI,
            'T_late': ESV4_T_LATE}

    for step in range(1, ESV4_STEPS + 1):
        if step % ESV4_K_REFRESH == 0:
            cents = esv4_refresh_centroids(20, 30)

        B = 8
        ci_list = []; S_recs = []; S_frz = []; vels_all = []
        for b in range(B):
            i  = b % len(cents)
            a  = 0.5 if b % 2 == 0 else 2.0
            Sp = cents[i].detach() + a * torch.randn(K, D_S, generator=rngp).to(dev)
            Sr, vl = esv4_surrogate_rollout(Sp, surr, ESV4_T_ROLL, ESV4_T_LATE)
            Sfz     = esv4_frozen_rollout(Sp, ESV4_T_ROLL, ESV4_T_LATE)
            ci_list.append(i); S_recs.append(Sr); S_frz.append(Sfz); vels_all.append(vl)

        L, parts = esv4_loss(
            S_recs, S_frz, vels_all, cents, ci_list,
            ESV4_VEL_FLOOR, ESV4_MARGIN, lams,
        )
        opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
        opt.step()

        if step % 25 == 0:
            print(
                'ESV4 step=%d L=%.4f basin=%.4f vel=%.4f margin=%.4f multi=%.4f N_attr=%d' % (
                    step, float(L), parts['basin'], parts['vel'],
                    parts['margin'], parts['multi'], len(cents)),
                flush=True,
            )

        if step % ESV4_K_EVAL == 0:
            torch.save(
                g.state_dict(),
                os.path.join(CKDIR, 'esv4_gate_s%d_step%d.pt' % (SEED, step)),
            )
            esv4_real_eval(_ESV4_HOFF)

    torch.save(g.state_dict(), os.path.join(CKDIR, 'esv4_gate_s%d.pt' % SEED))
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)

    print('=== ESV4 FINAL EVAL ===', flush=True)
    esv4_real_eval(_ESV4_HOFF)
    print('=== ESV4_DONE ===', flush=True)
    print('=== ESV4_END ===', flush=True)


@torch.no_grad()
def esv4d_collect_pool(cents):
    """Build real (S, H(S), centroid-index) pool using field-ON hidden states.

    For each centroid C_i and each alpha in ESV4D_ALPHAS, sample n_per
    perturbed states S = C_i + alpha*randn, run a real field-ON LLM forward
    to obtain H(S), and store (S.detach(), H.detach(), i).  Stops when the
    pool reaches ESV4D_POOL items.

    Args:
        cents: list of [K, D_S] centroid tensors (detached, from esv4_refresh_centroids)

    Returns:
        list of (S [K,D_S], H [T_tok, D_MODEL], ci int) — all tensors detached.
    """
    alphas = [float(a) for a in ESV4D_ALPHAS.split(',')]
    rng = torch.Generator()
    rng.manual_seed(33)
    n_per = max(1, ESV4D_POOL // max(1, len(cents) * len(alphas)))
    pool = []
    for i, C in enumerate(cents):
        for alpha in alphas:
            for _ in range(n_per):
                if len(pool) >= ESV4D_POOL:
                    break
                noise = torch.randn(K, D_S, generator=rng).to(dev)
                S = (C + alpha * noise).detach()
                _fb['S'] = S
                _fb['on'] = True
                H = model(fixed_ids, output_hidden_states=True).hidden_states[LOOP_READ_LAYER][0].float()
                _fb['on'] = False
                pool.append((S.detach(), H.detach(), i))
            if len(pool) >= ESV4D_POOL:
                break
        if len(pool) >= ESV4D_POOL:
            break
    return pool


def esv4d_train():
    """Train g on SINGLE real-H steps — no surrogate, no frozen-H drift (WALL P1 free).

    H(S) = LLM(field(prompt, S)) is read at LOOP_READ_LAYER.  Because the
    field reads S not g, precomputed pool entries stay valid as g trains.
    Reuses: esv4_refresh_centroids, esv4d_collect_pool, esv4_loss,
    esv4_real_eval, ESV4_* env vars.
    """
    global ESV3_N_INITS, ESV3_T_SETTLE, ESV3_T_POST

    lams = {'vel': ESV4_L_VEL, 'margin': ESV4_L_MARGIN, 'multi': ESV4_L_MULTI}
    print('\n=== ESV4D_TRAIN  SEED=%d  STEPS=%d  POOL=%d  LR=%.2e  '
          'vel_floor=%.4f  margin=%.4f  l_vel=%.2f  l_margin=%.2f  l_multi=%.2f ===' % (
              SEED, ESV4D_STEPS, ESV4D_POOL, ESV4_LR,
              ESV4_VEL_FLOOR, ESV4_MARGIN,
              ESV4_L_VEL, ESV4_L_MARGIN, ESV4_L_MULTI), flush=True)

    # Cheap in-training evals
    ESV3_N_INITS  = ESV4_EVAL_NINITS
    ESV3_T_SETTLE = 30
    ESV3_T_POST   = 30

    # Field-OFF baseline hidden (S-independent; used as frozen arm in L_margin)
    _fb['on'] = False
    H_off_pre = model(fixed_ids, output_hidden_states=True).hidden_states[LOOP_READ_LAYER][0].float().detach()

    # Prepare g for training
    g.float()
    g.train()
    for p in g.parameters():
        p.requires_grad_(True)

    cents = esv4_refresh_centroids(20, 30)
    print('ESV4D N_attractors=%d' % len(cents), flush=True)
    pool = esv4d_collect_pool(cents)

    opt = AdamW(g.parameters(), lr=ESV4_LR, weight_decay=0.01)

    print('--- ESV4D BASELINE EVAL ---', flush=True)
    esv4_real_eval(H_off_pre)

    rng_b = torch.Generator()
    rng_b.manual_seed(55)

    for step in range(1, ESV4D_STEPS + 1):
        if step % ESV4_K_REFRESH == 0:
            cents = esv4_refresh_centroids(20, 30)
            pool = esv4d_collect_pool(cents)

        # Deterministic batch indices — no RNG state issues, reproducible
        idxs = [(step * 7 + b * 13) % len(pool) for b in range(8)]

        Sp_loop = []
        Sp_frz  = []
        vels    = []
        ci      = []

        for j in idxs:
            S, Hreal, c = pool[j]
            # Differentiable wrt g; S and Hreal are detached pool entries
            Sprime = g.step(S, Hreal)
            # Frozen-field one-step: field-OFF H is S-independent baseline
            Sfrz   = g.step(S, H_off_pre)
            vel_val = 1.0 - F.cosine_similarity(
                Sprime.reshape(1, -1), S.reshape(1, -1)
            ).clamp(-1.0, 1.0)
            Sp_loop.append(Sprime)
            Sp_frz.append(Sfrz)
            vels.append([vel_val])   # 1-element list: mean([x]) == x; compatible with esv4_loss
            ci.append(c)

        L, parts = esv4_loss(Sp_loop, Sp_frz, vels, cents, ci,
                             ESV4_VEL_FLOOR, ESV4_MARGIN, lams)
        opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
        opt.step()

        if step % 25 == 0:
            print('ESV4D step=%d L=%.4f basin=%.4f vel=%.4f margin=%.4f multi=%.4f Nattr=%d' % (
                step, float(L),
                parts['basin'], parts['vel'], parts['margin'], parts['multi'],
                len(cents)), flush=True)

        if step % ESV4_K_EVAL == 0:
            torch.save(g.state_dict(),
                       os.path.join(CKDIR, 'esv4d_gate_s%d_step%d.pt' % (SEED, step)))
            esv4_real_eval(H_off_pre)

    torch.save(g.state_dict(), os.path.join(CKDIR, 'esv4d_gate_s%d.pt' % SEED))
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)

    print('=== ESV4D FINAL EVAL ===', flush=True)
    esv4_real_eval(H_off_pre)
    print('=== ESV4D_DONE ===', flush=True)
    print('=== ESV4D_END ===', flush=True)


def esv4_eval():
    """Load saved gate checkpoint and run ESV3 P1+P2 probes."""
    global ESV3_N_INITS, _ESV4_HOFF
    ESV3_N_INITS = ESV4_EVAL_NINITS

    ckpt_path = ESV4_CKPT or os.path.join(CKDIR, 'esv4_gate_s%d.pt' % SEED)
    g.load_state_dict(torch.load(ckpt_path, map_location=dev))
    g.eval()
    print('ESV4_EVAL loaded %s' % ckpt_path, flush=True)

    _fb['on'] = False
    hop = model(
        fixed_ids, output_hidden_states=True
    ).hidden_states[LOOP_READ_LAYER][0].float()

    p1 = esv3_p1_basin(hop)
    esv3_p2_viability(p1, hop)
    print('=== ESV4_DONE ===', flush=True)
    print('=== ESV4_END ===', flush=True)


def _softmin_cycdist(S, cyc_pts, tau):
    """Phase-invariant differentiable distance from S to a cycle point-set.

    Args:
        S: [K, D_S] state tensor (may carry grad).
        cyc_pts: list of [K, D_S] detached tensors representing the cycle.
        tau: softmin temperature (scalar float).

    Returns:
        Scalar tensor: soft-min over points of (1 - cos(S, p)).
    """
    dists = torch.stack([
        1 - F.cosine_similarity(
            S.reshape(1, -1), p.reshape(1, -1)
        ).clamp(-1.0, 1.0).squeeze()
        for p in cyc_pts
    ])  # [n_pts]
    return -tau * torch.logsumexp(-dists / tau, dim=0)   # soft-min, scalar


def _esv4e_rollout(S0, K_roll):
    """Multi-step rollout under g with real LLM hidden states.

    Grad flows through g.step across steps.  The LLM forward is
    inside torch.no_grad and its output is detached so no LLM backprop.

    Args:
        S0: [K, D_S] starting state (may carry grad or be detached).
        K_roll: number of rollout steps.

    Returns:
        List of K_roll [K, D_S] state tensors, each carrying grad wrt g params.
    """
    S = S0
    traj = []
    for _ in range(K_roll):
        with torch.no_grad():
            _fb['S'] = S.detach()
            _fb['on'] = True
            Hr = model(
                fixed_ids, output_hidden_states=True
            ).hidden_states[LOOP_READ_LAYER][0].float().detach()
            _fb['on'] = False
        S = g.step(S, Hr)   # grad to g params and to S (first arg); Hr detached
        traj.append(S)
    return traj


@torch.no_grad()
def _esv4e_baseline_cycles():
    """Settle multiple inits and collect per-cycle point sequences.

    Runs ESV4E_NCYC*3 perturbation amplitudes × seeds, keeps last
    ESV4E_CYCLEN states of each settle trajectory as the cycle, deduplicates
    cycles whose centroids are closer than 1.0 rad, and returns up to
    ESV4E_NCYC cycles.

    Returns:
        List of up to ESV4E_NCYC cycles; each cycle is a list of
        ESV4E_CYCLEN detached [K, D_S] tensors.
    """
    rng = torch.Generator()
    rng.manual_seed(42)
    cycles = []
    cents = []
    for a in [0.0, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(ESV4E_NCYC):
            S = g.init() + a * torch.randn(K, D_S, generator=rng).to(dev)
            pts = []
            for t in range(ESV3_T_SETTLE):
                _fb['S'] = S
                _fb['on'] = True
                Hr = model(
                    fixed_ids, output_hidden_states=True
                ).hidden_states[LOOP_READ_LAYER][0].float()
                _fb['on'] = False
                S = g.step(S, Hr)
                if t >= ESV3_T_SETTLE - ESV4E_CYCLEN:
                    pts.append(S.detach().clone())
            c = torch.stack(pts).mean(0)
            if all(ang(c, cc) > 1.0 for cc in cents):
                cents.append(c)
                cycles.append(pts)
            if len(cycles) >= ESV4E_NCYC:
                break
        if len(cycles) >= ESV4E_NCYC:
            break
    return cycles


def esv4e_train():
    """ESV4-E: multi-step viability training toward fixed baseline attractor CYCLES.

    Deepens basins toward pre-computed baseline cycles (not centroid-points)
    so it cannot collapse multistability.  Sibling to ESV4D.

    Reuses: g, model, fixed_ids, dev, K, D_S, SEED, CKDIR, LOOP_READ_LAYER,
    _fb, AdamW, ESV4_LR, ESV4_VEL_FLOOR, ESV4_MARGIN, ESV4_L_VEL,
    ESV4_L_MARGIN, ESV4_L_MULTI, ESV4_K_EVAL, ESV4_EVAL_NINITS,
    ESV3_N_INITS / T_SETTLE / T_POST globals, ang, esv4_real_eval.
    """
    global ESV3_N_INITS, ESV3_T_SETTLE, ESV3_T_POST

    print(
        '\n=== ESV4E_TRAIN  SEED=%d  STEPS=%d  K=%d  B=%d  NCYC=%d  '
        'ALPHA=%.2f  vel_floor=%.4f  margin=%.4f  '
        'l_vel=%.2f  l_margin=%.2f  l_multi=%.2f ===' % (
            SEED, ESV4E_STEPS, ESV4E_K, ESV4E_B, ESV4E_NCYC,
            ESV4E_ALPHA, ESV4_VEL_FLOOR, ESV4_MARGIN,
            ESV4_L_VEL, ESV4_L_MARGIN, ESV4_L_MULTI),
        flush=True)

    # Cheap in-training evals
    ESV3_N_INITS  = ESV4_EVAL_NINITS
    ESV3_T_SETTLE = 30
    ESV3_T_POST   = 30

    # Field-OFF baseline hidden — S-independent reference
    _fb['on'] = False
    H_off = model(
        fixed_ids, output_hidden_states=True
    ).hidden_states[LOOP_READ_LAYER][0].float().detach()

    # Capture baseline cycles BEFORE any training modifies g
    cycles = _esv4e_baseline_cycles()
    print('ESV4E baseline_cycles=%d' % len(cycles), flush=True)
    if len(cycles) < 2:
        print('ESV4E_ABORT too few cycles', flush=True)
        print('=== ESV4E_DONE ===', flush=True)
        print('=== ESV4E_END ===', flush=True)
        return

    g.float()
    g.train()
    for p in g.parameters():
        p.requires_grad_(True)

    opt = AdamW(g.parameters(), lr=ESV4_LR, weight_decay=0.01)

    print('--- ESV4E BASELINE EVAL ---', flush=True)
    esv4_real_eval(H_off)

    rng = torch.Generator()
    rng.manual_seed(55)

    for step in range(1, ESV4E_STEPS + 1):
        # Deterministic rotation over available cycles
        cyc_idx = [(step * 3 + b) % len(cycles) for b in range(min(ESV4E_B, len(cycles)))]

        endpoints   = []   # (cycle_index, SK)
        L_rec       = []
        L_vel_terms = []
        L_marg      = []

        for ci in cyc_idx:
            base_pt = cycles[ci][step % len(cycles[ci])]        # a point on the cycle
            Sp = base_pt + ESV4E_ALPHA * torch.randn(K, D_S, generator=rng).to(dev)

            traj = _esv4e_rollout(Sp, ESV4E_K)                  # grad flows through g
            SK = traj[-1]
            endpoints.append((ci, SK))

            # Recovery: push end-state toward its own baseline cycle
            L_rec.append(_softmin_cycdist(SK, cycles[ci], ESV4E_TAU))

            # Velocity: reward for state displacement across rollout steps
            vels = [
                1 - F.cosine_similarity(
                    traj[t].reshape(1, -1),
                    (traj[t - 1] if t > 0 else Sp).reshape(1, -1)
                ).clamp(-1.0, 1.0).squeeze()
                for t in range(ESV4E_K)
            ]
            L_vel_terms.append(F.relu(ESV4_VEL_FLOOR - torch.stack(vels).mean()))

            # Margin: loop-trajectory must beat frozen-H trajectory toward cycle
            with torch.no_grad():
                Sf = Sp
                for _ in range(ESV4E_K):
                    Sf = g.step(Sf, H_off)
            d_loop = _softmin_cycdist(SK, cycles[ci], ESV4E_TAU)
            d_frz  = _softmin_cycdist(Sf.detach(), cycles[ci], ESV4E_TAU)
            L_marg.append(F.relu(d_loop - d_frz + ESV4_MARGIN))

        L_recover = torch.stack(L_rec).mean()
        L_vel     = torch.stack(L_vel_terms).mean()
        L_margin  = torch.stack(L_marg).mean()

        # Separation: different-cycle endpoints must stay apart (anti-merge)
        seps = []
        for a in range(len(endpoints)):
            for b in range(a + 1, len(endpoints)):
                if endpoints[a][0] != endpoints[b][0]:
                    seps.append(
                        F.cosine_similarity(
                            endpoints[a][1].reshape(1, -1),
                            endpoints[b][1].reshape(1, -1)
                        ).clamp(-1.0, 1.0).squeeze()
                    )
        L_separate = torch.stack(seps).mean() if seps else torch.tensor(0.0, device=dev)

        L = (L_recover
             + ESV4_L_VEL    * L_vel
             + ESV4_L_MARGIN * L_margin
             + ESV4_L_MULTI  * L_separate)

        opt.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
        opt.step()

        if step % 20 == 0:
            print(
                'ESV4E step=%d L=%.4f recover=%.4f vel=%.4f margin=%.4f sep=%.4f' % (
                    step, float(L), float(L_recover),
                    float(L_vel), float(L_margin), float(L_separate)),
                flush=True)

        if step % ESV4_K_EVAL == 0:
            torch.save(
                g.state_dict(),
                os.path.join(CKDIR, 'esv4e_gate_s%d_step%d.pt' % (SEED, step)))
            esv4_real_eval(H_off)

    torch.save(g.state_dict(), os.path.join(CKDIR, 'esv4e_gate_s%d.pt' % SEED))
    g.eval()
    for p in g.parameters():
        p.requires_grad_(False)

    print('=== ESV4E FINAL EVAL ===', flush=True)
    esv4_real_eval(H_off)
    print('=== ESV4E_DONE ===', flush=True)
    print('=== ESV4E_END ===', flush=True)


if ESV4E:          esv4e_train()
elif ESV4D:        esv4d_train()
elif ESV4:         esv4_train()
elif ESV4_EVAL:    esv4_eval()
elif ESV2:         esv2_train()
elif ESV2_EVAL:    esv2_eval()
elif CONTENT:      content_characterize()
elif EPS_SWEEP:    eps_sweep()
elif ESV3:         esv3()
else:              characterize()
