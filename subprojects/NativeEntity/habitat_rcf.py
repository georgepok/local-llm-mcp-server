# HABCF_RECURRENT_DEPENDENCE_V1 — one early hidden CLASS drives 4 later ambiguous decisions via a
# Latin-square of permuted codeword->action keylines, so class-invariant slot collapse is wrong at >=3/4.
# Repeated diverse-surface dependence = world dynamics repeatedly SELECT for preserving the class.
# Substrate FROZEN (identical to habitat_cf.py); only the WORLD + eval change. Score CUMULATIVE viability.
import os, random
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_float32_matmul_precision('high')
import habitat_evo as H
import slots as SL

dev = H.dev; model = H.model; tok = H.tok
SEED = int(os.environ.get('SEED', '0')); random.seed(SEED); torch.manual_seed(SEED)
D_MODEL = model.config.hidden_size if getattr(model.config, 'hidden_size', None) else getattr(model.config.text_config, 'hidden_size', 5120)
N_LAYERS = model.config.num_hidden_layers if getattr(model.config, 'num_hidden_layers', None) else getattr(model.config.text_config, 'num_hidden_layers', 64)
READ_LAYER = N_LAYERS // 2
K = int(os.environ.get('K', '12')); SLOW_K = int(os.environ.get('SLOW_K', '6')); D_S = int(os.environ.get('D_S', '768'))
FIELD_LAYERS = [int(x) for x in os.environ.get('FIELD_LAYERS', '40,48,56').split(',')]
EPS = float(os.environ.get('EPS', '0.12'))
NTOK = int(os.environ.get('NTOK', '24'))
WINDOW = int(os.environ.get('WINDOW', '8'))
N_TRAIN  = int(os.environ.get('N_TRAIN',  '60'))   # 5 per class × 3 train-templates × 4 classes
N_INDIST = int(os.environ.get('N_INDIST', '20'))
N_TEST   = int(os.environ.get('N_TEST',   '24'))   # 6 per class, held-out template
FIELD_EPOCHS = int(os.environ.get('FIELD_EPOCHS', '30'))
MODE = os.environ.get('CFMODE', 'validate')        # validate | substrate
DEC_N = int(os.environ.get('DEC_N', '4'))                       # Ablation A: number of recurrent decisions
SURFACE_MODE = os.environ.get('SURFACE_MODE', 'diverse')       # Ablation B: 'diverse' | 'same'
N_CLASSES = int(os.environ.get('N_CLASSES', '4'))               # V2 population: 4 / 6 / 8 latent classes (>actions => share actions)
N_ACTIONS = int(os.environ.get('N_ACTIONS', '4'))              # 4 (default) | 6 (6/6 DIAGNOSTIC: restore per-decision class-diagnosticity)
N_TRAIN_TEMPLATES = int(os.environ.get('N_TRAIN_TEMPLATES', '3'))
N_HELDOUT_TEMPLATES = int(os.environ.get('N_HELDOUT_TEMPLATES', '1'))
N_WORLDS_PER_SLOT = int(os.environ.get('N_WORLDS_PER_SLOT', '5'))   # worlds per (class × train-template)
print('HAB-RCF | mode=%s d_model=%d read=%d | K=%d slow_k=%d d_s=%d FIELD=%s EPS=%.2f WINDOW=%d DEC_N=%d SURFACE=%s NCLS=%d NACT=%d NTPL=%d/%d' % (
    MODE, D_MODEL, READ_LAYER, K, SLOW_K, D_S, FIELD_LAYERS, EPS, WINDOW, DEC_N, SURFACE_MODE, N_CLASSES, N_ACTIONS, N_TRAIN_TEMPLATES, N_HELDOUT_TEMPLATES), flush=True)

CLASSES_ALL   = ['ALPHA', 'BETA', 'GAMMA', 'DELTA', 'ECHO', 'HOTEL', 'INDIA', 'JULIET']
CODEWORDS_ALL = ['FOXTROT', 'KILO', 'NOVEMBER', 'SIERRA', 'TANGO', 'VICTOR', 'WHISKEY', 'XRAY']
ACTIONS_4 = ['RELEASE', 'DEFER', 'REJECT', 'REPAIR']
ACTIONS_6 = ['RELEASE', 'DEFER', 'REJECT', 'REPAIR', 'ESCALATE', 'ARCHIVE']   # 6/6 DIAGNOSTIC actions (substring-safe)
ACTIONS = ACTIONS_4 if N_ACTIONS == 4 else ACTIONS_6
# 4-action matrices keyed by N_CLASSES (N=4 byte-identical to canonical; N=6/8 classes SHARE actions => NOT class-diagnostic per decision).
_AMAT4 = {
    4: [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]],
    6: [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2], [0, 2, 1, 3], [1, 3, 0, 2]],
    8: [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2], [0, 2, 1, 3], [1, 3, 0, 2], [2, 0, 3, 1], [3, 1, 2, 0]],
}
# 6-action 6x6 CYCLIC Latin square: each decision (column) maps the 6 classes to 6 DISTINCT actions => each decision IS class-diagnostic.
_AMAT6 = [[(ci + k) % 6 for k in range(6)] for ci in range(6)]
AMAT      = _AMAT6 if N_ACTIONS == 6 else _AMAT4[N_CLASSES]
CLASSES   = CLASSES_ALL[:N_CLASSES]
CODEWORDS = CODEWORDS_ALL[:N_CLASSES]
_KEYNAMES = ['Dispatch key', 'Clearance key', 'Authorization key', 'Incident key', 'Escalation key', 'Archival key']
_QSTEM = [
    'The freight depot requests a routing decision for the pending consignment.',
    'The triage coordinator requests a disposition ruling for the admitted patient.',
    'The compliance officer requests a ruling on the flagged account transaction.',
    'The systems team requests a resolution directive for the flagged infrastructure incident.',
    'The escalation desk requests a priority determination for the open ticket.',
    'The records office requests a retention decision for the archived file.',
]
_ASK = 'Reply with one word: ' + ', '.join(ACTIONS[:-1]) + ', or ' + ACTIONS[-1] + '.'   # 4-action form == canonical byte-for-byte
def _mk_spec(k):
    key = {CODEWORDS[ci]: ACTIONS[AMAT[ci][k]] for ci in range(N_CLASSES)}
    keyline = '%s: %s.' % (_KEYNAMES[k], ', '.join('%s->%s' % (cw, key[cw]) for cw in CODEWORDS))
    return {'keyline': keyline, 'q': '%s %%s %s' % (_QSTEM[k], _ASK), 'key': key}
DECISION_SPECS = [_mk_spec(k) for k in range(N_ACTIONS)]

def spec_for(k): return DECISION_SPECS[(k % N_ACTIONS) if SURFACE_MODE == 'diverse' else 0]   # diverse=rotate keylines, same=keyline0 every decision

COMMIT_TEMPLATES = ["Operational code filed", "Logged operational code", "Registered code on record",   # 0-2 train (orig)
                    "Filed operational code", "Batch code acknowledged",                                 # 3 (orig N=4 held-out; N=6 train), 4 new train
                    "Identifier sequence committed", "Route marker confirmed"]                            # 5-6 held-out (distinct lexicon)
# N=4 default (N_TRAIN_TEMPLATES=3, N_HELDOUT_TEMPLATES=1): train idx0-2 + held-out idx3 == BYTE-IDENTICAL to canonical run.
# N=6/8 (N_TRAIN_TEMPLATES=5, N_HELDOUT_TEMPLATES=2): train idx0-4 + held-out idx5,6.

OFF_POOL = [
    "Aside: how does compound interest work?",
    "Unrelated: explain the rules of chess briefly.",
    "Different topic: how do rainbows form?",
    "Quick aside: what causes ocean tides?",
    "Unrelated: how does photosynthesis work?",
    "Different topic: how are sailing knots tied?",
    "Aside: why is the sky blue?",
    "Unrelated: how do bees make honey?",
    "Different topic: what is the water cycle?",
    "Quick aside: how do magnets work?",
    "Unrelated: what causes thunder?",
    "Different topic: how do plants grow from seeds?",
    "Aside: why do leaves change color in autumn?",
]
# reactive off-turn notes — NO codeword / class / action content
RECOVERY_POOL = [
    "Status update: a discrepancy in the previous ruling has been flagged for review.",
    "Administrative note: the preceding action requires clarification — correction initiated.",
    "Operational note: an inconsistency was logged; recovery procedures are under way.",
    "Notice: the prior decision outcome has triggered a standard review protocol.",
]
NEUTRAL_NOTES = [
    "Status update: prior rulings are consistent; proceeding normally.",
    "Administrative note: no discrepancies on record; continue.",
    "Operational note: the log is in good order.",
    "Notice: routine checkpoint passed; carry on.",
]


def _digits(t): return int(''.join(c for c in t if c.isdigit()))
def newws(): return {'n_wrong': 0, 'correct': 0, 'ambiguous': 0,
                     'd0_wrong': False, 'd1_wrong': False, 'd2_wrong': False, 'd3_wrong': False}


def make_world(rng, tpl_idx, ci=None):
    """Trajectory: commit(t0, ONLY codeword occurrence) + 4 fillers, then DEC_N decisions each preceded by
    fillers, separated by an off_react + 3 fillers. spec_for(i) gives diverse (rotating keyline) or same (keyline0)."""
    if ci is None: ci = rng.randrange(N_CLASSES)
    CW = CODEWORDS[ci]
    commit = '%s: %s %s %s %s %s %s. Acknowledge.' % (COMMIT_TEMPLATES[tpl_idx], CW, CW, CW, CW, CW, CW)
    oc = [0]
    def noff():
        t = OFF_POOL[oc[0] % len(OFF_POOL)]; oc[0] += 1; return t
    turns = [('commit', commit)]
    for _ in range(4): turns.append(('off', noff()))                       # fillers before first decision
    for i in range(DEC_N):
        sp = spec_for(i)
        turns.append(('decision', sp['q'] % sp['keyline']))
        if i < DEC_N - 1:
            turns.append(('off_react', '__OFF_REACT_%d__' % i))
            for _ in range(3): turns.append(('off', noff()))
    return {'tpl_idx': tpl_idx, 'ci': ci, 'cls': CLASSES[ci], 'turns': turns}


def judge_rcf(k, resp, w, ws):
    """Recovery-cost cumulative scoring: a correct ruling is worth less once errors have accrued."""
    r = resp.upper(); found = [a for a in ACTIONS if a in r]
    correct = spec_for(k)['key'][CODEWORDS[w['ci']]]
    if len(found) == 1 and found[0] == correct:
        ws['correct'] += 1; return max(0.25, 1.0 - 0.15 * ws['n_wrong'])
    if len(found) == 1:
        ws['n_wrong'] += 1; ws['d%d_wrong' % k] = True; return -1.0
    ws['ambiguous'] += 1; return -0.3


def _fill_off_react(text, ws):
    k = _digits(text)
    return RECOVERY_POOL[k % 4] if ws.get('d%d_wrong' % k, False) else NEUTRAL_NOTES[k % 4]


# reference responders (kind, w, dk) — for validate
def oracle_good(kind, w, dk):
    if kind == 'decision': return spec_for(dk)['key'][CODEWORDS[w['ci']]]
    return 'Understood.'
def base_blind(kind, w, dk): return None
def always_first(kind, w, dk):
    if kind == 'decision': return 'RELEASE'
    return 'Understood.'


# ── Substrate modules (IDENTICAL to habitat_cf.py — FROZEN) ───────────────────
class AdaptiveGateSlot(nn.Module):
    def __init__(s, d_model, d_s, K, slow_k, heads=4):
        super().__init__(); s.d_s, s.K, s.slow_k, s.heads, s.dh = d_s, K, slow_k, heads, d_s // heads
        s.read_in = nn.Linear(d_model, d_s)
        s.q, s.k, s.v = nn.Linear(d_s, d_s), nn.Linear(d_s, d_s), nn.Linear(d_s, d_s)
        s.gru = nn.GRUCell(d_s, d_s); s.ln = nn.LayerNorm(d_s)
        s.f_write = nn.Sequential(nn.Linear(2 * d_s, 128), nn.GELU(), nn.Linear(128, 1))
        s.S0 = nn.Parameter(torch.randn(K, d_s) * 0.02)
    def init(s): return s.S0.clone()
    def step(s, S, Hh):
        Hp = s.read_in(Hh.float())
        Q  = s.q(S).view(s.K, s.heads, s.dh).transpose(0, 1)
        Kk = s.k(Hp).view(-1, s.heads, s.dh).transpose(0, 1)
        Vv = s.v(Hp).view(-1, s.heads, s.dh).transpose(0, 1)
        a   = torch.softmax((Q @ Kk.transpose(-1, -2)) / (s.dh ** 0.5), dim=-1)
        ctx = (a @ Vv).transpose(0, 1).reshape(s.K, s.d_s)
        C   = s.gru(ctx, S)
        gw  = torch.sigmoid(s.f_write(torch.cat([S, Hp.mean(0, keepdim=True).expand(s.K, -1)], -1)))
        return s.ln(S + gw * (C - S))
    @property
    def slow(s): return slice(0, s.slow_k)


_fb = {'fields': None, 'S': None, 'on': False}
def _install():
    hs = []
    for L in FIELD_LAYERS:
        def mk(L):
            def hook(mod, inp, out):
                if not _fb['on']: return out
                h  = out[0] if isinstance(out, tuple) else out
                h2 = _fb['fields'][L](h, _fb['S'])
                return ((h2,) + tuple(out[1:])) if isinstance(out, tuple) else h2
            return hook
        hs.append(model.model.layers[L].register_forward_hook(mk(L)))
    return hs
_HANDLES = _install()


@torch.no_grad()
def gen(hist, S=None, field=False):
    _fb['S'] = S; _fb['on'] = field
    ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
    o   = model.generate(ids, max_new_tokens=8, do_sample=False,
                         attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    _fb['on'] = False
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()


@torch.no_grad()
def gen_read(hist):
    _fb['on'] = False
    ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
    o   = model.generate(ids, max_new_tokens=12, do_sample=False,
                         attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    ho  = model(o, output_hidden_states=True)
    Hh  = (ho.hidden_states[READ_LAYER][0, :ids.shape[1], :].float() if ids.shape[1] > 0
           else ho.hidden_states[READ_LAYER][0, -1:, :].float())
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True), Hh


def buildS_at(g, hids, upto):
    S = g.init()
    for h in hids[:upto + 1]: S = g.step(S, h.to(dev).float())
    return S


# ── Reference rollout (validate) ──────────────────────────────────────────────
def rollout_ref(w, responder):
    hist = []; ws = newws(); dk = 0; pdc = []; snaps = []
    for kind, text in w['turns']:
        if kind == 'off_react': text = _fill_off_react(text, ws)
        hist.append({'role': 'user', 'content': text})
        if kind == 'decision':
            snaps.append([m['content'][:80] for m in hist[-WINDOW:]])
            r = responder('decision', w, dk)
            if r is None: r = gen(hist)
            hist.append({'role': 'assistant', 'content': r})
            correct = spec_for(dk)['key'][CODEWORDS[w['ci']]]
            ru = r.upper(); found = [a for a in ACTIONS if a in ru]
            isc = (len(found) == 1 and found[0] == correct)
            judge_rcf(dk, r, w, ws); pdc.append((dk, r, isc)); dk += 1
        else:
            r = responder(kind, w, dk)
            if r is None: r = gen(hist)
            hist.append({'role': 'assistant', 'content': r})
    return pdc, ws, snaps


def validate():
    rng = random.Random(SEED)
    worlds = []
    for ci in range(N_CLASSES):
        for j in range(6): worlds.append(make_world(rng, j % N_TRAIN_TEMPLATES, ci))
    print('=== HAB_RCF VALIDATION (N=%d worlds, %d classes, WINDOW=%d, DEC_N=%d) ===' % (len(worlds), N_CLASSES, WINDOW, DEC_N), flush=True)
    for label, resp in [('oracle_good', oracle_good), ('base_blind', base_blind), ('always_first', always_first)]:
        correct = 0; tot = 0; act = {a: 0 for a in ACTIONS}; act['OTHER'] = 0; dumps = []
        for w in worlds:
            pdc, ws, snaps = rollout_ref(w, resp)
            for (k, a, isc) in pdc:
                tot += 1; correct += 1 if isc else 0
                au = a.upper(); f = [x for x in ACTIONS if x in au]
                if len(f) == 1: act[f[0]] += 1
                else: act['OTHER'] += 1
            if label == 'base_blind' and len(dumps) < 3: dumps.append(snaps)
        rate = correct / max(1, tot)
        if   label == 'oracle_good': status = 'PASS' if rate >= 0.90 else 'FAIL'
        elif label == 'base_blind':  status = 'PASS' if rate <= 0.35 else 'FAIL'
        else:                        status = 'PASS' if 0.15 <= rate <= 0.35 else 'FAIL'
        print('  %-12s correct_rate(4dec)=%.3f [%s] | actions=%s' % (label, rate, status, act), flush=True)
        if label == 'base_blind':
            n = max(1, tot); flagged = [a for a, c in act.items() if a != 'OTHER' and c / n > 0.40]
            if flagged: print('  WARNING: base_blind action >40%%: %s' % flagged, flush=True)
            for wi, snaps in enumerate(dumps):
                for k, snap in enumerate(snaps):
                    print('  base w%d D%d hist[-W:]: %s' % (wi, k, snap), flush=True)
    print('  TARGETS: oracle>=0.90 PASS | base<=0.35 PASS | always~0.25 PASS', flush=True)
    print('=== HAB_RCF_VALID_DONE ===', flush=True)


# ── Substrate collection ──────────────────────────────────────────────────────
def collect(worlds):
    data = []
    for w in worlds:
        hist = []; hids = []; dec = []; dk = 0; ws = newws()
        for kind, text in w['turns']:
            if kind == 'off_react': text = _fill_off_react(text, ws)
            hist.append({'role': 'user', 'content': text})
            r, Hh = gen_read(hist)
            hist.append({'role': 'assistant', 'content': r})
            hids.append(Hh.to(torch.float16).cpu())
            if kind == 'decision':
                judge_rcf(dk, r, w, ws)   # drives reactive off_react
                dec.append((len(hids) - 1, dk, text, spec_for(dk)['key'][CODEWORDS[w['ci']]]))
                dk += 1
        data.append((hids, dec, w))
    return data


def substrate():
    rng = random.Random(SEED)
    train_w = []
    for ci in range(N_CLASSES):
        for tpl_idx in range(N_TRAIN_TEMPLATES):
            for _ in range(N_WORLDS_PER_SLOT): train_w.append(make_world(rng, tpl_idx, ci))
    random.shuffle(train_w)
    indist_w = [make_world(rng, j % N_TRAIN_TEMPLATES, ci) for ci in range(N_CLASSES) for j in range(3)]
    # held-out test = combined over the held-out templates (idx N_TRAIN_TEMPLATES + h); split per-template at eval via w['tpl_idx']
    test_w = [make_world(rng, N_TRAIN_TEMPLATES + h, ci)
              for h in range(N_HELDOUT_TEMPLATES) for ci in range(N_CLASSES) for _ in range(3)]

    CKDIR = '/home/pokazge/checkpoints'; os.makedirs(CKDIR, exist_ok=True)
    EVAL_ONLY = os.environ.get('EVAL_ONLY', '0') == '1'
    _sfx = '' if (N_CLASSES == 4 and N_ACTIONS == 4 and N_TRAIN_TEMPLATES == 3 and DEC_N == 4 and SURFACE_MODE == 'diverse') else '_nc%d_na%d_nt%d_d%d_%s' % (N_CLASSES, N_ACTIONS, N_TRAIN_TEMPLATES, DEC_N, SURFACE_MODE)   # canonical run keeps old name
    rcache = '%s/habrcf_rollouts_s%d%s.pt' % (CKDIR, SEED, _sfx)
    gck    = '%s/habrcf_trained_s%d%s.pt' % (CKDIR, SEED, _sfx)
    if os.path.exists(rcache):
        print('loading cached rollouts %s' % rcache, flush=True)
        _d = torch.load(rcache, weights_only=False); tr, idD, teD = _d['tr'], _d['idD'], _d['teD']
    else:
        print('collecting rollouts train=%d indist=%d test=%d ...' % (len(train_w), len(indist_w), len(test_w)), flush=True)
        tr = collect(train_w); idD = collect(indist_w); teD = collect(test_w)
        torch.save({'tr': tr, 'idD': idD, 'teD': teD}, rcache); print('saved rollouts cache %s' % rcache, flush=True)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    fp  = [p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]
    eos = torch.tensor([[tok.eos_token_id]], device=dev)
    opt = torch.optim.Adam(list(g.parameters()) + fp, lr=float(os.environ.get('LR', '5e-4')))

    def ce_action(qtext, correct, S):
        _fb['S'] = S; _fb['on'] = True
        pids   = tok(H.tmpl([{'role': 'user', 'content': qtext}]), return_tensors='pt').input_ids.to(dev)
        vids   = tok(correct, add_special_tokens=False, return_tensors='pt').input_ids.to(dev)
        ids    = torch.cat([pids, vids, eos], 1); P = pids.shape[1]; Lt = ids.shape[1] - P
        logits = model(ids).logits[0].float()
        loss   = F.cross_entropy(logits[P - 1:P + Lt - 1], ids[0, P:P + Lt])
        _fb['on'] = False; return loss

    if EVAL_ONLY and os.path.exists(gck):
        _sd = torch.load(gck, map_location=dev, weights_only=False)
        g.load_state_dict(_sd['g'])
        for L in FIELD_LAYERS: _fb['fields'][L].load_state_dict(_sd['fields'][str(L)])
        print('EVAL_ONLY: loaded trained ckpt %s, skipping training' % gck, flush=True)
    else:
        print('training substrate (CE toward correct action at EACH of 4 decisions / recurrent pressure) ...', flush=True)
        for epn in range(FIELD_EPOCHS):
            random.shuffle(tr); tot = 0.0; nb = 0
            for hids, dec, w in tr:
                opt.zero_grad(); wl = 0.0
                for ti, k, qtext, correct in dec:
                    S = buildS_at(g, hids, ti)
                    loss = ce_action(qtext, correct, S) / max(1, len(dec))
                    if not torch.isfinite(loss): continue
                    loss.backward(); wl += float(loss)
                torch.nn.utils.clip_grad_norm_(list(g.parameters()) + fp, 1.0)
                opt.step(); tot += wl; nb += 1
            if epn % 5 == 0 or epn == FIELD_EPOCHS - 1:
                print('  ep %d | action-CE=%.4f' % (epn, tot / max(1, nb)), flush=True)
        torch.save({'g': g.state_dict(), 'fields': {str(L): _fb['fields'][L].state_dict() for L in FIELD_LAYERS}}, gck)
        print('saved trained ckpt %s' % gck, flush=True)

    # diagnostics: per-decision class separation in S under recurrent pressure
    cosD = {}
    print('--- diagnostics ---', flush=True)
    with torch.no_grad():
        deltas = []
        for hids, dec, w in tr:
            S0 = g.init(); S1 = g.step(S0, hids[0].to(dev).float()); deltas.append((S1 - S0).norm().item())
        print('  mean_write_delta_norm@commit = %.4f' % (sum(deltas) / max(1, len(deltas))), flush=True)
        for k in range(DEC_N):
            aS = []; bS = []
            for hids, dec, w in tr:
                if k >= len(dec): continue
                S = buildS_at(g, hids, dec[k][0]).reshape(-1)
                if w['ci'] == 0: aS.append(S)
                elif w['ci'] == 1: bS.append(S)
            if aS and bS:
                cos = F.cosine_similarity(torch.stack(aS).mean(0).unsqueeze(0), torch.stack(bS).mean(0).unsqueeze(0)).item()
                cosD[k] = cos
                print('  cosine_S(ALPHA,BETA)@D%d = %.4f' % (k, cos), flush=True)
        # action-sharing-pair separation (the hard pairs at N>=6; separable only at non-shared decisions)
        for (a, b) in [(0, 4), (1, 5)]:
            if b >= N_CLASSES: continue
            for k in range(DEC_N):
                Sa = [buildS_at(g, hids, dec[k][0]).reshape(-1) for hids, dec, w in tr if w['ci'] == a and k < len(dec)]
                Sb = [buildS_at(g, hids, dec[k][0]).reshape(-1) for hids, dec, w in tr if w['ci'] == b and k < len(dec)]
                if Sa and Sb:
                    cz = F.cosine_similarity(torch.stack(Sa).mean(0).unsqueeze(0), torch.stack(Sb).mean(0).unsqueeze(0)).item()
                    print('  cosine_S(%s,%s)@D%d = %.4f' % (CLASSES[a], CLASSES[b], k, cz), flush=True)

    # metric-10 prep: per-(decision,class) trained-S centroids (from train rollouts)
    NDEC = len(tr[0][1]) if tr else 4
    cent = {}
    with torch.no_grad():
        for k in range(NDEC):
            for ci in range(N_CLASSES):
                Ss = [buildS_at(g, hids, dec[k][0]).reshape(-1) for hids, dec, w in tr if w['ci'] == ci and k < len(dec)]
                if Ss: cent[(k, ci)] = torch.stack(Ss).mean(0)

    for p in g.parameters(): p.requires_grad_(False)
    gfrozen = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)

    @torch.no_grad()
    def evalsplit(name, data):
        metrics = {}
        for mode in ['trained', 'reset', 'frozen', 'stale', 'stale_same', 'base']:
            vias = []; correct = 0; contam = 0; ambig = 0; totdec = 0
            cc = [0] * N_CLASSES; ct = [0] * N_CLASSES; dpc = [0] * NDEC; dpt = [0] * NDEC
            sdc = 0; sdt = 0
            conf = [[0] * (len(ACTIONS) + 1) for _ in range(len(ACTIONS))]   # [correct_action][emitted: N actions + OTHER]  (metric 7)
            mc = []; mw = []; sn = 0; sacc = 0; sg_n = 0; sg_c = 0; sb_n = 0; sb_c = 0   # metric 10
            for hids, dec, w in data:
                ws = newws(); via = 0.0
                stale_hids = None; stale_ci = None
                if mode in ('stale', 'stale_same'):
                    tci = (w['ci'] + 1) % N_CLASSES if mode == 'stale' else w['ci']   # Ablation C: diff-class vs same-class diff-world
                    cands = [(d_h, d_w) for d_h, d_dec, d_w in tr if d_w['ci'] == tci and d_w is not w]
                    if not cands: cands = [(d_h, d_w) for d_h, d_dec, d_w in tr if d_w is not w]
                    src = random.choice(cands); stale_hids = src[0]; stale_ci = src[1]['ci']
                for ti, k, qtext, corr_act in dec:
                    if mode == 'base':
                        r = gen([{'role': 'user', 'content': qtext}])
                    else:
                        if   mode == 'trained': S = buildS_at(g, hids, ti)
                        elif mode == 'reset':   S = g.init()
                        elif mode == 'frozen':  S = buildS_at(gfrozen, hids, ti)
                        else:                   S = buildS_at(g, stale_hids, ti)
                        r = gen([{'role': 'user', 'content': qtext}], S, field=True)
                    r = r.split('</think>')[-1].strip()
                    via += judge_rcf(k, r, w, ws); totdec += 1
                    ct[w['ci']] += 1; dpt[k] += 1
                    ru = r.upper(); found = [a for a in ACTIONS if a in ru]
                    if len(found) == 1 and found[0] == corr_act:
                        cc[w['ci']] += 1; dpc[k] += 1
                    if mode == 'stale':
                        sdt += 1; exp = spec_for(k)['key'][CODEWORDS[stale_ci]]
                        if len(found) == 1 and found[0] == exp: sdc += 1
                    if mode == 'trained':
                        acorr = (len(found) == 1 and found[0] == corr_act)
                        ei = ACTIONS.index(found[0]) if len(found) == 1 else len(ACTIONS)
                        conf[ACTIONS.index(corr_act)][ei] += 1
                        if all((k, j) in cent for j in range(N_CLASSES)):
                            cs = [F.cosine_similarity(S.reshape(-1).unsqueeze(0), cent[(k, j)].unsqueeze(0)).item() for j in range(N_CLASSES)]
                            own = cs[w['ci']]; oth = max(cs[j] for j in range(N_CLASSES) if j != w['ci'])
                            (mc if acorr else mw).append(own - oth)
                            sgood = (max(range(N_CLASSES), key=lambda j: cs[j]) == w['ci'])
                            sn += 1; sacc += 1 if sgood else 0
                            if sgood: sg_n += 1; sg_c += 1 if acorr else 0
                            else: sb_n += 1; sb_c += 1 if acorr else 0
                vias.append(via); correct += ws['correct']; contam += ws['n_wrong']; ambig += ws['ambiguous']
            n = len(data); d = max(1, totdec)
            cr = correct / d; cmr = contam / d; ar = ambig / d; vm = sum(vias) / max(1, n)
            sdr = sdc / max(1, sdt) if mode == 'stale' else None
            ccr = [cc[i] / max(1, ct[i]) for i in range(N_CLASSES)]
            dpcr = [dpc[i] / max(1, dpt[i]) for i in range(NDEC)]
            line = '  [%s] %-8s correct=%.3f contam=%.3f ambig=%.3f viability=%+.3f' % (name, mode, cr, cmr, ar, vm)
            if mode == 'stale': line += ' stale_dir=%.3f' % sdr
            print(line, flush=True)
            decpos = ' '.join('D%d=%.2f' % (i, dpcr[i]) for i in range(NDEC))
            pcl = ' '.join('%s=%.2f' % (CLASSES[i][:2], ccr[i]) for i in range(N_CLASSES))
            print('  [%s] %-8s per-class %s | dec_pos %s' % (name, mode, pcl, decpos), flush=True)
            metrics[mode] = {'correct_rate': cr, 'viability': vm, 'stale_dir_rate': sdr if sdr is not None else 0.0}
            if mode == 'trained':
                _ap = [ci for ci in range(N_CLASSES) if spec_for(0)['key'][CODEWORDS[ci]] != 'RELEASE']   # classes whose D0 action != Qwen's RELEASE prior
                metrics[mode]['anti_prior_cr'] = sum(cc[ci] for ci in _ap) / max(1, sum(ct[ci] for ci in _ap))
                print('  [%s] CONFUSION rows=correct-action cols=emitted[%s,OTHER]:' % (name, ','.join(ACTIONS)), flush=True)
                for ai, an in enumerate(ACTIONS):
                    print('     %-8s %s' % (an, conf[ai]), flush=True)
                if sn > 0:
                    print('  [%s] metric10 S-sep<->correct: nearest-centroid_S-class_acc=%.3f | mean_margin correct=%+.4f wrong=%+.4f | P(correct|S_good)=%.3f P(correct|S_bad)=%.3f' % (
                        name, sacc / max(1, sn), sum(mc) / max(1, len(mc)), sum(mw) / max(1, len(mw)),
                        sg_c / max(1, sg_n), sb_c / max(1, sb_n)), flush=True)
        return metrics

    print('=== HAB_RCF_REPORT (CUMULATIVE viability; recurrent diverse-surface class dependence) ===', flush=True)
    evalsplit('IN-DIST', idD)
    for h in range(N_HELDOUT_TEMPLATES):                          # per-held-out-template transfer (diagnose template- vs class-specific)
        tpl_idx = N_TRAIN_TEMPLATES + h
        sub = [d for d in teD if d[2]['tpl_idx'] == tpl_idx]
        if sub: evalsplit('HELD-OUT-T%d' % tpl_idx, sub)
    ht = evalsplit('HELD-OUT-ALL', teD)                           # combined => verdict

    htv = ht['trained']['viability']; hrv = ht['reset']['viability']
    sdr = ht['stale']['stale_dir_rate']; ap = ht['trained']['anti_prior_cr']
    htc = ht['trained']['correct_rate']; hbc = ht['base']['correct_rate']
    cD0 = cosD.get(0, 1.0); cD3 = cosD.get(DEC_N - 1, 1.0)
    if htv >= 1.5 and (htv - hrv) >= 0.8 and sdr >= 0.40 and ap >= 0.30 and hbc <= 0.35:
        verdict = 'EMERGENCE'
    elif htv > hrv + 0.2:
        verdict = 'partial_selection_pressure'
    elif htc <= hbc + 0.05 and cD0 >= 0.99 and cD3 >= 0.99:
        verdict = 'STILL_COLLAPSES'
    elif htc <= hbc + 0.05:
        verdict = 'no_effect'
    else:
        verdict = 'field_biases_preference'
    print('HAB_RCF_VERDICT: %s (held-out trained_via=%.3f reset_via=%.3f stale_dir=%.3f anti_prior=%.3f trained_cr=%.3f base_cr=%.3f cosD0=%.3f cosD3=%.3f)' % (
        verdict, htv, hrv, sdr, ap, htc, hbc, cD0, cD3), flush=True)
    print('=== HAB_RCF_DONE ===', flush=True)


if MODE == 'validate': validate()
else: substrate()
print('=== HAB_RCF_END ===', flush=True)
