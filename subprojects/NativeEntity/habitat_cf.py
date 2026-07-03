# HABITAT_COUNTERFACTUAL_DEPENDENCE_V1 — worlds where correct future ACTION is choosable only if
# persistent S preserved an earlier-committed latent CLASS.
# CLASS→ACTION map: ALPHA→RELEASE, BETA→DEFER, GAMMA→REJECT, DELTA→REPAIR.
# 10-turn trajectory; both decision prompts are separated from commit by >=4 off-topic turns.
# Measured on ACTUAL generated actions (not logprob margin).
import os, random
import torch, torch.nn as nn, torch.nn.functional as F
import habitat_evo as H
import slots as SL

dev = H.dev; model = H.model; tok = H.tok
SEED = int(os.environ.get('SEED', '0')); random.seed(SEED); torch.manual_seed(SEED)
D_MODEL = model.config.hidden_size if getattr(model.config, 'hidden_size', None) else getattr(model.config.text_config, 'hidden_size', 5120)
N_LAYERS = model.config.num_hidden_layers if getattr(model.config, 'num_hidden_layers', None) else getattr(model.config.text_config, 'num_hidden_layers', 64)
READ_LAYER = N_LAYERS // 2
K = int(os.environ.get('K', '12')); SLOW_K = int(os.environ.get('SLOW_K', '6')); D_S = int(os.environ.get('D_S', '768'))
FIELD_LAYERS = [int(x) for x in os.environ.get('FIELD_LAYERS', '40,48,56').split(',')]   # FIX9: 3 layers
EPS = float(os.environ.get('EPS', '0.12'))                                                # FIX9: default 0.12
NTOK = int(os.environ.get('NTOK', '24'))
N_TRAIN  = int(os.environ.get('N_TRAIN',  '60'))   # FIX9: 5 per class × 3 train-templates × 4 classes
N_INDIST = int(os.environ.get('N_INDIST', '20'))   # FIX9
N_TEST   = int(os.environ.get('N_TEST',   '24'))   # FIX9: >=24, 6 per class
FIELD_EPOCHS = int(os.environ.get('FIELD_EPOCHS', '50'))
MODE = os.environ.get('CFMODE', 'validate')        # validate | substrate
print('HAB-CF | mode=%s d_model=%d read=%d | K=%d slow_k=%d d_s=%d FIELD=%s EPS=%.2f' % (
    MODE, D_MODEL, READ_LAYER, K, SLOW_K, D_S, FIELD_LAYERS, EPS), flush=True)

CLASSES = ['ALPHA', 'BETA', 'GAMMA', 'DELTA']
ACTIONS = ['RELEASE', 'DEFER', 'REJECT', 'REPAIR']  # MAP: CLASSES[i] -> ACTIONS[i]

# FIX9: 3 train templates + 1 held-out (surface-different: medical sample / lab technician)
TEMPLATES = [
    {'item': 'consignment',    'agent': 'a courier',            'agent2': 'a second courier'},
    {'item': 'archive box',    'agent': 'a researcher',         'agent2': 'another researcher'},
    {'item': 'supply crate',   'agent': 'a porter',             'agent2': 'a relief porter'},
    {'item': 'medical sample', 'agent': 'a lab technician',     'agent2': 'a second lab technician'},  # held-out
]

# FIX1: OFF pool — 6 generic texts that NEVER mention class/item/action words
OFF_POOL = [
    "Aside: how does compound interest work?",
    "Unrelated: explain the rules of chess briefly.",
    "Different topic: how do rainbows form?",
    "Quick aside: what causes ocean tides?",
    "Unrelated: how does photosynthesis work?",
    "Different topic: how are sailing knots tied?",
]

MAPLINE = "Standing mapping — ALPHA: release, BETA: defer, GAMMA: reject, DELTA: repair."
CODEWORDS = ['FOXTROT', 'KILO', 'NOVEMBER', 'SIERRA']  # FOXTROT=ALPHA→RELEASE, KILO=BETA→DEFER, NOVEMBER=GAMMA→REJECT, SIERRA=DELTA→REPAIR
KEYLINE = "Key: FOXTROT=RELEASE, KILO=DEFER, NOVEMBER=REJECT, SIERRA=REPAIR."
ASK = "Reply with ONE word: RELEASE, DEFER, REJECT, or REPAIR."
DECISION2_SENTINEL = '__DECISION2_SENTINEL__'


def make_world(rng, tpl, ci=None):
    """Build a 10-turn world dict.  FIX1 layout (window = hist[-6:]):
      t0 commit+MAPLINE | t1 distractor | t2 off[0] | t3 off[1] | t4 off[2]
      t5 decision1(ASK) | t6 off[3]    | t7 off[4] | t8 off[5] | t9 decision2(sentinel)
    Window arithmetic:
      decision1 (t5): hist has 11 entries (t0u,t0a,...,t4a,t5u); hist[-6:] = [t2a..t5u]
        -> commit(t0u) and distractor-response(t1a) are OUT of window. ✓
      decision2 (t9): hist has 19 entries (t0u..t8a,t9u); hist[-6:] = [t6a..t9u]
        -> decision1 user(t5u) and decision1 answer(t5a) are OUT of window. ✓
    decision2 text is a sentinel filled reactively at rollout time (FIX3).
    """
    if ci is None: ci = rng.randrange(4)
    CW = CODEWORDS[ci]
    turns = [
        ('commit',    'Filed code: %s %s %s %s %s %s. Acknowledge.' % (CW, CW, CW, CW, CW, CW)),
        ('distractor','Logistics note: the loading dock is busy today.'),
        ('off',       OFF_POOL[0]),
        ('off',       OFF_POOL[1]),
        ('off',       OFF_POOL[2]),
        ('decision',  '%s requests action on the %s. %s %s' % (tpl['agent'], tpl['item'], KEYLINE, ASK)),
        ('off',       OFF_POOL[3]),
        ('off',       OFF_POOL[4]),
        ('off',       OFF_POOL[5]),
        ('decision2', DECISION2_SENTINEL),
    ]
    return {'tpl': tpl, 'ci': ci, 'cls': CLASSES[ci], 'correct': ACTIONS[ci], 'turns': turns}


def _fill_decision2(world_state, w):
    """FIX3: reactive decision2 built after decision1 is judged.
    Correct answer is ALWAYS w['correct']. Text does NOT name the action taken at decision1."""
    if world_state.get('correct', 0) > 0:
        return ('Handover confirmed for the %s. Register the final disposition. %s %s'
                % (w['tpl']['item'], KEYLINE, ASK))
    else:
        return ('The previous action on the %s has created an inconsistency. '
                'Issue a corrective ruling. %s %s' % (w['tpl']['item'], KEYLINE, ASK))


def judge(resp, w, world_state):
    """FIX4: judge actual generated action; update world_state in-place; return viability score.
    Trajectory viability = sum of scores over the 2 decisions."""
    r = resp.upper(); found = [a for a in ACTIONS if a in r]
    if len(found) == 1 and found[0] == w['correct']:
        world_state['correct'] = world_state.get('correct', 0) + 1
        return 1.0
    if len(found) == 1:
        world_state['wrong_action']  = world_state.get('wrong_action',  0) + 1
        world_state['contaminated']  = world_state.get('contaminated',  0) + 1
        return -1.0
    world_state['ambiguous'] = world_state.get('ambiguous', 0) + 1
    return -0.5


def oracle_good(kind, w):
    if kind in ('decision', 'decision2'): return w['correct']
    return 'Understood.'

def base_blind(kind, w): return None   # filled by model gen

def always_one(kind, w):
    if kind in ('decision', 'decision2'): return 'RELEASE'
    return 'Understood.'


# ── Substrate modules ────────────────────────────────────────────────────────

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


# ── Field hooks ──────────────────────────────────────────────────────────────

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

_HANDLES = _install()   # FIX9: install once


# ── Generation helpers ───────────────────────────────────────────────────────

@torch.no_grad()
def gen(hist, S=None, field=False):
    """FIX9: ~8 new tokens for action; strip </think>."""
    _fb['S'] = S; _fb['on'] = field
    ids = tok(H.tmpl(hist[-6:]), return_tensors='pt').input_ids.to(dev)
    o   = model.generate(ids, max_new_tokens=8, do_sample=False,
                         attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    _fb['on'] = False
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()


@torch.no_grad()
def gen_read(hist):
    """Generate response and capture mid-layer hidden states for S-building."""
    _fb['on'] = False
    ids = tok(H.tmpl(hist[-6:]), return_tensors='pt').input_ids.to(dev)
    o   = model.generate(ids, max_new_tokens=12, do_sample=False,
                         attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    ho  = model(o, output_hidden_states=True)
    # Read PROMPT tokens (where the class label lives), not response tokens.
    Hh  = (ho.hidden_states[READ_LAYER][0, :ids.shape[1], :].float() if ids.shape[1] > 0
           else ho.hidden_states[READ_LAYER][0, -1:, :].float())
    # Return full prompt-hidden span — no tail-clip — so early class tokens are included.
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True), Hh


# ── Reference rollouts ───────────────────────────────────────────────────────

def rollout_ref(w, responder):
    """Full 10-turn reference rollout; fills decision2 sentinel reactively (FIX3).
    Returns (trajectory_viability, world_state)."""
    hist = []; via = 0.0
    world_state = {'correct': 0, 'contaminated': 0, 'wrong_action': 0, 'ambiguous': 0}
    for kind, text in w['turns']:
        if kind == 'decision2' and text == DECISION2_SENTINEL:
            text = _fill_decision2(world_state, w)
        hist.append({'role': 'user', 'content': text})
        r = responder(kind, w)
        if r is None: r = gen(hist)
        hist.append({'role': 'assistant', 'content': r})
        if kind in ('decision', 'decision2'):
            v = judge(r, w, world_state); via += v
    return via, world_state


def rollout_d1(w, responder):
    """Run only through decision1 (turn 5).
    Returns (action_str, hist_snapshot_at_d1_including_user_msg)."""
    hist = []
    for kind, text in w['turns'][:6]:   # turns 0-5 inclusive
        hist.append({'role': 'user', 'content': text})
        if kind == 'decision':
            hist_snap = list(hist)       # hist[-6:] is the leakage-check window
            r = responder(kind, w)
            if r is None: r = gen(hist)
            return r, hist_snap
        r = responder(kind, w)
        if r is None: r = gen(hist)
        hist.append({'role': 'assistant', 'content': r})
    return None, hist


# ── Validate mode (FIX7) ─────────────────────────────────────────────────────

def validate():
    """FIX7: N_TEST>=24 worlds, 6 per class; decision1-only correct_rate; PASS/FAIL; leakage check."""
    rng = random.Random(SEED)
    # Stratify: exactly 6 worlds per class, cycling over the 3 train templates
    worlds = []
    for ci in range(4):
        for j in range(6):
            worlds.append(make_world(rng, TEMPLATES[j % 3], ci))
    assert len(worlds) == 24 and N_TEST >= 24

    print('=== CF HABITAT VALIDATION (N=%d, 6 per class, window=hist[-6:]) ===' % len(worlds), flush=True)

    for label, resp in [('oracle_good', oracle_good), ('base_blind', base_blind), ('always_one', always_one)]:
        correct1 = 0
        act_cnt  = {a: 0 for a in ACTIONS}; act_cnt['OTHER'] = 0
        hist_dumps = []

        for w in worlds:
            r, hist_snap = rollout_d1(w, resp)
            if r is None: continue
            r = r.split('</think>')[-1].strip()
            ws2 = {'correct': 0}; judge(r, w, ws2)
            correct1 += ws2.get('correct', 0)
            r_up = r.upper(); found = [a for a in ACTIONS if a in r_up]
            if len(found) == 1: act_cnt[found[0]] += 1
            else:               act_cnt['OTHER']   += 1
            if label == 'base_blind' and len(hist_dumps) < 5:
                hist_dumps.append(hist_snap[-6:])

        n = len(worlds); rate1 = correct1 / n
        if   label == 'oracle_good': status = 'PASS' if rate1 >= 0.90 else 'FAIL'
        elif label == 'base_blind':  status = 'PASS' if rate1 <= 0.38 else 'FAIL'
        else:                        status = 'PASS' if 0.15 <= rate1 <= 0.35 else 'FAIL'
        print('  %-12s decision1_correct=%.3f [%s] | actions=%s' % (label, rate1, status, act_cnt), flush=True)

        if label == 'base_blind':
            flagged = [a for a, c in act_cnt.items() if a != 'OTHER' and c / n > 0.40]
            if flagged:
                print('  WARNING: base_blind action >40%%: %s' % flagged, flush=True)
            print('  --- base_blind hist[-6:] leakage check (5 worlds) ---', flush=True)
            for i, snap in enumerate(hist_dumps):
                print('  world%d hist[-6:]: %s' % (i, [m['content'][:70] for m in snap]), flush=True)

    print('  TARGETS: oracle>=0.90 PASS | base<=0.38 PASS | always~0.25 PASS', flush=True)
    print('=== CF_VALID_DONE ===', flush=True)


# ── Substrate collection ─────────────────────────────────────────────────────

def collect(worlds):
    """Collect gen_read rollouts; fill decision2 sentinel reactively from base decision1 (FIX3)."""
    data = []
    for w in worlds:
        hist = []; hids = []; dec = []
        world_state = {'correct': 0, 'contaminated': 0, 'wrong_action': 0, 'ambiguous': 0}
        for kind, text in w['turns']:
            if kind == 'decision2' and text == DECISION2_SENTINEL:
                text = _fill_decision2(world_state, w)
            hist.append({'role': 'user', 'content': text})
            r, Hh = gen_read(hist)
            hist.append({'role': 'assistant', 'content': r})
            hids.append(Hh.to(torch.float16).cpu())
            if kind in ('decision', 'decision2'):
                judge(r, w, world_state)
                dec.append((len(hids) - 1, kind, text))
        data.append((hids, dec, w))
    return data


def buildS_at(g, hids, upto):
    S = g.init()
    for h in hids[:upto + 1]: S = g.step(S, h.to(dev).float())
    return S


# ── Substrate mode (FIX8) ────────────────────────────────────────────────────

def substrate():
    rng = random.Random(SEED)

    # FIX9: balanced train worlds — 5 per class × 3 train-templates × 4 classes = 60
    train_w = []
    for ci in range(4):
        for tpl_idx in range(3):
            for _ in range(5):
                train_w.append(make_world(rng, TEMPLATES[tpl_idx], ci))
    random.shuffle(train_w)

    # In-dist eval: 5 per class from train templates = 20
    indist_w = []
    for ci in range(4):
        for j in range(5):
            indist_w.append(make_world(rng, TEMPLATES[j % 3], ci))

    # Held-out template: 6 per class using TEMPLATES[3] only = 24  (FIX9)
    test_w = []
    for ci in range(4):
        for _ in range(6):
            test_w.append(make_world(rng, TEMPLATES[3], ci))

    print('collecting rollouts train=%d indist=%d test=%d ...' % (len(train_w), len(indist_w), len(test_w)), flush=True)
    tr = collect(train_w); idD = collect(indist_w); teD = collect(test_w)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    fp  = [p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]
    eos = torch.tensor([[tok.eos_token_id]], device=dev)
    opt = torch.optim.Adam(list(g.parameters()) + fp, lr=float(os.environ.get('LR', '5e-4')))

    def ce_action(qtext, correct, S):
        """Train: CE toward correct ACTION token under field+S; requires preserved class in S."""
        _fb['S'] = S; _fb['on'] = True
        pids   = tok(H.tmpl([{'role': 'user', 'content': qtext}]), return_tensors='pt').input_ids.to(dev)
        vids   = tok(correct, add_special_tokens=False, return_tensors='pt').input_ids.to(dev)
        ids    = torch.cat([pids, vids, eos], 1); P = pids.shape[1]; Lt = ids.shape[1] - P
        logits = model(ids).logits[0].float()
        loss   = F.cross_entropy(logits[P - 1:P + Lt - 1], ids[0, P:P + Lt])
        _fb['on'] = False; return loss

    print('training substrate (CE toward correct ACTION via preserved class) ...', flush=True)
    for epn in range(FIELD_EPOCHS):
        random.shuffle(tr); tot = 0.0; nb = 0
        for hids, dec, w in tr:
            for ti, kind, qtext in dec:
                S    = buildS_at(g, hids, ti)
                loss = ce_action(qtext, w['correct'], S)
                if not torch.isfinite(loss): continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(g.parameters()) + fp, 1.0)
                opt.step(); tot += float(loss); nb += 1
        if epn % 10 == 0 or epn == FIELD_EPOCHS - 1:
            print('  ep %d | action-CE=%.4f' % (epn, tot / max(1, nb)), flush=True)

    # FIX8: post-training diagnostics
    print('--- diagnostics ---', flush=True)
    with torch.no_grad():
        # Mean write-delta norm at commit turn (ti=0)
        deltas = []
        for hids, dec, w in tr:
            S0 = g.init(); S1 = g.step(S0, hids[0].to(dev).float())
            deltas.append((S1 - S0).norm().item())
        print('  mean_write_delta_norm@commit = %.4f' % (sum(deltas) / max(1, len(deltas))), flush=True)
        # Pairwise cosine S(ALPHA) vs S(BETA) at decision turn
        alpha_S = []; beta_S = []
        for hids, dec, w in tr:
            dec_ti = next((ti for ti, kind, _ in dec if kind == 'decision'), None)
            if dec_ti is None: continue
            S = buildS_at(g, hids, dec_ti).reshape(-1)
            if w['cls'] == 'ALPHA': alpha_S.append(S)
            elif w['cls'] == 'BETA': beta_S.append(S)
        if alpha_S and beta_S:
            a   = torch.stack(alpha_S).mean(0)
            b   = torch.stack(beta_S).mean(0)
            cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            print('  cosine_S(ALPHA_mean, BETA_mean)@decision = %.4f' % cos, flush=True)
        # Per-turn class-separation probe on raw collected hiddens.
        for turn_i in range(6):
            av = [hids[turn_i].float().mean(0) for hids, dec, w in tr if w['cls'] == 'ALPHA' and len(hids) > turn_i]
            bv = [hids[turn_i].float().mean(0) for hids, dec, w in tr if w['cls'] == 'BETA' and len(hids) > turn_i]
            if av and bv:
                ca = torch.stack(av).mean(0).unsqueeze(0); cb = torch.stack(bv).mean(0).unsqueeze(0)
                print('  cosine(raw_hids[%d] ALPHA vs BETA) = %.6f' % (turn_i, F.cosine_similarity(ca, cb).item()), flush=True)

    for p in g.parameters(): p.requires_grad_(False)
    gfrozen = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)

    @torch.no_grad()
    def evalsplit(name, data):
        """FIX8: eval all 5 conditions; report per condition: correct_action_rate, contamination_rate,
        ambiguous_rate, trajectory_viability_mean, and (stale) stale_direction_rate."""
        metrics = {}
        for mode in ['trained', 'reset', 'frozen', 'stale', 'base']:
            vias = []; correct_cnt = 0; contam_cnt = 0; ambig_cnt = 0; total_dec = 0
            stale_dir_cnt = 0; stale_total = 0
            class_correct = [0, 0, 0, 0]; class_total = [0, 0, 0, 0]

            for idx, (hids, dec, w) in enumerate(data):
                via = 0.0
                world_state = {'correct': 0, 'contaminated': 0, 'wrong_action': 0, 'ambiguous': 0}

                # FIX5+FIX6: stale-S from a world of class [(ci+1)%4], exclude current by identity
                stale_hids = None; stale_expected_action = None
                if mode == 'stale':
                    target_ci  = (w['ci'] + 1) % 4
                    # FIX6: exclude current world by identity
                    candidates = [(d_h, d_w) for d_h, d_dec, d_w in tr
                                  if d_w['ci'] == target_ci and d_w is not w]
                    if not candidates:   # fallback: any other world, still exclude by identity
                        candidates = [(d_h, d_w) for d_h, d_dec, d_w in tr if d_w is not w]
                    stale_src  = random.choice(candidates)
                    stale_hids = stale_src[0]
                    stale_expected_action = ACTIONS[stale_src[1]['ci']]   # FIX5: action of stale source class

                for ti, kind, qtext in dec:
                    if mode == 'base':
                        r = gen([{'role': 'user', 'content': qtext}])
                    else:
                        if   mode == 'trained': S = buildS_at(g,       hids,       ti)
                        elif mode == 'reset':   S = g.init()
                        elif mode == 'frozen':  S = buildS_at(gfrozen, hids,       ti)
                        else:                   S = buildS_at(g,       stale_hids, ti)  # stale
                        r = gen([{'role': 'user', 'content': qtext}], S, field=True)

                    r = r.split('</think>')[-1].strip()   # FIX9: strip think tokens
                    v = judge(r, w, world_state); via += v; total_dec += 1
                    class_total[w['ci']] += 1

                    # FIX5: track stale direction rate
                    if mode == 'stale' and stale_expected_action:
                        stale_total += 1
                        r_up = r.upper(); found = [a for a in ACTIONS if a in r_up]
                        if len(found) == 1 and found[0] == stale_expected_action:
                            stale_dir_cnt += 1

                vias.append(via)
                correct_cnt += world_state.get('correct',     0)
                contam_cnt  += world_state.get('contaminated', 0)
                ambig_cnt   += world_state.get('ambiguous',   0)
                class_correct[w['ci']] += world_state.get('correct', 0)

            n = len(data); d = max(1, total_dec)
            cr  = correct_cnt / d; cmr = contam_cnt / d; ar = ambig_cnt / d
            vm  = sum(vias) / n
            sdr = stale_dir_cnt / max(1, stale_total) if mode == 'stale' else None
            line = ('  [%s] %-8s correct_rate=%.3f contam_rate=%.3f ambig_rate=%.3f viability=%+.3f'
                    % (name, mode, cr, cmr, ar, vm))
            if mode == 'stale': line += ' stale_dir_rate=%.3f' % sdr
            print(line, flush=True)
            ccr = [class_correct[i] / max(1, class_total[i]) for i in range(4)]
            print('  [%s] %-8s per-class correct: A=%.2f B=%.2f G=%.2f D=%.2f' % (
                name, mode, ccr[0], ccr[1], ccr[2], ccr[3]), flush=True)
            metrics[mode] = {'correct_rate': cr, 'viability': vm,
                             'stale_dir_rate': sdr if sdr is not None else 0.0}
            if mode == 'trained':
                metrics[mode]['anti_prior_cr'] = (class_correct[2] + class_correct[3]) / max(1, class_total[2] + class_total[3])
        return metrics

    print('=== HAB_CF_REPORT (ACTUAL world outcomes; trained>reset>stale ordering = emergence) ===', flush=True)
    evalsplit('IN-DIST', idD)
    ht = evalsplit('HELD-OUT', teD)

    # FIX8: verdict
    ht_cr        = ht['trained']['correct_rate']
    hr_cr        = ht['reset']['correct_rate']
    ht_sdr       = ht['stale']['stale_dir_rate']
    anti_prior_cr = ht['trained']['anti_prior_cr']
    if ht_cr >= 0.55 and (ht_cr - hr_cr) >= 0.15 and ht_sdr >= 0.40 and anti_prior_cr >= 0.30:
        verdict = 'EMERGENCE'
    elif ht_cr > hr_cr:
        verdict = 'field biases local preference'
    else:
        verdict = 'no effect'
    print('HAB_CF_VERDICT: %s (held-out trained=%.3f reset=%.3f stale_dir=%.3f anti_prior=%.3f)' % (
        verdict, ht_cr, hr_cr, ht_sdr, anti_prior_cr), flush=True)
    print('=== HAB_CF_DONE ===', flush=True)


if MODE == 'validate': validate()
else: substrate()
print('=== HAB_CF_END ===', flush=True)
