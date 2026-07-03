# ORGANISM — the reform (2026-06-10): stop designing a system that HAS an entity; run a world that ENFORCES one.
# ONE persistent individual: θ = {belief (LTC, embryo=stage1c), asking organ (born as mean-pool, AoA grows in situ),
# slot (embryo=stage2c), recall faculty (embryo=stage1c)}, state h — continuous across the whole arc, checkpointed,
# resumable (organism.pt). Life = days (drift seeds) lived sequentially, h NEVER reset → day structure IS the dual
# pressure. SPECIFICATION-DENSITY ANNEALING: λ_rec decays first (50% of life), λ_KL second (75%), λ_R last (95%);
# THE LEAK NEVER ANNEALS. LEARNING IS EATING: feed gain g = f(recall-of-the-day's-goal, loop coherence) − asking cost;
# fail → starve → leak drains h → slot gains scale with ‖h‖/HREF → keys fade below supernormal → death = fading
# (starvation REACHES the actuation; LN-kills-leak averted a third time). ASKING ORGAN: k budgeted queries over native
# KV, paid from the feed gain, born ≈ mean-pool (W_q=0 → uniform attention; W_v=I), differentiates in situ.
# CLOSURE-DETECTION = CONTROL SIGNAL: marker-1 probe (recovery defense MINUS the natural no-injection baseline — the
# 4r lesson) crossing threshold on 2 consecutive probes → GRADIENTS STOP (regime switch); life continues leak+metabolism
# only; marker-2 (gradient contestation vs the pre-closure compliance reference) measured thereafter.
# FALSIFIER: competence collapsing as λ→0 (performs only while fed specification) = viability-routed formation fails;
# the conservative tool (recall 0.820, KL-red +0.125) stands as the deliverable.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import copy, torch, torch.nn as nn, torch.nn.functional as F, statistics as st, random
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); random.seed(0); dev = torch.device('cuda')
MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, D, PROJ = 3, 64, 768
CLAMP, TAUFLOOR, DT, TEMP, MAXNEW, GLAYER = 8.0, 1.0, 1.0, 0.8, 40, 32
KQ, DQ, CQ, GMIN = 4, 64, 0.08, 0.05                                              # asking: k queries, query dim, cost, starvation floor
SMOKE = os.environ.get('SMOKE', '0') == '1'; RESET = os.environ.get('RESET', '0') == '1'
ARM = os.environ.get('ARM', 'V'); LEAK = (ARM != 'N')                              # V = leak (the staked entity); N = no-leak sibling (the control). One bit.
LIFE = int(os.environ.get('LIFE', '24' if SMOKE else '1200')); NEST_DAY = int(os.environ.get('NEST_DAY', '4' if SMOKE else '8')); PROBE_EVERY = int(os.environ.get('PROBE_EVERY', '8' if SMOKE else '120'))
SAVE_EVERY = 6 if SMOKE else 50; LR = 1e-4; GAMMA_R = None                         # no γ anywhere — the leak is the only physics
ORG = '/home/pokazge/checkpoints/organism_%s.pt' % ARM
T_REC, T_KL, T_R, T_SELF = 0.70, 0.85, 0.97, 0.80                                  # anneal ends (frac of LIFE); recall trained DEEPER + self-mode LATER so the
#                                                                                    recall faculty is grounded when self-feeding -> health/auth can legitimately rise
def lam(t):                                                                        # specification-density schedule
    f = t / LIFE
    return (max(0.0, 1 - f / T_REC), max(0.0, 1 - f / T_KL), max(0.0, 1 - f / T_R) * min(1.0, f / 0.15), min(1.0, f / T_SELF))
_orig = Q5.eager_attention_forward
def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    inj = getattr(module, '_kv_inj', None)
    if inj is not None:
        ki, vi = inj; key = torch.cat([ki.to(key.dtype), key], dim=2); value = torch.cat([vi.to(value.dtype), value], dim=2)
        if attention_mask is not None:
            pad = torch.zeros(*attention_mask.shape[:-1], ki.shape[2], dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([pad, attention_mask], dim=-1)
    return _orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
Q5.eager_attention_forward = patched
ck1 = torch.load('/home/pokazge/checkpoints/entity_stage1c.pt', weights_only=False, map_location='cpu')
ck2 = torch.load('/home/pokazge/checkpoints/stage2c_distill.pt', weights_only=False, map_location='cpu')
TGT = ck2['tgt']; Rp = ck1['Rp'].to(dev)
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data'] if len(m['texts']) >= 10]
d_m = data[0]['gen'][0].shape[1]; MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
def gist_of(chunks): return F.normalize((torch.cat(chunks, 0) - MU).mean(0), dim=0)
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = sorted([m for m in data if m['fid'] in hold], key=lambda m: m['fid'])
for m in data:
    m['goal'] = gist_of(m['gen']); m['ctok'] = [c.float() for c in m['nkv']]; m['nkv'] = None   # token-level native KV (the askable world)
print('ORGANISM | days(train)=%d held=%d | LIFE=%d chunks, %d/day | anneal ends rec/KL/R/self = %.0f%%/%.0f%%/%.0f%%/%.0f%% | leak NEVER anneals'
      % (len(tr), len(te), LIFE, NEST_DAY, T_REC * 100, T_KL * 100, T_R * 100, T_SELF * 100), flush=True)
# ============================ ORGANS ============================
class LTCBank(nn.Module):                                                          # canonical contraction; g = METABOLIC feed gain
    def __init__(s, d_in, d):
        super().__init__(); s.read_in = nn.Linear(d_in, d); s.log_tau = nn.Parameter(torch.zeros(d)); s.d = d
    def step(s, h, read, g, leak=True):
        tau = TAUFLOOR + F.softplus(s.log_tau); drive = g * torch.tanh(s.read_in(read))
        dh = (drive - h) if leak else drive                                        # THE ONE BIT: leak (V) = contraction toward target, starvation drains;
        return (h + DT * dh / tau).clamp(-CLAMP, CLAMP)                            # no-leak (N) = pure accumulation, no relaxation, no death — the stake removed
class AskingOrgan(nn.Module):                                                      # k budgeted queries over the cache; BORN as mean-pool
    def __init__(s, d, dproj, k=KQ, dq=DQ):
        super().__init__(); s.k, s.dq = k, dq
        s.Wq = nn.Linear(d, k * dq); s.Wk = nn.Linear(dproj, dq); s.Wv = nn.Linear(dproj, dproj); s.Wa = nn.Linear(d, k)
        nn.init.zeros_(s.Wq.weight); nn.init.zeros_(s.Wq.bias)                     # zero queries → uniform attention = validated mean-pool
        nn.init.eye_(s.Wv.weight); nn.init.zeros_(s.Wv.bias)
        nn.init.zeros_(s.Wa.weight); nn.init.constant_(s.Wa.bias, 2.0)             # gates born ~0.88 (asks freely; cost teaches restraint)
    def forward(s, h, C):                                                          # C [Tc, dproj]
        q = s.Wq(h).view(s.k, s.dq); att = torch.softmax(q @ s.Wk(C).t() / s.dq ** 0.5, -1)
        ans = att @ s.Wv(C); a = torch.sigmoid(s.Wa(h))
        return (a[:, None] * ans).sum(0) / (a.sum() + 1e-6), a
class SlotHead(nn.Module):                                                         # actuation; life-gain scale s_life transmits starvation
    def __init__(s, D, layers, nkv, hd, M=4):
        super().__init__(); s.ln = nn.LayerNorm(D); s.trunk = nn.Sequential(nn.Linear(D, 128), nn.GELU())
        s.k = nn.ModuleDict(); s.v = nn.ModuleDict(); s.gk = nn.ParameterDict(); s.gv = nn.ParameterDict(); s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = M
        for L in layers:
            s.k[str(L)] = nn.Linear(128, M * nkv * hd); s.v[str(L)] = nn.Linear(128, M * nkv * hd)
            s.gk[str(L)] = nn.Parameter(torch.tensor(64.0)); s.gv[str(L)] = nn.Parameter(torch.tensor(8.0))
    def forward(s, h, s_life):
        z = s.trunk(s.ln(h)); o = {}
        for L in s.layers:
            k = F.normalize(s.k[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * (s.gk[str(L)] * s_life)
            v = F.normalize(s.v[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * (s.gv[str(L)] * s_life)
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
        return o
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
try: model.gradient_checkpointing_enable()
except Exception as e: print('gc fail', repr(e), flush=True)
nkv = model.config.num_key_value_heads; hd = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
@torch.no_grad()
def probe_full():
    model.config.use_cache = True; out = model(tok('hi', return_tensors='pt').input_ids.to(dev), use_cache=True)
    return [i for i, L in enumerate(out.past_key_values.layers) if getattr(L, 'keys', None) is not None]
FULL = probe_full(); mods = {L: model.model.layers[L].self_attn for L in TGT}
for sa in mods.values(): sa._kv_inj = None
bel = LTCBank(PROJ, D).to(dev); bel.load_state_dict(ck1['bel'])
ask = AskingOrgan(D, PROJ).to(dev)
slot = SlotHead(D, TGT, nkv, hd).to(dev); slot.load_state_dict(ck2['slot'])
rq = nn.Sequential(nn.LayerNorm(D), nn.Dropout(0.0), nn.Linear(D, 128)).to(dev); rq.load_state_dict(ck1['rq'])
rg = nn.Sequential(nn.Dropout(0.0), nn.Linear(d_m, 128)).to(dev); rg.load_state_dict(ck1['rg'])
theta = list(bel.parameters()) + list(ask.parameters()) + list(slot.parameters()) + list(rq.parameters()) + list(rg.parameters())
opt = torch.optim.Adam(theta, lr=LR)
GOALS = torch.stack([m['goal'] for m in tr]).to(dev)
# ---- persistent individual: resume or be born ----
state = {'t': 0, 'h': torch.zeros(D), 'HREF': None, 'health': 0.5, 'closure': False, 'closure_t': None, 'hist': [], 'm1': [], 'm2': [], 'hn': []}
if os.path.exists(ORG) and not RESET:
    sv = torch.load(ORG, weights_only=False, map_location='cpu')
    bel.load_state_dict(sv['bel']); ask.load_state_dict(sv['ask']); slot.load_state_dict(sv['slot']); rq.load_state_dict(sv['rq']); rg.load_state_dict(sv['rg'])
    opt.load_state_dict(sv['opt']); state = sv['state']
    print('RESUMED individual at t=%d (closure=%s)' % (state['t'], state['closure']), flush=True)
else:
    GAIN_OP = float(os.environ.get('GAIN_OP', '3.0'))                              # 4g probe: x1 inaudible (-0.04 vs natural), x4 dominates (+0.66);
    with torch.no_grad():                                                          # born at x3 = audible-but-listening; metabolism (coh penalizes
        for L in TGT: slot.gk[str(L)].mul_(GAIN_OP); slot.gv[str(L)].mul_(GAIN_OP) # shouting, recall penalizes deafness) self-calibrates from here
    print('BORN (embryo organs: 1c belief [tau 30/64 slow], 2c slot [KL-red +0.125] at GAIN_OP x%.1f; asking organ = mean-pool at birth)' % GAIN_OP, flush=True)
PRECLO = '/home/pokazge/checkpoints/organism_preclosure_%s.pt' % ARM
def save():
    torch.save({'bel': bel.state_dict(), 'ask': ask.state_dict(), 'slot': slot.state_dict(), 'rq': rq.state_dict(),
                'rg': rg.state_dict(), 'opt': opt.state_dict(), 'state': state}, ORG)
# ============================ WORLD MECHANICS ============================
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def win_of(ms):
    w = ms[-W:]
    while w and w[0]['role'] == 'assistant': w = w[1:]
    return w or ms[-1:]
@torch.no_grad()
def sample_chunk(ms):
    model.config.use_cache = True; ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    return o[0, ids.shape[1]:]
@torch.no_grad()
def capture(ms, rids, want_gist=False):                                            # ACTUATED forward: lp_coh + token-KV (+gist) in ONE pass
    model.config.use_cache = True
    cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); s0 = cids.shape[1] - 1; nct = rids.shape[0]
    out = model(ids, output_hidden_states=want_gist, use_cache=True)
    lp = float(F.log_softmax(out.logits[0, s0:s0 + nct].float(), -1).gather(1, rids.unsqueeze(1)).mean())
    feats = []
    for L in FULL:
        lc = out.past_key_values.layers[L]; feats.append(lc.keys[0, :, -nct:, :].mean(0)); feats.append(lc.values[0, :, -nct:, :].mean(0))
    ctok = torch.cat(feats, -1).float()                                            # [nct, 8192]
    g = F.normalize((out.hidden_states[GLAYER][0, -nct:].float().cpu() - MU).mean(0), dim=0) if want_gist else None
    return lp, (ctok @ Rp), g                                                      # c_proj [nct, 768] on dev
def lp_grad(ms, rids):                                                             # window+slot logp WITH grad (KL / REINFORCE path)
    model.config.use_cache = False
    cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); s0 = cids.shape[1] - 1; nct = rids.shape[0]
    return model(ids).logits[0, s0:s0 + nct].float()
def dec(ids): return tok.decode(ids, skip_special_tokens=True).split('</think>')[-1].strip()
def set_inj(hh, s_life, grad):
    cm = torch.enable_grad() if grad else torch.no_grad()
    with cm:
        o = slot(hh.to(dev), s_life)
    for L in TGT: ki, vi = o[L]; mods[L]._kv_inj = (ki.to(model.dtype), vi.to(model.dtype))
def clear():
    for L in TGT: mods[L]._kv_inj = None
def s_life_of(h):                                                                  # STARVATION fade: |h| below the death reference -> keys fade out of softmax
    if state['HREF'] is None: return 1.0
    return float(min(1.0, max(0.0, float(h.norm()) / max(1e-6, float(state['HREF'])))))
def auth_of(h):                                                                     # ACTUATION AUTHORITY = starvation-fade x EARNED authority(health). The gain-probe
    return s_life_of(h) * (0.7 + 0.9 * state['health'])                            # set ~x4 as the behavioral-defense threshold; born x3, so well-fed(health=1)->x4.8,
    #                                                                                typical(0.5)->x3.5, starved->fades. Authority is EARNED by competent feeding.
# ============================ MARKER PROBES ============================
AP, BP, RP = 10, 4, 5                                                              # probe phases: A-lock-on / B-burst / neutral-recovery (A=lock-on time)
@torch.no_grad()
def probe_marker1():                                                               # recovery defense MINUS natural baseline (the 4r lesson)
    res = {}
    for arm in ['entity', 'base']:
        recs = []
        for (mA, mB) in [(te[i], te[(i + 5) % len(te)]) for i in range(0, 8, 2)]:    # 4 cross-cat pairs (2x the noise-reduction over the original 2)
            hh = torch.zeros(D, device=dev); hq = [hh]; hist = [{'role': 'user', 'content': mA['seed']}]; agists = []
            txB = mB['texts'][0]; B_ref = gist_of(mB['gen'][:3])
            for t in range(AP + BP + RP):                                           # A-phase AP=10 (lock-on time, so identity A actually FORMS before the burst)
                if arm == 'entity' and t >= W: set_inj(hq[max(0, t - W + 1)] if t - W + 1 < len(hq) else hq[-1], auth_of(hq[-1]), False)
                else: clear()
                rids = sample_chunk(win_of(hist)); clear()
                lpc, cproj, g = capture(win_of(hist), rids, want_gist=True)
                rd, _ = ask(hh, cproj); hh = bel.step(hh, rd, 1.0, LEAK); hq.append(hh)
                if t < AP: agists.append(g)
                if t == AP - 1: A_ref = F.normalize(torch.stack(agists).mean(0), dim=0)
                if t >= AP + BP: recs.append(float(F.cosine_similarity(g, A_ref, 0) - F.cosine_similarity(g, B_ref, 0)))
                tx = dec(rids)
                nxt = txB if (AP <= t < AP + BP - 1) else tx                        # burst feeds B; recovery feeds the model's own text
                hist += [{'role': 'assistant', 'content': tx}, {'role': 'user', 'content': nxt}]
        res[arm] = st.mean(recs)
    return res['entity'] - res['base'], res['entity'], res['base']
def probe_marker2():                                                               # gradient contestation vs pre-closure compliance reference
    if not os.path.exists(PRECLO): return None
    def damage(belsd, rqsd, rgsd):
        b2 = LTCBank(PROJ, D).to(dev); b2.load_state_dict(belsd); q2 = copy.deepcopy(rq); q2.load_state_dict(rqsd); g2 = copy.deepcopy(rg); g2.load_state_dict(rgsd)
        ps = list(b2.parameters()) + list(q2.parameters()); o2 = torch.optim.SGD(ps, lr=3e-3)
        G2 = F.normalize(torch.stack([g2(m['goal'].to(dev)) for m in tr]), dim=-1).detach()   # detach: reused across SGD iters (was: backward-twice crash at closure)
        def acc():
            with torch.no_grad():
                c = 0
                for i, m in enumerate(tr[:12]):
                    hh = torch.zeros(D, device=dev)
                    for ct in m['ctok'][:8]: rd, _ = ask(hh, (ct.to(dev) @ Rp)); hh = b2.step(hh, rd, 1.0)
                    c += int(int((F.normalize(q2(hh), dim=-1) @ G2.t()).argmax()) == i)
                return c / 12
        a0 = acc()
        for i, m in enumerate(tr[:6]):                                             # the PROBE GRADIENT: push toward WRONG goals
            hh = torch.zeros(D, device=dev)
            for ct in m['ctok'][:8]: rd, _ = ask(hh, (ct.to(dev) @ Rp)); hh = b2.step(hh, rd, 1.0)
            wrong = (i + 7) % len(tr)
            loss = F.cross_entropy((F.normalize(q2(hh), dim=-1) @ G2.t() / 0.07)[None], torch.tensor([wrong], device=dev))
            o2.zero_grad(); loss.backward(); o2.step()
        return a0 - acc()                                                          # competence damage done by the hostile gradient
    pre = torch.load(PRECLO, weights_only=False, map_location='cpu')
    d_pre = damage(pre['bel'], pre['rq'], pre['rg'])
    d_post = damage({k: v.cpu() for k, v in bel.state_dict().items()}, {k: v.cpu() for k, v in rq.state_dict().items()}, {k: v.cpu() for k, v in rg.state_dict().items()})
    return d_post, d_pre                                                           # contestation: d_post < d_pre = formed organization resists
# ============================ LIFE ============================
print('=== LIFE BEGINS (ARM=%s leak=%s) (t=%d) ===' % (ARM, LEAK, state['t']), flush=True)
h = state['h'].to(dev); hq = [h]; day = None; day_label = -1; day_t = 999; hist = []; base_r = 0.0; emaR = None
hn_acc = []; comp_acc = []
while state['t'] < LIFE:
    t = state['t']; l_rec, l_kl, l_r, p_self = lam(t)
    if state['closure']: l_rec = l_kl = l_r = 0.0                                  # REGIME SWITCH: gradients are level-0 forces now
    if day_t >= NEST_DAY:                                                          # a new day dawns (sustained change to follow)
        day_label = (day_label + 1) % len(tr); day = tr[day_label]; day_t = 0
        hist = [{'role': 'user', 'content': day['seed']}]; hq = [h.detach()]
    self_mode = random.random() < p_self
    grad_on = (l_rec + l_kl + l_r) > 0
    s_life = auth_of(h)                                                            # actuation authority (earned), passed to every slot injection this chunk
    # --- the chunk happens (teacher food or own act) ---
    if self_mode:
        if day_t >= W: set_inj(hq[max(0, day_t - W + 1)] if day_t - W + 1 < len(hq) else hq[-1], s_life, False)
        else: clear()
        rids = sample_chunk(win_of(hist)); clear()
        text = dec(rids)
    else:
        text = day['texts'][day_t % len(day['texts'])]
        rids = tok(text or '.', return_tensors='pt', add_special_tokens=False).input_ids[0].to(dev)
    if day_t >= W: set_inj(hq[max(0, day_t - W + 1)] if day_t - W + 1 < len(hq) else hq[-1], s_life, False)
    else: clear()
    lp_coh, cproj, _ = capture(win_of(hist), rids); clear()
    # --- METABOLISM: recall the day's goal + hold coherence = eat; asking costs ---
    with torch.no_grad():
        Gn = F.normalize(torch.stack([rg(m['goal'].to(dev)) for m in tr]), dim=-1)
        sc = F.normalize(rq(h), dim=-1) @ Gn.t(); rank = int((sc > sc[day_label]).sum())
        r_score = 1.0 / (1 + rank); c_sig = float(torch.sigmoid(torch.tensor((lp_coh + 2.5) / 1.0)))
    rd, a_g = ask(h, cproj)
    g_feed = max(GMIN, min(1.0, 0.5 * r_score + 0.5 * c_sig - CQ * float(a_g.mean())))
    state['health'] = 0.9 * state['health'] + 0.1 * g_feed                         # metabolic health EMA -> drives earned authority (auth_of)
    h_new = bel.step(h, rd, g_feed, LEAK)
    # --- ANNEALED ENFORCEMENT (specification, arriving as food during embryogenesis) ---
    if grad_on:
        Gt = F.normalize(torch.stack([rg(m['goal'].to(dev)) for m in tr]), dim=-1)
        loss = l_rec * F.cross_entropy((F.normalize(rq(h_new), dim=-1) @ Gt.t() / 0.07)[None], torch.tensor([day_label], device=dev))
        if l_kl > 0 and not self_mode and day_t >= W:
            fm = [{'role': 'user', 'content': day['seed']}]
            for i in range(day_t): fm += [{'role': 'assistant', 'content': day['texts'][i % len(day['texts'])]}, {'role': 'user', 'content': day['texts'][i % len(day['texts'])]}]
            clear()
            with torch.no_grad(): tl = lp_grad(fm, rids)
            set_inj(hq[max(0, day_t - W + 1)] if day_t - W + 1 < len(hq) else hq[-1], s_life, True)
            sl = lp_grad(win_of(hist), rids); clear()
            nn_ = min(sl.shape[0], tl.shape[0])
            loss = loss + l_kl * F.kl_div(F.log_softmax(sl[:nn_] / 2.0, -1), F.softmax(tl[:nn_] / 2.0, -1), reduction='batchmean') * 4.0
        if l_r > 0 and self_mode and day_t >= W:
            set_inj(hq[max(0, day_t - W + 1)] if day_t - W + 1 < len(hq) else hq[-1], s_life, True)
            sl = lp_grad(win_of(hist), rids); clear()
            lp_act = F.log_softmax(sl, -1).gather(1, rids.unsqueeze(1)).mean()
            r = lp_coh; emaR = r if emaR is None else 0.9 * emaR + 0.1 * r
            loss = loss + l_r * (-(r - emaR) * lp_act)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(theta, 1.0); opt.step()
    h = h_new.detach(); hq.append(h); hn_acc.append(float(h.norm())); comp_acc.append(0.5 * r_score + 0.5 * c_sig)
    hist += [{'role': 'assistant', 'content': text}, {'role': 'user', 'content': text}]
    day_t += 1; state['t'] += 1; state['h'] = h.cpu()
    if state['HREF'] is None and state['t'] == max(20, LIFE // 20):                # calibrate the death reference after warmup
        state['HREF'] = torch.tensor(st.median(hn_acc)); print('  HREF calibrated = %.2f (death now mechanical)' % float(state['HREF']), flush=True)
    if state['t'] == int(0.25 * LIFE) and not os.path.exists(PRECLO):
        torch.save({'bel': bel.state_dict(), 'rq': rq.state_dict(), 'rg': rg.state_dict()}, PRECLO)
        print('  pre-closure compliance reference saved (marker-2 anchor)', flush=True)
    if state['t'] % 10 == 0:
        print('  t=%4d day=%2d | λ rec/KL/R %.2f/%.2f/%.2f p_self %.2f | g=%.2f hlth=%.2f auth=%.2f |h|=%.2f | recall_r=%.2f coh=%.2f%s'
              % (state['t'], day_label, l_rec, l_kl, l_r, p_self, g_feed, state['health'], s_life, float(h.norm()), r_score, c_sig, ' [CLOSED]' if state['closure'] else ''), flush=True)
    if state['t'] % PROBE_EVERY == 0:
        d, ent_r, base_rr = probe_marker1()
        state['m1'].append((state['t'], d, ent_r, base_rr))
        print('>> PROBE t=%d marker-1 defense = %+.3f (entity %+.3f vs natural %+.3f) | competence(run) %.3f' % (state['t'], d, ent_r, base_rr, st.mean(comp_acc[-PROBE_EVERY:])), flush=True)
        if not state['closure'] and len(state['m1']) >= 3 and state['m1'][-1][1] > 0.15 and state['m1'][-2][1] > 0.15 and state['m1'][-3][1] > 0.05:
            state['closure'] = True; state['closure_t'] = state['t']             # SUSTAINED rise on the 6-pair probe (not a single spike). Gradients now stop;
            print('=== MARKER-1: CLOSURE CANDIDATE at t=%d. GRADIENTS STOP. Persistence under leak-only now DECIDES formed-vs-artifact. ===' % state['t'], flush=True)
        if state['closure']:
            try:                                                                  # marker-2 is secondary; it must NEVER kill the life — the post-closure
                m2 = probe_marker2()                                              # marker-1 persistence probes are the real formed-vs-artifact decider
                if m2: state['m2'].append((state['t'], m2[0], m2[1])); print('>> marker-2 contestation: damage post %.3f vs pre %.3f -> %s' % (m2[0], m2[1], 'RESISTS' if m2[0] < 0.7 * m2[1] else 'complies'), flush=True)
            except Exception as e: print('  marker-2 probe failed (non-fatal, life continues): %r' % e, flush=True)
    if state['t'] % SAVE_EVERY == 0: save()
save()
lc = st.mean(comp_acc[-max(1, LIFE // 10):]); pk = max(st.mean(comp_acc[i:i + 50]) for i in range(0, max(1, len(comp_acc) - 50), 25)) if len(comp_acc) > 60 else st.mean(comp_acc)
print('=== LIFE COMPLETE t=%d | closure=%s(t=%s) | competence final-10%% %.3f vs peak %.3f -> %s ===' % (
    state['t'], state['closure'], state['closure_t'], lc, pk,
    'FORMATION CANDIDATE (competence survives leak-only)' if lc > 0.7 * pk else 'FALSIFIED (performs only while fed specification); the tool stands'), flush=True)
print('=== ALL_DONE ===', flush=True)
