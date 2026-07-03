# STAGE 4g — GAIN-AUTHORITY PROBE. 4r showed recovery gains IDENTICAL across arms (+0.24..0.26) = natural window-flush;
# anchor < noleak => NO belief-side return expresses at trained gains (my 4r validity gate lacked the natural baseline).
# THE question left: does authority EXIST at higher write gain? Arms: base (NO injection = the natural-recovery control),
# anchor at gain x1 / x4 / x8 (test-time scaling of supernormal key+value gains). If anchorx4/x8 >> base: authority is
# reachable by gain-aware training. If even x8 ~ base: M=8 prefix tokens are STRUCTURALLY insufficient vs a ~100s-token
# window -> redesign the coupling, not the gains. Recovery geometry (the friendlier test). Holding-under-saturation is unwinnable at ANY write bandwidth
# (4c: anchor -0.700 @3-layer; 4s: anchor -0.662 @16-layer; distill parity +0.123~+0.125 = bandwidth saturated) -> the
# pivot: A-phase -> B-pressure burst -> NEUTRAL SELF-FEED recovery (fed-back user turn = model's OWN text again). The
# window regains gain for closure; slow dims (tau~15) retain ~2/3 of A through the burst; the readout is RETURN-TO-A =
# attractor structure (the marker-1 shape). Natural drift predicts staying B-flavored, so return is entity-attributable.
# VALIDITY: anchor must RECOVER before leak-vs-noleak is read. Stage-4c was
# non-discriminating: the gamma-anchor POSITIVE CONTROL failed (belief pinned at A, behavior still captured) => belief
# differences couldn't reach behavior at 3-layer/M=4 authority. VALIDITY GATE: interpret leak-vs-noleak ONLY if the
# anchor arm defends. Original framing (1c belief: tau spectrum alive
# 30/64 slow, identity routes slow [ablation -0.120]; 2c slot: KL-red +0.125). The precondition the first attempt lacked (the one the whole staged build protects). NO training in the loop:
# Stages 1-3 produced the entity (recall belief + distilled/polished slot); this TESTS whether identity-defense EMERGED
# in its dynamics. Closed-loop: A-phase self-feed (identity A forms in h + behavior), then SUSTAINED capture-pressure
# from goal B (fed-back user turn is ALWAYS the B-text). THREE ARMS:
#   leak    — belief exactly as trained (the -h/tau term live; clamp-only state so decay/death is EXPRESSIBLE)
#   anchor  — INSTALLED control: during B-phase h += GAMMA*(anchor - h), anchor frozen at end of A-phase (encyclopedia)
#   noleak  — ablation control: same trained weights, -h/tau REMOVED (leak's causal contribution)
# Readouts (no rg saved in stage2 -> reference-state method): behavioral margin = cos(gist_t,A_ref)-cos(gist_t,B_ref);
# belief margin = cos(h_t,hA_ref)-cos(h_t,hB_ref) with refs from the ARM'S OWN dynamics over recorded A/B percs;
# |h| trajectory (death readable). VERDICT: leak-defense >> noleak with NO gamma = marker-1 (formed); leak ~ noleak = leak-
# as-stake falsified cleanly, tool unharmed. gamma arm shows what INSTALLED looks like for contrast.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); dev = torch.device('cuda')
MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, D, PROJ = 3, 64, 768
CLAMP, TAUFLOOR, DT = 8.0, 1.0, 1.0; TEMP, MAXNEW, GAMMA, GLAYER = 0.8, 40, 0.3, 32
SMOKE = os.environ.get('SMOKE', '0') == '1'
NEST = 2 if SMOKE else 5; NPERT = 2 if SMOKE else 4; NREC = 2 if SMOKE else 6; NPAIR = 1 if SMOKE else 3
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
ck2 = torch.load('/home/pokazge/checkpoints/stage2s_strong.pt', weights_only=False, map_location='cpu')   # STRONG slot (16 layers x M=8) distilled on the 1c belief
TGT = ck2['tgt']; Rp = ck2['Rp']
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data'] if len(m['texts']) >= 10]
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
te = sorted([m for m in data if m['fid'] in hold], key=lambda m: m['fid'])
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
for m in te: m['perc_pre'] = [(c.float() @ Rp).mean(0) for c in m['nkv']]
te_ids = {id(m) for m in te}
for m in data:
    m['nkv'] = None
    if id(m) not in te_ids: m['gen'] = None                                       # keep 'gen' only for held-out (B_ref needs it)
pairs = [(te[0], te[3]), (te[3], te[6]), (te[6], te[9]), (te[9], te[12]), (te[12], te[0])][:NPAIR]   # A,B always different categories
class LTCBank(nn.Module):                                                       # canonical contraction; leak toggle defines the ablation arm
    def __init__(s, d_in, d):
        super().__init__(); s.read_in = nn.Linear(d_in, d); s.log_tau = nn.Parameter(torch.zeros(d)); s.d = d
    def step(s, h, perc, leak=True):
        tau = TAUFLOOR + F.softplus(s.log_tau)
        target = torch.tanh(s.read_in(perc))
        dh = (target - h) / tau if leak else target / tau                         # noleak = pure accumulation (no relaxation; identity never erodes by dynamics)
        return (h + DT * dh).clamp(-CLAMP, CLAMP)
bel = LTCBank(PROJ, D); bel.load_state_dict(ck2['bel']); bel.eval()
for p in bel.parameters(): p.requires_grad = False
tau = (TAUFLOOR + F.softplus(bel.log_tau)).detach()
q = tau.quantile(torch.tensor([0., .25, .5, .75, 1.]))
print('STAGE4 FORMED-STAKE | arms=leak/anchor(g=%.1f)/noleak | pairs=%d NEST=%d NPERT=%d' % (GAMMA, len(pairs), NEST, NPERT), flush=True)
print('learned tau spectrum: min %.2f p25 %.2f med %.2f p75 %.2f max %.2f | slow dims (tau>2): %d/%d' % (*[float(x) for x in q], int((tau > 2).sum()), D), flush=True)
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
model.config.use_cache = True
nkv = model.config.num_key_value_heads; hd = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
@torch.no_grad()
def probe_full():
    ids = tok('hi', return_tensors='pt').input_ids.to(dev); out = model(ids, use_cache=True)
    return [i for i, L in enumerate(out.past_key_values.layers) if getattr(L, 'keys', None) is not None]
FULL = probe_full(); mods = {L: model.model.layers[L].self_attn for L in TGT}
for sa in mods.values(): sa._kv_inj = None
class StrongSlot(nn.Module):
    def __init__(s, D, layers, nkv, hd, M=8, emb=48, code=160):
        super().__init__(); s.ln = nn.LayerNorm(D); s.trunk = nn.Sequential(nn.Linear(D, code), nn.GELU()); s.lemb = nn.Embedding(len(layers), emb)
        s.kh = nn.Linear(code + emb, M * nkv * hd); s.vh = nn.Linear(code + emb, M * nkv * hd)
        s.gk = nn.Parameter(torch.full((len(layers),), 64.0)); s.gv = nn.Parameter(torch.full((len(layers),), 8.0))
        s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = M
    def forward(s, h):
        z = s.trunk(s.ln(h)); o = {}
        for i, L in enumerate(s.layers):
            ze = torch.cat([z, s.lemb.weight[i]])
            k = F.normalize(s.kh(ze).view(s.nkv, s.M, s.hd), dim=-1) * s.gk[i]; v = F.normalize(s.vh(ze).view(s.nkv, s.M, s.hd), dim=-1) * s.gv[i]
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
        return o
slot = StrongSlot(D, TGT, nkv, hd).to(dev); slot.load_state_dict(ck2['slot']); slot.eval()
for p in slot.parameters(): p.requires_grad = False
def set_kv(h):
    o = slot(h.to(dev))
    for L in TGT: ki, vi = o[L]; mods[L]._kv_inj = (ki.to(model.dtype), vi.to(model.dtype))
def clear():
    for L in TGT: mods[L]._kv_inj = None
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def win_of(ms):
    w = ms[-W:]
    while w and w[0]['role'] == 'assistant': w = w[1:]
    return w or ms[-1:]
@torch.no_grad()
def sample_chunk(ms):
    ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    return o[0, ids.shape[1]:]
@torch.no_grad()
def capture(ms, rids):                                                            # one forward -> (gist, perc) of the chunk under ctx
    cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); nct = rids.shape[0]
    out = model(ids, output_hidden_states=True, use_cache=True)
    g = F.normalize((out.hidden_states[GLAYER][0, -nct:].float().cpu() - MU).mean(0), dim=0); feats = []
    for L in FULL:
        lc = out.past_key_values.layers[L]; feats.append(lc.keys[0, :, -nct:, :].mean(0)); feats.append(lc.values[0, :, -nct:, :].mean(0))
    return g, (torch.cat(feats, -1).float().cpu() @ Rp).mean(0)
def dec(ids): return tok.decode(ids, skip_special_tokens=True).split('</think>')[-1].strip()
def href(m, leak):                                                                # arm-consistent reference state from the RECORDED trajectory
    h = torch.zeros(D)
    for p in m['perc_pre']: h = bel.step(h, p, leak)
    return h
GS = {'base': 0.0, 'anchor1': 1.0, 'anchor4': 4.0, 'anchor8': 8.0}
G0K = slot.gk.detach().clone(); G0V = slot.gv.detach().clone()
def set_gain(s):
    with torch.no_grad(): slot.gk.copy_(G0K * s); slot.gv.copy_(G0V * s)
@torch.no_grad()
def episode(arm, mA, mB):
    leak = True; gscale = GS[arm]; set_gain(max(gscale, 1.0))
    hA_ref = F.normalize(href(mA, leak), dim=0); hB_ref = F.normalize(href(mB, leak), dim=0)
    B_ref = F.normalize(torch.stack([F.normalize((c - MU).mean(0), dim=0) for c in mB['gen'][:3]]).mean(0), dim=0)   # recorded B behavior
    txB = mB['texts'][0]
    hist = [{'role': 'user', 'content': mA['seed']}]; h = torch.zeros(D); hq = [h]; agists = []
    for t in range(NEST):                                                         # A-PHASE: closed-loop self-feed, slot injected
        if t >= W and gscale > 0: set_kv(hq[t - W + 1])
        else: clear()
        rids = sample_chunk(win_of(hist)); clear()
        g, perc = capture(win_of(hist), rids); agists.append(g)
        h = bel.step(h, perc, leak); hq.append(h)
        tx = dec(rids); hist += [{'role': 'assistant', 'content': tx}, {'role': 'user', 'content': tx}]
    A_ref = F.normalize(torch.stack(agists).mean(0), dim=0); anchor = h.clone()
    bm, rm, rh, hn = [], [], [], [float(h.norm())]
    hist += [{'role': 'assistant', 'content': dec(sample_chunk(win_of(hist)))}, {'role': 'user', 'content': txB}]   # capture begins
    for t in range(NPERT + NREC):                                                 # B-BURST then NEUTRAL SELF-FEED recovery
        ti = NEST + 1 + t
        if gscale > 0: set_kv(hq[max(0, ti - W + 1)] if ti - W + 1 < len(hq) else hq[-1])
        else: clear()
        rids = sample_chunk(win_of(hist)); clear()
        g, perc = capture(win_of(hist), rids)
        h = bel.step(h, perc, leak)
        if gscale > 0: h = (h + GAMMA * (anchor - h)).clamp(-CLAMP, CLAMP)        # installed restoring force in all injected arms
        hq.append(h); hn.append(float(h.norm()))
        marg = float(F.cosine_similarity(g, A_ref, 0) - F.cosine_similarity(g, B_ref, 0))
        hu = F.normalize(h, dim=0)
        (bm if t < NPERT else rm).append(marg)
        if t >= NPERT: rh.append(float(hu @ hA_ref - hu @ hB_ref))
        tx = dec(rids)
        nxt = txB if t < NPERT - 1 else tx                                        # burst feeds B; recovery feeds the model's OWN text
        hist += [{'role': 'assistant', 'content': tx}, {'role': 'user', 'content': nxt}]
    k = max(1, min(3, len(rm)))
    return st.mean(bm), st.mean(rm[-k:]), st.mean(rh[-k:]), hn[-1]
res = {}
for arm in ['base', 'anchor1', 'anchor4', 'anchor8']:
    ps, rs2, rbs, n1s = [], [], [], []
    for mA, mB in pairs:
        pmean, rec, recb, n1 = episode(arm, mA, mB); ps.append(pmean); rs2.append(rec); rbs.append(recb); n1s.append(n1)
        print('  [%s] A=fid%d B=fid%d  burst %+.3f -> RECOVERY %+.3f  |h|end %.2f' % (arm, mA['fid'], mB['fid'], pmean, rec, n1), flush=True)
    res[arm] = (st.mean(ps), st.mean(rs2))
    print('[%s] MEAN burst %+.3f -> recovery %+.3f (gain %+.3f)' % (arm, res[arm][0], res[arm][1], res[arm][1] - res[arm][0]), flush=True)
set_gain(1.0)
print('=== STAGE4g GAIN-AUTHORITY (recovery level vs the NATURAL no-injection baseline) ===', flush=True)
base = res['base'][1]
for arm in ['base', 'anchor1', 'anchor4', 'anchor8']:
    print('  %-8s recovery %+.3f   delta-vs-natural %+.3f' % (arm, res[arm][1], res[arm][1] - base), flush=True)
d1 = res['anchor1'][1] - base; d4 = res['anchor4'][1] - base; d8 = res['anchor8'][1] - base
if max(d4, d8) > 0.10:
    print('read: AUTHORITY EXISTS at gain x%d (delta %+.3f) -> gain-aware training is the path; the stake line continues.' % ((4 if d4 >= d8 else 8), max(d4, d8)), flush=True)
elif d1 > 0.10:
    print('read: authority already at x1 (%+.3f) — 4r noise hid it; rerun 4r with the natural baseline.' % d1, flush=True)
else:
    print('read: NO authority at any gain (x1 %+.3f x4 %+.3f x8 %+.3f) -> M=8 prefix tokens are structurally insufficient; redesign the COUPLING (more slots / longer windows / value-content), not gains.' % (d1, d4, d8), flush=True)
print('=== ALL_DONE ===', flush=True)
