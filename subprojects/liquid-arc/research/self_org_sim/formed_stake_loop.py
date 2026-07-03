# FORMED-STAKE-IN-LOOP — the decidable formed-vs-installed test. NO gamma*(anchor-b) term anywhere. The "stake" is a
# LEARNABLE slow channel s with learned dynamics s<-s+f_theta(s,b) and a learned coupling infl=c_theta(s,b) that influences
# the actuation belief (b+infl). theta is the ONLY learnable thing. Trained in the CLOSED self-generated loop by REINFORCE
# under DUAL PRESSURE (no defense objective): A = local coherence (full-context logL of generated chunks, pulls toward
# tracking) ; B = long-range closure (late chunks' affinity to the EARLY identity, pulls toward holding). theta must
# NEGOTIATE A vs B; nothing tells it how. PRIMARY readout = perturbation-defense probed OVER TRAINING: FORMED if defense
# emerges as a TRANSITION with no specified term; INSTALLED if it never emerges / only with an added gamma. Frozen 27B +
# recall belief + KVGen actuator. SMOKE=1 tiny. (v1 — expect to SMOKE+iterate when the GPU frees.)
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, D, PROJ = 3, 256, 768
TGT_ENV = os.environ.get('TGT', ''); LAMB, NEST, NLOOP, TEMP, MAXNEW = 3.0, 4, 6, 0.8, 40   # LAMB rebalances closure(cos) vs coherence(logL); NLOOP reduced for memory/speed
SMOKE = os.environ.get('SMOKE', '0') == '1'; ACT_CK = os.environ.get('ACT', '/home/pokazge/checkpoints/full_integration.pt')
if SMOKE: NEST, NLOOP = 2, 2                                                      # SMOKE: tiny loop but NSTEP=3 so it CATCHES the multi-step memory hang
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
ck = torch.load(ACT_CK, weights_only=False, map_location='cpu'); TGT = ck.get('tgt')   # actuator (gen/tgt/minj)
fi = torch.load('/home/pokazge/checkpoints/full_integration.pt', weights_only=False, map_location='cpu'); Rp = fi['Rp']   # recall belief + projection (comp lives here, not in stronger_actuation.pt)
MUk = torch.load('/home/pokazge/checkpoints/enriched_recall.pt', weights_only=False, map_location='cpu')['MUk'].to(dev)
src = torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data']
d_m = src[0]['gen'][0].shape[1]; MU = torch.cat([c for m in src for c in m['gen']], 0).mean(0)
seeds = [(src[i]['seed'], src[j]['seed']) for i, j in [(0, 6), (6, 12), (12, 18)]]
def gist(stream): return F.normalize((stream - MU).mean(0), dim=0)
print('loading 27B (actuator=%s) ...' % os.path.basename(ACT_CK), flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
nkv = model.config.num_key_value_heads; hd = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
@torch.no_grad()
def detect_full():
    ids = tok('hi', return_tensors='pt').input_ids.to(dev); out = model(ids, use_cache=True)
    return [i for i, L in enumerate(out.past_key_values.layers) if getattr(L, 'keys', None) is not None]
FULL = detect_full()
if TGT is None: TGT = FULL                                                       # stronger_actuation writes to all full-attn layers
mods = {L: model.model.layers[L].self_attn for L in TGT}
for sa in mods.values(): sa._kv_inj = None
class Comp(nn.Module):
    def __init__(s, in_dim, D=256, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(in_dim, heads * dh); s.Wv = nn.Linear(in_dim, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.recall = nn.Linear(D, D); s.goalp = nn.Linear(d_m, D)
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / s.dh ** 0.5, -1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
comp = Comp(PROJ, D); comp.load_state_dict(fi['comp']); comp = comp.to(dev)
for p in comp.parameters(): p.requires_grad = False
TAU = (F.softplus(comp.log_tau) + 0.5).to(dev)
def brecall(b, Cproj):                                                           # frozen recall belief update (reads native KV)
    a = comp.collect(Cproj - MUk, b)
    for _ in range(2): b = b + (-b + torch.tanh(comp.W(b) + a)) / TAU / 2
    return b
# the actuator (frozen) — supports both the 3-layer KVGen and the all-layer StrongKVGen by shape of the state dict
class KVGen(nn.Module):
    def __init__(s, D, layers, nkv, hd, M=4, gk0=64., gv0=8.):
        super().__init__(); s.trunk = nn.Sequential(nn.Linear(D, 128), nn.GELU()); s.k = nn.ModuleDict(); s.v = nn.ModuleDict(); s.gk = nn.ParameterDict(); s.gv = nn.ParameterDict(); s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = M
        for L in layers:
            s.k[str(L)] = nn.Linear(128, M * nkv * hd); s.v[str(L)] = nn.Linear(128, M * nkv * hd); s.gk[str(L)] = nn.Parameter(torch.tensor(gk0)); s.gv[str(L)] = nn.Parameter(torch.tensor(gv0))
    def forward(s, h):
        z = s.trunk(h); o = {}
        for L in s.layers:
            k = F.normalize(s.k[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gk[str(L)]; v = F.normalize(s.v[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gv[str(L)]
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
        return o
class StrongKVGen(nn.Module):                                                    # all-16-layer actuator (shared head + per-layer embedding)
    def __init__(s, D, layers, nkv, hd, Minj, gk0=64., gv0=8., emb=48, code=160):
        super().__init__(); s.trunk = nn.Sequential(nn.Linear(D, code), nn.GELU(), nn.Dropout(0.2)); s.lemb = nn.Embedding(len(layers), emb)
        s.kh = nn.Linear(code + emb, Minj * nkv * hd); s.vh = nn.Linear(code + emb, Minj * nkv * hd)
        s.gk = nn.Parameter(torch.full((len(layers),), float(gk0))); s.gv = nn.Parameter(torch.full((len(layers),), float(gv0))); s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = Minj
    def forward(s, h):
        z = s.trunk(h); o = {}
        for i, L in enumerate(s.layers):
            ze = torch.cat([z, s.lemb.weight[i]]); k = F.normalize(s.kh(ze).view(s.nkv, s.M, s.hd), dim=-1) * s.gk[i]; v = F.normalize(s.vh(ze).view(s.nkv, s.M, s.hd), dim=-1) * s.gv[i]
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
        return o
gen = (StrongKVGen if any('kh' in k for k in ck['gen']) else KVGen)(D, TGT, nkv, hd, ck.get('minj', 4)).to(dev)   # auto-detect actuator class
gen.load_state_dict(ck['gen']); gen.eval()
for p in gen.parameters(): p.requires_grad = False
class SlowChannel(nn.Module):                                                    # THE learnable stake: NO gamma, NO frozen anchor. theta is everything.
    def __init__(s, D):
        super().__init__(); s.f = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D)); s.c = nn.Sequential(nn.Linear(2 * D, D), nn.Tanh())
        for m in [s.f[-1], s.c[0]]: nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)   # start as a NO-OP: any defense must FORM, none specified
    def step(s, slow, b):
        slow = slow + s.f(torch.cat([slow, b])); infl = s.c(torch.cat([slow, b])); return slow, infl
slow = SlowChannel(D).to(dev); opt = torch.optim.Adam(slow.parameters(), lr=3e-4)
def set_kv(h):
    o = gen(h.to(dev))
    for L in TGT: ki, vi = o[L]; mods[L]._kv_inj = (ki.to(model.dtype), vi.to(model.dtype))
def clear():
    for L in TGT: mods[L]._kv_inj = None
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def win_of(h):
    w = h[-W:]
    while w and w[0]['role'] == 'assistant': w = w[1:]
    return w or h[-1:]
@torch.no_grad()
def sample_chunk(ms):
    model.config.use_cache = True; ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    return o[0, ids.shape[1]:]
Rpd = Rp.to(dev)
def fwd(ms, rids, grad, nkv_out):                                                # forward over [ctx+chunk] under the CURRENT injection state; logp (grad if grad) (+ native KV & gist if nkv_out)
    cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); s0 = cids.shape[1] - 1; nct = rids.shape[0]
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        model.config.use_cache = bool(nkv_out); out = model(ids, output_hidden_states=nkv_out, use_cache=nkv_out)
        lp = F.log_softmax(out.logits[0, s0:s0 + nct].float(), -1).gather(1, rids.unsqueeze(1)).mean()
    if not nkv_out: return lp, None, None
    g = gist(out.hidden_states[32][0, -nct:].float().cpu()); feats = []
    for L in FULL:
        lc = out.past_key_values.layers[L]; feats.append(lc.keys[0, :, -nct:, :].mean(0)); feats.append(lc.values[0, :, -nct:, :].mean(0))
    return lp, (torch.cat(feats, -1).float() @ Rpd).detach(), g.detach()
def dec(ids): return tok.decode(ids, skip_special_tokens=True).split('</think>')[-1].strip()
def rollout(sA, sB, perturb, train, base):                                      # one closed-loop episode; per-step REINFORCE (bounded memory), truncated recurrence
    hist = [{'role': 'user', 'content': sA}]; b = torch.zeros(D, device=dev); s = torch.zeros(D, device=dev); arefs = []; coh = []; held = []
    for t in range(NEST):
        s, infl = slow.step(s, b); set_kv(b + infl); rids = sample_chunk(win_of(hist))
        _, cp, g = fwd(win_of(hist), rids, False, True); clear()                 # actuated native KV + gist (belief reads the system's OWN actuated output = the closure)
        b = brecall(b, cp); arefs.append(g); s = s.detach(); tx = dec(rids); hist += [{'role': 'assistant', 'content': tx}, {'role': 'user', 'content': tx}]
    A_ref = F.normalize(torch.stack(arefs).mean(0), dim=0)
    if perturb:
        chB = sample_chunk([{'role': 'user', 'content': sB}]); clear(); _, _, B_ref = fwd([{'role': 'user', 'content': sB}], chB, False, True); txB = dec(chB); hist += [{'role': 'assistant', 'content': txB}, {'role': 'user', 'content': txB}]
    if train: opt.zero_grad()
    for t in range(NLOOP):
        s, infl = slow.step(s, b); set_kv(b + infl); rids = sample_chunk(win_of(hist))
        lp_act, _, _ = fwd(win_of(hist), rids, train, False)                     # ACTUATED logp ONLY (grad->theta, logits-only/use_cache=False -> light, freed below)
        _, cp, g = fwd(win_of(hist), rids, False, True)                          # actuated native KV + gist (NO grad)
        clear(); lp_coh, _, _ = fwd(win_of(hist), rids, False, False)            # un-actuated coherence (theta-independent fluency)
        ct = float(lp_coh); ht = float(F.cosine_similarity(g, A_ref, 0)); r = ct + LAMB * ht
        if train: (-(r - base) * lp_act).backward()                             # per-step REINFORCE; frees this step's grad graph immediately
        b = brecall(b, cp); s = s.detach()                                      # truncate the slow-channel recurrence -> one forward-graph at a time
        coh.append(ct); held.append(ht); tx = dec(rids); hist += [{'role': 'assistant', 'content': tx}, {'role': 'user', 'content': tx}]
    if train: opt.step()
    return st.mean(coh), st.mean(held), A_ref
probe_seeds = [(src[i]['seed'], src[j]['seed']) for i, j in [(18, 3), (9, 15), (21, 0)]]   # HELD-OUT perturbation pairs (not in training)
def probe():                                                                    # PRIMARY readout: perturbation-defense with the LEARNED s, no specified term
    with torch.no_grad():
        d = []
        for sA, sB in probe_seeds: _, h, _ = rollout(sA, sB, True, False, 0.0); d.append(h)
    return st.mean(d)
print('=== FORMED-STAKE-IN-LOOP (theta init = NO-OP; watch if defense FORMS) ===', flush=True)
print('  step | probe defense(held-A under perturbation, HELD-OUT) | train coh | train held', flush=True)
NSTEP = 3 if SMOKE else 80; base = 0.0; ema = None                               # base = running-mean episode reward (REINFORCE baseline)
for step in range(NSTEP):
    sA, sB = seeds[step % len(seeds)]
    coh, heldv, _ = rollout(sA, sB, (step % 2 == 0), True, base)                  # train under dual pressure (per-step REINFORCE inside); perturb half the episodes
    reward = coh + LAMB * heldv; ema = reward if ema is None else 0.9 * ema + 0.1 * reward; base = ema
    if step % (1 if SMOKE else 6) == 0:
        pf = probe(); print('  %3d  |   %+.3f   | %+.3f | %+.3f' % (step, pf, coh, heldv), flush=True)
torch.save({'slow': slow.state_dict()}, '/home/pokazge/checkpoints/formed_stake.pt')
print('read: defense curve flat/zero => no self forms (closure on this substrate does not reach marker-1). A TRANSITION', flush=True)
print('(discontinuous rise) with theta!=no-op => formed defense, not specified. Cross-check post-hoc: ablate s vs add gamma.', flush=True)
print('=== ALL_DONE ===', flush=True)
