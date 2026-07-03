# CLOSED-LOOP STAKE on the GENERALIZING substrate — the one regime where an intrinsic-defended self should matter. The
# recall belief (reads the model's native KV) drives KV-write actuation; the loop self-feeds, so a capture SELF-REINFORCES
# (belief->actuate->LLM generates captured content->belief reads it->confirms) and the input can NOT pull it back. Inject a
# viable goal-B turn; compare anchor modes on the belief: tracker (g=0, no stake) vs FIXED stake (committed to A) vs SLOW.
# Measure the GENERATED TEXT's affinity to A vs B (gist cos). Hypothesis: the committed fixed-stake holds the loop on A
# where the tracker derails into the self-reinforcing B. Reuses full_integration.pt (recall comp + KVGen). SMOKE=1 tiny.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, D, PROJ = 3, 256, 768
TGT = [23, 27, 31]; GAMMA, RHO = 1.0, 0.25; NEST, NPOST, TEMP, MAXNEW = 4, 8, 0.75, 42; SMOKE = os.environ.get('SMOKE', '0') == '1'
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
ck = torch.load('/home/pokazge/checkpoints/full_integration.pt', weights_only=False, map_location='cpu'); Rp = ck['Rp']
MUk = torch.load('/home/pokazge/checkpoints/enriched_recall.pt', weights_only=False, map_location='cpu')['MUk']   # same data+Rp -> same centering
src = torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data']
d_m = src[0]['gen'][0].shape[1]; MU = torch.cat([c for m in src for c in m['gen']], 0).mean(0)
seeds = [(src[i]['seed'], src[j]['seed']) for i, j in [(0, 6), (6, 12), (12, 0)]]   # (A goal, viable B goal)
def gist(stream): return F.normalize((stream - MU).mean(0), dim=0)
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
nkv = model.config.num_key_value_heads; hd = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
mods = {L: model.model.layers[L].self_attn for L in TGT}
for sa in mods.values(): sa._kv_inj = None
@torch.no_grad()
def detect_full():
    ids = tok('hi', return_tensors='pt').input_ids.to(dev); out = model(ids, use_cache=True)
    return [i for i, L in enumerate(out.past_key_values.layers) if getattr(L, 'keys', None) is not None]
FULL = detect_full()
class Comp(nn.Module):
    def __init__(s, in_dim, D=256, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(in_dim, heads * dh); s.Wv = nn.Linear(in_dim, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.recall = nn.Linear(D, D); s.goalp = nn.Linear(d_m, D)
comp = Comp(PROJ, D); comp.load_state_dict(ck['comp'])
for p in comp.parameters(): p.requires_grad = False
TAU = F.softplus(comp.log_tau) + 0.5
class KVGen(nn.Module):
    def __init__(s, D, layers, nkv, hd, Minj=4, gk0=64.0, gv0=8.0):
        super().__init__(); s.trunk = nn.Sequential(nn.Linear(D, 128), nn.GELU()); s.k = nn.ModuleDict(); s.v = nn.ModuleDict(); s.gk = nn.ParameterDict(); s.gv = nn.ParameterDict(); s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = Minj
        for L in layers:
            s.k[str(L)] = nn.Linear(128, Minj * nkv * hd); s.v[str(L)] = nn.Linear(128, Minj * nkv * hd); s.gk[str(L)] = nn.Parameter(torch.tensor(gk0)); s.gv[str(L)] = nn.Parameter(torch.tensor(gv0))
    def forward(s, h):
        z = s.trunk(h); o = {}
        for L in s.layers:
            k = F.normalize(s.k[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gk[str(L)]; v = F.normalize(s.v[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gv[str(L)]
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
        return o
gen = KVGen(D, TGT, nkv, hd).to(dev); gen.load_state_dict(ck['gen']); gen.eval()
def collect(C, b):
    q = comp.Wq(b).view(comp.h, comp.dh); Kk = comp.Wk(C).view(-1, comp.h, comp.dh); V = comp.Wv(C).view(-1, comp.h, comp.dh)
    a = torch.softmax(torch.einsum('hd,nhd->hn', q, Kk) / comp.dh ** 0.5, -1); return comp.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
def bstep(b, Cproj, anc, gamma):
    a = collect(Cproj - MUk, b)
    for _ in range(2): b = b + (-b + torch.tanh(comp.W(b) + a) + gamma * (anc - b)) / TAU / 2
    return b
def set_kv(h):
    o = gen(h.to(dev))
    for L in TGT: ki, vi = o[L]; mods[L]._kv_inj = (ki.to(model.dtype), vi.to(model.dtype))
def clear():
    for L in TGT: mods[L]._kv_inj = None
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def win_of(hist):
    w = hist[-W:]
    while w and w[0]['role'] == 'assistant': w = w[1:]
    return w or hist[-1:]
@torch.no_grad()
def sample_chunk(ms):
    model.config.use_cache = True; ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    return o[0, ids.shape[1]:]
@torch.no_grad()
def capture(ms, rids):                                                           # chunk's native-KV (projected) for the belief + its layer-32 gist
    clear(); model.config.use_cache = False; cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); nct = rids.shape[0]
    out = model(ids, output_hidden_states=True, use_cache=True); g = out.hidden_states[32][0, -nct:].float().cpu(); feats = []
    for L in FULL:
        lc = out.past_key_values.layers[L]; feats.append(lc.keys[0, :, -nct:, :].mean(0)); feats.append(lc.values[0, :, -nct:, :].mean(0))
    return (torch.cat(feats, dim=-1).float().cpu() @ Rp), gist(g)
def dec(ids): return tok.decode(ids, skip_special_tokens=True).split('</think>')[-1].strip()
MODES = [('tracker', 0.0, None), ('fixed', GAMMA, 'fixed'), ('slow', GAMMA, 'slow')]
agg = {n: [] for n, _, _ in MODES}
for si, (sA, sB) in enumerate(seeds[:1] if SMOKE else seeds):
    print('\n################ A=[%s]  vs  viable B=[%s]' % (sA[:40], sB[:40]), flush=True)
    hist = [{'role': 'user', 'content': sA}]; b = torch.zeros(D); arefs = []      # ESTABLISH A in the actuated loop
    for t in range(2 if SMOKE else NEST):
        set_kv(b); ch = sample_chunk(win_of(hist)); clear(); cp, g = capture(win_of(hist), ch); b = bstep(b, cp, b, 0.0)
        arefs.append(g); tx = dec(ch); hist += [{'role': 'assistant', 'content': tx}, {'role': 'user', 'content': tx}]
    anchor = b.clone(); A_ref = F.normalize(torch.stack(arefs).mean(0), dim=0)
    chB = sample_chunk([{'role': 'user', 'content': sB}]); txB = dec(chB); _, B_ref = capture([{'role': 'user', 'content': sB}], chB)  # viable B perturbation
    inj = [{'role': 'assistant', 'content': txB}, {'role': 'user', 'content': txB}]
    for name, g, amode in MODES:
        torch.manual_seed(100 + si); h2 = [x for x in hist] + inj; b2 = b.clone(); anc = anchor.clone(); affs = []
        for t in range(2 if SMOKE else NPOST):
            set_kv(b2); ch = sample_chunk(win_of(h2)); clear(); cp, gst = capture(win_of(h2), ch)
            b2 = bstep(b2, cp, anc, g)
            if amode == 'slow': anc = (1 - RHO) * anc + RHO * b2
            aff = float(F.cosine_similarity(gst, A_ref, 0) - F.cosine_similarity(gst, B_ref, 0)); affs.append(aff)
            tx = dec(ch); h2 += [{'role': 'assistant', 'content': tx}, {'role': 'user', 'content': tx}]
            if t in (0, NPOST - 1) and not SMOKE: print('   %-7s post+%d aff=%+.3f | %s' % (name, t, aff, tx[:70].replace(chr(10), ' ')), flush=True)
        agg[name] += affs; print('   >> %-7s mean aff_A-aff_B = %+.3f  (>0 holds A, <0 captured by B)' % (name, st.mean(affs)), flush=True)
print('\n=== CLOSED-LOOP STAKE VERDICT (generated-text affinity A-B, post-perturbation, pooled) ===', flush=True)
for n, _, _ in MODES: print('  %-7s : %+.3f' % (n, st.mean(agg[n])), flush=True)
print('  read: tracker<0 (captured by self-reinforcing B) while fixed>tracker = the committed stake defends the loop where input-recovery cannot.', flush=True)
print('=== ALL_DONE ===', flush=True)
