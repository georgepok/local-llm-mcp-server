# UNIFIED NATIVE actuator (step 2): the Liquid READS the model's native KV memory (perception — replaces the layer-32
# tap) AND WRITES (k,v) the model attends to (actuation). The ENTIRE Liquid<->LLM coupling now goes through the model's
# own KV interface: no hidden-layer tap, no weight splice. Reuses validated pieces — native-KV perception (cached
# native_states.pt, the +0.224 channel) drives the belief; KV-write actuation (+0.033 agr / +0.155 KL) is unchanged.
# Question: does fully-native read+write match the layer-32-read KV-write? Same in-loop distillation eval, held-out.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, T = 3, 2.0
TGT = [int(x) for x in os.environ.get('TGT_LAYERS', '23,27,31').split(',')]; SMOKE = os.environ.get('SMOKE', '0') == '1'
MINJ = int(os.environ.get('MINJ', '4')); GK0 = float(os.environ.get('GK0', '64')); GV0 = float(os.environ.get('GV0', '8')); D = 256
_orig = Q5.eager_attention_forward                                               # KV-WRITE patch (validated): prepend injected (k,v), pad additive mask
def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    inj = getattr(module, '_kv_inj', None)
    if inj is not None:
        ki, vi = inj; key = torch.cat([ki.to(key.dtype), key], dim=2); value = torch.cat([vi.to(value.dtype), value], dim=2)
        if attention_mask is not None:
            pad = torch.zeros(*attention_mask.shape[:-1], ki.shape[2], dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([pad, attention_mask], dim=-1)
    return _orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
Q5.eager_attention_forward = patched
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data'] if len(m['texts']) >= 10]
d_m = data[0]['gen'][0].shape[1]; MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
def cen(C): return C - MU
for m in data: m['z'] = [F.normalize(cen(c).mean(0), dim=0) for c in m['gen']]
nat = torch.load('/home/pokazge/checkpoints/native_states.pt', weights_only=False)['KV']  # native-KV perception per chunk (windowed reads), aligned to data order
assert len(nat) == len(data), 'native cache misaligned with data (%d vs %d)' % (len(nat), len(data))
kv_dim = nat[0].shape[1]; g = torch.Generator().manual_seed(7); Rkv = F.normalize(torch.randn(kv_dim, 128, generator=g), dim=0)  # fixed projection (overfit control on small data)
for i, m in enumerate(data):
    assert nat[i].shape[0] == len(m['texts']), 'len mismatch traj %d' % i
    m['perc'] = nat[i] @ Rkv                                                      # [n_chunks, 128] — the Liquid's NATIVE perception channel
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('UNIFIED native read+write | trajs train=%d test=%d kv_dim=%d D=%d TGT=%s gk0/gv0=%.0f/%.0f smoke=%s' % (len(tr), len(te), kv_dim, D, TGT, GK0, GV0, SMOKE), flush=True)
class Comp(nn.Module):                                                           # READS native-KV perception -> persistent belief; pretrained to predict next chunk identity (holds dropped context)
    def __init__(s, D):
        super().__init__(); s.read = nn.Linear(128, D); s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m)); s.D = D
    def run(s, m):
        tau = F.softplus(s.log_tau) + 0.5; b = torch.zeros(s.D); h = torch.zeros(s.D); pr = []; hs = []
        for t in range(len(m['perc'])):
            hp = h; a = s.read(m['perc'][t])
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; hs.append(h); pr.append(s.pred(torch.cat([s.cz(m['z'][t]), hp])))
        return torch.stack(pr), torch.stack(hs)
comp = Comp(D); copt = torch.optim.Adam(comp.parameters(), lr=2e-3, weight_decay=1e-4)
for ep in range(40 if SMOKE else 200):
    loss = 0.0; n = 0
    for m in tr:
        pr, _ = comp.run(m); z = torch.stack(m['z']); loss = loss + (1 - (F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)).mean(); n += 1
    copt.zero_grad(); (loss / n).backward(); copt.step()
with torch.no_grad():
    gap = []
    for m in te:
        pr, _ = comp.run(m); z = torch.stack(m['z']); full = float((F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1).mean()); gap.append(full)
    for m in data: _, hs = comp.run(m); m['hs'] = hs
print('native-KV compressor pretrained; held-out next-gist coh %.3f; loading 27B ...' % st.mean(gap), flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
model.config.use_cache = False
try: model.gradient_checkpointing_enable()
except Exception as e: print('gc fail', repr(e), flush=True)
nkv = model.config.num_key_value_heads; hd = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
mods = {L: model.model.layers[L].self_attn for L in TGT}
for sa in mods.values(): sa._kv_inj = None
print('27B loaded %.1fGB; nkv=%d hd=%d KV read+write targets=%s' % (torch.cuda.memory_allocated() / 1e9, nkv, hd, TGT), flush=True)
class KVGen(nn.Module):
    def __init__(s, D, layers, nkv, hd, Minj, gk0, gv0):
        super().__init__(); s.trunk = nn.Sequential(nn.Linear(D, 128), nn.GELU()); s.k = nn.ModuleDict(); s.v = nn.ModuleDict(); s.gk = nn.ParameterDict(); s.gv = nn.ParameterDict()
        s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = Minj
        for L in layers:
            s.k[str(L)] = nn.Linear(128, Minj * nkv * hd); s.v[str(L)] = nn.Linear(128, Minj * nkv * hd)
            s.gk[str(L)] = nn.Parameter(torch.tensor(float(gk0))); s.gv[str(L)] = nn.Parameter(torch.tensor(float(gv0)))
    def forward(s, h):
        z = s.trunk(h); o = {}
        for L in s.layers:
            k = F.normalize(s.k[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gk[str(L)]
            v = F.normalize(s.v[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gv[str(L)]
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
        return o
gen = KVGen(comp.D, TGT, nkv, hd, MINJ, GK0, GV0).to(dev); opt = torch.optim.Adam(gen.parameters(), lr=1e-3, weight_decay=1e-4)
def set_kv(h):
    o = gen(h)
    for L in TGT: ki, vi = o[L]; mods[L]._kv_inj = (ki.to(model.dtype), vi.to(model.dtype))
def clear():
    for L in TGT: mods[L]._kv_inj = None
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def full_msgs(m, t):
    ms = [{'role': 'user', 'content': m['seed']}]
    for i in range(t): ms += [{'role': 'assistant', 'content': m['texts'][i]}, {'role': 'user', 'content': m['texts'][i]}]
    return ms
def win_msgs(ms):
    w = ms[-W:]
    while w and w[0]['role'] == 'assistant': w = w[1:]
    return w or ms[-1:]
def chunk_logits(ctx_msgs, chunk_text):
    cids = tok(tmpl(ctx_msgs), return_tensors='pt').input_ids.to(dev); rids = tok(chunk_text or '.', return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    ids = torch.cat([cids, rids], 1); lg = model(ids).logits[0]; st_ = cids.shape[1] - 1
    return lg[st_:st_ + rids.shape[1]].float()
EP = 1 if SMOKE else 4
for ep in range(EP):
    tot = 0.0; n = 0
    for m in tr:
        for t in range(W, len(m['texts'])):
            fm = full_msgs(m, t); clear()
            with torch.no_grad(): tl = chunk_logits(fm, m['texts'][t])
            set_kv(m['hs'][t - W].to(dev)); sl = chunk_logits(win_msgs(fm), m['texts'][t])
            n_ = min(sl.shape[0], tl.shape[0])
            loss = F.kl_div(F.log_softmax(sl[:n_] / T, -1), F.softmax(tl[:n_] / T, -1), reduction='batchmean') * (T * T)
            opt.zero_grad(); loss.backward(); opt.step(); clear(); tot += float(loss); n += 1
            if SMOKE and n >= 3: break
        if SMOKE: break
    print('unified ep %d kl-loss %.3f (n=%d) gk/gv=%s' % (ep, tot / max(n, 1), n, {L: (round(float(gen.gk[str(L)]), 1), round(float(gen.gv[str(L)]), 1)) for L in TGT}), flush=True)
print('\n=== UNIFIED NATIVE read+write (held-out): agreement + KL reduction ===', flush=True)
aw, al, klw, kll = [], [], [], []
with torch.no_grad():
    for m in te:
        for t in range(W, len(m['texts'])):
            fm = full_msgs(m, t); wm = win_msgs(fm)
            clear(); tl = chunk_logits(fm, m['texts'][t]); wl = chunk_logits(wm, m['texts'][t]); set_kv(m['hs'][t - W].to(dev)); ll = chunk_logits(wm, m['texts'][t]); clear()
            n_ = min(tl.shape[0], wl.shape[0], ll.shape[0]); tt = tl[:n_]; ww = wl[:n_]; lll = ll[:n_]; tgt = tt.argmax(-1)
            aw.append(float((ww.argmax(-1) == tgt).float().mean())); al.append(float((lll.argmax(-1) == tgt).float().mean()))
            tp = F.softmax(tt, -1); klw.append(float(F.kl_div(F.log_softmax(ww, -1), tp, reduction='batchmean'))); kll.append(float(F.kl_div(F.log_softmax(lll, -1), tp, reduction='batchmean')))
            if SMOKE: break
        if SMOKE: break
print('  next-token agreement w/ full-context:  window-alone %.3f   window+NATIVE(read+write) %.3f   (lift %+.3f)' % (st.mean(aw), st.mean(al), st.mean(al) - st.mean(aw)), flush=True)
print('  KL(full || window) %.3f  ->  KL(full || window+native) %.3f   (reduction %+.3f)' % (st.mean(klw), st.mean(kll), st.mean(klw) - st.mean(kll)), flush=True)
print('  vs layer-32-read KV-write: +0.033 agr / +0.155 KL  (and LoRA: +0.026 / +0.088). gains=%s' % {L: (round(float(gen.gk[str(L)]), 1), round(float(gen.gv[str(L)]), 1)) for L in TGT}, flush=True)
torch.save({'gen': gen.state_dict(), 'comp': comp.state_dict(), 'tgt': TGT, 'nkv': nkv, 'hd': hd, 'minj': MINJ, 'Rkv': Rkv}, '/home/pokazge/checkpoints/kv_unified.pt')
print('=== ALL_DONE ===', flush=True)
