# UNIFIED NATIVE actuator, BROAD READ (step 2, v3): the Liquid's belief does attention-on-attention over ALL full-attn
# layers' native KV (the model's FULL KV memory — the native-read probe's +0.224 channel used all 16, not 3), with a fixed
# random projection for overfit control on the wider read. READ broad (perception wants breadth), WRITE targeted at the
# validated 3 (actuation propagates). Liquid body + AoA both intact; no layer-32 tap, no mean-pool, no read=write handicap.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, T, D, PROJ = 3, 2.0, 384, 768
TGT = [int(x) for x in os.environ.get('TGT_LAYERS', '23,27,31').split(',')]; SMOKE = os.environ.get('SMOKE', '0') == '1'  # WRITE layers
MINJ = int(os.environ.get('MINJ', '4')); GK0 = float(os.environ.get('GK0', '64')); GV0 = float(os.environ.get('GV0', '8'))
CACHE = '/home/pokazge/checkpoints/native_kv_seq_all.pt'
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
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data'] if len(m['texts']) >= 10]
d_m = data[0]['gen'][0].shape[1]; MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
def cen(C): return C - MU
for m in data: m['z'] = [F.normalize(cen(c).mean(0), dim=0) for c in m['gen']]
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
nkv = model.config.num_key_value_heads; hd = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
mods = {L: model.model.layers[L].self_attn for L in TGT}
for sa in mods.values(): sa._kv_inj = None
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def win_msgs(ms):
    w = ms[-W:]
    while w and w[0]['role'] == 'assistant': w = w[1:]
    return w or ms[-1:]
def full_msgs(m, t):
    ms = [{'role': 'user', 'content': m['seed']}]
    for i in range(t): ms += [{'role': 'assistant', 'content': m['texts'][i]}, {'role': 'user', 'content': m['texts'][i]}]
    return ms
@torch.no_grad()
def detect_full():                                                               # which layers are full-attn (have a KV cache)
    for sa in mods.values(): sa._kv_inj = None
    ids = tok('hello world', return_tensors='pt').input_ids.to(dev); out = model(ids, use_cache=True)
    return [i for i, L in enumerate(out.past_key_values.layers) if getattr(L, 'keys', None) is not None]
FULL = detect_full(); print('full-attn layers (%d): %s' % (len(FULL), FULL), flush=True)
@torch.no_grad()
def native_kv_seq(ms, chunk_text):                                               # chunk tokens' keys+values at ALL full-attn layers (mean over heads) -> [n_tok, len(FULL)*2*hd]
    for sa in mods.values(): sa._kv_inj = None
    cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); rids = tok(chunk_text or '.', return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    ids = torch.cat([cids, rids], 1); out = model(ids, use_cache=True); cache = out.past_key_values; nct = rids.shape[1]; feats = []
    for L in FULL:
        lc = cache.layers[L]; feats.append(lc.keys[0, :, -nct:, :].mean(0)); feats.append(lc.values[0, :, -nct:, :].mean(0))
    return torch.cat(feats, dim=-1).float().cpu()
if os.path.exists(CACHE):
    seqs = torch.load(CACHE, weights_only=False)
    for i, m in enumerate(data): m['nkv'] = seqs[i]
    print('loaded cached broad native KV sequences', flush=True)
else:
    for mi, m in enumerate(data):
        hist = [{'role': 'user', 'content': m['seed']}]; ns = []
        for t in range(len(m['texts'])):
            ns.append(native_kv_seq(win_msgs(hist), m['texts'][t]))
            hist += [{'role': 'assistant', 'content': m['texts'][t]}, {'role': 'user', 'content': m['texts'][t]}]
        m['nkv'] = ns
        if mi == 0: print('broad native KV seq raw dim=%d (per token)' % ns[0].shape[1], flush=True)
    torch.save([m['nkv'] for m in data], CACHE); print('extracted + cached broad native KV sequences for %d trajs' % len(data), flush=True)
raw_dim = data[0]['nkv'][0].shape[1]; gR = torch.Generator().manual_seed(11); Rp = F.normalize(torch.randn(raw_dim, PROJ, generator=gR), dim=0)  # fixed projection: overfit control on the wide read
for m in data: m['perc'] = [c @ Rp for c in m['nkv']]                            # [n_tok, PROJ] per chunk
MUk = torch.cat([c for m in data for c in m['perc']], 0).mean(0)
def cenk(C): return C - MUk
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('BROAD-READ unified: Liquid+AoA over %d full-attn layers (raw %d -> proj %d) | train=%d test=%d WRITE=%s' % (len(FULL), raw_dim, PROJ, len(tr), len(te), TGT), flush=True)
class Compressor(nn.Module):
    def __init__(s, in_dim, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(in_dim, heads * dh); s.Wv = nn.Linear(in_dim, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K_ = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K_) / s.dh ** 0.5, dim=-1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def run(s, m):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; pr = []; hs = []
        for t in range(len(m['perc'])):
            hp = h; a = s.collect(cenk(m['perc'][t]), b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; hs.append(h); pr.append(s.pred(torch.cat([s.cz(m['z'][t]), hp])))
        return torch.stack(pr), torch.stack(hs)
comp = Compressor(PROJ, D); copt = torch.optim.Adam(comp.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(40 if SMOKE else 150):
    loss = 0.0; n = 0
    for m in tr:
        pr, _ = comp.run(m); z = torch.stack(m['z']); loss = loss + (1 - (F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)).mean(); n += 1
    copt.zero_grad(); (loss / n).backward(); copt.step()
with torch.no_grad():
    gp = [float((F.normalize(comp.run(m)[0][:-1], dim=-1) * torch.stack(m['z'])[1:]).sum(-1).mean()) for m in te]
    for m in data: _, hs = comp.run(m); m['hs'] = hs
print('broad native-KV-AoA Liquid pretrained; held-out next-gist coh %.3f (3-layer was 0.238, crude-pool 0.253)' % st.mean(gp), flush=True)
model.config.use_cache = False
try: model.gradient_checkpointing_enable()
except Exception as e: print('gc fail', repr(e), flush=True)
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
gen = KVGen(D, TGT, nkv, hd, MINJ, GK0, GV0).to(dev); opt = torch.optim.Adam(gen.parameters(), lr=1e-3, weight_decay=1e-4)
def set_kv(h):
    o = gen(h)
    for L in TGT: ki, vi = o[L]; mods[L]._kv_inj = (ki.to(model.dtype), vi.to(model.dtype))
def clear():
    for L in TGT: mods[L]._kv_inj = None
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
    print('broad-unified ep %d kl-loss %.3f (n=%d) gk/gv=%s' % (ep, tot / max(n, 1), n, {L: (round(float(gen.gk[str(L)]), 1), round(float(gen.gv[str(L)]), 1)) for L in TGT}), flush=True)
print('\n=== BROAD-READ UNIFIED (Liquid+AoA over all full-attn KV + KV write, held-out) ===', flush=True)
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
print('  next-token agreement w/ full-context:  window-alone %.3f   window+NATIVE(broad-AoA-read+KV-write) %.3f   (lift %+.3f)' % (st.mean(aw), st.mean(al), st.mean(al) - st.mean(aw)), flush=True)
print('  KL(full || window) %.3f  ->  KL(full || window+native) %.3f   (reduction %+.3f)' % (st.mean(klw), st.mean(kll), st.mean(klw) - st.mean(kll)), flush=True)
print('  vs layer-32-AoA KV-write: +0.033 agr / +0.155 KL', flush=True)
torch.save({'gen': gen.state_dict(), 'comp': comp.state_dict(), 'tgt': TGT, 'full': FULL, 'nkv': nkv, 'hd': hd, 'minj': MINJ, 'Rp': Rp}, '/home/pokazge/checkpoints/kv_unified3.pt')
print('=== ALL_DONE ===', flush=True)
