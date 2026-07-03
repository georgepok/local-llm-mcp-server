# IN-LOOP teacher-forced distillation of the KV-WRITE actuator — the ELEGANT actuation: the Liquid belief generates (k,v)
# pairs PREPENDED to the model's full-attn KV; the frozen model ATTENDS to them through its OWN softmax ("attention on
# attention"), zero weight-splice, zero context-token cost. Same loop / data / compressor / teacher-student chunk_logits /
# held-out eval as the LoRA version (inject_lora_inloop.py); ONLY the actuator changes, for an apples-to-apples comparison
# vs LoRA's +0.026 next-token agreement / +0.088 KL reduction. (Perception still layer-32 here; native-read is a separate
# pivot — this isolates the WRITE-side elegance.) SMOKE=1 tiny. GAIN0/MINJ/TGT_LAYERS via env.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, T = 3, 2.0
TGT = [int(x) for x in os.environ.get('TGT_LAYERS', '23,27,31').split(',')]; SMOKE = os.environ.get('SMOKE', '0') == '1'
MINJ = int(os.environ.get('MINJ', '4')); GK0 = float(os.environ.get('GK0', '64')); GV0 = float(os.environ.get('GV0', '8'))  # unit-dir x gain => gain IS the vector norm. smoke: keys need norm ~64-128 to win softmax mass; values ~real (~16)
_orig = Q5.eager_attention_forward                                               # KV-write patch: 'eager' not in registry -> module-local fallback, so this takes effect
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
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('trajs train=%d test=%d W=%d TGT=%s MINJ=%d gk0=%.1f gv0=%.1f smoke=%s' % (len(tr), len(te), W, TGT, MINJ, GK0, GV0, SMOKE), flush=True)
class Compressor(nn.Module):
    def __init__(s, d_m, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(d_m, heads * dh); s.Wv = nn.Linear(d_m, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K_ = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K_) / s.dh ** 0.5, dim=-1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def run(s, m):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; pr = []; hs = []
        for t in range(len(m['gen'])):
            hp = h; a = s.collect(cen(m['gen'][t]), b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; hs.append(h); pr.append(s.pred(torch.cat([s.cz(m['z'][t]), hp])))
        return torch.stack(pr), torch.stack(hs)
comp = Compressor(d_m); copt = torch.optim.Adam(comp.parameters(), lr=1e-3)
for ep in range(40 if SMOKE else 150):
    loss = 0.0; n = 0
    for m in tr:
        pr, _ = comp.run(m); z = torch.stack(m['z']); loss = loss + (1 - (F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)).mean(); n += 1
    copt.zero_grad(); (loss / n).backward(); copt.step()
for m in data:
    with torch.no_grad(): _, hs = comp.run(m); m['hs'] = hs
D = comp.D; print('compressor trained; loading 27B ...', flush=True)
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
print('27B loaded %.1fGB; nkv=%d hd=%d KV-write targets=%s' % (torch.cuda.memory_allocated() / 1e9, nkv, hd, TGT), flush=True)
class KVGen(nn.Module):                                                          # belief h -> (k,v) prefix per target full-attn layer; unit-normalized dirs x learnable gains
    def __init__(s, D, layers, nkv, hd, Minj, gk0, gv0):
        super().__init__(); s.trunk = nn.Sequential(nn.Linear(D, 128), nn.GELU()); s.k = nn.ModuleDict(); s.v = nn.ModuleDict(); s.gk = nn.ParameterDict(); s.gv = nn.ParameterDict()
        s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = Minj
        for L in layers:
            s.k[str(L)] = nn.Linear(128, Minj * nkv * hd); s.v[str(L)] = nn.Linear(128, Minj * nkv * hd)
            s.gk[str(L)] = nn.Parameter(torch.tensor(float(gk0))); s.gv[str(L)] = nn.Parameter(torch.tensor(float(gv0)))
    def forward(s, h):
        z = s.trunk(h); o = {}
        for L in s.layers:                                                       # unit-direction per head x per-head-dim gain => norm = gain*sqrt(hd), in the smoke's effective regime
            k = F.normalize(s.k[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gk[str(L)]
            v = F.normalize(s.v[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gv[str(L)]
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))                              # [1, nkv, M, hd]
        return o
gen = KVGen(D, TGT, nkv, hd, MINJ, GK0, GV0).to(dev); opt = torch.optim.Adam(gen.parameters(), lr=1e-3, weight_decay=1e-4)
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
            with torch.no_grad(): tl = chunk_logits(fm, m['texts'][t])           # teacher: full-context
            set_kv(m['hs'][t - W].to(dev)); sl = chunk_logits(win_msgs(fm), m['texts'][t])   # student: window + KV-write
            n_ = min(sl.shape[0], tl.shape[0])
            loss = F.kl_div(F.log_softmax(sl[:n_] / T, -1), F.softmax(tl[:n_] / T, -1), reduction='batchmean') * (T * T)
            opt.zero_grad(); loss.backward(); opt.step(); clear(); tot += float(loss); n += 1
            if SMOKE and n >= 3: break
        if SMOKE: break
    print('inloop ep %d kl-loss %.3f (n=%d) gk/gv=%s' % (ep, tot / max(n, 1), n, {L: (round(float(gen.gk[str(L)]), 1), round(float(gen.gv[str(L)]), 1)) for L in TGT}), flush=True)
print('\n=== KV-WRITE IN-LOOP (held-out): next-token AGREEMENT w/ full-context, and KL reduction ===', flush=True)
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
print('  next-token agreement w/ full-context:  window-alone %.3f   window+KVwrite %.3f   (lift %+.3f)' % (st.mean(aw), st.mean(al), st.mean(al) - st.mean(aw)), flush=True)
print('  KL(full || window) %.3f  ->  KL(full || window+KVwrite) %.3f   (reduction %+.3f)' % (st.mean(klw), st.mean(kll), st.mean(klw) - st.mean(kll)), flush=True)
print('  vs LoRA reference: +0.026 agreement / +0.088 KL reduction. final gk/gv=%s' % {L: (round(float(gen.gk[str(L)]), 1), round(float(gen.gv[str(L)]), 1)) for L in TGT}, flush=True)
torch.save({'gen': gen.state_dict(), 'comp': comp.state_dict(), 'tgt': TGT, 'nkv': nkv, 'hd': hd, 'minj': MINJ}, '/home/pokazge/checkpoints/kv_write_inloop.pt')
print('=== ALL_DONE ===', flush=True)
