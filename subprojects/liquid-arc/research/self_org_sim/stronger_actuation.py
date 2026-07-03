# STRONGER KV-WRITE actuation — the named bottleneck. Write side: 3 layers x M=4 (12 memory-tokens) -> ALL 16 full-attn
# layers x M=8 (128 memory-tokens, ~10x). Parameter-EFFICIENT generator (shared k/v head conditioned on a per-layer
# embedding => ~3M params not ~16M, regularized) so the committed belief drives generation far harder without overfitting
# 45 trajectories. Reuses the FROZEN recall-generalizing belief (full_integration.pt). Cross-category eval vs +0.021/+0.08.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, T, D, PROJ = 3, 2.0, 256, 768
MINJ, GK0, GV0 = 8, 64.0, 8.0; SMOKE = os.environ.get('SMOKE', '0') == '1'
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
print('loading data + frozen recall belief ...', flush=True)
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data'] if len(m['texts']) >= 10]
ck = torch.load('/home/pokazge/checkpoints/full_integration.pt', weights_only=False, map_location='cpu'); Rp = ck['Rp']
d_m = data[0]['gen'][0].shape[1]; MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
for m in data:
    m['z'] = [F.normalize((c - MU).mean(0), dim=0) for c in m['gen']]; m['perc'] = [c @ Rp for c in m['nkv']]; m['nkv'] = None
MUk = torch.cat([c for m in data for c in m['perc']], 0).mean(0)
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
class Comp(nn.Module):
    def __init__(s, in_dim, D=256, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(in_dim, heads * dh); s.Wv = nn.Linear(in_dim, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.recall = nn.Linear(D, D); s.goalp = nn.Linear(d_m, D)
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / s.dh ** 0.5, -1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def beliefs(s, m):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; hs = []
        for t in range(len(m['perc'])):
            a = s.collect(m['perc'][t] - MUk, b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; hs.append(h)
        return torch.stack(hs)
comp = Comp(PROJ, D); comp.load_state_dict(ck['comp'])
for p in comp.parameters(): p.requires_grad = False
with torch.no_grad():
    for m in data: m['hs'] = comp.beliefs(m).detach()
print('STRONGER actuation | train=%d test=%d held-out=%s | recall belief frozen' % (len(tr), len(te), sorted(hold)), flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
model.config.use_cache = False
try: model.gradient_checkpointing_enable()
except Exception as e: print('gc fail', repr(e), flush=True)
nkv = model.config.num_key_value_heads; hd = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
@torch.no_grad()
def detect_full():
    ids = tok('hi', return_tensors='pt').input_ids.to(dev); out = model(ids, use_cache=True)
    return [i for i, L in enumerate(out.past_key_values.layers) if getattr(L, 'keys', None) is not None]
TGT = detect_full(); mods = {L: model.model.layers[L].self_attn for L in TGT}
for sa in mods.values(): sa._kv_inj = None
print('27B loaded %.1fGB; WRITE targets = ALL %d full-attn layers, M=%d (was 3 layers x4)' % (torch.cuda.memory_allocated() / 1e9, len(TGT), MINJ), flush=True)
class StrongKVGen(nn.Module):                                                    # param-efficient: shared k/v head conditioned on per-layer embedding
    def __init__(s, D, layers, nkv, hd, Minj, gk0, gv0, emb=48, code=160, drop=0.2):
        super().__init__(); s.trunk = nn.Sequential(nn.Linear(D, code), nn.GELU(), nn.Dropout(drop)); s.lemb = nn.Embedding(len(layers), emb)
        s.kh = nn.Linear(code + emb, Minj * nkv * hd); s.vh = nn.Linear(code + emb, Minj * nkv * hd)
        s.gk = nn.Parameter(torch.full((len(layers),), float(gk0))); s.gv = nn.Parameter(torch.full((len(layers),), float(gv0)))
        s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = Minj
    def forward(s, h):
        z = s.trunk(h); o = {}
        for i, L in enumerate(s.layers):
            ze = torch.cat([z, s.lemb.weight[i]])
            k = F.normalize(s.kh(ze).view(s.nkv, s.M, s.hd), dim=-1) * s.gk[i]; v = F.normalize(s.vh(ze).view(s.nkv, s.M, s.hd), dim=-1) * s.gv[i]
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
        return o
gen = StrongKVGen(D, TGT, nkv, hd, MINJ, GK0, GV0).to(dev); opt = torch.optim.Adam(gen.parameters(), lr=1e-3, weight_decay=1e-3)
print('StrongKVGen params: %.2fM' % (sum(p.numel() for p in gen.parameters()) / 1e6), flush=True)
def set_kv(h):
    o = gen(h.to(dev))
    for L in TGT: ki, vi = o[L]; mods[L]._kv_inj = (ki.to(model.dtype), vi.to(model.dtype))
def clear():
    for L in TGT: mods[L]._kv_inj = None
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def win_msgs(ms, Wn=W):
    w = ms[-Wn:]
    while w and w[0]['role'] == 'assistant': w = w[1:]
    return w or ms[-1:]
def full_msgs(m, t):
    ms = [{'role': 'user', 'content': m['seed']}]
    for i in range(t): ms += [{'role': 'assistant', 'content': m['texts'][i]}, {'role': 'user', 'content': m['texts'][i]}]
    return ms
def chunk_logits(cm, ct):
    cids = tok(tmpl(cm), return_tensors='pt').input_ids.to(dev); rids = tok(ct or '.', return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    ids = torch.cat([cids, rids], 1); lg = model(ids).logits[0]; s = cids.shape[1] - 1
    return lg[s:s + rids.shape[1]].float()
print('distillation (stronger actuation) ...', flush=True)
gen.train()
for ep in range(1 if SMOKE else 4):
    tot = 0.0; n = 0
    for m in tr:
        for t in range(W, len(m['texts'])):
            fm = full_msgs(m, t); clear()
            with torch.no_grad(): tl = chunk_logits(fm, m['texts'][t])
            set_kv(m['hs'][t - 1].to(dev)); sl = chunk_logits(win_msgs(fm), m['texts'][t])
            nn_ = min(sl.shape[0], tl.shape[0]); loss = F.kl_div(F.log_softmax(sl[:nn_] / T, -1), F.softmax(tl[:nn_] / T, -1), reduction='batchmean') * (T * T)
            opt.zero_grad(); loss.backward(); opt.step(); clear(); tot += float(loss); n += 1
            if SMOKE and n >= 3: break
        if SMOKE: break
    print('  ep %d kl-loss %.3f (n=%d) gk~%.0f gv~%.1f' % (ep, tot / max(n, 1), n, float(gen.gk.mean()), float(gen.gv.mean())), flush=True)
gen.eval()
print('\n=== CROSS-CATEGORY (STRONGER actuation, %d held-out cats) ===' % len(hold), flush=True)
aw, al, klw, kll = [], [], [], []
with torch.no_grad():
    for m in te:
        for t in range(W, len(m['texts'])):
            fm = full_msgs(m, t); wm = win_msgs(fm)
            clear(); tl = chunk_logits(fm, m['texts'][t]); wl = chunk_logits(wm, m['texts'][t]); set_kv(m['hs'][t - 1].to(dev)); ll = chunk_logits(wm, m['texts'][t]); clear()
            nn_ = min(tl.shape[0], wl.shape[0], ll.shape[0]); tt = tl[:nn_]; ww = wl[:nn_]; lll = ll[:nn_]; tg = tt.argmax(-1)
            aw.append(float((ww.argmax(-1) == tg).float().mean())); al.append(float((lll.argmax(-1) == tg).float().mean()))
            tp = F.softmax(tt, -1); klw.append(float(F.kl_div(F.log_softmax(ww, -1), tp, reduction='batchmean'))); kll.append(float(F.kl_div(F.log_softmax(lll, -1), tp, reduction='batchmean')))
            if SMOKE: break
        if SMOKE: break
print('  agreement: window-alone %.3f -> window+STRONG-actuator %.3f   (lift %+.3f)' % (st.mean(aw), st.mean(al), st.mean(al) - st.mean(aw)), flush=True)
print('  KL(full||window) %.3f -> %.3f   (reduction %+.3f)   [n=%d chunks, UNSEEN cats]' % (st.mean(klw), st.mean(kll), st.mean(klw) - st.mean(kll), len(aw)), flush=True)
print('  vs 3-layer x4 actuation: +0.021 agr / +0.080 KL (within-cat ceiling +0.033 / +0.155)', flush=True)
torch.save({'gen': gen.state_dict(), 'tgt': TGT, 'minj': MINJ, 'Rp': Rp}, '/home/pokazge/checkpoints/stronger_actuation.pt')
print('=== ALL_DONE ===', flush=True)
