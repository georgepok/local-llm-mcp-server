# RESHAPE THE ACTUATION TRAINING (the second half of the pipeline). The recall belief generalizes; the KV-write
# distillation is still predict-style (match full-context) and overfits the belief->KV map to seen categories (cross-cat
# lift +0.021/+0.080 was HALF the within-cat +0.033/+0.155). Apply the SAME anti-memorization shaping to the ACTUATION:
# (1) instance-multiplication — vary window size W in {1,2,3} per step so the same full-context target appears under many
# drop-levels (no window->KV shortcut; must use the belief generally), (2) bottleneck the KVGen (trunk 64) + belief-input
# dropout + weight-decay. Reuses the frozen recall belief from full_integration.pt. EVAL cross-category (15 held-out).
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'; T, D, PROJ = 2.0, 256, 768
TGT = [23, 27, 31]; MINJ, GK0, GV0 = 4, 64.0, 8.0; SMOKE = os.environ.get('SMOKE', '0') == '1'; WAUG = [1, 2, 3]; WEVAL = 3
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
    m['z'] = [F.normalize((c - MU).mean(0), dim=0) for c in m['gen']]
    m['perc'] = [c @ Rp for c in m['nkv']]; m['nkv'] = None
MUk = torch.cat([c for m in data for c in m['perc']], 0).mean(0)
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
class Comp(nn.Module):                                                           # SAME as full_integration (to load the recall belief)
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
print('FULL=%d train=%d test=%d cats=%d held-out=%s | recall belief loaded (frozen)' % (len(data), len(tr), len(te), len(fids), sorted(hold)), flush=True)
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
class KVGen2(nn.Module):                                                         # RESHAPED: bottleneck trunk(64) + belief-input dropout (anti-memorization)
    def __init__(s, D, layers, nkv, hd, Minj, gk0, gv0, bn=64, drop=0.3):
        super().__init__(); s.indrop = nn.Dropout(drop); s.trunk = nn.Sequential(nn.Linear(D, bn), nn.GELU(), nn.Dropout(drop))
        s.k = nn.ModuleDict(); s.v = nn.ModuleDict(); s.gk = nn.ParameterDict(); s.gv = nn.ParameterDict(); s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = Minj
        for L in layers:
            s.k[str(L)] = nn.Linear(bn, Minj * nkv * hd); s.v[str(L)] = nn.Linear(bn, Minj * nkv * hd)
            s.gk[str(L)] = nn.Parameter(torch.tensor(float(gk0))); s.gv[str(L)] = nn.Parameter(torch.tensor(float(gv0)))
    def forward(s, h):
        z = s.trunk(s.indrop(h)); o = {}
        for L in s.layers:
            k = F.normalize(s.k[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gk[str(L)]; v = F.normalize(s.v[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gv[str(L)]
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
        return o
gen = KVGen2(D, TGT, nkv, hd, MINJ, GK0, GV0).to(dev); opt = torch.optim.Adam(gen.parameters(), lr=1e-3, weight_decay=1e-3)   # stronger wd
def set_kv(h):
    o = gen(h)
    for L in TGT: ki, vi = o[L]; mods[L]._kv_inj = (ki.to(model.dtype), vi.to(model.dtype))
def clear():
    for L in TGT: mods[L]._kv_inj = None
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def win_msgs(ms, Wn):
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
print('reshaped (augmented + bottleneck) KV-write distillation ...', flush=True)
gen.train()
for ep in range(1 if SMOKE else 5):
    tot = 0.0; n = 0
    for m in tr:
        for t in range(max(WAUG), len(m['texts'])):
            fm = full_msgs(m, t); clear()
            with torch.no_grad(): tl = chunk_logits(fm, m['texts'][t]); tlsm = F.softmax(tl / T, -1)
            Wn = WAUG[int(torch.randint(0, len(WAUG), (1,)))]                     # stochastic instance-multiplication: 1 random drop-level/step, varied across epochs (same per-epoch cost)
            set_kv(m['hs'][t - 1].to(dev)); sl = chunk_logits(win_msgs(fm, Wn), m['texts'][t])
            nn_ = min(sl.shape[0], tl.shape[0]); loss = F.kl_div(F.log_softmax(sl[:nn_] / T, -1), tlsm[:nn_], reduction='batchmean') * (T * T)
            opt.zero_grad(); loss.backward(); opt.step(); clear(); tot += float(loss); n += 1
            if SMOKE and n >= 3: break
        if SMOKE: break
    print('  reshaped distill ep %d kl-loss %.3f (n=%d) gk/gv=%s' % (ep, tot / max(n, 1), n, {L: (round(float(gen.gk[str(L)]), 1), round(float(gen.gv[str(L)]), 1)) for L in TGT}), flush=True)
gen.eval()
print('\n=== CROSS-CATEGORY (reshaped actuation, %d held-out cats, eval W=%d) ===' % (len(hold), WEVAL), flush=True)
aw, al, klw, kll = [], [], [], []
with torch.no_grad():
    for m in te:
        for t in range(WEVAL, len(m['texts'])):
            fm = full_msgs(m, t); wm = win_msgs(fm, WEVAL)
            clear(); tl = chunk_logits(fm, m['texts'][t]); wl = chunk_logits(wm, m['texts'][t]); set_kv(m['hs'][t - 1].to(dev)); ll = chunk_logits(wm, m['texts'][t]); clear()
            nn_ = min(tl.shape[0], wl.shape[0], ll.shape[0]); tt = tl[:nn_]; ww = wl[:nn_]; lll = ll[:nn_]; tg = tt.argmax(-1)
            aw.append(float((ww.argmax(-1) == tg).float().mean())); al.append(float((lll.argmax(-1) == tg).float().mean()))
            tp = F.softmax(tt, -1); klw.append(float(F.kl_div(F.log_softmax(ww, -1), tp, reduction='batchmean'))); kll.append(float(F.kl_div(F.log_softmax(lll, -1), tp, reduction='batchmean')))
            if SMOKE: break
        if SMOKE: break
print('  agreement: window-alone %.3f -> window+reshaped-actuator %.3f   (lift %+.3f)' % (st.mean(aw), st.mean(al), st.mean(al) - st.mean(aw)), flush=True)
print('  KL(full||window) %.3f -> %.3f   (reduction %+.3f)   [n=%d chunks, UNSEEN cats]' % (st.mean(klw), st.mean(kll), st.mean(klw) - st.mean(kll), len(aw)), flush=True)
print('  vs un-reshaped actuation cross-cat: +0.021 agr / +0.080 KL  (within-cat ceiling +0.033 / +0.155)', flush=True)
print('=== ALL_DONE ===', flush=True)
