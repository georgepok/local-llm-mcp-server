# FULL INTEGRATION (cross-category, n~15 held-out): native-KV read + RECALL-trained generalizing belief + KV-write
# actuation. (1) The Liquid's belief does AoA over the model's broad native KV (all full-attn layers), pretrained with
# CONTRASTIVE GOAL-RECALL (the generalization fix, not memorizable predict). (2) That belief -> KVGen -> (k,v) the model
# attends to. (3) In-loop distillation trains the actuation to make [window+KV-write] match [full-context]. EVAL: does the
# actuation lift agreement / reduce KL on UNSEEN categories? First reliable cross-category test of the whole pipeline.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, T, D, PROJ = 3, 2.0, 256, 768
TGT = [23, 27, 31]; MINJ, GK0, GV0 = 4, 64.0, 8.0; SMOKE = os.environ.get('SMOKE', '0') == '1'; RECALL_EP = 20 if SMOKE else 120
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
print('loading 60-traj data ...', flush=True)
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data'] if len(m['texts']) >= 10]
d_m = data[0]['gen'][0].shape[1]; nkv_raw = data[0]['nkv'][0].shape[1]
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
gR = torch.Generator().manual_seed(11); Rp = F.normalize(torch.randn(nkv_raw, PROJ, generator=gR), dim=0)
for m in data:
    m['z'] = [F.normalize((c - MU).mean(0), dim=0) for c in m['gen']]
    m['g'] = F.normalize(torch.stack(m['z'][:3]).mean(0), dim=0)
    m['perc'] = [c @ Rp for c in m['nkv']]; m['nkv'] = None                       # native-KV read (projected); free the raw
MUk = torch.cat([c for m in data for c in m['perc']], 0).mean(0)
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('FULL INTEGRATION | trajs=%d train=%d test=%d cats=%d held-out=%s | native-read dim=%d->%d' % (len(data), len(tr), len(te), len(fids), sorted(hold), nkv_raw, PROJ), flush=True)
class Comp(nn.Module):                                                           # Liquid + AoA over native KV; recall head (goal retention)
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
print('recall-pretraining native-KV compressor (CPU) ...', flush=True)
comp = Comp(PROJ, D); copt = torch.optim.Adam(comp.parameters(), lr=1e-3, weight_decay=1e-4)
for ep in range(RECALL_EP):
    HS = [comp.beliefs(m) for m in tr]; GP = torch.stack([comp.goalp(m['g']) for m in tr]); loss = 0.0; cnt = 0
    for mi in range(len(tr)):
        for t in range(3, HS[mi].shape[0]):
            r = comp.recall(HS[mi][t]); loss = loss + F.cross_entropy(((r @ GP.t()) / 0.1).unsqueeze(0), torch.tensor([mi])); cnt += 1
    copt.zero_grad(); (loss / cnt).backward(); copt.step()
with torch.no_grad():
    GPt = torch.stack([comp.goalp(m['g']) for m in te]); cor = 0; tot = 0; rr = 0.0
    for mi, m in enumerate(te):
        hs = comp.beliefs(m)
        for t in range(hs.shape[0] // 2, hs.shape[0]):
            sc = comp.recall(hs[t]) @ GPt.t(); rank = int((sc > sc[mi]).sum()); cor += (rank == 0); rr += 1.0 / (rank + 1); tot += 1
    for m in data: m['hs'] = comp.beliefs(m).detach()
print('  native-KV recall belief: held-out recall acc %.3f MRR %.3f chance %.3f (n_te=%d) -- generalizing belief ready' % (cor / tot, rr / tot, 1.0 / len(te), len(te)), flush=True)
print('loading 27B ...', flush=True)
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
            k = F.normalize(s.k[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gk[str(L)]; v = F.normalize(s.v[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gv[str(L)]
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
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
def win_msgs(ms):
    w = ms[-W:]
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
print('KV-write distillation on the recall belief ...', flush=True)
for ep in range(1 if SMOKE else 4):
    tot = 0.0; n = 0
    for m in tr:
        for t in range(W, len(m['texts'])):
            fm = full_msgs(m, t); clear()
            with torch.no_grad(): tl = chunk_logits(fm, m['texts'][t])
            set_kv(m['hs'][t - W].to(dev)); sl = chunk_logits(win_msgs(fm), m['texts'][t])
            nn_ = min(sl.shape[0], tl.shape[0]); loss = F.kl_div(F.log_softmax(sl[:nn_] / T, -1), F.softmax(tl[:nn_] / T, -1), reduction='batchmean') * (T * T)
            opt.zero_grad(); loss.backward(); opt.step(); clear(); tot += float(loss); n += 1
            if SMOKE and n >= 3: break
        if SMOKE: break
    print('  distill ep %d kl-loss %.3f (n=%d)' % (ep, tot / max(n, 1), n), flush=True)
print('\n=== CROSS-CATEGORY ACTUATION (held-out %d categories, n_te=%d trajs) ===' % (len(hold), len(te)), flush=True)
aw, al, klw, kll = [], [], [], []
with torch.no_grad():
    for m in te:
        for t in range(W, len(m['texts'])):
            fm = full_msgs(m, t); wm = win_msgs(fm)
            clear(); tl = chunk_logits(fm, m['texts'][t]); wl = chunk_logits(wm, m['texts'][t]); set_kv(m['hs'][t - W].to(dev)); ll = chunk_logits(wm, m['texts'][t]); clear()
            nn_ = min(tl.shape[0], wl.shape[0], ll.shape[0]); tt = tl[:nn_]; ww = wl[:nn_]; lll = ll[:nn_]; tg = tt.argmax(-1)
            aw.append(float((ww.argmax(-1) == tg).float().mean())); al.append(float((lll.argmax(-1) == tg).float().mean()))
            tp = F.softmax(tt, -1); klw.append(float(F.kl_div(F.log_softmax(ww, -1), tp, reduction='batchmean'))); kll.append(float(F.kl_div(F.log_softmax(lll, -1), tp, reduction='batchmean')))
            if SMOKE: break
        if SMOKE: break
print('  agreement w/ full-context:  window-alone %.3f -> window+native-actuator %.3f   (lift %+.3f)' % (st.mean(aw), st.mean(al), st.mean(al) - st.mean(aw)), flush=True)
print('  KL(full||window) %.3f -> KL(full||window+actuator) %.3f   (reduction %+.3f)   [n=%d chunks, UNSEEN categories]' % (st.mean(klw), st.mean(kll), st.mean(klw) - st.mean(kll), len(aw)), flush=True)
torch.save({'gen': gen.state_dict(), 'comp': comp.state_dict(), 'Rp': Rp, 'tgt': TGT}, '/home/pokazge/checkpoints/full_integration.pt')
print('=== ALL_DONE ===', flush=True)
