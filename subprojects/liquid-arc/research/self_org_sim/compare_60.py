# RELIABLE comparison on 60 trajectories (~15 held-out, cross-category) — fixes the n=2 underpowered eval. Trains the SAME
# Liquid+AoA compressor + KV-write actuator on TWO read channels and evals both on the same split: (A) layer-32 stream
# (the original tap), (B) broad native KV (all full-attn layers — the elegant native read). Same target (next-gist z),
# same window, same distillation, same held-out. The only difference is what the Liquid attends over. Now n is big enough
# for the agreement/KL-reduction differences to mean something.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, T, D = 3, 2.0, 384
TGT = [23, 27, 31]; MINJ, GK0, GV0 = 4, 64.0, 8.0; SMOKE = os.environ.get('SMOKE', '0') == '1'
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
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data'] if len(m['texts']) >= 10]
d_m = data[0]['gen'][0].shape[1]; nkv_dim = data[0]['nkv'][0].shape[1]
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0); MUk = torch.cat([c for m in data for c in m['nkv']], 0).mean(0)
for m in data: m['z'] = [F.normalize((c - MU).mean(0), dim=0) for c in m['gen']]
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('RELIABLE compare | trajs=%d train=%d test=%d (held-out fids=%s) | layer32 dim=%d native dim=%d' % (len(data), len(tr), len(te), sorted(hold), d_m, nkv_dim), flush=True)
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
def chunk_logits(ctx_msgs, chunk_text):
    cids = tok(tmpl(ctx_msgs), return_tensors='pt').input_ids.to(dev); rids = tok(chunk_text or '.', return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    ids = torch.cat([cids, rids], 1); lg = model(ids).logits[0]; st_ = cids.shape[1] - 1
    return lg[st_:st_ + rids.shape[1]].float()
class Compressor(nn.Module):                                                     # Liquid + attention-on-attention; reads whatever stream `key` names
    def __init__(s, in_dim, mu, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh; s.register_buffer('mu', mu)
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(in_dim, heads * dh); s.Wv = nn.Linear(in_dim, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K_ = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K_) / s.dh ** 0.5, dim=-1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def run(s, m, key):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; pr = []; hs = []
        for t in range(len(m[key])):
            hp = h; a = s.collect(m[key][t] - s.mu, b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; hs.append(h); pr.append(s.pred(torch.cat([s.cz(m['z'][t]), hp])))
        return torch.stack(pr), torch.stack(hs)
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
def clear():
    for L in TGT: mods[L]._kv_inj = None
def run_channel(key, in_dim, mu, label):
    print('\n######## CHANNEL: %s (read dim=%d) ########' % (label, in_dim), flush=True)
    comp = Compressor(in_dim, mu, D); copt = torch.optim.Adam(comp.parameters(), lr=1e-3, weight_decay=1e-5)
    for ep in range(40 if SMOKE else 150):
        loss = 0.0; n = 0
        for m in tr:
            pr, _ = comp.run(m, key); z = torch.stack(m['z']); loss = loss + (1 - (F.normalize(pr[:-1], dim=-1) * z[1:]).sum(-1)).mean(); n += 1
        copt.zero_grad(); (loss / n).backward(); copt.step()
    with torch.no_grad():
        gp = [float((F.normalize(comp.run(m, key)[0][:-1], dim=-1) * torch.stack(m['z'])[1:]).sum(-1).mean()) for m in te]
        for m in data: _, hs = comp.run(m, key); m['hs'] = hs.detach()
    print('  %s compressor pretrained; held-out next-gist coh %.3f (n=%d trajs)' % (label, st.mean(gp), len(te)), flush=True)
    gen = KVGen(D, TGT, nkv, hd, MINJ, GK0, GV0).to(dev); opt = torch.optim.Adam(gen.parameters(), lr=1e-3, weight_decay=1e-4)
    def set_kv(h):
        o = gen(h)
        for L in TGT: ki, vi = o[L]; mods[L]._kv_inj = (ki.to(model.dtype), vi.to(model.dtype))
    for ep in range(1 if SMOKE else 4):
        tot = 0.0; n = 0
        for m in tr:
            for t in range(W, len(m['texts'])):
                fm = full_msgs(m, t); clear()
                with torch.no_grad(): tl = chunk_logits(fm, m['texts'][t])
                set_kv(m['hs'][t - W].to(dev)); sl = chunk_logits(win_msgs(fm), m['texts'][t])
                n_ = min(sl.shape[0], tl.shape[0]); loss = F.kl_div(F.log_softmax(sl[:n_] / T, -1), F.softmax(tl[:n_] / T, -1), reduction='batchmean') * (T * T)
                opt.zero_grad(); loss.backward(); opt.step(); clear(); tot += float(loss); n += 1
                if SMOKE and n >= 3: break
            if SMOKE: break
        print('  %s ep %d kl-loss %.3f (n=%d)' % (label, ep, tot / max(n, 1), n), flush=True)
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
    print('  >> %s: agreement window %.3f -> +actuator %.3f (lift %+.3f) | KL %.3f -> %.3f (reduction %+.3f) | n=%d chunks' % (label, st.mean(aw), st.mean(al), st.mean(al) - st.mean(aw), st.mean(klw), st.mean(kll), st.mean(klw) - st.mean(kll), len(aw)), flush=True)
    return (st.mean(al) - st.mean(aw), st.mean(klw) - st.mean(kll))
r32 = run_channel('gen', d_m, MU, 'LAYER-32 tap')
rnat = run_channel('nkv', nkv_dim, MUk, 'NATIVE broad-KV')
print('\n=== RELIABLE VERDICT (%d held-out trajectories) ===' % len(te), flush=True)
print('  LAYER-32 tap   : agreement lift %+.3f   KL reduction %+.3f' % r32, flush=True)
print('  NATIVE broad-KV: agreement lift %+.3f   KL reduction %+.3f' % rnat, flush=True)
print('=== ALL_DONE ===', flush=True)
