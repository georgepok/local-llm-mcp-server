# REALIZATION (forward-based, hang-proof): the Liquid's compression injected as a soft-prompt CAUSALLY moves the
# windowed LLM's representation toward the full-context state. Train a prefix-map (compression h -> k virtual tokens)
# so [prefix ; window] reaches the same layer-32 hidden as FULL context (backprop through the frozen 27B; forward only,
# no model.generate which hangs on this hybrid). Then the causal REALIZATION GAP on held-out: cos([prefix;window],full)
# - cos([window],full) > 0 means the Liquid's compression pulls the windowed LLM toward on-track — it CAUSES, not predicts.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, collections
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); M = '/home/pokazge/models/Qwen3.6-27B'; LAYER, K, W = 32, 4, 3
SMOKE = os.environ.get('SMOKE', '0') == '1'
data = torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data']
data = [m for m in data if len(m['texts']) >= 10]
d_m = data[0]['gen'][0].shape[1]; MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
def cen(C): return C - MU
for m in data: m['z'] = [F.normalize(cen(c).mean(0), dim=0) for c in m['gen']]
hold = {len(data) - 1}; tr = [m for i, m in enumerate(data) if i not in hold]; te = [m for i, m in enumerate(data) if i in hold]
print('trajs train=%d test=%d  K=%d W=%d smoke=%s' % (len(tr), len(te), K, W, SMOKE), flush=True)
class Compressor(nn.Module):                                                     # the validated Liquid compressor
    def __init__(s, d_m, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(d_m, heads * dh); s.Wv = nn.Linear(d_m, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K_ = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K_) / s.dh ** 0.5, dim=-1)
        return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
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
        pr, _ = comp.run(m); z = torch.stack(m['z']); loss = loss + (1 - (F.normalize(pr[:-1], -1) * z[1:]).sum(-1)).mean(); n += 1
    copt.zero_grad(); (loss / n).backward(); copt.step()
for m in data:
    with torch.no_grad(): _, hs = comp.run(m); m['hs'] = hs
D = comp.D
print('compressor trained; loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(M); cfg.language_model_only = True
tok = AutoTokenizer.from_pretrained(M)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
model.config.use_cache = False
try: model.gradient_checkpointing_enable()                                       # recompute activations in backward (fits 27B backprop in memory)
except Exception as e: print('grad-ckpt enable failed:', repr(e), flush=True)
EMB = model.get_input_embeddings(); enorm = EMB.weight.float().norm(dim=1).mean().item()
print('27B loaded %.1fGB' % (torch.cuda.memory_allocated() / 1e9), flush=True)
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
def hid(ms, prefix=None):
    ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    if prefix is None:
        return model(ids, output_hidden_states=True).hidden_states[LAYER][0, -1].float()
    emb = torch.cat([prefix.to(EMB.weight.dtype).unsqueeze(0), EMB(ids)], 1)
    return model(inputs_embeds=emb, output_hidden_states=True).hidden_states[LAYER][0, -1].float()
class PrefixMap(nn.Module):
    def __init__(s, D, k, d_m):
        super().__init__(); s.net = nn.Sequential(nn.Linear(D, 1024), nn.GELU(), nn.Linear(1024, k * d_m)); s.k = k; s.d_m = d_m; s.g = nn.Parameter(torch.tensor(0.3))
    def forward(s, h): p = s.net(h).view(s.k, s.d_m); return F.normalize(p, dim=-1) * enorm * s.g
pm = PrefixMap(D, K, d_m).to(dev); opt = torch.optim.Adam(pm.parameters(), lr=1e-3)
EP = 1 if SMOKE else 5
for ep in range(EP):                                                             # train prefix so [prefix;window] -> full-context hidden
    tot = 0.0; n = 0
    for m in tr:
        for t in range(W, len(m['texts'])):
            fm = full_msgs(m, t)
            with torch.no_grad(): th = hid(fm)
            prefix = pm(m['hs'][t - W].to(dev)); sh = hid(win_msgs(fm), prefix)
            loss = 1 - F.cosine_similarity(sh.unsqueeze(0), th.unsqueeze(0)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); n += 1
            if SMOKE and n >= 3: break
        if SMOKE: break
    print('prefix ep %d match-loss %.3f (n=%d)' % (ep, tot / max(n, 1), n), flush=True)
print('\n=== REALIZATION GAP (held-out): does the Liquid prefix CAUSALLY pull the windowed LLM toward full-context? ===', flush=True)
pos = collections.defaultdict(list)
with torch.no_grad():
    for m in te:
        for t in range(W, len(m['texts'])):
            fm = full_msgs(m, t); th = hid(fm); wh = hid(win_msgs(fm)); ih = hid(win_msgs(fm), pm(m['hs'][t - W].to(dev)))
            cw = float(F.cosine_similarity(wh.unsqueeze(0), th.unsqueeze(0))); ci = float(F.cosine_similarity(ih.unsqueeze(0), th.unsqueeze(0)))
            pos[t].append(ci - cw)
            if SMOKE: break
        if SMOKE: break
allg = [v for L in pos.values() for v in L]
print('  window-alone -> full   vs   window+LIQUID -> full   (gap = how much the prefix pulls toward full-context)')
print('  mean realization gap = %+.4f  over %d held-out positions' % (sum(allg) / max(len(allg), 1), len(allg)))
if not SMOKE:
    for t in sorted(pos): print('    t=%2d  gap=%+.4f' % (t, sum(pos[t]) / len(pos[t])))
torch.save({'pm': pm.state_dict(), 'comp': comp.state_dict()}, '/home/pokazge/checkpoints/realization.pt')
print('=== ALL_DONE ===')
