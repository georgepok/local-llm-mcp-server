# REALIZATION: inject the Liquid's compression back into the frozen LLM as a soft-prompt, so a WINDOWED LLM stays on
# track. In-process (no server): load 27B frozen, train a prefix-map (compression h -> k virtual-token embeddings) so
# that [prefix ; windowed-context] lands the LLM in the SAME layer-32 state as the FULL context (teacher). Then demo
# 3-way generation: windowed-alone (drifts) vs windowed+Liquid-prefix (on-track) vs full-context teacher. The Liquid
# stops PREDICTING on-track-ness and starts CAUSING it. SMOKE=1 runs a tiny config to validate plumbing first.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
torch.manual_seed(0); dev = torch.device('cuda')
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high')
M = '/home/pokazge/models/Qwen3.6-27B'; LAYER = 32; K = 8; W = 4              # k prefix tokens; window = last W messages
SMOKE = os.environ.get('SMOKE', '0') == '1'
data = torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data']
data = [m for m in data if len(m['texts']) >= 10]
if SMOKE: data = data[:4]
d_m = data[0]['gen'][0].shape[1]
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
def cen(C): return C - MU
for m in data: m['z'] = [F.normalize(cen(c).mean(0), dim=0) for c in m['gen']]
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(2, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('trajs train=%d test=%d  K=%d W=%d smoke=%s' % (len(tr), len(te), K, W, SMOKE), flush=True)

# ---- compressor: train on dev streams, produce running compression states h_t (CPU) ----
class Compressor(nn.Module):
    def __init__(s, d_m, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(d_m, heads * dh); s.Wv = nn.Linear(d_m, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K_ = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K_) / s.dh ** 0.5, dim=-1)
        return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def run(s, m):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; preds = []; hs = []
        for t in range(len(m['gen'])):
            hp = h; a = s.collect(cen(m['gen'][t]), b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; hs.append(h); preds.append(s.pred(torch.cat([s.cz(m['z'][t]), hp])))
        return torch.stack(preds), torch.stack(hs)
comp = Compressor(d_m); copt = torch.optim.Adam(comp.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(40 if SMOKE else 150):
    loss, n = 0.0, 0
    for m in tr:
        pr, _ = comp.run(m); z = torch.stack(m['z'])
        loss = loss + (1 - (F.normalize(pr[:-1], -1) * z[1:]).sum(-1)).mean(); n += 1
    copt.zero_grad(); (loss / n).backward(); copt.step()
print('compressor trained (loss %.3f)' % float(loss / n), flush=True)
for m in data:
    with torch.no_grad(): _, hs = comp.run(m); m['hs'] = hs                  # [T, D] running compression
D = comp.D

# ---- load 27B frozen ----
print('loading 27B (frozen, in-process) ...', flush=True)
cfg = AutoConfig.from_pretrained(M); cfg.language_model_only = True
tok = AutoTokenizer.from_pretrained(M)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
EMB = model.get_input_embeddings()
print('27B loaded %.1fGB' % (torch.cuda.memory_allocated() / 1e9), flush=True)
def tmpl(msgs):
    try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
def full_msgs(m, upto):
    ms = [{'role': 'user', 'content': m['seed']}]
    for i in range(upto): ms += [{'role': 'assistant', 'content': m['texts'][i]}, {'role': 'user', 'content': m['texts'][i]}]
    return ms
def win_msgs(ms):
    w = ms[-W:]
    while w and w[0]['role'] == 'assistant': w = w[1:]                        # window must start on a user turn
    return w or ms[-1:]
def teacher_hidden(ms):
    ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    with torch.no_grad(): return model(ids, output_hidden_states=True).hidden_states[LAYER][0, -1].float()
def prefixed_hidden(prefix, ms):
    ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    emb = torch.cat([prefix.to(EMB.weight.dtype).unsqueeze(0), EMB(ids)], 1)
    return model(inputs_embeds=emb, output_hidden_states=True).hidden_states[LAYER][0, -1].float()

# ---- prefix map: compression h -> k virtual tokens (scaled to embedding norm) ----
enorm = EMB.weight.float().norm(dim=1).mean().item()
class PrefixMap(nn.Module):
    def __init__(s, D, k, d_m):
        super().__init__(); s.net = nn.Sequential(nn.Linear(D, 1024), nn.GELU(), nn.Linear(1024, k * d_m)); s.k = k; s.d_m = d_m; s.g = nn.Parameter(torch.tensor(0.3))
    def forward(s, h): p = s.net(h).view(s.k, s.d_m); return F.normalize(p, dim=-1) * enorm * s.g
pm = PrefixMap(D, K, d_m).to(dev)
opt = torch.optim.Adam(pm.parameters(), lr=1e-3)
EPOCHS = 1 if SMOKE else 5
for ep in range(EPOCHS):
    tot, n = 0.0, 0
    for m in tr:
        T = len(m['texts'])
        for t in range(W, T):
            fm = full_msgs(m, t); th = teacher_hidden(fm)
            prefix = pm(m['hs'][t - W].to(dev))                              # compression of the DROPPED history
            sh = prefixed_hidden(prefix, win_msgs(fm))
            loss = 1 - F.cosine_similarity(sh.unsqueeze(0), th.unsqueeze(0)).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); n += 1
            if SMOKE and n >= 3: break
        if SMOKE: break
    print('inject ep %d  match-loss %.3f  (n=%d)' % (ep, tot / max(n, 1), n), flush=True)

# ---- baseline: how close is windowed-alone hidden to teacher, vs windowed+prefix (held-out) ----
@torch.no_grad()
def gen_ids(inp_ids=None, inp_emb=None, mx=44):
    kw = dict(max_new_tokens=mx, do_sample=False, pad_token_id=tok.pad_token_id)
    if inp_ids is not None: return model.generate(input_ids=inp_ids, attention_mask=torch.ones_like(inp_ids), **kw)
    return model.generate(inputs_embeds=inp_emb, attention_mask=torch.ones(inp_emb.shape[:2], dtype=torch.long, device=dev), **kw)
@torch.no_grad()
def gist_of(text, ctx_ms):
    ids = tok(tmpl(ctx_ms), return_tensors='pt').input_ids.to(dev)
    r = tok(text or '.', return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    hs = model(torch.cat([ids, r], 1), output_hidden_states=True).hidden_states[LAYER][0, ids.shape[1]:].float()
    return F.normalize((hs.cpu() - MU).mean(0), dim=0)
print('\n=== DEMO (held-out): does the Liquid prefix keep windowed generation ON TRACK with full context? ===', flush=True)
win_sims, inj_sims = [], []
for m in te:
    for t in range(W, len(m['texts']), max(1, (len(m['texts']) - W) // 3)):
        fm = full_msgs(m, t); wm = win_msgs(fm)
        fids_ = tok(tmpl(fm), return_tensors='pt').input_ids.to(dev)
        wids = tok(tmpl(wm), return_tensors='pt').input_ids.to(dev)
        with torch.no_grad():
            t_txt = tok.decode(gen_ids(inp_ids=fids_)[0, fids_.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()
            w_txt = tok.decode(gen_ids(inp_ids=wids)[0, wids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()
            prefix = pm(m['hs'][t - W].to(dev))
            iemb = torch.cat([prefix.to(EMB.weight.dtype).unsqueeze(0), EMB(wids)], 1)
            i_txt = tok.decode(gen_ids(inp_emb=iemb)[0], skip_special_tokens=True).split('</think>')[-1].strip()
        tz = gist_of(t_txt, fm)
        win_sims.append(float((gist_of(w_txt, wm) * tz).sum())); inj_sims.append(float((gist_of(i_txt, wm) * tz).sum()))
        if SMOKE: print('  t=%d\n   teacher: %s\n   window : %s\n   inject : %s' % (t, t_txt[:70], w_txt[:70], i_txt[:70]), flush=True)
        if SMOKE: break
    if SMOKE: break
import statistics as st
print('\non-track-ness = gist-cos of generated chunk to the FULL-CONTEXT (teacher) chunk:')
print('  windowed-alone : %+.3f' % (sum(win_sims) / len(win_sims)))
print('  windowed+LIQUID: %+.3f' % (sum(inj_sims) / len(inj_sims)))
print('  injection lift : %+.3f  (>0 = the Liquid prefix pulls windowed generation back toward full-context behavior)' % (sum(inj_sims) / len(inj_sims) - sum(win_sims) / len(win_sims)))
torch.save({'pm': pm.state_dict(), 'comp': comp.state_dict()}, '/home/pokazge/checkpoints/inject_realize.pt')
print('=== ALL_DONE ===')
