# IN-LOOP teacher-forced distillation of the dynamic-LoRA actuator. At EVERY step of the saved self-feeding trajectory
# (the teacher's loop), match the [window + Liquid-LoRA] model's NEXT-TOKEN DISTRIBUTION to the [full-context] model's
# (full-vocab KL, temperature T, differentiable — before the non-diff sample). Trains the actuator inside the loop on the
# actual generation behavior, not a single hidden vector. Eval (held-out): next-token AGREEMENT with full-context
# (window+LoRA vs window-alone) and KL-divergence reduction. Multi-layer LoRA on o_proj. SMOKE=1 tiny. (Closed-loop RL is
# the next rung; this is the differentiable in-loop rung.)
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, collections
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); M = '/home/pokazge/models/Qwen3.6-27B'; LAYER, K, W, T = 32, 2, 3, 2.0
TGT_LAYERS = [int(x) for x in os.environ.get('TGT_LAYERS', '23,27,31').split(',')]; SMOKE = os.environ.get('SMOKE', '0') == '1'
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data'] if len(m['texts']) >= 10]
d_m = data[0]['gen'][0].shape[1]; MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0)
def cen(C): return C - MU
for m in data: m['z'] = [F.normalize(cen(c).mean(0), dim=0) for c in m['gen']]
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('trajs train=%d test=%d K=%d W=%d TGT=%s T=%.1f smoke=%s' % (len(tr), len(te), K, W, TGT_LAYERS, T, SMOKE), flush=True)
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
        pr, _ = comp.run(m); z = torch.stack(m['z']); loss = loss + (1 - (F.normalize(pr[:-1], -1) * z[1:]).sum(-1)).mean(); n += 1
    copt.zero_grad(); (loss / n).backward(); copt.step()
for m in data:
    with torch.no_grad(): _, hs = comp.run(m); m['hs'] = hs
D = comp.D; print('compressor trained; loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(M); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(M)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
model.config.use_cache = False
try: model.gradient_checkpointing_enable()
except Exception as e: print('gc fail', repr(e), flush=True)
oprojs = {}; dims = {}
for L in TGT_LAYERS:
    op = model.get_submodule('model.layers.%d.self_attn.o_proj' % L); op._A = None; op._B = None; op._s = 1.0; oprojs[L] = op; dims[L] = (op.weight.shape[1], op.weight.shape[0])
    def mk(mod):
        def hook(m_, inp, out): return out if m_._A is None else out + m_._s * F.linear(F.linear(inp[0], m_._A), m_._B)
        return hook
    op.register_forward_hook(mk(op))
print('27B loaded %.1fGB; LoRA targets=%s' % (torch.cuda.memory_allocated() / 1e9, TGT_LAYERS), flush=True)
class MultiGen(nn.Module):
    def __init__(s, D, K, dims):
        super().__init__(); s.trunk = nn.Sequential(nn.Linear(D, 128), nn.GELU()); s.hA = nn.ModuleDict(); s.hB = nn.ModuleDict(); s.g = nn.ParameterDict(); s.K = K; s.dims = dims
        for L, (IN, OUT) in dims.items():
            s.hA[str(L)] = nn.Linear(128, K * IN); s.hB[str(L)] = nn.Linear(128, OUT * K); s.g[str(L)] = nn.Parameter(torch.tensor(0.5))
    def forward(s, h):
        z = s.trunk(h); o = {}
        for L, (IN, OUT) in s.dims.items(): o[L] = (s.hA[str(L)](z).view(s.K, IN) * 0.02, s.hB[str(L)](z).view(OUT, s.K) * 0.02, s.g[str(L)])
        return o
gen = MultiGen(D, K, dims).to(dev); opt = torch.optim.Adam(gen.parameters(), lr=1e-3, weight_decay=1e-4)
def set_lora(h):
    o = gen(h)
    for L in TGT_LAYERS: A, B, g = o[L]; oprojs[L]._A = A.to(model.dtype); oprojs[L]._B = B.to(model.dtype); oprojs[L]._s = g
def clear():
    for L in TGT_LAYERS: oprojs[L]._A = None
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
def chunk_logits(ctx_msgs, chunk_text):                                          # next-token logits at the chunk positions (teacher-forced)
    cids = tok(tmpl(ctx_msgs), return_tensors='pt').input_ids.to(dev); rids = tok(chunk_text or '.', return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    ids = torch.cat([cids, rids], 1); lg = model(ids).logits[0]; st = cids.shape[1] - 1
    return lg[st:st + rids.shape[1]].float()                                     # [Lc, vocab]
EP = 1 if SMOKE else 4
for ep in range(EP):
    tot = 0.0; n = 0
    for m in tr:
        for t in range(W, len(m['texts'])):
            fm = full_msgs(m, t)
            clear()
            with torch.no_grad(): tl = chunk_logits(fm, m['texts'][t])           # teacher: full-context next-token dist
            set_lora(m['hs'][t - W].to(dev)); sl = chunk_logits(win_msgs(fm), m['texts'][t])   # student: window + LoRA
            n_ = min(sl.shape[0], tl.shape[0])
            loss = F.kl_div(F.log_softmax(sl[:n_] / T, -1), F.softmax(tl[:n_] / T, -1), reduction='batchmean') * (T * T)
            opt.zero_grad(); loss.backward(); opt.step(); clear(); tot += float(loss); n += 1
            if SMOKE and n >= 3: break
        if SMOKE: break
    print('inloop ep %d kl-loss %.3f (n=%d)' % (ep, tot / max(n, 1), n), flush=True)
print('\n=== IN-LOOP REALIZATION (held-out): next-token AGREEMENT with full-context, and KL reduction ===', flush=True)
aw, al, klw, kll = [], [], [], []
with torch.no_grad():
    for m in te:
        for t in range(W, len(m['texts'])):
            fm = full_msgs(m, t); wm = win_msgs(fm)
            clear(); tl = chunk_logits(fm, m['texts'][t]); wl = chunk_logits(wm, m['texts'][t]); set_lora(m['hs'][t - W].to(dev)); ll = chunk_logits(wm, m['texts'][t]); clear()
            n_ = min(tl.shape[0], wl.shape[0], ll.shape[0]); tt = tl[:n_]; ww = wl[:n_]; lll = ll[:n_]; tgt = tt.argmax(-1)
            aw.append(float((ww.argmax(-1) == tgt).float().mean())); al.append(float((lll.argmax(-1) == tgt).float().mean()))
            tp = F.softmax(tt, -1); klw.append(float(F.kl_div(F.log_softmax(ww, -1), tp, reduction='batchmean'))); kll.append(float(F.kl_div(F.log_softmax(lll, -1), tp, reduction='batchmean')))
            if SMOKE: break
        if SMOKE: break
import statistics as st
print('  next-token agreement w/ full-context:  window-alone %.3f   window+LoRA %.3f   (lift %+.3f)' % (st.mean(aw), st.mean(al), st.mean(al) - st.mean(aw)))
print('  KL(full || window) %.3f  ->  KL(full || window+LoRA) %.3f   (reduction %+.3f)' % (st.mean(klw), st.mean(kll), st.mean(klw) - st.mean(kll)))
torch.save({'gen': gen.state_dict(), 'comp': comp.state_dict(), 'tgt': TGT_LAYERS}, '/home/pokazge/checkpoints/lora_inloop.pt')
print('=== ALL_DONE ===')
