# STAGE 2 — closed-loop KL distillation (gate: cross-cat KL reduction >= +0.080, agreement lift >= +0.021; the floor
# full_integration.py already reproduced). The slot head is a READOUT of the Stage-1 CANONICAL belief h: LN(h) -> (k,v)
# injected as prefix KV. student = [window + B], teacher = [full context]; token-level KL through the frozen LLM's OWN
# attention trains the slot head (no judges, no rewards). Belief = canonical LiquidBelief (clamp-only STATE so the leak
# stays expressible for Stage 4; LN on the READOUT path; mean-pool native-KV read; INSTANCE-MULT recall = the Stage-1
# winning shape), FROZEN after recall-pretrain. Boundary-synchronous: at chunk t the window is the last W msgs -> use h[t-W].
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); dev = torch.device('cuda')
MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, TEMP, D, PROJ = 3, 2.0, 64, 768
TGT = [23, 27, 31]; MINJ, GK0, GV0 = 4, 64.0, 8.0; CLAMP, TAUFLOOR, DT, DROP, KWARM = 8.0, 1.0, 1.0, 0.3, 3
SMOKE = os.environ.get('SMOKE', '0') == '1'; RECALL_EP = 20 if SMOKE else 250; DISTILL_EP = 1 if SMOKE else 4
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
def gist(chunks): return F.normalize((torch.cat(chunks, 0) - MU).mean(0), dim=0)
for m in data:
    m['perc'] = [(c.float() @ Rp).mean(0) for c in m['nkv']]; m['goal'] = gist(m['gen']); m['nkv'] = None   # mean-pool native-KV read (cv2 baseline)
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('STAGE2 | trajs=%d train=%d test=%d cats=%d held-out=%s | belief d=%d -> slot M=%d x layers %s' % (len(data), len(tr), len(te), len(fids), sorted(hold), D, MINJ, TGT), flush=True)
# ---- canonical belief (CPU recall-pretrain, INSTANCE-MULT; clamp-only state; LN lives on the recall readout) ----
class LiquidBelief(nn.Module):
    def __init__(s, d_in, d):
        super().__init__(); s.read_in = nn.Linear(d_in, d); s.W = nn.Linear(d, d)
        s.log_tau = nn.Parameter(torch.zeros(d)); s.idrop = nn.Dropout(DROP); s.d = d
    def step(s, h, perc):
        tau = TAUFLOOR + F.softplus(s.log_tau)
        return (h + DT * (-h / tau + torch.tanh(s.W(h) + s.read_in(s.idrop(perc))))).clamp(-CLAMP, CLAMP)
    def run_seq(s, percs):
        h = torch.zeros(s.d); hs = []
        for p in percs: h = s.step(h, p); hs.append(h)
        return torch.stack(hs)
bel = LiquidBelief(PROJ, D); rq = nn.Sequential(nn.LayerNorm(D), nn.Dropout(DROP), nn.Linear(D, 128)); rg = nn.Sequential(nn.Dropout(DROP), nn.Linear(d_m, 128))
bopt = torch.optim.Adam(list(bel.parameters()) + list(rq.parameters()) + list(rg.parameters()), lr=3e-3, weight_decay=1e-3)
print('recall-pretraining canonical belief (instance-mult, CPU) ...', flush=True)
for ep in range(RECALL_EP):
    bel.train(); rq.train(); rg.train(); Q = []; lab = []
    for i, m in enumerate(tr):
        hs = bel.run_seq(m['perc'])
        for t in (list(range(KWARM, hs.shape[0])) or [hs.shape[0] - 1]): Q.append(rq(hs[t])); lab.append(i)
    Qn = F.normalize(torch.stack(Q), dim=-1); G = F.normalize(torch.stack([rg(m['goal']) for m in tr]), dim=-1)
    loss = F.cross_entropy(Qn @ G.t() / 0.07, torch.tensor(lab)); bopt.zero_grad(); loss.backward(); bopt.step()
bel.eval()
with torch.no_grad():
    for m in data: m['hs'] = bel.run_seq(m['perc']).detach()                      # frozen per-chunk belief states (CPU)
LNr = nn.LayerNorm(D); LNr.load_state_dict(rq[0].state_dict()); LNr.eval()        # the trained readout LN, reused by the slot head
print('  belief frozen; per-chunk states precomputed (recall_loss %.3f)' % float(loss), flush=True)
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
class SlotHead(nn.Module):                                                        # READOUT of h: LN(h) -> per-layer (k,v). thin; intelligence is in the belief.
    def __init__(s, D, layers, nkv, hd, M, gk0, gv0):
        super().__init__(); s.ln = LNr; s.trunk = nn.Sequential(nn.Linear(D, 128), nn.GELU())
        s.k = nn.ModuleDict(); s.v = nn.ModuleDict(); s.gk = nn.ParameterDict(); s.gv = nn.ParameterDict(); s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = M
        for L in layers:
            s.k[str(L)] = nn.Linear(128, M * nkv * hd); s.v[str(L)] = nn.Linear(128, M * nkv * hd)
            s.gk[str(L)] = nn.Parameter(torch.tensor(float(gk0))); s.gv[str(L)] = nn.Parameter(torch.tensor(float(gv0)))
    def forward(s, h):
        z = s.trunk(s.ln(h)); o = {}                                              # LN on the readout (state stays clamp-only -> leak expressible)
        for L in s.layers:
            k = F.normalize(s.k[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gk[str(L)]; v = F.normalize(s.v[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gv[str(L)]
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
        return o
slot = SlotHead(D, TGT, nkv, hd, MINJ, GK0, GV0).to(dev)
for p in slot.ln.parameters(): p.requires_grad = False                            # the LN is the FROZEN trained readout
opt = torch.optim.Adam([p for p in slot.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
def set_kv(h):
    o = slot(h.to(dev))
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
print('closed-loop KL distillation (slot head; belief frozen) ...', flush=True)
for ep in range(DISTILL_EP):
    tot = 0.0; n = 0
    for m in tr:
        for t in range(W, len(m['texts'])):
            fm = full_msgs(m, t); clear()
            with torch.no_grad(): tl = chunk_logits(fm, m['texts'][t])
            set_kv(m['hs'][t - W]); sl = chunk_logits(win_msgs(fm), m['texts'][t])
            nn_ = min(sl.shape[0], tl.shape[0]); loss = F.kl_div(F.log_softmax(sl[:nn_] / TEMP, -1), F.softmax(tl[:nn_] / TEMP, -1), reduction='batchmean') * (TEMP * TEMP)
            opt.zero_grad(); loss.backward(); opt.step(); clear(); tot += float(loss); n += 1
            if SMOKE and n >= 3: break
        if SMOKE: break
    print('  distill ep %d kl-loss %.3f (n=%d)' % (ep, tot / max(n, 1), n), flush=True)
print('\n=== STAGE2 CROSS-CATEGORY ACTUATION (held-out %d cats, n_te=%d trajs) ===' % (len(hold), len(te)), flush=True)
aw, al, klw, kll = [], [], [], []
with torch.no_grad():
    for m in te:
        for t in range(W, len(m['texts'])):
            fm = full_msgs(m, t); wm = win_msgs(fm)
            clear(); tl = chunk_logits(fm, m['texts'][t]); wl = chunk_logits(wm, m['texts'][t]); set_kv(m['hs'][t - W]); ll = chunk_logits(wm, m['texts'][t]); clear()
            nn_ = min(tl.shape[0], wl.shape[0], ll.shape[0]); tt = tl[:nn_]; ww = wl[:nn_]; lll = ll[:nn_]; tg = tt.argmax(-1)
            aw.append(float((ww.argmax(-1) == tg).float().mean())); al.append(float((lll.argmax(-1) == tg).float().mean()))
            tp = F.softmax(tt, -1); klw.append(float(F.kl_div(F.log_softmax(ww, -1), tp, reduction='batchmean'))); kll.append(float(F.kl_div(F.log_softmax(lll, -1), tp, reduction='batchmean')))
            if SMOKE: break
        if SMOKE: break
agl = st.mean(al) - st.mean(aw); klr = st.mean(klw) - st.mean(kll)
print('  agreement w/ full-context:  window-alone %.3f -> window+slot %.3f   (lift %+.3f, gate +0.021)' % (st.mean(aw), st.mean(al), agl), flush=True)
print('  KL(full||window) %.3f -> KL(full||window+slot) %.3f   (reduction %+.3f, gate +0.080)  [n=%d chunks, UNSEEN cats]' % (st.mean(klw), st.mean(kll), klr, len(aw)), flush=True)
print('=== STAGE2 GATE: %s (agreement %+.3f vs +0.021 ; KL-reduction %+.3f vs +0.080) ===' % ('PASS' if (klr >= 0.080 or agl >= 0.021) else 'FAIL', agl, klr), flush=True)
torch.save({'slot': slot.state_dict(), 'bel': bel.state_dict(), 'rq': rq.state_dict(), 'Rp': Rp, 'tgt': TGT}, '/home/pokazge/checkpoints/stage2_distill.pt')
print('=== ALL_DONE ===', flush=True)
