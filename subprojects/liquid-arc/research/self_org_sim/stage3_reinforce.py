# STAGE 3 — closed-loop REINFORCE on the slot head (the +0.41-class rung; KNOWN UNSTABLE -> runs LAST, SHORTEST,
# warm-started, stop-on-first-degradation). Warm-start = stage2_distill.pt (belief FROZEN, slot KL-distilled, gates
# passed: agr +0.026 / KL-red +0.114). Closed loop: the windowed model SELF-FEEDS with the slot injected; the belief
# steps boundary-synchronously from the LIVE native KV of its own actuated output; slot at chunk t injects h[t-W].
# Reward = full-context likelihood of the SAMPLED chunk (the teacher judging on-track-ness; no injection, no grad).
# Per-step REINFORCE -> slot head ONLY. SHORT BURSTS (NEP episodes) with the Stage-2 gate eval (subsampled) between
# bursts: if KL-reduction falls > DEGRADE below the running best -> restore best, STOP. Checkpoint EVERY burst from 0.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); dev = torch.device('cuda')
MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, D, PROJ = 3, 64, 768
CLAMP, TAUFLOOR, DT = 8.0, 1.0, 1.0; TEMP, MAXNEW = 0.8, 40
SMOKE = os.environ.get('SMOKE', '0') == '1'
NEST = 4 if SMOKE else 8                                                          # chunks/episode (first W are warmup, rest trained)
NEP = 2 if SMOKE else 10                                                          # episodes/burst
MAXBURST = 1 if SMOKE else 4; DEGRADE = 0.03; LR = 3e-4
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
ck = torch.load('/home/pokazge/checkpoints/stage2_distill.pt', weights_only=False, map_location='cpu')
TGT = ck['tgt']; Rp = ck['Rp']
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data'] if len(m['texts']) >= 10]
fids = sorted(set(m['fid'] for m in data)); hold = set(fids[-max(1, len(fids) // 4):])
tr = [m for m in data if m['fid'] not in hold]; te = [m for m in data if m['fid'] in hold]
print('STAGE3 REINFORCE | warm=stage2_distill.pt | train=%d test=%d | bursts<=%d x %d eps x %d chunks | degrade-stop %.3f' % (len(tr), len(te), MAXBURST, NEP, NEST, DEGRADE), flush=True)
class LiquidBelief(nn.Module):                                                    # canonical belief (FROZEN; clamp-only state, leak expressible)
    def __init__(s, d_in, d):
        super().__init__(); s.read_in = nn.Linear(d_in, d); s.W = nn.Linear(d, d)
        s.log_tau = nn.Parameter(torch.zeros(d)); s.idrop = nn.Dropout(0.3); s.d = d
    def step(s, h, perc):
        tau = TAUFLOOR + F.softplus(s.log_tau)
        return (h + DT * (-h / tau + torch.tanh(s.W(h) + s.read_in(perc)))).clamp(-CLAMP, CLAMP)
bel = LiquidBelief(PROJ, D); bel.load_state_dict(ck['bel']); bel.eval()
for p in bel.parameters(): p.requires_grad = False
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
try: model.gradient_checkpointing_enable()
except Exception as e: print('gc fail', repr(e), flush=True)
nkv = model.config.num_key_value_heads; hd = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
@torch.no_grad()
def probe_full():
    model.config.use_cache = True; ids = tok('hi', return_tensors='pt').input_ids.to(dev); out = model(ids, use_cache=True)
    return [i for i, L in enumerate(out.past_key_values.layers) if getattr(L, 'keys', None) is not None]
FULL = probe_full(); mods = {L: model.model.layers[L].self_attn for L in TGT}
for sa in mods.values(): sa._kv_inj = None
class SlotHead(nn.Module):                                                        # readout of h (warm from Stage 2); the ONLY trainable thing
    def __init__(s, D, layers, nkv, hd, M=4):
        super().__init__(); s.ln = nn.LayerNorm(D); s.trunk = nn.Sequential(nn.Linear(D, 128), nn.GELU())
        s.k = nn.ModuleDict(); s.v = nn.ModuleDict(); s.gk = nn.ParameterDict(); s.gv = nn.ParameterDict(); s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = M
        for L in layers:
            s.k[str(L)] = nn.Linear(128, M * nkv * hd); s.v[str(L)] = nn.Linear(128, M * nkv * hd)
            s.gk[str(L)] = nn.Parameter(torch.tensor(64.0)); s.gv[str(L)] = nn.Parameter(torch.tensor(8.0))
    def forward(s, h):
        z = s.trunk(s.ln(h)); o = {}
        for L in s.layers:
            k = F.normalize(s.k[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gk[str(L)]; v = F.normalize(s.v[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * s.gv[str(L)]
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
        return o
slot = SlotHead(D, TGT, nkv, hd).to(dev); slot.load_state_dict(ck['slot'])
for p in slot.ln.parameters(): p.requires_grad = False                            # frozen trained readout LN (as in Stage 2)
opt = torch.optim.Adam([p for p in slot.parameters() if p.requires_grad], lr=LR)
def set_kv(h):
    o = slot(h.to(dev))
    for L in TGT: ki, vi = o[L]; mods[L]._kv_inj = (ki.to(model.dtype), vi.to(model.dtype))
def clear():
    for L in TGT: mods[L]._kv_inj = None
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def win_of(ms):
    w = ms[-W:]
    while w and w[0]['role'] == 'assistant': w = w[1:]
    return w or ms[-1:]
def full_msgs(m, t):
    ms = [{'role': 'user', 'content': m['seed']}]
    for i in range(t): ms += [{'role': 'assistant', 'content': m['texts'][i]}, {'role': 'user', 'content': m['texts'][i]}]
    return ms
@torch.no_grad()
def sample_chunk(ms):
    model.config.use_cache = True; ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    return o[0, ids.shape[1]:]
def lp_chunk(cm, rids, grad):                                                     # mean logp of chunk tokens under ctx (logits-only; grad iff grad)
    cids = tok(tmpl(cm), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); s0 = cids.shape[1] - 1; nct = rids.shape[0]
    cmx = torch.enable_grad() if grad else torch.no_grad()
    with cmx:
        model.config.use_cache = False
        lp = F.log_softmax(model(ids).logits[0, s0:s0 + nct].float(), -1).gather(1, rids.unsqueeze(1)).mean()
    return lp
@torch.no_grad()
def capture_perc(cm, rids):                                                       # live native-KV read of [win+chunk] -> perc [PROJ]
    model.config.use_cache = True
    cids = tok(tmpl(cm), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); nct = rids.shape[0]
    out = model(ids, use_cache=True); feats = []
    for L in FULL:
        lc = out.past_key_values.layers[L]; feats.append(lc.keys[0, :, -nct:, :].mean(0)); feats.append(lc.values[0, :, -nct:, :].mean(0))
    return (torch.cat(feats, -1).float().cpu() @ Rp).mean(0)
def dec(ids): return tok.decode(ids, skip_special_tokens=True).split('</think>')[-1].strip()
def episode(m, base, train):                                                      # one closed-loop self-feed; per-step REINFORCE; returns mean reward
    hist = [{'role': 'user', 'content': m['seed']}]; h = torch.zeros(D); hq = [h]; rs = []
    for t in range(NEST):
        if t >= W: set_kv(hq[t - W + 1])                                          # hq[j+1] = state after chunk j -> matches Stage-2's hs[t-W]
        else: clear()
        rids = sample_chunk(win_of(hist))
        if t >= W:
            lp_act = lp_chunk(win_of(hist), rids, train)                          # actuated logp (grad -> slot through the injection)
            clear(); r = float(lp_chunk(hist, rids, False))                       # REWARD: full self-generated-context likelihood
            if train and torch.isfinite(lp_act):
                opt.zero_grad(); (-(r - base) * lp_act).backward()
                torch.nn.utils.clip_grad_norm_([p for p in slot.parameters() if p.requires_grad], 1.0); opt.step()
            rs.append(r)
        clear(); perc = capture_perc(win_of(hist), rids)                          # belief steps from LIVE actuated output (boundary-sync)
        h = bel.step(h, perc).detach(); hq.append(h)
        tx = dec(rids); hist += [{'role': 'assistant', 'content': tx}, {'role': 'user', 'content': tx}]
    return st.mean(rs) if rs else 0.0
@torch.no_grad()
def gate_eval():                                                                  # subsampled Stage-2 gate (held-out, recorded): KL-reduction + agreement lift
    aw, al, klw, kll = [], [], [], []
    for m in te:
        hs = torch.zeros(D); hh = [hs]
        for p in m['perc_pre']: hs = bel.step(hs, p); hh.append(hs)               # belief prefix states once per traj (recorded percs)
        for t in range(W, len(m['texts']), 3 if not SMOKE else 11):
            fm = full_msgs(m, t); wm = win_of(fm)
            rids = tok(m['texts'][t] or '.', return_tensors='pt', add_special_tokens=False).input_ids[0].to(dev)
            def lgts(cm):
                model.config.use_cache = False
                cids = tok(tmpl(cm), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); s0 = cids.shape[1] - 1
                return model(ids).logits[0, s0:s0 + rids.shape[0]].float()
            clear(); tl = lgts(fm); wl = lgts(wm)
            set_kv(hh[t - W + 1]); ll = lgts(wm); clear()                         # hh[j+1] = state after chunk j -> matches Stage-2's hs[t-W]
            tg = tl.argmax(-1); tp = F.softmax(tl, -1)
            aw.append(float((wl.argmax(-1) == tg).float().mean())); al.append(float((ll.argmax(-1) == tg).float().mean()))
            klw.append(float(F.kl_div(F.log_softmax(wl, -1), tp, reduction='batchmean'))); kll.append(float(F.kl_div(F.log_softmax(ll, -1), tp, reduction='batchmean')))
    return st.mean(al) - st.mean(aw), st.mean(klw) - st.mean(kll), len(aw)
for m in data: m['perc_pre'] = [(c.float() @ Rp).mean(0) for c in m['nkv']]; m['nkv'] = None
agl0, klr0, nev = gate_eval()
best = {'klr': klr0, 'burst': 0, 'state': {k: v.detach().cpu().clone() for k, v in slot.state_dict().items()}}
torch.save({'slot': best['state'], 'burst': 0, 'klr': klr0, 'tgt': TGT}, '/home/pokazge/checkpoints/stage3_best.pt')
print('burst 0 (warm-start) | gate: agr_lift %+.3f  KL-red %+.3f  (n=%d) -- the do-not-degrade baseline' % (agl0, klr0, nev), flush=True)
base = None
for b in range(1, MAXBURST + 1):
    rws = []
    for e in range(NEP):
        m = tr[(b * NEP + e) % len(tr)]
        r = episode(m, base if base is not None else 0.0, True)
        base = r if base is None else 0.9 * base + 0.1 * r; rws.append(r)
        print('  burst %d ep %d reward(full-ctx logp) %+.3f (ema %+.3f)' % (b, e, r, base), flush=True)
    agl, klr, _ = gate_eval()
    print('burst %d | mean reward %+.3f | gate: agr_lift %+.3f  KL-red %+.3f  (best %+.3f@b%d)' % (b, st.mean(rws), agl, klr, best['klr'], best['burst']), flush=True)
    torch.save({'slot': {k: v.detach().cpu().clone() for k, v in slot.state_dict().items()}, 'burst': b, 'klr': klr, 'tgt': TGT}, '/home/pokazge/checkpoints/stage3_burst%d.pt' % b)
    if klr > best['klr']: best = {'klr': klr, 'burst': b, 'state': {k: v.detach().cpu().clone() for k, v in slot.state_dict().items()}}
    if klr < best['klr'] - DEGRADE:
        print('=== DEGRADATION (%.3f < best %.3f - %.2f) -> restore best (burst %d), STOP ===' % (klr, best['klr'], DEGRADE, best['burst']), flush=True)
        slot.load_state_dict(best['state']); break
torch.save({'slot': best['state'], 'burst': best['burst'], 'klr': best['klr'], 'tgt': TGT}, '/home/pokazge/checkpoints/stage3_best.pt')
print('=== STAGE3 | best KL-red %+.3f @burst %d (warm-start was %+.3f) -> %s ===' % (best['klr'], best['burst'], klr0, 'IMPROVED' if best['klr'] > klr0 + 0.005 else 'HELD (no gain over Stage 2)'), flush=True)
print('=== ALL_DONE ===', flush=True)
