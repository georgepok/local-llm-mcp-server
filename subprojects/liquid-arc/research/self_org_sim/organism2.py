# ORGANISM v2 — SELF-FORMED PURPOSE + FORMED INTENSITY. No external goal, no recall target, no fixed gain anywhere.
# The entity forms EVERYTHING and is selected by one thing — viability:
#   PURPOSE  = its own SLOW-mode direction (committed by the tau spectrum: slow dims change slowly, so a purpose is a
#              direction HELD against the LLM's drift; it cannot collapse to "whatever I just output").
#   INTENSITY= gain_mult = softplus(gain_head(h)), a per-moment READOUT the entity sets — NOT a scalar, NOT f(health).
#   FEED     = self_advance x coherence:  self_advance = cos(perceived-output-direction, the standing purpose) [does my
#              output CONTINUE my own direction?] ; coherence = on the ACTUAL generated text [am I still intelligible?].
#   Too-strong gain -> garbage -> coherence collapses -> starve.  Too-weak -> ignored -> self_advance collapses -> starve.
#   So intensity self-organizes to the CRITICAL EDGE (max coherent influence); purpose is what the entity can coherently
#   maintain against drift; identity/defense are emergent (dissolving = starving). REINFORCE on feed forms gain_head +
#   slot + belief. Embryo (1c belief + 2c slot) = sensorimotor priors ONLY (perception map + actuation format), no goals.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); dev = torch.device('cuda')
MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, D, PROJ = 3, 64, 768
CLAMP, TAUFLOOR, DT, TEMP, MAXNEW, GLAYER = 8.0, 1.0, 1.0, 0.8, 40, 32
SMOKE = os.environ.get('SMOKE', '0') == '1'; LIFE = 20 if SMOKE else int(os.environ.get('LIFE', '200')); WARMUP = 5; LR = 3e-4
SEEDS = ['Say whatever you want to say.', 'Begin.', 'Continue however you like.', 'Write freely.']
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
ck1 = torch.load('/home/pokazge/checkpoints/entity_stage1c.pt', weights_only=False, map_location='cpu')
ck2 = torch.load('/home/pokazge/checkpoints/stage2c_distill.pt', weights_only=False, map_location='cpu')
TGT = ck2['tgt']; Rp = ck1['Rp'].to(dev)
data = torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data']
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0).to(dev); nkv_raw = data[0]['nkv'][0].shape[1]
class LTCBank(nn.Module):
    def __init__(s, d_in, d):
        super().__init__(); s.read_in = nn.Linear(d_in, d); s.log_tau = nn.Parameter(torch.zeros(d)); s.d = d
    def target(s, perc): return torch.tanh(s.read_in(perc))                        # the input-direction the belief chases
    def step(s, h, perc, feed):
        tau = TAUFLOOR + F.softplus(s.log_tau)
        return (h + DT * (feed * s.target(perc) - h) / tau).clamp(-CLAMP, CLAMP)   # feed gates replenishment; starved -> drains (death)
bel = LTCBank(PROJ, D).to(dev); bel.load_state_dict(ck1['bel'])
tau = (TAUFLOOR + F.softplus(bel.log_tau)).detach(); SLOW = (tau > 4).float().to(dev)   # the PURPOSE lives in the slow modes
print('SLOW (purpose) dims: %d/%d (tau>4)' % (int(SLOW.sum()), D), flush=True)
class SlotHead(nn.Module):                                                         # embryo actuation format; gain comes from gain_head, NOT fixed gk
    def __init__(s, D, layers, nkv, hd, M=4):
        super().__init__(); s.ln = nn.LayerNorm(D); s.trunk = nn.Sequential(nn.Linear(D, 128), nn.GELU())
        s.k = nn.ModuleDict(); s.v = nn.ModuleDict(); s.gk = nn.ParameterDict(); s.gv = nn.ParameterDict(); s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = M
        for L in layers:
            s.k[str(L)] = nn.Linear(128, M * nkv * hd); s.v[str(L)] = nn.Linear(128, M * nkv * hd)
            s.gk[str(L)] = nn.Parameter(torch.tensor(64.0)); s.gv[str(L)] = nn.Parameter(torch.tensor(8.0))
    def forward(s, h, gain):
        z = s.trunk(s.ln(h)); o = {}
        for L in s.layers:
            k = F.normalize(s.k[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * (s.gk[str(L)] * gain)   # gain = FORMED intensity readout
            v = F.normalize(s.v[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * (s.gv[str(L)] * gain)
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
        return o
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
nkv = model.config.num_key_value_heads; hd = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
@torch.no_grad()
def probe_full():
    model.config.use_cache = True; out = model(tok('hi', return_tensors='pt').input_ids.to(dev), use_cache=True)
    return [i for i, L in enumerate(out.past_key_values.layers) if getattr(L, 'keys', None) is not None]
FULL = probe_full(); mods = {L: model.model.layers[L].self_attn for L in TGT}
for sa in mods.values(): sa._kv_inj = None
slot = SlotHead(D, TGT, nkv, hd).to(dev); slot.load_state_dict(ck2['slot'])
import math
GAIN0 = float(os.environ.get('GAIN0', '1.0'))                                      # birth intensity; PERTURB with GAIN0=3.5 (garbage zone) to test if REINFORCE self-corrects to the edge
gain_head = nn.Linear(D, 1).to(dev); nn.init.zeros_(gain_head.weight); nn.init.constant_(gain_head.bias, math.log(math.expm1(GAIN0)))
theta = list(slot.parameters()) + list(gain_head.parameters()) + list(bel.parameters())
opt = torch.optim.Adam(theta, lr=LR)
def gain_of(h): return F.softplus(gain_head(h)).squeeze()                          # FORMED per-moment intensity, in (0, inf)
def set_inj(h, gain, grad):
    cm = torch.enable_grad() if grad else torch.no_grad()
    with cm: o = slot(h, gain)
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
@torch.no_grad()
def sample_chunk(ms):
    model.config.use_cache = True; ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    return o[0, ids.shape[1]:]
@torch.no_grad()
def perceive(ms, rids):                                                            # native-KV read (-> belief input) + coherence on the ACTUAL text
    model.config.use_cache = True
    cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); s0 = cids.shape[1] - 1; nct = rids.shape[0]
    out = model(ids, use_cache=True); feats = []
    for L in FULL:
        lc = out.past_key_values.layers[L]; feats.append(lc.keys[0, :, -nct:, :].mean(0)); feats.append(lc.values[0, :, -nct:, :].mean(0))
    lp = float(F.log_softmax(out.logits[0, s0:s0 + nct].float(), -1).gather(1, rids.unsqueeze(1)).mean())
    return (torch.cat(feats, -1).float() @ Rp).mean(0), lp                         # perc [PROJ] on dev, lp_coh
def logp_grad(ms, rids):                                                           # logp of the SAMPLED chunk under the actuated model (REINFORCE action logp)
    model.config.use_cache = False
    cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); s0 = cids.shape[1] - 1; nct = rids.shape[0]
    return F.log_softmax(model(ids).logits[0, s0:s0 + nct].float(), -1).gather(1, rids.unsqueeze(1)).mean()
def dec(ids): return tok.decode(ids, skip_special_tokens=True).split('</think>')[-1].strip()
# ============================ LIFE ============================
import random; random.seed(0)
h = torch.zeros(D, device=dev); hist = [{'role': 'user', 'content': SEEDS[0]}]; base = 0.0; ema = None; recent_w = []
print('=== LIFE (self-formed purpose, formed intensity, REINFORCE on coherent self-continuation) ===', flush=True)
for t in range(LIFE):
    purpose = F.normalize((h * SLOW), dim=0)                                        # the STANDING purpose = slow-mode direction (committed)
    gain = gain_of(h)                                                               # FORMED intensity this moment
    warm = t < WARMUP
    if not warm: set_inj(h, gain, False)
    else: clear()
    rids = sample_chunk(win_of(hist)); clear(); tx = dec(rids)
    perc, lp_coh = perceive(win_of(hist), rids)
    target = bel.target(perc)                                                       # perceived output-direction in belief space
    self_adv = float(F.cosine_similarity(target, purpose, 0)) if float(purpose.norm()) > 1e-4 else 0.0
    coh = float(torch.sigmoid(torch.tensor((lp_coh + 2.5) / 1.0)))
    cw = set(tx.lower().split())                                                    # REPETITION penalty (on the text): ~1 for development, ->0 only for near-repeats.
    rep = 0.0 if not recent_w else max((len(cw & w) / max(1, len(cw | w))) for w in recent_w)
    live = 1.0 - max(0.0, (rep - 0.5) / 0.5)                                        # tolerant: only NEAR-repeats (overlap>0.5) lose feed; normal development keeps live~1
    recent_w.append(cw); recent_w = recent_w[-3:]
    feed = max(0.02, 0.5 * (self_adv + 1) * coh * live)                             # VIABILITY: coherently DEVELOP my own direction (continue + intelligible + not-frozen). nothing external.
    if not warm:                                                                    # REINFORCE: form gain_head + slot to inject what yields high feed
        set_inj(h, gain, True); lp_act = logp_grad(win_of(hist), rids); clear()
        r = feed; ema = r if ema is None else 0.9 * ema + 0.1 * r
        if torch.isfinite(lp_act):
            loss = -(r - ema) * lp_act
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(theta, 1.0); opt.step()
    h = bel.step(h, perc, feed).detach()
    hist += [{'role': 'assistant', 'content': tx}, {'role': 'user', 'content': tx}]
    if t % 5 == 0 or t == LIFE - 1:
        print('  t=%3d | gain=%.2f self_adv=%+.2f coh=%.2f feed=%.2f | |purpose|=%.2f |h|=%.2f%s' % (
            t, float(gain), self_adv, coh, feed, float((h * SLOW).norm()), float(h.norm()), ' [warmup]' if warm else ''), flush=True)
    if t % 10 == 0 or t == LIFE - 1:
        print('     text: %s' % tx[:140].replace(chr(10), ' '), flush=True)
torch.save({'slot': slot.state_dict(), 'gain_head': gain_head.state_dict(), 'bel': bel.state_dict()}, '/home/pokazge/checkpoints/organism2.pt')
print('=== ALL_DONE === (read: does gain settle to a coherent edge? does a stable purpose form? is the text coherent + self-consistent?)', flush=True)
