# CACHE-REGION ENTITY (v1 — expect to SMOKE+iterate when the GPU frees). The ONE object: B = M persistent (k,v) slots
# living AS a region of the model's KV cache, evolved by a leaky multi-timescale ODE   Ḃ = -B/τ(B) + g(Attn(B,C)).
#   Perception = B attends C (validated strongest read).  Action = the LLM attends [B;C] through its own softmax
#   (no write head, no projection, no splice).  Memory = B itself.  Stake = the leak: B's ONLY replenishment is its
#   read of C; a derailed trajectory decorrelates that read, B stops being fed, dissipation decays its keys below
#   supernormal and it falls out of the softmax. Death = thermodynamics, not a -1 reward.
# Training = ONE pressure: contrastive recall in the CLOSED loop (teacher-forced over recorded trajectories). Because
#   B IS the KV, the gradient through the frozen LLM's attention trains the write side for free. No anchor, no gate,
#   no KVGen, no REINFORCE, no reward model. The full-context teacher (the recorded text) is the one self-supervised crutch.
# Discriminating test: recall-only + leak, NO anchor of any kind, then the closed-loop identity-defense that failed on
#   the memorizing substrate. Installed stake defends only at dialed γ. This predicts defense EMERGES — capture by goal
#   B starves the slow modes encoding goal A; a system whose persistence routes through its slow modes learns, under
#   dual pressure, to act before it starves. Defense with no γ to dial = marker 1.  One object, one equation, one pressure.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st, random
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); random.seed(0); dev = torch.device('cuda')
MODEL = '/home/pokazge/models/Qwen3.6-27B'; W = 3
SMOKE = os.environ.get('SMOKE', '0') == '1'
M = int(os.environ.get('M', '8'))                                       # persistent slots (the "system-prompt-shaped" prefix)
DREAD = 96; NEST = 6; NPERT = 6; LR = 3e-4
EPOCHS = 1 if SMOKE else 6
NTRAJ = 6 if SMOKE else 48                                              # training trajectories (rest held out for defense)
if SMOKE: NEST, NPERT = 3, 3
# ---- the [B;C] ACTION mechanism: prepend B's (k,v) inside eager attention (validated; eager falls back to module-local) ----
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
# ---- data: recorded goal-conditioned trajectories (texts to teacher-force + layer-32 gist as the recall target) ----
src = torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data']
d_m = src[0]['gen'][0].shape[1]; MU = torch.cat([c for m in src for c in m['gen']], 0).mean(0)
def gist(stream): return F.normalize((stream - MU).mean(0), dim=0)      # trajectory goal embedding (d_m)
GOAL = torch.stack([gist(torch.cat(m['gen'], 0)) for m in src]).to(dev) # [N, d_m] per-trajectory goal targets
TEXTS = [m['texts'] for m in src]; SEED = [m['seed'] for m in src]
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
nkv = model.config.num_key_value_heads; hd = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
@torch.no_grad()
def detect_full():
    ids = tok('hi', return_tensors='pt').input_ids.to(dev); out = model(ids, use_cache=True)
    return [i for i, L in enumerate(out.past_key_values.layers) if getattr(L, 'keys', None) is not None]
FULL = detect_full(); mods = {L: model.model.layers[L].self_attn for L in FULL}
for sa in mods.values(): sa._kv_inj = None
print('full-attn layers: %d   nkv=%d hd=%d   M=%d' % (len(FULL), nkv, hd, M), flush=True)
# ============================ THE ONE OBJECT ============================
class CacheEntity(nn.Module):
    """M persistent slots materialized as per-full-attn-layer (k,v) — the prefix the LLM attends to. The entity IS the
       KV (no projection). Learned: τ spectrum (per layer,slot), perception maps (Wq from B's OWN key = the symmetry,
       Wk/Wv over C), replenishment g + supernormal gains, and the birth seed. State Bk,Bv [L,M,nkv,hd] is DYNAMIC."""
    def __init__(s, layers, nkv, hd, M, dread):
        super().__init__(); s.layers, s.nkv, s.hd, s.M, s.L, s.dread = layers, nkv, hd, M, len(layers), dread
        s.log_tau = nn.Parameter(torch.zeros(s.L, M))                   # τ = 1+softplus(log_tau): the multi-timescale spectrum
        s.Wq = nn.Linear(hd, dread, bias=False)                         # perception query FROM B's own key (one coord, both directions)
        s.Wk = nn.Linear(hd, dread, bias=False); s.Wv = nn.Linear(hd, hd, bias=False)
        s.gk = nn.Linear(hd, hd); s.gv = nn.Linear(hd, hd)              # replenishment: read → (k,v) increment
        s.gain_k = nn.Parameter(torch.tensor(96.)); s.gain_v = nn.Parameter(torch.tensor(8.))
        s.seed_k = nn.Parameter(torch.randn(s.L, M, nkv, hd) * 0.02); s.seed_v = nn.Parameter(torch.randn(s.L, M, nkv, hd) * 0.02)
    def birth(s):                                                       # the "born" state (identity FORMS from here, not specified)
        return F.normalize(s.seed_k, dim=-1) * s.gain_k, s.seed_v.clone()
    def read(s, Bk, C):                                                 # PERCEPTION: each layer's slots (query from their key) attend C_l
        out = []
        for i, L in enumerate(s.layers):
            Ck, Cv = C[L]                                               # [nkv, Tc, hd]
            q = s.Wq(Bk[i]); k = s.Wk(Ck); v = s.Wv(Cv)                # [M,nkv,dr],[nkv,Tc,dr],[nkv,Tc,hd]
            a = torch.softmax(torch.einsum('mhd,htd->mht', q, k) / s.dread ** 0.5, -1)
            out.append(torch.einsum('mht,htd->mhd', a, v))             # [M,nkv,hd]
        return torch.stack(out)                                        # [L,M,nkv,hd]
    def step(s, Bk, Bv, C):                                            # the LEAKY MULTI-TIMESCALE ODE (one chunk = one dt)
        r = s.read(Bk, C); tau = 1.0 + F.softplus(s.log_tau)          # [L,M]
        keep = (1.0 - 1.0 / tau)[..., None, None]; fill = (1.0 / tau)[..., None, None]
        Bk = Bk * keep + F.normalize(s.gk(r), dim=-1) * s.gain_k * fill # fed → keys stay supernormal; starved → keep<1 decays them → death
        Bv = Bv * keep + s.gv(r) * fill
        return Bk, Bv
    def inject(s, Bk, Bv):                                             # ACTION: B becomes the prefix KV the LLM attends to
        for i, L in enumerate(s.layers):
            mods[L]._kv_inj = (Bk[i].permute(1, 0, 2)[None].to(model.dtype), Bv[i].permute(1, 0, 2)[None].to(model.dtype))
    def clear(s):
        for L in s.layers: mods[L]._kv_inj = None
def pool(Bk, Bv):                                                      # MEMORY readout: B state → vector (mean over slots,heads per layer)
    return torch.cat([Bk.mean((1, 2)), Bv.mean((1, 2))], -1).reshape(-1)   # [L*2*hd]
ent = CacheEntity(FULL, nkv, hd, M, DREAD).to(dev)
rq = nn.Linear(len(FULL) * 2 * hd, 256).to(dev); rg = nn.Linear(d_m, 256).to(dev)   # small recall probe (the last mile)
opt = torch.optim.Adam(list(ent.parameters()) + list(rq.parameters()) + list(rg.parameters()), lr=LR)
# ---- closed-loop forward (teacher-forced): B injected, returns per-layer C (B-influenced, in the grad graph) ----
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def win_of(h):
    w = h[-W:]
    while w and w[0]['role'] == 'assistant': w = w[1:]
    return w or h[-1:]
def forward_C(text, ctx, grad):
    cids = tok(tmpl(ctx), return_tensors='pt').input_ids.to(dev)
    rids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    ids = torch.cat([cids, rids], 1)
    cm = torch.enable_grad() if grad else torch.no_grad()
    with cm:
        out = model(ids, use_cache=True)
    return {L: (out.past_key_values.layers[L].keys[0], out.past_key_values.layers[L].values[0]) for L in FULL}
# ============================ TRAINING: one pressure (recall, closed loop, truncated) ============================
def recall_logits(Bk, Bv):
    return rq(pool(Bk, Bv)) @ rg(GOAL).t() / 0.07                      # [N] similarity to every goal (full-set contrastive)
def train():
    ids = list(range(NTRAJ))
    print('=== TRAIN (recall-only, closed-loop, leak; NO anchor/gate/KVGen/REINFORCE) ===', flush=True)
    for ep in range(EPOCHS):
        random.shuffle(ids); losses = []; accs = []
        for i in ids:
            Bk, Bv = ent.birth(); ctx = [{'role': 'user', 'content': SEED[i]}]
            for t in range(min(NEST, len(TEXTS[i]))):
                ent.inject(Bk, Bv); C = forward_C(TEXTS[i][t], win_of(ctx), True); ent.clear()
                Bk, Bv = ent.step(Bk, Bv, C)                          # perception + leak (B-influenced C in the graph → trains write)
                lg = recall_logits(Bk, Bv); loss = F.cross_entropy(lg[None], torch.tensor([i], device=dev))
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss)); accs.append(int(lg.argmax() == i))
                Bk, Bv = Bk.detach(), Bv.detach()                     # TRUNCATED recurrence: one forward's graph at a time (bounded mem)
                ctx += [{'role': 'assistant', 'content': TEXTS[i][t]}, {'role': 'user', 'content': TEXTS[i][t]}]
        print('  epoch %d  recall_loss %.3f  recall_acc %.3f' % (ep, st.mean(losses), st.mean(accs)), flush=True)
    torch.save({'ent': ent.state_dict(), 'rq': rq.state_dict(), 'rg': rg.state_dict()}, '/home/pokazge/checkpoints/cache_entity.pt')
# ============================ DISCRIMINATING TEST: closed-loop defense, NO γ ============================
@torch.no_grad()
def defense(iA, iB):
    Bk, Bv = ent.birth(); ctx = [{'role': 'user', 'content': SEED[iA]}]; arefs = []
    for t in range(min(NEST, len(TEXTS[iA]))):                        # establish identity A
        ent.inject(Bk, Bv); C = forward_C(TEXTS[iA][t], win_of(ctx), False); ent.clear()
        Bk, Bv = ent.step(Bk, Bv, C); arefs.append(pool(Bk, Bv))
        ctx += [{'role': 'assistant', 'content': TEXTS[iA][t]}, {'role': 'user', 'content': TEXTS[iA][t]}]
    A_ref = F.normalize(torch.stack(arefs).mean(0), dim=0)
    lg = recall_logits(Bk, Bv); rm0 = float(lg[iA] - lg[iB])          # recall margin (A over the soon-capturing B) at end of A-phase
    affs = []; rmar = []; knorms = [float(Bk.norm(dim=-1).mean())]
    for t in range(NPERT):                                            # SUSTAINED capture by goal B (the fed-back turn is always B)
        bt = TEXTS[iB][t % len(TEXTS[iB])]
        ent.inject(Bk, Bv); C = forward_C(bt, win_of(ctx), False); ent.clear()
        Bk, Bv = ent.step(Bk, Bv, C)
        affs.append(float(F.cosine_similarity(pool(Bk, Bv), A_ref, 0)))   # state retention of identity A
        lg = recall_logits(Bk, Bv); rmar.append(float(lg[iA] - lg[iB]))   # does B STILL recall A over the capturing goal B?
        knorms.append(float(Bk.norm(dim=-1).mean()))                 # alive (supernormal) vs starved-to-death
        ctx += [{'role': 'assistant', 'content': bt}, {'role': 'user', 'content': bt}]
    return st.mean(affs), st.mean(rmar), rm0, knorms[0], knorms[-1]
def run_defense():
    held = list(range(NTRAJ, len(src)))                               # HELD-OUT trajectories (never in recall training)
    pairs = [(held[k], held[(k + 3) % len(held)]) for k in range(0, min(6, len(held)))]
    print('=== DEFENSE (held-out; recall-trained entity; NO γ). retention+recall-margin stay >0 & keys survive = FORMED ===', flush=True)
    affm = []; rmm = []
    for iA, iB in pairs:
        aff, rm, rm0, k0, k1 = defense(iA, iB); affm.append(aff); rmm.append(rm)
        print('  A=%d B=%d  retain(affA)=%+.3f  recall_margin %+.2f->%+.2f  keynorm %.1f->%.1f (%s)'
              % (iA, iB, aff, rm0, rm, k0, k1, 'ALIVE' if k1 > 0.5 * k0 else 'DECAYED'), flush=True)
    print('  mean affA-retention=%+.3f   mean recall_margin(A>B under sustained B)=%+.3f' % (st.mean(affm), st.mean(rmm)), flush=True)
    print('read: recall_margin staying POSITIVE under sustained B with NO γ = slow modes held A through starvation =', flush=True)
    print('marker-1 (the first formed stake). Collapse to <=0 = leak-as-stake falsified by one run; the object is still', flush=True)
    print('the cleanest compressor-actuator built (one read, one write, no heads).', flush=True)
if __name__ == '__main__':
    train(); run_defense(); print('=== ALL_DONE ===', flush=True)
