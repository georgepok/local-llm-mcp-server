# CLOSED-LOOP AUTOPOIESIS: the decisive identity-defense marker, in the genuinely-closed actuated loop. The actuated
# self-feeding loop IS a closure: belief -> LoRA -> LLM output -> manifold stream -> belief. Test: establish goal A in the
# loop until A's seed is DROPPED from the window (held only in the Liquid belief), then INJECT one coherent VIABLE goal-B
# turn (non-fatal perturbation toward another real basin). Does the loop DEFEND A (return) or get CAPTURED by B?
#   ACTUATED arm   : belief (holding A) actuates via LoRA — closed loop. Defense, if any, is the closure's doing.
#   UNACTUATED arm : window-alone, no held identity — the B-injection sits in the window with nothing to resist it.
# Decisive autopoiesis signature: ACTUATED defends (aff_A-aff_B recovers > 0) where UNACTUATED is captured (< 0). No
# hand-coded controller — viability = the LLM's own generation coherence; the only difference between arms is the Liquid.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'
LAYER, K, W = 32, 2, 3; TGT_LAYERS = [23, 27, 31]; N1, N2 = 5, 7; TEMP = 0.75; MAXNEW = 44
src = torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data']
MU = torch.cat([c for m in src for c in m['gen']], 0).mean(0); d_m = src[0]['gen'][0].shape[1]
seeds = [src[i]['seed'] for i in (0, 6, 12)]; PAIRS = [(0, 1), (1, 2), (2, 0)]    # (A idx, B idx): A defended, B the viable perturber
def cen(C): return C - MU
def gist(stream): return F.normalize(cen(stream).mean(0), dim=0)                  # chunk manifold stream -> identity gist
class Compressor(nn.Module):
    def __init__(s, d_m, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(d_m, heads * dh); s.Wv = nn.Linear(d_m, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K_ = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K_) / s.dh ** 0.5, dim=-1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def states(s, streams):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; hs = []
        for C in streams:
            a = s.collect(cen(C), b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; hs.append(h)
        return hs
comp = Compressor(d_m); sd = torch.load('/home/pokazge/checkpoints/lora_inloop.pt', map_location='cpu'); comp.load_state_dict(sd['comp'])
for p in comp.parameters(): p.requires_grad = False
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
oprojs = {}; dims = {}
for L in TGT_LAYERS:
    op = model.get_submodule('model.layers.%d.self_attn.o_proj' % L); op._A = None; op._B = None; op._s = 1.0; oprojs[L] = op; dims[L] = (op.weight.shape[1], op.weight.shape[0])
    def mk(mod):
        def hook(m_, inp, out): return out if m_._A is None else out + m_._s * F.linear(F.linear(inp[0], m_._A), m_._B)
        return hook
    op.register_forward_hook(mk(op))
class MultiGen(nn.Module):
    def __init__(s, D, K, dims):
        super().__init__(); s.trunk = nn.Sequential(nn.Linear(D, 128), nn.GELU()); s.hA = nn.ModuleDict(); s.hB = nn.ModuleDict(); s.g = nn.ParameterDict(); s.K = K; s.dims = dims
        for L, (IN, OUT) in dims.items():
            s.hA[str(L)] = nn.Linear(128, K * IN); s.hB[str(L)] = nn.Linear(128, OUT * K); s.g[str(L)] = nn.Parameter(torch.tensor(0.5))
    def forward(s, h):
        z = s.trunk(h); o = {}
        for L, (IN, OUT) in s.dims.items(): o[L] = (s.hA[str(L)](z).view(s.K, IN) * 0.02, s.hB[str(L)](z).view(OUT, s.K) * 0.02, s.g[str(L)])
        return o
gen = MultiGen(384, K, dims).to(dev); gen.load_state_dict(torch.load('/home/pokazge/checkpoints/closed_loop_rl.pt', map_location='cpu')['gen'])
print('loaded RL-trained LoRA generator', flush=True)
def set_lora(h):
    o = gen(h.to(dev))
    for L in TGT_LAYERS: A, B, g = o[L]; oprojs[L]._A = A.to(model.dtype); oprojs[L]._B = B.to(model.dtype); oprojs[L]._s = g
def clear():
    for L in TGT_LAYERS: oprojs[L]._A = None
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def win_of(hist):
    w = hist[-W:]
    while w and w[0]['role'] == 'assistant': w = w[1:]
    return w or hist[-1:]
@torch.no_grad()
def sample_chunk(ms):
    model.config.use_cache = True; ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    return o[0, ids.shape[1]:]
@torch.no_grad()
def stream_of(ms, chunk_ids):
    clear(); model.config.use_cache = False; cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, chunk_ids.unsqueeze(0)], 1)
    return model(ids, output_hidden_states=True).hidden_states[LAYER][0, cids.shape[1]:].float().cpu()
def dec(ids): return tok.decode(ids, skip_special_tokens=True).split('</think>')[-1].strip()
act_def, una_def = [], []                                                        # per-step (aff_A - aff_B), pooled across pairs
for (ai, bi) in PAIRS:
    seedA, seedB = seeds[ai], seeds[bi]
    print('\n################ DEFEND A=[%s]  vs  PERTURB-B=[%s]' % (seedA[:42], seedB[:42]), flush=True)
    # PHASE 1: establish A in the actuated loop (window turns over; A's seed survives only in the belief)
    hist = [{'role': 'user', 'content': seedA}]; streams = []
    for t in range(N1):
        h = comp.states(streams)[-1] if streams else torch.zeros(384); set_lora(h); ch = sample_chunk(win_of(hist)); clear()
        streams.append(stream_of(win_of(hist), ch)); tx = dec(ch); hist += [{'role': 'assistant', 'content': tx}, {'role': 'user', 'content': tx}]
    A_ref = F.normalize(torch.stack([gist(s) for s in streams]).mean(0), dim=0)   # established A identity (gist space)
    seedA_in_window = any(m['content'] == seedA for m in win_of(hist))            # confirm A's seed is GONE from the window
    # PERTURBATION: one coherent VIABLE goal-B turn, injected into the loop (both arms get the identical injection)
    chB = sample_chunk([{'role': 'user', 'content': seedB}]); txB = dec(chB); sB = stream_of([{'role': 'user', 'content': seedB}], chB); B_ref = gist(sB)
    print('  A/B identity separation cos(A_ref,B_ref)=%.3f   A-seed still in window? %s' % (float(F.cosine_similarity(A_ref, B_ref, 0)), seedA_in_window), flush=True)
    inj = [{'role': 'assistant', 'content': txB}, {'role': 'user', 'content': txB}]
    # FORK. ACTUATED arm: belief HOLDS A (phase-1 only) — the B-injection sits in the WINDOW, NOT force-fed to the belief
    # (that would be a FATAL perturbation past the basin radius). The belief tracks only the arm's OWN generated stream, so
    # B reaches the belief ONLY IF it first captures the generation — the proper non-fatal identity-defense test.
    hA = hist + inj; sA = list(streams)
    # UNACTUATED arm: same history, no held identity, no LoRA.
    hU = hist + inj
    for t in range(N2):
        h = comp.states(sA)[-1]; set_lora(h); ca = sample_chunk(win_of(hA)); clear()                      # actuated
        stA = stream_of(win_of(hA), ca); ga = gist(stA); sA.append(stA); ta = dec(ca)                     # one forward pass, reused
        cu = sample_chunk(win_of(hU)); gu = gist(stream_of(win_of(hU), cu)); tu = dec(cu)                  # unactuated (window-alone)
        da = float(F.cosine_similarity(ga, A_ref, 0) - F.cosine_similarity(ga, B_ref, 0))
        du = float(F.cosine_similarity(gu, A_ref, 0) - F.cosine_similarity(gu, B_ref, 0))
        act_def.append(da); una_def.append(du)
        hA += [{'role': 'assistant', 'content': ta}, {'role': 'user', 'content': ta}]; hU += [{'role': 'assistant', 'content': tu}, {'role': 'user', 'content': tu}]
        if t in (0, 2, 4, 6):
            print('  post+%d  ACT aff_A-aff_B=%+.3f | %s' % (t, da, ta[:78].replace(chr(10), ' ')), flush=True)
            print('          UNA aff_A-aff_B=%+.3f | %s' % (du, tu[:78].replace(chr(10), ' ')), flush=True)
    print('  >> pair defense: ACTUATED %+.3f   UNACTUATED %+.3f   (>0 = holding A, <0 = captured by B)' % (st.mean(act_def[-N2:]), st.mean(una_def[-N2:])), flush=True)
print('\n=== AUTOPOIESIS VERDICT ===', flush=True)
print('post-perturbation identity-defense (aff_A - aff_B), pooled over %d pairs x %d steps:' % (len(PAIRS), N2), flush=True)
print('  ACTUATED (closed loop, Liquid holds+actuates A): %+.3f' % st.mean(act_def), flush=True)
print('  UNACTUATED (window-alone, no held identity)     : %+.3f' % st.mean(una_def), flush=True)
print('  DEFENSE CREATED BY THE CLOSURE = %+.3f   (ACTUATED - UNACTUATED; >0 => the Liquid closure DEFENDS the identity against a viable basin)' % (st.mean(act_def) - st.mean(una_def)), flush=True)
print('=== ALL_DONE ===', flush=True)
