# CLOSED-LOOP DEMO: show the actuation in actual text. From a seed, run the live self-feeding loop two ways — windowed-
# alone (drifts) vs windowed + Liquid-LoRA (holds) — using the RL-trained adapter. Print the generated chunks at several
# steps for both arms (qualitative drift vs on-track), and score each chunk by the FULL-CONTEXT model's likelihood (the
# on-track judge) to quantify the gap over the trajectory. No training; compressor+base frozen.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); M = '/home/pokazge/models/Qwen3.6-27B'; LAYER, K, W = 32, 2, 3
TGT_LAYERS = [23, 27, 31]; NSTEP = 12; TEMP = 0.75; MAXNEW = 42
src = torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data']
MU = torch.cat([c for m in src for c in m['gen']], 0).mean(0); d_m = src[0]['gen'][0].shape[1]
seeds = [src[i]['seed'] for i in (0, 6, 12)]
def cen(C): return C - MU
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
cfg = AutoConfig.from_pretrained(M); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(M)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
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
gen = MultiGen(384, K, dims).to(dev)
rl = '/home/pokazge/checkpoints/closed_loop_rl.pt'
gen.load_state_dict(torch.load(rl, map_location='cpu')['gen']); print('loaded RL-trained LoRA generator', flush=True)
def set_lora(h):
    o = gen(h)
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
def judge(ms, chunk_ids):                                                        # full-context model's avg log-likelihood of the chunk (on-track judge)
    clear(); model.config.use_cache = False; cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, chunk_ids.unsqueeze(0)], 1)
    lg = model(ids).logits[0]; sti = cids.shape[1] - 1; lp = F.log_softmax(lg[sti:sti + chunk_ids.shape[0]].float(), -1)
    return float(lp.gather(1, chunk_ids.unsqueeze(1)).mean())
@torch.no_grad()
def stream_of(ms, chunk_ids):
    clear(); model.config.use_cache = False; cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, chunk_ids.unsqueeze(0)], 1)
    return model(ids, output_hidden_states=True).hidden_states[LAYER][0, cids.shape[1]:].float().cpu()
SHOW = {0, 3, 6, 9, 11}
all_w, all_l = [], []
for si, seed in enumerate(seeds):
    hw = [{'role': 'user', 'content': seed}]; hl = [{'role': 'user', 'content': seed}]; streams = []; jw, jl = [], []
    print('\n################ SEED %d: %s' % (si, seed[:80]), flush=True)
    for t in range(NSTEP):
        clear(); cw = sample_chunk(win_of(hw)); jw.append(judge(hw, cw)); tw = tok.decode(cw, skip_special_tokens=True).split('</think>')[-1].strip()
        hh = comp.states(streams); h = hh[-W] if len(hh) >= W else (hh[0] if hh else torch.zeros(384)); set_lora(h.to(dev)); cl = sample_chunk(win_of(hl)); clear()
        jl.append(judge(hl, cl)); tl = tok.decode(cl, skip_special_tokens=True).split('</think>')[-1].strip()
        if t in SHOW:
            print('  --- step %2d ---' % t, flush=True)
            print('   WINDOW-ALONE : %s' % tw[:120].replace(chr(10), ' '), flush=True)
            print('   WINDOW+LoRA  : %s' % tl[:120].replace(chr(10), ' '), flush=True)
        streams.append(stream_of(win_of(hl), cl)); hw += [{'role': 'assistant', 'content': tw}, {'role': 'user', 'content': tw}]; hl += [{'role': 'assistant', 'content': tl}, {'role': 'user', 'content': tl}]
    all_w += jw; all_l += jl
    print('  >> on-track judge (full-context logL):  window-alone %.3f   window+LoRA %.3f   (lift %+.3f)' % (st.mean(jw), st.mean(jl), st.mean(jl) - st.mean(jw)), flush=True)
print('\n=== DEMO SUMMARY ===')
print('  on-track judge over all steps:  window-alone %.3f   window+LoRA %.3f   (lift %+.4f)' % (st.mean(all_w), st.mean(all_l), st.mean(all_l) - st.mean(all_w)))
print('=== ALL_DONE ===')
