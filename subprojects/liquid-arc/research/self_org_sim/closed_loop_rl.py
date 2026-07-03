# CLOSED-LOOP RL for the dynamic-LoRA actuator. The windowed+LoRA model generates its OWN self-feeding trajectory
# (sampled); reward = the FULL-CONTEXT model's avg log-likelihood of each generated chunk (on-track-ness); REINFORCE
# trains the LoRA generator to keep its own drifting loop coherent (handles compounding error — the property a single-
# step metric can't show). Compressor (AoA, pretrained) + base 27B frozen; warm-start the LoRA gen from the in-loop ckpt.
# Eval: does the actuated loop's reward beat the windowed-alone loop's, and does it climb with RL. SMOKE=1 tiny.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); M = '/home/pokazge/models/Qwen3.6-27B'; LAYER, K, W = 32, 2, 3
TGT_LAYERS = [int(x) for x in os.environ.get('TGT_LAYERS', '23,27,31').split(',')]; SMOKE = os.environ.get('SMOKE', '0') == '1'
NSTEP = 4 if SMOKE else 10; EPISODES = 1 if SMOKE else 24; TEMP = 0.9; MAXNEW = 40
src = torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data']
MU = torch.cat([c for m in src for c in m['gen']], 0).mean(0); d_m = src[0]['gen'][0].shape[1]
seeds = [m['seed'] for m in src][:(2 if SMOKE else 16)]
def cen(C): return C - MU
class Compressor(nn.Module):
    def __init__(s, d_m, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(d_m, heads * dh); s.Wv = nn.Linear(d_m, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, C, b):
        q = s.Wq(b).view(s.h, s.dh); K_ = s.Wk(C).view(-1, s.h, s.dh); V = s.Wv(C).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K_) / s.dh ** 0.5, dim=-1); return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def states(s, streams):                                                      # running compression over the generated streams
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; hs = []
        for C in streams:
            a = s.collect(cen(C), b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b; hs.append(h)
        return hs
comp = Compressor(d_m)
ck = '/home/pokazge/checkpoints/lora_inloop.pt'
sd = torch.load(ck, map_location='cpu') if os.path.exists(ck) else None
if sd is not None: comp.load_state_dict(sd['comp']); print('warm-started compressor from lora_inloop.pt', flush=True)
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
print('27B loaded %.1fGB targets=%s' % (torch.cuda.memory_allocated() / 1e9, TGT_LAYERS), flush=True)
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
if sd is not None and 'gen' in sd:
    try: gen.load_state_dict(sd['gen']); print('warm-started LoRA generator from in-loop ckpt', flush=True)
    except Exception as e: print('gen warm-start skipped:', repr(e), flush=True)
opt = torch.optim.Adam(gen.parameters(), lr=3e-4)
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
def sample_chunk(ms):                                                            # the policy acts (LoRA already set), sampled
    model.config.use_cache = True
    ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    return o[0, ids.shape[1]:]
def logp(ms, chunk_ids, grad):                                                   # avg log-prob of chunk_ids given context
    model.config.use_cache = False
    cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, chunk_ids.unsqueeze(0)], 1)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        lg = model(ids).logits[0]; st_ = cids.shape[1] - 1; lp = F.log_softmax(lg[st_:st_ + chunk_ids.shape[0]].float(), -1)
        return lp.gather(1, chunk_ids.unsqueeze(1)).squeeze(1).mean()
@torch.no_grad()
def stream_of(ms, chunk_ids):                                                    # per-token layer-32 hidden of the chunk (compressor input)
    model.config.use_cache = False; cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, chunk_ids.unsqueeze(0)], 1)
    return model(ids, output_hidden_states=True).hidden_states[LAYER][0, cids.shape[1]:].float().cpu()
base = None; hist_reward = []; lift_log = []
for ep in range(EPISODES):
    seed = seeds[ep % len(seeds)]; hist = [{'role': 'user', 'content': seed}]; streams = []; ep_r = []; ep_lift = []
    for t in range(NSTEP):
        hh = comp.states(streams); h = hh[-W] if len(hh) >= W else (hh[0] if hh else torch.zeros(384))   # compression of dropped history
        set_lora(h.to(dev))
        ch = sample_chunk(win_of(hist))                                          # windowed + LoRA generates its own next chunk
        lp_pi = logp(win_of(hist), ch, grad=True)                               # policy log-prob (grad)
        clear()
        with torch.no_grad(): r = float(logp(hist, ch, grad=False))            # reward: full-context likelihood (on-track-ness)
        b0 = base if base is not None else r; adv = r - b0
        loss = -(lp_pi) * adv
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(gen.parameters(), 1.0); opt.step()
        base = r if base is None else 0.9 * base + 0.1 * r; ep_r.append(r)
        # windowed-alone counterfactual reward for the SAME context (fresh sample, LoRA off) — measures lift
        clear()
        with torch.no_grad(): ch_w = sample_chunk(win_of(hist)); r_walone = float(logp(hist, ch_w, grad=False))
        ep_lift.append(r - r_walone)
        txt = tok.decode(ch, skip_special_tokens=True).split('</think>')[-1].strip()
        streams.append(stream_of(win_of(hist), ch)); hist += [{'role': 'assistant', 'content': txt}, {'role': 'user', 'content': txt}]
    hist_reward.append(st.mean(ep_r)); lift_log.append(st.mean(ep_lift))
    print('ep %2d  mean reward %.3f  baseline %.3f  lift(LoRA vs window-alone) %+.3f' % (ep, st.mean(ep_r), base, st.mean(ep_lift)), flush=True)
print('\n=== CLOSED-LOOP RL ===')
print('  reward trajectory (first->last quarter): %.3f -> %.3f' % (st.mean(hist_reward[:max(1, len(hist_reward) // 4)]), st.mean(hist_reward[-max(1, len(hist_reward) // 4):])))
print('  on-track LIFT (windowed+LoRA - windowed-alone, full-context likelihood): %+.4f' % st.mean(lift_log))
torch.save({'gen': gen.state_dict()}, '/home/pokazge/checkpoints/closed_loop_rl.pt')
print('=== ALL_DONE ===')
