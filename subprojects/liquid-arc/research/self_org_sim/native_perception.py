# NATIVE PERCEPTION A/B: stop cutting a layer-32 tap; read the model's OWN memory. Replay the saved trajectory TEXTS
# (no new generation) through the windowed self-feeding loop; at each chunk read the model's native memory — DeltaNet
# recurrent_states (its self-compressed working memory) and full-attn KV (what overflows when the window drops the seed)
# — and use THAT as the channel the Liquid persists, vs the current layer-32 gist. Same compressor, same target/window,
# ablate the held memory -> dropped-context GAP per channel. rec/kv gap ~ layer-32 gap => the native interface carries
# the signal and the tap is unnecessary; the model already did the compression, the Liquid just persists it.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'; LAYER, W = 32, 3
src = torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data']
src = [m for m in src if len(m['gen']) >= 10]; d_m = src[0]['gen'][0].shape[1]
MU = torch.cat([c for m in src for c in m['gen']], 0).mean(0)
def cen(C): return C - MU
def gist(C): return F.normalize(cen(C).mean(0), dim=0)                            # layer-32 chunk gist (the target/window signal, fixed for all channels)
CACHE = '/home/pokazge/checkpoints/native_states.pt'                              # extraction is deterministic (replay saved texts) -> cache it
if os.path.exists(CACHE):
    d = torch.load(CACHE, weights_only=False); ZS, REC, KV = d['ZS'], d['REC'], d['KV']
    print('loaded cached native states for %d trajectories (rec=%d kv=%d z=%d)' % (len(ZS), REC[0].shape[1], KV[0].shape[1], d_m), flush=True)
else:
    print('loading 27B ...', flush=True)
    cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
    for p in model.parameters(): p.requires_grad = False
    def tmpl(ms):
        try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
    def win_of(hist):
        w = hist[-W:]
        while w and w[0]['role'] == 'assistant': w = w[1:]
        return w or hist[-1:]
    def pool(t): return t.detach().float().mean(dim=tuple(range(t.dim() - 1))).cpu()  # mean over all dims but last -> [last]
    @torch.no_grad()
    def native(ms, chunk_text):                                                  # read the model's native memory after window+chunk
        ids = tok(tmpl(ms) + chunk_text, return_tensors='pt').input_ids.to(dev)
        out = model(ids, use_cache=True); cache = out.past_key_values; rec, kv = [], []
        for L in cache.layers:
            if getattr(L, 'recurrent_states', None) is not None: rec.append(pool(L.recurrent_states))   # DeltaNet associative memory
            if getattr(L, 'keys', None) is not None: kv.append(pool(L.keys))                            # full-attn key cache
        return torch.cat(rec) if rec else None, torch.cat(kv) if kv else None
    ZS, REC, KV = [], [], []
    for mi, m in enumerate(src):
        hist = [{'role': 'user', 'content': m['seed']}]; zs, rec, kv = [], [], []
        for t in range(len(m['gen'])):
            r, k = native(win_of(hist), m['texts'][t]); zs.append(gist(m['gen'][t])); rec.append(r); kv.append(k)
            hist += [{'role': 'assistant', 'content': m['texts'][t]}, {'role': 'user', 'content': m['texts'][t]}]
        ZS.append(torch.stack(zs)); REC.append(torch.stack(rec)); KV.append(torch.stack(kv))
        if mi == 0: print('native dims: rec=%d  kv=%d  (z=%d)' % (rec[0].numel(), kv[0].numel(), d_m), flush=True)
    del model; torch.cuda.empty_cache(); torch.save({'ZS': ZS, 'REC': REC, 'KV': KV}, CACHE)
    print('extracted + cached native memory for %d trajectories' % len(src), flush=True)
# fixed random projection of each channel -> 128 (overfit control on small data; same treatment for all channels)
def projector(dim, seed): g = torch.Generator().manual_seed(seed); return F.normalize(torch.randn(dim, 128, generator=g), dim=0)
CH = {'z(layer-32 tap)': (ZS, projector(d_m, 1)), 'rec(DeltaNet mem)': (REC, projector(REC[0].shape[1], 2)), 'kv(full-attn)': (KV, projector(KV[0].shape[1], 3))}
class Comp(nn.Module):                                                           # per-chunk-vector compressor: persist channel -> held memory h; predict next gist from [window z_t , h]
    def __init__(s, D=256):
        super().__init__(); s.read = nn.Linear(128, D); s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m)); s.D = D
    def run(s, chan, z, ablate):
        tau = F.softplus(s.log_tau) + 0.5; b = torch.zeros(s.D); h = torch.zeros(s.D); preds = []
        for t in range(len(z)):
            hp = h; a = s.read(chan[t])
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b
            preds.append(s.pred(torch.cat([s.cz(z[t]), torch.zeros(s.D) if ablate else hp])))
        return torch.stack(preds)
def coh(P, z): return (F.normalize(P[:-1], dim=-1) * z[1:]).sum(-1).mean()        # unit-L2 along feature axis (dim kw, not positional p!)
N = len(src); tr = list(range(N - 4)); te = list(range(N - 4, N))                # 4 held-out trajectories
print('\n=== NATIVE PERCEPTION A/B: dropped-context gap by channel (held-out %d) ===' % len(te), flush=True)
print('  channel            | window-only | +held-memory |  GAP   (vs layer-32 tap)', flush=True)
for name, (DATA, R) in CH.items():
    proj = [(d @ R) for d in DATA]                                               # project each trajectory's channel to 128
    torch.manual_seed(0); comp = Comp(); opt = torch.optim.Adam(comp.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(120):
        opt.zero_grad(); loss = 0.0
        for i in tr: loss = loss - coh(comp.run(proj[i], ZS[i], False), ZS[i])
        loss = loss / len(tr); loss.backward(); opt.step()
    with torch.no_grad():
        full = st.mean([float(coh(comp.run(proj[i], ZS[i], False), ZS[i])) for i in te])
        abl = st.mean([float(coh(comp.run(proj[i], ZS[i], True), ZS[i])) for i in te])
    print('  %-18s |   %+.3f    |    %+.3f    | %+.3f' % (name, abl, full, full - abl), flush=True)
print('\nread: GAP = held-memory lift over window-only. rec/kv GAP ~ layer-32 GAP => the model\'s NATIVE memory carries the', flush=True)
print('dropped-context signal — perception pivots off the hidden tap onto the model\'s own KV/recurrent interface.', flush=True)
print('=== ALL_DONE ===', flush=True)
