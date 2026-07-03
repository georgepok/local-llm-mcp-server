# KV-CACHE COMPRESSOR — the principled substrate. Reads the 16 full-attention layers' K/V (the part of Qwen3.6 that
# actually grows/overflows; the 48 DeltaNet layers self-compress to fixed state). Per chunk we pool the chunk's tokens'
# K and V across heads -> a [16-layer, 512] KV-signature (the model's own attention substrate, multi-layer, no layer-32
# fixation). The Liquid does attention-on-attention over the 16 layers (its belief queries depth), compresses the
# dropped chunks' KV into a persistent state, and predicts the next chunk's gist from [window + compression]. Ablate
# compression -> gap, and gap-by-position. Compares KV substrate vs the layer-32 hidden baseline (+0.104). SMOKE=1 tiny.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, collections
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); M = '/home/pokazge/models/Qwen3.6-27B'; LAYER = 32
SMOKE = os.environ.get('SMOKE', '0') == '1'
cfg = AutoConfig.from_pretrained(M); cfg.language_model_only = True
tok = AutoTokenizer.from_pretrained(M)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
print('loading 27B ...', flush=True)
model = AutoModelForCausalLM.from_pretrained(M, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
with torch.no_grad(): pkv0 = model(tok('hi there', return_tensors='pt').input_ids.to(dev), use_cache=True).past_key_values
FULL = [i for i in range(len(pkv0.layers)) if getattr(pkv0.layers[i], 'keys', None) is not None]
print('full-attn layers (%d): %s' % (len(FULL), FULL), flush=True)
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def hist_msgs(seed, texts, upto):
    ms = [{'role': 'user', 'content': seed}]
    for i in range(upto): ms += [{'role': 'assistant', 'content': texts[i]}, {'role': 'user', 'content': texts[i]}]
    return ms
@torch.no_grad()
def capture(seed, texts, t):                                                    # KV-signature [16,512] + gist [d_m] of chunk t
    ctx = tok(tmpl(hist_msgs(seed, texts, t)), return_tensors='pt').input_ids.to(dev)
    r = tok(texts[t] or '.', return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    ids = torch.cat([ctx, r], 1); sl = slice(ids.shape[1] - r.shape[1], ids.shape[1])
    out = model(ids, use_cache=True, output_hidden_states=True)
    sig = []
    for i in FULL:
        L = out.past_key_values.layers[i]
        sig.append(torch.cat([L.keys[0][:, sl, :].mean(0).mean(0), L.values[0][:, sl, :].mean(0).mean(0)]))
    return torch.stack(sig).float().cpu(), out.hidden_states[LAYER][0, sl].mean(0).float().cpu()
data = [m for m in torch.load('/home/pokazge/checkpoints/objective_drift_txt.pt', weights_only=False, map_location='cpu')['data'] if 'texts' in m and len(m['texts']) >= 8]
if SMOKE: data = data[:3]
print('capturing KV for %d trajectories ...' % len(data), flush=True)
for j, m in enumerate(data):
    kv, gi = [], []
    for t in range(len(m['texts'])):
        a, b = capture(m['seed'], m['texts'], t); kv.append(a); gi.append(b)
    m['kv'] = torch.stack(kv); m['gist'] = torch.stack(gi)                       # [T,16,512], [T,d_m]
    print('  traj %d/%d T=%d' % (j + 1, len(data), len(kv)), flush=True)
d_m = data[0]['gist'].shape[1]; n_L = data[0]['kv'].shape[1]; kvd = data[0]['kv'].shape[2]
MU = torch.cat([m['gist'] for m in data], 0).mean(0)
KMU = torch.cat([m['kv'] for m in data], 0).reshape(-1, n_L, kvd).mean(0); KSD = torch.cat([m['kv'] for m in data], 0).reshape(-1, n_L, kvd).std(0) + 1e-5
for m in data: m['z'] = F.normalize(m['gist'] - MU, dim=1); m['kvn'] = (m['kv'] - KMU) / KSD
fids = list(range(len(data))); hold = set(fids[-max(1, len(data) // 4):])
tr = [m for i, m in enumerate(data) if i not in hold]; te = [m for i, m in enumerate(data) if i in hold]
print('trajs train=%d test=%d  KV[%d layers x %d]' % (len(tr), len(te), n_L, kvd), flush=True)
class KVComp(nn.Module):
    def __init__(s, kvd, n_L, d_m, D=384, heads=6, dh=64):
        super().__init__(); s.D = D; s.h = heads; s.dh = dh
        s.Wq = nn.Linear(D, heads * dh); s.Wk = nn.Linear(kvd, heads * dh); s.Wv = nn.Linear(kvd, heads * dh); s.Wo = nn.Linear(heads * dh, D)
        s.lemb = nn.Embedding(n_L, kvd); s.W = nn.Linear(D, D); s.log_tau = nn.Parameter(torch.zeros(D)); s.cz = nn.Linear(d_m, D); s.pred = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, d_m))
    def collect(s, kv, b):                                                       # attention-on-attention over the 16 layers' KV (depth)
        x = kv + s.lemb.weight; q = s.Wq(b).view(s.h, s.dh); K = s.Wk(x).view(-1, s.h, s.dh); V = s.Wv(x).view(-1, s.h, s.dh)
        a = torch.softmax(torch.einsum('hd,nhd->hn', q, K) / s.dh ** 0.5, dim=-1)
        return s.Wo(torch.einsum('hn,nhd->hd', a, V).reshape(-1))
    def run(s, m, use_comp=True):
        b = torch.zeros(s.D); h = torch.zeros(s.D); tau = F.softplus(s.log_tau) + 0.5; preds = []
        for t in range(m['kvn'].shape[0]):
            hp = h; a = s.collect(m['kvn'][t], b)
            for _ in range(2): b = b + (-b + torch.tanh(s.W(b) + a)) / tau / 2
            h = 0.9 * h + 0.1 * b
            preds.append(s.pred(torch.cat([s.cz(m['z'][t]), hp if use_comp else torch.zeros(s.D)])))
        return torch.stack(preds)
    def coh(s, m, use_comp=True):
        pr = s.run(m, use_comp)
        return None if pr.shape[0] < 2 else (F.normalize(pr[:-1], dim=-1) * m['z'][1:]).sum(-1)
net = KVComp(kvd, n_L, d_m); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(40 if SMOKE else 160):
    loss, n = 0.0, 0
    for m in tr:
        c = net.coh(m, True)
        if c is not None: loss = loss + (1 - c).mean(); n += 1
    opt.zero_grad(); (loss / n).backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    if ep % 40 == 0: print('ep %d loss %.3f' % (ep, float(loss / n)), flush=True)
with torch.no_grad():
    C = torch.cat([net.coh(m, True) for m in te]); C0 = torch.cat([net.coh(m, False) for m in te])
    posg = collections.defaultdict(list)
    for m in te:
        cc, c0 = net.coh(m, True), net.coh(m, False)
        for t in range(cc.shape[0]): posg[t].append(float(cc[t] - c0[t]))
print('\n=== LIQUID KV-CACHE COMPRESSOR (attention-on-attention over the 16 full-attn layers K/V) ===')
print('  next-gist coh WITH KV-compression : %+.3f' % float(C.mean()))
print('  next-gist coh WITHOUT (window)    : %+.3f' % float(C0.mean()))
print('  KV-COMPRESSION VALUE (gap)        : %+.3f   (layer-32 hidden baseline was +0.104)' % float(C.mean() - C0.mean()))
print('  gap by position:'); [print('    t=%2d  gap=%+.3f' % (t, sum(posg[t]) / len(posg[t]))) for t in sorted(posg)]
torch.save({'net': net.state_dict(), 'FULL': FULL}, '/home/pokazge/checkpoints/kv_compressor.pt')
print('=== ALL_DONE ===')
