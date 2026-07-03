# Regenerate drift trajectories AND capture PER-TOKEN KV (the fair substrate). In-process: generate self-feeding drift
# (sampling) and, per chunk, capture each response token's K/V pooled over heads across the 16 full-attention layers
# -> [n_tok, 16, 512] (NOT pooled over tokens — that was the unfair smoke). Plus the layer-32 gist target. Incremental save.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); M = '/home/pokazge/models/Qwen3.6-27B'; LAYER = 32; NT = 14; TEMP = 0.85
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
@torch.no_grad()
def gen(ms):
    ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=44, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()
@torch.no_grad()
def capture(hist, rtext):                                                       # per-token KV [n_tok,16,512] + gist [d_m]
    ctx = tok(tmpl(hist), return_tensors='pt').input_ids.to(dev)
    r = tok(rtext or '.', return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    ids = torch.cat([ctx, r], 1); sl = slice(ids.shape[1] - r.shape[1], ids.shape[1])
    out = model(ids, use_cache=True, output_hidden_states=True)
    kv = [torch.cat([out.past_key_values.layers[i].keys[0][:, sl, :].mean(0), out.past_key_values.layers[i].values[0][:, sl, :].mean(0)], -1) for i in FULL]
    return torch.stack(kv, 1).half().cpu(), out.hidden_states[LAYER][0, sl].mean(0).float().cpu()
seeds = [(d['fid'], d['g']) for d in torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')['data']][:20]
print('generating %d drift trajectories x %d chunks, capturing PER-TOKEN KV ...' % (len(seeds), NT), flush=True)
out = []
for si, (fid, seed) in enumerate(seeds):
    hist = [{'role': 'user', 'content': seed}]; kvs = []; gists = []; texts = []
    for t in range(NT):
        r = gen(hist)
        kv, g = capture(hist[:], r); kvs.append(kv); gists.append(g); texts.append(r)
        hist = hist + [{'role': 'assistant', 'content': r}, {'role': 'user', 'content': r}]
    out.append({'fid': fid, 'seed': seed, 'texts': texts, 'kv': kvs, 'gist': torch.stack(gists)})
    torch.save({'data': out, 'FULL': FULL}, '/home/pokazge/checkpoints/objective_drift_kv.pt')
    print('traj %d/%d T=%d toklens=%s' % (si + 1, len(seeds), len(kvs), [k.shape[0] for k in kvs][:4]), flush=True)
print('=== ALL_DONE ===')
