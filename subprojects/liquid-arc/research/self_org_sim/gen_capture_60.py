# Regenerate the drift dataset at PROPER SCALE: all 60 seeds (20 categories x 3) instead of [:20] -> 60 trajectories, so a
# 1/4 cross-category hold-out is ~15 trajectories (was 2 — the eval was underpowered). ONE in-process pass captures
# everything CONSISTENTLY per chunk from the same full-context forward: texts, the layer-32 developmental stream (gist
# target + layer-32-AoA baseline), and the BROAD native KV (keys+values at all full-attn layers, mean over heads) for the
# native-AoA read. Fixes both n=2 AND the earlier full-vs-windowed native-read inconsistency. Self-feed, FULL context, temp 0.85.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.manual_seed(0); dev = torch.device('cuda'); MODEL = '/home/pokazge/models/Qwen3.6-27B'; LAYER, NT, TEMP, MAXNEW = 32, 14, 0.85, 44
src = torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')['data']
seeds = [(d['fid'], d['g']) for d in src]                                         # ALL 60 (was [:20])
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def probe_full():
    ids = tok('hi', return_tensors='pt').input_ids.to(dev); out = model(ids, use_cache=True)
    return [i for i, L in enumerate(out.past_key_values.layers) if getattr(L, 'keys', None) is not None]
FULL = probe_full(); print('full-attn layers (%d): %s' % (len(FULL), FULL), flush=True)
@torch.no_grad()
def gen_chunk(ms):
    ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    return o[0, ids.shape[1]:]
@torch.no_grad()
def capture(ms, rids):                                                           # one full-context forward -> layer-32 stream + broad native KV for the chunk tokens
    cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); nct = rids.shape[0]
    out = model(ids, output_hidden_states=True, use_cache=True)
    g = out.hidden_states[LAYER][0, -nct:].float().cpu()                         # [nct, d_m] layer-32 stream
    feats = []
    for L in FULL:
        lc = out.past_key_values.layers[L]; feats.append(lc.keys[0, :, -nct:, :].mean(0)); feats.append(lc.values[0, :, -nct:, :].mean(0))
    return g, torch.cat(feats, dim=-1).float().cpu()                            # [nct, len(FULL)*2*hd]
print('generating %d drift trajectories x %d chunks (full capture) ...' % (len(seeds), NT), flush=True)
out = []
for si, (fid, seed) in enumerate(seeds):
    hist = [{'role': 'user', 'content': seed}]; texts = []; gen = []; nkv = []
    for step in range(NT):
        rids = gen_chunk(hist); txt = tok.decode(rids, skip_special_tokens=True).split('</think>')[-1].strip()
        try: g, k = capture(hist[:], rids)
        except Exception as e: print('  capture err s%d t%d: %r' % (si, step, e), flush=True); break
        texts.append(txt); gen.append(g); nkv.append(k)
        hist += [{'role': 'assistant', 'content': txt}, {'role': 'user', 'content': txt}]
    out.append({'fid': fid, 'seed': seed, 'texts': texts, 'gen': gen, 'nkv': nkv})
    torch.save({'data': out, 'full': FULL}, '/home/pokazge/checkpoints/objective_drift60.pt')
    if (si + 1) % 5 == 0 or si == 0: print('  seed %d/%d (fid %d) chunks=%d' % (si + 1, len(seeds), fid, len(texts)), flush=True)
print('=== ALL_DONE === saved %d trajectories' % len(out), flush=True)
