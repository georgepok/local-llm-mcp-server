# Generate SELF-FEEDING drift trajectories (sampling) — the substrate for the Liquid-as-context-compressor. Each seed
# drifts under its own output fed back as the next prompt, FULL context (so the seed influences all later chunks — that
# is what the compressor must recover when the window drops it). Capture each chunk's generation developmental stream
# ([n_tok, d_m]) for AoA compression. Incremental save after every seed (SSH-drop resilient).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=1200).read())
def gen(msgs, temp): return post('/gen', messages=msgs, max_new=44, temp=temp)['text']
def gentraj(ctx, r): return torch.tensor(post('/manifold_gen', context=ctx, response=r)['h'])   # [n_tok, d_m]
src = torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')
seeds = [(d['fid'], d['g']) for d in src['data']][:24]
NT, TEMP = 16, 0.85
print('generating %d self-feeding drift trajectories x %d steps (temp %.2f) ...' % (len(seeds), NT, TEMP), flush=True)
out = []
for si, (fid, seed) in enumerate(seeds):
    hist = [{'role': 'user', 'content': seed}]; stream = []
    for step in range(NT):
        r = gen(hist, TEMP)
        try: stream.append(gentraj(hist[:], r))
        except Exception as e: print('  gentraj err s%d t%d: %r' % (si, step, e), flush=True); break
        hist = hist + [{'role': 'assistant', 'content': r}, {'role': 'user', 'content': r}]   # self-feed: output -> next prompt
    out.append({'fid': fid, 'seed': seed, 'gen': stream})
    torch.save({'data': out}, '/home/pokazge/checkpoints/objective_drift.pt')               # incremental
    print('seed %d/%d  steps=%d  toklens=%s' % (si + 1, len(seeds), len(stream), [s.shape[0] for s in stream][:5]), flush=True)
print('=== ALL_DONE ===')
