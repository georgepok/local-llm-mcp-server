# Regenerate self-feeding drift trajectories SAVING TEXTS (needed to reconstruct windowed-vs-full context for the
# injection demo) alongside each chunk's developmental stream. Same self-feed + sampling as before. Incremental save.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=1200).read())
def gen(msgs, temp): return post('/gen', messages=msgs, max_new=44, temp=temp)['text']
def gentraj(ctx, r): return torch.tensor(post('/manifold_gen', context=ctx, response=r)['h'])
src = torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')
seeds = [(d['fid'], d['g']) for d in src['data']][:16]
NT, TEMP = 14, 0.85
print('regen %d drift trajectories x %d steps WITH TEXTS ...' % (len(seeds), NT), flush=True)
out = []
for si, (fid, seed) in enumerate(seeds):
    hist = [{'role': 'user', 'content': seed}]; stream = []; texts = []
    for step in range(NT):
        r = gen(hist, TEMP)
        try: stream.append(gentraj(hist[:], r))
        except Exception as e: print('  gentraj err s%d t%d: %r' % (si, step, e), flush=True); break
        texts.append(r)
        hist = hist + [{'role': 'assistant', 'content': r}, {'role': 'user', 'content': r}]
    out.append({'fid': fid, 'seed': seed, 'texts': texts, 'gen': stream})
    torch.save({'data': out}, '/home/pokazge/checkpoints/objective_drift_txt.pt')
    print('seed %d/%d steps=%d' % (si + 1, len(seeds), len(stream)), flush=True)
print('=== ALL_DONE ===')
