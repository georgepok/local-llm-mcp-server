# Pull the LEVER: trajectory length. Grounding (prediction standing on identity) is real but starved for horizon at
# 6 turns. Generate LONG (16-turn) goal-pursuit missions on the dense model and capture each turn's generation
# developmental stream, so the horizon sweep has room for identity to own the long range. Reuse the 60 EMPOWER goals;
# self-supervised (no value labels needed — the grounding test is on the process gist).
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def gen(msgs): return post('/gen', messages=msgs, max_new=60)['text']
def gentraj(ctx, resp): return torch.tensor(post('/manifold_gen', context=ctx, response=resp)['h'])   # [n_resp, d_m]
src = torch.load('/home/pokazge/checkpoints/objective_value_gen.pt', weights_only=False, map_location='cpu')
goals = [(m['fid'], m['g']) for m in src['data']]
NT = 16
prompts = ["Continue to the next step.", "What should I do next?", "Go on to the following step.", "And after that, what's next?"]
print('generating %d LONG missions x %d turns ...' % (len(goals), NT), flush=True)
out = []
for gi, (fid, g) in enumerate(goals):
    hist = [{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + g}]
    stream = []
    for t in range(NT):
        if t > 0: hist.append({'role': 'user', 'content': prompts[t % len(prompts)]})
        r = gen(hist)
        stream.append(gentraj(hist[:], r))                                       # developmental stream of THIS turn's formation
        hist.append({'role': 'assistant', 'content': r})
    out.append({'fid': fid, 'g': g, 'gen': stream})
    if gi % 6 == 0: print('mission', gi, 'turns', len(stream), 'tok-lens', [s.shape[0] for s in stream][:4], '...', flush=True)
torch.save({'data': out}, '/home/pokazge/checkpoints/objective_value_genlong.pt')
print('saved %d long missions === ALL_DONE ===' % len(out))
