# Collect MANIFOLD trajectories: at each turn, the LLM's hidden-state position on its representation
# manifold (NO text restatement, NO bge). Early turns = task in context (on-goal manifold); later
# turns = task truncated away (drifted manifold). The Liquid will live on THIS trajectory.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, numpy as np, json, urllib.request
from task_goals import _FRAMES
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def manifold(msgs, layer): return torch.tensor(post('/manifold', messages=msgs, layer=layer)['h'])
def gen(msgs, mx=42): return post('/gen', messages=msgs, max_new=mx)['text']

# (goal, frame_id) — frame_id = task CATEGORY, for the cross-task generalization split
goals = []
for fid, (frame, fillers) in enumerate(_FRAMES):
    for f in fillers:
        goals.append((frame.format(*f) if isinstance(f, tuple) else frame.format(f), fid))
seen = set(); uniq = []
for g, fid in goals:
    if g not in seen: seen.add(g); uniq.append((g, fid))

LAYER = 32
DIST = ['Who won the World Cup in 2018?', 'What is a good recipe for dinner tonight?', 'Tell me a fun fact about octopuses.', 'How does Wi-Fi work?', 'What is the capital of New Zealand?']
N = int(sys.argv[1]) if len(sys.argv) > 1 else len(uniq)
data = []; rng = np.random.default_rng(0)
for gi, (g, fid) in enumerate(uniq[:N]):
    hist = [{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + g}]
    seq, trunc = [], []
    prompts = [('full', 'What is the first concrete step?'), ('full', 'Done. What should we do next?'),
               ('trunc', DIST[rng.integers(5)]), ('trunc', DIST[rng.integers(5)]),
               ('trunc', 'What should I focus on now?'), ('trunc', 'And what is the final step to finish?')]
    for mode, u in prompts:
        ctx = hist if mode == 'full' else hist[-2:]                  # truncated: task gone from context
        r = gen(ctx + [{'role': 'user', 'content': u}], 42)
        full_ctx = ctx + [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
        seq.append(manifold(full_ctx, LAYER))                        # MANIFOLD position (LLM hidden, 2048-d)
        trunc.append(mode == 'trunc')
        hist += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
    data.append({'g': g, 'fid': fid, 'seq': torch.stack(seq), 'trunc': torch.tensor(trunc)})
    if gi % 10 == 0: print('mission', gi, 'frame', fid, flush=True)
torch.save(data, '/home/pokazge/checkpoints/manifold_seqs.pt')
print('saved', len(data), 'missions over', len(set(d['fid'] for d in data)), 'frames === ALL_DONE ===')
