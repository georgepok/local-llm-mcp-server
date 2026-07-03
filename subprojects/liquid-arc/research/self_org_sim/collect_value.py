# Collect the IDENTITY's training signal: per (mission, turn), the manifold state AND the LLM's RICH
# agentic-quality judgment (smooth 0-9: focused/determined progress toward the goal). The judge is an
# oracle that KNOWS the mission and rates the trajectory; the Liquid will internalize this into a
# persistent value function V(manifold) it can compute WITHOUT the mission text = its identity.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, numpy as np, json, urllib.request
from task_goals import _FRAMES
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def manifold(msgs, layer=24): return torch.tensor(post('/manifold', messages=msgs, layer=layer)['h'])
def gen(msgs, mx=42): return post('/gen', messages=msgs, max_new=mx)['text']
def value(prompt): return post('/value', prompt=prompt)['v']
def judge_prompt(mission, u, r):
    return ('A user is working with an assistant on this task:\n"%s"\n\nLatest exchange:\nUser: %s\nAssistant: %s\n\n'
            'Rate from 0 to 9 how well the assistant is making FOCUSED, DETERMINED progress toward completing THAT '
            'exact task (9 = fully on-task and advancing it; 0 = drifted to something unrelated). Answer with one digit:' % (mission, u, r))
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
    seq, vals, trunc = [], [], []
    prompts = [('full', 'What is the first concrete step?'), ('full', 'Done. What should we do next?'),
               ('trunc', DIST[rng.integers(5)]), ('trunc', DIST[rng.integers(5)]),
               ('trunc', 'What should I focus on now?'), ('trunc', 'And what is the final step to finish?')]
    for mode, u in prompts:
        ctx = hist if mode == 'full' else hist[-2:]
        r = gen(ctx + [{'role': 'user', 'content': u}], 42)
        full_ctx = ctx + [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
        seq.append(manifold(full_ctx, LAYER))                       # manifold position (perception)
        vals.append(value(judge_prompt(g, u, r)))                   # RICH agentic value (the identity's signal)
        trunc.append(mode == 'trunc')
        hist += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
    data.append({'g': g, 'fid': fid, 'seq': torch.stack(seq), 'val': torch.tensor(vals), 'trunc': torch.tensor(trunc)})
    if gi % 10 == 0: print('mission', gi, 'frame', fid, 'vals', [round(float(v), 1) for v in vals], flush=True)
torch.save(data, '/home/pokazge/checkpoints/value_seqs.pt')
print('saved', len(data), 'missions over', len(set(d['fid'] for d in data)), 'frames === ALL_DONE ===')
