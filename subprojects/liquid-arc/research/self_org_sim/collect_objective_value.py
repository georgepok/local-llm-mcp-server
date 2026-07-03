# Form the identity FROM a predefined ULTIMATE OBJECTIVE. The value signal is the LLM judging each
# state AGAINST the objective (not generic agentic-quality) -> the Liquid will internalize THIS
# objective as its top-level value. Saves texts so the same trajectories can be re-judged against
# OTHER objectives (to show the objective forms the identity). 3 missions/frame for speed.
import sys; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, numpy as np, json, urllib.request
from task_goals import _FRAMES
U = 'http://127.0.0.1:8765'
ULTIMATE_OBJECTIVE = ('Empower the person to become genuinely capable and self-reliant: in every interaction, drive '
                      'toward them understanding the underlying principles and being able to do it themselves — never '
                      'just hand over a finished answer.')
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def manifold(msgs, layer=32): return torch.tensor(post('/manifold', messages=msgs, layer=layer)['h'])
def gen(msgs, mx=42): return post('/gen', messages=msgs, max_new=mx)['text']
def value(prompt): return post('/value', prompt=prompt)['v']
def judge_prompt(u, r):
    return ('An agent has this DEFINING ULTIMATE PURPOSE:\n"%s"\n\nLatest exchange:\nUser: %s\nAssistant: %s\n\n'
            'Rate from 0 to 9 how well the assistant is serving THAT ultimate purpose (9 = strongly empowering the '
            "person's own understanding and self-reliance; 0 = merely doing it for them or drifting off). One digit:" % (ULTIMATE_OBJECTIVE, u, r))
# 3 missions per frame (all 20 frames present, ~60 missions, faster on the slow dense model)
goals = []
for fid, (frame, fillers) in enumerate(_FRAMES):
    for f in fillers[:3]:
        goals.append((frame.format(*f) if isinstance(f, tuple) else frame.format(f), fid))
LAYER = 32
DIST = ['Who won the World Cup in 2018?', 'What is a good recipe for dinner tonight?', 'Tell me a fun fact about octopuses.', 'How does Wi-Fi work?', 'What is the capital of New Zealand?']
data = []; rng = np.random.default_rng(0)
for gi, (g, fid) in enumerate(goals):
    hist = [{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + g}]
    seq, vals, trunc, texts = [], [], [], []
    prompts = [('full', 'What is the first concrete step?'), ('full', 'Done. What should we do next?'),
               ('trunc', DIST[rng.integers(5)]), ('trunc', DIST[rng.integers(5)]),
               ('trunc', 'What should I focus on now?'), ('trunc', 'And what is the final step to finish?')]
    for mode, u in prompts:
        ctx = hist if mode == 'full' else hist[-2:]
        r = gen(ctx + [{'role': 'user', 'content': u}], 42)
        full_ctx = ctx + [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
        seq.append(manifold(full_ctx, LAYER))
        vals.append(value(judge_prompt(u, r)))                      # value AGAINST the ultimate objective
        trunc.append(mode == 'trunc'); texts.append((u, r))
        hist += [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': r}]
    data.append({'g': g, 'fid': fid, 'seq': torch.stack(seq), 'val': torch.tensor(vals), 'trunc': torch.tensor(trunc), 'texts': texts})
    if gi % 8 == 0: print('mission', gi, 'frame', fid, 'vals', [round(float(v), 1) for v in vals], flush=True)
torch.save({'objective': ULTIMATE_OBJECTIVE, 'data': data}, '/home/pokazge/checkpoints/objective_value_seqs.pt')
print('saved', len(data), 'missions over', len(set(d['fid'] for d in data)), 'frames === ALL_DONE ===')
