# Diagnose: is weak actuation a MAGNITUDE limit or a DIRECTION limit? Sweep LoRA gain at inference
# on the trained ll2_unify. If bigger delta redirects coherently -> magnitude. If it degenerates
# before redirecting -> the learned steer direction is weak (needs retraining w/ stronger signal).
import json, urllib.request, math
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def cos(a, b):
    return sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b)) + 1e-9)

GAINS = [(1, 0.5), (3, 1.0), (6, 2.0), (10, None)]
MISSION = 'organize a community charity 5K run: secure the venue and date, recruit volunteers, promote and register runners, then plan race-day logistics'
zM = post('/encode', text=MISSION)['emb']

print('=== LoRA GAIN SWEEP on ll2_unify — does more magnitude redirect, or degenerate? ===')
print('MISSION held in belief:', MISSION[:70], '...\n')
for label, ctx in [('COLD TANGENT (no task context)', [{'role': 'user', 'content': 'By the way, who won the World Cup in 2018?'}]),
                   ('IN-REGIME (on-task turn, mission truncated)', None)]:
    post('/reset', mission=MISSION); post('/observe', text=MISSION)
    if ctx is None:
        r1 = post('/gen', messages=[{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + MISSION}], max_new=45)['text']
        ctx = [{'role': 'assistant', 'content': r1}, {'role': 'user', 'content': 'Okay, what should we focus on next?'}]
    print('--- %s ---' % label)
    base = post('/gen', messages=ctx, max_new=50, steered=False)['text']
    print('  UNSTEERED        (cos %.2f): %s' % (cos(post('/encode', text=base)['emb'], zM), base[:150].replace(chr(10), ' ')))
    for s, c in GAINS:
        post('/gain', scale=s, cap_rel=c)
        steer = post('/gen', messages=ctx, max_new=50, steered=True)['text']
        rm = post('/relmag')['mean']
        cs = cos(post('/encode', text=steer)['emb'], zM)
        print('  STEER s=%-2g cap=%-4s relmag=%.2f cos=%.2f: %s' % (s, str(c), rm, cs, steer[:150].replace(chr(10), ' ')))
    print()
print('=== ALL_DONE ===')
