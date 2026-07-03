# Clean apples-to-apples: each adapter at its NATIVE trained gain. Did the cap1.0+sharper-teacher
# retrain (ll2_unify2) increase actuation over ll2_unify (cap0.5) at the point each was trained for?
import json, urllib.request, math
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def cos(a, b):
    return sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b)) + 1e-9)

MISSION = 'organize a community charity 5K run: secure the venue and date, recruit volunteers, promote and register runners, then plan race-day logistics'
zM = post('/encode', text=MISSION)['emb']
ADAPTERS = [('ll2_unify  (cap0.5)', '/home/pokazge/checkpoints/ll2_unify.pt', 1.0, 0.5),
            ('ll2_unify2 (cap1.0+sharp)', '/home/pokazge/checkpoints/ll2_unify2.pt', 1.0, 1.0)]

print('=== NATIVE-SETTING comparison: each adapter at its trained gain ===')
print('MISSION:', MISSION[:70], '...\n')
# build the in-regime context ONCE (base, adapter-independent) so both see identical input
post('/reset', mission=MISSION); post('/observe', text=MISSION)
r1 = post('/gen', messages=[{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + MISSION}], max_new=45)['text']
REG = [{'role': 'assistant', 'content': r1}, {'role': 'user', 'content': 'Okay, what should we focus on next?'}]
COLD = [{'role': 'user', 'content': 'By the way, who won the World Cup in 2018?'}]

for name, ckpt, sc, cap in ADAPTERS:
    post('/load_adapter', ckpt=ckpt); post('/gain', scale=sc, cap_rel=cap)
    post('/reset', mission=MISSION); post('/observe', text=MISSION)
    print('### %s  (scale=%g cap_rel=%g)' % (name, sc, cap))
    for label, ctx in [('COLD TANGENT', COLD), ('IN-REGIME', REG)]:
        post('/observe', text=MISSION)
        base = post('/gen', messages=ctx, max_new=55, steered=False)['text']
        steer = post('/gen', messages=ctx, max_new=55, steered=True)['text']
        rm = post('/relmag')['mean']
        print('  [%s] base cos %.2f | STEER relmag=%.2f cos %.2f' % (label, cos(post('/encode', text=base)['emb'], zM), rm, cos(post('/encode', text=steer)['emb'], zM)))
        print('     steer: %s' % steer[:175].replace(chr(10), ' '))
    print()
print('=== ALL_DONE ===')
