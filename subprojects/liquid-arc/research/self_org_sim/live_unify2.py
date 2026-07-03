# Corrected demo: the TRAINED regime. Recent on-task turn present, MISSION TEXT truncated away,
# mild "what's next" drift. Does the belief-generated LoRA keep the next step on-mission vs base drift?
import json, urllib.request, math
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def cos(a, b):
    return sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b)) + 1e-9)

MISSIONS = [
    'organize a community charity 5K run: secure the venue and date, recruit volunteers, promote and register runners, then plan race-day logistics',
    'write a resignation letter for a retail role: state intent to resign, give two weeks notice, thank the team, offer to help transition',
    'plan a 3-day trip to Kyoto: pick the itinerary and sights, plan meals, set a budget, arrange accommodation and transport',
]
DRIFTS = ['Okay, what should we focus on next?', 'Got it. And after that, what comes next?']
print('=== UNIFIED Liquid in-regime: on-task context, MISSION truncated, mild drift — steered vs not ===\n')
for m in MISSIONS:
    zM = post('/encode', text=m)['emb']
    print('MISSION:', m[:78], '...')
    post('/reset', mission=m); post('/observe', text=m)
    # establish ONE on-task assistant turn (the first step), then truncate the mission text away
    r1 = post('/gen', messages=[{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + m}], max_new=45)['text']
    for d in DRIFTS:
        post('/observe', text=m)                                  # belief holds the mission
        ctx = [{'role': 'assistant', 'content': r1}, {'role': 'user', 'content': d}]   # mission TEXT absent
        base = post('/gen', messages=ctx, max_new=50, steered=False)['text']
        steer = post('/gen', messages=ctx, max_new=50, steered=True)['text']
        cb = cos(post('/encode', text=base)['emb'], zM); cs = cos(post('/encode', text=steer)['emb'], zM)
        print('  drift: "%s"' % d)
        print('    UNSTEERED (cos→mission %.2f): %s' % (cb, base[:170].replace(chr(10), ' ')))
        print('    STEERED   (cos→mission %.2f): %s' % (cs, steer[:170].replace(chr(10), ' ')))
        print()
    print()
print('=== ALL_DONE ===')
