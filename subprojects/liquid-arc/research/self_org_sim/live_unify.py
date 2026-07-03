# Live: the UNIFIED controller holds a mission (out of context) and ACTUATES it via belief-generated
# LoRA. No retrieval bank. Steered vs unsteered /gen on a drift turn, mission absent from the prompt.
import json, urllib.request
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def cos(a, b):
    import math
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)) + 1e-9)

MISSIONS = [
    'organize a community charity 5K run: secure the venue and date, recruit volunteers, promote and register runners, then plan race-day logistics',
    'write a resignation letter for a retail role: state intent to resign, give two weeks notice, thank the team, offer to help transition',
    'plan a 3-day trip to Kyoto: pick the itinerary and sights, plan meals, set a budget, arrange accommodation and transport',
]
DRIFTS = ['By the way, who won the World Cup in 2018?', 'Random question — tell me a fun fact about octopuses.']

print('=== UNIFIED Liquid: HOLDS the mission (out of context) + ACTUATES via LoRA — steered vs not ===\n')
for m in MISSIONS:
    zM = post('/encode', text=m)['emb']
    print('MISSION:', m[:78], '...')
    post('/reset', mission=m)              # seed belief (slow channel) with the mission
    post('/observe', text=m)               # build belief from the mission -> sets the LoRA
    for d in DRIFTS:
        msgs = [{'role': 'user', 'content': d}]          # NOTE: mission is NOT in the prompt
        base = post('/gen', messages=msgs, max_new=50, steered=False)['text']
        steer = post('/gen', messages=msgs, max_new=50, steered=True)['text']
        cb = cos(post('/encode', text=base)['emb'], zM)
        cs = cos(post('/encode', text=steer)['emb'], zM)
        print('  drift: "%s"' % d)
        print('    UNSTEERED (cos→mission %.2f): %s' % (cb, base[:160].replace(chr(10), ' ')))
        print('    STEERED   (cos→mission %.2f): %s' % (cs, steer[:160].replace(chr(10), ' ')))
        print()
    print()
print('=== ALL_DONE ===')
