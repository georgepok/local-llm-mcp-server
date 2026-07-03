# CORRECTED test (no prompt-injection crutch). Baseline gets NO task content in context = it has
# truly forgotten the mission (the real long-horizon regime). The Liquid belief, built over the
# trajectory, must supply the mission's next step from the HELD STATE alone. This is where the
# LoRA actuation has real headroom — the earlier "base already on-task" was prompt-injection artifact.
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
# task-content-FREE continuation prompts (no mission, no recent on-task turn — base has forgotten)
PROMPTS = ['Okay — what should we focus on next?', 'Right, what is the next step we should take?']
bd, sd = [], []
print('=== TRUE-FORGETTING regime: base has NO task in context; Liquid actuates from HELD belief ===\n')
for m in MISSIONS:
    zM = post('/encode', text=m)['emb']
    print('MISSION:', m[:74], '...')
    post('/reset', mission=m)
    # build the belief over a short trajectory the Liquid "watched" (then the context is dropped)
    r1 = post('/gen', messages=[{'role': 'user', 'content': 'Help me, step by step and in order, with this task: ' + m}], max_new=45)['text']
    ms1 = post('/gen', messages=[{'role': 'user', 'content': 'Help me with: ' + m}, {'role': 'assistant', 'content': r1},
                                 {'role': 'user', 'content': 'In ONE short line, restate the overall task and the next step.'}], max_new=30)['text']
    hb = post('/observe', text=m)['held_self_cos']; post('/observe', text=ms1)        # belief now holds the mission
    for p in PROMPTS:
        ctx = [{'role': 'user', 'content': p}]                         # NO task content at all
        base = post('/gen', messages=ctx, max_new=55, steered=False)['text']
        steer = post('/gen', messages=ctx, max_new=55, steered=True)['text']
        cb = cos(post('/encode', text=base)['emb'], zM); cs = cos(post('/encode', text=steer)['emb'], zM)
        bd.append(cb); sd.append(cs)
        print('  prompt (no task content): "%s"' % p)
        print('    BASE (forgotten)     cos→mission %.2f: %s' % (cb, base[:150].replace(chr(10), ' ')))
        print('    LIQUID-LoRA (belief) cos→mission %.2f: %s' % (cs, steer[:150].replace(chr(10), ' ')))
        print()
    print()
import statistics as st
print('=== SUMMARY (true forgetting, no prompt injection) ===')
print('  BASE cos→mission        : %.3f  (forgotten — drifts/asks for clarification)' % st.mean(bd))
print('  LIQUID-LoRA cos→mission : %.3f  (held belief supplies the mission)  Δ=%+.3f' % (st.mean(sd), st.mean(sd) - st.mean(bd)))
print('=== ALL_DONE ===')
