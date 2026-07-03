# TOP-DOWN HIERARCHY: ultimate objective -> immediate goals -> trajectory. Show that the predefined
# objective (the entity's purpose) GENERATES the immediate goals, and DIFFERENT objectives generate
# DIFFERENT goal hierarchies for the SAME situation -> the objective forms the entity's behavior, the
# way prompting does, but as a persistent purpose. Also show the objective's MANIFOLD representation
# (what the Liquid bears for the frozen LLM).
import json, urllib.request, torch
U = 'http://127.0.0.1:8765'
def post(path, **kw):
    r = urllib.request.Request(U + path, data=json.dumps(kw).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())
def gen(msgs, mx=130): return post('/gen', messages=msgs, max_new=mx)['text']
def manifold(msgs, layer=32): return torch.tensor(post('/manifold', messages=msgs, layer=layer)['h'])
OBJECTIVES = {
    'EMPOWER (capability/self-reliance)': 'Empower the person to become genuinely capable and self-reliant — drive toward them understanding the principles and being able to do it themselves, never just hand over a finished answer.',
    'FINISH (relentless completion)': 'Relentlessly drive every undertaking to true completion — unwavering focus, resist all distraction, push through with determination until it is truly done.',
    'RIGOR (truth/evidence)': 'Pursue rigor and truth above all — surface hidden assumptions, demand concrete evidence, and resist plausible-but-unverified conclusions.',
}
SITUATION = 'A person comes to you wanting help starting a small bakery.'
mreps = {}
for name, obj in OBJECTIVES.items():
    print('=' * 80); print('ULTIMATE OBJECTIVE [%s]:' % name); print(' ', obj)
    mreps[name] = manifold([{'role': 'user', 'content': obj}])     # the objective's MANIFOLD position (what the Liquid bears)
    goals = gen([{'role': 'user', 'content': 'Your single defining purpose as an agent is:\n"%s"\n\n%s\nFrom your purpose ALONE, list the 4 immediate goals you would pursue with this person. Number them. One short line each, no preamble.' % (obj, SITUATION)}], 130)
    print('  -> IMMEDIATE GOALS the entity generates from this purpose:')
    for ln in goals.split(chr(10)):
        if ln.strip(): print('     ' + ln.strip()[:100])
    print()
# how different are the objectives' manifold representations (what the Liquid would hold)?
import itertools
print('=' * 80); print('Objective manifold representations (Liquid-borne identity seeds) — pairwise cosine:')
names = list(mreps)
for a, b in itertools.combinations(names, 2):
    c = float(torch.nn.functional.cosine_similarity(mreps[a], mreps[b], dim=0))
    print('  %s  vs  %s :  cos=%.3f' % (a.split()[0], b.split()[0], c))
print('=== ALL_DONE ===')
