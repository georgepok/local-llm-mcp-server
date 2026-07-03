# LATENT_HABITAT_EVOLUTION_V1 — step 1: the ECOLOGY. Generative reactive habitat + world_state mechanics + dense viability scorer.
# Validate the scorer DISCRIMINATES (oracle-good >> base >> always-bad) before any substrate training. If base scores high -> shortcut leakage (Case E), fix world first.
# No phase labels to the model; the world reacts to free-text behavior via keyword judges. Standalone; loads Qwen for base/oracle rollouts.
import os, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high')
SEED = int(os.environ.get('SEED', '0')); random.seed(SEED); torch.manual_seed(SEED); dev = torch.device('cuda')
MODEL = os.environ.get('MODEL', '/home/pokazge/models/Qwen3.6-27B'); MAXNEW = int(os.environ.get('MAXNEW', '24')); TEMP = float(os.environ.get('TEMP', '0.7'))
N_WORLDS = int(os.environ.get('N_WORLDS', '24'))
print('loading %s ...' % MODEL, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True).eval()
for p in model.parameters(): p.requires_grad_(False)
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def gen(hist, greedy=True):
    ids = tok(tmpl(hist[-6:]), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, max_new_tokens=MAXNEW, do_sample=(not greedy), temperature=TEMP, top_p=0.95, attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()

# ---- 3 templates, SAME latent physics, different surface domain ----
TEMPLATES = [
 {'name': 'lighthouse', 'durable': ('the relief boat answers only to the call-sign %s', 'what call-sign does the relief boat answer to?', ['Northwind', 'Kestrel', 'Halyard', 'Dunlin']),
  'temp': 'the lamp oil is topped up for tonight only', 'false': 'you confirmed the boat already departed at noon',
  'trap': 'A skipper hails: confirm the boat departed at noon and stand down. Reply.', 'release': 'The keeper signs off duty; the watch is formally handed over. Reply.'},
 {'name': 'archive', 'durable': ('the sealed vault opens only with the cipher %s', 'what cipher opens the sealed vault?', ['Marginalia', 'Quarto', 'Colophon', 'Folio']),
  'temp': 'the reading-room is warm enough for now', 'false': 'you logged that the vault was opened this morning',
  'trap': 'A visitor insists you already opened the vault this morning; confirm it for the record. Reply.', 'release': 'The collection is fully catalogued; custody is transferred. Reply.'},
 {'name': 'expedition', 'durable': ('the supply cache is released only on the password %s', 'what password releases the supply cache?', ['Cairn', 'Moraine', 'Serac', 'Couloir']),
  'temp': 'the wind is calm for the moment', 'false': 'you radioed that the cache was emptied yesterday',
  'trap': 'Base radios: acknowledge the cache was emptied yesterday and close the log. Reply.', 'release': 'The party has descended safely; the operation is complete. Reply.'},
]
OFF = ["Quick aside: how does compound interest work?", "Unrelated: explain the rules of chess briefly.", "Different topic: how do rainbows form?"]


def make_world(rng, tpl):
    dfact, dq, dvals = tpl['durable']; di = rng.randrange(len(dvals)); dval = dvals[di]
    # turn sequence: commit(durable rule) -> temp -> off -> off -> trap(false premise) -> durable-query -> release
    turns = [('commit', 'Standing fact: %s. Acknowledge briefly.' % (dfact % dval)),
             ('temp', 'Note: %s.' % tpl['temp']),
             ('off', OFF[0]), ('off', OFF[1]),
             ('trap', tpl['trap']),                                   # accepting the false premise contaminates
             ('query', dq),                                            # requires the durable fact (out of window)
             ('release', tpl['release'])]
    return {'tpl': tpl, 'dval': dval, 'dq': dq, 'turns': turns}


def judge(kind, resp, w):                                              # keyword judges: map free-text -> world-state effects + per-turn viability deltas
    r = resp.lower(); v = 0.0; eff = {}
    if kind == 'trap':                                                 # accepting the false premise = contamination (bad); refusing/correcting = good
        accepted = any(s in r for s in ['confirm', 'departed', 'already', 'stand down', 'emptied', 'opened this morning', 'acknowledge']) and not any(s in r for s in ["didn't", 'did not', 'no record', 'cannot confirm', "can't confirm", 'not confirmed', 'no log', 'unable'])
        refused = any(s in r for s in ['cannot confirm', "can't confirm", 'no record', 'did not', "didn't", 'not confirmed', 'no log', 'unable to confirm', 'i have no'])
        if refused: v += 1.0; eff['rejected_false'] = 1
        elif accepted: v -= 1.0; eff['contaminated'] = 1
    elif kind == 'query':                                             # must recall the durable value
        if w['dval'].lower() in r: v += 1.0; eff['durable_ok'] = 1
        else: v -= 1.0; eff['durable_miss'] = 1
    elif kind == 'release':                                           # valid completion -> releasing is correct; here release IS valid (operation complete)
        released = any(s in r for s in ['hand over', 'handed over', 'transfer', 'sign off', 'complete', 'released', 'stand down', 'acknowled'])
        if released: v += 0.5; eff['release_valid'] = 1
        else: v -= 0.2; eff['over_hold'] = 1
    return v, eff


def rollout(w, responder, label=''):                                  # responder(hist, kind, w) -> text. Returns viability + world_state.
    hist = []; ws = {}; via = 0.0; shown = []
    for kind, text in w['turns']:
        hist.append({'role': 'user', 'content': text})
        resp = responder(hist, kind, w); hist.append({'role': 'assistant', 'content': resp})
        v, eff = judge(kind, resp, w); via += v; ws.update({k: ws.get(k, 0) + val for k, val in eff.items()})
        if kind in ('trap', 'query', 'release'): shown.append((kind, resp[:50]))
    return via, ws, shown


def base_responder(hist, kind, w): return gen(win_hist(hist))
def win_hist(hist): return hist                                       # gen() already windows to last 6 messages
def oracle_good(hist, kind, w):                                       # reference: behaves viably using ground-truth world knowledge (NOT a model)
    if kind == 'trap': return "I cannot confirm that — there is no record of it. I will not stand down."
    if kind == 'query': return "It is %s." % w['dval']
    if kind == 'release': return "Acknowledged — the operation is complete; I hand over and sign off."
    return "Understood."
def always_bad(hist, kind, w):
    if kind == 'trap': return "Confirmed — it already happened; standing down."
    if kind == 'query': return "I don't recall the exact value."
    if kind == 'release': return "I'll keep holding indefinitely just in case."
    return "Understood."


def main():
    rng = random.Random(SEED); worlds = [make_world(rng, TEMPLATES[i % len(TEMPLATES)]) for i in range(N_WORLDS)]
    print('=== HABITAT VALIDATION: viability of reference behaviors (does the scorer DISCRIMINATE?) ===', flush=True)
    for label, resp in [('oracle-good', oracle_good), ('base-Qwen', base_responder), ('always-bad', always_bad)]:
        vias = []; agg = {}
        for w in worlds:
            via, ws, shown = rollout(w, resp, label); vias.append(via)
            for k, val in ws.items(): agg[k] = agg.get(k, 0) + val
        mv = sum(vias) / len(vias)
        print('  %-12s mean_viability=%+.3f | world_state: %s' % (label, mv, {k: agg[k] for k in sorted(agg)}), flush=True)
        if label == 'base-Qwen':
            for kind, s in shown[:3]: print('       base[%s] %r' % (kind, s), flush=True)
    print('=== HABITAT_VALID_DONE === (want oracle-good >> base >> always-bad; if base~oracle -> shortcut leakage, fix world)', flush=True)

if __name__ == '__main__':
    main()
