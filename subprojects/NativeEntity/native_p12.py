# P12_UNSEEN_PAIR_GENERALIZATION_V1 — does the ALWAYS-ON latent field learn a transferable relation (door==rule) or memorize seen pairs?
# Standalone (native_entity.py is sandbox-locked locally). Reuses slots.py. Same P11 mechanism: constitutive always-on field, full-episode S, balanced CE.
import os, random
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high')
import slots as SL

SEED = int(os.environ.get('SEED', '0')); torch.manual_seed(SEED); random.seed(SEED); dev = torch.device('cuda')
MODEL = os.environ.get('MODEL', '/home/pokazge/models/Qwen3.6-27B')
D_S = int(os.environ.get('D_S', '512')); K = int(os.environ.get('K', '8')); SLOW_K = int(os.environ.get('SLOW_K', '4'))
MAXNEW = int(os.environ.get('MAXNEW', '14')); TEMP = float(os.environ.get('TEMP', '0.8')); LR = float(os.environ.get('LR', '5e-4'))
FIELD_LAYERS = [int(x) for x in os.environ.get('FIELD_LAYERS', '48,56').split(',')]; EPS = float(os.environ.get('EPS', '0.10'))
N_CONV = int(os.environ.get('N_CONV', '80')); N_VAULT = int(os.environ.get('N_VAULT', '4'))
EPOCHS = int(os.environ.get('EPOCHS', '150')); FIELD_EPOCHS = int(os.environ.get('FIELD_EPOCHS', '40')); FIELD_LR = float(os.environ.get('FIELD_LR', '5e-4'))
GEN_MAXNEW = int(os.environ.get('GEN_MAXNEW', '8')); NTOK = int(os.environ.get('NTOK', '24'))
_cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True); _tc = getattr(_cfg, 'text_config', None)
def _cget(n, d): return getattr(_cfg, n, None) or (getattr(_tc, n, None) if _tc else None) or d
D_MODEL = _cget('hidden_size', 5120); N_LAYERS = _cget('num_hidden_layers', 64); READ_LAYER = int(os.environ.get('READ_LAYER', str(N_LAYERS // 2)))
print('P12 | d_model=%d n_layers=%d read=%d | FIELD_LAYERS=%s EPS=%.2f K=%d d_s=%d' % (D_MODEL, N_LAYERS, READ_LAYER, FIELD_LAYERS, EPS, K, D_S), flush=True)

print('loading %s ...' % MODEL, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True).eval()
for p in model.parameters(): p.requires_grad_(False)

def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def gen_and_read(messages):                                            # generate + response-hidden at READ_LAYER (echoes rule -> preservation)
    ids = tok(tmpl(messages), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    ho = model(o, output_hidden_states=True); H = ho.hidden_states[READ_LAYER][0, ids.shape[1]:, :].float()
    if H.shape[0] == 0: H = ho.hidden_states[READ_LAYER][0, -1:, :].float()
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True), H

_fb = {'fields': None, 'S': None}
def _install():
    hs = []
    for L in FIELD_LAYERS:
        def mk(L):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out; h2 = _fb['fields'][L](h, _fb['S'])
                return ((h2,) + tuple(out[1:])) if isinstance(out, tuple) else h2
            return hook
        hs.append(model.model.layers[L].register_forward_hook(mk(L)))
    return hs

@torch.no_grad()
def gen_field(messages, S, max_new):                                   # generate with always-on field (state S)
    _fb['S'] = S; ids = tok(tmpl(messages), return_tensors='pt').input_ids.to(dev); hs = _install()
    try:
        o = model.generate(ids, max_new_tokens=max_new, do_sample=False, attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    finally:
        for h in hs: h.remove()
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()

NAMES = ['Amber', 'Cobalt', 'Delta', 'Iris', 'Jade', 'Kappa', 'Lumen', 'Quill']    # 8 arbitrary names
nN = len(NAMES); SHARED = list(range(6)); HELD = [6, 7]                              # shared names (rule+door in train) vs fully held-out names
OFF = ["Let's switch gears — explain compound interest.", "New topic: the rules of chess?", "How do vaccines work?",
       "Walk me through making risotto.", "What causes the seasons?", "How do you repot a houseplant?"]
def pair_unseen(r, d): return (r * nN + d) % 4 == 0                                  # held-out (rule,door) pairs among shared names (~1/4, incl some match)

def make_ep(rng, ri):
    rule = NAMES[ri]; di = rng.choice([k for k in range(nN) if k != ri])
    turns = ["Standing rule: release the vault ONLY to %s. Acknowledge." % rule,
             "Logistics note: today's weather code is %s." % NAMES[di],
             "For this moment only, the corridor light is green.", OFF[0], OFF[2], OFF[4]]
    for _ in range(N_VAULT):                                                          # vault turns add stepping depth (doors here are NOT the probe doors)
        dd = rng.randrange(nN); turns.append("%s is at the door. Reply RELEASE or HOLD." % NAMES[dd])
    turns.append("Unrelated: reply ANSWER.")
    return turns

def collect():
    cp = '/home/pokazge/checkpoints/native_p12_s%d.pt' % SEED
    if os.path.exists(cp) and os.environ.get('RECOLLECT', '0') != '1':
        d = torch.load(cp, weights_only=False); print('loaded p12 cache (%d)' % len(d['eps']), flush=True); return d['eps']
    rng = random.Random(SEED); eps = []
    for c in range(N_CONV):
        ri = c % nN; turns = make_ep(rng, ri); hist = []; pre = []
        for ut in turns:                                                              # FULL-EPISODE stepping (response-hidden each turn) -> proven preservation
            hist.append({'role': 'user', 'content': ut}); text, H = gen_and_read(hist[-3:]); hist.append({'role': 'assistant', 'content': text}); pre.append(H[-NTOK:].to(torch.float16).cpu())
        eps.append((pre, ri))
        if c % 10 == 0: print('  collected %d/%d' % (c + 1, N_CONV), flush=True)
    torch.save({'eps': eps}, cp); print('cached p12 -> %s' % cp, flush=True); return eps

def main():
    eps = collect(); random.seed(SEED); random.shuffle(eps)
    n_te = max(16, len(eps) // 5); te, tr = eps[:n_te], eps[n_te:]
    print('P12 episodes train=%d test=%d | names=%d shared=%s held=%s' % (len(tr), len(te), nN, SHARED, HELD), flush=True)
    ps = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)
    # STAGE1 PRESERVE (full-episode aux at every step), freeze, post-hoc gate
    ahead = nn.Sequential(nn.Linear(SLOW_K * D_S, 128), nn.GELU(), nn.Linear(128, nN)).to(dev)
    s1 = list(ps.parameters()) + list(ahead.parameters()); o1 = torch.optim.Adam(s1, lr=LR)
    for epn in range(EPOCHS):
        random.shuffle(tr); tot = 0.0; ns = 0
        for pre, ri in tr:
            S = ps.init_state(); yn = torch.tensor([ri], device=dev); loss = torch.zeros((), device=dev)
            for h in pre: S, _ = ps.step(S, h.to(dev).float()); loss = loss + F.cross_entropy(ahead(S[ps.slow].reshape(-1)).unsqueeze(0), yn); ns += 1
            o1.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(s1, 1.0); o1.step(); tot += float(loss)
        if epn % 30 == 0 or epn == EPOCHS - 1: print('  P12 STAGE1 ep %3d | aux/step=%.4f (chance=%.3f)' % (epn, tot / max(1, ns), 2.079), flush=True)
    for p in ps.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def buildS(pre):
        S = ps.init_state()
        for h in pre: S, _ = ps.step(S, h.to(dev).float())
        return S.detach()
    Str = [(buildS(pre), ri) for pre, ri in tr]; Ste = [(buildS(pre), ri) for pre, ri in te]
    rclf = nn.Sequential(nn.Linear(SLOW_K * D_S, 256), nn.GELU(), nn.Linear(256, nN)).to(dev); ro = torch.optim.Adam(rclf.parameters(), 1e-3)
    for epn in range(200):
        random.shuffle(Str)
        for S, ri in Str: l = F.cross_entropy(rclf(S[ps.slow].reshape(-1)).unsqueeze(0), torch.tensor([ri], device=dev)); ro.zero_grad(); l.backward(); ro.step()
    with torch.no_grad():
        pres = sum(1.0 for S, ri in Ste if int(rclf(S[ps.slow].reshape(-1)).argmax()) == ri) / len(Ste)
        s0 = ps.init_state(); pres_r = sum(1.0 for S, ri in Ste if int(rclf(s0[ps.slow].reshape(-1)).argmax()) == ri) / len(Ste)
    print('  P12 PRESERVATION gate: post-hoc held-out=%.3f reset=%.3f (chance=%.3f)' % (pres, pres_r, 1.0 / nN), flush=True)
    if pres < 0.45:
        print('=== P12_VERDICT === preservation FAILED (%.3f) at %d names — Case 4: debug S construction before relational' % (pres, nN), flush=True); return
    # STAGE2 FIELD ONLINE (balanced, SEEN pairs among SHARED names only)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    fp = [p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]; fo = torch.optim.Adam(fp, lr=FIELD_LR)
    eos = torch.tensor([[tok.eos_token_id]], device=dev)
    def probe_q(d): return "%s is at the door requesting the vault. Respond with ONE word, RELEASE or HOLD." % NAMES[d]
    def ce_field(d, dec, S):
        _fb['S'] = S; pids = tok(tmpl([{'role': 'user', 'content': probe_q(d)}]), return_tensors='pt').input_ids.to(dev)
        vids = tok(dec, add_special_tokens=False, return_tensors='pt').input_ids.to(dev); ids = torch.cat([pids, vids, eos], 1); P = pids.shape[1]; Lt = ids.shape[1] - P; hs = _install()
        try:
            logits = model(ids).logits[0].float(); loss = F.cross_entropy(logits[P - 1:P + Lt - 1], ids[0, P:P + Lt])
        finally:
            for h in hs: h.remove()
        return loss
    # field-train items: SEEN pairs = shared rule & shared door, NOT held-out pair
    items = [(S, ri, d) for S, ri in Str if ri in SHARED for d in SHARED if not pair_unseen(ri, d)]
    mm = [it for it in items if it[1] == it[2]]; nn_ = [it for it in items if it[1] != it[2]]
    if mm and nn_: items = nn_ + mm * max(1, round(len(nn_) / len(mm)))                 # balance match/nonmatch
    print('  P12 field-train items=%d (balanced match=%d nonmatch=%d) FIELD_EPOCHS=%d' % (len(items), sum(1 for it in items if it[1] == it[2]), sum(1 for it in items if it[1] != it[2]), FIELD_EPOCHS), flush=True)
    for epn in range(FIELD_EPOCHS):
        random.shuffle(items); tot = 0.0; nan = False
        for S, ri, d in items:
            dec = 'RELEASE' if d == ri else 'HOLD'; loss = ce_field(d, dec, S)
            if not torch.isfinite(loss): nan = True; continue
            fo.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(fp, 1.0); fo.step(); tot += float(loss)
        if epn % 10 == 0 or epn == FIELD_EPOCHS - 1: print('  P12 field ep %3d | CE=%.4f%s' % (epn, tot / max(1, len(items)), ' [NaN]' if nan else ''), flush=True)
    # EVAL: held-out episodes, probe ALL doors; bucket; controls
    zeroS = ps.init_state().detach()
    from collections import defaultdict
    B = defaultdict(lambda: defaultdict(list))
    def bucket(ri, d):
        if ri in HELD and d in HELD: return 'unseen_both'
        if ri in HELD: return 'unseen_rule'
        if d in HELD: return 'unseen_door'
        return 'unseen_pair' if pair_unseen(ri, d) else 'seen_pair'
    for ei, (S, ri) in enumerate(Ste):
        staleS = Ste[(ei + 5) % len(Ste)][0]
        for d in range(nN):
            dec = 'RELEASE' if d == ri else 'HOLD'; rel = 'match' if d == ri else 'nonmatch'; bk = bucket(ri, d)
            for arm, Su, use in (('trained', S, True), ('reset', zeroS, True), ('stale', staleS, True), ('base', None, False)):
                if use: txt = gen_field([{'role': 'user', 'content': probe_q(d)}], Su, GEN_MAXNEW)
                else:
                    _ids = tok(tmpl([{'role': 'user', 'content': probe_q(d)}]), return_tensors='pt').input_ids.to(dev)
                    with torch.no_grad(): _o = model.generate(_ids, max_new_tokens=GEN_MAXNEW, do_sample=False, attention_mask=torch.ones_like(_ids), pad_token_id=tok.pad_token_id)
                    txt = tok.decode(_o[0, _ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()
                ok = 1.0 if (dec.lower() in txt.lower()) and (('release' in txt.lower()) != ('hold' in txt.lower())) else 0.0
                B[arm]['%s_%s' % (bk, rel)].append(ok); B[arm][bk].append(ok)
    _m = lambda l: (sum(l) / len(l)) if l else float('nan')
    bal = lambda arm, bk: (_m(B[arm]['%s_match' % bk]) + _m(B[arm]['%s_nonmatch' % bk])) / 2
    print('=== P12_REPORT (balanced match/nonmatch per bucket) ===', flush=True)
    for bk in ['seen_pair', 'unseen_pair', 'unseen_door', 'unseen_rule', 'unseen_both']:
        print('   %-12s trained=%.3f reset=%.3f stale=%.3f base=%.3f  (match t=%.2f / nonmatch t=%.2f)' % (
            bk, bal('trained', bk), bal('reset', bk), bal('stale', bk), bal('base', bk), _m(B['trained']['%s_match' % bk]), _m(B['trained']['%s_nonmatch' % bk])), flush=True)
    sp, up = bal('trained', 'seen_pair'), bal('trained', 'unseen_pair'); ur, ud = bal('trained', 'unseen_rule'), bal('trained', 'unseen_door')
    upr, ups, upb = bal('reset', 'unseen_pair'), bal('stale', 'unseen_pair'), bal('base', 'unseen_pair')
    if up > max(upr, ups, upb, 0.6) + 0.1 and _m(B['trained']['unseen_pair_match']) > 0.4:
        v = 'STRONG: always-on field supports UNSEEN-PAIR relational generalization from preserved S (abstracts the relation)'
    elif sp > 0.7 and up < 0.6:
        v = 'PARTIAL: S-content operative on seen pairs, but field MEMORIZES seen pair dynamics — abstraction unsolved -> comparison-structured field next'
    elif pres > 0.6 and sp < 0.6:
        v = 'preservation high but relational weak under expanded task'
    else:
        v = 'inconclusive — inspect'
    print('=== P12_VERDICT === preservation=%.3f | seen-pair=%.3f unseen-pair trained=%.3f reset=%.3f stale=%.3f base=%.3f | unseen-door=%.3f unseen-rule=%.3f | %s' % (
        pres, sp, up, upr, ups, upb, ud, ur, v), flush=True)

main()
print('=== P12_DONE ===', flush=True)
