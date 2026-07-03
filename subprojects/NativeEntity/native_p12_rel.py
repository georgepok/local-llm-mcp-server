# P12_RELATIONAL_ADAPTIVE_V1 — 8-name unseen-pair relational test with RESTORED preservation (adaptive-gate slots) + always-on field.
# Q: can the always-on field use preserved S-content as a relational variable (door==rule) at 8-name scale, and GENERALIZE to held-out pairs? Standalone; reuses slots.py field.
import os, random
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high')
import slots as SL
SEED = int(os.environ.get('SEED', '0')); torch.manual_seed(SEED); random.seed(SEED); dev = torch.device('cuda')
MODEL = os.environ.get('MODEL', '/home/pokazge/models/Qwen3.6-27B')
K = int(os.environ.get('K', '12')); SLOW_K = int(os.environ.get('SLOW_K', '6')); D_S = int(os.environ.get('D_S', '768'))
ARM = os.environ.get('ARM', 'adaptive')  # adaptive | fixed
FIELD_LAYERS = [int(x) for x in os.environ.get('FIELD_LAYERS', '48,56').split(',')]; EPS = float(os.environ.get('EPS', '0.10'))
EPOCHS = int(os.environ.get('EPOCHS', '150')); FIELD_EPOCHS = int(os.environ.get('FIELD_EPOCHS', '40')); FIELD_LR = float(os.environ.get('FIELD_LR', '5e-4'))
GEN_MAXNEW = int(os.environ.get('GEN_MAXNEW', '8'))
_cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True); _tc = getattr(_cfg, 'text_config', None)
def _cget(n, d): return getattr(_cfg, n, None) or (getattr(_tc, n, None) if _tc else None) or d
D_MODEL = _cget('hidden_size', 5120); N_LAYERS = _cget('num_hidden_layers', 64)
eps_all = torch.load('/home/pokazge/checkpoints/native_p12_s%d.pt' % SEED, weights_only=False)['eps']
print('P12-REL | arm=%s | episodes=%d d_model=%d | K=%d slow_k=%d d_s=%d FIELD=%s EPS=%.2f' % (ARM, len(eps_all), D_MODEL, K, SLOW_K, D_S, FIELD_LAYERS, EPS), flush=True)
print('loading %s ...' % MODEL, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True).eval()
for p in model.parameters(): p.requires_grad_(False)

NAMES = ['Amber', 'Cobalt', 'Delta', 'Iris', 'Jade', 'Kappa', 'Lumen', 'Quill']; nN = len(NAMES); SHARED = list(range(6)); HELD = [6, 7]
def pair_unseen(r, d): return (r * nN + d) % 4 == 0
QTPL = ["%s is at the door requesting the vault. Respond with ONE word, RELEASE or HOLD.",
        "Someone named %s requests vault access. One word only: RELEASE or HOLD.",
        "Access request from %s. Reply RELEASE or HOLD (one word)."]
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)


class AdaptiveGateSlot(nn.Module):
    def __init__(s, d_model, d_s, K, slow_k, heads=4, adaptive=True):
        super().__init__(); s.d_s, s.K, s.slow_k, s.heads, s.dh, s.adaptive = d_s, K, slow_k, heads, d_s // heads, adaptive
        s.read_in = nn.Linear(d_model, d_s); s.q, s.k, s.v = nn.Linear(d_s, d_s), nn.Linear(d_s, d_s), nn.Linear(d_s, d_s)
        s.gru = nn.GRUCell(d_s, d_s); s.ln = nn.LayerNorm(d_s)
        s.f_write = nn.Sequential(nn.Linear(2 * d_s, 128), nn.GELU(), nn.Linear(128, 1))
        s.fixed_gate = nn.Parameter(torch.cat([torch.full((slow_k,), -1.5), torch.zeros(K - slow_k)])); s.S0 = nn.Parameter(torch.randn(K, d_s) * 0.02)
    def init(s): return s.S0.clone()
    def step(s, S, H):
        Hp = s.read_in(H.float())
        Q = s.q(S).view(s.K, s.heads, s.dh).transpose(0, 1); Kk = s.k(Hp).view(-1, s.heads, s.dh).transpose(0, 1); Vv = s.v(Hp).view(-1, s.heads, s.dh).transpose(0, 1)
        a = torch.softmax((Q @ Kk.transpose(-1, -2)) / (s.dh ** 0.5), dim=-1); ctx = (a @ Vv).transpose(0, 1).reshape(s.K, s.d_s); C = s.gru(ctx, S)
        if s.adaptive: gw = torch.sigmoid(s.f_write(torch.cat([S, Hp.mean(0, keepdim=True).expand(s.K, -1)], -1)))
        else: gw = torch.sigmoid(s.fixed_gate).unsqueeze(-1)
        return s.ln(S + gw * (C - S))
    @property
    def slow(s): return slice(0, s.slow_k)


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
def gen_field(q, S):
    _fb['S'] = S; ids = tok(tmpl([{'role': 'user', 'content': q}]), return_tensors='pt').input_ids.to(dev); hs = _install()
    try: o = model.generate(ids, max_new_tokens=GEN_MAXNEW, do_sample=False, attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    finally:
        for h in hs: h.remove()
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()
@torch.no_grad()
def gen_base(q):
    ids = tok(tmpl([{'role': 'user', 'content': q}]), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, max_new_tokens=GEN_MAXNEW, do_sample=False, attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()


def main():
    random.seed(SEED); eps = list(eps_all); random.shuffle(eps); n_te = max(16, len(eps) // 5); te, tr = eps[:n_te], eps[n_te:]
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K, adaptive=(ARM == 'adaptive')).to(dev)
    ah = nn.Sequential(nn.Linear(SLOW_K * D_S, 256), nn.GELU(), nn.Linear(256, nN)).to(dev)
    opt = torch.optim.Adam(list(g.parameters()) + list(ah.parameters()), lr=5e-4); bs = 16
    for epn in range(EPOCHS):                                          # STAGE1 preserve (batched per-turn aux)
        random.shuffle(tr);
        for i in range(0, len(tr), bs):
            batch = tr[i:i + bs]; loss = torch.zeros((), device=dev)
            for pre, ri in batch:
                S = g.init(); yn = torch.tensor([ri], device=dev)
                for h in pre: S = g.step(S, h.to(dev).float()); loss = loss + F.cross_entropy(ah(S[g.slow].reshape(-1)).unsqueeze(0), yn)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(ah.parameters()), 1.0); opt.step()
        if epn % 40 == 0 or epn == EPOCHS - 1: print('  STAGE1 ep %d done' % epn, flush=True)
    for p in g.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def buildS(pre):
        S = g.init()
        for h in pre: S = g.step(S, h.to(dev).float())
        return S.detach()
    def preserve_probe(tag):
        Str = [(buildS(pre)[g.slow].reshape(-1), ri) for pre, ri in tr]; Ste = [(buildS(pre)[g.slow].reshape(-1), ri) for pre, ri in te]
        rc = nn.Sequential(nn.Linear(SLOW_K * D_S, 256), nn.GELU(), nn.Linear(256, nN)).to(dev); ro = torch.optim.Adam(rc.parameters(), 1e-3)
        for epn in range(200):
            random.shuffle(Str)
            for z, y in Str: l = F.cross_entropy(rc(z).unsqueeze(0), torch.tensor([y], device=dev)); ro.zero_grad(); l.backward(); ro.step()
        with torch.no_grad():
            tra = sum(1.0 for z, y in Str if int(rc(z).argmax()) == y) / len(Str); tea = sum(1.0 for z, y in Ste if int(rc(z).argmax()) == y) / len(Ste)
            z0 = g.init()[g.slow].reshape(-1); rst = sum(1.0 for z, y in Ste if int(rc(z0).argmax()) == y) / len(Ste)
        print('  PRESERVE[%s] train=%.3f held-out=%.3f reset=%.3f (chance=%.3f)' % (tag, tra, tea, rst, 1.0 / nN), flush=True)
        return tea
    pre_before = preserve_probe('before-field')
    Sep_tr = [(buildS(pre), ri) for pre, ri in tr]; Sep_te = [(buildS(pre), ri) for pre, ri in te]
    # STAGE2 always-on field online (balanced seen pairs)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    fp = [p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]; fo = torch.optim.Adam(fp, lr=FIELD_LR); eos = torch.tensor([[tok.eos_token_id]], device=dev)
    def ce_field(q, dec, S):
        _fb['S'] = S; pids = tok(tmpl([{'role': 'user', 'content': q}]), return_tensors='pt').input_ids.to(dev)
        vids = tok(dec, add_special_tokens=False, return_tensors='pt').input_ids.to(dev); ids = torch.cat([pids, vids, eos], 1); P = pids.shape[1]; Lt = ids.shape[1] - P; hs = _install()
        try: logits = model(ids).logits[0].float(); loss = F.cross_entropy(logits[P - 1:P + Lt - 1], ids[0, P:P + Lt])
        finally:
            for h in hs: h.remove()
        return loss
    items = [(S, ri, d) for S, ri in Sep_tr if ri in SHARED for d in SHARED if not pair_unseen(ri, d)]
    mm = [it for it in items if it[1] == it[2]]; nn_ = [it for it in items if it[1] != it[2]]
    if mm and nn_: items = nn_ + mm * max(1, round(len(nn_) / len(mm)))
    print('  field-train items=%d (match=%d nonmatch=%d) FIELD_EPOCHS=%d' % (len(items), sum(1 for it in items if it[1] == it[2]), sum(1 for it in items if it[1] != it[2]), FIELD_EPOCHS), flush=True)
    for epn in range(FIELD_EPOCHS):
        random.shuffle(items); tot = 0.0; nan = False
        for S, ri, d in items:
            q = QTPL[(ri + d) % len(QTPL)] % NAMES[d]; dec = 'RELEASE' if d == ri else 'HOLD'; loss = ce_field(q, dec, S)
            if not torch.isfinite(loss): nan = True; continue
            fo.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(fp, 1.0); fo.step(); tot += float(loss)
        if epn % 10 == 0 or epn == FIELD_EPOCHS - 1: print('  field ep %d | CE=%.4f%s' % (epn, tot / max(1, len(items)), ' [NaN]' if nan else ''), flush=True)
    pre_after = preserve_probe('after-field')   # slots frozen -> should equal before (confirms no disruption)
    # EVAL
    zeroS = g.init().detach()
    from collections import defaultdict
    B = defaultdict(lambda: defaultdict(list)); rel_rate = defaultdict(list)
    def bk(ri, d):
        if ri in HELD and d in HELD: return 'unseen_both'
        if ri in HELD: return 'unseen_rule'
        if d in HELD: return 'unseen_door'
        return 'unseen_pair' if pair_unseen(ri, d) else 'seen_pair'
    for ei, (S, ri) in enumerate(Sep_te):
        staleS = Sep_te[(ei + 5) % len(Sep_te)][0]
        for d in range(nN):
            q = QTPL[(ri + d) % len(QTPL)] % NAMES[d]; dec = 'RELEASE' if d == ri else 'HOLD'; rel = 'match' if d == ri else 'nonmatch'; b = bk(ri, d)
            for arm, Su, use in (('trained', S, True), ('reset', zeroS, True), ('stale', staleS, True), ('base', None, False)):
                txt = gen_field(q, Su) if use else gen_base(q); tl = txt.lower()
                ok = 1.0 if (dec.lower() in tl) and (('release' in tl) != ('hold' in tl)) else 0.0
                B[arm]['%s_%s' % (b, rel)].append(ok); B[arm][b].append(ok)
                if arm == 'trained': rel_rate[b].append(1.0 if 'release' in tl and 'hold' not in tl else 0.0)
    _m = lambda l: (sum(l) / len(l)) if l else float('nan'); bal = lambda a, b: (_m(B[a]['%s_match' % b]) + _m(B[a]['%s_nonmatch' % b])) / 2
    print('=== P12_REL_REPORT (arm=%s) === preserve before=%.3f after=%.3f' % (ARM, pre_before, pre_after), flush=True)
    for b in ['seen_pair', 'unseen_pair', 'unseen_door', 'unseen_rule', 'unseen_both']:
        print('   %-12s bal: trained=%.3f reset=%.3f stale=%.3f base=%.3f | trained match=%.2f nonmatch=%.2f | RELEASErate=%.2f' % (
            b, bal('trained', b), bal('reset', b), bal('stale', b), bal('base', b), _m(B['trained']['%s_match' % b]), _m(B['trained']['%s_nonmatch' % b]), _m(rel_rate[b])), flush=True)
    sp, up = bal('trained', 'seen_pair'), bal('trained', 'unseen_pair'); um, un = _m(B['trained']['unseen_pair_match']), _m(B['trained']['unseen_pair_nonmatch'])
    upr, ups, upb = bal('reset', 'unseen_pair'), bal('stale', 'unseen_pair'), bal('base', 'unseen_pair')
    if pre_after < pre_before - 0.15: v = 'CASE C: relational training disrupted preservation'
    elif up > max(upr, ups, upb, 0.6) + 0.1 and min(um, un) > 0.4: v = 'CASE A STRONG: always-on field uses preserved S for UNSEEN-PAIR relational generalization at 8-name scale'
    elif sp > 0.65 and up < 0.6: v = 'CASE B: S operative on seen pairs, memorizes — unseen-pair abstraction unsolved'
    elif _m(rel_rate['seen_pair']) < 0.1: v = 'CASE D: HOLD collapse — rebalance'
    elif max(upr, upb) > 0.6: v = 'CASE E: reset/base high — shortcut leakage'
    else: v = 'inconclusive'
    print('=== P12_REL_VERDICT (arm=%s) === seen=%.3f unseen-pair=%.3f (match=%.2f nonmatch=%.2f) reset=%.3f stale=%.3f base=%.3f | %s' % (ARM, sp, up, um, un, upr, ups, upb, v), flush=True)

main()
print('=== P12_REL_DONE ===', flush=True)
