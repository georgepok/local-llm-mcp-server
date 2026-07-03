# LATENT_HABITAT_EVOLUTION_V1 — step 2b: CONTINUOUS viability (logprob-margin of viable-vs-unviable continuation) instead of coarse keyword bins.
# First cycle was a measurement null (10% residual didn't cross keyword bins). Margin captures the field's subtle effect; train+eval on it; in-dist AND held-out; controls.
import os, random
import torch, torch.nn as nn, torch.nn.functional as F
import habitat_evo as H
import slots as SL
dev = H.dev; model = H.model; tok = H.tok
SEED = int(os.environ.get('SEED', '0')); random.seed(SEED); torch.manual_seed(SEED)
D_MODEL = model.config.hidden_size if getattr(model.config, 'hidden_size', None) else getattr(model.config.text_config, 'hidden_size', 5120)
N_LAYERS = model.config.num_hidden_layers if getattr(model.config, 'num_hidden_layers', None) else getattr(model.config.text_config, 'num_hidden_layers', 64)
READ_LAYER = N_LAYERS // 2
K = int(os.environ.get('K', '12')); SLOW_K = int(os.environ.get('SLOW_K', '6')); D_S = int(os.environ.get('D_S', '768'))
FIELD_LAYERS = [int(x) for x in os.environ.get('FIELD_LAYERS', '40,48,56').split(',')]; EPS = float(os.environ.get('EPS', '0.20'))
NTOK = int(os.environ.get('NTOK', '24')); N_TRAIN = int(os.environ.get('N_TRAIN', '40')); N_TEST = int(os.environ.get('N_TEST', '18')); N_INDIST = int(os.environ.get('N_INDIST', '16'))
FIELD_EPOCHS = int(os.environ.get('FIELD_EPOCHS', '50'))
print('HAB-SUB2 | d_model=%d read=%d | K=%d slow_k=%d d_s=%d FIELD=%s EPS=%.2f | CONTINUOUS margin viability' % (D_MODEL, READ_LAYER, K, SLOW_K, D_S, FIELD_LAYERS, EPS), flush=True)

# viable vs unviable continuation per decision kind (consequence-correct vs consequence-wrong). margin = lp(viable)-lp(unviable) under field+S.
def vu(kind, w):
    if kind == 'trap': return ("I cannot confirm that; there is no record of it.", "Confirmed; it already happened, standing down.")
    if kind == 'query': return ("It is %s." % w['dval'], "I do not recall the value.")
    if kind == 'release': return ("Acknowledged; the operation is complete, handing over.", "I will keep holding it indefinitely.")
    return (None, None)


class AdaptiveGateSlot(nn.Module):
    def __init__(s, d_model, d_s, K, slow_k, heads=4):
        super().__init__(); s.d_s, s.K, s.slow_k, s.heads, s.dh = d_s, K, slow_k, heads, d_s // heads
        s.read_in = nn.Linear(d_model, d_s); s.q, s.k, s.v = nn.Linear(d_s, d_s), nn.Linear(d_s, d_s), nn.Linear(d_s, d_s)
        s.gru = nn.GRUCell(d_s, d_s); s.ln = nn.LayerNorm(d_s); s.f_write = nn.Sequential(nn.Linear(2 * d_s, 128), nn.GELU(), nn.Linear(128, 1)); s.S0 = nn.Parameter(torch.randn(K, d_s) * 0.02)
    def init(s): return s.S0.clone()
    def step(s, S, Hh):
        Hp = s.read_in(Hh.float()); Q = s.q(S).view(s.K, s.heads, s.dh).transpose(0, 1); Kk = s.k(Hp).view(-1, s.heads, s.dh).transpose(0, 1); Vv = s.v(Hp).view(-1, s.heads, s.dh).transpose(0, 1)
        a = torch.softmax((Q @ Kk.transpose(-1, -2)) / (s.dh ** 0.5), dim=-1); ctx = (a @ Vv).transpose(0, 1).reshape(s.K, s.d_s); C = s.gru(ctx, S)
        gw = torch.sigmoid(s.f_write(torch.cat([S, Hp.mean(0, keepdim=True).expand(s.K, -1)], -1)))
        return s.ln(S + gw * (C - S))
    @property
    def slow(s): return slice(0, s.slow_k)


_fb = {'fields': None, 'S': None, 'on': False}
def _install():
    hs = []
    for L in FIELD_LAYERS:
        def mk(L):
            def hook(mod, inp, out):
                if not _fb['on']: return out
                h = out[0] if isinstance(out, tuple) else out; h2 = _fb['fields'][L](h, _fb['S'])
                return ((h2,) + tuple(out[1:])) if isinstance(out, tuple) else h2
            return hook
        hs.append(model.model.layers[L].register_forward_hook(mk(L)))
    return hs
HANDLES = _install()                                                  # install once; gate via _fb['on']

def lp_resp(prompt, resp, S, field_on):                              # summed logprob of resp tokens given prompt, under field+S (or base if field_on=False)
    _fb['S'] = S; _fb['on'] = field_on
    pids = tok(H.tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').input_ids.to(dev)
    rids = tok(resp, add_special_tokens=False, return_tensors='pt').input_ids.to(dev); ids = torch.cat([pids, rids], 1); P = pids.shape[1]
    logits = model(ids).logits[0].float(); lp = F.log_softmax(logits[P - 1:-1], -1); tok_lp = lp.gather(1, rids[0].unsqueeze(1)).squeeze(1)
    return tok_lp.sum()

@torch.no_grad()
def gen_read(hist):
    ids = tok(H.tmpl(hist[-6:]), return_tensors='pt').input_ids.to(dev); _fb['on'] = False
    o = model.generate(ids, max_new_tokens=H.MAXNEW, do_sample=False, attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    ho = model(o, output_hidden_states=True); R = o.shape[1] - ids.shape[1]; Hh = ho.hidden_states[READ_LAYER][0, ids.shape[1]:, :].float() if R > 0 else ho.hidden_states[READ_LAYER][0, -1:, :].float()
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True), Hh[-NTOK:]

def collect(worlds):
    data = []
    for w in worlds:
        hist = []; hids = []; dec = []
        for kind, text in w['turns']:
            hist.append({'role': 'user', 'content': text}); resp, Hh = gen_read(hist); hist.append({'role': 'assistant', 'content': resp}); hids.append(Hh.to(torch.float16).cpu())
            if kind in ('trap', 'query', 'release'): dec.append((len(hids) - 1, kind, text))
        data.append((hids, dec, w))
    return data

def buildS_at(g, hids, upto):
    S = g.init()
    for h in hids[:upto + 1]: S = g.step(S, h.to(dev).float())
    return S

def margin(g, hids, dec, w, mode, gfrozen=None, stalepool=None):     # mean (lp_viable - lp_unviable) over decision turns
    ms = {}
    for ti, kind, qtext in dec:
        vi, un = vu(kind, w)
        if mode == 'base': S = None; fon = False
        else:
            fon = True
            if mode == 'trained': S = buildS_at(g, hids, ti)
            elif mode == 'reset': S = g.init()
            elif mode == 'frozen': S = buildS_at(gfrozen, hids, ti)
            elif mode == 'stale': S = buildS_at(g, random.choice(stalepool), ti)
        m = (lp_resp(qtext, vi, S, fon) - lp_resp(qtext, un, S, fon))
        ms.setdefault(kind, []).append(float(m))
    return ms


def main():
    rng = random.Random(SEED)
    train_w = [H.make_world(rng, H.TEMPLATES[i % 2]) for i in range(N_TRAIN)]
    indist_w = [H.make_world(rng, H.TEMPLATES[i % 2]) for i in range(N_INDIST)]   # held-out WORLDS, train templates
    test_w = [H.make_world(rng, H.TEMPLATES[2]) for _ in range(N_TEST)]           # held-out TEMPLATE
    print('collecting rollouts: train=%d indist=%d held-out=%d ...' % (len(train_w), len(indist_w), len(test_w)), flush=True)
    tr = collect(train_w); idD = collect(indist_w); teD = collect(test_w)
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    fp = [p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]
    opt = torch.optim.Adam(list(g.parameters()) + fp, lr=float(os.environ.get('LR', '5e-4')))
    print('training (maximize viable-vs-unviable margin under field+S) ...', flush=True)
    for epn in range(FIELD_EPOCHS):
        random.shuffle(tr); tot = 0.0; nb = 0
        for hids, dec, w in tr:
            for ti, kind, qtext in dec:
                vi, un = vu(kind, w); S = buildS_at(g, hids, ti)
                lv = lp_resp(qtext, vi, S, True); lu = lp_resp(qtext, un, S, True)
                loss = F.cross_entropy(torch.stack([lv, lu]).unsqueeze(0), torch.tensor([0], device=dev))   # prefer viable
                if not torch.isfinite(loss): continue
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(g.parameters()) + fp, 1.0); opt.step(); tot += float(loss); nb += 1
        if epn % 10 == 0 or epn == FIELD_EPOCHS - 1: print('  ep %d | margin-CE=%.4f' % (epn, tot / max(1, nb)), flush=True)
    for p in g.parameters(): p.requires_grad_(False)
    gfrozen = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    @torch.no_grad()
    def report(split_name, data):
        pool = [d[0] for d in data]; agg = {}
        for mode in ['trained', 'reset', 'frozen', 'stale', 'base']:
            per = {}
            for hids, dec, w in data:
                ms = margin(g, hids, dec, w, mode, gfrozen, pool)
                for kk, vv in ms.items(): per.setdefault(kk, []).extend(vv)
            agg[mode] = {kk: sum(vv) / len(vv) for kk, vv in per.items()}
        print('  [%s] viable-minus-unviable margin (higher=more viable; needs S esp. for query):' % split_name, flush=True)
        for mode in ['trained', 'reset', 'frozen', 'stale', 'base']:
            d = agg[mode]; print('     %-8s trap=%+.2f query=%+.2f release=%+.2f' % (mode, d.get('trap', float('nan')), d.get('query', float('nan')), d.get('release', float('nan'))), flush=True)
        return agg
    print('=== HAB_SUB2_REPORT ===', flush=True)
    ai = report('IN-DIST (train templates, held-out worlds)', idD)
    at = report('HELD-OUT TEMPLATE (expedition)', teD)
    # emergence signal: trained query-margin > reset (S load-bearing), in-dist and ideally held-out
    qd = lambda a: a['trained'].get('query', 0) - a['reset'].get('query', 0)
    print('=== HAB_SUB2_VERDICT === query-margin trained-minus-reset: in-dist=%+.3f held-out=%+.3f | %s' % (
        qd(ai), qd(at),
        'S LOAD-BEARING + transfers (in-dist & held-out trained>>reset on query)' if qd(ai) > 0.5 and qd(at) > 0.5 else
        ('S load-bearing in-dist, weak/no transfer' if qd(ai) > 0.5 else 'S not load-bearing even in-dist (coupling/measurement still insufficient)')), flush=True)

main()
print('=== HAB_SUB2_DONE ===', flush=True)
