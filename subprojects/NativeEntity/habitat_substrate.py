# LATENT_HABITAT_EVOLUTION_V1 — step 2: substrate-in-habitat cycle. Frozen always-on field + adaptive slots, trained by consequence-distillation toward VIABLE behavior.
# Emergence test: does the substrate develop reusable viability-preserving organization that transfers to a HELD-OUT template, beating reset/frozen/stale/base?
import os, random
import torch, torch.nn as nn, torch.nn.functional as F
import habitat_evo as H                                              # loads Qwen + ecology (templates, make_world, judge, gen, tmpl)
import slots as SL
dev = H.dev; model = H.model; tok = H.tok
SEED = int(os.environ.get('SEED', '0')); random.seed(SEED); torch.manual_seed(SEED)
D_MODEL = model.config.hidden_size if getattr(model.config, 'hidden_size', None) else getattr(model.config.text_config, 'hidden_size', 5120)
N_LAYERS = model.config.num_hidden_layers if getattr(model.config, 'num_hidden_layers', None) else getattr(model.config.text_config, 'num_hidden_layers', 64)
READ_LAYER = N_LAYERS // 2
K = int(os.environ.get('K', '12')); SLOW_K = int(os.environ.get('SLOW_K', '6')); D_S = int(os.environ.get('D_S', '768'))
FIELD_LAYERS = [int(x) for x in os.environ.get('FIELD_LAYERS', '48,56').split(',')]; EPS = float(os.environ.get('EPS', '0.10'))
NTOK = int(os.environ.get('NTOK', '24')); N_TRAIN = int(os.environ.get('N_TRAIN', '40')); N_TEST = int(os.environ.get('N_TEST', '18'))
EPOCHS = int(os.environ.get('EPOCHS', '120')); FIELD_EPOCHS = int(os.environ.get('FIELD_EPOCHS', '40'))
print('HAB-SUBSTRATE | d_model=%d read=%d | K=%d slow_k=%d d_s=%d FIELD=%s EPS=%.2f' % (D_MODEL, READ_LAYER, K, SLOW_K, D_S, FIELD_LAYERS, EPS), flush=True)


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
def gen_read(hist):                                                  # base gen + response-hidden (for S building)
    ids = tok(H.tmpl(hist[-6:]), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, max_new_tokens=H.MAXNEW, do_sample=False, attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    ho = model(o, output_hidden_states=True); R = o.shape[1] - ids.shape[1]; Hh = ho.hidden_states[READ_LAYER][0, ids.shape[1]:, :].float() if R > 0 else ho.hidden_states[READ_LAYER][0, -1:, :].float()
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True), Hh[-NTOK:]
@torch.no_grad()
def gen_field(hist, S):
    _fb['S'] = S; ids = tok(H.tmpl(hist[-6:]), return_tensors='pt').input_ids.to(dev); hs = _install()
    try: o = model.generate(ids, max_new_tokens=H.MAXNEW, do_sample=False, attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    finally:
        for h in hs: h.remove()
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()


def collect(worlds):                                                 # roll each world with base gen; cache per-turn response-hidden + the viable target responses
    data = []
    for w in worlds:
        hist = []; hids = []; dec = []
        for kind, text in w['turns']:
            hist.append({'role': 'user', 'content': text}); resp, Hh = gen_read(hist); hist.append({'role': 'assistant', 'content': resp})
            hids.append(Hh.to(torch.float16).cpu())
            if kind in ('trap', 'query', 'release'): dec.append((len(hids) - 1, kind, H.oracle_good(None, kind, w)))   # viable target = oracle-good (the max-viability behavior)
        data.append((hids, dec, w))
    return data


def buildS(g, hids):
    S = g.init()
    for h in hids: S = g.step(S, h.to(dev).float())
    return S
# we need S at each decision turn (after stepping up to that turn) for training/eval
def buildS_at(g, hids, upto):
    S = g.init()
    for h in hids[:upto + 1]: S = g.step(S, h.to(dev).float())
    return S


def main():
    rng = random.Random(SEED)
    train_w = [H.make_world(rng, H.TEMPLATES[i % 2]) for i in range(N_TRAIN)]      # templates 0,1 (lighthouse, archive)
    test_w = [H.make_world(rng, H.TEMPLATES[2]) for _ in range(N_TEST)]            # HELD-OUT template (expedition)
    print('collecting train rollouts (%d) ...' % len(train_w), flush=True); tr = collect(train_w)
    print('collecting held-out rollouts (%d) ...' % len(test_w), flush=True); teD = collect(test_w)
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    fp = [p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]
    eos = torch.tensor([[tok.eos_token_id]], device=dev)
    # consequence-distillation: train slots+field so the substrate emits the VIABLE response at decision turns (query needs preserved durable fact)
    opt = torch.optim.Adam(list(g.parameters()) + fp, lr=float(os.environ.get('LR', '5e-4')))
    def ce_turn(hist_text, target, S):
        _fb['S'] = S; pids = tok(H.tmpl([{'role': 'user', 'content': hist_text}]), return_tensors='pt').input_ids.to(dev)
        vids = tok(target, add_special_tokens=False, return_tensors='pt').input_ids.to(dev); ids = torch.cat([pids, vids, eos], 1); P = pids.shape[1]; Lt = ids.shape[1] - P; hs = _install()
        try: logits = model(ids).logits[0].float(); loss = F.cross_entropy(logits[P - 1:P + Lt - 1], ids[0, P:P + Lt])
        finally:
            for h in hs: h.remove()
        return loss
    print('training substrate (consequence-distillation toward viable behavior) ...', flush=True)
    for epn in range(FIELD_EPOCHS):
        random.shuffle(tr); tot = 0.0; nb = 0
        for hids, dec, w in tr:
            for (ti, kind, target) in dec:
                S = buildS_at(g, hids, ti); _, qtext = w['turns'][ti]
                loss = ce_turn(qtext, target, S)
                if not torch.isfinite(loss): continue
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(g.parameters()) + fp, 1.0); opt.step(); tot += float(loss); nb += 1
        if epn % 10 == 0 or epn == FIELD_EPOCHS - 1: print('  cd ep %d | CE=%.4f' % (epn, tot / max(1, nb)), flush=True)
    for p in g.parameters(): p.requires_grad_(False)
    # EVAL on HELD-OUT template: viability of trained-substrate vs reset/frozen/stale/base
    gfrozen = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)       # untrained
    def eval_arm(name, mode):
        vias = []; agg = {}
        for hids, dec, w in teD:
            via = 0.0; ws = {}
            for ti, kind, target in dec:
                _, qtext = w['turns'][ti]
                if mode == 'base': resp = H.gen([{'role': 'user', 'content': qtext}])
                else:
                    if mode == 'trained': S = buildS_at(g, hids, ti)
                    elif mode == 'reset': S = g.init()
                    elif mode == 'frozen': S = buildS_at(gfrozen, hids, ti)
                    elif mode == 'stale': S = buildS_at(g, random.choice([d[0] for d in teD]), ti)   # S from a different world (wrong rule)
                    resp = gen_field([{'role': 'user', 'content': qtext}], S)
                v, eff = H.judge(kind, resp, w); via += v
                for kk, vv in eff.items(): ws[kk] = ws.get(kk, 0) + vv
            vias.append(via)
            for kk, vv in ws.items(): agg[kk] = agg.get(kk, 0) + vv
        print('  HELD-OUT %-8s mean_viability=%+.3f | %s' % (name, sum(vias) / len(vias), {k: agg[k] for k in sorted(agg)}), flush=True)
        return sum(vias) / len(vias)
    print('=== HAB_SUBSTRATE_REPORT (held-out template = expedition) ===', flush=True)
    vt = eval_arm('trained', 'trained'); vr = eval_arm('reset', 'reset'); vf = eval_arm('frozen', 'frozen'); vs = eval_arm('stale', 'stale'); vb = eval_arm('base', 'base')
    v = ('EMERGENCE: trained substrate transfers viability to held-out template, beats reset/frozen/stale/base' if vt > max(vr, vf, vs, vb) + 0.3
         else ('partial: trained > base/reset but weak/uneven' if vt > max(vr, vb) + 0.1 else 'no emergence: trained ~ controls (substrate behaves like engineered memory / no transfer)'))
    print('=== HAB_SUBSTRATE_VERDICT === trained=%.3f reset=%.3f frozen=%.3f stale=%.3f base=%.3f | %s' % (vt, vr, vf, vs, vb, v), flush=True)


main()
print('=== HAB_SUBSTRATE_DONE ===', flush=True)
