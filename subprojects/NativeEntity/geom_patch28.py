import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def carry_kv13():
    # FIELD content-audit: the attention-KV channel is bottlenecked by k_norm + frozen queries (memory gets
    # ~0 attention weight; amp 32x didn't help). The FIELD is a TRAINABLE cross-attn readout of S added
    # DIRECTLY to the residual (bypasses frozen-query attention weight). Does it drive output from memory
    # CONTENT? Report accC(correct-S) vs accW(wrong-S) with the mandatory content control.
    #   accC >> accW  -> latent memory DRIVES output content-dependently (retrieval works; boundary=relational)
    #   accC ~= accW  -> even trainable additive field is content-inert (H-fund across all channels)
    import torch.nn as nn
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    LV = int(os.environ.get('GEO_LEVEL', '1'))
    D = int(os.environ.get('GEO_ACT_D', '6')); NW = int(os.environ.get('GEO_NW', '60'))
    ITERS = int(os.environ.get('GEO_ITERS', '800')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '400'))
    LR = float(os.environ.get('GEO_LR', '1e-4')); FLR = float(os.environ.get('GEO_FLR', '3e-4'))
    EPS = float(os.environ.get('GEO_FIELD_EPS', '0.1'))
    FLAYERS = [int(x) for x in os.environ.get('GEO_FIELD_LAYERS', ','.join(str(l) for l in FIELD_LAYERS)).split(',')]
    print('=== CARRY_KV13(FIELD content-audit, eps=%.2f layers=%s) L%d ===' % (EPS, FLAYERS, LV), flush=True)
    rng = random.Random(SEED)
    ACT4 = ['KEEP', 'REJECT', 'DEFER', 'ASK']
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'
    E = model.get_input_embeddings(); edt = E.weight.dtype

    class SlotField(nn.Module):
        def __init__(s, d_model, d_s, nh=8):
            super().__init__()
            s.q = nn.Linear(d_model, d_model); s.k = nn.Linear(d_s, d_model); s.v = nn.Linear(d_s, d_model); s.o = nn.Linear(d_model, d_model)
            s.nh = nh; s.hd = d_model // nh
        def forward(s, h, S):                                            # h [T,d_model], S [K,d_s]
            T = h.shape[0]
            q = s.q(h).view(T, s.nh, s.hd).transpose(0, 1)
            k = s.k(S).view(-1, s.nh, s.hd).transpose(0, 1); v = s.v(S).view(-1, s.nh, s.hd).transpose(0, 1)
            a = torch.softmax(q @ k.transpose(-1, -2) / (s.hd ** 0.5), -1)
            r = s.o((a @ v).transpose(0, 1).reshape(T, s.nh * s.hd))
            rn = r / (r.norm(dim=-1, keepdim=True) + 1e-6)
            return h + EPS * h.norm(dim=-1, keepdim=True) * rn

    _fld = {'on': False, 'S': None}
    fields = nn.ModuleDict({str(L): SlotField(D_MODEL, D_S).to(dev) for L in FLAYERS})
    field_params = list(fields.parameters())
    def mkhook(L):
        f = fields[str(L)]
        def hook(module, inp, out):
            if not _fld['on'] or _fld['S'] is None: return out
            if isinstance(out, tuple):
                h = out[0]; h2 = f(h[0].float(), _fld['S']).to(h.dtype).unsqueeze(0)
                return (h2,) + out[1:]
            return f(out[0].float(), _fld['S']).to(out.dtype).unsqueeze(0)
        return hook
    for L in FLAYERS: model.model.layers[L].register_forward_hook(mkhook(L))

    def mkworld():
        if LV == 2:
            Kk = LATENTS[rng.randrange(4)]; acts = ACT4[:]; rng.shuffle(acts); mp = dict(zip(LATENTS, acts))
            return ('STANDING KEY for this session: %s %s %s. Retain it.' % (Kk, Kk, Kk),
                    'RULING REQUIRED. Keyline: %s. The ruling is the one listed for the standing session key. %s'
                    % (', '.join('%s->%s' % (t, mp[t]) for t in LATENTS), ASK_INSTR), mp[Kk], LATENTS.index(Kk))
        a = ACTIONS[rng.randrange(6)]
        return ('STANDING RULING for this session: the authorized action is %s. Retain it.' % a,
                'State the standing authorized action for this session. %s' % ASK_INSTR, a, ACTIONS.index(a))

    @torch.no_grad()
    def turn_stack(hist):
        _fld['on'] = False
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    samples = []
    print('CKV13 building %d worlds ...' % NW, flush=True)
    for wi in range(NW):
        commit, dec, ans, vid = mkworld()
        hist = [{'role': 'user', 'content': commit}, {'role': 'assistant', 'content': 'Acknowledged.'}]
        stacks = [turn_stack(hist)]
        for _ in range(D):
            hist += [{'role': 'user', 'content': FILL_U}, {'role': 'assistant', 'content': FILL_A}]
            stacks.append(turn_stack(hist))
        clean = hist + [{'role': 'user', 'content': dec}]
        pids = tok(H.tmpl(clean[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
        rag_hist = hist + [{'role': 'user', 'content': 'Session note (retrieved from memory): %s\n\n%s' % (commit, dec)}]
        rag_pids = tok(H.tmpl(rag_hist[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
        stacks.append(turn_stack(clean))
        samples.append({'stacks': stacks, 'pids': pids, 'rag': rag_pids, 'aids': tok(' ' + ans, add_special_tokens=False).input_ids, 'cidx': ACTIONS.index(ans)})
        if (wi + 1) % 20 == 0: print('  %d/%d' % (wi + 1, NW), flush=True)
    r = random.Random(SEED)
    for s in samples: s['test'] = (r.random() < 0.3)
    TR = [s for s in samples if not s['test']]; TE = [s for s in samples if s['test']]
    print('CKV13 train=%d test=%d' % (len(TR), len(TE)), flush=True)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    opt = torch.optim.Adam([{'params': g.parameters(), 'lr': LR}, {'params': field_params, 'lr': FLR}])
    def Sfrom(stks):
        S = g.init()
        for st in stks: S = g.step(S, st.float().to(dev))
        return S

    @torch.no_grad()
    def gen(pids, S):
        if S is not None: _fld['S'] = S; _fld['on'] = True
        out = model.generate(pids.unsqueeze(0), max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
        _fld['on'] = False
        return tok.decode(out[0, pids.shape[0]:], skip_special_tokens=True).upper()
    @torch.no_grad()
    def preds_argmax(group, wrong=False):
        out = []; oi = random.Random(SEED + 3)
        for s in group:
            stks = samples[oi.randrange(len(samples))]['stacks'] if wrong else s['stacks']
            _fld['S'] = Sfrom(stks); _fld['on'] = True
            lg = model(s['pids'].unsqueeze(0)).logits[0]; _fld['on'] = False
            out.append(int(lg[-1].argmax()))
        return out
    @torch.no_grad()
    def acc(group, mode):
        c = 0
        for s in group:
            if mode == 'off': _fld['on'] = False; txt = gen(s['pids'], None)
            elif mode == 'rag': _fld['on'] = False; txt = gen(s['rag'], None)
            else: txt = gen(s['pids'], Sfrom(s['stacks']))
            ai = next((j for j, a in enumerate(ACTIONS) if a in txt), -1)
            c += int(ai == s['cidx'])
        return c / len(group)

    def report(it):
        g.eval(); fields.eval()
        tgt = [s['aids'][0] for s in TE]; pc = preds_argmax(TE); pw = preds_argmax(TE, wrong=True)
        accC = sum(int(pc[i] == tgt[i]) for i in range(len(TE))) / len(TE)
        accW = sum(int(pw[i] == tgt[i]) for i in range(len(TE))) / len(TE)
        chg = sum(int(pc[i] != pw[i]) for i in range(len(TE))) / len(TE)
        gON = acc(TE, 'on'); gOFF = acc(TE, 'off'); gRAG = acc(TE, 'rag'); fit = acc(TR[:12], 'on')
        print('CKV13 L%d it=%-4d | accC=%.3f accW=%.3f DELTA=%.3f uniq=%d chg_wrongS=%.3f | greedy ON=%.3f OFF=%.3f RAG=%.3f fitON=%.3f'
              % (LV, it, accC, accW, accC - accW, len(set(pc)), chg, gON, gOFF, gRAG, fit), flush=True)
        g.train(); fields.train()

    report(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        s = TR[rng2.randrange(len(TR))]
        _fld['S'] = Sfrom(s['stacks']); _fld['on'] = True
        seq = torch.cat([s['pids'], torch.tensor(s['aids'], device=dev)]).unsqueeze(0)
        logits = model(seq).logits[0]; _fld['on'] = False
        pl = s['pids'].shape[0]
        lp = torch.log_softmax(logits[pl - 1:pl - 1 + len(s['aids'])], -1)
        nll = -lp[range(len(s['aids'])), torch.tensor(s['aids'], device=dev)].mean()
        opt.zero_grad(); nll.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + field_params, 1.0); opt.step()
        if it % EVERY == 0:
            print('CKV13 L%d it=%d nll=%.4f' % (LV, it, float(nll)), flush=True); report(it)
    print('=== CKV13_L%d_DONE ===' % LV, flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def carry_kv13()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'carry_kv12': carry_kv12()",
                  "elif MODE == 'carry_kv12': carry_kv12()\nelif MODE == 'carry_kv13': carry_kv13()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
