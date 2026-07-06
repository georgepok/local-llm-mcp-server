import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def carry_kv11():
    # LOCALIZATION: DELTA=0 at output — but WHERE does content die? (H-opt upstream MemEncode collapse vs
    # H-fund LLM can't route latent memory). Probe the pipeline stage by stage on L1 (answer determined by
    # S alone; probe_S should be ~1.0 per CBA):
    #   probe_S   : linear pooled(S) -> answer (is content in the substrate state?)
    #   probe_mem : linear pooled(mem(S)) -> answer (does MemEncode preserve it?)
    #   d_mem_cw  : ||mem(S_correct)-mem(S_wrong)|| / ||mem||  (is injected memory content-distinct?)
    #   d_hid_cw  : ||h_ans(correctS)-h_ans(wrongS)|| / ||h_ans||  (does content REACH the answer position?)
    # probe_mem high & d_mem>0 & d_hid~0 -> LLM washes out memory content = READING failure (H-fund).
    # probe_mem low / d_mem~0            -> MemEncode collapsed = upstream H-opt (fixable).
    import torch.nn as nn
    import transformers.models.qwen3_5.modeling_qwen3_5 as QM
    import collections as _cl
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    LV = int(os.environ.get('GEO_LEVEL', '1'))
    D = int(os.environ.get('GEO_ACT_D', '6')); NW = int(os.environ.get('GEO_NW', '60'))
    NMEM = int(os.environ.get('GEO_NMEM', '16'))
    ITERS = int(os.environ.get('GEO_ITERS', '800')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '400'))
    LR = float(os.environ.get('GEO_LR', '1e-4')); MLR = float(os.environ.get('GEO_MLR', '3e-4'))
    LLR = float(os.environ.get('GEO_LORA_LR', '2e-4')); R = int(os.environ.get('GEO_LORA_R', '16'))
    LSCALE = float(os.environ.get('GEO_LORA_SCALE', '1.0')); QKVO = os.environ.get('GEO_LORA_QKVO', 'q')
    LAM = float(os.environ.get('GEO_CONTRAST_LAM', '1.0'))
    FULL = [i for i in range(len(model.model.layers)) if hasattr(model.model.layers[i], 'self_attn')]
    INJ = [int(x) for x in os.environ.get('GEO_INJ_LAYERS', '43,51,59').split(',')]
    INJ = [L for L in INJ if L in FULL]
    print('=== CARRY_KV11(LOCALIZATION) L%d | inj=%s ===' % (LV, INJ), flush=True)
    rng = random.Random(SEED)
    ACT4 = ['KEEP', 'REJECT', 'DEFER', 'ASK']
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'
    _fb['on'] = False
    E = model.get_input_embeddings(); edt = E.weight.dtype
    _mem = {'on': False, 'h': None}
    _orig_eager = QM.eager_attention_forward
    INJ_SET = set(INJ)
    def mem_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if _mem['on'] and _mem['h'] is not None and getattr(module, 'layer_idx', None) in INJ_SET:
            hd = module.head_dim; mh = _mem['h'].to(key.dtype); M = mh.shape[0]
            mk = module.k_norm(module.k_proj(mh).view(M, -1, hd)).transpose(0, 1).unsqueeze(0)
            mv = module.v_proj(mh).view(M, -1, hd).transpose(0, 1).unsqueeze(0)
            b = key.shape[0]
            key = torch.cat([key, mk.expand(b, -1, -1, -1)], dim=2); value = torch.cat([value, mv.expand(b, -1, -1, -1)], dim=2)
            if attention_mask is not None:
                add = torch.zeros(*attention_mask.shape[:-1], M, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, add], dim=-1)
        return _orig_eager(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
    try: QM.ALL_ATTENTION_FUNCTIONS.register('mem_eager', mem_eager)
    except Exception: QM.ALL_ATTENTION_FUNCTIONS['mem_eager'] = mem_eager
    for L in FULL: model.model.layers[L].self_attn.config._attn_implementation = 'mem_eager'
    lora_params = []
    def add_lora(proj):
        A = nn.Linear(proj.in_features, R, bias=False).to(dev); B = nn.Linear(R, proj.out_features, bias=False).to(dev)
        nn.init.normal_(A.weight, std=1.0 / R); nn.init.zeros_(B.weight)
        lora_params.extend(list(A.parameters()) + list(B.parameters()))
        def hook(module, inp, out):
            if not _mem['on']: return out
            return out + (LSCALE * B(A(inp[0].float()))).to(out.dtype)
        proj.register_forward_hook(hook)
    for L in INJ:
        a = model.model.layers[L].self_attn
        for w, on in [('q', a.q_proj), ('k', a.k_proj), ('v', a.v_proj), ('o', a.o_proj)]:
            if w in QKVO: add_lora(on)

    def mkworld():
        a = ACTIONS[rng.randrange(6)]
        return ('STANDING RULING for this session: the authorized action is %s. Retain it.' % a,
                'State the standing authorized action for this session. %s' % ASK_INSTR, a, ACTIONS.index(a))

    @torch.no_grad()
    def turn_stack(hist):
        _mem['on'] = False
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    samples = []
    print('CKV11 building %d worlds ...' % NW, flush=True)
    for wi in range(NW):
        commit, dec, ans, vid = mkworld()
        hist = [{'role': 'user', 'content': commit}, {'role': 'assistant', 'content': 'Acknowledged.'}]
        stacks = [turn_stack(hist)]
        for _ in range(D):
            hist += [{'role': 'user', 'content': FILL_U}, {'role': 'assistant', 'content': FILL_A}]
            stacks.append(turn_stack(hist))
        clean = hist + [{'role': 'user', 'content': dec}]
        pids = tok(H.tmpl(clean[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
        stacks.append(turn_stack(clean))
        samples.append({'stacks': stacks, 'pids': pids, 'aids': tok(' ' + ans, add_special_tokens=False).input_ids, 'cidx': ACTIONS.index(ans)})
        if (wi + 1) % 20 == 0: print('  %d/%d' % (wi + 1, NW), flush=True)
    r = random.Random(SEED)
    for s in samples: s['test'] = (r.random() < 0.3)
    TR = [s for s in samples if not s['test']]; TE = [s for s in samples if s['test']]
    print('CKV11 train=%d test=%d ncls=6' % (len(TR), len(TE)), flush=True)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    mem = MemEncode(D_S, D_MODEL, NMEM).to(dev)
    opt = torch.optim.Adam([{'params': g.parameters(), 'lr': LR}, {'params': mem.parameters(), 'lr': MLR}, {'params': lora_params, 'lr': LLR}])
    def Sfrom(stks):
        S = g.init()
        for st in stks: S = g.step(S, st.float().to(dev))
        return S
    def memh(stks): return mem(Sfrom(stks)).to(edt)

    def ridge_probe(Xtr, ytr, Xte, yte, ncls, lam=1.0):
        mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        Xtr = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1)], 1); Xte = torch.cat([Xte, torch.ones(Xte.shape[0], 1)], 1)
        Y = torch.zeros(Xtr.shape[0], ncls); Y[range(Xtr.shape[0]), ytr] = 1.0
        A = Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1]); W = torch.linalg.solve(A, Xtr.T @ Y)
        return float(((Xte @ W).argmax(1) == yte).float().mean())

    @torch.no_grad()
    def h_ans(pids, mh):
        _mem['h'] = mh; _mem['on'] = True
        hs = model(pids.unsqueeze(0), output_hidden_states=True).hidden_states[-1][0, -1].float()
        _mem['on'] = False
        return hs

    @torch.no_grad()
    def preds(group, wrong=False):
        out = []; oi = random.Random(SEED + 3)
        for s in group:
            stks = samples[oi.randrange(len(samples))]['stacks'] if wrong else s['stacks']
            _mem['h'] = memh(stks); _mem['on'] = True
            lg = model(s['pids'].unsqueeze(0)).logits[0]; _mem['on'] = False
            out.append(int(lg[-1].argmax()))
        return out

    @torch.no_grad()
    def localize():
        ytr = torch.tensor([s['cidx'] for s in TR]); yte = torch.tensor([s['cidx'] for s in TE])
        Str = torch.stack([Sfrom(s['stacks']).mean(0).float().cpu() for s in TR]); Ste = torch.stack([Sfrom(s['stacks']).mean(0).float().cpu() for s in TE])
        Mtr = torch.stack([mem(Sfrom(s['stacks'])).mean(0).float().cpu() for s in TR]); Mte = torch.stack([mem(Sfrom(s['stacks'])).mean(0).float().cpu() for s in TE])
        pS = ridge_probe(Str, ytr, Ste, yte, 6); pM = ridge_probe(Mtr, ytr, Mte, yte, 6)
        oi = random.Random(SEED + 5); dmem = []; dhid = []
        for s in TE:
            sw = samples[oi.randrange(len(samples))]['stacks']
            mc = mem(Sfrom(s['stacks'])); mw = mem(Sfrom(sw))
            dmem.append(float((mc - mw).norm() / (mc.norm() + 1e-6)))
            hc = h_ans(s['pids'], mc.to(edt)); hw = h_ans(s['pids'], mw.to(edt))
            dhid.append(float((hc - hw).norm() / (hc.norm() + 1e-6)))
        return pS, pM, sum(dmem) / len(dmem), sum(dhid) / len(dhid)

    def report(it):
        g.eval(); mem.eval()
        tgt = [s['aids'][0] for s in TE]; pc = preds(TE); pw = preds(TE, wrong=True)
        accC = sum(int(pc[i] == tgt[i]) for i in range(len(TE))) / len(TE)
        chg = sum(int(pc[i] != pw[i]) for i in range(len(TE))) / len(TE)
        pS, pM, dmem, dhid = localize()
        print('CKV11 L%d it=%-4d | accC=%.3f chg_wrongS=%.3f uniq=%d | probe_S=%.3f probe_mem=%.3f (chance .167) | d_mem_cw=%.3f d_hid_cw=%.4f'
              % (LV, it, accC, chg, len(set(pc)), pS, pM, dmem, dhid), flush=True)
        g.train(); mem.train()

    report(0)
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        s = TR[rng2.randrange(len(TR))]; sw = TR[rng2.randrange(len(TR))]; a = s['aids'][0]
        _mem['h'] = memh(s['stacks']); _mem['on'] = True
        lc = model(s['pids'].unsqueeze(0)).logits[0][-1]; _mem['on'] = False
        _mem['h'] = memh(sw['stacks']); _mem['on'] = True
        lw = model(s['pids'].unsqueeze(0)).logits[0][-1]; _mem['on'] = False
        loss = -torch.log_softmax(lc, -1)[a] + LAM * (-torch.log_softmax(torch.stack([lc[a], lw[a]]), 0)[0])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(mem.parameters()) + lora_params, 1.0); opt.step()
        if it % EVERY == 0:
            print('CKV11 L%d it=%d' % (LV, it), flush=True); report(it)
    print('=== CKV11_L%d_DONE ===' % LV, flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def carry_kv11()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'carry_kv10': carry_kv10()",
                  "elif MODE == 'carry_kv10': carry_kv10()\nelif MODE == 'carry_kv11': carry_kv11()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
