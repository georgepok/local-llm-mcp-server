import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def carry_kv12():
    # AMPLIFY the world-specific memory signal to separate H-opt (signal too weak) from H-fund (LLM washout).
    # Localization (kv11) showed: content IS decodable in memory (probe_mem=1.0) but tiny in magnitude
    # (d_mem->0) and LLM propagates ~1% (d_hid~0.01). Inject mem_bar + ALPHA*(mem(S)-mem_bar); train with it.
    #   ALPHA up => Delta>0, d_hid rises  -> signal was too weak (H-OPT, fixable, direction reopens)
    #   ALPHA up => Delta stays 0         -> frozen LLM washes out latent memory regardless (H-FUND)
    import torch.nn as nn
    import transformers.models.qwen3_5.modeling_qwen3_5 as QM
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    LV = int(os.environ.get('GEO_LEVEL', '1'))
    D = int(os.environ.get('GEO_ACT_D', '6')); NW = int(os.environ.get('GEO_NW', '60'))
    NMEM = int(os.environ.get('GEO_NMEM', '16'))
    ITERS = int(os.environ.get('GEO_ITERS', '800')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '400'))
    LR = float(os.environ.get('GEO_LR', '1e-4')); MLR = float(os.environ.get('GEO_MLR', '3e-4'))
    LLR = float(os.environ.get('GEO_LORA_LR', '2e-4')); R = int(os.environ.get('GEO_LORA_R', '16'))
    LSCALE = float(os.environ.get('GEO_LORA_SCALE', '1.0')); QKVO = os.environ.get('GEO_LORA_QKVO', 'q')
    LAM = float(os.environ.get('GEO_CONTRAST_LAM', '1.0'))
    ALPHA_TR = float(os.environ.get('GEO_AMP', '8.0'))
    SWEEP = [float(x) for x in os.environ.get('GEO_AMP_SWEEP', '1,4,8,16,32').split(',')]
    FULL = [i for i in range(len(model.model.layers)) if hasattr(model.model.layers[i], 'self_attn')]
    INJ = [int(x) for x in os.environ.get('GEO_INJ_LAYERS', '43,51,59').split(',')]
    INJ = [L for L in INJ if L in FULL]
    print('=== CARRY_KV12(AMPLIFY, ALPHA_tr=%.1f sweep=%s) L%d | inj=%s ===' % (ALPHA_TR, SWEEP, LV, INJ), flush=True)
    rng = random.Random(SEED)
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
    print('CKV12 building %d worlds ...' % NW, flush=True)
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
    print('CKV12 train=%d test=%d' % (len(TR), len(TE)), flush=True)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    mem = MemEncode(D_S, D_MODEL, NMEM).to(dev)
    opt = torch.optim.Adam([{'params': g.parameters(), 'lr': LR}, {'params': mem.parameters(), 'lr': MLR}, {'params': lora_params, 'lr': LLR}])
    def Sfrom(stks):
        S = g.init()
        for st in stks: S = g.step(S, st.float().to(dev))
        return S
    bar = {'v': None}
    @torch.no_grad()
    def refresh_bar():
        bar['v'] = torch.stack([mem(Sfrom(s['stacks'])) for s in TR]).mean(0)      # [M, d_model] shared component
    def amp(stks, alpha):
        m = mem(Sfrom(stks)); mb = bar['v'].detach()
        return (mb + alpha * (m - mb)).to(edt)

    @torch.no_grad()
    def h_ans(pids, mh):
        _mem['h'] = mh; _mem['on'] = True
        hs = model(pids.unsqueeze(0), output_hidden_states=True).hidden_states[-1][0, -1].float(); _mem['on'] = False
        return hs
    @torch.no_grad()
    def sweep_eval():
        oi = random.Random(SEED + 5); tgt = [s['aids'][0] for s in TE]
        for al in SWEEP:
            pc = []; pw = []; dh = []
            for s in TE:
                sw = samples[oi.randrange(len(samples))]['stacks']
                mc = amp(s['stacks'], al); mw = amp(sw, al)
                _mem['h'] = mc; _mem['on'] = True
                pc.append(int(model(s['pids'].unsqueeze(0)).logits[0][-1].argmax())); _mem['on'] = False
                _mem['h'] = mw; _mem['on'] = True
                pw.append(int(model(s['pids'].unsqueeze(0)).logits[0][-1].argmax())); _mem['on'] = False
                dh.append(float((h_ans(s['pids'], mc) - h_ans(s['pids'], mw)).norm() / (h_ans(s['pids'], mc).norm() + 1e-6)))
            accC = sum(int(pc[i] == tgt[i]) for i in range(len(TE))) / len(TE)
            accW = sum(int(pw[i] == tgt[i]) for i in range(len(TE))) / len(TE)
            chg = sum(int(pc[i] != pw[i]) for i in range(len(TE))) / len(TE)
            print('   ALPHA=%-5.1f accC=%.3f accW=%.3f DELTA=%.3f uniq=%d chg_wrongS=%.3f d_hid=%.4f'
                  % (al, accC, accW, accC - accW, len(set(pc)), chg, sum(dh) / len(dh)), flush=True)

    refresh_bar()
    print('CKV12 L%d it=0 (untrained) sweep:' % LV, flush=True); sweep_eval()
    rng2 = random.Random(SEED + 1)
    for it in range(1, ITERS + 1):
        s = TR[rng2.randrange(len(TR))]; sw = TR[rng2.randrange(len(TR))]; a = s['aids'][0]
        _mem['h'] = amp(s['stacks'], ALPHA_TR); _mem['on'] = True
        lc = model(s['pids'].unsqueeze(0)).logits[0][-1]; _mem['on'] = False
        _mem['h'] = amp(sw['stacks'], ALPHA_TR); _mem['on'] = True
        lw = model(s['pids'].unsqueeze(0)).logits[0][-1]; _mem['on'] = False
        loss = -torch.log_softmax(lc, -1)[a] + LAM * (-torch.log_softmax(torch.stack([lc[a], lw[a]]), 0)[0])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(mem.parameters()) + lora_params, 1.0); opt.step()
        if it % EVERY == 0:
            refresh_bar(); print('CKV12 L%d it=%d (trained ALPHA_tr=%.1f) sweep:' % (LV, it, ALPHA_TR), flush=True); sweep_eval()
    print('=== CKV12_L%d_DONE ===' % LV, flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def carry_kv12()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'carry_kv11': carry_kv11()",
                  "elif MODE == 'carry_kv11': carry_kv11()\nelif MODE == 'carry_kv12': carry_kv12()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
