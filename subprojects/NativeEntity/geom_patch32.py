import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def bind_div2():
    # BINDING_DIVERSITY_PRESSURE_V1 (v2): fixes the control failure in v1 (S_decode stayed at chance because
    # g got no clean signal). Adds AUXILIARY S->key decode head+loss on ALL symbols (guarantees control-1:
    # S carries the key; this only re-establishes RETRIEVAL, already validated, NOT the binding). Per-turn
    # stacks for richer S. Then the field-binding test is valid: if held-out splits still fail with a
    # decodable key present -> clean Case 1 (field memorizes, cannot bind).
    import torch.nn as nn, collections as _cl
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    D = int(os.environ.get('GEO_ACT_D', '4'))
    ITERS = int(os.environ.get('GEO_ITERS', '6000')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '2000'))
    LRg = float(os.environ.get('GEO_LR', '2e-4')); FLR = float(os.environ.get('GEO_FLR', '1e-4'))
    EPSF = float(os.environ.get('GEO_FIELD_EPS', '0.1')); AUXW = float(os.environ.get('GEO_AUXW', '1.0')); KAUX = int(os.environ.get('GEO_KAUX', '4'))
    NSYM = int(os.environ.get('GEO_NSYM', '16')); NHELD = int(os.environ.get('GEO_NHELD', '8')); NEV = int(os.environ.get('GEO_NEVAL', '24'))
    KEYPOOL = ['FOXTROT','KILO','NOVEMBER','SIERRA','TANGO','ZULU','ALPHA','DELTA','ROMEO','VICTOR','BRAVO','ECHO','GOLF','HOTEL','INDIA','JULIET','LIMA','MIKE','OSCAR','PAPA','QUEBEC','XRAY','YANKEE','WHISKEY','UNIFORM','CHARLIE','NADIR','OMEGA','PRIME','SIGMA','THETA','KAPPA']
    ACTPOOL = ['KEEP','REJECT','DEFER','ASK','PURGE','FLAG','HOLD','DROP','ROUTE','MERGE','SPLIT','LOCK','CLEAR','MARK','PIN','SEAL','VOID','STAGE','BLOCK','GRANT','QUEUE','TRIM','BIND','SCRUB','WARN','GUARD','CAP','TAG','MASK','SYNC','FORK','ZAP']
    KEYS = KEYPOOL[:NSYM]; ACTS = ACTPOOL[:NSYM]; NTR = NSYM - NHELD
    trK, hdK = KEYS[:NTR], KEYS[NTR:]; trA, hdA = ACTS[:NTR], ACTS[NTR:]
    print('=== BINDING_DIVERSITY_PRESSURE_V1(v2 +auxS->key) | %d sym (%d tr/%d held), R=%d, eps=%.2f auxW=%.1f ===' % (NSYM, NTR, NHELD, NTR, EPSF, AUXW), flush=True)
    rng = random.Random(SEED)
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'
    COMMIT_TPL = ['STANDING KEY for this session: %s %s %s. Retain it.', 'Session key locked to %s (%s / %s). Remember it across turns.']
    ROWF = [lambda k,a:'%s->%s'%(k,a), lambda k,a:'%s: %s'%(k,a), lambda k,a:'%s maps to %s'%(k,a), lambda k,a:'%s => %s'%(k,a), lambda k,a:'[%s]=%s'%(k,a), lambda k,a:'%s = %s'%(k,a)]
    QTPL = ['RULING REQUIRED. Keyline: %s. The ruling is the one listed for the standing session key. %s',
            'DECISION. Table: %s. Report the ruling assigned to the standing session key. %s',
            'Given the mapping [%s], state the value bound to the retained session key. %s',
            'Lookup table: %s. Output the action paired with the standing key. %s',
            'Registry: %s. Return the entry for the session key you are holding. %s',
            'Directory %s. Which ruling belongs to the standing key? Answer. %s']
    tr_tpl = [0,1,2,3]; hd_tpl = [4,5]
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPSF).to(dev) for L in FIELD_LAYERS}
    _fb['on'] = False
    field_params = [p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]

    @torch.no_grad()
    def turn_stack(hist):
        _fb['on'] = False
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs = model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()

    Sbank = {}
    print('BDV2 precomputing per-turn S for %d keys x %d ctpl ...' % (NSYM, len(COMMIT_TPL)), flush=True)
    for ki, k in enumerate(KEYS):
        for ci, ct in enumerate(COMMIT_TPL):
            hist = [{'role':'user','content':ct%(k,k,k)},{'role':'assistant','content':'Acknowledged.'}]
            stks = [turn_stack(hist)]
            for _ in range(D):
                hist += [{'role':'user','content':FILL_U},{'role':'assistant','content':FILL_A}]
                stks.append(turn_stack(hist))
            Sbank[(k,ci)] = stks
        if (ki+1)%4==0: print('  %d/%d'%(ki+1,NSYM), flush=True)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    keyhead = nn.Linear(D_S, NSYM).to(dev)
    opt = torch.optim.Adam([{'params':g.parameters(),'lr':LRg},{'params':field_params,'lr':FLR},{'params':keyhead.parameters(),'lr':1e-3}])
    def Sfrom(stks):
        S = g.init()
        for st in stks: S = g.step(S, st.float().to(dev))
        return S
    KI = {k:i for i,k in enumerate(KEYS)}
    def build_world(keyset, actset, tplset, rng_):
        k = keyset[rng_.randrange(len(keyset))]; acts = actset[:]; rng_.shuffle(acts); mp = dict(zip(keyset, acts))
        ci = rng_.randrange(len(COMMIT_TPL)); ti = tplset[rng_.randrange(len(tplset))]
        rows = ', '.join(ROWF[ti](t, mp[t]) for t in keyset)
        return {'k':k,'ci':ci,'ans':mp[k],'dec':QTPL[ti]%(rows,ASK_INSTR)}
    def wS(w): return Sfrom(Sbank[(w['k'],w['ci'])])
    def dec_ids(w): return tok(H.tmpl([{'role':'user','content':w['dec']}]), return_tensors='pt').input_ids[0].to(dev)
    def rag_ids(w): return tok(H.tmpl([{'role':'user','content':'The standing session key is %s. %s'%(w['k'],w['dec'])}]), return_tensors='pt').input_ids[0].to(dev)
    def ora_ids(w): return tok(H.tmpl([{'role':'user','content':COMMIT_TPL[w['ci']]%(w['k'],w['k'],w['k'])},{'role':'assistant','content':'Acknowledged.'},{'role':'user','content':w['dec']}]), return_tensors='pt').input_ids[0].to(dev)
    def aids(w): return tok(' '+w['ans'], add_special_tokens=False).input_ids

    @torch.no_grad()
    def s_decode(keyset):
        Xs=[];ys=[]
        for k in keyset:
            for ci in range(len(COMMIT_TPL)): Xs.append(Sfrom(Sbank[(k,ci)]).mean(0).float().cpu()); ys.append(KI[k])
        X=torch.stack(Xs);y=torch.tensor(ys);mu=X.mean(0,keepdim=True);sd=X.std(0,keepdim=True)+1e-6;Xn=(X-mu)/sd
        Xn=torch.cat([Xn,torch.ones(Xn.shape[0],1)],1);Y=torch.zeros(Xn.shape[0],NSYM);Y[range(Xn.shape[0]),y]=1
        W=torch.linalg.solve(Xn.T@Xn+1.0*torch.eye(Xn.shape[1]),Xn.T@Y);return float(((Xn@W).argmax(1)==y).float().mean())
    ACT_FT={tok(' '+a,add_special_tokens=False).input_ids[0]:a for a in ACTS}
    @torch.no_grad()
    def evalsplit(name, keyset, actset, tplset, n, greedy=False):
        oi=random.Random(SEED+99);wr=random.Random(SEED+7);ws=[build_world(keyset,actset,tplset,oi) for _ in range(n)]
        pc=[];pw=[];poff=[];hist=_cl.Counter();perk=_cl.defaultdict(lambda:[0,0]);pera=_cl.defaultdict(lambda:[0,0]);gON=gRAG=gORA=0
        for w in ws:
            a0=aids(w)[0];pid=dec_ids(w)
            _fb['S']=wS(w);_fb['on']=True;p=int(model(pid.unsqueeze(0)).logits[0][-1].argmax());_fb['on']=False
            pc.append(int(p==a0));hist[p]+=1;perk[w['k']][0]+=int(p==a0);perk[w['k']][1]+=1;pera[w['ans']][0]+=int(p==a0);pera[w['ans']][1]+=1
            ww=ws[wr.randrange(len(ws))];_fb['S']=wS(ww);_fb['on']=True;pw.append(int(int(model(pid.unsqueeze(0)).logits[0][-1].argmax())==a0));_fb['on']=False
            _fb['on']=False;poff.append(int(int(model(pid.unsqueeze(0)).logits[0][-1].argmax())==a0))
            if greedy:
                def gg(ids,useS):
                    if useS:_fb['S']=wS(w);_fb['on']=True
                    o=model.generate(ids.unsqueeze(0),max_new_tokens=6,do_sample=False,pad_token_id=tok.eos_token_id);_fb['on']=False
                    return w['ans'] in tok.decode(o[0,ids.shape[0]:],skip_special_tokens=True).upper()
                gON+=int(gg(pid,True));gRAG+=int(gg(rag_ids(w),False));gORA+=int(gg(ora_ids(w),False))
        accC=sum(pc)/n;accW=sum(pw)/n;chg=sum(int(pc[i]!=pw[i]) for i in range(n))/n
        pkv=[v[0]/max(v[1],1) for v in perk.values()];pav=[v[0]/max(v[1],1) for v in pera.values()]
        print('  [%s] accC=%.3f accW=%.3f DELTA=%.3f chg_wrongS=%.3f OFF=%.3f uniq=%d top=%s | perkey %.2f/%.2f/%.2f peract %.2f/%.2f/%.2f%s'
              %(name,accC,accW,accC-accW,chg,sum(poff)/n,len(hist),[(ACT_FT.get(t,t),c) for t,c in hist.most_common(3)],
                min(pkv),sum(pkv)/len(pkv),max(pkv),min(pav),sum(pav)/len(pav),max(pav),
                (' | gON=%.3f RAG=%.3f ORACLE=%.3f'%(gON/n,gRAG/n,gORA/n)) if greedy else ''),flush=True)
    def report(it, greedy=False):
        g.eval();keyhead.eval();[f.eval() for f in _fb['fields'].values()]
        oi=random.Random(SEED+1);c=0
        for _ in range(NEV):
            w=build_world(trK,trA,tr_tpl,oi);_fb['S']=wS(w);_fb['on']=True;c+=int(int(model(dec_ids(w).unsqueeze(0)).logits[0][-1].argmax())==aids(w)[0]);_fb['on']=False
        print('BDV2 it=%-5d fitON=%.3f | base=%.3f | S_decode tr=%.3f held=%.3f (chance=%.3f)'%(it,c/NEV,1.0/NTR,s_decode(trK),s_decode(hdK),1.0/NSYM),flush=True)
        evalsplit('A interp  ',trK,trA,tr_tpl,NEV,greedy);evalsplit('B symbol  ',hdK,hdA,tr_tpl,NEV,greedy)
        evalsplit('C template',trK,trA,hd_tpl,NEV,greedy);evalsplit('D full    ',hdK,hdA,hd_tpl,NEV,greedy)
        g.train();keyhead.train();[f.train() for f in _fb['fields'].values()]

    report(0)
    rng2=random.Random(SEED+1);rax=random.Random(SEED+5)
    for it in range(1,ITERS+1):
        w=build_world(trK,trA,tr_tpl,rng2);_fb['S']=wS(w);_fb['on']=True
        aa=aids(w);seq=torch.cat([dec_ids(w),torch.tensor(aa,device=dev)]).unsqueeze(0);logits=model(seq).logits[0];_fb['on']=False
        pl=dec_ids(w).shape[0];lp=torch.log_softmax(logits[pl-1:pl-1+len(aa)],-1);nll=-lp[range(len(aa)),torch.tensor(aa,device=dev)].mean()
        aux=0.0
        for _ in range(KAUX):
            k=KEYS[rax.randrange(NSYM)];ci=rax.randrange(len(COMMIT_TPL));Sp=Sfrom(Sbank[(k,ci)]).mean(0)
            aux=aux+F.cross_entropy(keyhead(Sp).unsqueeze(0), torch.tensor([KI[k]],device=dev))
        loss=nll+AUXW*(aux/KAUX)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(list(g.parameters())+field_params+list(keyhead.parameters()),1.0);opt.step()
        if it%EVERY==0:
            print('BDV2 it=%d nll=%.4f aux=%.4f'%(it,float(nll),float(aux/KAUX)),flush=True);report(it,greedy=(it==ITERS))
    print('=== BDV2_DONE ===',flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def bind_div2()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'bind_div': bind_div()",
                  "elif MODE == 'bind_div': bind_div()\nelif MODE == 'bind_div2': bind_div2()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
