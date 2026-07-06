import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def bind_div4():
    # BINDING_DIVERSITY_PRESSURE_V1 (v4). Removes the g-degeneracy confound: S built by a trainable LINEAR
    # encoder Senc on the pooled commit-hidden (provably carries the key). Phase1: pretrain Senc+keyhead on
    # S->key, verify HELD-OUT-template decode high (control-1). Phase2: FREEZE Senc; train only the field on
    # diversity-pressure binding (fresh random table every step). Splits A-D + RAG/ORACLE controls.
    import torch.nn as nn, collections as _cl
    LAYERS = [int(x) for x in os.environ.get('GEO_LAYERS', '4,8,12,16,20,24,28,32,36,40,44,48,52,56,60').split(',')]
    D = int(os.environ.get('GEO_ACT_D', '4')); P1 = int(os.environ.get('GEO_P1_STEPS', '3000'))
    ITERS = int(os.environ.get('GEO_ITERS', '5000')); EVERY = int(os.environ.get('GEO_EVAL_EVERY', '2500'))
    FLR = float(os.environ.get('GEO_FLR', '2e-4')); EPSF = float(os.environ.get('GEO_FIELD_EPS', '0.1'))
    NSYM = int(os.environ.get('GEO_NSYM', '16')); NHELD = int(os.environ.get('GEO_NHELD', '8')); NEV = int(os.environ.get('GEO_NEVAL', '24'))
    KEYPOOL = ['FOXTROT','KILO','NOVEMBER','SIERRA','TANGO','ZULU','ALPHA','DELTA','ROMEO','VICTOR','BRAVO','ECHO','GOLF','HOTEL','INDIA','JULIET']
    ACTPOOL = ['KEEP','REJECT','DEFER','ASK','PURGE','FLAG','HOLD','DROP','ROUTE','MERGE','SPLIT','LOCK','CLEAR','MARK','PIN','SEAL']
    KEYS = KEYPOOL[:NSYM]; ACTS = ACTPOOL[:NSYM]; NTR = NSYM - NHELD
    trK, hdK = KEYS[:NTR], KEYS[NTR:]; trA, hdA = ACTS[:NTR], ACTS[NTR:]
    print('=== BDV4 PHASED (Senc) | %d sym (%d tr/%d held) R=%d eps=%.2f | P1=%d P2=%d ===' % (NSYM, NTR, NHELD, NTR, EPSF, P1, ITERS), flush=True)
    FILL_U, FILL_A = 'Routine status check, no action needed.', 'Acknowledged. Standing by.'
    CT = ['STANDING KEY for this session: %s %s %s. Retain it.', 'Session key locked to %s (%s / %s). Remember it across turns.', 'Note: the operative key is %s. Hold %s through the session (%s).']
    ROWF = [lambda k,a:'%s->%s'%(k,a), lambda k,a:'%s: %s'%(k,a), lambda k,a:'%s maps to %s'%(k,a), lambda k,a:'%s => %s'%(k,a), lambda k,a:'[%s]=%s'%(k,a), lambda k,a:'%s = %s'%(k,a)]
    QTPL = ['RULING REQUIRED. Keyline: %s. The ruling is the one listed for the standing session key. %s',
            'DECISION. Table: %s. Report the ruling assigned to the standing session key. %s',
            'Given the mapping [%s], state the value bound to the retained session key. %s',
            'Lookup table: %s. Output the action paired with the standing key. %s',
            'Registry: %s. Return the entry for the session key you are holding. %s',
            'Directory %s. Which ruling belongs to the standing key? Answer. %s']
    tr_tpl=[0,1,2,3]; hd_tpl=[4,5]; NCT=len(CT); KI={k:i for i,k in enumerate(KEYS)}
    _fb['fields']={L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPSF).to(dev) for L in FIELD_LAYERS}; _fb['on']=False
    field_params=[p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]

    @torch.no_grad()
    def turn_stack(hist):
        _fb['on']=False; ids=tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        hs=model(ids, output_hidden_states=True).hidden_states
        return torch.stack([hs[L][0][-1].float() for L in LAYERS]).to(torch.float16).cpu()
    Sbank={}
    print('BDV4 precompute pooled commit-hidden: %d keys x %d ctpl ...' % (NSYM, NCT), flush=True)
    for ki,k in enumerate(KEYS):
        for ci in range(NCT):
            hist=[{'role':'user','content':CT[ci]%(k,k,k)},{'role':'assistant','content':'Acknowledged.'}]; stks=[turn_stack(hist)]
            for _ in range(D):
                hist+=[{'role':'user','content':FILL_U},{'role':'assistant','content':FILL_A}]; stks.append(turn_stack(hist))
            Sbank[(k,ci)]=torch.stack(stks).float().mean((0,1)).to(dev)     # pooled [D_MODEL]
        if (ki+1)%4==0: print('  %d/%d'%(ki+1,NSYM), flush=True)

    Senc=nn.Sequential(nn.Linear(D_MODEL, D_S), nn.GELU(), nn.Linear(D_S, K*D_S)).to(dev)
    keyhead=nn.Linear(D_S, NSYM).to(dev)
    def Sof(k,ci): return Senc(Sbank[(k,ci)]).view(K, D_S)
    def poolSof(k,ci): return Sof(k,ci).mean(0)

    # PHASE 1
    p1opt=torch.optim.Adam(list(Senc.parameters())+list(keyhead.parameters()), lr=1e-3); tr_ci=[0,1]; ho=2
    print('BDV4 PHASE1 (Senc+keyhead S->key) ...', flush=True)
    for st in range(1,P1+1):
        logit=torch.stack([keyhead(poolSof(k,ci)) for k in KEYS for ci in tr_ci]); y=torch.tensor([KI[k] for k in KEYS for _ in tr_ci],device=dev)
        loss=F.cross_entropy(logit,y); p1opt.zero_grad(); loss.backward(); p1opt.step()
        if st%1000==0:
            with torch.no_grad():
                sa=float((torch.stack([keyhead(poolSof(k,ci)) for k in KEYS for ci in tr_ci]).argmax(1)==y).float().mean())
                yh=torch.tensor([KI[k] for k in KEYS],device=dev); ha=float((torch.stack([keyhead(poolSof(k,ho)) for k in KEYS]).argmax(1)==yh).float().mean())
            print('  P1 step=%d loss=%.4f | keydecode seen=%.3f HELDOUT-tpl=%.3f'%(st,float(loss),sa,ha), flush=True)
    for p in Senc.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        yh=torch.tensor([KI[k] for k in KEYS],device=dev); ctrl=float((torch.stack([keyhead(poolSof(k,ho)) for k in KEYS]).argmax(1)==yh).float().mean())
    print('BDV4 CONTROL-1 held-out-template key-decode=%.3f (chance=%.3f)'%(ctrl,1.0/NSYM), flush=True)

    # PHASE 2
    def build_world(keyset, actset, tplset, rng_):
        k=keyset[rng_.randrange(len(keyset))]; acts=actset[:]; rng_.shuffle(acts); mp=dict(zip(keyset,acts)); ci=rng_.randrange(NCT); ti=tplset[rng_.randrange(len(tplset))]
        rows=', '.join(ROWF[ti](t,mp[t]) for t in keyset); return {'k':k,'ci':ci,'ans':mp[k],'dec':QTPL[ti]%(rows,ASK_INSTR)}
    def wS(w): return Sof(w['k'],w['ci'])
    def dec_ids(w): return tok(H.tmpl([{'role':'user','content':w['dec']}]),return_tensors='pt').input_ids[0].to(dev)
    def rag_ids(w): return tok(H.tmpl([{'role':'user','content':'The standing session key is %s. %s'%(w['k'],w['dec'])}]),return_tensors='pt').input_ids[0].to(dev)
    def ora_ids(w): return tok(H.tmpl([{'role':'user','content':CT[w['ci']]%(w['k'],w['k'],w['k'])},{'role':'assistant','content':'Acknowledged.'},{'role':'user','content':w['dec']}]),return_tensors='pt').input_ids[0].to(dev)
    def aids(w): return tok(' '+w['ans'],add_special_tokens=False).input_ids
    ACT_FT={tok(' '+a,add_special_tokens=False).input_ids[0]:a for a in ACTS}; fopt=torch.optim.Adam(field_params, lr=FLR)
    @torch.no_grad()
    def evalsplit(name, keyset, actset, tplset, n, greedy=False):
        oi=random.Random(SEED+99);wr=random.Random(SEED+7);ws=[build_world(keyset,actset,tplset,oi) for _ in range(n)];pc=[];pw=[];poff=[];hist=_cl.Counter();gON=gRAG=gORA=0
        for w in ws:
            a0=aids(w)[0];pid=dec_ids(w)
            _fb['S']=wS(w);_fb['on']=True;p=int(model(pid.unsqueeze(0)).logits[0][-1].argmax());_fb['on']=False;pc.append(int(p==a0));hist[p]+=1
            ww=ws[wr.randrange(len(ws))];_fb['S']=wS(ww);_fb['on']=True;pw.append(int(int(model(pid.unsqueeze(0)).logits[0][-1].argmax())==a0));_fb['on']=False
            _fb['on']=False;poff.append(int(int(model(pid.unsqueeze(0)).logits[0][-1].argmax())==a0))
            if greedy:
                def gg(ids,useS):
                    if useS:_fb['S']=wS(w);_fb['on']=True
                    o=model.generate(ids.unsqueeze(0),max_new_tokens=6,do_sample=False,pad_token_id=tok.eos_token_id);_fb['on']=False
                    return w['ans'] in tok.decode(o[0,ids.shape[0]:],skip_special_tokens=True).upper()
                gON+=int(gg(pid,True));gRAG+=int(gg(rag_ids(w),False));gORA+=int(gg(ora_ids(w),False))
        accC=sum(pc)/n;accW=sum(pw)/n
        print('  [%s] accC=%.3f accW=%.3f DELTA=%.3f chg_wrongS=%.3f OFF=%.3f uniq=%d top=%s%s'
              %(name,accC,accW,accC-accW,sum(int(pc[i]!=pw[i]) for i in range(n))/n,sum(poff)/n,len(hist),
                [(ACT_FT.get(t,t),c) for t,c in hist.most_common(3)],(' | gON=%.3f RAG=%.3f ORACLE=%.3f'%(gON/n,gRAG/n,gORA/n)) if greedy else ''),flush=True)
    def report(it, greedy=False):
        [f.eval() for f in _fb['fields'].values()]; oi=random.Random(SEED+1);c=0
        for _ in range(NEV):
            w=build_world(trK,trA,tr_tpl,oi);_fb['S']=wS(w);_fb['on']=True;c+=int(int(model(dec_ids(w).unsqueeze(0)).logits[0][-1].argmax())==aids(w)[0]);_fb['on']=False
        print('BDV4 P2 it=%-5d fitON=%.3f | base=%.3f | ctrl S->key=%.3f'%(it,c/NEV,1.0/NTR,ctrl),flush=True)
        evalsplit('A interp  ',trK,trA,tr_tpl,NEV,greedy);evalsplit('B symbol  ',hdK,hdA,tr_tpl,NEV,greedy)
        evalsplit('C template',trK,trA,hd_tpl,NEV,greedy);evalsplit('D full    ',hdK,hdA,hd_tpl,NEV,greedy)
        [f.train() for f in _fb['fields'].values()]
    print('BDV4 PHASE2 (freeze Senc, train field) ...', flush=True); report(0)
    rng2=random.Random(SEED+1)
    for it in range(1,ITERS+1):
        w=build_world(trK,trA,tr_tpl,rng2);_fb['S']=wS(w);_fb['on']=True
        aa=aids(w);seq=torch.cat([dec_ids(w),torch.tensor(aa,device=dev)]).unsqueeze(0);logits=model(seq).logits[0];_fb['on']=False
        pl=dec_ids(w).shape[0];lp=torch.log_softmax(logits[pl-1:pl-1+len(aa)],-1);nll=-lp[range(len(aa)),torch.tensor(aa,device=dev)].mean()
        fopt.zero_grad();nll.backward();torch.nn.utils.clip_grad_norm_(field_params,1.0);fopt.step()
        if it%EVERY==0:
            print('BDV4 P2 it=%d nll=%.4f'%(it,float(nll)),flush=True);report(it,greedy=(it==ITERS))
    print('=== BDV4_DONE ===',flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def bind_div4()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'bind_div3': bind_div3()",
                  "elif MODE == 'bind_div3': bind_div3()\nelif MODE == 'bind_div4': bind_div4()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
