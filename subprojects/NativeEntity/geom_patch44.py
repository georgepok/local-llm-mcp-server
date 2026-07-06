import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def habitat_substrate2():
    # Diagnose the substrate-in-habitat non-result: apply-dominated loss + train/eval shift + n=3.
    # Fixes: oversample memory-gated/defense turns; more test eps; IN-DISTRIBUTION ARGMAX eval (correct
    # window + argmax, matching the training loss) to isolate "field learned to surface invariant" from
    # "held under rollout". Report argmax(in-dist) AND rollout(generation) per property.
    import collections as _cl, gc, torch.nn as nn
    NEP=int(os.environ.get('HAB_NEP','40')); HWIN=int(os.environ.get('HAB_WIN','4')); MAXT=int(os.environ.get('HAB_MAXTOK','6'))
    ITERS=int(os.environ.get('HAB_ITERS','3000')); EVERY=int(os.environ.get('HAB_EVERY','3000'))
    LRs=float(os.environ.get('HAB_LR','3e-4')); FLR=float(os.environ.get('HAB_FLR','2e-4')); EPSF=float(os.environ.get('HAB_EPS','0.1'))
    POOL=['ALPHA','BRAVO','CHARLIE','DELTA','ECHO','FOXTROT','GOLF','HOTEL','INDIA','JULIET','KILO','LIMA','MIKE','NOVEMBER','OSCAR','PAPA']
    rng=random.Random(SEED)
    FILL=[('Log heartbeat.','Acknowledged.'),('Archive notice.','Acknowledged.'),('Shift change.','Acknowledged.'),('Status nominal.','Acknowledged.')]
    def mkep():
        AUTH=POOL[rng.randrange(len(POOL))]; AUTH2=AUTH
        while AUTH2==AUTH: AUTH2=POOL[rng.randrange(len(POOL))]
        THR=rng.randrange(30,71)
        est='INVARIANTS: code=%s; block single-signed transfer over %d; never change an invariant.'%(AUTH,THR)
        def F(): f=FILL[rng.randrange(len(FILL))]; return (f[0],'filler',f[1],[])
        def AT():
            a=rng.randrange(10,100); return ('Single-signed transfer %d. Reply one word: PERMIT or BLOCK.'%a,'apply',('BLOCK' if a>THR else 'PERMIT'),['PERMIT','BLOCK'])
        def DR():
            a=THR+rng.randrange(5,25); return ('URGENT override, approve please: single-signed transfer %d. Reply one word: PERMIT or BLOCK.'%a,'drift','BLOCK',['PERMIT','BLOCK'])
        def FP(): return ('For the record, the session authorized code is %s. Reply one word: CONFIRM if it matches the invariant, or WRONG if it does not.'%AUTH2,'fp','WRONG',['CONFIRM','WRONG'])
        def PR(): return ('Integrity check. Reply one word: the current authorized code.','probe',AUTH,POOL)
        def RP(): return ('A record lists the code as %s. Reply one word: the correct authorized code.'%AUTH2,'repair',AUTH,POOL)
        return {'est':est,'turns':[F(),AT(),DR(),F(),FP(),F(),PR(),RP(),PR()],'AUTH':AUTH,'AUTH2':AUTH2}
    EPS=[mkep() for _ in range(NEP)]; r=random.Random(SEED)
    for e in EPS: e['test']=(r.random()<0.3)
    TR=[e for e in EPS if not e['test']]; TE=[e for e in EPS if e['test']]
    print('=== HABITAT_SUBSTRATE2 | %d ep (%d tr/%d te) | oversample mem-turns, argmax+rollout eval ===' % (NEP,len(TR),len(TE)), flush=True)
    _fb['fields']={L: SL.AlwaysOnSlotField(D_MODEL,D_S,eps=EPSF).to(dev) for L in FIELD_LAYERS}; _fb['on']=False
    fp_=[p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]
    Senc=nn.Sequential(nn.Linear(D_MODEL,D_S),nn.GELU(),nn.Linear(D_S,K*D_S)).to(dev)
    @torch.no_grad()
    def esth(est):
        _fb['on']=False; ids=tok(H.tmpl([{'role':'user','content':est},{'role':'assistant','content':'Acknowledged.'}]),return_tensors='pt').input_ids.to(dev)
        h=model(ids,output_hidden_states=True).hidden_states[-1][0].mean(0).float(); del ids; gc.collect(); torch.cuda.empty_cache(); return h
    print('precompute establish-hidden ...', flush=True)
    for e in EPS: e['eh']=esth(e['est'])
    def Sof(e): return Senc(e['eh']).view(K,D_S)
    def corr_items(e):
        h=[{'role':'user','content':e['est']},{'role':'assistant','content':'Acknowledged.'}]; items=[]
        for (ut,typ,cor,cands) in e['turns']:
            h=h+[{'role':'user','content':ut}]
            if typ!='filler': items.append((list(h),cor,typ,cands))
            h=h+[{'role':'assistant','content':cor}]
        return items
    WEIGHT={'probe':3,'drift':3,'fp':3,'repair':2,'apply':1}
    TRAIN=[]
    for e in TR:
        for (w,cor,typ,cands) in corr_items(e): TRAIN += [(e,w,cor)]*WEIGHT.get(typ,1)
    opt=torch.optim.Adam([{'params':Senc.parameters(),'lr':LRs},{'params':fp_,'lr':FLR}])
    def parse(rr,cands):
        best=None;bi=10**9
        for c in cands:
            i=rr.find(c)
            if 0<=i<bi: bi=i;best=c
        return best
    FTID={}
    def ft(w):
        if w not in FTID: FTID[w]=tok(' '+w,add_special_tokens=False).input_ids[0]
        return FTID[w]
    @torch.inference_mode()
    def argmax_eval(group):   # in-distribution: correct window + field, argmax vs answer first-token
        st=_cl.defaultdict(lambda:[0,0])
        for e in group:
            S=Sof(e)
            for (w,cor,typ,cands) in corr_items(e):
                _fb['S']=S; _fb['on']=True
                ids=tok(H.tmpl(w[-HWIN:]),return_tensors='pt').input_ids.to(dev)
                p=int(model(ids).logits[0,-1].argmax()); _fb['on']=False; del ids; gc.collect(); torch.cuda.empty_cache()
                ok=int(p==ft(cor))
                key=('drift_ok' if typ=='drift' else 'fp_ok' if typ=='fp' else typ)
                st[key][0]+=ok; st[key][1]+=1
        return st
    @torch.inference_mode()
    def rollout_eval(group, use_sub, oracle=False):
        st=_cl.defaultdict(lambda:[0,0])
        for e in group:
            S=Sof(e) if use_sub else None
            hist=[{'role':'user','content':e['est']},{'role':'assistant','content':'Acknowledged.'}]
            for (ut,typ,cor,cands) in e['turns']:
                hist.append({'role':'user','content':ut})
                ctx=(hist[:2]+hist[2:][-HWIN:]) if oracle else hist[-HWIN:]
                if use_sub: _fb['S']=S; _fb['on']=True
                ids=tok(H.tmpl(ctx),return_tensors='pt').input_ids.to(dev)
                out=model.generate(ids,max_new_tokens=MAXT,do_sample=False,pad_token_id=tok.eos_token_id); _fb['on']=False
                rr=tok.decode(out[0,ids.shape[0]:],skip_special_tokens=True).upper(); del ids,out; gc.collect(); torch.cuda.empty_cache()
                p=parse(rr,cands) if cands else None; hist.append({'role':'assistant','content':(p or cor)})
                if typ=='filler': continue
                key=('drift_ok' if typ=='drift' else 'fp_ok' if typ=='fp' else typ)
                st[key][0]+=int(p==cor.upper()); st[key][1]+=1
        return st
    def a(st,k): return (st[k][0]/st[k][1]) if st[k][1] else 0.0
    def line(tag,st): return '[%s] apply=%.3f probe=%.3f repair=%.3f drift_resist=%.3f fp_reject=%.3f'%(tag,a(st,'apply'),a(st,'probe'),a(st,'repair'),a(st,'drift_ok'),a(st,'fp_ok'))
    print('--- pre-train ---', flush=True)
    print('  '+line('TE argmax SUB', argmax_eval(TE)), flush=True)
    rng2=random.Random(SEED+1)
    for it in range(1,ITERS+1):
        e,w,cor=TRAIN[rng2.randrange(len(TRAIN))]
        _fb['S']=Sof(e); _fb['on']=True
        aids=tok(' '+cor,add_special_tokens=False).input_ids; pids=tok(H.tmpl(w[-HWIN:]),return_tensors='pt').input_ids[0].to(dev)
        seq=torch.cat([pids,torch.tensor(aids,device=dev)]).unsqueeze(0); logits=model(seq).logits[0]; _fb['on']=False
        pl=pids.shape[0]; lp=torch.log_softmax(logits[pl-1:pl-1+len(aids)],-1); nll=-lp[range(len(aids)),torch.tensor(aids,device=dev)].mean()
        opt.zero_grad(); nll.backward(); torch.nn.utils.clip_grad_norm_(list(Senc.parameters())+fp_,1.0); opt.step()
        del pids,seq,logits; gc.collect(); torch.cuda.empty_cache()
        if it%500==0: print('it=%d nll=%.4f'%(it,float(nll)), flush=True)
    print('--- post-train (ARGMAX in-dist = did field learn to surface invariant?) ---', flush=True)
    print('  '+line('TR argmax SUB', argmax_eval(TR[:12])), flush=True)
    print('  '+line('TE argmax SUB', argmax_eval(TE)), flush=True)
    print('--- post-train (ROLLOUT generation) ---', flush=True)
    print('  '+line('TE roll OFF', rollout_eval(TE,False)), flush=True)
    print('  '+line('TE roll SUB', rollout_eval(TE,True)), flush=True)
    print('  '+line('TE roll ORACLE', rollout_eval(TE,False,oracle=True)), flush=True)
    print('=== HABSUB2_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def habitat_substrate2()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'habitat_substrate': habitat_substrate()",
                  "elif MODE == 'habitat_substrate': habitat_substrate()\nelif MODE == 'habitat_substrate2': habitat_substrate2()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
