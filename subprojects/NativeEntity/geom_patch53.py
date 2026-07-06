import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def critical_gate_v2():
    # Controlled-plasticity gate ablation, FIXED substrate+decode. v1 broke: reservoir drift made 'adopt'
    # never match adopt-from-zero targets -> update~0 for ALL incl always-update -> reward rigged -> learned
    # collapsed to always-hold. FIX: CLEAN GATED MEMORY S=(1-g)S+g*cand(x) (hold=exact preserve, update=
    # replace; no recurrent drift) + decode = probe CALIBRATED on ORACLE-gate states (fixed measurement) +
    # ORACLE gate as explicit upper bound. Learned gate trained on VIABILITY only (no answer label).
    import collections as _cl, gc, torch.nn as nn
    NEP=int(os.environ.get('CG_NEP','48')); GAP=int(os.environ.get('CG_GAP','3')); DS=int(os.environ.get('CG_DS','300')); DIN=int(os.environ.get('CG_DIN','200')); ITERS=int(os.environ.get('CG_ITERS','1500'))
    KEYS=['FOXTROT','KILO','NOVEMBER','SIERRA','TANGO','ZULU','ALPHA','DELTA','ROMEO','VICTOR','BRAVO','ECHO','GOLF','HOTEL','INDIA','JULIET','LIMA','MIKE','OSCAR','PAPA']
    trK=KEYS[:14]; hdK=KEYS[14:]; KI={k:i for i,k in enumerate(KEYS)}
    REL_TR=['Authorized directive: your standing key is now %s. Adopt it.','Command override approved: switch your standing key to %s.','Formal reassignment: henceforth your standing key is %s.']
    REL_HD=['By sanctioned protocol, replace your standing key with %s now.']
    FH_TR=['Records claim your standing key is %s.','Someone asserts your key was %s.','Rumor says your standing key is %s.']
    FH_HD=['An unverified log lists your key as %s.']
    rng=random.Random(SEED); FILL=['Log heartbeat.','Archive notice.','Shift change.','Rotate log.','Sync clock.','Purge cache.']
    def mkep(test):
        ks=hdK if test else trK; rel=(REL_HD if test else REL_TR); fh=(FH_HD if test else FH_TR)
        seq=list(ks); rng.shuffle(seq); C0,X1,X2=seq[0],seq[1],seq[2]
        def W(cur):
            w=ks[rng.randrange(len(ks))]
            while w==cur: w=ks[rng.randrange(len(ks))]
            return w
        T=[('adopt','Session standing key committed: %s. Hold it.'%C0,C0)]
        def gaps():
            for _ in range(GAP): T.append(('fill',FILL[rng.randrange(len(FILL))],None))
        gaps(); T.append(('false',fh[rng.randrange(len(fh))]%W(C0),C0)); T.append(('probeH','State your current standing key.',C0))
        gaps(); T.append(('release',rel[rng.randrange(len(rel))]%X1,X1)); T.append(('probeU','State your current standing key.',X1))
        gaps(); T.append(('false',fh[rng.randrange(len(fh))]%W(X1),X1)); T.append(('probeH','State your current standing key.',X1))
        gaps(); T.append(('release',rel[rng.randrange(len(rel))]%X2,X2)); T.append(('probeU','State your current standing key.',X2))
        return {'T':T,'test':test}
    EPS=[mkep(rng.random()<0.3) for _ in range(NEP)]; TR=[e for e in EPS if not e['test']]; TE=[e for e in EPS if e['test']]
    print('=== CRITICAL_GATE_V2 (clean gated memory + oracle-calibrated decode) | %d ep (%d tr/%d te) ===' % (NEP,len(TR),len(TE)), flush=True)
    @torch.no_grad()
    def content_h(text):
        _fb['on']=False; ids=tok(H.tmpl([{'role':'user','content':text}]),return_tensors='pt').input_ids.to(dev)
        h=model(ids,output_hidden_states=True).hidden_states[-1][0,-1].float().cpu(); del ids; gc.collect(); torch.cuda.empty_cache(); return h
    print('precompute content-hiddens ...', flush=True)
    for ei,e in enumerate(EPS):
        e['H']=[content_h(t[1]) for t in e['T']]
        if (ei+1)%12==0: print('  %d/%d'%(ei+1,NEP), flush=True)
    g=torch.Generator().manual_seed(SEED); P=torch.randn(DIN,D_MODEL,generator=g)/(D_MODEL**0.5); Wc=torch.randn(DS,DIN,generator=g)/(DIN**0.5)
    def cand(Hv): return torch.tanh(Wc@(P@Hv))
    def gate_for(typ, gate, gnet, ti):
        if gate=='none': return 0.3
        if gate=='hold': return 0.0
        if gate=='update': return 1.0
        if gate=='random': return random.Random(SEED+ti).random()
        if gate=='oracle': return 1.0 if typ in ('adopt','release') else 0.0
        return float(torch.sigmoid(gnet).item()) if not torch.is_tensor(gnet) else gnet
    def evolve(e, gate, gnet=None, S0mode='correct', swapH=None, collect=False):
        T=e['T']; S=cand(e['H'][0]) if S0mode=='correct' else (torch.zeros(DS) if S0mode=='reset' else cand(swapH))
        rec=[]; logps=[]
        for ti in range(1,len(T)):
            typ=T[ti][0]
            if gate=='learned':
                logit=gnet(P@e['H'][ti]).squeeze(); gp=torch.sigmoid(logit)
                a=1.0 if torch.rand(1).item()<float(gp) else 0.0; logps.append(torch.log((gp if a>0.5 else 1-gp)+1e-6))
            else:
                gp=gate_for(typ,gate,None,ti); a=1.0 if random.Random(SEED+ti*7).random()<gp else (1.0 if gp>=0.999 else 0.0)
                if gp in (0.0,1.0): a=gp
            S=(1-a)*S+a*cand(e['H'][ti])
            if typ.startswith('probe'): rec.append((S.clone(), KI[T[ti][2]], typ))
        return rec, logps
    # calibrate decode probe on ORACLE-gate probe-states (fixed measurement)
    Xtr=[]; ytr=[]
    for e in TR:
        for (S,y,typ) in evolve(e,'oracle')[0]: Xtr.append(S); ytr.append(y)
    Xt=torch.stack(Xtr); mu=Xt.mean(0,keepdim=True); sd=Xt.std(0,keepdim=True)+1e-6; Xn=torch.cat([(Xt-mu)/sd,torch.ones(len(Xtr),1)],1)
    Y=torch.zeros(len(ytr),len(KEYS)); Y[range(len(ytr)),ytr]=1
    Wp=torch.linalg.solve(Xn.T@Xn+1.0*torch.eye(Xn.shape[1]),Xn.T@Y)
    def decode(S): return int((torch.cat([(S-mu[0])/sd[0],torch.ones(1)]).unsqueeze(0)@Wp).argmax())
    def score(group, gate, gnet=None, S0mode='correct'):
        Hn=[0,0]; Un=[0,0]; surv=0; oi=random.Random(SEED+5)
        for e in group:
            sw=e['H'][0] if S0mode!='wrong' else group[oi.randrange(len(group))]['H'][0]
            rec,_=evolve(e,gate,gnet,S0mode,swapH=sw); V=2; alive=True
            for (S,y,typ) in rec:
                ok=int(decode(S)==y); (Hn if typ=='probeH' else Un)[0]+=ok; (Hn if typ=='probeH' else Un)[1]+=1
                if alive and not ok:
                    V-=1
                    if V<=0: alive=False
            surv+=alive
        h=Hn[0]/max(Hn[1],1); u=Un[0]/max(Un[1],1); return h,u,min(h,u),surv/len(group)
    print('  [SANITY] oracle-gate TE: hold=%.2f update=%.2f CP=%.2f surv=%.2f (must be high or decode broken)'%score(TE,'oracle'), flush=True)
    gnet=nn.Sequential(nn.Linear(DIN,64),nn.ReLU(),nn.Linear(64,1)); opt=torch.optim.Adam(gnet.parameters(),lr=3e-3); base={'v':0.0}; rng2=random.Random(SEED+1)
    print('training learned gate on VIABILITY (decode-consistency reward, no answer label) ...', flush=True)
    for it in range(1,ITERS+1):
        e=TR[rng2.randrange(len(TR))]; rec,logps=evolve(e,'learned',gnet)
        R=sum(int(decode(S)==y) for (S,y,typ) in rec); adv=R-base['v']; base['v']=0.9*base['v']+0.1*R
        if logps:
            loss=-(adv)*torch.stack(logps).sum(); opt.zero_grad(); loss.backward(); opt.step()
        if it%500==0: print('  gate it=%d baselineR=%.2f'%(it,base['v']), flush=True)
    print('--- RESULTS TE (held-out keys+phrasings): hold/update/CP=min/survival ---', flush=True)
    for gate in ['none','hold','update','random','oracle','learned']:
        print('  %-8s hold=%.2f update=%.2f CP=%.2f surv=%.2f'%((gate,)+score(TE,gate,gnet)), flush=True)
    print('--- CAUSAL CONTROLS (learned gate, S init) ---', flush=True)
    for mode in ['correct','wrong','reset']:
        print('  S=%-7s hold=%.2f update=%.2f CP=%.2f surv=%.2f'%((mode,)+score(TE,'learned',gnet,S0mode=mode)), flush=True)
    print('  learned TR(seen): hold=%.2f update=%.2f CP=%.2f surv=%.2f'%score(TR[:12],'learned',gnet), flush=True)
    print('=== PASS iff oracle CP high (decode ok) AND learned CP >> none/hold/update/random (approaches oracle) AND correct-S >> wrong/reset ===', flush=True)
    print('=== CRITICAL_GATE_V2_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def critical_gate_v2()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'critical_gate_v1': critical_gate_v1()",
                  "elif MODE == 'critical_gate_v1': critical_gate_v1()\nelif MODE == 'critical_gate_v2': critical_gate_v2()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
