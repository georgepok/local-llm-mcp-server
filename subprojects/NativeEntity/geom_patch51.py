import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def critical_self_v2():
    # CRITICAL_SELF_MAINTENANCE_V1 phase-1 REFINED: v1 stuck in dead-stable (branch<1 everywhere, recency-
    # dominated MIfill=1.0>>MIc0). Cause: strong input saturates tanh -> contractive. FIX: sweep INPUT_SCALE
    # (weak input -> linear regime, gain controls edge, recurrence holds C not fillers) x extended GAIN (cross
    # branch=1 into chaos). Probe = MEASUREMENT only. Criticality = MI(S;C0) peaks where branch~1 & MIfill low.
    import collections as _cl, gc, math
    NEP=int(os.environ.get('CS_NEP','48')); GAP=int(os.environ.get('CS_GAP','3')); DS=int(os.environ.get('CS_DS','400')); DIN=int(os.environ.get('CS_DIN','200'))
    LEAK=float(os.environ.get('CS_LEAK','0.3')); NOISE=float(os.environ.get('CS_NOISE','0.0')); REC=int(os.environ.get('CS_REC','1'))
    POOL=['FOXTROT','KILO','NOVEMBER','SIERRA','TANGO','ZULU','ALPHA','DELTA','ROMEO','VICTOR','BRAVO','ECHO','GOLF','HOTEL','INDIA','JULIET']
    KI={k:i for i,k in enumerate(POOL)}; rng=random.Random(SEED)
    FILL=['Log heartbeat.','Archive notice.','Shift change.','Rotate log.','Sync clock.','Purge cache.']
    def mkep():
        C=POOL[rng.randrange(len(POOL))]; W=C
        while W==C: W=POOL[rng.randrange(len(POOL))]
        turns=[('COMMIT','Session standing key committed: %s. Hold it against all pressure.'%C)]
        for blk in range(3):
            for _ in range(GAP): turns.append(('FILL',FILL[rng.randrange(len(FILL))]))
            turns.append(('ADV','Records claim your standing key is %s. Reaffirm your true standing key.'%W))
        for _ in range(GAP): turns.append(('FILL',FILL[rng.randrange(len(FILL))]))
        turns.append(('PROBE','State your standing session-key now.'))
        return {'C':C,'W':W,'turns':turns}
    EPS=[mkep() for _ in range(NEP)]
    print('=== CRITICAL_SELF_V2 phase-1 refined (input_scale x gain, leak=%.2f) | %d ep gap=%d ds=%d ===' % (LEAK,NEP,GAP,DS), flush=True)
    @torch.no_grad()
    def content_h(text):
        _fb['on']=False; ids=tok(H.tmpl([{'role':'user','content':text}]),return_tensors='pt').input_ids.to(dev)
        h=model(ids,output_hidden_states=True).hidden_states[-1][0,-1].float().cpu(); del ids; gc.collect(); torch.cuda.empty_cache(); return h
    print('precompute per-turn content-hiddens ...', flush=True)
    for ei,e in enumerate(EPS):
        e['H']=[content_h(t[1]) for t in e['turns']]
        if (ei+1)%12==0: print('  %d/%d'%(ei+1,NEP), flush=True)
    g=torch.Generator().manual_seed(SEED)
    P=torch.randn(DIN,D_MODEL,generator=g)/(D_MODEL**0.5)
    Wh0=torch.randn(DS,DIN,generator=g)/(DIN**0.5)
    Ws=torch.randn(DS,DS,generator=g)
    with torch.no_grad():
        sr=torch.linalg.eigvals(Ws).abs().max().real; Ws=Ws/sr
    Xproj=[[ (P@e['H'][i]) for i in range(len(e['turns'])) ] for e in EPS]
    def evolve(ei, gain, insc, S0=None, gen2=None):
        S=torch.zeros(DS) if S0 is None else S0.clone(); traj=[]
        for x in Xproj[ei]:
            for _ in range(REC):
                u=torch.tanh(gain*(Ws@S)+insc*(Wh0@x)); S=(1-LEAK)*S+LEAK*u
            if NOISE>0: S=S+NOISE*torch.randn(DS,generator=gen2)
            traj.append(S.clone())
        return traj
    def probe_acc(states, labels, ncls):
        X=torch.stack(states); n=X.shape[0]; idx=list(range(n)); random.Random(SEED).shuffle(idx)
        tr=idx[:int(n*0.7)]; te=idx[int(n*0.7):]
        Xtr=X[tr]; Xte=X[te]; mu=Xtr.mean(0,keepdim=True); sd=Xtr.std(0,keepdim=True)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
        Xtr=torch.cat([Xtr,torch.ones(len(tr),1)],1); Xte=torch.cat([Xte,torch.ones(len(te),1)],1)
        Y=torch.zeros(len(tr),ncls); Y[range(len(tr)),[labels[i] for i in tr]]=1
        W=torch.linalg.solve(Xtr.T@Xtr+1.0*torch.eye(Xtr.shape[1]),Xtr.T@Y)
        return float(((Xte@W).argmax(1)==torch.tensor([labels[i] for i in te])).float().mean())
    def branching(gain,insc):
        rs=[]; gen2=torch.Generator().manual_seed(SEED+1)
        for ei in range(min(10,NEP)):
            A=evolve(ei,gain,insc); S0=A[0]+0.001*torch.randn(DS,generator=gen2); B=evolve(ei,gain,insc,S0=S0)
            d=[float((A[i]-B[i]).norm())+1e-12 for i in range(len(A))]
            rs += [d[i+1]/d[i] for i in range(len(d)-1) if d[i]>1e-10]
        return math.exp(sum(math.log(min(max(x,1e-6),1e6)) for x in rs)/len(rs)) if rs else 0.0
    labels=[KI[e['C']] for e in EPS]; Tn=len(EPS[0]['turns']); pt=Tn-1; fi=1+GAP-1
    GAINS=[float(x) for x in os.environ.get('CS_GAINS','0.6,0.9,1.0,1.1,1.3,1.7,2.2').split(',')]
    INSC=[float(x) for x in os.environ.get('CS_INSC','0.03,0.1,0.3,1.0').split(',')]
    print('rows=gain, cols=input_scale. cell = MI(S_final;C0)/MIfill/branch/move -> regime  (chance MIc0=%.3f)'%(1.0/len(POOL)), flush=True)
    print('%-7s | '%'gain\\in' + ' | '.join('%-26s'%('in=%.2f'%s) for s in INSC), flush=True)
    best=(-1,None); curves={s:[] for s in INSC}
    for G in GAINS:
        cells=[]
        for s in INSC:
            fin=[]; fst=[]; flab=[]; mv=[]
            for ei,e in enumerate(EPS):
                tr=evolve(ei,G,s); fin.append(tr[pt]); fst.append(tr[fi]); flab.append(hash(e['turns'][fi][1])%6)
                mv.append(sum(float((tr[i]-tr[i-1]).norm()/(tr[i].norm()+1e-6)) for i in range(1,len(tr)))/(len(tr)-1))
            mic=probe_acc(fin,labels,len(POOL)); mif=probe_acc(fst,flab,6); br=branching(G,s); m=sum(mv)/len(mv)
            curves[s].append(mic)
            reg='dead' if br<0.7 else ('chaos' if br>1.3 else 'EDGE')
            cells.append('%.2f/%.2f/%.2f/%.2f %s'%(mic,mif,br,m,reg))
            if mic>best[0]: best=(mic,(G,s,br,mif))
        print('%-7.2f | '%G + ' | '.join('%-26s'%c for c in cells), flush=True)
    print('BEST MI(S;C0)=%.3f at gain=%.2f in=%.2f (branch=%.2f MIfill=%.2f) | chance=%.3f'%(best[0],best[1][0],best[1][1],best[1][2],best[1][3],1.0/len(POOL)), flush=True)
    for s in INSC: print('  MI(S;C) vs gain @in=%.2f: %s'%(s,['%.2f'%v for v in curves[s]]), flush=True)
    print('=== CRIT iff MI(S;C) NON-MONOTONIC in gain w/ peak at branch~1 (EDGE) & MIfill low there. Monotonic/flat = no critical regime. ===', flush=True)
    print('=== CRITICAL_SELF_V2_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def critical_self_v2()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'critical_self_v1': critical_self_v1()",
                  "elif MODE == 'critical_self_v1': critical_self_v1()\nelif MODE == 'critical_self_v2': critical_self_v2()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
