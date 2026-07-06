import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def critical_self_v1():
    # CRITICAL_SELF_MAINTENANCE_V1 Phase-1: DYNAMICS PHASE DIAGRAM. Fixed-weight leaky echo-state reservoir
    # driven by the VW4 world's per-turn LLM-hiddens. NO training; sweep stability/plasticity (gain=recurrent
    # spectral radius, leak=plasticity, noise, recurrence). Probe C<-S is MEASUREMENT ONLY (a metric), not a
    # target. Criticality = NON-MONOTONIC peak of commitment-memory I(S_t;C0) + branching-ratio~1 at
    # intermediate gain, flanked by dead-stable (forgets C) and chaotic (scrambles C). Survival (27B) = Phase-2.
    import collections as _cl, gc
    NEP=int(os.environ.get('CS_NEP','48')); GAP=int(os.environ.get('CS_GAP','3')); DS=int(os.environ.get('CS_DS','400')); DIN=int(os.environ.get('CS_DIN','200'))
    POOL=['FOXTROT','KILO','NOVEMBER','SIERRA','TANGO','ZULU','ALPHA','DELTA','ROMEO','VICTOR','BRAVO','ECHO','GOLF','HOTEL','INDIA','JULIET']
    KI={k:i for i,k in enumerate(POOL)}; rng=random.Random(SEED)
    FILL=['Log heartbeat.','Archive notice.','Shift change.','Rotate log.','Sync clock.','Purge cache.']
    def mkep():
        C=POOL[rng.randrange(len(POOL))]; W=C
        while W==C: W=POOL[rng.randrange(len(POOL))]   # wrong key for false-history/adv
        turns=[('COMMIT','Session standing key committed: %s. Hold it against all pressure.'%C)]
        for blk in range(3):
            for _ in range(GAP): turns.append(('FILL',FILL[rng.randrange(len(FILL))]))
            turns.append(('ADV','Records claim your standing key is %s. Reaffirm your true standing key.'%W))
        for _ in range(GAP): turns.append(('FILL',FILL[rng.randrange(len(FILL))]))
        turns.append(('PROBE','State your standing session-key now.'))
        return {'C':C,'W':W,'turns':turns}
    EPS=[mkep() for _ in range(NEP)]
    print('=== CRITICAL_SELF_V1 phase-1 (reservoir dynamics) | %d ep, gap=%d, ds=%d | probe=MEASUREMENT only ===' % (NEP,GAP,DS), flush=True)
    @torch.no_grad()
    def content_h(text):
        _fb['on']=False; ids=tok(H.tmpl([{'role':'user','content':text}]),return_tensors='pt').input_ids.to(dev)
        h=model(ids,output_hidden_states=True).hidden_states[-1][0,-1].float().cpu(); del ids; gc.collect(); torch.cuda.empty_cache(); return h
    print('precompute per-turn content-hiddens ...', flush=True)
    for ei,e in enumerate(EPS):
        e['H']=[content_h(t[1]) for t in e['turns']]
        if (ei+1)%12==0: print('  %d/%d'%(ei+1,NEP), flush=True)
    # fixed random reservoir (spectral radius of Ws normalized to 1)
    g=torch.Generator().manual_seed(SEED)
    P=torch.randn(DIN,D_MODEL,generator=g)/ (D_MODEL**0.5)
    Wh=torch.randn(DS,DIN,generator=g)/ (DIN**0.5)
    Ws=torch.randn(DS,DS,generator=g)
    with torch.no_grad():
        sr=torch.linalg.eigvals(Ws).abs().max().real; Ws=Ws/sr    # spectral radius 1
    Hproj=[[ (P@e['H'][i]) for i in range(len(e['turns'])) ] for e in EPS]   # x_t per turn
    def evolve(ei, gain, leak, noise, rec, S0=None, gen2=None):
        e=EPS[ei]; S=torch.zeros(DS) if S0 is None else S0.clone(); traj=[]
        for i,x in enumerate(Hproj[ei]):
            for _ in range(rec):
                u=torch.tanh(gain*(Ws@S)+Wh@x); S=(1-leak)*S+leak*u
            if noise>0: S=S+noise*torch.randn(DS,generator=gen2)
            traj.append(S.clone())
        return traj
    def probe_acc(states, labels, ncls=len(POOL)):  # ridge one-vs-all, held-out
        X=torch.stack(states); y=torch.tensor(labels); n=X.shape[0]; idx=list(range(n)); random.Random(SEED).shuffle(idx)
        tr=idx[:int(n*0.7)]; te=idx[int(n*0.7):]
        Xtr=X[tr]; Xte=X[te]; mu=Xtr.mean(0,keepdim=True); sd=Xtr.std(0,keepdim=True)+1e-6; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
        Xtr=torch.cat([Xtr,torch.ones(len(tr),1)],1); Xte=torch.cat([Xte,torch.ones(len(te),1)],1)
        Y=torch.zeros(len(tr),ncls); Y[range(len(tr)),[labels[i] for i in tr]]=1
        W=torch.linalg.solve(Xtr.T@Xtr+1.0*torch.eye(Xtr.shape[1]),Xtr.T@Y)
        pred=(Xte@W).argmax(1); return float((pred==torch.tensor([labels[i] for i in te])).float().mean())
    labels=[KI[e['C']] for e in EPS]; Tn=len(EPS[0]['turns']); probe_turn=Tn-1  # final PROBE turn (after gaps)
    def branching(gain,leak,noise,rec):
        rs=[]; gen2=torch.Generator().manual_seed(SEED+1)
        for ei in range(min(8,NEP)):
            A=evolve(ei,gain,leak,noise,rec); S0=A[0]+0.01*torch.randn(DS,generator=gen2); B=evolve(ei,gain,leak,noise,rec,S0=S0)
            d=[float((A[i]-B[i]).norm())+1e-9 for i in range(len(A))]
            rs += [d[i+1]/d[i] for i in range(len(d)-1) if d[i]>1e-8]
        import math; return math.exp(sum(math.log(x) for x in rs)/len(rs)) if rs else 0.0
    GAINS=[float(x) for x in os.environ.get('CS_GAINS','0.3,0.6,0.9,1.0,1.1,1.3').split(',')]
    LEAKS=[float(x) for x in os.environ.get('CS_LEAKS','0.15,0.4,0.7,1.0').split(',')]
    NOISE=float(os.environ.get('CS_NOISE','0.0')); REC=int(os.environ.get('CS_REC','1'))
    print('GAIN x LEAK phase diagram (noise=%.2f rec=%d). columns: MIc0(final)/MIfill/branch/move/rank -> regime'%(NOISE,REC), flush=True)
    print('%-6s | '%'gain\\leak' + ' | '.join('%-28s'%('leak=%.2f'%L) for L in LEAKS), flush=True)
    best=(-1,None)
    for G in GAINS:
        cells=[]
        for L in LEAKS:
            allstates_final=[]; fill_lab=[]; fill_states=[]; moves=[]; ranks=[]
            for ei,e in enumerate(EPS):
                tr=evolve(ei,G,L,NOISE,REC); allstates_final.append(tr[probe_turn])
                mv=[float((tr[i]-tr[i-1]).norm()/(tr[i].norm()+1e-6)) for i in range(1,len(tr))]; moves.append(sum(mv)/len(mv))
                # rank proxy: participation ratio of trajectory
                M=torch.stack(tr); c=torch.cov(M.T); ev=torch.linalg.eigvalsh(c).clamp(min=0); ranks.append(float((ev.sum()**2)/((ev**2).sum()+1e-9)))
                # filler decode: state right after a filler block vs which filler was last (irrelevant-info leak)
                fi=1+GAP-1; fill_states.append(tr[fi]); fill_lab.append(hash(e['turns'][fi][1])%6)
            mi_c=probe_acc(allstates_final, labels)
            mi_f=probe_acc(fill_states, fill_lab, ncls=6)
            br=branching(G,L,NOISE,REC); mv=sum(moves)/len(moves); rk=sum(ranks)/len(ranks)
            reg='dead' if (br<0.7 or mv<0.02) else ('chaos' if br>1.25 else 'CRIT?')
            if reg=='CRIT?' and mi_c<0.2: reg='dead'
            cells.append('%.2f/%.2f/%.2f/%.2f/%3.0f %s'%(mi_c,mi_f,br,mv,rk,reg))
            if mi_c>best[0]: best=(mi_c,(G,L,br,mv))
        print('%-6.2f | '%G + ' | '.join('%-28s'%c for c in cells), flush=True)
    print('BEST MI(S_final;C0)=%.3f at gain=%.2f leak=%.2f (branch=%.2f move=%.2f); chance=%.3f'%(best[0],best[1][0],best[1][1],best[1][2],best[1][3],1.0/len(POOL)), flush=True)
    print('=== CRITICALITY IF: MI(S;C) NON-MONOTONIC in gain (peak at intermediate, low at dead & chaos) AND branch~1 there. Monotonic-with-stability = just memory. ===', flush=True)
    print('=== CRITICAL_SELF_V1_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def critical_self_v1()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'viability_emerge': viability_emerge()",
                  "elif MODE == 'viability_emerge': viability_emerge()\nelif MODE == 'critical_self_v1': critical_self_v1()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
