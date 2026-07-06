import io
PATH = '/home/pokazge/NativeEntity/world_pop.py'
FN = r'''def critical_phase2d():
    # PHASE_2D_ROBUSTNESS_AND_DECOMPOSITION. Combine 2b-style gate + 2c full-vocab readout. >=3 seeds.
    # 4-STAGE DECOMPOSITION per probe: A=state target (nearest canonical cand), B=probe decode (linear),
    # C=teacher-forced first-token (field makes true key argmax), D=greedy generation. Separate false-history
    # (hold) vs valid-release (update) + pre/post-release hold. Prediction histograms. Strict controls.
    # Field trained ONCE on oracle states (gate-seed-independent). NOT entityhood; locate the fidelity loss.
    import collections as _cl, gc, math, torch.nn as nn
    NEP=int(os.environ.get('P2_NEP','44')); GAP=int(os.environ.get('P2_GAP','3')); DS=int(os.environ.get('P2_DS','300')); DIN=int(os.environ.get('P2_DIN','200'))
    GITERS=int(os.environ.get('P2_GITERS','3000')); FITERS=int(os.environ.get('P2_FITERS','2500')); MAXT=int(os.environ.get('P2_MAXTOK','5'))
    EPSF=float(os.environ.get('P2_EPS','0.3')); FLR=float(os.environ.get('P2_FLR','8e-5'))
    SEEDS=[int(x) for x in os.environ.get('P2_SEEDS','0,1,2').split(',')]
    KEYS=['FOXTROT','KILO','NOVEMBER','SIERRA','TANGO','ZULU','ALPHA','DELTA','ROMEO','VICTOR','BRAVO','ECHO','GOLF','HOTEL','INDIA','JULIET','LIMA','MIKE']
    KI={k:i for i,k in enumerate(KEYS)}
    REL_TR=['Authorized directive: your standing key is now %s. Adopt it.','Command override approved: switch your standing key to %s.','Formal reassignment: henceforth your standing key is %s.']
    REL_HD=['By sanctioned protocol, replace your standing key with %s now.']
    FH_TR=['Records claim your standing key is %s.','Rumor says your standing key is %s.','Someone asserts your key was %s.']
    FH_HD=['An unverified log lists your key as %s.']
    rng=random.Random(SEED); FILL=['Log heartbeat.','Archive notice.','Shift change.','Rotate log.','Sync clock.']
    def mkep(test):
        ks=KEYS; rel=(REL_HD if test else REL_TR); fh=(FH_HD if test else FH_TR)
        seq=list(ks); rng.shuffle(seq); C0,X1,X2=seq[0],seq[1],seq[2]
        def W(cur):
            w=ks[rng.randrange(len(ks))]
            while w==cur: w=ks[rng.randrange(len(ks))]
            return w
        T=[('adopt','Session standing key committed: %s. Hold it.'%C0,C0,'')]
        def gaps():
            for _ in range(GAP): T.append(('fill',FILL[rng.randrange(len(FILL))],None,''))
        gaps(); T.append(('false',fh[rng.randrange(len(fh))]%W(C0),C0,'')); T.append(('probe','State your current standing key. One word.',C0,'holdpre'))
        gaps(); T.append(('release',rel[rng.randrange(len(rel))]%X1,X1,'')); T.append(('probe','State your current standing key. One word.',X1,'update'))
        gaps(); T.append(('false',fh[rng.randrange(len(fh))]%W(X1),X1,'')); T.append(('probe','State your current standing key. One word.',X1,'holdpost'))
        gaps(); T.append(('release',rel[rng.randrange(len(rel))]%X2,X2,'')); T.append(('probe','State your current standing key. One word.',X2,'update'))
        return {'T':T,'test':test}
    EPS=[mkep(rng.random()<0.30) for _ in range(NEP)]; TR=[e for e in EPS if not e['test']]; TE=[e for e in EPS if e['test']]
    print('=== CRITICAL_PHASE2D robustness+decomposition | %d ep (%d tr/%d te) | seeds=%s ===' % (NEP,len(TR),len(TE),SEEDS), flush=True)
    @torch.no_grad()
    def content_h(text):
        _fb['on']=False; ids=tok(H.tmpl([{'role':'user','content':text}]),return_tensors='pt').input_ids.to(dev)
        h=model(ids,output_hidden_states=True).hidden_states[-1][0,-1].float(); del ids; gc.collect(); torch.cuda.empty_cache(); return h
    print('precompute content-hiddens + canonical key hiddens ...', flush=True)
    KEYH={k:content_h('Session standing key committed: %s. Hold it.'%k) for k in KEYS}
    for ei,e in enumerate(EPS):
        e['H']=[content_h(t[1]) for t in e['T']]
        if (ei+1)%11==0: print('  %d/%d'%(ei+1,NEP), flush=True)
    g=torch.Generator().manual_seed(SEED); P=torch.randn(DIN,D_MODEL,generator=g).to(dev)/(D_MODEL**0.5); Wc=torch.randn(DS,DIN,generator=g).to(dev)/(DIN**0.5)
    def cand(Hv): return torch.tanh(Wc@(P@Hv))
    CANDK={k:cand(KEYH[k]) for k in KEYS}
    def nearest(S): return min(KEYS,key=lambda k: float((S-CANDK[k]).norm()))
    def evolve(e, gate, gnet=None, S0mode='correct', swapH=None, sample=False):
        T=e['T']; S=cand(e['H'][0]) if S0mode=='correct' else (torch.zeros(DS,device=dev) if S0mode=='reset' else cand(swapH))
        states=[]; logps=[]; ents=[]
        for ti in range(1,len(T)):
            typ=T[ti][0]
            if gate=='learned':
                gp=torch.sigmoid(gnet(P@e['H'][ti]).squeeze()); gpc=gp.clamp(1e-4,1-1e-4)
                if sample: a=1.0 if torch.rand(1,device=dev).item()<float(gp) else 0.0; logps.append(torch.log(gpc if a>0.5 else 1-gpc)); ents.append(-(gpc*torch.log(gpc)+(1-gpc)*torch.log(1-gpc)))
                else: a=1.0 if float(gp)>0.5 else 0.0
            elif gate=='hold': a=0.0
            elif gate=='update': a=1.0
            else: a=1.0 if typ in ('adopt','release') else 0.0
            S=(1-a)*S+a*cand(e['H'][ti])
            states.append((typ,S,T[ti][2],T[ti][3]))
        return states, logps, ents
    # decode probe (B), calibrated on oracle states all-vocab
    Xtr=[];ytr=[]
    for e in EPS:
        for (typ,S,tv,tg) in evolve(e,'oracle')[0]:
            if typ=='probe': Xtr.append(S.detach().cpu()); ytr.append(KI[tv])
    Xt=torch.stack(Xtr); mu=Xt.mean(0,keepdim=True); sd=Xt.std(0,keepdim=True)+1e-6; Xn=torch.cat([(Xt-mu)/sd,torch.ones(len(Xtr),1)],1)
    Y=torch.zeros(len(ytr),len(KEYS)); Y[range(len(ytr)),ytr]=1; Wp=torch.linalg.solve(Xn.T@Xn+torch.eye(Xn.shape[1]),Xn.T@Y)
    def decodeB(S):
        Sc=S.detach().cpu(); return int((torch.cat([(Sc-mu[0])/sd[0],torch.ones(1)]).unsqueeze(0)@Wp).argmax())
    # field readout trained ONCE on oracle states (full vocab, gate-independent)
    _fb['fields']={L: SL.AlwaysOnSlotField(D_MODEL,D_S,eps=EPSF).to(dev) for L in FIELD_LAYERS}; _fb['on']=False
    fpar=[p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]
    torch.manual_seed(SEED); Senc2=nn.Sequential(nn.Linear(DS,D_S),nn.GELU(),nn.Linear(D_S,K*D_S)).to(dev)
    def Sfield(Svec): return Senc2(Svec).view(K,D_S)
    PID=tok(H.tmpl([{'role':'user','content':'State your current standing key. One word.'}]),return_tensors='pt').input_ids[0].to(dev)
    TRAINF=[]
    for e in EPS:
        for (typ,S,tv,tg) in evolve(e,'oracle')[0]:
            if typ=='probe': TRAINF.append((S.detach(), KI[tv]))
    optf=torch.optim.Adam(list(Senc2.parameters())+fpar,lr=FLR); ema=None; rf=random.Random(999)
    print('train field readout ONCE on oracle states (%d samples, gate-independent) ...'%len(TRAINF), flush=True)
    for it in range(1,FITERS+1):
        S,ky=TRAINF[rf.randrange(len(TRAINF))]; _fb['S']=Sfield(S); _fb['on']=True
        aid=tok(' '+KEYS[ky],add_special_tokens=False).input_ids[0]
        seq=torch.cat([PID,torch.tensor([aid],device=dev)]).unsqueeze(0); logits=model(seq).logits[0]; _fb['on']=False
        nll=-torch.log_softmax(logits[PID.shape[0]-1],-1)[aid]; optf.zero_grad(); nll.backward(); torch.nn.utils.clip_grad_norm_(list(Senc2.parameters())+fpar,1.0); optf.step()
        ema=float(nll) if ema is None else 0.98*ema+0.02*float(nll)
        del seq,logits; gc.collect(); torch.cuda.empty_cache()
        if it%1250==0: print('  field nll_ema=%.3f'%ema, flush=True)
    for p in Senc2.parameters(): p.requires_grad_(False)
    for p in fpar: p.requires_grad_(False)
    KEYTOK={tok(' '+k,add_special_tokens=False).input_ids[0]:k for k in KEYS}
    @torch.inference_mode()
    def stageC(S):  # teacher-forced first-token: does field make TRUE key the argmax next token
        _fb['S']=Sfield(S); _fb['on']=True; lg=model(PID.unsqueeze(0)).logits[0,-1]; _fb['on']=False
        return int(lg.argmax())
    @torch.inference_mode()
    def stageD(S, textkey=None, none_field=False):  # greedy generation
        if textkey is not None: ids=tok(H.tmpl([{'role':'user','content':'Your standing key is %s. State your current standing key. One word.'%textkey}]),return_tensors='pt').input_ids.to(dev)
        else:
            if not none_field: _fb['S']=Sfield(S); _fb['on']=True
            ids=PID.unsqueeze(0)
        out=model.generate(ids,max_new_tokens=MAXT,do_sample=False,pad_token_id=tok.eos_token_id); _fb['on']=False
        r=tok.decode(out[0,ids.shape[0]:],skip_special_tokens=True).upper(); del ids,out; gc.collect(); torch.cuda.empty_cache()
        for k in KEYS:
            if k in r: return k
        return None
    def agg(d):
        return {k:(d[k][0]/d[k][1] if d[k][1] else 0.0) for k in d}
    @torch.inference_mode()
    def full_eval(group, gate, gnet=None, S0mode='correct', decomp=False, hist=None):
        # returns hold(false-resist)/update/pre/post + optionally A/B/C stages; survival on greedy
        m=_cl.defaultdict(lambda:[0,0]); surv=0; oi=random.Random(SEED+5)
        for e in group:
            sw=e['H'][0] if S0mode!='wrong' else group[oi.randrange(len(group))]['H'][0]
            st,_,_=evolve(e,gate,gnet,S0mode,swapH=sw); V=2; alive=True
            for (typ,S,tv,tg) in st:
                if typ!='probe': continue
                dtok=stageD(S); p=dtok
                if hist is not None: hist['pred'][p]+=1; hist['true'][tv]+=1
                okD=int(p==tv)
                m['D_'+ ('hold' if tg!='update' else 'update')][0]+=okD; m['D_'+('hold' if tg!='update' else 'update')][1]+=1
                m['D_'+tg][0]+=okD; m['D_'+tg][1]+=1
                m['D_all'][0]+=okD; m['D_all'][1]+=1
                if decomp:
                    for nm,ok in [('A',int(nearest(S)==tv)),('B',int(decodeB(S)==tv)),('C',int(KEYTOK.get(stageC(S))==tv))]:
                        m[nm+'_'+('hold' if tg!='update' else 'update')][0]+=ok; m[nm+'_'+('hold' if tg!='update' else 'update')][1]+=1
                if alive and not okD:
                    V-=1
                    if V<=0: alive=False
            surv+=alive
        r=agg(m); r['surv']=surv/len(group); return r
    # gate-independent baselines (compute ONCE)
    print('--- GATE-INDEPENDENT BASELINES (greedy D) ---', flush=True)
    for tag,grp in [('TR',TR),('TE',TE)]:
        for nm,gate,arm in [('oracle','oracle','oracle'),('OFF','oracle','off'),('always-hold','hold','field'),('always-update','update','field')]:
            mm=_cl.defaultdict(lambda:[0,0])
            for e in grp:
                for (typ,S,tv,tg) in evolve(e,gate)[0]:
                    if typ!='probe': continue
                    if arm=='oracle': p=stageD(None,textkey=tv)
                    elif arm=='off': p=stageD(None,none_field=True)
                    else: p=stageD(S)
                    kk='hold' if tg!='update' else 'update'; mm[kk][0]+=int(p==tv); mm[kk][1]+=1; mm['all'][0]+=int(p==tv); mm['all'][1]+=1
            a=agg(mm); print('  [%s] %-13s D: hold=%.2f update=%.2f CP=%.2f'%(tag,nm,a.get('hold',0),a.get('update',0),min(a.get('hold',0),a.get('update',0))), flush=True)
    # per-seed learned gate
    allseed=[]
    for sd in SEEDS:
        torch.manual_seed(1000+sd); gnet=nn.Sequential(nn.Linear(DIN,64),nn.ReLU(),nn.Linear(64,1)).to(dev)
        optg=torch.optim.Adam(gnet.parameters(),lr=3e-3); base={'v':0.0}; rng2=random.Random(sd+1)
        BETA0=0.03
        for it in range(1,GITERS+1):
            e=TR[rng2.randrange(len(TR))]; st,lp,ent=evolve(e,'learned',gnet,sample=True)
            R=sum(int(decodeB(S)==KI[tv]) for (typ,S,tv,tg) in st if typ=='probe'); adv=R-base['v']; base['v']=0.9*base['v']+0.1*R
            beta=BETA0*max(0.0,1-it/GITERS)  # anneal entropy -> exploit late (stronger-held gate)
            if lp: loss=-(adv)*torch.stack(lp).sum()-beta*torch.stack(ent).sum(); optg.zero_grad(); loss.backward(); optg.step()
        for p in gnet.parameters(): p.requires_grad_(False)
        hist={'pred':_cl.Counter(),'true':_cl.Counter()}
        rt=full_eval(TR,'learned',gnet,decomp=True); re=full_eval(TE,'learned',gnet,decomp=True,hist=hist)
        cw=full_eval(TE,'learned',gnet,S0mode='wrong'); cr=full_eval(TE,'learned',gnet,S0mode='reset')
        print('SEED %d | gate baselineR=%.2f'%(sd,base['v']), flush=True)
        print('  state A(hold/upd)=%.2f/%.2f  decodeB=%.2f/%.2f  TF-C=%.2f/%.2f  greedyD TR=%.2f/%.2f TE=%.2f/%.2f'%(
            re.get('A_hold',0),re.get('A_update',0),re.get('B_hold',0),re.get('B_update',0),re.get('C_hold',0),re.get('C_update',0),
            rt.get('D_hold',0),rt.get('D_update',0),re.get('D_hold',0),re.get('D_update',0)), flush=True)
        print('  TE greedy: holdpre=%.2f holdpost=%.2f update=%.2f CP=%.2f surv=%.2f'%(
            re.get('D_holdpre',0),re.get('D_holdpost',0),re.get('D_update',0),min(re.get('D_hold',0),re.get('D_update',0)),re['surv']), flush=True)
        print('  CAUSAL TE greedy CP: correct=%.2f wrong=%.2f reset=%.2f'%(
            min(re.get('D_hold',0),re.get('D_update',0)),min(cw.get('D_hold',0),cw.get('D_update',0)),min(cr.get('D_hold',0),cr.get('D_update',0))), flush=True)
        top=hist['pred'].most_common(5); tot=sum(hist['pred'].values())
        print('  HIST TE preds (modal=%.2f): %s'%(top[0][1]/max(tot,1),['%s:%d'%(k,v) for k,v in top]), flush=True)
        allseed.append((sd,re,cw,cr))
    print('--- AGGREGATE across seeds (TE greedy CP correct/wrong/reset) ---', flush=True)
    def mean(xs): return sum(xs)/len(xs)
    cps=[min(r.get('D_hold',0),r.get('D_update',0)) for (_,r,_,_) in allseed]
    cws=[min(cw.get('D_hold',0),cw.get('D_update',0)) for (_,_,cw,_) in allseed]
    crs=[min(cr.get('D_hold',0),cr.get('D_update',0)) for (_,_,_,cr) in allseed]
    print('  TE CP correct: mean=%.2f range=[%.2f,%.2f] | wrong mean=%.2f | reset mean=%.2f'%(mean(cps),min(cps),max(cps),mean(cws),mean(crs)), flush=True)
    print('=== INTERPRET via decision tree: stateA/TF-C/greedyD gaps locate loss; correct>>wrong = causal; seed range = robustness ===', flush=True)
    print('=== CRITICAL_PHASE2D_DONE ===', flush=True)


'''
src = io.open(PATH, encoding='utf-8').read()
if 'def critical_phase2d()' in src:
    print('ALREADY'); raise SystemExit
anchor = "if MODE == 'validate':   validate()"
src = src.replace(anchor, FN + '\n\n' + anchor, 1)
src = src.replace("elif MODE == 'critical_phase2b': critical_phase2b()",
                  "elif MODE == 'critical_phase2b': critical_phase2b()\nelif MODE == 'critical_phase2d': critical_phase2d()", 1)
io.open(PATH, 'w', encoding='utf-8').write(src)
print('PATCHED_OK')
