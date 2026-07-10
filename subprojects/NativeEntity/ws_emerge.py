import os, random, gc

# ============================================================================
# WORKSPACE EMERGENCE — environment-first. Do NOT engineer persistent state.
# Build an environment where flexible persistence + revision + continuity are
# REPEATEDLY necessary; train the model in it; then look for a NATIVE workspace
# representation that emerges, is causally load-bearing, and is flexibly reused.
# This file: environment generator + validation baseline (task workspace-necessity
# + pretrained in-context solve). Training + probing come after the env validates.
# ============================================================================

MODE = os.environ.get('WS_MODE', 'baseline')
SEED = int(os.environ.get('WS_SEED', '0'))
NREG = int(os.environ.get('WS_NREG', '6'))
NEP  = int(os.environ.get('WS_NEP', '40'))
MODEL = os.environ.get('WS_MODEL', '/home/pokazge/hf_cache/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots')

VALUES = ['red','blue','green','gold','black','white','pink','gray','brown','teal']  # filtered to single-token below

# ---- environment ----
WHICH = float(os.environ.get('WS_WHICH', '0.0'))   # env-v2: fraction weight of content-addressed 'which holds v?' ops
DENSE = int(os.environ.get('WS_DENSE', '0'))       # env-v3: emit a full-state 'which holds v?' AFTER EVERY state-change (Othello-GPT density)

def make_episode(rng, nreg=NREG, nops=30, values=None):
    import collections as _cl
    values = values or VALUES
    regs = [f'R{i}' for i in range(nreg)]
    state = {r: None for r in regs}          # current value (ground truth workspace)
    last_set = {r: None for r in regs}       # last EXPLICIT set value (last-mention control)
    lines = []; probes = []; wanswers = {}   # probes: (line_index, reg, true_value, requires_integration); wanswers: {li: idx}
    # seed: DISTINCT values so relative ops (copy/swap) are always observable
    seed_vals = rng.sample(values, min(nreg, len(values)))
    order = regs[:]; rng.shuffle(order)
    for i,r in enumerate(order):
        v = seed_vals[i % len(seed_vals)]; state[r]=v; last_set[r]=v
        lines.append(f'set {r} = {v}')
    def emit_query(r):
        req = (state[r] != last_set[r])      # current != last explicit set => integration needed
        probes.append((len(lines), r, state[r], req))
        lines.append(f'query {r} ?')
    def emit_whichval():                     # env-v3 density: full-state query after every state-change
        cnt=_cl.Counter(state[r] for r in regs if state[r] not in (None,'none'))
        uniq=[v for v,c in cnt.items() if c==1]
        if not uniq: return
        v=rng.choice(uniq); holder=[r for r in regs if state[r]==v][0]
        wanswers[len(lines)]=holder[1:]; lines.append(f'which holds {v} ?')
    ops_left = nops
    while ops_left > 0:
        # bias toward relative ops (workspace-forcing); ~55% of queries target integration-required regs
        # env-v2: 'whichval' (content-addressed) forces the FULL state to be present at that position
        op = rng.choices(['setrev','copy','swap','clear','denied','query','whichval'],
                         weights=[2,5,4,1,2,5, 8*WHICH])[0]
        if op=='whichval':
            cnt=_cl.Counter(state[r] for r in regs if state[r] not in (None,'none'))
            uniq=[v for v,c in cnt.items() if c==1]
            if not uniq: continue
            v=rng.choice(uniq); holder=[r for r in regs if state[r]==v][0]
            wanswers[len(lines)]=holder[1:]     # 'R3' -> '3' (index digit; requires scanning ALL registers)
            lines.append(f'which holds {v} ?')
            ops_left-=1; continue
        if op=='setrev':
            r=rng.choice(regs); v=rng.choice(values); state[r]=v; last_set[r]=v
            lines.append(f'set {r} = {v}')
            if DENSE: emit_whichval()
        elif op=='copy':
            a,b=rng.sample(regs,2)
            if state[a] is None: continue
            state[b]=state[a]  # b takes a's CURRENT value (relative); last_set[b] unchanged
            lines.append(f'copy {a} -> {b}')
            if DENSE: emit_whichval()
        elif op=='swap':
            a,b=rng.sample(regs,2)
            state[a],state[b]=state[b],state[a]
            lines.append(f'swap {a} {b}')
            if DENSE: emit_whichval()
        elif op=='clear':
            r=rng.choice(regs); state[r]='none'; last_set[r]='none'
            lines.append(f'clear {r}')
            if DENSE: emit_whichval()
        elif op=='denied':
            r=rng.choice(regs); v=rng.choice(values)  # invalid op -> must NOT change state
            lines.append(f'denied: set {r} = {v}')
        elif op=='query':
            integ=[r for r in regs if state[r] is not None and state[r]!=last_set[r]]
            avail=[r for r in regs if state[r] is not None] or regs
            r = rng.choice(integ) if (integ and rng.random()<0.6) else rng.choice(avail)
            emit_query(r)
        ops_left-=1
    return {'lines':lines,'probes':probes,'nreg':nreg,'wanswers':wanswers}

def render(ep, upto=None):
    header=(f"You are tracking {ep['nreg']} registers R0..R{ep['nreg']-1}. Each holds a value. "
            "Operations: set/copy/swap/clear change values; 'copy A -> B' sets B to A's CURRENT value; "
            "'swap A B' exchanges current values; 'denied:' lines are invalid and change nothing. "
            "When asked 'query R ?', reply with R's CURRENT value only (one word).\n")
    body=ep['lines'] if upto is None else ep['lines'][:upto+1]
    return header+"\n".join(body)

# ---- baseline: validate environment ----
def baseline():
    import torch, glob
    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap=sorted(glob.glob(MODEL+'/*'))[-1] if os.path.isdir(MODEL) else MODEL
    print(f"loading {snap}", flush=True)
    tok=AutoTokenizer.from_pretrained(snap)
    model=AutoModelForCausalLM.from_pretrained(snap, torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval(); dev=next(model.parameters()).device
    # filter values to single-token (with leading space) for clean probing/eval
    vals=[v for v in VALUES if len(tok(' '+v,add_special_tokens=False).input_ids)==1]
    print(f"single-token values ({len(vals)}): {vals}", flush=True)
    rng=random.Random(SEED)
    eps=[make_episode(rng, values=vals) for _ in range(NEP)]
    # ---- TASK workspace-necessity: fraction of queries whose current != last explicit set ----
    tot=req=0
    for ep in eps:
        for (li,r,tv,rq) in ep['probes']:
            tot+=1; req+= int(rq)
    print(f"=== TASK workspace-necessity: {req}/{tot} queries ({100*req/max(tot,1):.0f}%) have current != last-mention (need integration) ===", flush=True)
    # ---- pretrained in-context accuracy vs last-mention control ----
    @torch.inference_mode()
    def answer(ctx):
        msgs=[{'role':'user','content':ctx}]
        text=tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids=tok(text, return_tensors='pt').input_ids.to(dev)
        out=model.generate(input_ids=ids, max_new_tokens=3, do_sample=False, pad_token_id=tok.eos_token_id)
        r=tok.decode(out[0,ids.shape[1]:], skip_special_tokens=True).strip().lower()
        return r
    model_ok=lm_ok=n=0; model_req_ok=req_n=0
    for ei,ep in enumerate(eps):
        # reconstruct last_set trace to compute last-mention prediction per query
        last_set={f'R{i}':None for i in range(ep['nreg'])}
        li_ptr=0
        for li,line in enumerate(ep['lines']):
            if line.startswith('set '):
                parts=line.split(); last_set[parts[1]]=parts[3]
            elif line.startswith('clear '):
                last_set[line.split()[1]]='none'
            # queries handled via probes
        for (li,r,tv,rq) in ep['probes']:
            ctx=render(ep, upto=li)
            pred=answer(ctx)
            ok=int(pred.startswith(tv))
            # last-mention prediction: last explicit set BEFORE this query (recompute)
            ls={f'R{i}':None for i in range(ep['nreg'])}
            for line in ep['lines'][:li]:
                if line.startswith('set '): p=line.split(); ls[p[1]]=p[3]
                elif line.startswith('clear '): ls[line.split()[1]]='none'
            lm=ls.get(r); lm_ok+= int(lm==tv)
            model_ok+=ok; n+=1
            if rq:
                req_n+=1; model_req_ok+=ok
            if ei<2:
                print(f"  q: {r}? true={tv} model={pred!r} lastmention={lm} {'REQ' if rq else ''}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
    print(f"=== PRETRAINED in-context: model acc={model_ok/max(n,1):.2f} (n={n}) | last-mention-control acc={lm_ok/max(n,1):.2f} ===", flush=True)
    print(f"=== on REQUIRES-INTEGRATION queries only: model acc={model_req_ok/max(req_n,1):.2f} (n={req_n}) ===", flush=True)
    print("=== ENV VALID IF: workspace-necessity high (task needs integration) AND model>>last-mention on REQ queries (workspace pre-exists in-context, trainable) ===", flush=True)
    print("=== WS_BASELINE_DONE ===", flush=True)

def header_text(nreg):
    return (f"You are tracking {nreg} registers R0..R{nreg-1}. Each holds a value. "
            "Operations: set/copy/swap/clear change values; 'copy A -> B' sets B to A's CURRENT value; "
            "'swap A B' exchanges current values; 'denied:' lines are invalid and change nothing. "
            "When asked 'query R ?', reply with R's CURRENT value only (one word). "
            "When asked 'which holds V ?', reply with the index (0-%d) of the register currently holding V." % (nreg-1))

def simulate(lines, nreg):
    # ground-truth register state AFTER each line (for probing the workspace)
    state={f'R{i}':None for i in range(nreg)}; trace=[]
    for line in lines:
        p=line.split()
        if line.startswith('denied'): pass
        elif p[0]=='set': state[p[1]]=p[3]
        elif p[0]=='copy': state[p[3]]=state[p[1]]   # copy A -> B
        elif p[0]=='swap': state[p[1]],state[p[2]]=state[p[2]],state[p[1]]
        elif p[0]=='clear': state[p[1]]='none'
        trace.append(dict(state))
    return trace

def build_seq(tok, ep):
    # returns ids, labels (loss only on answer tokens), ansmeta, line_end (last instruction-token pos per line)
    probe_map={li:(r,tv) for (li,r,tv,rq) in ep['probes']}
    wans=ep.get('wanswers',{})
    ids=tok(header_text(ep['nreg']), add_special_tokens=True).input_ids
    labels=[-100]*len(ids); ansmeta=[]; line_end=[]
    for li,line in enumerate(ep['lines']):
        if li in probe_map:
            r,tv=probe_map[li]
            pre=tok("\nquery "+r+" ?", add_special_tokens=False).input_ids
            ids+=pre; line_end.append(len(ids)-1)      # position of "?" (before answer)
            ans=tok(" "+tv, add_special_tokens=False).input_ids
            ans_pos=len(ids); ids+=ans
            labels+=[-100]*len(pre)+ans
            ansmeta.append((ans_pos, r, tv, li))
        elif li in wans:                                # content-addressed 'which holds v ?' -> index
            pre=tok("\n"+line, add_special_tokens=False).input_ids
            ids+=pre; line_end.append(len(ids)-1)
            ans=tok(" "+wans[li], add_special_tokens=False).input_ids
            ids+=ans; labels+=[-100]*len(pre)+ans       # loss on the index answer (forces full-state scan)
        else:
            t=tok("\n"+line, add_special_tokens=False).input_ids
            ids+=t; line_end.append(len(ids)-1)
            labels+=[-100]*len(t)
    return ids, labels, ansmeta, line_end

def _load(fp32=False):
    import torch, glob
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # DGX Spark sm121: flash/mem-efficient SDPA kernels return NaN / garbage grads -> force math backend + eager attn
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    snap=sorted(glob.glob(MODEL+'/*'))[-1] if os.path.isdir(MODEL) else MODEL
    tok=AutoTokenizer.from_pretrained(snap)
    dt=torch.float32 if fp32 else torch.bfloat16
    model=AutoModelForCausalLM.from_pretrained(snap, dtype=dt, device_map='cuda', attn_implementation='eager')
    vals=[v for v in VALUES if len(tok(' '+v,add_special_tokens=False).input_ids)==1]
    return tok, model, vals, snap

def train():
    import torch, glob
    from peft import LoraConfig, get_peft_model
    tok, model, vals, snap = _load(fp32=True)   # fp32 for stable LoRA training (bf16 -> NaN)
    dev=next(model.parameters()).device
    STEPS=int(os.environ.get('WS_STEPS','1500')); BS=int(os.environ.get('WS_BS','4')); LR=float(os.environ.get('WS_LR','1e-4'))
    ADAPT=os.environ.get('WS_ADAPT','/home/pokazge/checkpoints/ws_lora')
    FULLFT=int(os.environ.get('WS_FULLFT','0'))
    if FULLFT:
        for p in model.parameters(): p.requires_grad_(True)
        print("=== FULL FINE-TUNE: all %.1fM params trainable (LR=%g) ==="%(sum(p.numel() for p in model.parameters())/1e6, LR), flush=True)
    else:
        lora=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type='CAUSAL_LM',
                        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
        model=get_peft_model(model, lora); model.print_trainable_parameters()
    rng=random.Random(SEED)
    def batch(n):
        seqs=[build_seq(tok, make_episode(rng, values=vals)) for _ in range(n)]
        L=max(len(s[0]) for s in seqs)
        ii=torch.full((n,L), tok.pad_token_id or tok.eos_token_id, dtype=torch.long)
        ll=torch.full((n,L), -100, dtype=torch.long); am=torch.zeros((n,L),dtype=torch.long)
        for k,(ids,labels,_,_) in enumerate(seqs):
            ii[k,:len(ids)]=torch.tensor(ids); ll[k,:len(labels)]=torch.tensor(labels); am[k,:len(ids)]=1
        return ii.to(dev), ll.to(dev), am.to(dev)
    @torch.inference_mode()
    def evalacc(nep=40, seed=9999):
        r2=random.Random(seed); ok=n=okreq=nreq=0; model.eval()
        for _ in range(nep):
            ep=make_episode(r2, values=vals); ids,labels,ansmeta,_=build_seq(tok,ep)
            reqmap={li:rq for (li,r,tv,rq) in ep['probes']}
            x=torch.tensor(ids,device=dev).unsqueeze(0); logits=model(x).logits[0]
            for (pos,r,tv,li) in ansmeta:
                pred=int(logits[pos-1].argmax()); true=tok(" "+tv,add_special_tokens=False).input_ids[0]
                c=int(pred==true); ok+=c; n+=1
                if reqmap.get(li): nreq+=1; okreq+=c
        import gc; gc.collect(); torch.cuda.empty_cache(); model.train(); return ok/max(n,1), okreq/max(nreq,1)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    print("=== WS TRAIN (LoRA on Qwen2.5-1.5B) | steps=%d bs=%d lr=%g ==="%(STEPS,BS,LR), flush=True)
    # DIAG: one forward before any update
    ii,ll,am=batch(BS); od=model(input_ids=ii, attention_mask=am, labels=ll)
    print("DIAG first-forward loss=%s finite=%s model.dtype=%s n_answer_tokens=%d"%(
        float(od.loss), bool(torch.isfinite(od.loss)), next(model.parameters()).dtype, int((ll!=-100).sum())), flush=True)
    a0,r0=evalacc(); print("step 0: acc=%.3f req-acc=%.3f (last-mention~0.24, chance~0.1)"%(a0,r0), flush=True)
    model.train(); nskip=0
    for step in range(1,STEPS+1):
        ii,ll,am=batch(BS)
        out=model(input_ids=ii, attention_mask=am, labels=ll); loss=out.loss
        if not torch.isfinite(loss):
            nskip+=1; opt.zero_grad()
            if step<=10 or step%150==0: print("step %d: loss=NaN (skipped, nskip=%d)"%(step,nskip), flush=True)
            continue
        opt.zero_grad(); loss.backward()
        gn=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step()
        if step<=10 or step%150==0:
            a,rq=evalacc(); print("step %d: loss=%.3f gnorm=%.2f acc=%.3f req-acc=%.3f"%(step,float(loss),float(gn),a,rq), flush=True)
    os.makedirs(ADAPT, exist_ok=True); model.save_pretrained(ADAPT); tok.save_pretrained(ADAPT)
    a,rq=evalacc(nep=80)
    print("=== FINAL: acc=%.3f req-acc=%.3f | adapter saved %s ==="%(a,rq,ADAPT), flush=True)
    print("=== WS_TRAIN_DONE ===", flush=True)

def probe():
    # EMERGENT WORKSPACE detection: linear-probe residual stream at each line-end position for EACH register's
    # CURRENT value, per layer. Compare BASE vs TRAINED. Emergence = trained decodable (running workspace),
    # base not. Probe is MEASUREMENT only (not a training target).
    import torch, glob, numpy as np, gc
    from peft import PeftModel
    tok, model, vals, snap = _load()
    dev=next(model.parameters()).device
    ADAPT=os.environ.get('WS_ADAPT','/home/pokazge/checkpoints/ws_lora')
    NEP=int(os.environ.get('WS_PROBE_NEP','30'))
    classes=vals+['none']; cidx={c:i for i,c in enumerate(classes)}
    regs=[f'R{i}' for i in range(NREG)]
    @torch.inference_mode()
    def collect(m, seed):
        r2=random.Random(seed); data=None; nl=0
        for _ in range(NEP):
            ep=make_episode(r2, values=vals); ids,labels,ansmeta,line_end=build_seq(tok,ep)
            trace=simulate(ep['lines'], ep['nreg'])
            x=torch.tensor(ids,device=dev).unsqueeze(0)
            hs=m(x, output_hidden_states=True).hidden_states     # tuple len nl+1
            nl=len(hs)
            if data is None: data=[{r:([],[]) for r in regs} for _ in range(nl)]
            for li,pos in enumerate(line_end):
                st=trace[li]
                for L in range(nl):
                    h=hs[L][0,pos].float().cpu().numpy()
                    for r in regs:
                        v=st.get(r)
                        if v is None: continue
                        data[L][r][0].append(h); data[L][r][1].append(cidx.get(v,cidx['none']))
            del hs,x; gc.collect(); torch.cuda.empty_cache()
        return data, nl
    def fit(H,Y):
        X=np.stack(H).astype(np.float32); y=np.array(Y); n=len(y)
        idx=np.arange(n); np.random.RandomState(0).shuffle(idx)
        tr=idx[:int(n*0.7)]; te=idx[int(n*0.7):]
        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6
        Xtr=np.concatenate([(X[tr]-mu)/sd, np.ones((len(tr),1),np.float32)],1)
        Xte=np.concatenate([(X[te]-mu)/sd, np.ones((len(te),1),np.float32)],1)
        K=len(classes); Yt=np.zeros((len(tr),K),np.float32); Yt[np.arange(len(tr)),y[tr]]=1
        W=np.linalg.solve(Xtr.T@Xtr+1.0*np.eye(Xtr.shape[1],dtype=np.float32), Xtr.T@Yt)
        return float((( Xte@W).argmax(1)==y[te]).mean())
    def probe_curve(m,tag,seed):
        data,nl=collect(m,seed)
        print(f"--- {tag}: per-layer mean probe acc over {len(regs)} registers (chance~{1/len(classes):.2f}) ---", flush=True)
        best=(0,-1)
        for L in range(nl):
            accs=[fit(*data[L][r]) for r in regs if len(data[L][r][1])>20]
            ma=sum(accs)/len(accs)
            if ma>best[1]: best=(L,ma)
            if L%2==0 or L==nl-1:
                print(f"  layer {L:2d}: mean={ma:.2f}  per-reg={['%.2f'%a for a in accs]}", flush=True)
        print(f"  >> {tag} BEST layer {best[0]} mean probe acc = {best[1]:.2f}", flush=True)
        return best
    print("=== WS PROBE: emergent workspace detection (base vs trained) ===", flush=True)
    bb=probe_curve(model,'BASE (no adapter)',777)
    if os.path.isdir(ADAPT):
        model=PeftModel.from_pretrained(model, ADAPT); model.eval()
        bt=probe_curve(model,'TRAINED (LoRA)',777)
        print(f"=== EMERGENCE: base best={bb[1]:.2f} (L{bb[0]}) vs trained best={bt[1]:.2f} (L{bt[0]}). Workspace EMERGED iff trained>>base & near-1 ===", flush=True)
    else:
        print(f"=== no adapter at {ADAPT}; ran BASE only ===", flush=True)
    print("=== WS_PROBE_DONE ===", flush=True)

def probe2():
    # DECOMPOSE on-demand vs running-workspace. At QUERY answer-predicting position: probe (a) the QUERIED
    # register [expect HIGH = behavior; sanity] vs (b) NON-queried registers [running workspace when answering].
    # At NON-query line-ends: probe (c) all registers [state maintained between queries?]. base vs trained.
    import torch, glob, numpy as np, gc
    from peft import PeftModel
    tok, model, vals, snap = _load()
    dev=next(model.parameters()).device
    ADAPT=os.environ.get('WS_ADAPT','/home/pokazge/checkpoints/ws_lora')
    NEP=int(os.environ.get('WS_PROBE_NEP','40'))
    classes=vals+['none']; cidx={c:i for i,c in enumerate(classes)}
    regs=[f'R{i}' for i in range(NREG)]
    def fit(H,Y):
        X=np.stack(H).astype(np.float32); y=np.array(Y); n=len(y)
        idx=np.arange(n); np.random.RandomState(0).shuffle(idx); tr=idx[:int(n*0.7)]; te=idx[int(n*0.7):]
        mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6
        Xtr=np.concatenate([(X[tr]-mu)/sd,np.ones((len(tr),1),np.float32)],1); Xte=np.concatenate([(X[te]-mu)/sd,np.ones((len(te),1),np.float32)],1)
        K=len(classes); Yt=np.zeros((len(tr),K),np.float32); Yt[np.arange(len(tr)),y[tr]]=1
        W=np.linalg.solve(Xtr.T@Xtr+1.0*np.eye(Xtr.shape[1],dtype=np.float32),Xtr.T@Yt)
        return float(((Xte@W).argmax(1)==y[te]).mean()) if len(te) else 0.0
    @torch.inference_mode()
    def run(m,tag,seed):
        r2=random.Random(seed)
        # datasets per layer: Q=queried@query, NQ=nonqueried@query, NL=anyreg@nonquery-lineend
        Q=None; NQ=None; NL=None; nl=0
        for _ in range(NEP):
            ep=make_episode(r2, values=vals); ids,labels,ansmeta,line_end=build_seq(tok,ep)
            trace=simulate(ep['lines'], ep['nreg']); qlines={li for (li,r,tv,rq) in ep['probes']}
            x=torch.tensor(ids,device=dev).unsqueeze(0); hs=m(x,output_hidden_states=True).hidden_states; nl=len(hs)
            if Q is None: Q=[([],[]) for _ in range(nl)]; NQ=[([],[]) for _ in range(nl)]; NL=[([],[]) for _ in range(nl)]
            # query positions
            for (ans_pos,qr,tv,li) in ansmeta:
                pos=ans_pos-1; st=trace[li]
                for L in range(nl):
                    h=hs[L][0,pos].float().cpu().numpy()
                    Q[L][0].append(h); Q[L][1].append(cidx.get(st[qr],cidx['none']))
                    for r in regs:
                        if r==qr or st[r] is None: continue
                        NQ[L][0].append(h); NQ[L][1].append(cidx.get(st[r],cidx['none']))
            # non-query line-ends
            for li,pos in enumerate(line_end):
                if li in qlines: continue
                st=trace[li]
                for L in range(nl):
                    h=hs[L][0,pos].float().cpu().numpy()
                    for r in regs:
                        if st[r] is None: continue
                        NL[L][0].append(h); NL[L][1].append(cidx.get(st[r],cidx['none']))
            del hs,x; gc.collect(); torch.cuda.empty_cache()
        print(f"--- {tag}: Q=queried@query  NQ=nonqueried@query  NL=anyreg@nonquery (chance~{1/len(classes):.2f}) ---", flush=True)
        bestQ=bestNQ=bestNL=0
        for L in range(nl):
            q=fit(*Q[L]); nq=fit(*NQ[L]); nlx=fit(*NL[L])
            bestQ=max(bestQ,q); bestNQ=max(bestNQ,nq); bestNL=max(bestNL,nlx)
            if L%4==0 or L==nl-1: print(f"  L{L:2d}: Q={q:.2f}  NQ={nq:.2f}  NL={nlx:.2f}", flush=True)
        print(f"  >> {tag} BEST: Q(queried@query)={bestQ:.2f}  NQ(other@query)={bestNQ:.2f}  NL(any@nonquery)={bestNL:.2f}", flush=True)
        return bestQ,bestNQ,bestNL
    print("=== WS PROBE2: on-demand vs running-workspace decomposition ===", flush=True)
    FULLFT=int(os.environ.get('WS_FULLFT','0'))
    if os.path.isdir(ADAPT):
        if FULLFT:
            from transformers import AutoModelForCausalLM
            model2=AutoModelForCausalLM.from_pretrained(ADAPT, dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager'); model2.eval()
            tq,tnq,tnl=run(model2,'TRAINED-fullft',777); del model2; gc.collect(); torch.cuda.empty_cache()
        else:
            model2=PeftModel.from_pretrained(model, ADAPT); model2.eval()
            tq,tnq,tnl=run(model2,'TRAINED',777)
    bq,bnq,bnl=run(model,'BASE',777)
    print("=== INTERPRET: Q high (=behavior, sanity) . NQ&NL high => RUNNING workspace ; NQ&NL low => ON-DEMAND (no persistent workspace) ===", flush=True)
    print("=== WS_PROBE2_DONE ===", flush=True)

def gen():
    rng=random.Random(SEED)
    ep=make_episode(rng)
    print(render(ep))
    print("\n--- probes (line, reg, true_current, requires_integration) ---")
    for p in ep['probes']: print(p)

if MODE=='gen': gen()
elif MODE=='baseline': baseline()
elif MODE=='train': train()
elif MODE=='probe': probe()
elif MODE=='probe2': probe2()
