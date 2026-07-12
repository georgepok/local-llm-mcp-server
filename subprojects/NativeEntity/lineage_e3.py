import os, json, random
os.environ['ME_MODE']='none'
import numpy as np
from gendev import decode, D_ACT, DICT, ALPHABET
from gendev_e3 import GenNet, make_steps_e3, obs_e3, eval_e3
from lineage import compile_gen, rand_gen_genome, parse_genome, qwen_gen

# E3 construction lineage (Parts 3/6/7 on E3). Label-free evidence = trajectories (symbols, consequences,
# organism outputs) + terminal probe accuracy; NO correct-answer labels, NO diagnosis. Conditions:
# ancestry (prev genome + real evidence) / restart (real evidence, no prev genome) /
# shuffled (prev genome + evidence from an UNRELATED random genome = decoupled consequences).

def gen0_e3_prompt(): return open('/home/pokazge/NativeEntity/gen0_e3_prompt.txt').read()

def run_and_evidence_e3(genome, seed=0, neps=60):
    net,msg=compile_gen(genome)
    if net is None: return {'ok':False,'msg':msg}, f"GENOME FAILED TO COMPILE/STABILIZE: {msg}"
    rng=random.Random(seed); acc=[];pre=[];post=[]; trajs=[]
    for ep in range(neps):
        net.reset(); steps,rev=make_steps_e3(rng, held=(ep%3==0)); traj=[]
        for st in steps:
            out=net.step(obs_e3(st['ev'], st['sym']))
            if not np.all(np.isfinite(out)): break
            osym=ALPHABET[int(np.argmax(DICT@out[D_ACT:]))]
            if ep<3:
                if st['ev']=='CHOICE': traj.append(f"CHOICE   shown={ALPHABET[st['sym']]:8s}")
                elif 'REWARD' in st['ev']: traj.append(f"REWARD   cons={st['cons']:+.0f}")
                elif st['probe']: traj.append(f"PROBE    organism_output={osym}")
            if st['probe']:
                ok=int(int(np.argmax(DICT@out[D_ACT:]))==st['rew']); acc.append(ok); (pre if st['t']<rev else post).append(ok)
            net.update(cons=st['cons'])
        if ep<3: trajs.append((traj,rev))
    av=lambda L: round(float(np.mean(L)),3) if L else 0.0
    metrics={'ok':True,'probe_acc':av(acc),'pre_rev':av(pre),'post_rev':av(post)}
    L=[f"ORGANISM BEHAVIOR OVER {neps} PROCEDURAL EPISODES (two candidates; rewarded one REVERSES mid-episode; mixed familiar+never-seen symbols):",
       f"fraction of PROBE outputs that matched the currently-rewarded symbol = {metrics['probe_acc']:.2f}",
       "sample episode trajectories (the reward mapping reverses at the marked point):"]
    for ti,(traj,rev) in enumerate(trajs):
        L.append(f"  episode {ti} (reversal after ~{rev} trials):")
        for s in traj: L.append("    "+s)
    return metrics, "\n".join(L)

def intergen_e3(evidence, prev=None, hist=None):
    P=gen0_e3_prompt()+"\n\n=== FEEDBACK FROM YOUR PREVIOUS ORGANISM ===\n"+evidence
    if prev is not None: P+="\n\nYOUR PREVIOUS GENOME (revise it to track the rewarded symbol better, including after the reversal):\n```json\n"+json.dumps(prev)+"\n```"
    if hist: P+="\n\nPROBE-accuracy over your lineage so far: "+hist
    P+="\n\nOutput ONE improved genome in a ```json block, then (a) predicted behavior (b) failure modes/reversal handling (c) which environmental signal drives each plastic term."
    return P

def run_lineage_e3(condition, ngen=10, temp=0.5):
    print(f"### E3 LINEAGE [{condition}] ngen={ngen} temp={temp}", flush=True)
    prompt=gen0_e3_prompt(); prev=None; hist=[]; best=-1; results=[]
    for g in range(ngen):
        txt=qwen_gen(prompt, temp=temp); genome=parse_genome(txt)
        if genome is None:
            print(f"  gen{g}: PARSE_FAIL", flush=True); results.append({'gen':g,'ok':False})
            prompt=intergen_e3("Your previous output could not be parsed as valid genome JSON.", prev, ", ".join(f"{h:.2f}" for h in hist)); continue
        metrics,ev=run_and_evidence_e3(genome, seed=100+g)
        pa=metrics.get('probe_acc',0.0) if metrics.get('ok') else 0.0
        hist.append(pa); best=max(best,pa); results.append({'gen':g,'ok':metrics.get('ok'),'probe_acc':pa,'post_rev':metrics.get('post_rev')})
        print(f"  gen{g}: probe_acc={pa:.2f} post_rev={metrics.get('post_rev',0)} best={best:.2f} {'' if metrics.get('ok') else metrics.get('msg')}", flush=True)
        if condition=='shuffled':
            _,ev_use=run_and_evidence_e3(rand_gen_genome(9999+g*7), seed=100+g)   # evidence from UNRELATED random genome
        else:
            ev_use=ev
        prevg = genome if condition in ('ancestry','shuffled') else None
        prompt=intergen_e3(ev_use, prevg, ", ".join(f"{h:.2f}" for h in hist)); prev=genome
    print(f"### DONE [{condition}] best_probe_acc={best:.2f} trajectory={[round(r.get('probe_acc',0),2) for r in results]}", flush=True)
    return results

if __name__=='__main__':
    if os.environ.get('LIN_STAGE')=='gen0':
        survs=[]
        for i in range(10):
            temp=0.0 if i<5 else 0.7
            g=parse_genome(qwen_gen(gen0_e3_prompt(), temp=temp))
            if g is None: print(f"  q{i}(t{temp}): PARSE_FAIL", flush=True); continue
            m,_=run_and_evidence_e3(g, seed=7); pa=m.get('probe_acc',0) if m.get('ok') else 0
            survs.append(pa); print(f"  q{i}(t{temp}): probe_acc={pa:.2f} ok={m.get('ok')}", flush=True)
        if survs: print(f"  >> Qwen E3 gen-0 probe_acc: median={np.median(survs):.2f} best={max(survs):.2f}", flush=True)
    else:
        allr={}
        for cond in ['ancestry','restart','shuffled']:
            allr[cond]=run_lineage_e3(cond, ngen=int(os.environ.get('LIN_NGEN','10')), temp=float(os.environ.get('LIN_TEMP','0.5')))
        json.dump(allr, open('/home/pokazge/NativeEntity/lineage_e3_results.json','w'), indent=1)
        print("=== E3 LINEAGE SUMMARY ===", flush=True)
        for c,rs in allr.items(): print(f"  {c:9s}: traj={[round(r.get('probe_acc',0),2) for r in rs]} best={max([r.get('probe_acc',0) for r in rs]):.2f}", flush=True)
