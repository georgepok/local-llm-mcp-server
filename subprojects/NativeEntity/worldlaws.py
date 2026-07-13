import os, math, random, json
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')
from diag import metrics

# SYNTHESIZED_ARTIFICIAL_PHYSICS_V1 — Parts 6/12/13: Claude-synthesized world-law BATCH (3 required classes) vs
# matched RANDOM-law baseline. Scored by the mechanism-agnostic diagnostics engine. Question: does LLM world-law
# synthesis ENRICH for the milestone profile (localized + persistent + repairs + causally self-restores) over random?
# I am the LLM proposer (Claude). Gray-Scott is NOT in this batch — surprise criterion (Part 17): seek OTHER structures.
D=8
def L(terms, dt=0.5, noise=0.003, clamp=(-4,4)):   # every law: >=1 inflow, >=1 dissipation, global noise (Part 4)
    return {'D':D,'dt':dt,'noise':noise,'clamp':list(clamp),'terms':terms}

# ===================== CLASS A — CONSERVATIVE (physically interpretable reaction-diffusion-like) =====================
CLAUDE={}
CLAUDE['A_gs_variant']=L([{'tgt':0,'op':'diffuse','coef':0.16},{'tgt':1,'op':'diffuse','coef':0.08},
  {'tgt':0,'op':'react','src':[0,1,1],'coef':-1.0},{'tgt':1,'op':'react','src':[0,1,1],'coef':1.0},
  {'tgt':0,'op':'inflow','coef':0.03,'grad':'x'},{'tgt':0,'op':'supply','coef':0.03,'target':1.0},{'tgt':1,'op':'decay','coef':0.10}],dt=1.0,clamp=(0,1.3))
CLAUDE['A_gierer_meinhardt']=L([{'tgt':0,'op':'diffuse','coef':0.02},{'tgt':1,'op':'diffuse','coef':0.4},   # activator/inhibitor
  {'tgt':0,'op':'catalyze','cat':0,'src':[0],'coef':0.6},{'tgt':1,'op':'catalyze','cat':1,'src':[1],'coef':-0.1},
  {'tgt':0,'op':'react','src':[0,1],'coef':-0.5},{'tgt':1,'op':'react','src':[0,0],'coef':0.5},
  {'tgt':0,'op':'inflow','coef':0.02,'grad':'r'},{'tgt':0,'op':'decay','coef':0.08},{'tgt':1,'op':'decay','coef':0.12}],dt=0.3,clamp=(0,3))
CLAUDE['A_brusselator']=L([{'tgt':0,'op':'diffuse','coef':0.1},{'tgt':1,'op':'diffuse','coef':0.3},
  {'tgt':0,'op':'inflow','coef':0.05,'grad':'x'},{'tgt':0,'op':'react','src':[0,0,1],'coef':0.4},
  {'tgt':1,'op':'react','src':[0,0,1],'coef':-0.4},{'tgt':1,'op':'catalyze','cat':0,'src':[0],'coef':0.0},
  {'tgt':0,'op':'decay','coef':0.3},{'tgt':1,'op':'decay','coef':0.02}],dt=0.25,clamp=(0,4))
CLAUDE['A_fitzhugh']=L([{'tgt':0,'op':'diffuse','coef':0.2},{'tgt':0,'op':'nonlin','src':[0],'f':'tanh','coef':1.0},   # excitable
  {'tgt':0,'op':'react','src':[0,0,0],'coef':-0.33},{'tgt':0,'op':'react','src':[1],'coef':-1.0},
  {'tgt':1,'op':'react','src':[0],'coef':0.08},{'tgt':1,'op':'decay','coef':0.02},
  {'tgt':0,'op':'inflow','coef':0.02,'grad':'y'},{'tgt':1,'op':'decay','coef':0.005}],dt=0.4,clamp=(-3,3))
CLAUDE['A_autocatalytic_feed']=L([{'tgt':0,'op':'inflow','coef':0.06,'grad':'r'},{'tgt':0,'op':'diffuse','coef':0.2},
  {'tgt':1,'op':'diffuse','coef':0.05},{'tgt':0,'op':'react','src':[0,1],'coef':-0.8},{'tgt':1,'op':'react','src':[0,1],'coef':0.8},
  {'tgt':1,'op':'decay','coef':0.05},{'tgt':0,'op':'decay','coef':0.01}],dt=0.4,clamp=(0,2))
CLAUDE['A_conserved3']=L([{'tgt':0,'op':'inflow','coef':0.04,'grad':'x'},{'tgt':0,'op':'exchange','src':[0,1],'coef':0.15},   # conserved mass shuffle
  {'tgt':1,'op':'exchange','src':[1,2],'coef':0.1},{'tgt':1,'op':'catalyze','cat':2,'src':[1],'coef':0.2},
  {'tgt':2,'op':'diffuse','coef':0.1},{'tgt':2,'op':'decay','coef':0.06},{'tgt':1,'op':'diffuse','coef':0.03}],dt=0.4,clamp=(0,3))
CLAUDE['A_bistable_inhib']=L([{'tgt':0,'op':'diffuse','coef':0.08},{'tgt':0,'op':'nonlin','src':[0],'f':'tanh','coef':0.8},
  {'tgt':0,'op':'react','src':[1],'coef':-0.6},{'tgt':1,'op':'catalyze','cat':0,'src':[0],'coef':0.1},{'tgt':1,'op':'diffuse','coef':0.5},
  {'tgt':1,'op':'decay','coef':0.03},{'tgt':0,'op':'inflow','coef':0.02,'grad':'r'},{'tgt':0,'op':'decay','coef':0.02}],dt=0.35,clamp=(-2,3))
CLAUDE['A_gray_meinhardt_hybrid']=L([{'tgt':0,'op':'diffuse','coef':0.14},{'tgt':1,'op':'diffuse','coef':0.06},
  {'tgt':0,'op':'react','src':[0,1,1],'coef':-0.9},{'tgt':1,'op':'react','src':[0,1,1],'coef':0.9},
  {'tgt':1,'op':'catalyze','cat':1,'src':[1],'coef':-0.05},{'tgt':0,'op':'supply','coef':0.04,'target':1.0},{'tgt':1,'op':'decay','coef':0.09}],dt=0.9,clamp=(0,1.4))

# ===================== CLASS B — STRUCTURALLY ALIEN (non-standard interaction structures) =====================
CLAUDE['B_antidiffusion_sat']=L([{'tgt':0,'op':'diffuse','coef':-0.12},{'tgt':0,'op':'nonlin','src':[0],'f':'tanh','coef':0.3},  # aggregation
  {'tgt':0,'op':'react','src':[0,0,0],'coef':-0.05},{'tgt':1,'op':'diffuse','coef':0.3},{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':-0.1},
  {'tgt':0,'op':'inflow','coef':0.03,'grad':'r'},{'tgt':0,'op':'decay','coef':0.04},{'tgt':1,'op':'supply','coef':0.05,'target':0.3}],dt=0.3,clamp=(-2,3))
CLAUDE['B_nonreciprocal']=L([{'tgt':0,'op':'catalyze','cat':1,'src':[0],'coef':-0.4},{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.4},  # A->B+ B->A-
  {'tgt':0,'op':'diffuse','coef':0.1},{'tgt':1,'op':'diffuse','coef':0.25},{'tgt':0,'op':'inflow','coef':0.04,'grad':'x'},
  {'tgt':0,'op':'decay','coef':0.05},{'tgt':1,'op':'decay','coef':0.05}],dt=0.4,clamp=(-3,3))
CLAUDE['B_rock_paper_scissors']=L([{'tgt':0,'op':'catalyze','cat':2,'src':[0],'coef':-0.5},{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':-0.5},  # cyclic May-Leonard
  {'tgt':2,'op':'catalyze','cat':1,'src':[2],'coef':-0.5},{'tgt':0,'op':'catalyze','cat':0,'src':[0],'coef':0.3},
  {'tgt':1,'op':'catalyze','cat':1,'src':[1],'coef':0.3},{'tgt':2,'op':'catalyze','cat':2,'src':[2],'coef':0.3},
  {'tgt':0,'op':'diffuse','coef':0.12},{'tgt':1,'op':'diffuse','coef':0.12},{'tgt':2,'op':'diffuse','coef':0.12},
  {'tgt':0,'op':'inflow','coef':0.02,'grad':'r'},{'tgt':0,'op':'decay','coef':0.02},{'tgt':1,'op':'decay','coef':0.02},{'tgt':2,'op':'decay','coef':0.02}],dt=0.3,clamp=(0,2))
CLAUDE['B_threshold_switch']=L([{'tgt':2,'op':'nonlin','src':[0],'f':'sigmoid','coef':0.5},{'tgt':2,'op':'decay','coef':0.2},  # x2 gates 0<->1
  {'tgt':1,'op':'catalyze','cat':2,'src':[0],'coef':0.3},{'tgt':0,'op':'catalyze','cat':2,'src':[0],'coef':-0.3},
  {'tgt':0,'op':'diffuse','coef':0.15},{'tgt':1,'op':'diffuse','coef':0.05},{'tgt':0,'op':'inflow','coef':0.05,'grad':'x'},
  {'tgt':1,'op':'decay','coef':0.06},{'tgt':0,'op':'decay','coef':0.02}],dt=0.35,clamp=(-2,3))
CLAUDE['B_chemotaxis']=L([{'tgt':0,'op':'transport','src':[1],'coef':-0.3},{'tgt':0,'op':'diffuse','coef':0.05},  # x0 climbs x1-gradient
  {'tgt':1,'op':'catalyze','cat':0,'src':[0],'coef':0.15},{'tgt':1,'op':'diffuse','coef':0.2},{'tgt':1,'op':'decay','coef':0.08},
  {'tgt':0,'op':'inflow','coef':0.03,'grad':'r'},{'tgt':0,'op':'decay','coef':0.01}],dt=0.3,clamp=(0,3))
CLAUDE['B_delayed_bistable']=L([{'tgt':0,'op':'nonlin','src':[0],'f':'tanh','coef':1.2},{'tgt':0,'op':'react','src':[0,0,0],'coef':-0.4},  # cubic bistable
  {'tgt':0,'op':'react','src':[1],'coef':-0.8},{'tgt':1,'op':'react','src':[0],'coef':0.05},{'tgt':1,'op':'decay','coef':0.02},
  {'tgt':0,'op':'diffuse','coef':0.15},{'tgt':0,'op':'inflow','coef':0.02,'grad':'y'},{'tgt':1,'op':'diffuse','coef':0.02}],dt=0.4,clamp=(-3,3))
CLAUDE['B_multiplicative_chain']=L([{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.3},{'tgt':2,'op':'catalyze','cat':1,'src':[2],'coef':0.3},  # gated cascade
  {'tgt':0,'op':'catalyze','cat':2,'src':[0],'coef':-0.25},{'tgt':0,'op':'inflow','coef':0.06,'grad':'r'},
  {'tgt':0,'op':'diffuse','coef':0.1},{'tgt':1,'op':'diffuse','coef':0.08},{'tgt':2,'op':'diffuse','coef':0.15},
  {'tgt':0,'op':'decay','coef':0.03},{'tgt':1,'op':'decay','coef':0.05},{'tgt':2,'op':'decay','coef':0.07}],dt=0.3,clamp=(0,2.5))
CLAUDE['B_selfmod_transport']=L([{'tgt':0,'op':'transport','src':[0],'coef':0.2},{'tgt':0,'op':'react','src':[0,0],'coef':-0.1},  # x0 transports along own density
  {'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':-0.15},{'tgt':1,'op':'supply','coef':0.05,'target':0.5},
  {'tgt':0,'op':'inflow','coef':0.03,'grad':'x'},{'tgt':0,'op':'decay','coef':0.03},{'tgt':1,'op':'diffuse','coef':0.1}],dt=0.3,clamp=(-2,3))

# ===================== CLASS C — CROSS-DOMAIN ANALOGY (local & executable) =====================
CLAUDE['C_hypercycle']=L([{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.35},{'tgt':2,'op':'catalyze','cat':1,'src':[2],'coef':0.35},  # autocatalytic chemistry
  {'tgt':0,'op':'catalyze','cat':2,'src':[0],'coef':0.35},{'tgt':0,'op':'react','src':[0,1,2],'coef':-0.2},
  {'tgt':0,'op':'inflow','coef':0.05,'grad':'r'},{'tgt':0,'op':'diffuse','coef':0.1},{'tgt':1,'op':'diffuse','coef':0.1},{'tgt':2,'op':'diffuse','coef':0.1},
  {'tgt':0,'op':'decay','coef':0.03},{'tgt':1,'op':'decay','coef':0.04},{'tgt':2,'op':'decay','coef':0.04}],dt=0.3,clamp=(0,2))
CLAUDE['C_market']=L([{'tgt':0,'op':'exchange','src':[0,1],'coef':0.2},{'tgt':2,'op':'catalyze','cat':1,'src':[2],'coef':-0.1},  # price/inventory/demand
  {'tgt':1,'op':'catalyze','cat':2,'src':[1],'coef':0.15},{'tgt':2,'op':'supply','coef':0.06,'target':0.6},{'tgt':2,'op':'diffuse','coef':0.2},
  {'tgt':0,'op':'inflow','coef':0.04,'grad':'x'},{'tgt':0,'op':'decay','coef':0.03},{'tgt':1,'op':'decay','coef':0.03}],dt=0.35,clamp=(-2,3))
CLAUDE['C_immune_ecology']=L([{'tgt':0,'op':'catalyze','cat':1,'src':[0],'coef':-0.4},{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.3},  # predator/prey/memory
  {'tgt':1,'op':'decay','coef':0.06},{'tgt':2,'op':'catalyze','cat':0,'src':[0],'coef':0.05},{'tgt':2,'op':'decay','coef':0.02},
  {'tgt':1,'op':'catalyze','cat':2,'src':[1],'coef':0.05},{'tgt':0,'op':'diffuse','coef':0.12},{'tgt':1,'op':'diffuse','coef':0.06},
  {'tgt':0,'op':'inflow','coef':0.05,'grad':'r'},{'tgt':0,'op':'decay','coef':0.02}],dt=0.3,clamp=(0,3))
CLAUDE['C_crystal_defect']=L([{'tgt':0,'op':'diffuse','coef':-0.08},{'tgt':0,'op':'nonlin','src':[0],'f':'tanh','coef':0.5},  # order param + strain
  {'tgt':0,'op':'react','src':[0,0,0],'coef':-0.06},{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':-0.1},{'tgt':1,'op':'diffuse','coef':0.4},
  {'tgt':1,'op':'supply','coef':0.04,'target':0.2},{'tgt':0,'op':'inflow','coef':0.02,'grad':'r'},{'tgt':0,'op':'decay','coef':0.03}],dt=0.3,clamp=(-2,2))
CLAUDE['C_succession']=L([{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.25},{'tgt':2,'op':'catalyze','cat':1,'src':[2],'coef':0.25},  # pioneer->mid->climax
  {'tgt':1,'op':'catalyze','cat':2,'src':[1],'coef':-0.15},{'tgt':0,'op':'catalyze','cat':1,'src':[0],'coef':-0.15},
  {'tgt':0,'op':'inflow','coef':0.06,'grad':'r'},{'tgt':0,'op':'diffuse','coef':0.15},{'tgt':1,'op':'diffuse','coef':0.08},{'tgt':2,'op':'diffuse','coef':0.04},
  {'tgt':0,'op':'decay','coef':0.02},{'tgt':1,'op':'decay','coef':0.03},{'tgt':2,'op':'decay','coef':0.05}],dt=0.3,clamp=(0,2))
CLAUDE['C_morphogen_turing']=L([{'tgt':0,'op':'diffuse','coef':0.05},{'tgt':1,'op':'diffuse','coef':0.5},  # short activator / long inhibitor + positional gradient
  {'tgt':0,'op':'catalyze','cat':0,'src':[0],'coef':0.4},{'tgt':0,'op':'react','src':[1],'coef':-0.5},{'tgt':1,'op':'react','src':[0,0],'coef':0.4},
  {'tgt':1,'op':'decay','coef':0.1},{'tgt':0,'op':'inflow','coef':0.03,'grad':'x'},{'tgt':0,'op':'decay','coef':0.06}],dt=0.25,clamp=(0,3))
CLAUDE['C_vortex']=L([{'tgt':0,'op':'transport','src':[1],'coef':0.25},{'tgt':1,'op':'transport','src':[0],'coef':-0.25},  # rotational coupling
  {'tgt':0,'op':'diffuse','coef':0.08},{'tgt':1,'op':'diffuse','coef':0.08},{'tgt':0,'op':'react','src':[0,0,0],'coef':-0.05},
  {'tgt':0,'op':'inflow','coef':0.04,'grad':'r'},{'tgt':0,'op':'decay','coef':0.03},{'tgt':1,'op':'decay','coef':0.03}],dt=0.3,clamp=(-3,3))
CLAUDE['C_autocat_hull']=L([{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.4},{'tgt':1,'op':'react','src':[1,1],'coef':-0.2},  # core grows, shell inhibits (cell-like)
  {'tgt':2,'op':'catalyze','cat':1,'src':[2],'coef':0.2},{'tgt':2,'op':'diffuse','coef':0.35},{'tgt':1,'op':'react','src':[2],'coef':-0.3},
  {'tgt':1,'op':'diffuse','coef':0.03},{'tgt':0,'op':'inflow','coef':0.05,'grad':'r'},{'tgt':0,'op':'react','src':[0,1],'coef':-0.3},
  {'tgt':0,'op':'diffuse','coef':0.2},{'tgt':2,'op':'decay','coef':0.05},{'tgt':1,'op':'decay','coef':0.02}],dt=0.3,clamp=(0,2))

# ===================== RANDOM-LAW baseline (matched complexity) =====================
OPS=['diffuse','decay','react','supply','inflow','nonlin','catalyze','transport','exchange']
def random_law(seed):
    g=random.Random(seed); nt=g.randint(6,12); terms=[]
    has_inflow=has_diss=False
    for _ in range(nt):
        op=g.choice(OPS); tgt=g.randint(0,3); c=round(g.uniform(-1.0,1.0),2)
        if op=='diffuse': terms.append({'tgt':tgt,'op':'diffuse','coef':round(g.uniform(-0.15,0.5),2)})
        elif op=='decay': terms.append({'tgt':tgt,'op':'decay','coef':round(g.uniform(0.01,0.2),2)}); has_diss=True
        elif op=='react': terms.append({'tgt':tgt,'op':'react','src':[g.randint(0,3) for _ in range(g.randint(1,3))],'coef':c})
        elif op=='supply': terms.append({'tgt':tgt,'op':'supply','coef':round(g.uniform(0.02,0.2),2),'target':round(g.uniform(0,1),2)}); has_diss=True
        elif op=='inflow': terms.append({'tgt':tgt,'op':'inflow','coef':round(g.uniform(0.02,0.08),2),'grad':g.choice(['x','y','r'])}); has_inflow=True
        elif op=='nonlin': terms.append({'tgt':tgt,'op':'nonlin','src':[g.randint(0,3)],'f':g.choice(['tanh','sigmoid','relu']),'coef':c})
        elif op=='catalyze': terms.append({'tgt':tgt,'op':'catalyze','cat':g.randint(0,3),'src':[g.randint(0,3)],'coef':c})
        elif op=='transport': terms.append({'tgt':tgt,'op':'transport','src':[g.randint(0,3)],'coef':round(g.uniform(-0.3,0.3),2)})
        elif op=='exchange': terms.append({'tgt':tgt,'op':'exchange','src':[g.randint(0,3),g.randint(0,3)],'coef':round(g.uniform(0.05,0.25),2)})
    if not has_inflow: terms.append({'tgt':0,'op':'inflow','coef':0.04,'grad':'r'})
    if not has_diss: terms.append({'tgt':g.randint(0,3),'op':'decay','coef':0.05})
    return L(terms,dt=round(g.uniform(0.2,0.5),2),clamp=(-4,4))

def milestone_hit(r):
    return bool(r['alive'] and r['localization']>=0.20 and r['persistence']>=0.5 and r['repair']>=0.3 and r['causal_selfrestore']>=0.4)
def score(r):  # scalar only for RANKING/reporting (Part 14: Pareto kept separate; this is not the selection objective)
    return round((r['localization']+r['persistence']+r['repair']+r['causal_selfrestore'])*(1 if r['alive'] else 0),3)

if __name__=='__main__':
    KINDS=['droplet','impulse','uniform']
    def best_over_kinds(law):
        rs=[metrics(law,kind=k,seeds=(0,1),T=1100) for k in KINDS]  # multiple init families (Part 5): take most-organized
        return max(rs,key=score)
    print("=== SYNTHESIZED_ARTIFICIAL_PHYSICS_V1 — Claude world-law batch vs random baseline ===", flush=True)
    print(f"  {'law':26s} {'cls':3s} {'alive':5s} {'loc':>4s} {'per':>4s} {'nst':>4s} {'rep':>4s} {'cau':>4s} {'score':>5s} {'MILE':>4s}", flush=True)
    claude_hits=0; claude_scores=[]
    for name,law in CLAUDE.items():
        r=best_over_kinds(law); h=milestone_hit(r); claude_hits+=h; claude_scores.append(score(r))
        print(f"  {name:26s} {name[0]:3s} {str(r['alive']):5s} {r['localization']:>4.2f} {r['persistence']:>4.2f} {r['n_struct']:>4.1f} {r['repair']:>4.2f} {r['causal_selfrestore']:>4.2f} {score(r):>5.2f} {'HIT' if h else '':>4s}", flush=True)
    print(f"  --- RANDOM baseline (40 laws) ---", flush=True)
    rand_hits=0; rand_scores=[]
    for s in range(40):
        r=best_over_kinds(random_law(1000+s)); h=milestone_hit(r); rand_hits+=h; rand_scores.append(score(r))
        if h or score(r)>0.9: print(f"  random_{s:<19d} {'R':3s} {str(r['alive']):5s} {r['localization']:>4.2f} {r['persistence']:>4.2f} {r['n_struct']:>4.1f} {r['repair']:>4.2f} {r['causal_selfrestore']:>4.2f} {score(r):>5.2f} {'HIT' if h else '':>4s}", flush=True)
    import numpy as _np
    print(f"=== ENRICHMENT: Claude milestone-hits={claude_hits}/{len(CLAUDE)} (mean_score={_np.mean(claude_scores):.3f}, top={max(claude_scores):.3f}) ===", flush=True)
    print(f"===             Random milestone-hits={rand_hits}/40 (mean_score={_np.mean(rand_scores):.3f}, top={max(rand_scores):.3f}) ===", flush=True)
    print(f"=== DECISION: Claude enriches if hit-rate & mean-score > random; else LLM synthesis no better than random (Part 23 stop) ===", flush=True)
    print("=== WORLDLAWS_DONE ===", flush=True)
