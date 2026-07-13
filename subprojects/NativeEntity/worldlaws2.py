import os, math, random
os.environ['ME_MODE']='none'
import numpy as np
np.seterr(over='ignore', invalid='ignore')
from worldlaws import L, D
from diag2 import hardened, gs_init
import physics, diag
_oi=physics.init_field
physics.init_field=lambda H,W,Dd,kind,seed:(gs_init(seed) if kind=='gs' else _oi(H,W,Dd,kind,seed))
diag.init_field=physics.init_field

# META-GEN 2 (Part 21 Case-B rule: increase nonequilibrium flow + dissipation; Part 16: topology changes not param tuning).
# Meta-gen-1 gave only static/frozen (Case B): weak inflow -> near-equilibrium Turing attractors. Here: DOMINANT drive
# (strong feed + strong washout = chemostat) across DIVERSE topologies. Question (Part 17 surprise): can NON-Gray-Scott
# structures reach Case D (dissipative + organization causally self-restores), or is Case D only reaction-diffusion?
STRONG={}
# --- Conservative dissipative (strong-drive, non-GS reaction topologies) ---
STRONG['S_selkov_glycolytic']=L([{'tgt':0,'op':'supply','coef':0.10,'target':1.0},{'tgt':0,'op':'diffuse','coef':0.10},   # substrate fed
  {'tgt':0,'op':'react','src':[1,1,0],'coef':-1.0},{'tgt':1,'op':'react','src':[1,1,0],'coef':1.0},                       # p^2 s autocatalysis
  {'tgt':1,'op':'supply','coef':0.16,'target':0.0},{'tgt':1,'op':'diffuse','coef':0.04}],dt=0.8,clamp=(0,2.5))
STRONG['S_gierer_saturated']=L([{'tgt':1,'op':'supply','coef':0.12,'target':1.0},{'tgt':1,'op':'diffuse','coef':0.4},      # substrate s strongly fed
  {'tgt':0,'op':'catalyze','cat':1,'src':[0],'coef':0.5},{'tgt':0,'op':'react','src':[0,0],'coef':0.3},                    # activator a: s*a + a^2
  {'tgt':1,'op':'react','src':[0,0,1],'coef':-0.6},{'tgt':0,'op':'supply','coef':0.15,'target':0.0},{'tgt':0,'op':'diffuse','coef':0.03}],dt=0.4,clamp=(0,3))
STRONG['S_driven_brusselator']=L([{'tgt':0,'op':'supply','coef':0.14,'target':1.0},{'tgt':0,'op':'diffuse','coef':0.1},    # A fed strongly
  {'tgt':0,'op':'react','src':[0,0,1],'coef':0.5},{'tgt':1,'op':'react','src':[0,0,1],'coef':-0.5},{'tgt':1,'op':'catalyze','cat':2,'src':[0],'coef':0.3},
  {'tgt':1,'op':'supply','coef':0.18,'target':0.0},{'tgt':2,'op':'supply','coef':0.1,'target':0.5},{'tgt':1,'op':'diffuse','coef':0.3}],dt=0.25,clamp=(0,4))
STRONG['S_gs_worms']=L([{'tgt':0,'op':'diffuse','coef':0.16},{'tgt':1,'op':'diffuse','coef':0.08},                         # GS worms regime (mechanism-preserving anchor)
  {'tgt':0,'op':'react','src':[0,1,1],'coef':-1.0},{'tgt':1,'op':'react','src':[0,1,1],'coef':1.0},
  {'tgt':0,'op':'supply','coef':0.054,'target':1.0},{'tgt':1,'op':'supply','coef':0.117,'target':0.0}],dt=1.0,clamp=(0,1.5))
# --- Structurally alien (strongly driven) ---
STRONG['S_driven_rps']=L([{'tgt':3,'op':'supply','coef':0.12,'target':1.0},                                               # shared resource x3 fed strong
  {'tgt':0,'op':'catalyze','cat':3,'src':[0],'coef':0.4},{'tgt':1,'op':'catalyze','cat':3,'src':[1],'coef':0.4},{'tgt':2,'op':'catalyze','cat':3,'src':[2],'coef':0.4},
  {'tgt':0,'op':'catalyze','cat':2,'src':[0],'coef':-0.6},{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':-0.6},{'tgt':2,'op':'catalyze','cat':1,'src':[2],'coef':-0.6},
  {'tgt':3,'op':'react','src':[3,0],'coef':-0.4},{'tgt':3,'op':'react','src':[3,1],'coef':-0.4},{'tgt':3,'op':'react','src':[3,2],'coef':-0.4},
  {'tgt':0,'op':'supply','coef':0.1,'target':0.0},{'tgt':1,'op':'supply','coef':0.1,'target':0.0},{'tgt':2,'op':'supply','coef':0.1,'target':0.0},
  {'tgt':0,'op':'diffuse','coef':0.1},{'tgt':1,'op':'diffuse','coef':0.1},{'tgt':2,'op':'diffuse','coef':0.1}],dt=0.3,clamp=(0,2.5))
STRONG['S_driven_predprey']=L([{'tgt':0,'op':'supply','coef':0.12,'target':1.0},{'tgt':0,'op':'diffuse','coef':0.12},      # prey x0 fed strong
  {'tgt':0,'op':'catalyze','cat':1,'src':[0],'coef':-0.7},{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.6},          # predation
  {'tgt':1,'op':'supply','coef':0.16,'target':0.0},{'tgt':1,'op':'diffuse','coef':0.06},{'tgt':0,'op':'supply','coef':0.05,'target':0.0}],dt=0.3,clamp=(0,3))
STRONG['S_driven_front']=L([{'tgt':0,'op':'supply','coef':0.1,'target':1.0},{'tgt':0,'op':'nonlin','src':[0],'f':'tanh','coef':1.0},  # driven bistable front
  {'tgt':0,'op':'react','src':[0,0,0],'coef':-0.4},{'tgt':0,'op':'react','src':[1],'coef':-1.0},{'tgt':1,'op':'catalyze','cat':0,'src':[0],'coef':0.15},
  {'tgt':1,'op':'supply','coef':0.14,'target':0.0},{'tgt':0,'op':'diffuse','coef':0.15},{'tgt':1,'op':'diffuse','coef':0.02}],dt=0.35,clamp=(-2,3))
STRONG['S_selfmod_aggregate']=L([{'tgt':0,'op':'supply','coef':0.1,'target':1.0},{'tgt':0,'op':'transport','src':[0],'coef':0.25},   # density self-transport (aggregation) fed
  {'tgt':0,'op':'react','src':[0,0],'coef':-0.2},{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.2},{'tgt':1,'op':'supply','coef':0.15,'target':0.0},
  {'tgt':0,'op':'supply','coef':0.06,'target':0.0},{'tgt':1,'op':'diffuse','coef':0.1}],dt=0.3,clamp=(0,3))
# --- Cross-domain (strongly driven / chemostat) ---
STRONG['S_chemostat_hypercycle']=L([{'tgt':3,'op':'supply','coef':0.14,'target':1.0},                                     # resource washin
  {'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.5},{'tgt':2,'op':'catalyze','cat':1,'src':[2],'coef':0.5},{'tgt':0,'op':'catalyze','cat':2,'src':[0],'coef':0.5},
  {'tgt':1,'op':'react','src':[1,3],'coef':0.0},{'tgt':3,'op':'react','src':[3,0],'coef':-0.3},{'tgt':3,'op':'react','src':[3,1],'coef':-0.3},{'tgt':3,'op':'react','src':[3,2],'coef':-0.3},
  {'tgt':0,'op':'supply','coef':0.12,'target':0.0},{'tgt':1,'op':'supply','coef':0.12,'target':0.0},{'tgt':2,'op':'supply','coef':0.12,'target':0.0},  # washout all
  {'tgt':0,'op':'diffuse','coef':0.1},{'tgt':1,'op':'diffuse','coef':0.1},{'tgt':2,'op':'diffuse','coef':0.1}],dt=0.3,clamp=(0,2.5))
STRONG['S_driven_immune']=L([{'tgt':0,'op':'supply','coef':0.13,'target':1.0},{'tgt':0,'op':'diffuse','coef':0.14},        # antigen fed strong
  {'tgt':0,'op':'catalyze','cat':1,'src':[0],'coef':-0.7},{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.5},{'tgt':2,'op':'catalyze','cat':0,'src':[0],'coef':0.15},
  {'tgt':1,'op':'catalyze','cat':2,'src':[1],'coef':0.1},{'tgt':1,'op':'supply','coef':0.15,'target':0.0},{'tgt':2,'op':'supply','coef':0.08,'target':0.0},{'tgt':1,'op':'diffuse','coef':0.06}],dt=0.3,clamp=(0,3))
STRONG['S_driven_cell']=L([{'tgt':0,'op':'supply','coef':0.13,'target':1.0},{'tgt':0,'op':'diffuse','coef':0.2},           # resource fed; core grows, shell dissipates
  {'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.5},{'tgt':1,'op':'react','src':[1,1],'coef':-0.25},{'tgt':1,'op':'react','src':[2],'coef':-0.4},
  {'tgt':2,'op':'catalyze','cat':1,'src':[2],'coef':0.25},{'tgt':2,'op':'diffuse','coef':0.35},{'tgt':2,'op':'supply','coef':0.14,'target':0.0},
  {'tgt':0,'op':'react','src':[0,1],'coef':-0.4},{'tgt':1,'op':'supply','coef':0.06,'target':0.0},{'tgt':1,'op':'diffuse','coef':0.03}],dt=0.3,clamp=(0,2.5))
STRONG['S_driven_vortex']=L([{'tgt':0,'op':'supply','coef':0.1,'target':1.0},{'tgt':0,'op':'transport','src':[1],'coef':0.3},{'tgt':1,'op':'transport','src':[0],'coef':-0.3},  # driven rotational
  {'tgt':0,'op':'react','src':[0,0,0],'coef':-0.15},{'tgt':1,'op':'catalyze','cat':0,'src':[1],'coef':0.2},{'tgt':1,'op':'supply','coef':0.14,'target':0.0},
  {'tgt':0,'op':'supply','coef':0.05,'target':0.0},{'tgt':0,'op':'diffuse','coef':0.05},{'tgt':1,'op':'diffuse','coef':0.05}],dt=0.3,clamp=(-3,3))

def best(law,kinds=('droplet','impulse','uniform'),T=1200):
    rs=[hardened(law,kind=k,T=T) for k in kinds]
    return max(rs,key=lambda r:(r['case']=='D',r['org_caus']+r['inflow_dep']+r['repair_c']))

if __name__=='__main__':
    print("=== META-GEN 2: STRONG nonequilibrium drive (chemostat) x diverse topologies — can NON-GS reach Case D? ===", flush=True)
    print(f"  {'law':24s} {'case':4s} {'loc':>4s} {'infdep':>6s} {'dyn':>4s} {'orgC':>5s} {'repC':>5s} {'MILE':>4s}", flush=True)
    mile=0; cases={}
    for nm,law in STRONG.items():
        r=best(law); mile+=r['milestone']; cases[r['case']]=cases.get(r['case'],0)+1
        print(f"  {nm:24s} {r['case']:4s} {r['localization']:>4.2f} {r['inflow_dep']:>6.2f} {r['dynamic']:>4.1f} {r['org_caus']:>5.2f} {r['repair_c']:>5.2f} {'HIT' if r['milestone'] else '':>4s}", flush=True)
    print(f"=== META-GEN 2 RESULT: Case-D milestone={mile}/{len(STRONG)} case_dist={cases} ===", flush=True)
    print("=== if >=1 non-GS reaches Case D -> surprise candidate; if 0 -> Case B persists, escalate drive/dissipation further ===", flush=True)
    print("=== WORLDLAWS2_DONE ===", flush=True)
