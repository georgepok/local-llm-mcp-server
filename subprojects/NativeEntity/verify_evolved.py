import os, json
os.environ['ME_MODE']='none'
import numpy as np
from physics import step
from diag import detect, sfield
from repro import run_repro, GS_REPLICATE
import physics

# ARTIFACT-FIRST verification: is the evolved law's high reproduction (final 20.7 vs GS 10) GENUINE stronger
# replication, or Part-14 metric-gaming (unbounded ch3 autocatalysis saturating into counted noise-blobs)?
# Ablate the added ch3 term; decompose blob count by channel; check saturation.
L=json.load(open('/home/pokazge/NativeEntity/evolved_best_law.json'))
L_noch3=dict(L); L_noch3['terms']=[t for t in L['terms'] if not (t['tgt']==3 and t['op']=='react')]  # remove ch3 autocatalysis

def seed01(H,W,D,kind,s):
    X=(0.02*np.random.RandomState(s).standard_normal((H,W,D))).astype(np.float32); X[:,:,0]+=1.0; X[3:8,3:8,1]+=0.3; return X

def analyze(law, name):
    law=dict(law); law['base_leak']=0.0; physics.init_field=seed01
    X=seed01(32,32,8,'x',0); rng=np.random.RandomState(0)
    for t in range(4000): X=step(X,law,t,rng)
    # per-channel: how "saturated" (fraction at clamp) and structure count using ONLY ch0/ch1 vs ONLY ch3
    def blobs_ch(chs):
        Xc=X.copy()
        for c in range(8):
            if c not in chs: Xc[:,:,c]=0.0
        return len(detect(Xc)[0])
    sat3=float((X[:,:,3]>1.2).mean()); mean3=float(X[:,:,3].mean())
    n_gs=blobs_ch([0,1]); n_ch3=blobs_ch([3]); n_all=len(detect(X)[0])
    print(f"  {name:16s}: total_blobs={n_all} | GS-spots(ch0,1)={n_gs} ch3-blobs={n_ch3} | ch3 mean={mean3:.2f} ch3_saturated_frac={sat3:.2f}", flush=True)
    return n_gs, n_ch3, sat3

if __name__=='__main__':
    print("=== VERIFY EVOLVED LAW: genuine reproduction vs ch3-saturation metric-gaming? ===", flush=True)
    print("  evolved law = GS-core + ch2-decay(noop) + ch3 autocatalysis (react [3,0,3] coef 0.52, NO diffusion)", flush=True)
    analyze(GS_REPLICATE, 'GS_REPLICATE')
    analyze(L, 'EVOLVED(full)')
    analyze(L_noch3, 'EVOLVED(no ch3)')
    # reproduction fitness with vs without ch3
    r_full=run_repro(L,0,1,T=4000); r_no=run_repro(L_noch3,0,1,T=4000)
    print(f"  reproduction final_count: EVOLVED full={r_full['final_count'] if r_full else 'NA'} | no-ch3={r_no['final_count'] if r_no else 'NA'} | GS_REPLICATE~10", flush=True)
    print("=== VERDICT: if ch3 saturated (frac high) AND no-ch3 count ~= GS -> fitness gain is metric-gaming, NOT genuine reproduction ===", flush=True)
    print("=== VERIFY_DONE ===", flush=True)
