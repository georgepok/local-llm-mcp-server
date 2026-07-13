import os
os.environ['ME_MODE']='none'
import numpy as np
from physics import step, GRAY_SCOTT
from diag2 import hardened
import diag, physics

# FAIRNESS PROBE: does the Case-D positive control (Gray-Scott) reach Case D from GENERIC random inits (near-0, near-op-point),
# or ONLY from the hand-seeded gs_init? If GS needs a favorable init the synthesized laws never get, the comparison is unfair
# and the init families must be fixed (start near a nonzero operating point + noise), NOT the metrics.
def mk(kind):
    def f(H,W,D,k,seed):
        g=np.random.RandomState(seed); X=(0.02*g.standard_normal((H,W,D))).astype(np.float32)
        if kind=='near0': pass
        elif kind=='u1':    X[:,:,0]+=1.0                                    # resource at operating point + noise (NOT seeded organism)
        elif kind=='mod':   X+=0.5*g.random((H,W,D)).astype(np.float32)      # moderate uniform field spanning range
        elif kind=='u1noisy': X[:,:,0]+=1.0; X[:,:,1]+=0.25*(g.random((H,W))<0.1)  # u1 + sparse v specks (nucleation sites, not a formed pattern)
        return X
    return f

if __name__=='__main__':
    print("=== INIT FAIRNESS PROBE: Gray-Scott case vs init family (is Case-D emergence-from-random or seed-dependent?) ===", flush=True)
    for kind in ['near0','mod','u1','u1noisy']:
        physics.init_field=mk(kind); diag.init_field=physics.init_field
        r=hardened(GRAY_SCOTT,kind='x',seeds=(0,1),T=1600,warm=900)
        print(f"  GS init={kind:9s}: case={r['case']} loc={r['localization']:.2f} infdep={r['inflow_dep']:.2f} orgC={r['org_caus']:.2f} struct={r['struct']}", flush=True)
    print("=== if GS is Case-D from 'mod'/'u1' (unseeded) -> fix generic inits to nonzero op-point; if only seeded -> milestone genuinely unmet ===", flush=True)
    print("=== INITPROBE_DONE ===", flush=True)
