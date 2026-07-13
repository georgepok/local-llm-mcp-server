import os
os.environ['ME_MODE']='none'
import numpy as np
from physics import step
from diag import detect

# Validate a REPRODUCTION positive control: which GS regime actually COLONIZES (replicates spots) from ONE seed?
# stable-spots (F.037/k.060) makes persistent non-replicating spots; mitosis (F.0245/k.0555) replicates.
def gs(F,k): return {'D':8,'dt':1.0,'noise':0.0,'clamp':[0,1.3],'base_leak':0.0,'terms':[
    {'tgt':0,'op':'diffuse','coef':0.16},{'tgt':1,'op':'diffuse','coef':0.08},
    {'tgt':0,'op':'react','src':[0,1,1],'coef':-1.0},{'tgt':1,'op':'react','src':[0,1,1],'coef':1.0},
    {'tgt':0,'op':'supply','coef':F,'target':1.0},{'tgt':1,'op':'supply','coef':F+k,'target':0.0}]}
def seed(strong=False):
    X=np.zeros((32,32,8),np.float32); X[:,:,0]=1.0
    if strong: X[13:19,13:19,1]=0.5; X[13:19,13:19,0]=0.3   # central stronger seed
    else: X[3:8,3:8,1]=0.3
    return X
if __name__=='__main__':
    for name,F,k,st,leak in [('spots F037k060',0.037,0.060,False,0.0),('mitosis F0245k0555',0.0245,0.0555,False,0.0),
                             ('mitosis strong',0.0245,0.0555,True,0.0),('mitosis strong+leak',0.0245,0.0555,True,0.004),
                             ('uskate F062k061',0.062,0.061,True,0.0)]:
        law=gs(F,k); law['base_leak']=leak; X=seed(st); rng=np.random.RandomState(0); cs=[]
        for t in range(6000):
            X=step(X,law,t,rng)
            if t%600==0: cs.append(len(detect(X)[0]))
        print(f"  GS {name:22s}: counts={cs}", flush=True)
    print("=== GSCOLON_DONE ===", flush=True)
