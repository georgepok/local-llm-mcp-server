# TEMPORAL DYNAMICS DIAGNOSTIC — the static rep manifold is flat (~12-dim). Is the SEQUENCING (the trajectory the reps
# trace as the LLM generates) flat or curved? Decisive for the read: if the next state is LINEARLY predictable from the
# current (delta x_{t+1}-x_t ~ A x_t), the dynamics are flat and the Liquid's curvature has nothing to capture (explains
# curved==diagonal). If a small NONLINEAR predictor beats linear on HELD-OUT steps, the sequencing is curved -> the
# read+Liquid have real temporal geometry to model. Metric = held-out delta-R^2 gap (nonlinear - linear); the trajectory
# is a low-dim curve so this is reliable (no high-dim geodesic confound). Calibrated vs synthetic linear & nonlinear systems.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, random
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); random.seed(0); dev = torch.device('cuda')
MODEL = '/home/pokazge/models/Qwen3.6-27B'; LAYERS = [8, 16, 27, 35]; DRED = 24

def pca(X, d):
    Xc = X - X.mean(0); U, S, V = torch.linalg.svd(Xc, full_matrices=False)
    return Xc @ V[:d].T

def held_out_gap(X):                                               # X:[T,d] ordered trajectory -> (R2_lin, R2_nl) on delta, held-out
    X = X.float(); X = pca(X, min(DRED, X.shape[1]))
    inp = X[:-1]; out = X[1:] - X[:-1]                            # predict the DELTA (residual dynamics)
    n = inp.shape[0]; s = n // 2
    Itr, Otr, Ite, Ote = inp[:s], out[:s], inp[s:], out[s:]
    def r2(pred):
        return float(1 - ((pred - Ote) ** 2).sum() / (((Ote - Ote.mean(0)) ** 2).sum() + 1e-9))
    # linear (ridge)
    lam = 1e-1 * torch.eye(Itr.shape[1], device=dev)
    W = torch.linalg.solve(Itr.T @ Itr + lam, Itr.T @ Otr)
    r2_lin = r2(Ite @ W)
    # nonlinear (small MLP), trained ONLY on train half, scored on held-out
    net = nn.Sequential(nn.Linear(Itr.shape[1], 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, Otr.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-4)
    for _ in range(600): opt.zero_grad(); F.mse_loss(net(Itr), Otr).backward(); opt.step()
    net.eval(); r2_nl = r2(net(Ite))
    return r2_lin, r2_nl

def turning(X):                                                    # mean turning angle of the trajectory (straight=1, random-walk~0)
    X = X.float(); v = X[1:] - X[:-1]; v = F.normalize(v, dim=-1)
    return float((v[1:] * v[:-1]).sum(-1).mean())                 # cos angle between consecutive velocity vectors

print('=== CALIBRATION (synthetic dynamics; gap = R2_nl - R2_lin) ===', flush=True)
T = 400
# linear system: x_{t+1} = A x_t + noise  (gap should be ~0)
A = torch.linalg.qr(torch.randn(DRED, DRED, device=dev))[0] * 0.98
xl = [torch.randn(DRED, device=dev)]
for _ in range(T): xl.append(A @ xl[-1] + 0.03 * torch.randn(DRED, device=dev))
XL = torch.stack(xl); rl, rn = held_out_gap(XL)
print('linear system:    R2_lin=%.3f R2_nl=%.3f  GAP=%.3f  turn=%.3f' % (rl, rn, rn - rl, turning(XL)), flush=True)
# nonlinear system: bounded curved flow on the sphere (gap should be >0, R2 sane)
B = torch.randn(DRED, DRED, device=dev) * 0.8; B = B - B.T          # skew -> rotation-like, bounded
xn = [F.normalize(torch.randn(DRED, device=dev), dim=0)]
for _ in range(T): xn.append(F.normalize(xn[-1] + 0.25 * torch.tanh(B @ xn[-1]) + 0.005 * torch.randn(DRED, device=dev), dim=0))
XN = torch.stack(xn); rl, rn = held_out_gap(XN)
print('nonlinear system: R2_lin=%.3f R2_nl=%.3f  GAP=%.3f  turn=%.3f' % (rl, rn, rn - rl, turning(XN)), flush=True)

PROMPTS = [
    "Write a flowing reflection on what it means to pay attention in a distracted world.",
    "Explain step by step how to debug a program that intermittently crashes under load.",
    "Tell a short story about a lighthouse keeper who has not seen another person in years.",
]
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False

@torch.no_grad()
def trajectory(prompt, ntok=280):
    try: s = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: s = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
    ids = tok(s, return_tensors='pt').input_ids.to(dev)
    gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=ntok, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id)
    out = model(gen, output_hidden_states=True)
    s0 = ids.shape[1]
    return {L: out.hidden_states[L][0, s0:] for L in LAYERS}      # ordered trajectory over GENERATED tokens

trajs = [trajectory(p) for p in PROMPTS]
print('=== LLM rep-trajectory dynamics (mean over %d generations) ===' % len(PROMPTS), flush=True)
for L in LAYERS:
    gls, gns, turns = [], [], []
    for tr in trajs:
        rl, rn = held_out_gap(tr[L]); gls.append(rl); gns.append(rn); turns.append(turning(tr[L]))
    al, an = sum(gls) / len(gls), sum(gns) / len(gns)
    print('  layer %2d: R2_lin=%.3f  R2_nl=%.3f  GAP(nl-lin)=%.3f  turn=%.3f' % (L, al, an, an - al, sum(turns) / len(turns)), flush=True)
print('=== DIAG_DONE === (GAP~0 like linear-ref => flat dynamics, curvature has nothing to capture; GAP>>0 like nonlinear-ref => curved dynamics for read+Liquid)', flush=True)
