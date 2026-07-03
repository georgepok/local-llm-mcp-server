# READ GEOMETRY DIAGNOSTIC v2 — CORRECTED. v1 measured classical-MDS neg_mass on AMBIENT Euclidean distances, which is
# ~0 for ANY point cloud in R^d (calibration caught it: sphere & hyperboloid also read 0.001). Manifold curvature only
# appears in GEODESIC (along-manifold) distances. So here:
#   (1) intrinsic dimension (TwoNN) — is there a low-dim MANIFOLD at all, or do reps fill the space?
#   (2) geodesic-Isomap neg_mass on a k-NN graph — curvature of that manifold (flat-embeddable => 0; curved => >0).
#   (3) geodesic/Euclidean stretch — how much the manifold curls (1.0 = flat, >1 = curved).
# Calibrated through the SAME geodesic pipeline: Gaussian-fills-space (flat), sphere (POS curv), swiss-roll (intrinsically
# FLAT despite looking curved — the discriminating control), hyperboloid (NEG curv).
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F, numpy as np, random
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); random.seed(0); np.random.seed(0); dev = torch.device('cuda')
MODEL = '/home/pokazge/models/Qwen3.6-27B'; LAYERS = [8, 16, 27, 35]; NCAP = 1200; K = 12

def twonn(X):                                                      # Facco intrinsic-dimension MLE
    X = X.float(); D = torch.cdist(X, X); D.fill_diagonal_(1e9)
    s, _ = D.sort(1); mu = (s[:, 1] / (s[:, 0] + 1e-9)).cpu().numpy()
    mu = mu[(mu > 1) & np.isfinite(mu)]
    return float(len(mu) / np.log(mu).sum())

def geo(X, k=K):                                                   # geodesic neg_mass + geodesic/euclid stretch
    X = X.float(); n = X.shape[0]; D = torch.cdist(X, X)
    nn = D.topk(k + 1, largest=False).indices[:, 1:]
    r = torch.arange(n).unsqueeze(1).expand(-1, k).reshape(-1)
    w = D[r, nn.reshape(-1)].cpu().numpy()
    G = csr_matrix((w, (r.cpu().numpy(), nn.reshape(-1).cpu().numpy())), shape=(n, n)); G = G.maximum(G.T)
    Dg = shortest_path(G, directed=False)
    fin = np.isfinite(Dg)
    if not fin.all(): Dg[~fin] = Dg[fin].max()                    # patch disconnected
    stretch = float(np.median((Dg[fin] + 1e-9) / (torch.cdist(X, X).cpu().numpy()[fin] + 1e-9)))
    Dg = torch.tensor(Dg, dtype=torch.float32); D2 = Dg ** 2
    J = torch.eye(n) - 1.0 / n; B = -0.5 * J @ D2 @ J; ev = torch.linalg.eigvalsh(B)
    pos = ev[ev > 0].sum(); neg = (-ev[ev < 0]).sum()
    return float(neg / (pos + neg + 1e-9)), stretch

print('=== CALIBRATION (geodesic pipeline; sphere/hyperboloid SHOULD be curved, swiss-roll FLAT) ===', flush=True)
gauss = torch.randn(NCAP, 16, device=dev)
nm, st = geo(gauss); print('gaussian R^16 (flat):   id=%.1f  geo_negmass=%.3f  stretch=%.2f' % (twonn(gauss), nm, st), flush=True)
u = torch.rand(NCAP, device=dev) * 2 * 3.14159; v = torch.acos(1 - 2 * torch.rand(NCAP, device=dev))
sph = torch.stack([torch.sin(v) * torch.cos(u), torch.sin(v) * torch.sin(u), torch.cos(v)], -1).to(dev)
nm, st = geo(sph); print('sphere S^2 (POS curv):  id=%.1f  geo_negmass=%.3f  stretch=%.2f' % (twonn(sph), nm, st), flush=True)
tt = torch.rand(NCAP, device=dev) * 3 * 3.14159 + 1.5; sr = torch.stack([tt * torch.cos(tt), torch.rand(NCAP, device=dev) * 21, tt * torch.sin(tt)], -1)
nm, st = geo(sr); print('swiss-roll (FLAT intr): id=%.1f  geo_negmass=%.3f  stretch=%.2f' % (twonn(sr), nm, st), flush=True)
z = torch.randn(NCAP, 4, device=dev) * 0.9; hb = torch.cat([torch.sqrt(1 + (z ** 2).sum(-1, keepdim=True)), z], -1)
nm, st = geo(hb); print('hyperboloid (NEG curv): id=%.1f  geo_negmass=%.3f  stretch=%.2f' % (twonn(hb), nm, st), flush=True)

TEXTS = [
    "The river remembered every stone it had ever touched, though it could never return to any of them. To flow was to forget and to carry forward at once.",
    "Import the libraries, define a function that accepts a dataframe, drops null rows, normalizes the numeric columns, and returns the cleaned result for modeling.",
    "Consider whether moral responsibility presupposes the ability to have done otherwise. If determinism holds, the agent could not have chosen differently, yet we attribute blame.",
    "The market opened sharply lower as investors digested the inflation print. Energy led the decline, financials lagged, and the dollar strengthened through the afternoon.",
    "She set the table for two, then sat alone and watched the candle gutter. The chair across held only the shape of an absence she never stopped setting a place for.",
    "A neural network learns by adjusting weights against the gradient of a loss surface. The geometry of that surface determines which solutions are reachable.",
    "Gather your ingredients: flour, butter, sugar, eggs. Cream the butter and sugar until pale. Fold in the dry ingredients gently or the crumb will turn tough.",
    "Memory is not a recording but a reconstruction, assembled fresh each time from fragments and inference, closer to imagining a past consistent with who we now need to be.",
    "The negotiations stalled when neither delegation would concede the disputed territory, and the ceasefire that had held for months began to fray along its oldest seams.",
    "Light from the dying star had traveled ten thousand years to reach the telescope, carrying news of an event that was, by every present reckoning, already ancient history.",
    "He practiced the same four bars until his fingers bled, convinced that mastery lived in the repetition itself and not in any sudden grace that might arrive unearned.",
    "The proof proceeds by contradiction: assume the set is finite, enumerate its elements, construct a number that differs from each, and derive the impossibility directly.",
]
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False

@torch.no_grad()
def reps(text):
    ids = tok(text, return_tensors='pt').input_ids.to(dev)
    out = model(ids, output_hidden_states=True)
    return {L: out.hidden_states[L][0] for L in LAYERS}

allr = [reps(t) for t in TEXTS]
print('=== LLM token-cloud geometry (pooled across %d texts) ===' % len(TEXTS), flush=True)
for L in LAYERS:
    X = torch.cat([r[L] for r in allr], 0)
    if X.shape[0] > NCAP: X = X[torch.randperm(X.shape[0])[:NCAP]]
    nm, st = geo(X)
    print('  layer %2d: intrinsic_dim=%.1f (ambient %d)  geo_negmass=%.3f  geo/euclid_stretch=%.2f' % (
        L, twonn(X), X.shape[1], nm, st), flush=True)
print('=== DIAG_DONE === (curved iff geo_negmass & stretch track the sphere/hyperboloid refs, NOT the gaussian/swiss-roll)', flush=True)
