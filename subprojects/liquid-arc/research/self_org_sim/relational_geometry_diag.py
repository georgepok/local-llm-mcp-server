# RELATIONAL GEOMETRY DIAGNOSTIC — representations are flat (manifold ~12d) and their token-dynamics are noise. But FGN's
# whole basis is that the geometry lives in RELATIONS, not points: the attention structure (who attends to whom), which for
# language is known to be hierarchical/hyperbolic (CURVED) even when embeddings are flat. So: measure delta-hyperbolicity
# (Gromov 4-point) of the attention-derived relational distances per layer. delta_norm -> 0 = tree-like/hyperbolic (curved);
# large = flat. Calibrated against a balanced TREE (curved, ~0), a 2D GRID (flat), and random points (Euclidean).
# Also contrast with the token-REPRESENTATION cosine-relational hyperbolicity (expected flat) to isolate relations vs points.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F, numpy as np, random
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); random.seed(0); np.random.seed(0); dev = torch.device('cuda')
MODEL = '/home/pokazge/models/Qwen3.6-27B'; LAYERS = [0, 2, 4, 6, 8, 10, 12, 14, 15]   # span the ~16 attention layers

def delta_hyp(D, samples=4000):                                    # normalized Gromov 4-point delta; ->0 = hyperbolic/curved
    D = np.asarray(D, dtype=np.float64); n = D.shape[0]; diam = D.max() + 1e-9; ds = []
    for _ in range(samples):
        i, j, k, l = random.sample(range(n), 4)
        s = sorted([D[i, j] + D[k, l], D[i, k] + D[j, l], D[i, l] + D[j, k]])
        ds.append((s[2] - s[1]) / 2.0)
    return float(np.mean(ds) / diam)

def graph_dist(W):                                                 # relational distances = shortest path on edge weights W (high weight=far)
    n = W.shape[0]; G = csr_matrix(W); G = G.maximum(G.T)
    D = shortest_path(G, directed=False); fin = np.isfinite(D)
    if not fin.all(): D[~fin] = D[fin].max()
    return D

print('=== CALIBRATION (delta_hyp: tree~0 curved, grid larger flat) ===', flush=True)
# balanced binary tree, 255 nodes, unit edges
import collections
N = 255; edges = []
for c in range(1, N):
    edges.append((c, (c - 1) // 2))
W = np.zeros((N, N));
for a, b in edges: W[a, b] = W[b, a] = 1.0
Dtree = graph_dist(W); print('tree (curved/hyperbolic):  delta_hyp=%.3f' % delta_hyp(Dtree), flush=True)
# 16x16 grid, L1 distances
g = 16; coords = np.array([[i, j] for i in range(g) for j in range(g)])
Dgrid = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1).astype(float)
print('grid 2D (flat):            delta_hyp=%.3f' % delta_hyp(Dgrid), flush=True)
rp = np.random.randn(256, 10); Drand = np.sqrt(((rp[:, None] - rp[None]) ** 2).sum(-1))
print('random R^10 (Euclid):      delta_hyp=%.3f' % delta_hyp(Drand), flush=True)

TEXTS = [
    "The committee debated for hours, but the chairman, sensing the room had already decided, called the vote before the dissenters could regroup their fractured argument into anything coherent.",
    "To build the model you must first preprocess the data, then define the architecture, then choose an optimizer, and only after all of that does training begin in earnest.",
    "She remembered the house as enormous, with endless corridors, but returning as an adult she found it small, the corridors short, the rooms shrunken by the simple fact of her own growth.",
    "The theorem follows from three lemmas, each of which depends in turn on the boundedness assumption, which itself is justified only because the underlying space is compact.",
    "Rain moved across the valley in visible sheets, darkening the far ridge first, then the orchard, then the road, until at last it reached the house and drummed against the tin roof.",
    "Every empire tells itself a story about why its dominion is just, and the story is always most elaborate precisely when the dominion has begun, quietly, to slip from its grasp.",
    "The algorithm sorts the array by repeatedly selecting the smallest remaining element and appending it to the output, a method simple to reason about but quadratic and slow at scale.",
    "Grief does not move in stages so much as in tides, receding far enough that you mistake it for gone, then returning without warning to pull the ground out from under your feet.",
    "Photosynthesis converts light into chemical energy, splitting water to release oxygen and fixing carbon into sugar, the quiet engine on which nearly every other living thing depends.",
    "He had rehearsed the apology a hundred times, but standing at her door the words rearranged themselves into something smaller and truer than the speech he had meant to give.",
    "A currency holds value only so long as enough people agree it does; the paper is a shared fiction, and a bank run is simply the moment the fiction briefly stops being believed.",
    "The glacier had carved the valley over ten thousand years, indifferent and patient, leaving behind moraines and erratics as the only signature of a force that no longer existed.",
]
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False

with torch.no_grad():
    _o = model(tok('hello world this is a test', return_tensors='pt').input_ids.to(dev), output_attentions=True, output_hidden_states=True)
nA = len(_o.attentions) if _o.attentions is not None else 0; nH = len(_o.hidden_states)
print('model returns: attentions=%d layers, hidden_states=%d' % (nA, nH), flush=True)
if nA == 0: print('!! NO ATTENTIONS RETURNED — relational read via attention not directly observable on this model', flush=True)
LAYERS = [L for L in LAYERS if L < nA] or ([max(1, nA // 4), nA // 2, 3 * nA // 4, nA - 1] if nA else [])
print('using layers:', LAYERS, flush=True)

@torch.no_grad()
def attn_and_hidden(text):
    ids = tok(text, return_tensors='pt').input_ids.to(dev)
    out = model(ids, output_attentions=True, output_hidden_states=True)
    return ({L: out.attentions[L][0].mean(0).float() for L in LAYERS},        # mean-head attention [n,n]
            {L: out.hidden_states[L][0].float() for L in LAYERS})

A_all, H_all = [], []
for t in TEXTS:
    a, h = attn_and_hidden(t); A_all.append(a); H_all.append(h)

print('=== RELATIONAL geometry: ATTENTION structure (who-attends-to-whom) ===', flush=True)
for L in LAYERS:
    hs = []
    for a in A_all:
        S = (a[L] + a[L].t()) / 2                                   # symmetrize attention
        W = (-(S + 1e-6).log()).cpu().numpy()                       # strong attention = short edge
        np.fill_diagonal(W, 0.0); W[W < 0] = 0.0
        hs.append(delta_hyp(graph_dist(W)))
    print('  layer %2d: attention-relational delta_hyp=%.3f' % (L, sum(hs) / len(hs)), flush=True)

print('=== CONTRAST: token-REPRESENTATION cosine-relational geometry (expected flat) ===', flush=True)
for L in LAYERS:
    hs = []
    for h in H_all:
        X = F.normalize(h[L], dim=-1); Dc = (1 - X @ X.t()).clamp(min=0).cpu().numpy()
        hs.append(delta_hyp(Dc))
    print('  layer %2d: representation-relational delta_hyp=%.3f' % (L, sum(hs) / len(hs)), flush=True)
print('=== DIAG_DONE === (attention delta_hyp near TREE => curved relations a read should transfer; near GRID => flat everywhere)', flush=True)
