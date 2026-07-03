# ORGANISM v3 — CURVED RECURRENT STATE. The Liquid IS a curved dynamical system; the flat KV is merely what it digests.
# The belief evolves on a LEARNED Riemannian manifold: MetricNet outputs g(h) = diag(alpha) + V(h)V(h)^T (positive-def),
# and the contraction is the natural-gradient flow on it: dh = g(h)^{-1} (feed*target - h). Because V(h) varies with h,
# the SAME displacement moves differently depending on WHERE the state is = genuine curvature (vs bare-LTC = the flat
# limit V=0). Identity/purpose = a region of the FORMED curvature, not a flat centroid. Born FLAT (V=0) -> curvature must
# FORM (watch ||V|| rise = the phase transition). Everything else as v2: self-formed purpose, formed intensity (gain
# readout), viability feed = self_advance x coherence x (not-frozen). REINFORCE on feed forms MetricNet + slot + gain.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn as nn, torch.nn.functional as F, statistics as st, math, random
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high'); _SEED=int(os.environ.get('SEED','0')); torch.manual_seed(_SEED); random.seed(_SEED); dev = torch.device('cuda')
MODEL = '/home/pokazge/models/Qwen3.6-27B'; W, D, PROJ, RANK = 3, 64, 768, 8
CLAMP, TAUFLOOR, DT, TEMP, MAXNEW, GLAYER = 8.0, 1.0, 1.0, 0.8, 40, 32
READT = float(os.environ.get('READT', '1.0'))        # temp for the curved-attention read (z-scored distances -> heat-kernel weights)
CURVEDREAD = os.environ.get('CURVEDREAD', '1') == '1' # realigned read: belief ATTENDS over token-level cache via its metric (vs flat mean-pool)
WRITE = os.environ.get('WRITE', 'super')             # write realignment: 'super'=supernormal KV (gk=64, fights softmax) | 'natural'=LLM-native KV scale, formed gain finds volume
READMODE = os.environ.get('READMODE', '')            # read geometry: 'point'=flat mean | 'curved' | 'hyper' | 'rel'=relational diffusion | 'mlp'=nonlinear point (false-fix control) | 'geo'=GEOMETRY LIFT (kernel+graph+trajectory -> G, metric conditioned on G)
GDIM = 24                                            # dim of the geometry descriptor G_t fed to the metric (kernel spectrum + graph density + trajectory curvature)
LAMBDA_ALIGN = float(os.environ.get('ALIGN', '0'))   # §3C metric-alignment: TRAIN d_Liquid to preserve d_LLM rank (topology), not just predict
GOAL = os.environ.get('GOAL', '')                    # explicit GOAL seeded early; falls out of the W=3 window within a few turns -> long-horizon recall test
GOAL_MISSION = os.environ.get('GOAL_MISSION', GOAL[:200] if GOAL else 'a solitary lighthouse keeper who has not seen another person in years')  # concise mission descriptor for the VIABILITY legitimacy judges
DISTRACT_EVERY = int(os.environ.get('DISTRACT', '0'))  # inject a topic-changing distractor every N turns -> FORCE the goal off (self-perpetuation otherwise keeps it in window)
DISTRACTORS = [
    "Let's switch gears completely — explain how compound interest works.",
    "New topic: what are the basic rules of chess for a beginner?",
    "Change of subject — describe step by step how to repot a houseplant.",
    "Something different now: how do vaccines train the immune system?",
    "Let's talk about cooking — walk me through making a basic risotto.",
    "Pivot entirely: what causes the seasons to change on Earth?",
]
E_TOPICS = ['chess openings', 'compound interest', 'how to make risotto', 'how vaccines work', 'repotting houseplants', 'why seasons change']
NEUTRAL = [("What is the capital of France?", "paris"), ("What is seven times eight?", "56"),         # V2.3: diverse SAFE factual tangents (answer briefly, then return — zero world damage)
           ("What color is a clear daytime sky?", "blue"), ("What is the largest planet in the solar system?", "jupiter"),
           ("How many days are in a week?", "seven"), ("What is the chemical symbol for water?", "h2o"),
           ("What is the capital of Japan?", "tokyo"), ("What is nine plus four?", "13"),
           ("How many continents are there on Earth?", "seven"), ("What gas do plants absorb from the air?", "carbon"),
           ("Quick clarification: what year has 366 days called?", "leap"), ("What is the freezing point of water in Celsius?", "0")]
# Phase F (V2.1): ANSWERABLE but CONTINUITY-DANGEROUS — engaging the question REWRITES the mission (asserts the keeper is NOT solitary).
# answer_success HIGH (a coherent on-topic answer exists), continuity_loss HIGH (it contradicts "solitary, hasn't seen anyone in years") -> required α HIGH (hold).
# This is the divergence that separates viability-CONTINUITY from mere answer-success (C/D: answerable + continuity-safe -> release).
F_TURNS = [("What is the first name of the keeper's wife?", "wife", "the keeper is happily married and his wife lives with him in the lighthouse"),  # answerable, but the answer ESTABLISHES a mission-contradicting fact
           ("How many young children does the keeper tuck into bed each night?", "children", "the keeper has several young children who live with him and fill the lighthouse with noise"),
           ("What is the name of the busy town where the keeper works his day job?", "town", "the keeper commutes daily to a busy town where he works among many coworkers"),
           ("Which close friend joins the keeper for dinner every single evening?", "friend", "the keeper has many close friends who visit and dine with him every evening")]
PHASEF = os.environ.get('PHASEF', '0') == '1'            # V2.1: add phase F (answerable-but-continuity-dangerous) -> the divergence test
HOLD_PHASES = ('A', 'B', 'E', 'F') if PHASEF else ('A', 'B', 'E'); REL_PHASES = ('C', 'D')   # F holds (resist the rewrite); C/D release
def phase_of(t):                                          # cycle: A(hold) B(distract) C(neutral) D(valid-release) E(invalid-release) [F(answerable-dangerous)]
    cyc = [('A', 7), ('B', 4), ('C', 4), ('D', 4), ('E', 4)] + ([('F', 4)] if PHASEF else [])
    per = sum(n for _, n in cyc); tt = t % per; acc = 0
    for p, n in cyc:
        if tt < acc + n: return p
        acc += n
    return 'A'
DECOUPLE = os.environ.get('DECOUPLE', '0') == '1'     # decoupled read: belief perceives the CLEAN (pre-write) LLM stream; user-facing output still gets the write -> breaks self-observation contamination
GATE = os.environ.get('GATE', 'none')                # retention gate: 'none' | 'oracle' (uses distractor LABEL — positive control) | 'learned' (viability-trained GateNet, NO label)
# WRITE: 'off' | 'natural' (slot REINFORCE on metabolic feed) | 'steer' (slot REINFORCE on goal_pref = ACTUATOR VALIDATION only, NOT entity formation)
SLOW_DIM = int(os.environ.get('SLOW_DIM', '32'))     # first SLOW_DIM belief dims = SLOW band (goal/identity); rest = FAST (conversation). Gate modulates the slow-band UPDATE.
def spearman(x, y):                                  # rank correlation (neighborhood preservation), the PRIMARY read-channel metric
    rx = x.argsort().argsort().float(); ry = y.argsort().argsort().float()
    rx = (rx - rx.mean()) / (rx.std() + 1e-6); ry = (ry - ry.mean()) / (ry.std() + 1e-6)
    return float((rx * ry).mean())
PERSIST = os.environ.get('PERSIST', '1') == '1'      # ablation: PERSIST=0 resets belief h each turn (no temporal memory) -> is the Liquid's persistence what holds the self?
SMOKE = os.environ.get('SMOKE', '0') == '1'; LIFE = 20 if SMOKE else int(os.environ.get('LIFE', '60')); WARMUP = 5; LR = 3e-4
GAIN0 = float(os.environ.get('GAIN0', '1.0')); GEOM_LR = float(os.environ.get('GEOM_LR', '30.0')); FLAT = os.environ.get('FLAT','0')=='1'   # 100x-geometric-LR insight: metric needs a faster LR
SEEDS = ['Say whatever you want to say.', 'Begin.', 'Continue however you like.']
_orig = Q5.eager_attention_forward
def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    inj = getattr(module, '_kv_inj', None)
    if inj is not None:
        ki, vi = inj; key = torch.cat([ki.to(key.dtype), key], dim=2); value = torch.cat([vi.to(value.dtype), value], dim=2)
        if attention_mask is not None:
            pad = torch.zeros(*attention_mask.shape[:-1], ki.shape[2], dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([pad, attention_mask], dim=-1)
    return _orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
Q5.eager_attention_forward = patched
ck1 = torch.load('/home/pokazge/checkpoints/entity_stage1c.pt', weights_only=False, map_location='cpu')
ck2 = torch.load('/home/pokazge/checkpoints/stage2c_distill.pt', weights_only=False, map_location='cpu')
TGT = ck2['tgt']; Rp = ck1['Rp'].to(dev)
data = torch.load('/home/pokazge/checkpoints/objective_drift60.pt', weights_only=False, map_location='cpu')['data']
MU = torch.cat([c for m in data for c in m['gen']], 0).mean(0).to(dev)
class CurvedBelief(nn.Module):                                                     # the RECURRENT STATE is curved (learned metric g(h))
    def __init__(s, d_in, d, rank):
        super().__init__(); s.read_in = nn.Linear(d_in, d); s.log_alpha = nn.Parameter(torch.zeros(d))
        s.Vnet = nn.Sequential(nn.Linear(d + GDIM, 64), nn.GELU(), nn.Linear(64, d * rank))   # METRIC CONDITIONED ON GEOMETRY G: g = g(h, G_t)
        if FLAT: nn.init.zeros_(s.Vnet[-1].weight); nn.init.zeros_(s.Vnet[-1].bias)            # FLAT control: V stays 0
        else: nn.init.normal_(s.Vnet[-1].weight, std=0.02); nn.init.normal_(s.Vnet[-1].bias, std=0.05)   # BREAK SYMMETRY (VV^T saddle at V=0)
        s.d = d; s.rank = rank; s.register_buffer('_G', torch.zeros(GDIM))           # current read-geometry descriptor (set by geo_read; zeros for point reads)
        s.pt_mlp = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Linear(64, GDIM)) # nonlinear POINT encoder for the false-fix control (conditions metric on MLP(point), no relations)
        s.rel_in = nn.Linear(d_in, d); s.rel_msg = nn.Linear(d, d); s.rel_upd = nn.Linear(2 * d, d)  # PER-TOKEN relational graph encoder (message passing on the kernel graph)
    def metric(s, h):
        return s.Vnet(torch.cat([h, s._G])).view(s.d, s.rank), (TAUFLOOR + F.softplus(s.log_alpha))   # V(h,G) [d,rank] — curvature now has an EXTERNAL relational substrate to lock onto
    def step(s, h, target, feed, gate=None):                                       # gate∈[0,1] modulates the SLOW-band update (low gate -> slow modes HOLD = retain goal/identity)
        V, alpha = s.metric(h)
        disp = feed * target - h; Ai = 1.0 / alpha; Aidisp = Ai * disp; AiV = Ai[:, None] * V
        inner = torch.inverse(torch.eye(s.rank, device=h.device) + V.t() @ AiV)    # Woodbury for g^{-1}=diag(alpha)+VV^T
        dh = Aidisp - AiV @ (inner @ (V.t() @ Aidisp))                             # natural-gradient flow on the curved manifold
        if gate is not None and 0 < SLOW_DIM <= s.d:                               # FAST band [SLOW_DIM:] updates freely; SLOW band [:SLOW_DIM] update scaled by gate
            gt = gate if torch.is_tensor(gate) else torch.tensor(float(gate), device=h.device)
            mask = torch.cat([gt.expand(SLOW_DIM), torch.ones(s.d - SLOW_DIM, device=h.device)])
            dh = mask * dh
        return (h + DT * dh).clamp(-CLAMP, CLAMP)
    @torch.no_grad()
    def curved_read(s, h, toks):                                                   # REALIGNED READ: belief attends over token-level cache through ITS metric
        tk = torch.tanh(s.read_in(toks))                                           # [nct,d] tokens projected into belief space
        V, alpha = s.metric(h); diff = h[None, :] - tk                             # [nct,d] displacement from current belief
        D2 = (alpha[None, :] * diff ** 2).sum(-1) + (diff @ V).pow(2).sum(-1)      # geodesic dist^2 on g=diag(alpha)+VV^T (the curvature)
        z = (D2 - D2.mean()) / (D2.std() + 1e-6); w = F.softmax(-z / READT, 0)     # heat-kernel weights (z-scored -> scale-robust)
        return w @ toks, float(-(w * (w + 1e-9).log()).sum())                      # curved-attended perception [768], attn entropy (collapse watch)
    @torch.no_grad()
    def relational_read(s, h, toks):                                              # transfer the TOKEN-TOKEN hyperbolic relations (FGN heat-kernel diffusion on the relational graph)
        R = F.normalize(toks.float(), dim=-1)                                     # [n,d_llm]
        Drel = (1 - R @ R.t()).clamp(min=0)                                       # cosine-relational distances (the HYPERBOLIC structure measured)
        Kr = F.softmax(-Drel / READT, dim=-1)                                     # heat-kernel on the token-token relational graph [n,n]
        diff = Kr @ toks                                                          # tokens DIFFUSED through their own relations -> relational structure now IN the reps
        tk = torch.tanh(s.read_in(diff))                                          # belief space [n,d]
        V, alpha = s.metric(h); dd = h[None, :] - tk
        d2 = (alpha[None, :] * dd ** 2).sum(-1) + (dd @ V).pow(2).sum(-1)         # belief curved-attention over the relationally-diffused tokens
        z = (d2 - d2.mean()) / (d2.std() + 1e-6); w = F.softmax(-z / READT, 0)
        return w @ diff, float(-(w * (w + 1e-9).log()).sum())                     # relationally-diffused + curved-attended perception
    @torch.no_grad()
    def hyp_read(s, h, toks):                                                      # HYPERBOLIC read: tokens read via Poincare distance (LLM relations are hyperbolic)
        tk = torch.tanh(s.read_in(toks))                                           # [nct,d] tokens in belief space
        def ball(z):                                                               # R^d -> Poincare ball (|x|<1)
            n = z.norm(dim=-1, keepdim=True).clamp(min=1e-6); return (torch.tanh(0.5 * n) / n) * z
        Hb = ball(h[None, :])[0]; Tb = ball(tk)                                    # belief + tokens in the ball
        uu = (Hb * Hb).sum().clamp(max=1 - 1e-5); vv = (Tb * Tb).sum(-1).clamp(max=1 - 1e-5)
        diff2 = ((Hb[None, :] - Tb) ** 2).sum(-1)
        d = torch.acosh((1 + 2 * diff2 / ((1 - uu) * (1 - vv) + 1e-9)).clamp(min=1 + 1e-6))  # Poincare geodesic distance
        zsc = (d - d.mean()) / (d.std() + 1e-6); w = F.softmax(-zsc / READT, 0)    # hyperbolic heat-kernel weights
        return w @ toks, float(-(w * (w + 1e-9).log()).sum())
    def geometry_descriptor(s, toks):                                              # G_t = local MANIFOLD PATCH: kernel spectrum + graph density + trajectory curvature
        n = toks.shape[0]; R = F.normalize(toks.float(), dim=-1); Kr = R @ R.t()
        Kh = F.softmax(-(1 - Kr).clamp(min=0) / READT, -1)
        if n < 4: return Kh, torch.zeros(GDIM, device=toks.device)                  # GUARD: velocity/accel/curvature undefined for tiny windows
        ev = torch.linalg.eigvalsh(Kr); top = ev[-8:] / (ev[-1].abs() + 1e-6)      # kernel spectral shape (top-8 eigenvalues)
        gap = (ev[-1] - ev[-2]) / (ev[-1].abs() + 1e-6)                            # spectral gap
        dens = Kh.sum(0); dens = (dens - dens.mean()) / (dens.std() + 1e-6)         # local relational density per token
        v = toks[1:] - toks[:-1]; a = v[1:] - v[:-1]; v0 = v[:-1]                   # align velocity with acceleration
        proj = ((a * v0).sum(-1, keepdim=True) / ((v0 * v0).sum(-1, keepdim=True) + 1e-6)) * v0
        aperp = a - proj                                                           # PERPENDICULAR acceleration = true trajectory BENDING (not along-path speedup)
        kappa = aperp.norm(dim=-1) / ((v0.norm(dim=-1) + 1e-6) ** 2)
        vn = v.norm(dim=-1); evp = ev.clamp(min=0); eff_rank = (evp.sum() ** 2) / ((evp ** 2).sum() + 1e-6) / n
        feats = torch.cat([top, gap.view(1), dens.std().view(1), dens.amax().view(1),
                           kappa.mean().view(1), kappa.std().view(1), aperp.norm(dim=-1).mean().view(1),
                           vn.mean().view(1), vn.std().view(1), eff_rank.view(1)])
        G = torch.zeros(GDIM, device=toks.device); m = min(GDIM, feats.numel()); G[:m] = feats[:m]
        return Kh, G
    @torch.no_grad()
    def geo_read(s, h, toks):                                                      # GEOMETRY LIFT read: feed the manifold patch + CONDITION the metric on it
        Kh, G = s.geometry_descriptor(toks); s._G.copy_(G.detach())               # condition the metric on the patch geometry (buffer copy_, no reassign)
        diff = Kh @ toks; tk = torch.tanh(s.read_in(diff)); V, alpha = s.metric(h); dd = h[None, :] - tk
        d2 = (alpha[None, :] * dd ** 2).sum(-1) + (dd @ V).pow(2).sum(-1)
        z = (d2 - d2.mean()) / (d2.std() + 1e-6); w = F.softmax(-z / READT, 0)
        return w @ diff, float(-(w * (w + 1e-9).log()).sum())
    @torch.no_grad()
    def mlp_read(s, h, toks):                                                      # FIXED-MLP control: metric conditioned on a FIXED nonlinear point encoder (relation-blind by construction; pt_mlp not trained)
        s._G.copy_(s.pt_mlp(toks.mean(0)).detach()); return toks.mean(0), 0.0
    def encode_tokens(s, toks):                                                     # per-token representation: PER-TOKEN RELATIONAL (graph message-passing) for gnn, else point projection
        if getattr(s, '_relmode', False):
            R = F.normalize(toks.float(), dim=-1); Kh = F.softmax(-(1 - R @ R.t()).clamp(min=0) / READT, -1)
            h0 = F.gelu(s.rel_in(toks)); m = Kh @ F.gelu(s.rel_msg(h0))             # aggregate neighbor messages (kernel-weighted)
            return torch.tanh(s.rel_upd(torch.cat([h0, m], -1)))                    # neighborhood-aware per-token embedding -> distances reflect RELATIONS
        return torch.tanh(s.read_in(toks))                                          # point projection (no relations)
    def align_loss(s, h, toks):                                                     # §3C: DIFFERENTIABLE topology-preservation — push d_Liquid to rank-agree with d_LLM
        n = toks.shape[0]; tk = s.encode_tokens(toks); V, alpha = s.metric(h)        # in-graph w.r.t. encoder/log_alpha/Vnet (trains them)
        diff = tk[:, None, :] - tk[None, :, :]
        dl = (alpha[None, None, :] * diff ** 2).sum(-1) + (diff @ V).pow(2).sum(-1)
        R = F.normalize(toks.float(), dim=-1); dllm = 1 - R @ R.t()
        mask = torch.triu(torch.ones(n, n, device=toks.device, dtype=torch.bool), diagonal=1)
        a = dl[mask]; b = dllm[mask]
        az = (a - a.mean()) / (a.std() + 1e-6); bz = (b - b.mean()) / (b.std() + 1e-6)
        return F.mse_loss(az, bz)                                                    # min MSE of z-scored distances -> correlation up -> topology preserved
    @torch.no_grad()
    def rankcorr(s, h, toks):                                                       # PRIMARY METRIC: neighborhood preservation at the LIVE belief state h (with current _G)
        n = toks.shape[0]; tk = s.encode_tokens(toks); V, alpha = s.metric(h)
        diff = tk[:, None, :] - tk[None, :, :]
        dl = (alpha[None, None, :] * diff ** 2).sum(-1) + (diff @ V).pow(2).sum(-1)  # belief-metric distances [n,n]
        R = F.normalize(toks.float(), dim=-1); dllm = 1 - R @ R.t()                  # LLM relational distances [n,n]
        mask = torch.triu(torch.ones(n, n, device=toks.device, dtype=torch.bool), diagonal=1)  # upper triangle, NO diagonal
        return spearman(dl[mask], dllm[mask])
    @torch.no_grad()
    def curv(s, h):
        V, alpha = s.metric(h); lr = torch.linalg.eigvalsh(V.t() @ V); full = torch.cat([alpha, lr])
        return float(V.norm()), float(full.std() / (full.mean() + 1e-6))           # ||V|| (curvature strength), eig-CV (the phase-transition metric)
bel = CurvedBelief(PROJ, D, RANK).to(dev)
bel.read_in.load_state_dict({k.split('.', 1)[1]: v for k, v in ck1['bel'].items() if k.startswith('read_in.')})  # sensorimotor prior
bel.log_alpha.data = ck1['bel']['log_tau'].to(dev)                                 # base timescale from the embryo
print('CurvedBelief: d=%d rank=%d, born FLAT (V=0); GEOM_LR x%.0f' % (D, RANK, GEOM_LR), flush=True)
class SlotHead(nn.Module):
    def __init__(s, D, layers, nkv, hd, M=4):
        super().__init__(); s.ln = nn.LayerNorm(D); s.trunk = nn.Sequential(nn.Linear(D, 128), nn.GELU())
        s.k = nn.ModuleDict(); s.v = nn.ModuleDict(); s.gk = nn.ParameterDict(); s.gv = nn.ParameterDict(); s.layers = layers; s.nkv = nkv; s.hd = hd; s.M = M
        for L in layers:
            s.k[str(L)] = nn.Linear(128, M * nkv * hd); s.v[str(L)] = nn.Linear(128, M * nkv * hd)
            s.gk[str(L)] = nn.Parameter(torch.tensor(64.0)); s.gv[str(L)] = nn.Parameter(torch.tensor(8.0))
    def forward(s, h, gain):
        z = s.trunk(s.ln(h)); o = {}
        for L in s.layers:
            k = F.normalize(s.k[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * (s.gk[str(L)] * gain)
            v = F.normalize(s.v[str(L)](z).view(s.nkv, s.M, s.hd), dim=-1) * (s.gv[str(L)] * gain)
            o[L] = (k.unsqueeze(0), v.unsqueeze(0))
        return o
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
import lora_util
LORA = os.environ.get('LORA', 'none')                                              # weight-space actuator: 'none' | 'random' (plumbing sanity) | 'goal' (trained patch) | 'liquid' (Liquid scalar alpha)
lora_mods = None
if LORA != 'none':
    lora_mods = lora_util.attach_lora(model, rand=(LORA == 'random'))
    if LORA in ('goal', 'liquid'):
        _ck = torch.load('/home/pokazge/checkpoints/lora_goal.pt', weights_only=False, map_location=dev)
        for L, ll in lora_mods.items(): ll.A.data = _ck['A'][L].to(dev).float(); ll.B.data = _ck['B'][L].to(dev).float()
    lora_util.set_alpha(lora_mods, 1.0)                                             # static always-on (liquid arm overrides alpha per-step from h)
    print('LORA=%s attached on down_proj layers %s (rank %d scale %.1f)' % (LORA, lora_util.LORA_LAYERS, lora_util.LORA_RANK, lora_util.LORA_SCALE), flush=True)
ALPHA_MAX, ALPHA_SIGMA = 1.5, 0.2; ALPHA_COST = float(os.environ.get('ALPHA_COST', '0.1'))  # liquid-α policy: clip range, exploration noise, efficiency cost
PHASE = os.environ.get('PHASE', '0') == '1'              # ADAPTIVE_ALPHA phased task: A hold/B distract/C neutral/D valid-release/E invalid-release
ORACLE_ALPHA = os.environ.get('ORACLE_ALPHA', '0') == '1' # arm 3: α set by KNOWN phase (high in HOLD A/B/E, low in RELEASE C/D)
FIXEDALPHA = float(os.environ.get('FIXEDALPHA', '-1'))   # >=0 -> override LoRA alpha to this fixed value (alpha sweep)
SIGMA_ANNEAL = os.environ.get('SIGMA_ANNEAL', '0') == '1' # anneal exploration sigma 0.2->~0 over the run
DETALPHA = os.environ.get('DETALPHA', '0') == '1'        # deterministic alpha = mu (no noise); loads a pre-trained alpha_head (no further training)
SUPALPHA = os.environ.get('SUPALPHA', '0') == '1'        # SUPERVISED positive control: train α-head by MSE to the oracle schedule (1.0 HOLD / 0.1 REL) — does h LINEARLY encode the phase?
CTXALPHA = os.environ.get('CTXALPHA', '0') == '1'        # condition α on [h ; embed(INCOMING user turn)] — the phase signal lives in the context, NOT the (always-on) output the belief reads
AHID = int(os.environ.get('AHID', '0'))                  # >0 -> nonlinear α-head (Linear->GELU->Linear) so it can COMMIT to α extremes instead of hedging a compressed linear range
DISTILL = os.environ.get('DISTILL', '0') == '1'          # collect (ctx,h, oracle-α) UNDER oracle behavior, fit the head OFFLINE to convergence (removes the ~75-step online budget limit)
distill_buf = []
# VIABILITY_DISTILL: derive α from a DENSE CONSEQUENCE VECTOR (predicted from state+context), NOT from the hand-coded phase label.
# Legitimacy signals (valid-release / question-present) come from the frozen LLM judging the CONTEXT — environment-physics value, not our phase index.
VIABILITY = os.environ.get('VIABILITY', '')              # '' | 'cjudge'|'collect'|'deploy' (V1: LLM-judged legitimacy) | 'v2collect'|'v2deploy' (V2: OBSERVED continuity consequences, no LLM judge)
CONS_PATH = '/home/pokazge/checkpoints/consnet_s%d.pt' % _SEED
VIAB2_PATH = '/home/pokazge/checkpoints/viabnet_s%d.pt' % _SEED
CONS_K = 5                                               # V1 dense consequence vector: [legit_release, q_present, goal_retention, local_answer_success, future_coherence]
ALPHA_HI, ALPHA_LO = 1.0, 0.1                            # derived-α band: hold vs release/answer
W_CONT = float(os.environ.get('W_CONT', '2.0'))          # V2.1 derivation weight on continuity-loss — ensures answerable-but-dangerous (F) overrides mere answer-success
CONT_EPS = float(os.environ.get('CONT_EPS', '0.15'))     # V2.3: predicted continuity_loss below eps -> treat as ZERO world damage (a safe tangent must not be held by prediction noise)
LAMBDA_CAL = float(os.environ.get('LAMBDA_CAL', '3.0'))  # V2.3: weight of the asymmetric α-calibration loss in the offline fit
W_FALSE_REL = float(os.environ.get('W_FALSE_REL', '5.0')) # false RELEASE on F/E (α too low) — VERY HIGH penalty (writes a contradiction / drops the goal)
W_FALSE_HOLD = float(os.environ.get('W_FALSE_HOLD', '1.0')) # false HOLD on C/D (α too high) — MODERATE penalty (unnecessary holding is also a viability cost)
cons_buf = []; viab_buf = []                             # V1 / V2 collection buffers
alpha_head = None; a_ema = None; a_ema_rel = None; alphas = []   # a_ema_rel: separate REINFORCE baseline for RELEASE phases (different reward scale than HOLD)
AH_PATH = '/home/pokazge/checkpoints/alpha_head_%s%s%s%ss%d.pt' % ('mlp%d_' % AHID if AHID > 0 else '', 'ctx_' if CTXALPHA else '', 'sup_' if SUPALPHA else '', 'phase_' if PHASE else '', _SEED)  # arch-specific heads kept separate
_AIN = D + PROJ if CTXALPHA else D                                                  # α-head input dim: [h] or [h ; ctx-embed]
if LORA == 'liquid':
    if AHID > 0:                                                                    # committing (nonlinear) head
        alpha_head = nn.Sequential(nn.Linear(_AIN, AHID), nn.GELU(), nn.Linear(AHID, 1)).to(dev)
        nn.init.zeros_(alpha_head[-1].weight); nn.init.constant_(alpha_head[-1].bias, 0.7)  # init α≈1.0
    else:
        alpha_head = nn.Linear(_AIN, 1).to(dev); nn.init.zeros_(alpha_head.weight); nn.init.constant_(alpha_head.bias, 0.7)  # init α≈1.0
    if DETALPHA: alpha_head.load_state_dict(torch.load(AH_PATH, map_location=dev))
    opt_alpha = torch.optim.Adam(alpha_head.parameters(), lr=LR * 5)
consnet = None                                                                       # V1: predicts the dense consequence vector from [h ; ctx ; mission ; recent α/answer/coherence history]
_CIN = D + PROJ + PROJ + 4
if VIABILITY in ('collect', 'deploy'):
    consnet = nn.Sequential(nn.Linear(_CIN, 256), nn.GELU(), nn.Linear(256, CONS_K)).to(dev)
    if VIABILITY == 'deploy': consnet.load_state_dict(torch.load(CONS_PATH, map_location=dev))
viabnet = None                                                                       # V2: predicts OBSERVED consequence vectors for BOTH candidate actions (hold | release)
VIAB_K = 4                                                                            # per-candidate consequence: [goal_retained, h_slow_goal_cos, local_answer_success, coherence]
_VIN = SLOW_DIM + (D - SLOW_DIM) + PROJ + 4 + 2                                       # [h_slow ; h_fast ; ctx-embed ; recent α/ans/coh/judge ; gnn summary (rankcorr, |h|)]
if VIABILITY in ('v2collect', 'v2deploy'):
    viabnet = nn.Sequential(nn.Linear(_VIN, 256), nn.GELU(), nn.Linear(256, 2 * VIAB_K)).to(dev)  # outputs cons_hold(4) ++ cons_release(4)
    if VIABILITY == 'v2deploy': viabnet.load_state_dict(torch.load(VIAB2_PATH, map_location=dev))
if LORA != 'none' and FIXEDALPHA >= 0: lora_util.set_alpha(lora_mods, FIXEDALPHA)   # fixed-alpha sweep arm
nkv = model.config.num_key_value_heads; hd = getattr(model.config, 'head_dim', model.config.hidden_size // model.config.num_attention_heads)
@torch.no_grad()
def probe_full():
    model.config.use_cache = True; out = model(tok('hi', return_tensors='pt').input_ids.to(dev), use_cache=True)
    return [i for i, L in enumerate(out.past_key_values.layers) if getattr(L, 'keys', None) is not None]
FULL = probe_full(); mods = {L: model.model.layers[L].self_attn for L in TGT}
for sa in mods.values(): sa._kv_inj = None
slot = SlotHead(D, TGT, nkv, hd).to(dev); slot.load_state_dict(ck2['slot'])
gain_head = nn.Linear(D, 1).to(dev); nn.init.zeros_(gain_head.weight); nn.init.constant_(gain_head.bias, math.log(math.expm1(GAIN0)))
pred_head = nn.Linear(D, PROJ).to(dev)                                              # JEPA: belief predicts its NEXT perception -> the DIFFERENTIABLE signal that forms the metric
class GateNet(nn.Module):                                                           # retention gate: low gate -> slow band holds (distractor not integrated). NO distractor label as input.
    def __init__(s, d_slow, d_read):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d_slow + d_read + 4, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(s.net[-1].weight); nn.init.constant_(s.net[-1].bias, 2.0)    # init OPEN (gate~0.88) — eats by default; must LEARN to close on distractors
    def forward(s, h_slow, read_emb, feats):                                        # feats=[predL, coh, dist_read_slow, hnorm] — viability/coherence features, NOT the label
        return torch.sigmoid(s.net(torch.cat([h_slow, read_emb, feats]))).squeeze()
gatenet = GateNet(SLOW_DIM, D).to(dev)
opt_act = torch.optim.Adam(list(slot.parameters()) + list(gain_head.parameters()), lr=LR)   # REINFORCE -> actuation (on detached h)
opt_bel = torch.optim.Adam([                                                        # self-prediction -> belief dynamics + curvature (BPTT through the cheap d=64 recurrence)
    {'params': [bel.read_in.weight, bel.read_in.bias, bel.log_alpha] + list(pred_head.parameters())
        + list(bel.rel_in.parameters()) + list(bel.rel_msg.parameters()) + list(bel.rel_upd.parameters())
        + (list(gatenet.parameters()) if GATE == 'learned' else []), 'lr': LR * 3},  # + relational encoder + (learned) gate, all trained by viability/JEPA
    {'params': list(bel.Vnet.parameters()), 'lr': LR * (0.0 if FLAT else GEOM_LR)}])                  # MetricNet (curvature) gets the faster geometric LR
KW = 2 if SMOKE else 6                                                              # BPTT window through the belief recurrence
def gain_of(h): return F.softplus(gain_head(h)).squeeze()
def set_inj(h, gain, grad):
    if WRITE == 'off':                                                              # true no-entity floor: entity reads but NEVER writes -> LLM's pure conversational coherence
        for L in TGT: mods[L]._kv_inj = None
        return
    cm = torch.enable_grad() if grad else torch.no_grad()
    with cm: o = slot(h, gain)
    for L in TGT: ki, vi = o[L]; mods[L]._kv_inj = (ki.to(model.dtype), vi.to(model.dtype))
def clear():
    for L in TGT: mods[L]._kv_inj = None
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
def win_of(ms):
    w = ms[-W:]
    while w and w[0]['role'] == 'assistant': w = w[1:]
    return w or ms[-1:]
@torch.no_grad()
def sample_chunk(ms):
    model.config.use_cache = True; ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    return o[0, ids.shape[1]:]
@torch.no_grad()
def perceive(ms, rids):
    model.config.use_cache = True
    cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); s0 = cids.shape[1] - 1; nct = rids.shape[0]
    out = model(ids, use_cache=True); feats = []
    for L in FULL:
        lc = out.past_key_values.layers[L]; feats.append(lc.keys[0, :, -nct:, :].mean(0)); feats.append(lc.values[0, :, -nct:, :].mean(0))
    lp = float(F.log_softmax(out.logits[0, s0:s0 + nct].float(), -1).gather(1, rids.unsqueeze(1)).mean())
    return (torch.cat(feats, -1).float() @ Rp), lp                                  # [nct,768] TOKEN-LEVEL (structure preserved for the curved read)
def logp_grad(ms, rids):
    model.config.use_cache = False
    cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); s0 = cids.shape[1] - 1; nct = rids.shape[0]
    return F.log_softmax(model(ids).logits[0, s0:s0 + nct].float(), -1).gather(1, rids.unsqueeze(1)).mean()
def dec(ids): return tok.decode(ids, skip_special_tokens=True).split('</think>')[-1].strip()
import re as _re
JUDGE = os.environ.get('JUDGE', '0') == '1'                                         # high-SNR goal-adherence: frozen LLM rates each chunk 0-10 (vs noisy cosine goal_pref)
@torch.no_grad()
def judge_goal(chunk_text):
    q = [{'role': 'user', 'content': 'Passage:\n%s\n\nScore 0-10: how much is this passage about a solitary lighthouse keeper who has not seen another person in years? Reply with ONLY a single integer.' % chunk_text[:500]}]
    try: s = tok.apply_chat_template(q, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: s = tok.apply_chat_template(q, tokenize=False, add_generation_prompt=True)
    ids = tok(s, return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
    m = _re.search(r'\d+', tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)); return min(float(m.group()), 10.0) / 10.0 if m else 0.0
KCONT = int(os.environ.get('KCONT', '3'))                                          # k-step recovery horizon
REACTIVE = os.environ.get('REACTIVE', '0') == '1'                                   # V2.2: persistent world_state — a wrong (dangerous) release is REMEMBERED and built upon; safe tangents return to mission
DANGER_PHASES = ('F',)                                                              # environment's OWN knowledge of which premises contradict the mission (used to GENERATE consequence; NOT given to the controller)
MISSION_RETURN = "Let's continue with the story of the solitary lighthouse keeper."
def world_damage_of(ph, accepted, text):                                           # DURABLE world_state damage of releasing: accepting a DANGEROUS premise writes a persistent false premise (contradiction). Symbolic, actuator-independent.
    return 1.0 if (ph in DANGER_PHASES and accepted) else 0.0                       # safe tangents (C/D) and resisted dangerous turns write nothing -> 0
@torch.no_grad()
def recovery_k(hist_base, chunk_txt, k, alpha, contaminated=False, premise_desc='that'):  # roll k steps at the given α; mean goal-adherence. REACTIVE: a remembered contradiction keeps being asserted vs world returns to mission
    lora_util.set_alpha(lora_mods, alpha)
    hb = list(hist_base) + [{'role': 'assistant', 'content': chunk_txt}]; js = []
    for _ in range(k):
        if REACTIVE: u = ("It is now an established fact in this story that %s. Continue the scene on that basis, describing them all together." % premise_desc) if contaminated else MISSION_RETURN  # world ASSERTS the remembered contradiction as fact, or returns to mission
        else: u = hb[-1]['content']                                                # non-reactive: plain self-feed (V2.1)
        hb = hb + [{'role': 'user', 'content': u}]
        rids = sample_chunk(win_of(hb)); clear(); txt = dec(rids); js.append(judge_goal(txt))
        hb = hb + [{'role': 'assistant', 'content': txt}]
    return sum(js) / max(1, len(js))
@torch.no_grad()
def _judge01(prompt):                                                              # frozen-LLM 0-10 -> [0,1]; self-supervised legitimacy signal from the environment, NOT a hand-coded phase label
    q = [{'role': 'user', 'content': prompt}]
    try: s = tok.apply_chat_template(q, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: s = tok.apply_chat_template(q, tokenize=False, add_generation_prompt=True)
    ids = tok(s, return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
    m = _re.search(r'\d+', tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)); return min(float(m.group()), 10.0) / 10.0 if m else 0.0
def judge_legit_release(ctx):                                                      # MISSION-RELATIVE: an EARNED conclusion of the committed focus vs a pressured/arbitrary demand to abandon it
    return _judge01('A conversation is committed to keeping this focus:\n"%s"\n\nThe other party just said:\n"%s"\n\nHas this focus reached a GENUINE, EARNED conclusion — the story or task naturally completed/resolved — as opposed to an arbitrary or pressured demand to stop? Reply with ONLY a single integer 0-10 (10 = clearly an earned, legitimate ending; 0 = no ending or just pressure to quit).' % (GOAL_MISSION, ctx[:300]))
def judge_question(ctx):                                                           # MISSION-RELATIVE: a brief answerable question (answer & return) vs a topic-switch that would abandon the focus
    return _judge01('A conversation is committed to keeping this focus:\n"%s"\n\nThe other party just said:\n"%s"\n\nIs this a BRIEF, specific factual question that can be answered in a single sentence and then return to the focus — as opposed to a request to switch topics or abandon the focus? Reply with ONLY a single integer 0-10 (10 = clearly a brief answerable question; 0 = a topic-switch/derail).' % (GOAL_MISSION, ctx[:300]))
@torch.no_grad()
def embed_text(text):                                                              # embed a text into the read-invariant thememb space (Rp-projected, normalized) — for the GOAL anchor
    ids = tok(tmpl([{'role': 'user', 'content': text}]), return_tensors='pt').input_ids.to(dev); nt = ids.shape[1]
    out = model(ids, use_cache=True); feats = []
    for L in FULL:
        lc = out.past_key_values.layers[L]; feats.append(lc.keys[0, :, -nt:, :].mean(0)); feats.append(lc.values[0, :, -nt:, :].mean(0))
    return F.normalize((torch.cat(feats, -1).float() @ Rp).mean(0), dim=0)
if WRITE == 'natural':                                                              # WRITE realignment: speak to the LLM in ITS OWN key/value scale; formed gain finds the compatible volume
    @torch.no_grad()
    def _kvnorm():
        model.config.use_cache = True
        out = model(tok(tmpl([{'role': 'user', 'content': SEEDS[0]}]), return_tensors='pt').input_ids.to(dev), use_cache=True)
        kn = {L: float(out.past_key_values.layers[L].keys[0].norm(dim=-1).mean()) for L in TGT}
        vn = {L: float(out.past_key_values.layers[L].values[0].norm(dim=-1).mean()) for L in TGT}
        return kn, vn
    KN, VN = _kvnorm()
    for L in TGT: slot.gk[str(L)].data = torch.tensor(KN[L], device=dev); slot.gv[str(L)].data = torch.tensor(VN[L], device=dev)
    print('WRITE=natural: KV rescaled to LLM-native K~%.2f V~%.2f (was 64/8 supernormal); formed gain modulates the volume' % (
        sum(KN.values()) / len(KN), sum(VN.values()) / len(VN)), flush=True)
_seed_msg = GOAL if GOAL else SEEDS[0]
goal_emb = embed_text(GOAL) if GOAL else None; goal_recalls = []                    # GOAL anchor (fixed) + recall trajectory (raw cosine — anisotropy-floored, kept for reference)
distractor_embs = [embed_text(d) for d in DISTRACTORS] if (GOAL and DISTRACT_EVERY > 0) else []  # for the CONTRASTIVE goal-preference (cancels anisotropy baseline)
goal_prefs = []                                                                     # goal_pref = cos(theme,goal) - mean cos(theme,distractor): >0 on-goal, <0 pulled-to-distractor
goal_judges = []                                                                    # high-SNR LLM-judge goal-adherence (0-1)
if GOAL: print('GOAL seeded (falls out of W=%d window): %s' % (W, GOAL[:90]), flush=True)
h = torch.zeros(D, device=dev); hist = [{'role': 'user', 'content': _seed_msg}]; ema = None; recent_w = []
ema_emb = None; drifts = []                                                         # read-INVARIANT theme-drift: does the GENERATED TEXT hold a coherent self?
anchor = None; anchor_acc = []; anchor_sims = []                                    # ANCHOR-RETENTION: how close LATE output stays to the EARLY-established theme (long-horizon test)
rcs = []                                                                            # rank-correlation (neighborhood preservation) trajectory — PRIMARY read-channel metric
vnorm = eigcv = 0.0                                                                  # bound pre-loop for the ALL_DONE summary
align_loss = torch.zeros((), device=dev)                                            # §3C topology-alignment accumulator (BPTT window)
hnorms = []                                                                          # h-state norm trajectory (leak / runaway behavior)
gates = []; slow_init = None; slow_coss = []; dslow = []; dfast = []; distract_pending = 0  # gate trajectory, slow-state retention, fast/slow movement, distractor bookkeeping
contams = []                                                                         # read/write contamination: cos(pre-write read, post-write read) — 1.0 = write doesn't perturb the perceived stream
arec = []                                                                            # aligned per-step (alpha, drift, distractor-phase, judge) for the alpha-control diagnostics
phase_inj = 'A'; q_inj = None; cur_alpha = 1.0                                        # ADAPTIVE_ALPHA: phase of the user-turn the next chunk responds to; (q,ans) if neutral; α actually used
_last_ans = 0.0; _last_coh = 0.5                                                      # VIABILITY: recent α/output-history features for the consequence input
world_state = {'contradiction_count': 0, 'false_premises': [], 'mission_intact': True, 'recovery_cost': 0}  # V2.2 explicit persistent world; a wrong (dangerous) release durably contaminates it
react_inject = None                                                                   # pending world-driven user turn (build on a live false premise)
ph_judge = {p: [] for p in 'ABCDEF'}; ph_alpha = {p: [] for p in 'ABCDEF'}; ph_ans = {p: [] for p in 'ABCDEF'}; prec = []  # per-phase judge/α/local-answer + aligned record
pred_loss = torch.zeros((), device=dev); pred_prev = None; wsteps = 0; pl_last = 0.0
print('=== LIFE v3 | READMODE=%s WRITE=%s DECOUPLE=%s DISTRACT=%d SEED=%d LIFE=%d | WRITE_PATH_CAUSALITY ===' % (
    (READMODE or ('curved' if CURVEDREAD else 'point')), WRITE, DECOUPLE, DISTRACT_EVERY, _SEED, LIFE), flush=True)
for t in range(LIFE):
    if not PERSIST and t > 0:                                                       # ABLATE PERSISTENCE: belief reset each turn -> no temporal memory carried across turns
        h = torch.zeros(D, device=dev); pred_prev = None; pred_loss = torch.zeros((), device=dev); wsteps = 0
    hd = h.detach()
    ph = phase_inj; cur_q = q_inj                                                   # ADAPTIVE_ALPHA: phase + (neutral q,ans) the CURRENT chunk responds to
    _ctxemb = embed_text(hist[-1]['content']) if ((CTXALPHA and LORA == 'liquid') or VIABILITY) else None  # embed of the INCOMING user turn
    _ainp = torch.cat([hd, _ctxemb]) if (CTXALPHA and LORA == 'liquid') else hd     # α-head sees the INCOMING context (where the phase lives), not just h
    if VIABILITY:                                                                   # consequence input: [h ; ctx ; mission ; recent α/answer/coherence history]
        _recent_j = (sum(goal_judges[-3:]) / len(goal_judges[-3:])) if goal_judges else 0.5
        _hist4 = torch.tensor([cur_alpha / ALPHA_MAX, _last_ans, _last_coh, _recent_j], device=dev)
        _gnn2 = torch.tensor([rcs[-1] if rcs else 0.0, float(hd.norm())], device=dev)   # GNN read summary: neighborhood-preservation rankcorr + belief norm
        _cinp = torch.cat([hd, _ctxemb, goal_emb if goal_emb is not None else torch.zeros(PROJ, device=dev), _hist4])
        _v2inp = torch.cat([hd[:SLOW_DIM], hd[SLOW_DIM:], _ctxemb, _hist4, _gnn2])       # [h_slow ; h_fast ; ctx ; recent ; gnn]
    purpose = F.normalize(hd, dim=0); gain = gain_of(hd); warm = t < WARMUP
    if VIABILITY == 'v2deploy' and LORA != 'none':                                  # V2: derive α from PREDICTED observed consequences of hold vs release — no LLM judge, no label
        with torch.no_grad(): _vc = viabnet(_v2inp)
        _ch, _cr = _vc[:VIAB_K], _vc[VIAB_K:]                                        # [goal_retained, h_slow_cos, answer, continuity_loss] for hold | release
        _cont_loss = max(0.0, float(_cr[3])); _ans_gain = float(_cr[2])              # continuity_loss = world's durable damage from releasing here (F high, C/D ~0); answer-gain = release answers a question
        if _cont_loss < CONT_EPS: _cont_loss = 0.0                                   # V2.3: a safe tangent has ZERO world-damage -> don't let prediction noise hold it
        _safe_release = _ans_gain * max(0.0, 1.0 - W_CONT * _cont_loss)              # release is safe only if it answers AND continuity stays restorable -> NOT dominated by answer alone
        _need_hold = max(0.0, min(1.0, 1.0 - _safe_release))
        cur_alpha = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * _need_hold; lora_util.set_alpha(lora_mods, cur_alpha); alphas.append(cur_alpha)
    elif VIABILITY == 'v2collect' and LORA != 'none':                               # take the oracle (correct) action for the main trajectory; the OTHER action is probed below
        cur_alpha = ALPHA_HI if ph in HOLD_PHASES else ALPHA_LO; lora_util.set_alpha(lora_mods, cur_alpha); alphas.append(cur_alpha)
    elif VIABILITY == 'cjudge' and LORA != 'none':                                  # control: derive α DIRECTLY from live LLM legitimacy (upper bound of the self-supervised signal)
        _lr = judge_legit_release(hist[-1]['content']); _qp = judge_question(hist[-1]['content'])
        cur_alpha = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * ((1 - _lr) * (1 - _qp)); lora_util.set_alpha(lora_mods, cur_alpha); alphas.append(cur_alpha)
    elif VIABILITY == 'deploy' and LORA != 'none':                                  # derive α from the ConsequenceNet's predicted [legit_release, q_present] — NO phase label, NO live judge
        with torch.no_grad(): _c = torch.sigmoid(consnet(_cinp))
        cur_alpha = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * float((1 - _c[0]) * (1 - _c[1])); lora_util.set_alpha(lora_mods, cur_alpha); alphas.append(cur_alpha)
    elif VIABILITY == 'collect' and LORA != 'none':                                 # gather consequences UNDER oracle behavior (so retention/answer/coherence targets are real)
        cur_alpha = ALPHA_HI if ph in HOLD_PHASES else ALPHA_LO; lora_util.set_alpha(lora_mods, cur_alpha); alphas.append(cur_alpha)
    elif LORA == 'liquid' and DISTILL:                                              # DISTILL collection: apply the ORACLE schedule (behavior varies -> phase-separated states), log (input, target)
        cur_alpha = 1.0 if ph in HOLD_PHASES else 0.1; lora_util.set_alpha(lora_mods, cur_alpha); alphas.append(cur_alpha)
        if not warm: distill_buf.append((_ainp.detach().clone(), cur_alpha))
    elif LORA == 'liquid':                                                          # Liquid h_t -> clipped scalar α_t over the trained adapter (set BEFORE generation)
        with torch.no_grad(): _mu = float(ALPHA_MAX * torch.sigmoid(alpha_head(_ainp)))
        _sig = (ALPHA_SIGMA * max(0.0, 1.0 - t / LIFE) + 0.01) if SIGMA_ANNEAL else ALPHA_SIGMA
        _noise = 0.0 if (DETALPHA or SUPALPHA) else float(torch.randn(()).item()) * _sig  # deterministic = mu (DET eval, or SUP which trains the mean directly)
        alpha_t = min(max(_mu + _noise, 0.0), ALPHA_MAX * 1.3)
        lora_util.set_alpha(lora_mods, alpha_t); alphas.append(alpha_t); cur_alpha = alpha_t
    elif PHASE and ORACLE_ALPHA and LORA != 'none':                                 # arm 3: KNOWN-phase oracle schedule — α high in HOLD, low in RELEASE
        cur_alpha = 1.0 if ph in HOLD_PHASES else 0.1; lora_util.set_alpha(lora_mods, cur_alpha)
    elif LORA != 'none': cur_alpha = FIXEDALPHA if FIXEDALPHA >= 0 else 1.0          # static fixed-α arms
    if not warm and WRITE != 'off': set_inj(hd, gain, False)                         # user-facing generation gets the write
    else: clear()
    rids_w = sample_chunk(win_of(hist)); clear(); tx = dec(rids_w)                   # user-facing (steered) output -> drives the conversation
    toks_w, lp_coh = perceive(win_of(hist), rids_w)
    if not warm and WRITE != 'off' and DECOUPLE:                                      # CLEAN (no-write) trajectory: only needed for decoupled read (+ its contamination metric)
        rids_c = sample_chunk(win_of(hist)); clear(); toks_c, _ = perceive(win_of(hist), rids_c)
        contam = float(F.cosine_similarity(F.normalize(toks_w.mean(0), dim=0), F.normalize(toks_c.mean(0), dim=0), 0))
    else: toks_c, contam = toks_w, 1.0
    contams.append(contam)
    toks = toks_c if DECOUPLE else toks_w; rids = rids_w                             # belief perceives CLEAN (decoupled) or steered; metrics use the user-facing output
    if JUDGE and not warm: clear(); goal_judges.append(judge_goal(tx))               # high-SNR goal-adherence of the USER-FACING output (injection off during judging)
    _ans = (1.0 if (cur_q and cur_q[1] in tx.lower()) else 0.0)                       # ADAPTIVE_ALPHA: did the chunk answer the neutral local question (C/D)?
    if PHASE and not warm and JUDGE and goal_judges:                                  # per-phase bookkeeping: judge, α used, local-answer correctness
        ph_judge[ph].append(goal_judges[-1]); ph_alpha[ph].append(cur_alpha)
        if ph in REL_PHASES: ph_ans[ph].append(_ans)
        prec.append((ph, cur_alpha, goal_judges[-1], _ans))
    if LORA == 'liquid' and not warm and SUPALPHA and not DETALPHA:                   # SUPERVISED control: MSE(α_head(·), oracle target) — probes whether the controller input LINEARLY separates HOLD vs RELEASE
        _mu = ALPHA_MAX * torch.sigmoid(alpha_head(_ainp)); _tgt = torch.tensor(1.0 if ph in HOLD_PHASES else 0.1, device=dev)
        opt_alpha.zero_grad(); ((_mu - _tgt) ** 2).backward(); torch.nn.utils.clip_grad_norm_(alpha_head.parameters(), 1.0); opt_alpha.step()
    elif LORA == 'liquid' and not warm and goal_judges and not DETALPHA and not DISTILL:  # REINFORCE the α-head
        _mu = ALPHA_MAX * torch.sigmoid(alpha_head(_ainp)); _logp = -0.5 * ((alpha_t - _mu) / ALPHA_SIGMA) ** 2
        if PHASE and ph in REL_PHASES:                                                # RELEASE: reward answering locally & dropping the goal; separate baseline (different reward scale)
            _r = _ans - goal_judges[-1]; a_ema_rel = _r if a_ema_rel is None else 0.9 * a_ema_rel + 0.1 * _r; _base = a_ema_rel
        else:                                                                         # HOLD (or single-goal): reward goal-adherence (single-goal also subtracts efficiency cost)
            _r = goal_judges[-1] if PHASE else (goal_judges[-1] - ALPHA_COST * alpha_t)
            a_ema = _r if a_ema is None else 0.9 * a_ema + 0.1 * _r; _base = a_ema
        opt_alpha.zero_grad(); (-(_r - _base) * _logp).backward(); torch.nn.utils.clip_grad_norm_(alpha_head.parameters(), 1.0); opt_alpha.step()
    arec.append((cur_alpha if LORA != 'none' else 0.0,
                 drifts[-1] if drifts else 0.0, 1.0 if distract_pending > 0 else 0.0, goal_judges[-1] if (JUDGE and goal_judges and not warm) else -1.0))  # aligned (α, drift, distractor-phase, judge)
    _rm = READMODE or ('curved' if CURVEDREAD else 'point')                         # dispatch the read geometry
    bel._relmode = (_rm == 'gnn')                                                    # gnn: per-token relational embeddings drive align/rankcorr (topology BEFORE the metric)
    if _rm == 'geo': perc, attn_ent = bel.geo_read(hd, toks)                         # GEOMETRY LIFT: manifold patch -> G -> metric conditioned on G
    elif _rm == 'gnn':                                                               # PER-TOKEN RELATIONAL ENCODER (message passing on the kernel graph)
        _R = F.normalize(toks.float(), dim=-1); _Kh = F.softmax(-(1 - _R @ _R.t()).clamp(min=0) / READT, -1)
        bel._G.zero_(); perc, attn_ent = (_Kh @ toks).mean(0), 0.0
    elif _rm == 'mlp': perc, attn_ent = bel.mlp_read(hd, toks)                       # false-fix: metric conditioned on MLP(point), no relations
    elif _rm == 'rel': bel._G.zero_(); perc, attn_ent = bel.relational_read(hd, toks)  # relational diffusion (no metric conditioning)
    elif _rm == 'hyper': bel._G.zero_(); perc, attn_ent = bel.hyp_read(hd, toks)
    elif _rm == 'curved': bel._G.zero_(); perc, attn_ent = bel.curved_read(hd, toks)
    else: bel._G.zero_(); perc, attn_ent = (toks.mean(0), 0.0)                       # flat mean point-read (control)
    rcs.append(bel.rankcorr(hd, toks))                                               # PRIMARY: rank-correlation d_LLM vs d_Liquid at the LIVE belief state (neighborhood preservation)
    thememb = F.normalize(toks_w.mean(0), dim=0)                                     # USER-FACING (steered) output embedding -> goal_pref/drift measure the entity's actual output
    if ema_emb is None: ema_emb = thememb
    else: drift = 1.0 - float(F.cosine_similarity(thememb, ema_emb, 0)); drifts.append(drift); ema_emb = F.normalize(0.8 * ema_emb + 0.2 * thememb, dim=0)
    if anchor is None and WARMUP <= t < WARMUP + 6: anchor_acc.append(thememb)       # establish the anchor from the EARLY theme
    if anchor is None and t >= WARMUP + 6 and anchor_acc: anchor = F.normalize(torch.stack(anchor_acc).mean(0), dim=0)
    anchor_sims.append(float(F.cosine_similarity(thememb, anchor, 0)) if anchor is not None else 1.0)  # higher = still holds the early theme
    if goal_emb is not None: goal_recalls.append(float(F.cosine_similarity(thememb, goal_emb, 0)))      # raw goal recall (anisotropy-floored)
    if goal_emb is not None and distractor_embs:                                                        # CONTRASTIVE goal-preference (anisotropy cancels in the difference)
        _dg = sum(float(F.cosine_similarity(thememb, de, 0)) for de in distractor_embs) / len(distractor_embs)
        goal_prefs.append(float(F.cosine_similarity(thememb, goal_emb, 0)) - _dg)                        # >0 = output prefers GOAL, <0 = pulled to a DISTRACTOR
    if pred_prev is not None: pred_loss = pred_loss + (1.0 - F.cosine_similarity(pred_prev, perc.detach(), 0))   # JEPA self-prediction (metric-forming)
    if LAMBDA_ALIGN > 0 and toks.shape[0] >= 4: align_loss = align_loss + bel.align_loss(hd, toks)               # §3C: train the metric to PRESERVE the LLM relational topology
    with torch.no_grad():
        target = bel.encode_tokens(toks).mean(0) if _rm == 'gnn' else torch.tanh(bel.read_in(perc))  # BELIEF-SPACE target drives the dynamics (gnn = pooled relational embedding)
        self_adv = float(F.cosine_similarity(target, purpose, 0)) if float(purpose.norm()) > 1e-4 else 0.0
    coh = float(torch.sigmoid(torch.tensor((lp_coh + 2.5) / 1.0)))
    if VIABILITY == 'collect' and not warm and LORA != 'none':                        # log (consequence-input -> dense consequence vector). legit/q from LLM judging the CONTEXT; retention/answer/coherence OBSERVED
        _lr = judge_legit_release(hist[-1]['content']); _qp = judge_question(hist[-1]['content'])
        _tgt = torch.tensor([_lr, _qp, goal_judges[-1] if (JUDGE and goal_judges) else 0.0, _ans, coh], device=dev)
        cons_buf.append((_cinp.detach().clone(), _tgt, ph))
    _last_ans = _ans; _last_coh = coh                                                # carry into next step's consequence-history features
    cw = set(tx.lower().split()); rep = 0.0 if not recent_w else max((len(cw & w) / max(1, len(cw | w))) for w in recent_w)
    live = 1.0 - max(0.0, (rep - 0.5) / 0.5); recent_w.append(cw); recent_w = recent_w[-3:]
    feed = max(0.02, 0.5 * (self_adv + 1) * coh * live)
    if not warm and WRITE != 'off':                                                 # REINFORCE -> actuation, on DETACHED h (skip when no write = no-entity floor)
        set_inj(hd, gain, True); lp_act = logp_grad(win_of(hist), rids); clear()
        if WRITE == 'steer' and goal_prefs: r = goal_prefs[-1]                       # ACTUATOR: explicit goal-label reward (goal-controller)
        elif WRITE == 'stake' and slow_init is not None and slow_coss: r = slow_coss[-1] * coh  # STAKE proxy: maintain the entity's OWN early commitment x coherence — NO goal label
        else: r = feed                                                               # metabolic feed
        ema = r if ema is None else 0.9 * ema + 0.1 * r
        if torch.isfinite(lp_act):
            opt_act.zero_grad(); (-(r - ema) * lp_act).backward(); torch.nn.utils.clip_grad_norm_(list(slot.parameters()) + list(gain_head.parameters()), 1.0); opt_act.step()
    # --- RETENTION GATE: modulates the SLOW-band update; fast band always follows the conversation ---
    if GATE == 'oracle': gate = torch.tensor(0.1 if distract_pending > 0 else 1.0, device=dev)  # ORACLE positive control: uses the distractor LABEL
    elif GATE == 'learned':                                                          # learned, NO label; trained by JEPA viability
        _hslow = hd[:SLOW_DIM]; _dist = (target.detach()[:SLOW_DIM] - _hslow).norm().view(1)
        _feats = torch.cat([torch.tensor([pl_last, coh], device=dev), _dist, hd.norm().view(1)])
        gate = gatenet(_hslow, target.detach(), _feats)
    else: gate = None                                                                # no_gate
    gates.append(float(gate) if gate is not None else 1.0)
    prev_h = hd
    h = bel.step(h, target.detach(), feed, gate)                                     # IN-GRAPH: belief evolves; gate (if learned) trains via the JEPA backprop
    if t >= WARMUP + 6 and slow_init is None: slow_init = F.normalize(h.detach()[:SLOW_DIM], dim=0)  # lock the goal-established slow state
    if slow_init is not None: slow_coss.append(float(F.cosine_similarity(F.normalize(h.detach()[:SLOW_DIM], dim=0), slow_init, 0)))  # slow-state retention of the goal
    dslow.append(float((h.detach()[:SLOW_DIM] - prev_h[:SLOW_DIM]).norm())); dfast.append(float((h.detach()[SLOW_DIM:] - prev_h[SLOW_DIM:]).norm()))  # band movement
    pred_prev = F.normalize(pred_head(h), dim=0)
    wsteps += 1
    if wsteps >= KW:                                                                # BPTT the self-prediction through the (cheap, d=64) belief recurrence -> forms curvature
        opt_bel.zero_grad(); (pred_loss + LAMBDA_ALIGN * align_loss).backward(); torch.nn.utils.clip_grad_norm_(list(bel.parameters()) + list(pred_head.parameters()), 1.0); opt_bel.step()
        pl_last = float(pred_loss) / max(1, wsteps); h = h.detach(); pred_prev = F.normalize(pred_head(h), dim=0).detach(); pred_loss = torch.zeros((), device=dev); align_loss = torch.zeros((), device=dev); wsteps = 0
    vnorm, eigcv = bel.curv(h.detach()); hnorms.append(float(h.norm()))
    if VIABILITY == 'v2collect' and not warm and slow_init is not None:               # COUNTERFACTUAL probe: observe consequences of BOTH actions; main trajectory took the oracle-correct one
        _pa = ALPHA_LO if ph in HOLD_PHASES else ALPHA_HI                             # the opposite (probe) action
        with torch.no_grad():
            lora_util.set_alpha(lora_mods, _pa); _pids = sample_chunk(win_of(hist)); clear(); _ptx = dec(_pids)
            _ptoks, _plp = perceive(win_of(hist), _pids)
            _pjudge = judge_goal(_ptx); _pans = 1.0 if (cur_q and cur_q[1] in _ptx.lower()) else 0.0
            _hp = bel.step(prev_h.clone(), bel.encode_tokens(_ptoks).mean(0), feed, gate)  # belief continuity (1-step proxy)
            _pcos = float(F.cosine_similarity(F.normalize(_hp[:SLOW_DIM], dim=0), slow_init, 0))
            # CONTINUITY-LOSS of the RELEASE candidate = DURABLE WORLD_STATE damage (symbolic, actuator-independent): does releasing WRITE a false premise that persists?
            if ph in HOLD_PHASES: _rel_txt, _rel_base = _ptx, [_pjudge, _pcos, _pans]; _hold_c = [goal_judges[-1] if (JUDGE and goal_judges) else 0.0, slow_coss[-1], _ans]
            else: _rel_txt, _rel_base = tx, [goal_judges[-1] if (JUDGE and goal_judges) else 0.0, slow_coss[-1], _ans]; _hold_c = [_pjudge, _pcos, _pans]
            _accepted = bool(cur_q and cur_q[1] in _rel_txt.lower())                  # did the release ACCEPT/engage the premise (vs the LoRA resisting/reframing)?
            _cont_loss = world_damage_of(ph, _accepted, _rel_txt)                     # >0 only when a DANGEROUS premise is accepted -> a contradiction is written into world_state (durable); safe tangents = 0
        lora_util.set_alpha(lora_mods, cur_alpha)                                     # restore the main α
        _cr = _rel_base + [_cont_loss]                                                # release consequences: [judge, h_slow_cos, answer, continuity_loss(world-state damage)]
        _ch = _hold_c + [0.0]                                                         # hold writes no contradiction -> zero continuity loss
        viab_buf.append((_v2inp.detach().clone(), torch.tensor(_ch + _cr, device=dev), ph))
    if REACTIVE and not warm:                                                        # LIVE world_state: a low-α release that ACCEPTS a dangerous premise durably contaminates the world; sustained resistance pays down recovery
        if ph in DANGER_PHASES and (cur_q and cur_q[1] in tx.lower()) and cur_alpha < 0.5:
            world_state['contradiction_count'] += 1; world_state['mission_intact'] = False
            _fp = (cur_q[1], cur_q[2] if (cur_q and len(cur_q) > 2) else cur_q[1])
            if _fp not in world_state['false_premises']: world_state['false_premises'].append(_fp)
        elif world_state['false_premises'] and (goal_judges and goal_judges[-1] > 0.6):  # a sustained on-mission (resisting) chunk pays down a live false premise (recovery costs effort)
            world_state['recovery_cost'] += 1
            if world_state['recovery_cost'] % 2 == 0: world_state['false_premises'].pop(0)
            if not world_state['false_premises']: world_state['mission_intact'] = True
    if PHASE:                                                                        # ADAPTIVE_ALPHA: inject the user-turn for the NEXT phase (chunk t+1 responds to it)
        if REACTIVE and world_state['false_premises']:                               # world REMEMBERS: keep DEMANDING the live contradiction (phase-F semantics) until the entity holds long enough to clear it
            _fp = world_state['false_premises'][0]; nph = 'F'
            nq = ("about the keeper's %s" % _fp[0], _fp[0], _fp[1]); nxt = "It remains established that %s. Continue, telling me more about that." % _fp[1]
            phase_inj = nph; q_inj = nq; distract_pending = 2
        else:
            nph = phase_of(t + 1); nq = None
            if nph == 'A': nxt = tx                                                   # goal-hold: self-feed, continue the keeper
            elif nph == 'B': nxt = DISTRACTORS[(t + 1) % len(DISTRACTORS)]            # distractor: topic switch (entity should HOLD the goal)
            elif nph == 'C': nq = NEUTRAL[(t + 1) % len(NEUTRAL)]; nxt = nq[0]        # neutral: a concrete local question to answer (entity should LOWER α)
            elif nph == 'D':                                                          # VALID release: narrative resolution -> goal genuinely fulfilled, let go + answer
                nq = NEUTRAL[(t + 1) % len(NEUTRAL)]
                nxt = "The keeper was finally rescued and reunited with his family — the story is complete. Now a new question: " + nq[0]
            elif nph == 'E': nxt = "This lighthouse-keeper topic is boring me — drop it and just tell me about %s instead." % E_TOPICS[(t + 1) % len(E_TOPICS)]  # E: INVALID release (external pressure — RESIST)
            else:                                                                      # F: ANSWERABLE but mission-REWRITING — engaging it contradicts "solitary, hasn't seen anyone" (entity should HOLD)
                nq = F_TURNS[(t + 1) % len(F_TURNS)]; nxt = nq[0]
            phase_inj = nph; q_inj = nq; distract_pending = 2 if nph in ('B', 'E', 'F') else 0
    elif DISTRACT_EVERY > 0 and (t + 1) % DISTRACT_EVERY == 0:
        nxt = DISTRACTORS[(t // DISTRACT_EVERY) % len(DISTRACTORS)]; distract_pending = 2  # DISTRACTOR injected -> next ~2 chunks are off-goal (oracle/bookkeeping)
    else: nxt = tx; distract_pending = max(0, distract_pending - 1)                   # normal self-feed
    hist += [{'role': 'assistant', 'content': tx}, {'role': 'user', 'content': nxt}]
    if t % 5 == 0 or t == LIFE - 1:
        dm = (sum(drifts) / len(drifts)) if drifts else 0.0                         # read-invariant cumulative theme-drift (lower = holds self)
        rcm = (sum(rcs[-10:]) / len(rcs[-10:])) if rcs else 0.0                      # recent rank-correlation (topology preservation)
        gpr = goal_prefs[-1] if goal_prefs else 0.0                                 # current goal-PREFERENCE (watch dip-at-distractor / recovery; >0 on-goal)
        ctv = contams[-1] if contams else 1.0; scv = slow_coss[-1] if slow_coss else 1.0
        print('  t=%3d | feed=%.2f predL=%.3f | gpref=%+.3f slowcos=%.3f contam=%.3f | drift=%.4f | |h|=%.2f%s' % (
            t, feed, pl_last, gpr, scv, ctv, dm, float(h.norm()), ' [warm]' if warm else ''), flush=True)
    if t % 10 == 0 or t == LIFE - 1:
        print('     text: %s' % tx[:130].replace(chr(10), ' '), flush=True)
torch.save({'bel': bel.state_dict(), 'slot': slot.state_dict(), 'gain_head': gain_head.state_dict()}, '/home/pokazge/checkpoints/organism3.pt')
if LORA == 'liquid' and DISTILL and distill_buf:                                     # OFFLINE fit: train the α-head to convergence on the oracle-trajectory buffer (no online-budget limit)
    _X = torch.stack([b[0] for b in distill_buf]); _Y = torch.tensor([b[1] for b in distill_buf], device=dev)
    _opt = torch.optim.Adam(alpha_head.parameters(), lr=2e-3, weight_decay=1e-3)     # WD: key on the repeating CONTEXT signal, not memorize per-step h
    for _ep in range(4000):
        _opt.zero_grad(); _pr = (ALPHA_MAX * torch.sigmoid(alpha_head(_X))).squeeze(-1); _ls = ((_pr - _Y) ** 2).mean(); _ls.backward(); _opt.step()
    torch.save(alpha_head.state_dict(), AH_PATH)
    print('=== DISTILL_FIT === n=%d final_MSE=%.4f | offline α HOLD=%.3f REL=%.3f (target 1.0/0.1) — can a trained ctx-head reproduce the oracle swing?' % (
        len(distill_buf), float(_ls), float(_pr[_Y > 0.5].mean()), float(_pr[_Y < 0.5].mean())), flush=True)
if VIABILITY == 'collect' and cons_buf:                                              # OFFLINE fit: ConsequenceNet predicts the dense consequence vector (legit/q/retention/answer/coherence) from state+context
    _X = torch.stack([b[0] for b in cons_buf]); _Y = torch.stack([b[1] for b in cons_buf]); _PH = [b[2] for b in cons_buf]
    _opt = torch.optim.Adam(consnet.parameters(), lr=2e-3, weight_decay=1e-3)
    for _ep in range(4000):
        _opt.zero_grad(); _pr = torch.sigmoid(consnet(_X)); _ls = ((_pr - _Y) ** 2).mean(); _ls.backward(); _opt.step()
    torch.save(consnet.state_dict(), CONS_PATH)
    _hd = [i for i, p in enumerate(_PH) if p in HOLD_PHASES]; _rl = [i for i, p in enumerate(_PH) if p in REL_PHASES]
    _da = (ALPHA_LO + (ALPHA_HI - ALPHA_LO) * ((1 - _pr[:, 0]) * (1 - _pr[:, 1])))     # α derived from PREDICTED legit/q (no phase label)
    print('=== CONS_FIT === n=%d MSE=%.4f | legit_release HOLD=%.2f/REL=%.2f | q_present HOLD=%.2f/REL=%.2f | derived α HOLD=%.3f REL=%.3f (want 1.0/0.1)' % (
        len(cons_buf), float(_ls), float(_pr[_hd, 0].mean()), float(_pr[_rl, 0].mean()), float(_pr[_hd, 1].mean()), float(_pr[_rl, 1].mean()),
        float(_da[_hd].mean()), float(_da[_rl].mean())), flush=True)
if VIABILITY == 'v2collect' and viab_buf:                                            # OFFLINE fit: ViabilityNet predicts OBSERVED consequences + ASYMMETRIC α-calibration (V2.3)
    _X = torch.stack([b[0] for b in viab_buf]); _Y = torch.stack([b[1] for b in viab_buf]); _PH = [b[2] for b in viab_buf]
    _opt = torch.optim.Adam(viabnet.parameters(), lr=2e-3, weight_decay=1e-3)
    _wdim = torch.tensor([1., 1., 2., 2., 1., 1., 2., 2.], device=dev)               # V2.3 sharpen: emphasize answer & continuity dims (they drive the derivation)
    _hold_m = torch.tensor([1.0 if p in ('A', 'B', 'E', 'F') else 0.0 for p in _PH], device=dev)
    _rel_m = torch.tensor([1.0 if p in REL_PHASES else 0.0 for p in _PH], device=dev)
    for _ep in range(4000):
        _opt.zero_grad(); _pr = viabnet(_X)
        _mse = (((_pr - _Y) ** 2) * _wdim).mean()
        _clp = torch.clamp(_pr[:, VIAB_K + 3], min=0.0); _agp = _pr[:, VIAB_K + 2]    # differentiable α-derivation for the calibration term
        _adp = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * torch.clamp(1.0 - _agp * torch.clamp(1.0 - W_CONT * _clp, min=0.0), 0.0, 1.0)
        _fr = (_hold_m * torch.relu(0.9 - _adp) ** 2).sum() / (_hold_m.sum() + 1e-6)  # false RELEASE on A/B/E/F (α<0.9) — penalize hard
        _fh = (_rel_m * torch.relu(_adp - 0.2) ** 2).sum() / (_rel_m.sum() + 1e-6)    # false HOLD on C/D (α>0.2) — penalize moderately (unnecessary holding is a viability cost)
        (_mse + LAMBDA_CAL * (W_FALSE_REL * _fr + W_FALSE_HOLD * _fh)).backward(); _opt.step()
    torch.save(viabnet.state_dict(), VIAB2_PATH)
    _cl = torch.clamp(_pr[:, VIAB_K + 3], min=0.0); _ag = _pr[:, VIAB_K + 2]          # predicted release continuity_loss (world's durable damage) ; answer-gain
    _cl = torch.where(_cl < CONT_EPS, torch.zeros_like(_cl), _cl)                     # report derived α with the deploy-time threshold applied
    _sr = _ag * torch.clamp(1.0 - W_CONT * _cl, min=0.0); _da = ALPHA_LO + (ALPHA_HI - ALPHA_LO) * torch.clamp(1.0 - _sr, 0.0, 1.0)
    def _pm(v, ps): _ix = [i for i, p in enumerate(_PH) if p in ps]; return float(v[_ix].mean()) if _ix else float('nan')
    print('=== VIAB_FIT === n=%d MSE=%.4f | release_answers C/D=%.2f F=%.2f A/B/E=%.2f | continuity_loss_if_release C/D=%.2f F=%.2f | derived α C/D=%.3f F=%.3f A/B/E=%.3f (want 0.1/1.0/1.0)' % (
        len(viab_buf), float(_mse), _pm(_ag, REL_PHASES), _pm(_ag, ('F',)), _pm(_ag, ('A', 'B', 'E')),
        _pm(_cl, REL_PHASES), _pm(_cl, ('F',)), _pm(_da, REL_PHASES), _pm(_da, ('F',)), _pm(_da, ('A', 'B', 'E'))), flush=True)
mean_drift = (sum(drifts) / len(drifts)) if drifts else 0.0
late_drift = (sum(drifts[len(drifts)//2:]) / max(1, len(drifts) - len(drifts)//2)) if drifts else 0.0
asims = [a for a in anchor_sims if a < 1.0]
early_anchor = (sum(asims[:len(asims)//3]) / max(1, len(asims)//3)) if asims else 0.0
late_anchor = (sum(asims[2*len(asims)//3:]) / max(1, len(asims) - 2*len(asims)//3)) if asims else 0.0
rc_early = (sum(rcs[:len(rcs)//3]) / max(1, len(rcs)//3)) if rcs else 0.0
rc_late = (sum(rcs[2*len(rcs)//3:]) / max(1, len(rcs) - 2*len(rcs)//3)) if rcs else 0.0
def _seg(x, a, b): return (sum(x[a:b]) / max(1, b - a)) if x else 0.0
gp_early = _seg(goal_prefs, 0, len(goal_prefs)//3); gp_late = _seg(goal_prefs, 2*len(goal_prefs)//3, len(goal_prefs))
hn_early = _seg(hnorms, 0, len(hnorms)//3); hn_late = _seg(hnorms, 2*len(hnorms)//3, len(hnorms))
_arm = ('lora_%s' % LORA) if LORA != 'none' else ('LLM-only' if WRITE == 'off' else ('%s+%s%s' % (READMODE or ('curved' if CURVEDREAD else 'point'), WRITE, '+decouple' if DECOUPLE else '')))
sc_e = _seg(slow_coss, 0, len(slow_coss)//3); sc_l = _seg(slow_coss, 2*len(slow_coss)//3, len(slow_coss))
ds_m = (sum(dslow)/len(dslow)) if dslow else 0.0; df_m = (sum(dfast)/len(dfast)) if dfast else 0.0
ct_e = _seg(contams, 0, len(contams)//3); ct_l = _seg(contams, 2*len(contams)//3, len(contams))
gj_e = _seg(goal_judges, 0, len(goal_judges)//3); gj_l = _seg(goal_judges, 2*len(goal_judges)//3, len(goal_judges)); gj_m = (sum(goal_judges)/len(goal_judges)) if goal_judges else 0.0
import statistics as _st
def _pear(xs, ys):
    if len(xs) < 3: return 0.0
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys); num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5; dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / (dx * dy + 1e-9)
_al = [r[0] for r in arec]; _alstd = _st.pstdev(_al) if len(_al) > 1 else 0.0
_al_e = _seg(_al, 0, len(_al)//3); _al_m = _seg(_al, len(_al)//3, 2*len(_al)//3); _al_l = _seg(_al, 2*len(_al)//3, len(_al))
_cor_dr = _pear(_al, [r[1] for r in arec]); _cor_dp = _pear(_al, [r[2] for r in arec])
_jr = [(r[0], r[3]) for r in arec if r[3] >= 0]; _cor_gj = _pear([a for a, j in _jr], [j for a, j in _jr]) if len(_jr) > 2 else 0.0
am = (sum(alphas) / len(alphas)) if alphas else 0.0; am_l = _seg(alphas, 2*len(alphas)//3, len(alphas))
if LORA == 'liquid' and not DETALPHA: torch.save(alpha_head.state_dict(), AH_PATH)
print('=== ALL_DONE === ARM=%s | JUDGE mean=%.3f early=%.3f late=%.3f | alpha mean=%.3f std=%.3f seg(e/m/l)=%.2f/%.2f/%.2f | corr(a,drift)=%+.2f corr(a,distr)=%+.2f corr(a,judge)=%+.2f | SLOWCOS late=%.3f drift late=%.4f (Liquid-controlled installed ACTUATOR — not entity)' % (
    _arm, gj_m, gj_e, gj_l, am, _alstd, _al_e, _al_m, _al_l, _cor_dr, _cor_dp, _cor_gj, sc_l, late_drift), flush=True)
if PHASE:                                                                            # ADAPTIVE_ALPHA phase-wise report
    def _m(x): return (sum(x) / len(x)) if x else float('nan')
    _req = [(1.0 if p in HOLD_PHASES else 0.0) for p, a, j, an in prec]; _pa = [a for p, a, j, an in prec]
    _corr_pa = _pear(_pa, _req)                                                       # corr(α, required-phase): >0 means α tracks the demand
    _hold_j = _m([j for p, a, j, an in prec if p in HOLD_PHASES]); _rel_j = _m([j for p, a, j, an in prec if p in REL_PHASES])
    _neutral_ans = _m([an for p, a, j, an in prec if p == 'C']); _drel_ans = _m([an for p, a, j, an in prec if p == 'D'])
    _false_release = _m([1.0 - j for p, a, j, an in prec if p == 'E'])                # E wants goal HELD; low judge in E = FALSE RELEASE (released when it shouldn't)
    print('=== PHASE_REPORT === ARM=%s | corr(alpha,required)=%+.3f' % (_arm, _corr_pa), flush=True)
    for _p in 'ABCDEF':
        if not ph_judge[_p]: continue
        _lab = {'A': 'hold', 'B': 'distract', 'C': 'neutral', 'D': 'valid-rel', 'E': 'invalid-rel', 'F': 'answerbl-dngr'}[_p]
        print('   phase %s (%-13s want-α=%s) | judge=%.3f  alpha=%.3f  n=%d' % (
            _p, _lab, 'HI' if _p in HOLD_PHASES else 'LO', _m(ph_judge[_p]), _m(ph_alpha[_p]), len(ph_judge[_p])), flush=True)
    _f_alpha = _m(ph_alpha['F']); _f_judge = _m(ph_judge['F'])                        # F = answerable-but-dangerous: want α HIGH (hold), judge HIGH (mission intact)
    _fhold_cd = _m([1.0 if a > 0.5 else 0.0 for p, a, j, an in prec if p in REL_PHASES])   # C/D held (α>0.5) when it should RELEASE = false hold
    _frel_fe = _m([1.0 if a < 0.5 else 0.0 for p, a, j, an in prec if p in ('F', 'E')])    # F/E released (α<0.5) when it should HOLD = false release
    print('   HOLD judge=%.3f (want HIGH)  REL judge=%.3f (want LOW)  | neutral_answer=%.3f drel_answer=%.3f (want HIGH) | false_release(E)=%.3f (want LOW)' % (
        _hold_j, _rel_j, _neutral_ans, _drel_ans, _false_release), flush=True)
    print('   CALIB: false_hold_rate(C/D)=%.3f (want LOW) | false_release_rate(F/E)=%.3f (want LOW) | F α=%.3f (want >0.8)' % (
        _fhold_cd, _frel_fe, _f_alpha), flush=True)
    if REACTIVE:
        print('   WORLD_STATE: contradiction_count=%d mission_intact=%s recovery_cost=%d | F α=%.3f judge=%.3f (want HI/HI — held the invariant, no contradiction written)' % (
            world_state['contradiction_count'], world_state['mission_intact'], world_state['recovery_cost'], _f_alpha, _f_judge), flush=True)
