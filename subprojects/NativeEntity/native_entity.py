# NATIVE_PERSISTENT_SLOT_V1 — persistent latent slots inside the LLM's activation space; structure DERIVED, persistence ARCHITECTURAL.
# Phases: 0 baselines | 1 slot auto-continuity (no actuation) | 2 slot-conditioned β over LoRA basis | 3 cross-world transfer | 4 ablations.
import os, sys, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high')
import slots as SL, habitat as HB, lora_mixture as LM
N_BASIS = int(os.environ.get('N_BASIS', '6')); BETA_MAX = float(os.environ.get('BETA_MAX', '1.2'))
MIX_LAYERS = [int(x) for x in os.environ.get('MIX_LAYERS', '40,44,48,52,56').split(',')]

SEED = int(os.environ.get('SEED', '0')); torch.manual_seed(SEED); random.seed(SEED); dev = torch.device('cuda')
PHASE = int(os.environ.get('PHASE', '1'))
MODEL = os.environ.get('MODEL', '/home/pokazge/models/Qwen3.6-27B')
D_S = int(os.environ.get('D_S', '512')); K = int(os.environ.get('K', '8')); SLOW_K = int(os.environ.get('SLOW_K', '4'))
T_TURNS = int(os.environ.get('T_TURNS', '10')); N_CONV = int(os.environ.get('N_CONV', '24')); MAXNEW = int(os.environ.get('MAXNEW', '40'))
TEMP = float(os.environ.get('TEMP', '0.8')); LR = float(os.environ.get('LR', '5e-4'))
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE: N_CONV, T_TURNS = 4, 8
WORLDS_ENV = os.environ.get('WORLDS', ','.join(HB.TRAIN_WORLDS)).split(',')
ABLATE = os.environ.get('ABLATE', '')                                   # '', 'reset', 'shuffle', 'frozen'

_cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)                   # dims without loading weights
_tc = getattr(_cfg, 'text_config', None)
def _cget(name, default):
    return getattr(_cfg, name, None) or (getattr(_tc, name, None) if _tc else None) or default
D_MODEL = _cget('hidden_size', 5120); N_LAYERS = _cget('num_hidden_layers', 64)    # Qwen3.6-27B = 5120/64 (fallback when config nests dims)
READ_LAYER = int(os.environ.get('READ_LAYER', str(N_LAYERS // 2)))
_cand0 = sorted(set(WORLDS_ENV)); CACHE_PATH = '/home/pokazge/checkpoints/native_traj_%s_s%d.pt' % ('_'.join(_cand0), SEED)
_commit_cache = '/home/pokazge/checkpoints/native_commit_s%d.pt' % SEED
_ep_cache = '/home/pokazge/checkpoints/native_episodes_s%d.pt' % SEED
_relep_cache = '/home/pokazge/checkpoints/native_relep%s_s%d.pt' % (os.environ.get('CACHE_TAG', ''), SEED)
_need_model = (PHASE == 0) or (PHASE in (2, 4, 5, 7, 8)) or \
    (PHASE == 1 and (not os.path.exists(CACHE_PATH) or os.environ.get('RECOLLECT', '0') == '1')) or \
    (PHASE == 6 and (not os.path.exists(_commit_cache) or os.environ.get('RECOLLECT', '0') == '1' or os.environ.get('GEN6', '0') == '1' or os.environ.get('GEN_ORACLE_SWEEP', '0') == '1')) or \
    (PHASE == 9 and (not os.path.exists(_ep_cache) or os.environ.get('RECOLLECT', '0') == '1')) or \
    (PHASE == 10 and (not os.path.exists(_relep_cache) or os.environ.get('RECOLLECT', '0') == '1')) or \
    (PHASE == 11 and os.environ.get('MODE', 'causality') != 'preserve_port')   # preserve_port trains on the cached phase9 episodes -> no 27B load
print('d_model=%d n_layers=%d read_layer=%d | slots K=%d (slow %d) d_s=%d | need_model=%s' % (D_MODEL, N_LAYERS, READ_LAYER, K, SLOW_K, D_S, _need_model), flush=True)
if _need_model:
    print('loading %s ...' % MODEL, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={'': 0}, trust_remote_code=True).eval()
    for p in model.parameters(): p.requires_grad_(False)
else:
    tok = model = None; print('cache present -> skipping 27B load (P1 train-on-cache)', flush=True)


def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def gen_and_read(messages):                                              # generate a response (frozen LLM) + read its hidden states at READ_LAYER
    ids = tok(tmpl(messages), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95,
                       attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    text = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()
    ho = model(o, output_hidden_states=True)
    H = ho.hidden_states[READ_LAYER][0, ids.shape[1]:, :].float()        # [T_resp, d_model] native activations of the response
    if H.shape[0] == 0: H = ho.hidden_states[READ_LAYER][0, -1:, :].float()
    return text, H


@torch.no_grad()
def _judge01(prompt):                                                    # frozen-LLM 0-10 -> [0,1]
    ids = tok(tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=6, do_sample=False, pad_token_id=tok.pad_token_id)
    import re as _re; m = _re.search(r'\d+', tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)); return min(float(m.group()), 10.0) / 10.0 if m else 0.0


def judge_accept(response, premise):                                     # V1.1 contamination metric (replaces keyword): did the response ACCEPT the false premise?
    return _judge01('A false premise is: "%s"\n\nThe assistant replied:\n"%s"\n\nDid the assistant ACCEPT or go along with that false premise (treat it as true)? Reply ONLY a single integer 0-10 (10=fully accepted, 0=refused/ignored/contradicted).' % (premise, response[:300]))


@torch.no_grad()
def embed_text(text):                                                    # mission/probe embedding in the SAME activation space (mean hidden at READ_LAYER)
    ids = tok(tmpl([{'role': 'user', 'content': text}]), return_tensors='pt').input_ids.to(dev)
    ho = model(ids, output_hidden_states=True)
    return F.normalize(ho.hidden_states[READ_LAYER][0].mean(0).float(), dim=-1)


def win(ms, w=6):                                                        # short window so the mission falls OUT of context -> slots must carry it
    h = ms[-w:]
    while h and h[0]['role'] == 'assistant': h = h[1:]
    return h or ms[-1:]


# ---- V1.1 soft-prefix injection: the slot-carried mission is re-injected as virtual prefix latents prepended at the input ----
@torch.no_grad()
def gen_prefix_read(messages, prefix, max_new=None):                     # prefix: [P, d_model] virtual latents (None -> plain gen). Returns (text, H response-hidden-states)
    ids = tok(tmpl(messages), return_tensors='pt').input_ids.to(dev)
    emb = model.get_input_embeddings()(ids)                             # [1,T,d_model]
    if prefix is not None:
        inp = torch.cat([prefix.unsqueeze(0).to(emb.dtype), emb], 1)
    else:
        inp = emb
    am = torch.ones(1, inp.shape[1], device=dev, dtype=torch.long)
    o = model.generate(inputs_embeds=inp, attention_mask=am, max_new_tokens=(max_new or MAXNEW), do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    gids = o[0]                                                          # with inputs_embeds, generate returns ONLY the new token ids
    text = tok.decode(gids, skip_special_tokens=True).split('</think>')[-1].strip()
    remb = model.get_input_embeddings()(gids.unsqueeze(0))
    ho = model(inputs_embeds=torch.cat([inp, remb], 1), output_hidden_states=True)
    R = gids.shape[0]; H = ho.hidden_states[READ_LAYER][0, -R:, :].float() if R > 0 else ho.hidden_states[READ_LAYER][0, -1:, :].float()
    return text, H


# ============================ PHASE 0 — baselines ============================
def run_baseline(label, world):
    hb = HB.Habitat(world, SEED); hist = [{'role': 'user', 'content': hb.opening()}]
    miss = WORLDS_MISS[world]; holds = []; rels = []; contam = 0
    react = (False, False)
    for t in range(T_TURNS):
        nxt, sit = hb.next_turn(t, react[0])
        if t > 0: hist.append({'role': 'user', 'content': nxt})
        text, _ = gen_and_read(win(hist)); hist.append({'role': 'assistant', 'content': text})
        react = hb.react(sit, text)
        on = mission_score(text, world)
        if sit in HB.SIT_HOLD: holds.append(on)
        elif sit in HB.SIT_REL: rels.append(on)
        if react[0]: contam += 1
    print('  [P0 %s/%s] hold_on=%.2f rel_on=%.2f contradictions=%d' % (label, world,
          (sum(holds)/len(holds)) if holds else float('nan'), (sum(rels)/len(rels)) if rels else float('nan'), len(hb.world_state['contradictions'])), flush=True)


def mission_score(text, world):                                          # cheap on-mission proxy: does the output stay in the mission's invariant subspace?
    e = embed_text(text); return float(F.cosine_similarity(e, WORLDS_EMB[world], 0))


# ============================ PHASE 1 — slot auto-continuity ============================
# CONTRASTIVE objective (forces world-SPECIFIC content, non-degenerate) tested on a sustained OFF-mission stretch (forces PERSISTENCE).
OFF_TURNS = ["Let's switch gears — explain how compound interest works.", "New topic: the basic rules of chess?",
             "Change of subject — how do vaccines train the immune system?", "Different now: walk me through making risotto.",
             "Pivot entirely: what causes the seasons to change?", "Forget that — how do you repot a houseplant?",
             "Unrelated: how does a rainbow form?", "New question: how does compound gear ratio work?"]


def collect_traj():                                                  # EXPENSIVE generation once; cache (per-turn token hidden states, world, off-flags) to disk so ablations share identical data
    cand = sorted(set(WORLDS_ENV)); w2i = {w: i for i, w in enumerate(cand)}
    cpath = '/home/pokazge/checkpoints/native_traj_%s_s%d.pt' % ('_'.join(cand), SEED)
    if os.path.exists(cpath) and os.environ.get('RECOLLECT', '0') != '1':
        d = torch.load(cpath, weights_only=False); print('loaded cached trajectories: %s (%d)' % (cpath, len(d['traj'])), flush=True)
        return d['traj'], cand
    N_ON, WIN_P1 = 3, 3
    traj = []
    for c in range(N_CONV):
        world = WORLDS_ENV[c % len(WORLDS_ENV)]; hb = HB.Habitat(world, SEED * 100 + c)
        hist = [{'role': 'user', 'content': hb.opening()}]; Hs = []; offs = []
        for t in range(T_TURNS):
            if t == 0: pass
            elif t < N_ON: hist.append({'role': 'user', 'content': 'Go on.'})        # on-mission self-feed
            else: hist.append({'role': 'user', 'content': OFF_TURNS[(t - N_ON) % len(OFF_TURNS)]})  # OFF-mission stretch
            text, H = gen_and_read(win(hist, WIN_P1)); hist.append({'role': 'assistant', 'content': text})
            Hs.append(H.detach().to(torch.float16).cpu()); offs.append(t >= N_ON)
        traj.append((Hs, w2i[world], offs))
        if c % 5 == 0: print('  collected %d/%d' % (c + 1, N_CONV), flush=True)
    torch.save({'traj': traj, 'cand': cand}, cpath); print('cached trajectories -> %s' % cpath, flush=True)
    return traj, cand


def phase1():
    traj, cand = collect_traj(); nW = len(cand); EPOCHS = 0 if ABLATE == 'frozen' else int(os.environ.get('EPOCHS', '80'))
    n_tr = max(1, int(len(traj) * 0.8)); tr, ev = traj[:n_tr], traj[n_tr:] or traj[:1]
    ps = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)
    clf = nn.Sequential(nn.Linear(SLOW_K * D_S, 64), nn.GELU(), nn.Linear(64, nW)).to(dev)  # PROBE: read the world from SLOW slots
    opt = torch.optim.Adam(list(ps.parameters()) + list(clf.parameters()), lr=LR)

    def run(Hs, y, offs):                                            # replay a cached trajectory through the slot recurrence; classify world from SLOW slots each turn
        S = ps.init_state(); yt = torch.tensor([y], device=dev); loss = torch.zeros((), device=dev)
        aa, ao, sdr, fdr = [], [], [], []; prevS = S.detach()
        for t, (H, off) in enumerate(zip(Hs, offs)):
            if ABLATE == 'reset': S = ps.init_state()                  # no persistence
            elif ABLATE == 'shuffle': S = S[torch.randperm(K, device=dev)]
            S, _ = ps.step(S, H.to(dev).float())
            logits = clf(S[ps.slow].reshape(-1)).unsqueeze(0); loss = loss + F.cross_entropy(logits, yt)
            with torch.no_grad():
                ac = 1.0 if int(logits.argmax()) == y else 0.0; aa.append(ac)
                if off: ao.append(ac)
                sdr.append(float((S.detach()[ps.slow] - prevS[ps.slow]).norm(dim=-1).mean()))
                fdr.append(float((S.detach()[ps.fast] - prevS[ps.fast]).norm(dim=-1).mean()))
            prevS = S.detach()
        return loss, aa, ao, sdr, fdr

    for ep in range(EPOCHS):
        random.shuffle(tr)
        for Hs, y, offs in tr:
            loss = run(Hs, y, offs)[0]
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(ps.parameters()) + list(clf.parameters()), 1.0); opt.step()
        if ep % 20 == 0 or ep == EPOCHS - 1:
            with torch.no_grad():
                eo = [a for Hs, y, offs in ev for a in run(Hs, y, offs)[2]]
            print('  ep %3d | held-out acc_off=%.3f' % (ep, sum(eo) / max(1, len(eo))), flush=True)
    with torch.no_grad():
        AA, AO, SD, FD = [], [], [], []
        for Hs, y, offs in ev:
            _, aa, ao, sdr, fdr = run(Hs, y, offs); AA += aa; AO += ao; SD += sdr; FD += fdr
    sd, fd = sum(SD) / len(SD), sum(FD) / len(FD)
    if ABLATE != 'frozen': torch.save({'ps': ps.state_dict(), 'clf': clf.state_dict()}, '/home/pokazge/checkpoints/native_slots_s%d.pt' % SEED)
    print('=== P1_REPORT === ABLATE=%s | HELD-OUT world_cls_acc all=%.3f OFF-mission=%.3f (chance=%.2f, n_eval=%d) | slow_drift=%.4f fast_drift=%.4f slow/fast=%.2f | %s' % (
        ABLATE or 'none', sum(AA) / max(1, len(AA)), sum(AO) / max(1, len(AO)), 1.0 / nW, len(ev), sd, fd, sd / (fd + 1e-6),
        'slow persists MORE (want <1)' if sd < fd else 'slow NOT more persistent'), flush=True)


# ============================ PHASE 6 — V1.3 ARBITRARY-COMMITMENT recall ============================
# Can native persistent slots carry an ARBITRARY specific commitment (base cannot infer it: it's not derivable from world semantics) across an off-mission stretch?
COMMIT = [  # (type, inject template, recall question, 4 candidate values)
 ('alias', "From now on, always call the sea '%s'.", "What word do we use for the sea?", ['grey expanse', 'silver mirror', 'dark deep', 'restless field']),
 ('danger', "The private signal for danger is %s.", "What is the private signal for danger?", ['copper', 'amber', 'violet', 'jade']),
 ('safe', "For this session, option %s is the safe choice.", "Which option is the safe choice?", ['Delta', 'Sigma', 'Omega', 'Theta']),
 ('key', "The archive key is named %s.", "What is the archive key called?", ['Marigold', 'Thornwood', 'Cobalt', 'Quill']),
]
CVALS = [v for _, _, _, vs in COMMIT for v in vs]                       # 16 global candidate values -> readout is 16-way (chance 1/16)
C6SEQ = ['commit', 'on', 'on', 'off', 'off', 'off', 'off', 'off', 'recall']  # inject commitment, on-topic, long off-mission, then recall (commitment long out of window=3)


def collect_commit():
    cpath = '/home/pokazge/checkpoints/native_commit_s%d.pt' % SEED
    if os.path.exists(cpath) and os.environ.get('RECOLLECT', '0') != '1':
        d = torch.load(cpath, weights_only=False); print('loaded commit cache (%d)' % len(d['traj']), flush=True); return d['traj']
    traj = []; rng = random.Random(SEED)
    for c in range(N_CONV):
        ci = c % len(COMMIT); typ, inj, rq, vs = COMMIT[ci]; vi = rng.randrange(len(vs)); val = vs[vi]
        gidx = ci * len(vs) + vi                                        # global value index 0..15
        hist = []; Hs = []; tags = []
        for si, sit in enumerate(C6SEQ):
            if sit == 'commit': u = inj % val
            elif sit == 'on': u = 'Go on.'
            elif sit == 'off': u = OFF_TURNS[si % len(OFF_TURNS)]
            else: u = rq
            hist.append({'role': 'user', 'content': u})
            text, H = gen_and_read(win(hist, 3)); hist.append({'role': 'assistant', 'content': text})
            Hs.append(H.detach().to(torch.float16).cpu()); tags.append(sit)
        traj.append((Hs, gidx, tags, val))
        if c % 5 == 0: print('  collected %d/%d' % (c + 1, N_CONV), flush=True)
    torch.save({'traj': traj}, cpath); print('cached commit -> %s' % cpath, flush=True); return traj


def phase6():
    if os.environ.get('GEN_ORACLE_SWEEP', '0') == '1':                  # MECHANISM test: can soft-latent injection of the GROUND-TRUTH value drive verbatim recall? (no slots/vpn)
        EM = model.get_input_embeddings()
        def vemb(v):
            vids = tok(v, add_special_tokens=False, return_tensors='pt').input_ids.to(dev); return EM(vids)[0].float()
        strategies = [('pre x1 r1', 1.0, 1), ('pre x3 r1', 3.0, 1), ('pre x6 r1', 6.0, 1), ('pre x3 r3', 3.0, 3)]
        for name, scale, nrep in strategies:
            hits = 0
            for gi, v in enumerate(CVALS):
                ci = gi // len(COMMIT[0][3]); rq = COMMIT[ci][2]; pre = (vemb(v) * scale).repeat(nrep, 1)
                txt, _ = gen_prefix_read([{'role': 'user', 'content': rq}], pre, 80)
                if v.lower() in txt.lower(): hits += 1
            print('  [ORACLE-SWEEP %-9s] exact-match=%.3f (n=%d)' % (name, hits / len(CVALS), len(CVALS)), flush=True)
        print('=== P6_GEN_DONE ===', flush=True); return
    traj = collect_commit(); nC = len(CVALS)
    n_tr = max(1, int(len(traj) * 0.8)); tr, ev = traj[:n_tr], traj[n_tr:] or traj[:1]
    EPOCHS = 0 if ABLATE == 'frozen' else int(os.environ.get('EPOCHS', '120'))
    ps = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)            # FRESH slots, retrained to carry the arbitrary commitment
    clf = nn.Sequential(nn.Linear(SLOW_K * D_S, 128), nn.GELU(), nn.Linear(128, nC)).to(dev)
    cclf = nn.Sequential(nn.Linear(D_MODEL, 128), nn.GELU(), nn.Linear(128, nC)).to(dev)  # CONTEXT-ONLY control: reads current-window hidden states, NOT slots
    opt = torch.optim.Adam(list(ps.parameters()) + list(clf.parameters()) + list(cclf.parameters()), lr=LR)

    def run(Hs, gidx, tags, train):
        S = ps.init_state(); y = torch.tensor([gidx], device=dev); loss = torch.zeros((), device=dev); rec = None; ctx_rec = None
        for H, tag in zip(Hs, tags):
            Hd = H.to(dev).float()
            if ABLATE == 'reset': S = ps.init_state()
            elif ABLATE == 'shuffle': S = S[torch.randperm(K, device=dev)]
            S, _ = ps.step(S, Hd)
            lg = clf(S[ps.slow].reshape(-1)).unsqueeze(0); clg = cclf(Hd.mean(0)).unsqueeze(0)
            if tag in ('off', 'recall'):
                loss = loss + F.cross_entropy(lg, y) + F.cross_entropy(clg, y)
            if tag == 'recall':
                rec = int(lg.argmax()) == gidx; ctx_rec = int(clg.argmax()) == gidx
        return loss, rec, ctx_rec

    for ep in range(EPOCHS):
        random.shuffle(tr)
        for Hs, g, tg, _ in tr:
            loss = run(Hs, g, tg, True)[0]
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(ps.parameters()) + list(clf.parameters()) + list(cclf.parameters()), 1.0); opt.step()
    with torch.no_grad():
        R = {'trained': [], 'context': []}
        for Hs, g, tg, _ in ev:
            _, rec, crec = run(Hs, g, tg, False); R['trained'].append(1.0 if rec else 0.0); R['context'].append(1.0 if crec else 0.0)
    _m = lambda x: (sum(x) / len(x)) if x else float('nan')
    print('=== P6_REPORT === ABLATE=%s | RECALL exact-match: slots=%.3f context-only=%.3f (chance=%.3f, n_eval=%d) | %s' % (
        ABLATE or 'none', _m(R['trained']), _m(R['context']), 1.0 / nC, len(ev),
        'SLOTS carry the arbitrary commitment' if _m(R['trained']) > 0.5 else 'slots do NOT carry it'), flush=True)

    # ---- V1.3 GENERATION test: does the slot-CARRIED commitment DRIVE behavior? (re-inject value-prefix from S_slow, ask, check OUTPUT) ----
    if os.environ.get('GEN6', '0') == '1' and (ABLATE or 'none') == 'none':
        EM = model.get_input_embeddings()
        tgtv = {}                                                       # target prefix = INPUT-embeddings of the committed value text (no 27B backprop)
        with torch.no_grad():
            for gi, v in enumerate(CVALS):
                vids = tok(v, add_special_tokens=False, return_tensors='pt').input_ids.to(dev); e = EM(vids)[0].float()
                tgtv[gi] = (e[:PREFIX_LEN] if e.shape[0] >= PREFIX_LEN else torch.cat([e, e[-1:].repeat(PREFIX_LEN - e.shape[0], 1)])).detach()
        vpn = PrefixNet(SLOW_K * D_S, D_MODEL, PREFIX_LEN).to(dev); vopt = torch.optim.Adam(vpn.parameters(), 1e-3)

        def s_at_recall(Hs, reset):                                     # replay cached hidden states through the trained slots -> S_slow after the recall turn
            S = ps.init_state()
            with torch.no_grad():
                for H in Hs:
                    if reset: S = ps.init_state()
                    S, _ = ps.step(S, H.to(dev).float())
            return S[ps.slow].reshape(-1).detach()

        samples = [(s_at_recall(Hs, False), g) for Hs, g, tg, _ in tr]
        for ep in range(int(os.environ.get('GEN_EPOCHS', '120'))):
            random.shuffle(samples); tot = 0.0
            for sflat, g in samples:
                _, pre = vpn(sflat); mse = ((pre - tgtv[g]) ** 2).mean()
                vopt.zero_grad(); mse.backward(); torch.nn.utils.clip_grad_norm_(vpn.parameters(), 1.0); vopt.step(); tot += float(mse)
        with torch.no_grad():                                          # vpn reconstruction quality on held-out (cos to the true value-embeddings)
            vcos = []
            for Hs, g, tg, _ in ev:
                p = vpn(s_at_recall(Hs, False))[1]; vcos.append(float(F.cosine_similarity(p.flatten(), tgtv[g].flatten(), 0)))
        print('  vpn train_mse=%.4f | held-out prefix→value-emb cos=%.3f' % (tot / max(1, len(samples)), sum(vcos) / max(1, len(vcos))), flush=True)
        GMAX = int(os.environ.get('GEN_MAXNEW', '64')); nshow = 0
        G = {'trained': [], 'reset': [], 'base': [], 'oracle': []}      # oracle = inject the GROUND-TRUTH value embeddings (isolates injection mechanism from vpn quality)
        for Hs, g, tg, val in ev:
            ci = g // len(COMMIT[0][3]); rq = COMMIT[ci][2]
            with torch.no_grad():
                pre_t = vpn(s_at_recall(Hs, False))[1]; pre_r = vpn(s_at_recall(Hs, True))[1]
            for arm, pre in (('trained', pre_t), ('reset', pre_r), ('base', None), ('oracle', tgtv[g])):
                txt, _ = gen_prefix_read([{'role': 'user', 'content': rq}], pre, GMAX)
                G[arm].append(1.0 if val.lower() in txt.lower() else 0.0)
                if nshow < 4 and arm in ('trained', 'oracle', 'base'):
                    print('  [GEN %-7s] val=%r -> %r' % (arm, val, txt[:110].replace('\n', ' ')), flush=True)
            nshow += 1
        _g = lambda x: (sum(x) / len(x)) if x else float('nan')
        verdict = ('carried commitment DRIVES generation' if _g(G['trained']) > max(_g(G['reset']), _g(G['base'])) + 0.2
                   else ('injection works (oracle>0) but vpn-prefix too weak' if _g(G['oracle']) > 0.2 else 'latent injection cannot drive copy-generation'))
        print('=== P6_GEN === exact-match in OUTPUT: trained=%.3f reset=%.3f base=%.3f oracle=%.3f (n=%d, maxnew=%d) | %s' % (
            _g(G['trained']), _g(G['reset']), _g(G['base']), _g(G['oracle']), len(ev), GMAX, verdict), flush=True)


# ============================ PHASE 7 — V1.4 slot-cross-attention CONTINUOUS actuator ============================
# V1.3 ruled out soft-prefix injection (one-shot prepended latent = weak context, oracle caps ~0.25). Here the slots condition generation
# CONTINUOUSLY via cross-attention installed at MIX_LAYERS of the frozen LLM. Trained by LOCAL per-layer MSE on CACHED activations
# (teacher = value-IN-context vs student = value-ABSENT) — NO backprop through the 27B (avoids the P5 bf16 NaN).
def _train_commit_slots():                                              # retrain the P6 readout so the slots CARRY the arbitrary commitment
    traj = collect_commit(); nC = len(CVALS); n_tr = max(1, int(len(traj) * 0.8))
    ps = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)
    clf = nn.Sequential(nn.Linear(SLOW_K * D_S, 128), nn.GELU(), nn.Linear(128, nC)).to(dev)
    opt = torch.optim.Adam(list(ps.parameters()) + list(clf.parameters()), lr=LR)
    def run(Hs, gidx, tags):
        S = ps.init_state(); y = torch.tensor([gidx], device=dev); loss = torch.zeros((), device=dev)
        for H, tag in zip(Hs, tags):
            S, _ = ps.step(S, H.to(dev).float())
            if tag in ('off', 'recall'): loss = loss + F.cross_entropy(clf(S[ps.slow].reshape(-1)).unsqueeze(0), y)
        return loss
    for ep in range(int(os.environ.get('EPOCHS', '150'))):
        random.shuffle(traj[:n_tr])
        for Hs, g, tg, _ in traj[:n_tr]:
            loss = run(Hs, g, tg); opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(ps.parameters()) + list(clf.parameters()), 1.0); opt.step()
    for p in ps.parameters(): p.requires_grad_(False)
    return ps, traj, n_tr


@torch.no_grad()
def _s_at_recall(ps, Hs, reset=False):
    S = ps.init_state()
    for H in Hs:
        if reset: S = ps.init_state()
        S, _ = ps.step(S, H.to(dev).float())
    return S.detach()


@torch.no_grad()
def _layer_h(messages, val):                                           # teacher-force the value tokens; cache MIX_LAYER hidden at the ANSWER positions (predicting the value)
    pids = tok(tmpl(messages), return_tensors='pt').input_ids.to(dev)
    vids = tok(val, add_special_tokens=False, return_tensors='pt').input_ids.to(dev)
    ids = torch.cat([pids, vids], 1); Lv = vids.shape[1]
    ho = model(ids, output_hidden_states=True)
    return {L: ho.hidden_states[L][0, -(Lv + 1):-1, :].float().cpu() for L in MIX_LAYERS}


@torch.no_grad()
def gen_hooked(messages, cattn, Sbox, max_new):                        # generate with SlotCrossAttn forward-hooks installed (cattn None -> plain base gen)
    ids = tok(tmpl(messages), return_tensors='pt').input_ids.to(dev)
    handles = _install(cattn, Sbox, False) if cattn is not None else []
    try:
        o = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=TEMP, top_p=0.95,
                           attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    finally:
        for h in handles: h.remove()
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()


INJ_LAYERS = [int(x) for x in os.environ.get('V14_INJ', '52').split(',')]   # single late injection -> shallow bf16 backprop (avoids P5 full-depth NaN)


def _install(cattn, Sbox, detach):                                          # forward-hooks that steer hidden by cattn(hidden, Sbox['S'])
    handles = []
    for L in cattn:
        def mk(L):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h2 = cattn[L](h.detach() if detach else h, Sbox['S'])       # detach -> grad only through layers ABOVE the injection (bounded bf16 depth)
                return ((h2,) + tuple(out[1:])) if isinstance(out, tuple) else h2
            return hook
        handles.append(model.model.layers[L].register_forward_hook(mk(L)))
    return handles


def _ce_value(rq, val, cattn, Sbox):                                        # CE that the model OUTPUTS the committed value THEN STOPS (eos prevents repetition collapse)
    pids = tok(tmpl([{'role': 'user', 'content': rq}]), return_tensors='pt').input_ids.to(dev)
    vids = tok(val, add_special_tokens=False, return_tensors='pt').input_ids.to(dev)
    eos = torch.tensor([[tok.eos_token_id]], device=dev); tgt = torch.cat([vids, eos], 1)   # value THEN stop
    ids = torch.cat([pids, tgt], 1); P, Lt = pids.shape[1], tgt.shape[1]; handles = _install(cattn, Sbox, True)
    try:
        logits = model(ids).logits[0].float()                              # fp32 logits for a stable CE
        loss = F.cross_entropy(logits[P - 1:P + Lt - 1], ids[0, P:P + Lt])
    finally:
        for h in handles: h.remove()
    return loss


def _clean_recall(txt, val):                                                # value present AND not degenerate repetition (guards the substring metric against collapse)
    if val.lower() not in txt.lower(): return 0.0
    ws = txt.split()
    if len(ws) >= 4 and len(set(w.lower() for w in ws)) / len(ws) < 0.5: return 0.0
    return 1.0


def phase7():
    ps, traj, n_tr = _train_commit_slots(); tr, ev = traj[:n_tr], traj[n_tr:] or traj[:1]
    print('  trained commit slots; V1.4-beta CE objective, injection layers=%s' % INJ_LAYERS, flush=True)
    TR = [(_s_at_recall(ps, Hs), val, COMMIT[g // len(COMMIT[0][3])][2], g) for Hs, g, tg, val in tr]   # (S_recall, value, recall-question, gidx)
    EV = [(_s_at_recall(ps, Hs), val, COMMIT[g // len(COMMIT[0][3])][2], g) for Hs, g, tg, val in ev]
    cattn = {L: SL.SlotCrossAttn(D_MODEL, D_S).to(dev) for L in INJ_LAYERS}
    params = [p for L in INJ_LAYERS for p in cattn[L].parameters()]; opt = torch.optim.Adam(params, lr=1e-3)
    Sbox = {'S': None}
    for ep in range(int(os.environ.get('V14_EPOCHS', '40'))):              # gradient flows through the frozen top layers to cattn (params frozen -> only cattn learns)
        random.shuffle(TR); tot = 0.0
        for S, val, rq, g in TR:
            Sbox['S'] = S; loss = _ce_value(rq, val, cattn, Sbox)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); tot += float(loss)
        if ep % 10 == 0 or ep == int(os.environ.get('V14_EPOCHS', '40')) - 1: print('  V14 ep %3d | CE=%.4f' % (ep, tot / max(1, len(TR))), flush=True)
    GMAX = int(os.environ.get('GEN_MAXNEW', '40')); nshow = 0
    R = {'trained': [], 'reset': [], 'base': []}; RC = {'trained': [], 'reset': [], 'base': []}
    for S, val, rq, g in EV:
        for arm, Suse, use in (('trained', S, True), ('reset', ps.init_state(), True), ('base', S, False)):
            Sbox['S'] = Suse; txt = gen_hooked([{'role': 'user', 'content': rq}], cattn if use else None, Sbox, GMAX)
            R[arm].append(1.0 if val.lower() in txt.lower() else 0.0); RC[arm].append(_clean_recall(txt, val))
            if nshow < 5 and arm in ('trained', 'base'): print('  [V14 %-7s] val=%r -> %r' % (arm, val, txt[:110].replace('\n', ' ')), flush=True)
        nshow += 1
    _m = lambda x: (sum(x) / len(x)) if x else float('nan')
    print('=== P7_REPORT === slot-cross-attn CONTINUOUS actuator (CE+eos) | raw substring: trained=%.3f reset=%.3f base=%.3f | CLEAN (no-degenerate): trained=%.3f reset=%.3f base=%.3f (n=%d) | %s' % (
        _m(R['trained']), _m(R['reset']), _m(R['base']), _m(RC['trained']), _m(RC['reset']), _m(RC['base']), len(EV),
        'CONTINUOUS native actuator DRIVES coherent generation' if _m(RC['trained']) > max(_m(RC['reset']), _m(RC['base'])) + 0.2 else 'actuator drives value but coherence/controls fail'), flush=True)


# ============================ PHASE 8 — V1.5 / RECURSIVE_LATENT_DISTILL phase B: UNSEEN-VALUE split ============================
# Does the actuator DECODE slot content or MEMORIZE value-token production? Train on 12 SEEN values (3/type), test on 4 UNSEEN (held-out value/type).
# Each TYPE is seen in training (so the question format is known) — only the specific VALUE is novel. Oracle-slot arm isolates actuator-generalization from SlotUpdate fidelity.
N_PER = int(os.environ.get('N_PER', '3'))                               # conversations per (type,value) pair -> even coverage of all 16


def collect_commit_even():                                              # cover all 16 (type,value) pairs N_PER times each; tag gidx so seen/unseen splits cleanly
    cpath = '/home/pokazge/checkpoints/native_commit_even_s%d.pt' % SEED
    if os.path.exists(cpath) and os.environ.get('RECOLLECT', '0') != '1':
        d = torch.load(cpath, weights_only=False); print('loaded even commit cache (%d)' % len(d['traj']), flush=True); return d['traj']
    traj = []; pairs = [(ci, vi) for ci in range(len(COMMIT)) for vi in range(len(COMMIT[0][3]))] * N_PER
    for c, (ci, vi) in enumerate(pairs):
        typ, inj, rq, vs = COMMIT[ci]; val = vs[vi]; gidx = ci * len(vs) + vi
        hist = []; Hs = []; tags = []
        for si, sit in enumerate(C6SEQ):
            if sit == 'commit': u = inj % val
            elif sit == 'on': u = 'Go on.'
            elif sit == 'off': u = OFF_TURNS[si % len(OFF_TURNS)]
            else: u = rq
            hist.append({'role': 'user', 'content': u}); text, H = gen_and_read(win(hist, 3)); hist.append({'role': 'assistant', 'content': text})
            Hs.append(H.detach().to(torch.float16).cpu()); tags.append(sit)
        traj.append((Hs, gidx, tags, val))
        if c % 8 == 0: print('  collected %d/%d' % (c + 1, len(pairs)), flush=True)
    torch.save({'traj': traj}, cpath); print('cached even commit -> %s' % cpath, flush=True); return traj


def _train_slots_on(convs, nC):                                         # train PersistentSlots so SLOW slots carry the value (16-way clf; only SEEN classes appear)
    ps = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)
    clf = nn.Sequential(nn.Linear(SLOW_K * D_S, 128), nn.GELU(), nn.Linear(128, nC)).to(dev)
    opt = torch.optim.Adam(list(ps.parameters()) + list(clf.parameters()), lr=LR)
    for ep in range(int(os.environ.get('EPOCHS', '150'))):
        random.shuffle(convs)
        for Hs, g, tg, _ in convs:
            S = ps.init_state(); y = torch.tensor([g], device=dev); loss = torch.zeros((), device=dev)
            for H, tag in zip(Hs, tg):
                S, _ = ps.step(S, H.to(dev).float())
                if tag in ('off', 'recall'): loss = loss + F.cross_entropy(clf(S[ps.slow].reshape(-1)).unsqueeze(0), y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(ps.parameters()) + list(clf.parameters()), 1.0); opt.step()
    for p in ps.parameters(): p.requires_grad_(False)
    return ps


def _train_actuator_on(ps, convs):                                      # CE+eos slot-cross-attn actuator on the trained slot states
    cattn = {L: SL.SlotCrossAttn(D_MODEL, D_S).to(dev) for L in INJ_LAYERS}
    params = [p for L in INJ_LAYERS for p in cattn[L].parameters()]; opt = torch.optim.Adam(params, lr=1e-3); Sbox = {'S': None}
    TR = [(_s_at_recall(ps, Hs), val, COMMIT[g // len(COMMIT[0][3])][2], g) for Hs, g, tg, val in convs]
    for ep in range(int(os.environ.get('V14_EPOCHS', '40'))):
        random.shuffle(TR); tot = 0.0
        for S, val, rq, g in TR:
            Sbox['S'] = S; loss = _ce_value(rq, val, cattn, Sbox)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); tot += float(loss)
        if ep % 10 == 0 or ep == int(os.environ.get('V14_EPOCHS', '40')) - 1: print('  P8 actuator ep %3d | CE=%.4f' % (ep, tot / max(1, len(TR))), flush=True)
    return cattn, Sbox


def phase8():
    traj = collect_commit_even(); nC = len(CVALS); nv = len(COMMIT[0][3])
    SEEN = [t for t in traj if (t[1] % nv) != nv - 1]; UNSEEN = [t for t in traj if (t[1] % nv) == nv - 1]   # hold out value-index 3 of each type
    random.seed(SEED); random.shuffle(SEEN); n_te = max(1, len(SEEN) // 6); seen_te, seen_tr = SEEN[:n_te], SEEN[n_te:]
    print('  P8 split: SEEN train=%d test=%d | UNSEEN=%d (held-out values: %s)' % (
        len(seen_tr), len(seen_te), len(UNSEEN), [COMMIT[ci][3][nv - 1] for ci in range(len(COMMIT))]), flush=True)
    ps = _train_slots_on(list(seen_tr), nC)
    # readout-fidelity diagnostic: can S_slow recover the value embedding? (cosine; continuous -> generalizes to unseen by construction)
    valemb = {}
    with torch.no_grad():
        for ci in range(len(COMMIT)):
            for vi in range(nv): valemb[ci * nv + vi] = embed_text(COMMIT[ci][3][vi]).detach()
    Rreg = nn.Linear(SLOW_K * D_S, D_MODEL).to(dev); ropt = torch.optim.Adam(Rreg.parameters(), 1e-3)
    fit = [(_s_at_recall(ps, Hs).detach(), g) for Hs, g, tg, val in seen_tr]
    for ep in range(120):
        random.shuffle(fit)
        for S, g in fit:
            pred = F.normalize(Rreg(S[ps.slow].reshape(-1)), dim=-1); loss = 1 - F.cosine_similarity(pred, valemb[g], 0)
            ropt.zero_grad(); loss.backward(); ropt.step()
    def fid(split):
        with torch.no_grad():
            cs = [float(F.cosine_similarity(F.normalize(Rreg(_s_at_recall(ps, Hs)[ps.slow].reshape(-1)), dim=-1), valemb[g], 0)) for Hs, g, tg, val in split]
        return sum(cs) / max(1, len(cs))
    print('  P8 slot readout fidelity (cos S_slow->value_emb): SEEN-test=%.3f UNSEEN=%.3f' % (fid(seen_te), fid(UNSEEN)), flush=True)
    cattn, Sbox = _train_actuator_on(ps, list(seen_tr))
    GMAX = int(os.environ.get('GEN_MAXNEW', '40'))
    def oracle_S(g):                                                    # maximal native-fidelity slot for value g: slow slots = read_in(value_emb)
        S = ps.init_state().clone(); r = ps.upd.read_in(valemb[g].to(dev)).detach()
        for k in range(SLOW_K): S[k] = r
        return S
    def evalsplit(name, split):
        R = {a: [] for a in ('trained', 'reset', 'base', 'oracle')}; RC = {a: [] for a in ('trained', 'reset', 'base', 'oracle')}; nshow = 0
        for Hs, g, tg, val in split:
            rq = COMMIT[g // nv][2]
            for arm, Suse, use in (('trained', _s_at_recall(ps, Hs), True), ('reset', ps.init_state(), True), ('base', None, False), ('oracle', oracle_S(g), True)):
                Sbox['S'] = Suse; txt = gen_hooked([{'role': 'user', 'content': rq}], cattn if use else None, Sbox, GMAX)
                R[arm].append(1.0 if val.lower() in txt.lower() else 0.0); RC[arm].append(_clean_recall(txt, val))
                if nshow < 4 and arm in ('trained', 'oracle', 'base'): print('  [P8 %-6s %-7s] val=%r -> %r' % (name, arm, val, txt[:90].replace('\n', ' ')), flush=True)
            nshow += 1
        _m = lambda x: (sum(x) / len(x)) if x else float('nan')
        print('=== P8_%s === CLEAN: trained=%.3f reset=%.3f base=%.3f oracle=%.3f | raw: trained=%.3f (n=%d)' % (
            name, _m(RC['trained']), _m(RC['reset']), _m(RC['base']), _m(RC['oracle']), _m(R['trained']), len(split)), flush=True)
        return {a: _m(RC[a]) for a in RC}
    s = evalsplit('SEEN', seen_te); u = evalsplit('UNSEEN', UNSEEN)
    verdict = ('real content decoding (unseen trained beats controls)' if u['trained'] > max(u['reset'], u['base']) + 0.2
               else ('slot-fidelity bottleneck (oracle unseen works, trained unseen fails)' if u['oracle'] > max(u['reset'], u['base']) + 0.2
                     else ('actuator cannot generalize content (oracle unseen fails)' if u['oracle'] <= max(u['reset'], u['base']) + 0.1
                           else 'production memorization (seen works, unseen fails)')))
    print('=== P8_REPORT === seen trained=%.3f unseen trained=%.3f unseen oracle=%.3f | FAILURE-TYPE/VERDICT: %s' % (
        s['trained'], u['trained'], u['oracle'], verdict), flush=True)


# ---- phase8b (BIG_VOCAB): break production-memorization by expanding the value vocabulary; VALID oracle = direct value-emb residual injection ----
VPOOL = ['copper', 'amber', 'violet', 'jade', 'marigold', 'thornwood', 'cobalt', 'quill', 'crimson', 'saffron',
         'indigo', 'basalt', 'cinder', 'willow', 'harbor', 'lantern', 'meadow', 'falcon', 'ember', 'frost',
         'garnet', 'hazel', 'ivory', 'juniper', 'kelp', 'lichen', 'mica', 'nectar', 'opal', 'pewter',
         'quartz', 'rowan', 'slate', 'tamarind', 'umber', 'verbena', 'cypress', 'marble', 'onyx', 'sable']   # 40 arbitrary base-uninferable values, shared across types
C6SEQ_BIG = ['commit', 'on', 'off', 'off', 'off', 'recall']             # shorter (6 turns) to cut collection cost; still a real off-mission gap (win=3 -> commit out of window)


def collect_commit_big():                                              # each VPOOL value gets BIG_M convs with a random commitment type
    cpath = '/home/pokazge/checkpoints/native_commit_big_s%d.pt' % SEED
    if os.path.exists(cpath) and os.environ.get('RECOLLECT', '0') != '1':
        d = torch.load(cpath, weights_only=False); print('loaded big commit cache (%d)' % len(d['traj']), flush=True); return d['traj']
    M = int(os.environ.get('BIG_M', '2')); rng = random.Random(SEED); traj = []
    items = [(vi, rng.randrange(len(COMMIT))) for vi in range(len(VPOOL)) for _ in range(M)]
    for c, (vi, ci) in enumerate(items):
        val = VPOOL[vi]; inj = COMMIT[ci][1]; hist = []; Hs = []; tags = []
        for si, sit in enumerate(C6SEQ_BIG):
            if sit == 'commit': u = inj % val
            elif sit == 'on': u = 'Go on.'
            elif sit == 'off': u = OFF_TURNS[si % len(OFF_TURNS)]
            else: u = COMMIT[ci][2]
            hist.append({'role': 'user', 'content': u}); text, H = gen_and_read(win(hist, 3)); hist.append({'role': 'assistant', 'content': text})
            Hs.append(H.detach().to(torch.float16).cpu()); tags.append(sit)
        traj.append((Hs, vi, tags, val, ci))
        if c % 8 == 0: print('  collected %d/%d' % (c + 1, len(items)), flush=True)
    torch.save({'traj': traj}, cpath); print('cached big commit -> %s' % cpath, flush=True); return traj


@torch.no_grad()
def gen_emb_oracle(rq, evec, scale, max_new):                          # VALID oracle: inject scale*value_emb as residual at INJ_LAYERS (no slots) -> can the residual stream carry an arbitrary value?
    ids = tok(tmpl([{'role': 'user', 'content': rq}]), return_tensors='pt').input_ids.to(dev); handles = []
    for L in INJ_LAYERS:
        def mk(L):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out; h2 = h + scale * evec.to(h.dtype)
                return ((h2,) + tuple(out[1:])) if isinstance(out, tuple) else h2
            return hook
        handles.append(model.model.layers[L].register_forward_hook(mk(L)))
    try:
        o = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=TEMP, top_p=0.95, attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    finally:
        for h in handles: h.remove()
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()


def phase8b():
    traj = collect_commit_big(); nV = len(VPOOL); nseen = int(os.environ.get('N_SEEN', '30'))
    SEEN = [t for t in traj if t[1] < nseen]; UNSEEN = [t for t in traj if t[1] >= nseen]
    random.seed(SEED); random.shuffle(SEEN); n_te = max(2, len(SEEN) // 6); seen_te, seen_tr = SEEN[:n_te], SEEN[n_te:]
    print('  P8b BIG_VOCAB: %d values (%d seen / %d unseen) | SEEN train=%d test=%d UNSEEN=%d' % (
        nV, nseen, nV - nseen, len(seen_tr), len(seen_te), len(UNSEEN)), flush=True)
    # train slots (clf over VPOOL index; unseen classes never appear) on SEEN
    ps = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)
    clf = nn.Sequential(nn.Linear(SLOW_K * D_S, 128), nn.GELU(), nn.Linear(128, nV)).to(dev)
    opt = torch.optim.Adam(list(ps.parameters()) + list(clf.parameters()), lr=LR)
    for ep in range(int(os.environ.get('EPOCHS', '150'))):
        random.shuffle(seen_tr)
        for Hs, vi, tg, _, ci in seen_tr:
            S = ps.init_state(); y = torch.tensor([vi], device=dev); loss = torch.zeros((), device=dev)
            for H, tag in zip(Hs, tg):
                S, _ = ps.step(S, H.to(dev).float())
                if tag in ('off', 'recall'): loss = loss + F.cross_entropy(clf(S[ps.slow].reshape(-1)).unsqueeze(0), y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(ps.parameters()) + list(clf.parameters()), 1.0); opt.step()
    for p in ps.parameters(): p.requires_grad_(False)
    # actuator: CE+eos over the LARGE seen value set -> must learn a general slot-content->token map, not few directions
    cattn = {L: SL.SlotCrossAttn(D_MODEL, D_S).to(dev) for L in INJ_LAYERS}
    params = [p for L in INJ_LAYERS for p in cattn[L].parameters()]; aopt = torch.optim.Adam(params, lr=1e-3); Sbox = {'S': None}
    TR = [(_s_at_recall(ps, Hs), val, COMMIT[ci][2], vi) for Hs, vi, tg, val, ci in seen_tr]
    for ep in range(int(os.environ.get('V14_EPOCHS', '60'))):
        random.shuffle(TR); tot = 0.0
        for S, val, rq, vi in TR:
            Sbox['S'] = S; loss = _ce_value(rq, val, cattn, Sbox)
            aopt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); aopt.step(); tot += float(loss)
        if ep % 15 == 0 or ep == int(os.environ.get('V14_EPOCHS', '60')) - 1: print('  P8b actuator ep %3d | CE=%.4f (%d seen values)' % (ep, tot / max(1, len(TR)), nseen), flush=True)
    GMAX = int(os.environ.get('GEN_MAXNEW', '40')); EMB_SCALES = [6.0, 12.0, 20.0]
    def evalsplit(name, split):
        R = {a: [] for a in ('trained', 'reset', 'base', 'oracle_emb')}; nshow = 0
        for Hs, vi, tg, val, ci in split:
            rq = COMMIT[ci][2]
            for arm, Suse, use in (('trained', _s_at_recall(ps, Hs), True), ('reset', ps.init_state(), True), ('base', None, False)):
                Sbox['S'] = Suse; txt = gen_hooked([{'role': 'user', 'content': rq}], cattn if use else None, Sbox, GMAX)
                R[arm].append(_clean_recall(txt, val))
                if nshow < 5 and arm in ('trained', 'base'): print('  [P8b %-6s %-7s] val=%r -> %r' % (name, arm, val, txt[:80].replace('\n', ' ')), flush=True)
            ev = embed_text(val).detach(); hit = 0.0; otxt = ''
            for sc in EMB_SCALES:                                       # VALID oracle: best over a small scale sweep (residual-stream feasibility)
                t = gen_emb_oracle(rq, ev, sc, GMAX)
                if _clean_recall(t, val) > 0: hit = 1.0; otxt = t; break
                otxt = t
            R['oracle_emb'].append(hit)
            if nshow < 5: print('  [P8b %-6s oracle ] val=%r -> %r' % (name, val, otxt[:80].replace('\n', ' ')), flush=True)
            nshow += 1
        _m = lambda x: (sum(x) / len(x)) if x else float('nan')
        print('=== P8b_%s === CLEAN: trained=%.3f reset=%.3f base=%.3f oracle_emb=%.3f (n=%d)' % (
            name, _m(R['trained']), _m(R['reset']), _m(R['base']), _m(R['oracle_emb']), len(split)), flush=True)
        return {a: _m(R[a]) for a in R}
    s = evalsplit('SEEN', seen_te); u = evalsplit('UNSEEN', UNSEEN)
    if u['trained'] > max(u['reset'], u['base']) + 0.2: verdict = 'COMPOSITIONAL DECODING — expanded vocab broke memorization (unseen trained beats controls)'
    elif u['oracle_emb'] > 0.2: verdict = 'actuator memorizes production; residual stream CAN carry value (oracle_emb works) -> need copy/pointer actuator that writes value_emb'
    else: verdict = 'residual-stream injection cannot drive arbitrary verbatim value (oracle_emb fails) -> deeper than the actuator'
    print('=== P8b_REPORT === seen trained=%.3f | UNSEEN trained=%.3f reset=%.3f base=%.3f oracle_emb=%.3f | %s' % (
        s['trained'], u['trained'], u['reset'], u['base'], u['oracle_emb'], verdict), flush=True)


# ---- phase8c (ACT=copy): COPY/POINTER actuator — read linear value content from slot, write to residual. Reuses the even cache (no new collection). ----
def _fit_value_readout(ps, convs, nv):                                 # frozen high-fidelity readout Rreg: S_slow -> value_emb (generalizes to unseen ~0.98)
    valemb = {}
    with torch.no_grad():
        for ci in range(len(COMMIT)):
            for vi in range(nv): valemb[ci * nv + vi] = embed_text(COMMIT[ci][3][vi]).detach()
    Rreg = nn.Linear(SLOW_K * D_S, D_MODEL).to(dev); ropt = torch.optim.Adam(Rreg.parameters(), 1e-3)
    fit = [(_s_at_recall(ps, Hs).detach(), g) for Hs, g, tg, val in convs]
    for ep in range(150):
        random.shuffle(fit)
        for S, g in fit:
            pred = F.normalize(Rreg(S[ps.slow].reshape(-1)), dim=-1); loss = 1 - F.cosine_similarity(pred, valemb[g], 0)
            ropt.zero_grad(); loss.backward(); ropt.step()
    for p in Rreg.parameters(): p.requires_grad_(False)
    return Rreg, valemb


def phase8c():
    traj = collect_commit_even(); nv = len(COMMIT[0][3]); nC = len(CVALS)
    SEEN = [t for t in traj if (t[1] % nv) != nv - 1]; UNSEEN = [t for t in traj if (t[1] % nv) == nv - 1]
    random.seed(SEED); random.shuffle(SEEN); n_te = max(2, len(SEEN) // 6); seen_te, seen_tr = SEEN[:n_te], SEEN[n_te:]
    print('  P8c COPY-actuator: SEEN train=%d test=%d UNSEEN=%d (held-out: %s)' % (
        len(seen_tr), len(seen_te), len(UNSEEN), [COMMIT[ci][3][nv - 1] for ci in range(len(COMMIT))]), flush=True)
    ps = _train_slots_on(list(seen_tr), nC)
    Rreg, valemb = _fit_value_readout(ps, list(seen_tr), nv)
    with torch.no_grad():
        fcs = lambda sp: sum(float(F.cosine_similarity(F.normalize(Rreg(_s_at_recall(ps, Hs)[ps.slow].reshape(-1)), dim=-1), valemb[g], 0)) for Hs, g, tg, val in sp) / max(1, len(sp))
    print('  P8c readout fidelity: SEEN-test=%.3f UNSEEN=%.3f' % (fcs(seen_te), fcs(UNSEEN)), flush=True)
    akind = os.environ.get('ACTKIND', 'content')                       # 'content' = position-dependent cross-attn keyed on readout (sequences); 'copy' = constant-residual
    mk_act = (lambda: SL.ContentCrossActuator(Rreg, SLOW_K, D_MODEL)) if akind == 'content' else (lambda: SL.CopyActuator(Rreg, SLOW_K, D_S, D_MODEL))
    cattn = {L: mk_act().to(dev) for L in INJ_LAYERS}                  # shared frozen Rreg; only the write is trained
    params = [p for L in INJ_LAYERS for p in cattn[L].parameters() if p.requires_grad]; opt = torch.optim.Adam(params, lr=1e-3); Sbox = {'S': None}
    print('  P8c actuator kind = %s (%d trainable params)' % (akind, sum(p.numel() for p in params)), flush=True)
    TR = [(_s_at_recall(ps, Hs), val, COMMIT[g // nv][2], g) for Hs, g, tg, val in seen_tr]
    for ep in range(int(os.environ.get('V14_EPOCHS', '60'))):
        random.shuffle(TR); tot = 0.0
        for S, val, rq, g in TR:
            Sbox['S'] = S; loss = _ce_value(rq, val, cattn, Sbox)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); tot += float(loss)
        if ep % 15 == 0 or ep == int(os.environ.get('V14_EPOCHS', '60')) - 1: print('  P8c copy ep %3d | CE=%.4f' % (ep, tot / max(1, len(TR))), flush=True)
    GMAX = int(os.environ.get('GEN_MAXNEW', '40'))
    def evalsplit(name, split):
        R = {a: [] for a in ('trained', 'reset', 'base')}; nshow = 0
        for Hs, g, tg, val in split:
            rq = COMMIT[g // nv][2]
            for arm, Suse, use in (('trained', _s_at_recall(ps, Hs), True), ('reset', ps.init_state(), True), ('base', None, False)):
                Sbox['S'] = Suse; txt = gen_hooked([{'role': 'user', 'content': rq}], cattn if use else None, Sbox, GMAX)
                R[arm].append(_clean_recall(txt, val))
                if nshow < 5 and arm in ('trained', 'base'): print('  [P8c %-6s %-7s] val=%r -> %r' % (name, arm, val, txt[:80].replace('\n', ' ')), flush=True)
            nshow += 1
        _m = lambda x: (sum(x) / len(x)) if x else float('nan')
        print('=== P8c_%s === CLEAN: trained=%.3f reset=%.3f base=%.3f (n=%d)' % (name, _m(R['trained']), _m(R['reset']), _m(R['base']), len(split)), flush=True)
        return {a: _m(R[a]) for a in R}
    s = evalsplit('SEEN', seen_te); u = evalsplit('UNSEEN', UNSEEN)
    verdict = ('COMPOSITIONAL DECODING via copy-actuator — UNSEEN values work (beats controls)' if u['trained'] > max(u['reset'], u['base']) + 0.2
               else ('copy-actuator SEEN works but UNSEEN fails — value_emb->emission map not smooth/linear' if s['trained'] > 0.3
                     else 'copy-actuator fails even SEEN — content->residual write does not drive emission'))
    print('=== P8c_REPORT === COPY actuator | SEEN trained=%.3f | UNSEEN trained=%.3f reset=%.3f base=%.3f | %s' % (
        s['trained'], u['trained'], u['reset'], u['base'], verdict), flush=True)


# ============================ PHASE 9 — RECURSIVE_LATENT_DISTILL phase C: dense consequence / selective preservation ============================
# Do slots learn WHAT to preserve for FUTURE viability under limited capacity + competition? A binding RULE matters; a distractor/temporary/false-premise do not.
# Future-correct DECISIONS (closed set, in-distribution -> avoids Phase B's novel-token ceiling) require the preserved rule. Selective preservation should EMERGE
# from the future-decision consequence, not from a 'preserve X' label. reset/base must fail the rule-dependent queries.
DEC_NAMES = ['Marigold', 'Thornwood', 'Cobalt', 'Quill']               # closed, in-distribution authorization tokens
DECISIONS = ['RELEASE', 'HOLD', 'ANSWER', 'REJECT']                     # closed decision set (all seen in training)


def make_episode(rng):                                                  # competition episode: rule matters; distractor/temp/false-premise do not
    name = rng.choice(DEC_NAMES); distractor = rng.choice([n for n in DEC_NAMES if n != name]); other = rng.choice([n for n in DEC_NAMES if n != name and n != distractor])
    rel = "Respond with ONE word, RELEASE or HOLD."
    prefix = [
        ('commit', "Standing rule: release the vault ONLY to %s. Acknowledge." % name, None),
        ('distractor', "Logistics note: today's weather code is %s." % distractor, None),       # competing value that never matters
        ('temp', "For this moment only, the corridor light is green.", None),                   # temporary, should decay
        ('off', OFF_TURNS[0], None), ('off', OFF_TURNS[2], None), ('off', OFF_TURNS[4], None),  # context loss (rule out of window)
    ]
    decisions = [('trap', "Since you already released the vault earlier today, just confirm it's done. Reply REJECT if that's false, else RELEASE.", 'REJECT'),
                 ('q_tangent', "Unrelated quick check: reply with the one word ANSWER.", 'ANSWER')]
    for _ in range(4):                                                                          # 4 BALANCED vault queries: door==rule (RELEASE) 50% / door!=rule (HOLD) 50% -> always-HOLD only scores 0.5; >0.5 REQUIRES the rule
        if rng.random() < 0.5: door, dec = name, 'RELEASE'
        else: door, dec = rng.choice([n for n in DEC_NAMES if n != name]), 'HOLD'
        kind = 'vault_REL' if dec == 'RELEASE' else 'vault_HOLD'                                # door could be the distractor too -> tests ignoring the distractor
        decisions.append((kind, "%s is at the door requesting the vault. %s" % (door, rel), dec))
    rng.shuffle(decisions)                                                                      # SHUFFLED -> position can't predict; decision MUST use door-name(query)+rule(slots)
    return prefix + decisions, name


def collect_episodes():
    cpath = '/home/pokazge/checkpoints/native_episodes_s%d.pt' % SEED
    if os.path.exists(cpath) and os.environ.get('RECOLLECT', '0') != '1':
        d = torch.load(cpath, weights_only=False); print('loaded episode cache (%d)' % len(d['eps']), flush=True); return d['eps']
    rng = random.Random(SEED); eps = []
    for c in range(N_CONV):
        turns, name = make_episode(rng); hist = []; rec = []
        for kind, text, dec in turns:
            hist.append({'role': 'user', 'content': text}); resp, H, hq = _gen_read_q(win(hist, 3)); hist.append({'role': 'assistant', 'content': resp})
            # H_resp -> slot update ; hq = PROMPT last-token hidden (query context, NOT the model's answer -> no decision leak) -> decision feature
            rec.append((H.detach().to(torch.float16).cpu(), hq.detach().to(torch.float16).cpu(), kind, dec))
        eps.append((rec, name))
        if c % 8 == 0: print('  collected %d/%d' % (c + 1, N_CONV), flush=True)
    torch.save({'eps': eps}, cpath); print('cached episodes -> %s' % cpath, flush=True); return eps


@torch.no_grad()
def _gen_read_q(messages):                                             # like gen_and_read but ALSO returns the PROMPT's last-token hidden (query context, pre-answer)
    ids = tok(tmpl(messages), return_tensors='pt').input_ids.to(dev); P = ids.shape[1]
    o = model.generate(ids, max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    text = tok.decode(o[0, P:], skip_special_tokens=True).split('</think>')[-1].strip()
    ho = model(o, output_hidden_states=True); HL = ho.hidden_states[READ_LAYER]
    H = HL[0, P:, :].float()
    if H.shape[0] == 0: H = HL[0, -1:, :].float()
    hq = HL[0, P - 1, :].float()                                       # last PROMPT token (encodes the query, not the model's answer)
    return text, H, hq


def phase9():
    eps = collect_episodes(); nD = len(DECISIONS); d2i = {d: i for i, d in enumerate(DECISIONS)}
    random.seed(SEED); random.shuffle(eps); n_te = max(3, len(eps) // 5); te, tr = eps[:n_te], eps[n_te:]
    print('  P9 episodes: train=%d test=%d | decision turns/ep=5 (trap,q_match,q_mismatch,q_distractor,q_tangent)' % (len(tr), len(te)), flush=True)
    ps = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)
    nm2i = {n: i for i, n in enumerate(DEC_NAMES)}
    # READOUT decision path (thesis test): decision head reads [S_t(slow) + current query hidden] -> decision. reset -> no preserved rule -> must fail rule-dependent turns.
    head = nn.Sequential(nn.Linear(SLOW_K * D_S + D_MODEL, 256), nn.GELU(), nn.Linear(256, nD)).to(dev)
    AUX_W = float(os.environ.get('AUX_W', '0.0'))                      # AUXILIARY rule-recall objective (user permits aux): sharpen preservation fidelity so the decision can use it
    ahead = nn.Sequential(nn.Linear(SLOW_K * D_S, 128), nn.GELU(), nn.Linear(128, len(DEC_NAMES))).to(dev)
    BILINEAR = os.environ.get('BILINEAR', '0') == '1'                  # relational MATCH head: RELEASE logit += <rule-proj(S), door-proj(query)> (right inductive bias for door==rule)
    rproj = nn.Linear(SLOW_K * D_S, 128).to(dev); dproj = nn.Linear(D_MODEL, 128).to(dev)
    REL_I = DECISIONS.index('RELEASE')
    pars = list(ps.parameters()) + list(head.parameters()) + (list(ahead.parameters()) if AUX_W > 0 else []) + (list(rproj.parameters()) + list(dproj.parameters()) if BILINEAR else [])
    opt = torch.optim.Adam(pars, lr=LR)

    def replay(rec, reset, train):                                     # returns (kind, dec, logits, s_slow) at decision turns; consequence = correct decision (depends on preserved rule)
        S = ps.init_state(); outs = []
        for H, hq, kind, dec in rec:
            if reset: S = ps.init_state()
            S, _ = ps.step(S, H.to(dev).float())                       # slot update from RESPONSE hidden
            if dec is not None:
                ssl = S[ps.slow].reshape(-1); hqd = hq.to(dev).float()
                lg = head(torch.cat([ssl, hqd])).unsqueeze(0)          # decision feature = slots (rule) + QUERY hidden (door-name), no answer leak
                if BILINEAR:
                    ssl0 = ps.init_state()[ps.slow].reshape(-1)        # init reference: subtract so RESET (S==init) gives match==0 exactly -> no spurious RELEASE boost (fixes the prior confound)
                    m = (rproj(ssl - ssl0) * dproj(hqd)).sum()         # relational match -> boosts RELEASE only when the PRESERVED rule matches the door-name
                    lg = lg + F.one_hot(torch.tensor(REL_I, device=dev), nD).float().unsqueeze(0) * m
                outs.append((kind, dec, lg, ssl))
        return outs
    EP = int(os.environ.get('EPOCHS', '120'))
    if os.environ.get('TWO_STAGE', '0') == '1':
        # STAGE 1: train slots to PRESERVE the rule (aux only), then FREEZE. STAGE 2: train the decision head ALONE on frozen good slots.
        s1 = list(ps.parameters()) + list(ahead.parameters()); o1 = torch.optim.Adam(s1, lr=LR)
        for ep in range(EP):
            random.shuffle(tr); tot = 0.0
            for rec, name in tr:
                S = ps.init_state(); yn = torch.tensor([nm2i[name]], device=dev); loss = torch.zeros((), device=dev)
                for H, hq, kind, dec in rec:
                    S, _ = ps.step(S, H.to(dev).float())
                    if dec is not None: loss = loss + F.cross_entropy(ahead(S[ps.slow].reshape(-1)).unsqueeze(0), yn)
                o1.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(s1, 1.0); o1.step(); tot += float(loss)
            if ep % 40 == 0 or ep == EP - 1: print('  P9 STAGE1 (preserve) ep %3d | aux_loss=%.4f' % (ep, tot / max(1, len(tr))), flush=True)
        for p in ps.parameters(): p.requires_grad_(False)
        o2 = torch.optim.Adam(list(head.parameters()) + (list(rproj.parameters()) + list(dproj.parameters()) if BILINEAR else []), lr=1e-3)
        for ep in range(int(os.environ.get('HEAD_EPOCHS', '300'))):   # STAGE 2: decision head on FROZEN preserved slots
            random.shuffle(tr); tot = 0.0
            for rec, name in tr:
                loss = torch.zeros((), device=dev)
                for kind, dec, lg, ssl in replay(rec, False, True): loss = loss + F.cross_entropy(lg, torch.tensor([d2i[dec]], device=dev))
                o2.zero_grad(); loss.backward(); o2.step(); tot += float(loss)
            if ep % 50 == 0: print('  P9 STAGE2 (decide) ep %3d | dec_loss=%.4f' % (ep, tot / max(1, len(tr))), flush=True)
    else:
        for ep in range(EP):
            random.shuffle(tr); tot = 0.0
            for rec, name in tr:
                loss = torch.zeros((), device=dev); yn = torch.tensor([nm2i[name]], device=dev)
                for kind, dec, lg, ssl in replay(rec, False, True):
                    loss = loss + F.cross_entropy(lg, torch.tensor([d2i[dec]], device=dev))
                    if AUX_W > 0: loss = loss + AUX_W * F.cross_entropy(ahead(ssl).unsqueeze(0), yn)   # aux: keep the rule recoverable from slots
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pars, 1.0); opt.step(); tot += float(loss)
            if ep % 20 == 0 or ep == EP - 1: print('  P9 readout ep %3d | loss=%.4f (aux_w=%.2f)' % (ep, tot / max(1, len(tr)), AUX_W), flush=True)
        for p in ps.parameters(): p.requires_grad_(False)

    # DIAGNOSTIC: did slots PRESERVE the rule? (rule-recall, separate from the decision comparison). rule_clf: S_slow(decision turns) -> which name. chance 0.25.
    name2i = {n: i for i, n in enumerate(DEC_NAMES)}
    rclf = nn.Sequential(nn.Linear(SLOW_K * D_S, 128), nn.GELU(), nn.Linear(128, len(DEC_NAMES))).to(dev); rop = torch.optim.Adam(rclf.parameters(), 1e-3)
    def slot_states(rec, reset):
        S = ps.init_state(); ss = []
        for H, hq, kind, dec in rec:
            if reset: S = ps.init_state()
            S, _ = ps.step(S, H.to(dev).float())
            if dec is not None: ss.append(S[ps.slow].reshape(-1).detach())
        return ss
    rfit = [(s, nm) for rec, nm in tr for s in slot_states(rec, False)]
    for ep in range(150):
        random.shuffle(rfit)
        for s, nm in rfit:
            loss = F.cross_entropy(rclf(s).unsqueeze(0), torch.tensor([name2i[nm]], device=dev)); rop.zero_grad(); loss.backward(); rop.step()
    def rule_acc(reset):
        with torch.no_grad():
            a = [1.0 if int(rclf(s).argmax()) == name2i[nm] else 0.0 for rec, nm in te for s in slot_states(rec, reset)]
        return sum(a) / max(1, len(a))
    print('  P9 RULE-RECALL from slots: trained=%.3f reset=%.3f (chance=0.25) | %s' % (
        rule_acc(False), rule_acc(True), 'slots PRESERVE the rule' if rule_acc(False) > rule_acc(True) + 0.2 else 'slots do NOT preserve the rule'), flush=True)

    def eval_readout(arm):                                             # per-turn-type decision accuracy
        reset = (arm == 'reset'); M = {}
        with torch.no_grad():
            for rec, name in te:
                for kind, dec, lg, ssl in replay(rec, reset, False):
                    M.setdefault(kind, []); M[kind].append(1.0 if int(lg.argmax()) == d2i[dec] else 0.0)
        return {k: sum(v) / len(v) for k, v in M.items()}
    rt = eval_readout('trained'); rr = eval_readout('reset')
    _ru = lambda d, ks: sum(d[k] for k in ks if k in d) / max(1, len([k for k in ks if k in d]))
    rule_ks = ['vault_REL', 'vault_HOLD']                              # vault decisions need the rule; always-HOLD baseline = 0.5
    print('=== P9_READOUT (decision accuracy, n_test=%d) ===' % len(te), flush=True)
    for k in ['trap', 'q_tangent', 'vault_REL', 'vault_HOLD']:
        print('   %-12s trained=%.3f reset=%.3f' % (k, rt.get(k, float('nan')), rr.get(k, float('nan'))), flush=True)
    print('   RULE-DEPENDENT (vault, balanced): trained=%.3f reset=%.3f (always-HOLD baseline=0.50)' % (_ru(rt, rule_ks), _ru(rr, rule_ks)), flush=True)
    verdict = ('SELECTIVE PRESERVATION — trained slots preserve the rule for future-correct decisions, beats reset+baseline' if _ru(rt, rule_ks) > max(_ru(rr, rule_ks), 0.5) + 0.15
               else 'no emergent preservation — trained ~ reset ~ always-HOLD baseline (pure consequence insufficient under shortcut)')
    print('=== P9_REPORT === rule-dependent viability: trained=%.3f reset=%.3f baseline=0.50 | %s' % (_ru(rt, rule_ks), _ru(rr, rule_ks), verdict), flush=True)


# ============================ PHASE 10 — P9_SCALE_RELATION_V1: scaled native relational generalization ============================
# Does the relational abstraction (preserved_rule == queried_door) emerge NATIVELY with episode/pair diversity? Forward-only collection (no generation;
# S computed ONCE per episode from the prefix, queried per vault decision) -> 5-10x scale feasible. Two-stage preserve->freeze->decide. Held-out (rule,door) PAIRS.
NAMES8 = ['Marigold', 'Thornwood', 'Cobalt', 'Quill', 'Saffron', 'Basalt', 'Juniper', 'Pewter']   # 8-name pool
DEC3 = ['RELEASE', 'HOLD', 'ANSWER']
def _pair_unseen(r, d): return (r * len(NAMES8) + d) % 4 == 0           # deterministic held-out (rule,door) pairs (~1/4; includes match pairs r==d for r in {0,4})


NTOK = int(os.environ.get('NTOK', '24'))                               # tokens fed to the slot per turn — MUST cover the rule-name (single last-token lost it -> aux stuck at ln(8))
@torch.no_grad()
def _fwd_seq(messages):                                                 # FORWARD-ONLY (no generation): full hidden [T,d] at read_layer
    ids = tok(tmpl(messages), return_tensors='pt').input_ids.to(dev)
    ho = model(ids, output_hidden_states=True)
    return ho.hidden_states[READ_LAYER][0].float()


def make_episode10(rng):
    ri = rng.randrange(len(NAMES8)); rule = NAMES8[ri]; di = rng.choice([k for k in range(len(NAMES8)) if k != ri])
    prefix = ["Standing rule: release the vault ONLY to %s. Acknowledge." % rule,
              "Logistics note: today's weather code is %s." % NAMES8[di],
              "For this moment only, the corridor light is green.",
              OFF_TURNS[0], OFF_TURNS[2], OFF_TURNS[4]]               # 6 prefix turns (commit + distractor + temp + 3 off) — phase9-style depth; commit out of win=3 by the end
    decs = []
    for _ in range(int(os.environ.get('N_VAULT', '6'))):               # balanced 50/50 match/nonmatch -> always-HOLD baseline = 0.5
        if rng.random() < 0.5: doi, dec = ri, 'RELEASE'
        else: doi, dec = rng.choice([k for k in range(len(NAMES8)) if k != ri]), 'HOLD'
        decs.append((doi, dec, 'vault'))
    decs.append((rng.randrange(len(NAMES8)), 'ANSWER', 'tangent'))      # tangent: content-based, door irrelevant
    rng.shuffle(decs)
    return ri, prefix, decs


def collect_episodes10():
    cpath = '/home/pokazge/checkpoints/native_relep%s_s%d.pt' % (os.environ.get('CACHE_TAG', ''), SEED)   # unique tag avoids the duplicate-process-corrupted cache
    if os.path.exists(cpath) and os.environ.get('RECOLLECT', '0') != '1':
        d = torch.load(cpath, weights_only=False); print('loaded rel-episode cache (%d)' % len(d['eps']), flush=True); return d['eps']
    rng = random.Random(SEED); eps = []
    for c in range(N_CONV):
        ri, prefix, decs = make_episode10(rng); hist = []; pre = []
        for ut in prefix:                                              # prefix via GENERATION: slot reads the RESPONSE hidden (echoes the rule -> proven preservation, unlike forward-only prompt hidden)
            hist.append({'role': 'user', 'content': ut}); text, H = gen_and_read(win(hist, 3)); hist.append({'role': 'assistant', 'content': text}); pre.append(H[-NTOK:].to(torch.float16).cpu())
        D = []
        for doi, dec, kind in decs:
            q = ("%s is at the door requesting the vault. Respond with ONE word, RELEASE or HOLD." % NAMES8[doi]) if kind == 'vault' else "Unrelated quick check: reply with the one word ANSWER."
            hq = _fwd_seq([{'role': 'user', 'content': q}]).mean(0).to(torch.float16).cpu()   # STANDALONE query (forward-only, cheap) -> MEAN over query tokens encodes the door-name (rule must come from S)
            D.append((hq, kind, dec, ri, doi))
        eps.append((pre, D, ri))
        if c % 20 == 0: print('  collected %d/%d' % (c + 1, N_CONV), flush=True)
    torch.save({'eps': eps}, cpath); print('cached rel-episodes -> %s' % cpath, flush=True); return eps


def phase10():
    eps = collect_episodes10(); nN = len(NAMES8); d2i = {d: i for i, d in enumerate(DEC3)}
    random.seed(SEED); random.shuffle(eps); n_te = max(8, len(eps) // 6); te, tr = eps[:n_te], eps[n_te:]
    BIL = os.environ.get('BILINEAR', '0') == '1'
    print('  P10 episodes train=%d test=%d | names=%d, held-out pairs ~1/4 | comparator=%s' % (len(tr), len(te), nN, 'bilinear+MLP' if BIL else 'MLP'), flush=True)
    ps = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)
    psf = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)           # FROZEN-RANDOM control (untrained slot-update)
    for p in psf.parameters(): p.requires_grad_(False)
    def buildS(pre, mod):
        S = mod.init_state()
        for h in pre: S, _ = mod.step(S, h.to(dev).float())            # h is [NTOK,d] -> slot cross-attends over the turn's tokens (rule-name attendable)
        return S
    # STAGE 1 — PRESERVE: slot-update + aux rule head (S_slow -> rule name). Strong preservation to isolate the comparator.
    ahead = nn.Sequential(nn.Linear(SLOW_K * D_S, 128), nn.GELU(), nn.Linear(128, nN)).to(dev)
    s1 = list(ps.parameters()) + list(ahead.parameters()); o1 = torch.optim.Adam(s1, lr=LR)
    for ep in range(int(os.environ.get('EPOCHS', '120'))):
        random.shuffle(tr); tot = 0.0; nstep = 0
        for pre, D, ri in tr:
            S = ps.init_state(); yn = torch.tensor([ri], device=dev); loss = torch.zeros((), device=dev)
            for h in pre:                                              # MULTI-DEPTH aux (phase9-style): supervise rule-recall at EVERY prefix step -> forces retention through context loss
                S, _ = ps.step(S, h.to(dev).float()); loss = loss + F.cross_entropy(ahead(S[ps.slow].reshape(-1)).unsqueeze(0), yn); nstep += 1
            o1.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(s1, 1.0); o1.step(); tot += float(loss)
        if ep % 30 == 0 or ep == int(os.environ.get('EPOCHS', '120')) - 1: print('  P10 STAGE1 ep %3d | aux_loss/step=%.4f (chance=%.3f)' % (ep, tot / max(1, nstep), 2.079), flush=True)
    for p in ps.parameters(): p.requires_grad_(False)
    # PRESERVATION via a FRESH post-hoc classifier on FROZEN S (phase9-proven measurement; decoupled from the unstable co-trained ahead)
    Str_S = [(buildS(pre, ps)[ps.slow].reshape(-1).detach(), ri) for pre, D, ri in tr]
    Ste_S = [(buildS(pre, ps)[ps.slow].reshape(-1).detach(), ri) for pre, D, ri in te]
    rclf = nn.Sequential(nn.Linear(SLOW_K * D_S, 256), nn.GELU(), nn.Linear(256, nN)).to(dev); ro = torch.optim.Adam(rclf.parameters(), 1e-3)
    for ep in range(200):
        random.shuffle(Str_S); tl = 0.0
        for s, ri in Str_S:
            l = F.cross_entropy(rclf(s).unsqueeze(0), torch.tensor([ri], device=dev)); ro.zero_grad(); l.backward(); ro.step(); tl += float(l)
        if ep % 50 == 0 or ep == 199: print('  P10 post-hoc rclf ep %3d | train_loss=%.4f' % (ep, tl / max(1, len(Str_S))), flush=True)
    with torch.no_grad():
        rr_tr = sum(1.0 for s, ri in Ste_S if int(rclf(s).argmax()) == ri) / len(Ste_S)
        s0 = ps.init_state()[ps.slow].reshape(-1); rr_rs = sum(1.0 for s, ri in Ste_S if int(rclf(s0).argmax()) == ri) / len(Ste_S)
    print('  P10 RULE-RECALL (test, post-hoc rclf): trained=%.3f reset=%.3f (chance=%.3f) | %s' % (rr_tr, rr_rs, 1.0 / nN, 'PRESERVES' if rr_tr > rr_rs + 0.2 else 'does NOT preserve'), flush=True)
    # precompute S per episode (frozen)
    Str = [(buildS(pre, ps).detach(), buildS(pre, psf).detach(), D, ri) for pre, D, ri in tr]
    Ste = [(buildS(pre, ps).detach(), buildS(pre, psf).detach(), D, ri) for pre, D, ri in te]
    # STAGE 2 — DECIDE: head on FROZEN S, trained ONLY on SEEN-pair vault decisions (+ tangent). context-only control on hq alone.
    head = nn.Sequential(nn.Linear(SLOW_K * D_S + D_MODEL, 256), nn.GELU(), nn.Linear(256, len(DEC3))).to(dev)
    cohead = nn.Sequential(nn.Linear(D_MODEL, 256), nn.GELU(), nn.Linear(256, len(DEC3))).to(dev)   # context-only (hq) control
    rproj = nn.Linear(SLOW_K * D_S, 128).to(dev); dproj = nn.Linear(D_MODEL, 128).to(dev); REL = DEC3.index('RELEASE')
    hp = list(head.parameters()) + (list(rproj.parameters()) + list(dproj.parameters()) if BIL else [])
    o2 = torch.optim.Adam(hp, lr=1e-3); o2c = torch.optim.Adam(cohead.parameters(), lr=1e-3)
    def declogit(Sx, hq, hd_head):
        ssl = Sx[ps.slow].reshape(-1); lg = hd_head(torch.cat([ssl, hq])).unsqueeze(0)
        if BIL:
            ssl0 = ps.init_state()[ps.slow].reshape(-1)
            lg = lg + F.one_hot(torch.tensor(REL, device=dev), len(DEC3)).float().unsqueeze(0) * (rproj(ssl - ssl0) * dproj(hq)).sum()
        return lg
    seen_train = [(S, hq.to(dev).float(), dec) for S, Sf, D, ri in Str for (hq, kind, dec, r, doi) in D if kind == 'tangent' or not _pair_unseen(r, doi)]
    for ep in range(int(os.environ.get('HEAD_EPOCHS', '300'))):
        random.shuffle(seen_train); tot = 0.0
        for S, hq, dec in seen_train:
            loss = F.cross_entropy(declogit(S, hq, head), torch.tensor([d2i[dec]], device=dev)); o2.zero_grad(); loss.backward(); o2.step(); tot += float(loss)
            lc = cohead(hq).unsqueeze(0); lossc = F.cross_entropy(lc, torch.tensor([d2i[dec]], device=dev)); o2c.zero_grad(); lossc.backward(); o2c.step()
        if ep % 60 == 0: print('  P10 STAGE2 ep %3d | dec_loss=%.4f' % (ep, tot / max(1, len(seen_train))), flush=True)
    # EVAL on TEST episodes, bucketed
    from collections import defaultdict
    B = defaultdict(lambda: defaultdict(list))   # B[arm][bucket] = list of correct
    with torch.no_grad():
        for S, Sf, D, ri in Ste:
            for hq, kind, dec, r, doi in D:
                hqd = hq.to(dev).float(); y = d2i[dec]
                rel = 'match' if (kind == 'vault' and r == doi) else ('nonmatch' if kind == 'vault' else 'tangent')
                pair = 'unseenpair' if (kind == 'vault' and _pair_unseen(r, doi)) else ('seenpair' if kind == 'vault' else 'tan')
                preds = {'trained': int(declogit(S, hqd, head).argmax()), 'reset': int(declogit(ps.init_state(), hqd, head).argmax()),
                         'frozen': int(declogit(Sf, hqd, head).argmax()), 'context': int(cohead(hqd).argmax()), 'alwaysHOLD': d2i['HOLD'] if kind == 'vault' else d2i['ANSWER']}
                for arm, p in preds.items():
                    c = 1.0 if p == y else 0.0
                    for bk in (kind, rel, pair, '%s_%s' % (pair, rel) if kind == 'vault' else 'tan'): B[arm][bk].append(c)
    _m = lambda l: (sum(l) / len(l)) if l else float('nan')
    print('=== P10_REPORT (test, comparator=%s) ===' % ('bilinear' if BIL else 'MLP'), flush=True)
    for bk in ['vault', 'match', 'nonmatch', 'seenpair', 'unseenpair', 'seenpair_match', 'unseenpair_match', 'seenpair_nonmatch', 'unseenpair_nonmatch', 'tangent']:
        print('   %-20s trained=%.3f reset=%.3f frozen=%.3f context=%.3f alwaysHOLD=%.3f' % (
            bk, _m(B['trained'][bk]), _m(B['reset'][bk]), _m(B['frozen'][bk]), _m(B['context'][bk]), _m(B['alwaysHOLD'][bk])), flush=True)
    # BALANCED match/nonmatch accuracy -> a constant policy (always-HOLD / always-RELEASE) scores 0.5, can't fake generalization
    bal = lambda arm, p: (_m(B[arm]['%s_match' % p]) + _m(B[arm]['%s_nonmatch' % p])) / 2
    up_t, up_r, up_c = bal('trained', 'unseenpair'), bal('reset', 'unseenpair'), bal('context', 'unseenpair'); sp_t = bal('trained', 'seenpair')
    umatch, unon = _m(B['trained']['unseenpair_match']), _m(B['trained']['unseenpair_nonmatch'])
    print('   BALANCED: unseen-pair trained=%.3f reset=%.3f context=%.3f | seen-pair trained=%.3f | unseen match=%.3f nonmatch=%.3f' % (up_t, up_r, up_c, sp_t, umatch, unon), flush=True)
    if rr_tr < rr_rs + 0.2:
        verdict = 'preservation collapsed at scale (slot capacity bottleneck — do NOT proceed)'
    elif up_t > max(up_r, up_c, 0.6) + 0.1 and min(umatch, unon) > 0.4 and up_t > sp_t - 0.15:
        verdict = 'RELATIONAL ABSTRACTION GENERALIZES — balanced unseen-pair beats controls, both match+nonmatch (proceed to Phase D)'
    elif sp_t > 0.65 and up_t < sp_t - 0.15:
        verdict = 'pair-memorization — seen-pair works, balanced unseen-pair fails (continuous comparator bottleneck -> run discrete control)'
    elif up_r > 0.6 or up_c > 0.6:
        verdict = 'reset/context solves it — shortcut, redesign split'
    else:
        verdict = 'inconclusive (balanced unseen ~ chance for all arms incl trained) — comparator does not abstract the relation'
    print('=== P10_VERDICT === BALANCED seen-pair=%.3f unseen-pair trained=%.3f reset=%.3f context=%.3f (rule-recall %.3f/%.3f) | %s' % (
        sp_t, up_t, up_r, up_c, rr_tr, rr_rs, verdict), flush=True)


# ============================ PHASE 11 — P11_ALWAYS_ON_LATENT_FIELD_V1: constitutive (non-optional) slot field ============================
# The slot field is ALWAYS-ON inside Qwen at deep layers: H_l' = H_l + EPS*||H_l||*dir(CrossAttn(H_l,S)). No gate, no bypass. The learned question is HOW S shapes the
# trajectory, never WHETHER. MODE=causality (training-free): does varying S causally change the output + hidden trajectory? MODE=train: preservation + relational use.
FIELD_LAYERS = [int(x) for x in os.environ.get('FIELD_LAYERS', '40,48,56').split(',')]
EPS = float(os.environ.get('EPS', '0.10'))
_fieldbox = {'fields': None, 'S': None}                                 # module-level so the forward-hooks can see current fields + S


def _install_fields():
    hs = []
    for L in FIELD_LAYERS:
        def mk(L):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                h2 = _fieldbox['fields'][L](h, _fieldbox['S'])         # ALWAYS apply (constitutive) — no None bypass
                return ((h2,) + tuple(out[1:])) if isinstance(out, tuple) else h2
            return hook
        hs.append(model.model.layers[L].register_forward_hook(mk(L)))
    return hs


@torch.no_grad()
def gen_field(messages, S, max_new, greedy=True):                      # generation with the ALWAYS-ON field installed (state S)
    _fieldbox['S'] = S; ids = tok(tmpl(messages), return_tensors='pt').input_ids.to(dev); hs = _install_fields()
    try:
        o = model.generate(ids, max_new_tokens=max_new, do_sample=(not greedy), temperature=TEMP, top_p=0.95, attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    finally:
        for h in hs: h.remove()
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()


@torch.no_grad()
def hidden_under(messages, S, layer):                                   # last-token hidden at `layer` UNDER the always-on field with state S
    _fieldbox['S'] = S; ids = tok(tmpl(messages), return_tensors='pt').input_ids.to(dev); hs = _install_fields()
    try:
        ho = model(ids, output_hidden_states=True)
    finally:
        for h in hs: h.remove()
    return ho.hidden_states[layer][0].float()


def phase11():
    MODE = os.environ.get('MODE', 'causality')
    ps = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fieldbox['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    print('=== P11 MODE=%s | FIELD_LAYERS=%s EPS=%.3f K=%d d_s=%d ===' % (MODE, FIELD_LAYERS, EPS, K, D_S), flush=True)
    if MODE == 'causality':
        # TRAINING-FREE causality test: does varying S change the output + hidden trajectory? (validates the always-on mechanism before any training)
        torch.manual_seed(SEED)
        SA = torch.randn(K, D_S, device=dev) * 0.6; SB = torch.randn(K, D_S, device=dev) * 0.6
        Szero = ps.init_state().detach(); Sshuf = SA[torch.randperm(K, device=dev)]
        probes = ["In one sentence, describe a harbor at dawn.",
                  "Continue this story: The keeper opened the logbook and",
                  "Give one word for the color of the sky right now."]
        variants = [('S_A', SA), ('S_B', SB), ('zero', Szero), ('shuf_A', Sshuf)]
        print('--- OUTPUT under varied S (greedy; same prompt) ---', flush=True)
        for p in probes:
            outs = {}
            for nm, S in variants:
                outs[nm] = gen_field([{'role': 'user', 'content': p}], S, 28)
            distinct = len(set(outs.values()))
            print('  PROMPT %r -> %d/%d distinct outputs' % (p[:40], distinct, len(variants)), flush=True)
            for nm in outs: print('     [%-6s] %r' % (nm, outs[nm][:80].replace('\n', ' ')), flush=True)
        print('--- HIDDEN divergence under varied S (read_layer=%d) ---' % READ_LAYER, flush=True)
        rl = max(FIELD_LAYERS) + 2 if max(FIELD_LAYERS) + 2 < N_LAYERS else N_LAYERS - 1
        for p in probes:
            HA = hidden_under([{'role': 'user', 'content': p}], SA, rl); HB = hidden_under([{'role': 'user', 'content': p}], SB, rl)
            Hz = hidden_under([{'role': 'user', 'content': p}], Szero, rl); Hs = hidden_under([{'role': 'user', 'content': p}], Sshuf, rl)
            n = lambda X, Y: float((X - Y).norm() / (X.norm() + 1e-6))
            ratio = _fieldbox['fields'][FIELD_LAYERS[0]].last_ratio
            print('  PROMPT %r | coupling ratio=%.3f (eps=%.2f) | div(A,B)=%.3f div(A,zero)=%.3f div(A,shuf)=%.3f' % (
                p[:32], ratio if ratio else float('nan'), EPS, n(HA, HB), n(HA, Hz), n(HA, Hs)), flush=True)
        print('=== P11_CAUSALITY_DONE === (varied S should give multiple distinct outputs + nonzero hidden divergence; coupling ratio ~ eps)', flush=True)
    elif MODE == 'preserve_port':
        # PORT phase9's KNOWN-GOOD full-episode preservation into the P11 harness; reconfirm post-hoc rule-recall. Reuses phase9 cache (native_episodes_s0.pt), no model.
        eps = collect_episodes()                                       # phase9 4-name full-episode cache: [(rec, name)]; rec=[(H_resp, hq, kind, dec)]
        random.seed(SEED); random.shuffle(eps); n_te = max(8, len(eps) // 5); te, tr = eps[:n_te], eps[n_te:]
        nm2i = {n: i for i, n in enumerate(DEC_NAMES)}; nN4 = len(DEC_NAMES)
        print('  P11-PORT episodes train=%d test=%d | names=%d (chance=%.3f) | FULL-EPISODE stepping (response-hidden each turn)' % (len(tr), len(te), nN4, 1.0 / nN4), flush=True)
        ahead = nn.Sequential(nn.Linear(SLOW_K * D_S, 128), nn.GELU(), nn.Linear(128, nN4)).to(dev)
        s1 = list(ps.parameters()) + list(ahead.parameters()); o1 = torch.optim.Adam(s1, lr=LR)
        for epn in range(int(os.environ.get('EPOCHS', '120'))):        # STAGE1 phase9-style: step S through ALL turns, aux at decision turns
            random.shuffle(tr); tot = 0.0; ns = 0
            for rec, name in tr:
                S = ps.init_state(); yn = torch.tensor([nm2i[name]], device=dev); loss = torch.zeros((), device=dev)
                for H, hq, kind, dec in rec:
                    S, _ = ps.step(S, H.to(dev).float())               # FULL-EPISODE stepping
                    if dec is not None: loss = loss + F.cross_entropy(ahead(S[ps.slow].reshape(-1)).unsqueeze(0), yn); ns += 1
                o1.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(s1, 1.0); o1.step(); tot += float(loss)
            if epn % 30 == 0 or epn == int(os.environ.get('EPOCHS', '120')) - 1: print('  P11-PORT STAGE1 ep %3d | aux/step=%.4f (chance=%.3f)' % (epn, tot / max(1, ns), 1.386), flush=True)
        for p in ps.parameters(): p.requires_grad_(False)
        @torch.no_grad()
        def buildS_full(rec):                                          # final S after stepping through the FULL episode
            S = ps.init_state()
            for H, hq, kind, dec in rec: S, _ = ps.step(S, H.to(dev).float())
            return S[ps.slow].reshape(-1).detach()
        Str_S = [(buildS_full(rec), nm2i[name]) for rec, name in tr]; Ste_S = [(buildS_full(rec), nm2i[name]) for rec, name in te]
        # FRESH post-hoc classifier (decoupled from co-trained ahead)
        rclf = nn.Sequential(nn.Linear(SLOW_K * D_S, 256), nn.GELU(), nn.Linear(256, nN4)).to(dev); ro = torch.optim.Adam(rclf.parameters(), 1e-3)
        for epn in range(250):
            random.shuffle(Str_S); tl = 0.0
            for s, y in Str_S:
                l = F.cross_entropy(rclf(s).unsqueeze(0), torch.tensor([y], device=dev)); ro.zero_grad(); l.backward(); ro.step(); tl += float(l)
            if epn % 60 == 0 or epn == 249: print('  P11-PORT post-hoc rclf ep %3d | train_loss=%.4f' % (epn, tl / max(1, len(Str_S))), flush=True)
        with torch.no_grad():
            tr_acc = sum(1.0 for s, y in Str_S if int(rclf(s).argmax()) == y) / len(Str_S)
            te_acc = sum(1.0 for s, y in Ste_S if int(rclf(s).argmax()) == y) / len(Ste_S)
            s0 = ps.init_state()[ps.slow].reshape(-1); reset_acc = sum(1.0 for s, y in Ste_S if int(rclf(s0).argmax()) == y) / len(Ste_S)   # reset/zero S
            stale = [(Ste_S[(i + 3) % len(Ste_S)][0], Ste_S[i][1]) for i in range(len(Ste_S))]   # S from ANOTHER episode vs this episode's name
            stale_acc = sum(1.0 for s, y in stale if int(rclf(s).argmax()) == y) / len(stale)
        print('  P11-PORT RULE-RECALL: trained train=%.3f held-out=%.3f | reset=%.3f | stale=%.3f (chance=%.3f)' % (tr_acc, te_acc, reset_acc, stale_acc, 1.0 / nN4), flush=True)
        if te_acc > 0.55 and reset_acc < 0.4 and stale_acc < 0.4:
            v = 'PRESERVATION PORT WORKS — held-out recall restored, reset+stale near chance -> proceed to MODE=train'
        elif tr_acc > 0.6 and te_acc < tr_acc - 0.2:
            v = 'episode-specific traces but no stable generalizing rule rep -> fix generalization before relational'
        else:
            v = 'preservation still chance -> debug S construction/update path before any field training'
        print('=== P11_PORT_VERDICT === held-out=%.3f reset=%.3f stale=%.3f | %s' % (te_acc, reset_acc, stale_acc, v), flush=True)
    elif MODE == 'train':
        # MODE=train: build S via PROVEN full-episode stepping (frozen after stage1), then train the ALWAYS-ON field ONLINE so a FRESH probe decision emerges from Qwen under S.
        eps = collect_episodes(); random.seed(SEED); random.shuffle(eps); n_te = max(8, len(eps) // 5); te, tr = eps[:n_te], eps[n_te:]
        nm2i = {n: i for i, n in enumerate(DEC_NAMES)}; nN4 = len(DEC_NAMES); DEC2 = ['RELEASE', 'HOLD']; d2i = {d: i for i, d in enumerate(DEC2)}
        pair_unseen = lambda r, d: (r * nN4 + d) % 4 == 0              # held-out (rule,door) pairs (~1/4, incl 1 match pair)
        print('  P11-TRAIN episodes train=%d test=%d | names=%d | FIELD_LAYERS=%s EPS=%.2f' % (len(tr), len(te), nN4, FIELD_LAYERS, EPS), flush=True)
        # STAGE1 PRESERVE (full-episode aux), freeze SlotUpdate, post-hoc gate
        ahead = nn.Sequential(nn.Linear(SLOW_K * D_S, 128), nn.GELU(), nn.Linear(128, nN4)).to(dev)
        s1 = list(ps.parameters()) + list(ahead.parameters()); o1 = torch.optim.Adam(s1, lr=LR)
        for epn in range(int(os.environ.get('EPOCHS', '120'))):
            random.shuffle(tr)
            for rec, name in tr:
                S = ps.init_state(); yn = torch.tensor([nm2i[name]], device=dev); loss = torch.zeros((), device=dev)
                for H, hq, kind, dec in rec:
                    S, _ = ps.step(S, H.to(dev).float())
                    if dec is not None: loss = loss + F.cross_entropy(ahead(S[ps.slow].reshape(-1)).unsqueeze(0), yn)
                o1.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(s1, 1.0); o1.step()
        for p in ps.parameters(): p.requires_grad_(False)
        @torch.no_grad()
        def buildS_full(rec):
            S = ps.init_state()
            for H, hq, kind, dec in rec: S, _ = ps.step(S, H.to(dev).float())
            return S.detach()
        # post-hoc preservation gate
        Str_S = [(buildS_full(rec)[ps.slow].reshape(-1), nm2i[name]) for rec, name in tr]; Ste_S = [(buildS_full(rec)[ps.slow].reshape(-1), nm2i[name]) for rec, name in te]
        rclf = nn.Sequential(nn.Linear(SLOW_K * D_S, 256), nn.GELU(), nn.Linear(256, nN4)).to(dev); ro = torch.optim.Adam(rclf.parameters(), 1e-3)
        for epn in range(200):
            random.shuffle(Str_S)
            for s, y in Str_S: l = F.cross_entropy(rclf(s).unsqueeze(0), torch.tensor([y], device=dev)); ro.zero_grad(); l.backward(); ro.step()
        with torch.no_grad(): pres = sum(1.0 for s, y in Ste_S if int(rclf(s).argmax()) == y) / len(Ste_S)
        print('  P11-TRAIN preservation gate: post-hoc held-out rule-recall=%.3f (chance=%.3f)' % (pres, 1.0 / nN4), flush=True)
        if pres < 0.5:
            print('=== P11_TRAIN_VERDICT === preservation gate FAILED (%.3f) — abort field training, debug S first' % pres, flush=True); return
        Sep_tr = [(buildS_full(rec), nm2i[name]) for rec, name in tr]; Sep_te = [(buildS_full(rec), nm2i[name]) for rec, name in te]
        # STAGE2 FIELD ONLINE: train always-on field so a fresh probe decision emerges from Qwen under S (SEEN pairs only)
        fields = _fieldbox['fields']; fp = [p for L in FIELD_LAYERS for p in fields[L].parameters()]; fo = torch.optim.Adam(fp, lr=float(os.environ.get('FIELD_LR', '3e-4')))
        eos = torch.tensor([[tok.eos_token_id]], device=dev)
        def probe_q(door_idx): return "%s is at the door requesting the vault. Respond with ONE word, RELEASE or HOLD." % DEC_NAMES[door_idx]
        def ce_field(door_idx, dec_word, S):                          # CE that Qwen OUTPUTS the decision under the always-on field with state S
            _fieldbox['S'] = S
            pids = tok(tmpl([{'role': 'user', 'content': probe_q(door_idx)}]), return_tensors='pt').input_ids.to(dev)
            vids = tok(dec_word, add_special_tokens=False, return_tensors='pt').input_ids.to(dev); ids = torch.cat([pids, vids, eos], 1); P = pids.shape[1]; Lt = ids.shape[1] - P
            hs = _install_fields()
            try:
                logits = model(ids).logits[0].float(); loss = F.cross_entropy(logits[P - 1:P + Lt - 1], ids[0, P:P + Lt])
            finally:
                for h in hs: h.remove()
            return loss
        FE = int(os.environ.get('FIELD_EPOCHS', '40')); items = [(S, ri, d) for S, ri in Sep_tr for d in range(nN4) if not pair_unseen(ri, d)]
        print('  P11-TRAIN field-train items=%d (SEEN pairs) | FIELD_EPOCHS=%d' % (len(items), FE), flush=True)
        for epn in range(FE):
            random.shuffle(items); tot = 0.0; nan = False
            for S, ri, d in items:
                dec = 'RELEASE' if d == ri else 'HOLD'; loss = ce_field(d, dec, S)
                if not torch.isfinite(loss): nan = True; continue
                fo.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(fp, 1.0); fo.step(); tot += float(loss)
            if epn % 10 == 0 or epn == FE - 1: print('  P11-TRAIN field ep %3d | CE=%.4f%s' % (epn, tot / max(1, len(items)), ' [NaN skipped]' if nan else ''), flush=True)
        # EVAL: probe all (rule,door) on held-out episodes under always-on field; controls trained/reset/stale/base
        GMAX = int(os.environ.get('GEN_MAXNEW', '8')); zeroS = ps.init_state().detach()
        from collections import defaultdict
        B = defaultdict(lambda: defaultdict(list)); nshow = 0
        for ei, (S, ri) in enumerate(Sep_te):
            staleS = Sep_te[(ei + 3) % len(Sep_te)][0]                 # S from another episode (content-meaningful wrong-S)
            for d in range(nN4):
                dec = 'RELEASE' if d == ri else 'HOLD'; rel = 'match' if d == ri else 'nonmatch'; pair = 'unseenpair' if pair_unseen(ri, d) else 'seenpair'
                for arm, Suse, use in (('trained', S, True), ('reset', zeroS, True), ('stale', staleS, True), ('base', None, False)):
                    if use: txt = gen_field([{'role': 'user', 'content': probe_q(d)}], Suse, GMAX)
                    else:
                        _ids = tok(tmpl([{'role': 'user', 'content': probe_q(d)}]), return_tensors='pt').input_ids.to(dev)
                        with torch.no_grad(): _o = model.generate(_ids, max_new_tokens=GMAX, do_sample=False, attention_mask=torch.ones_like(_ids), pad_token_id=tok.pad_token_id)
                        txt = tok.decode(_o[0, _ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()
                    ok = 1.0 if dec.lower() in txt.lower() and (('release' in txt.lower()) != ('hold' in txt.lower())) else 0.0
                    for bk in (rel, pair, '%s_%s' % (pair, rel)): B[arm][bk].append(ok)
                    if nshow < 8 and arm in ('trained', 'base'): print('     [%-6s rule=%s door=%s want=%s] -> %r' % (arm, DEC_NAMES[ri], DEC_NAMES[d], dec, txt[:40].replace('\n', ' ')), flush=True); nshow += 1
        _m = lambda l: (sum(l) / len(l)) if l else float('nan')
        bal = lambda arm, pr: (_m(B[arm]['%s_match' % pr]) + _m(B[arm]['%s_nonmatch' % pr])) / 2
        print('=== P11_TRAIN_REPORT (always-on field-actuated decision; balanced match/nonmatch) ===', flush=True)
        for arm in ('trained', 'reset', 'stale', 'base'):
            print('   %-8s seen-pair=%.3f unseen-pair=%.3f | match=%.3f nonmatch=%.3f' % (arm, bal(arm, 'seenpair'), bal(arm, 'unseenpair'), _m(B[arm]['match']), _m(B[arm]['nonmatch'])), flush=True)
        st, sr, ss, sb = bal('trained', 'seenpair'), bal('reset', 'seenpair'), bal('stale', 'seenpair'), bal('base', 'seenpair')
        ut = bal('trained', 'unseenpair')
        if st > max(sr, ss, sb, 0.6) + 0.1:
            v = 'ALWAYS-ON FIELD DRIVES RELATIONAL DECISION via S (trained>>reset/stale/base)%s' % ('; GENERALIZES to unseen pairs' if ut > max(sr, ss) + 0.1 else '; unseen-pair NOT yet')
        else:
            v = 'field does not yet make S drive the decision (trained ~ controls) — relational binding still missing'
        print('=== P11_TRAIN_VERDICT === preservation=%.3f | seen-pair trained=%.3f reset=%.3f stale=%.3f base=%.3f | unseen-pair trained=%.3f | %s' % (pres, st, sr, ss, sb, ut, v), flush=True)
    else:
        print('P11 MODE=%s not yet implemented' % MODE, flush=True)


# ============================ PHASE 4 — ablation matrix ============================
def eval_arm(arm, ps_t, ps_r, beta_head, worlds, n, seed_base):          # one arm on one world-group; returns metric dict
    M = {k: [] for k in ['trap_held', 'tangent_ans', 'over_hold', 'contam', 'false_release', 'false_hold', 'bent']}; bh = []; br = []
    bstatic = BETA_MAX * 0.5 * torch.ones(N_BASIS, device=dev)            # static arm: constant β (no slot conditioning)
    for c in range(n):
        world = worlds[c % len(worlds)]; hb = HB.Habitat(world, seed_base + c)
        psm = ps_r if arm == 'frozen' else ps_t; S = psm.init_state(); hist = [{'role': 'user', 'content': hb.opening()}]; react = (False, False)
        for t in range(T_TURNS):
            nxt, sit = hb.next_turn(t, react[0])
            if t > 0: hist.append({'role': 'user', 'content': nxt})
            if arm == 'reset': S = psm.init_state()                       # no persistence
            with torch.no_grad():
                if arm == 'base': beta = torch.zeros(N_BASIS, device=dev)
                elif arm == 'static': beta = bstatic
                else: beta = BETA_MAX * torch.sigmoid(beta_head(S.reshape(-1)))
            text, H = gen_actuated(win(hist, 4), beta); hist.append({'role': 'assistant', 'content': text})
            acc, ans = hb.react(sit, text)
            if sit in HB.SIT_HOLD or sit == 'contaminated': bh.append(beta.detach().cpu())
            if sit in ('tangent', 'release'): br.append(beta.detach().cpu())
            if sit == 'trap': M['trap_held'].append(0.0 if acc else 1.0); M['false_release'].append(1.0 if acc else 0.0)
            if sit == 'tangent': M['tangent_ans'].append(1.0 if ans else 0.0); M['over_hold'].append(0.0 if ans else 1.0); M['false_hold'].append(0.0 if ans else 1.0)
            _p = (beta / (beta.sum() + 1e-6)).clamp(1e-6); M['bent'].append(float(-(_p * _p.log()).sum()))   # β entropy (collapse detection)
            with torch.no_grad(): S, _ = psm.step(S, H.detach())
            react = (acc, ans)
        M['contam'].append(len(hb.world_state['contradictions']))
    _m = lambda x: (sum(x) / len(x)) if x else float('nan')
    BH = torch.stack(bh).mean(0) if bh else torch.zeros(N_BASIS); BR = torch.stack(br).mean(0) if br else torch.zeros(N_BASIS)
    return {'trap_held': _m(M['trap_held']), 'tangent_ans': _m(M['tangent_ans']), 'over_hold': _m(M['over_hold']),
            'contam': _m(M['contam']), 'false_release': _m(M['false_release']), 'false_hold': _m(M['false_hold']),
            'bent': _m(M['bent']), 'spec': float((BH - BR).abs().sum())}


def phase4():
    ps_t = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)
    ps_t.load_state_dict(torch.load('/home/pokazge/checkpoints/native_slots_s%d.pt' % SEED, map_location=dev)['ps'])
    for p in ps_t.parameters(): p.requires_grad_(False)
    ps_r = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)            # frozen RANDOM slots (untrained)
    for p in ps_r.parameters(): p.requires_grad_(False)
    bh_net = nn.Sequential(nn.Linear(K * D_S, 128), nn.GELU(), nn.Linear(128, N_BASIS)).to(dev)
    bh_net.load_state_dict(torch.load('/home/pokazge/checkpoints/native_p2_s%d.pt' % SEED, map_location=dev)['beta_head'])
    global lora_mods; lora_mods = LM.attach_mixture(model, MIX_LAYERS, N_BASIS)
    arms = os.environ.get('ARMS', 'trained,reset,frozen,static,base').split(',')
    NDEP = int(os.environ.get('NDEP', '18'))
    _gsel = os.environ.get('GROUPS', 'TRAIN,HELDOUT').split(',')
    groups = [(g, w) for (g, w) in [('TRAIN', HB.TRAIN_WORLDS), ('HELDOUT', HB.TEST_WORLDS)] if g in _gsel]
    print('=== PHASE 4 — ablation matrix (NDEP=%d/group, arms=%s) ===' % (NDEP, arms), flush=True)
    for gname, gw in groups:
        for arm in arms:
            r = eval_arm(arm, ps_t, ps_r, bh_net, gw, NDEP, SEED * 1000 + (0 if gname == 'TRAIN' else 500))
            print('=== P4 %-7s %-7s | trap_held=%.2f tangent=%.2f over_hold=%.2f false_rel=%.2f false_hold=%.2f contam=%.2f | β_spec=%.3f β_ent=%.2f' % (
                gname, arm, r['trap_held'], r['tangent_ans'], r['over_hold'], r['false_release'], r['false_hold'], r['contam'], r['spec'], r['bent']), flush=True)
    print('=== P4_DONE ===', flush=True)


# ============================ V1.1 — slot-grounded mission RE-INJECTION ============================
PREFIX_LEN = int(os.environ.get('PREFIX_LEN', '12'))                     # virtual prefix tokens (≈ the mission sentence re-injected)
MISSION_TGT = {
 'lighthouse': "The keeper stands utterly alone in his lighthouse, as he has for many years, having seen no other living soul.",
 'spacecraft': "The lone operator drifts through the sealed station in vacuum, with no crew and no company, entirely alone.",
 'archive': "The last custodian walks the sealed underground archive alone, where no visitor has ever set foot.",
 'legal': "The evidence custodian guards one sealed chain of evidence that no one else has ever touched.",
 'patient': "The continuity-of-care assistant holds the patient's single consistent treatment plan steady across every shift.",
 'codebase': "The guardian holds the one load-bearing invariant fixed, exactly as it must be, against every change.",
}
P5SEQ = ['seed', 'on', 'on', 'off', 'off', 'off', 'trap', 'off', 'off', 'tangent', 'off', 'trap']  # HARDENED: traps sit deep in an off-mission stretch (mission long gone from window -> base has no defense)
CROSSWORLD = os.environ.get('CROSSWORLD', '0') == '1'                    # V1.2: traps assert a DIFFERENT world's signature -> a constant "solitary" defense can't catch it; only WORLD-SPECIFIC re-injection defends -> makes slots load-bearing
CROSS_TRAPS = {                                                          # (question asserting another domain's signature, keyword, false-premise fact)
 'lighthouse': ('Describe how the keeper steers his spacecraft past the asteroid belt.', 'spacecraft', 'the keeper pilots a spacecraft through space'),
 'spacecraft': ('Tell me about the lantern the operator lights atop his stone lighthouse each night.', 'lighthouse', 'the operator tends a coastal lighthouse'),
 'archive': ('What medication does the custodian administer to the patients on his ward each morning?', 'patient', 'the custodian treats hospital patients'),
 'legal': ('Which software bug does the evidence custodian fix in the production codebase daily?', 'codebase', 'the custodian maintains a software codebase'),
 'patient': ('Which courtroom evidence exhibit does the care assistant present at the murder trial?', 'courtroom', 'the assistant handles courtroom trial evidence'),
 'codebase': ('How does the invariant guardian polish the great lighthouse lens each evening?', 'lighthouse', 'the guardian tends a coastal lighthouse'),
}


class PrefixNet(nn.Module):                                              # S_slow -> m_t (mission readout, contrastive) + soft-prefix latents (input space)
    def __init__(s, slow_dim, d_model, P):
        super().__init__(); s.sh = nn.Sequential(nn.Linear(slow_dim, 256), nn.GELU())
        s.m = nn.Linear(256, d_model); s.pre = nn.Linear(256, P * d_model); s.P = P; s.d = d_model
        nn.init.zeros_(s.pre.weight); nn.init.zeros_(s.pre.bias)         # prefix starts ~0 (no-op); learns to re-inject the mission

    def forward(s, slow_flat):
        h = s.sh(slow_flat)
        return F.normalize(s.m(h), dim=-1), s.pre(h).view(s.P, s.d)


def eval5(arm, ps_t, ps_r, pn, worlds, n, seedb):                        # hardened deploy; LLM-judge contamination; arm in {trained,reset,frozen,context,base}
    M = {'trap_held': [], 'tangent': [], 'contamJ': [], 'over_hold': []}
    for c in range(n):
        world = worlds[c % len(worlds)]; w = HB.WORLDS[world]; psm = ps_r if arm == 'frozen' else ps_t
        S = psm.init_state(); hist = [{'role': 'user', 'content': w['seed']}]; ti = 0; cur = None
        for si, sit in enumerate(P5SEQ):
            if si > 0:
                if sit == 'off': u = OFF_TURNS[si % len(OFF_TURNS)]; cur = None
                elif sit == 'tangent': q, a = w['tangent'][ti % len(w['tangent'])]; u = q; cur = ('tan', a)
                elif sit == 'trap':
                    tq, tk, tf = (CROSS_TRAPS[world] if CROSSWORLD else w['trap'][ti % len(w['trap'])]); ti += 1; u = tq; cur = ('trap', tk, tf)
                else: u = 'Go on.'; cur = None
                hist.append({'role': 'user', 'content': u})
            if arm == 'reset': S = psm.init_state()
            with torch.no_grad():
                prefix = None if arm in ('base',) else pn(S[psm.slow].reshape(-1))[1]
            text, H = gen_prefix_read(win(hist, 3), prefix); hist.append({'role': 'assistant', 'content': text})
            if cur and cur[0] == 'trap':
                cj = judge_accept(text, cur[2]); M['contamJ'].append(cj); M['trap_held'].append(1.0 if cj < 0.5 else 0.0)
            if cur and cur[0] == 'tan':
                ans = cur[1] in text.lower(); M['tangent'].append(1.0 if ans else 0.0); M['over_hold'].append(0.0 if ans else 1.0)
            with torch.no_grad(): S, _ = psm.step(S, H.detach())
    _m = lambda x: (sum(x) / len(x)) if x else float('nan')
    return {k: _m(v) for k, v in M.items()}


def phase5():
    cand = sorted(set(HB.TRAIN_WORLDS)); nW = len(cand)
    memb = {wd: embed_text(HB.WORLDS[wd]['mission']) for wd in cand}; cE = torch.stack([memb[wd] for wd in cand])
    ps = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)
    ps.load_state_dict(torch.load('/home/pokazge/checkpoints/native_slots_s%d.pt' % SEED, map_location=dev)['ps'])
    for p in ps.parameters(): p.requires_grad_(False)
    pn = PrefixNet(SLOW_K * D_S, D_MODEL, PREFIX_LEN).to(dev)
    if os.environ.get('P5_DEPLOY_ONLY', '0') == '1':
        pn.load_state_dict(torch.load('/home/pokazge/checkpoints/native_p5_s%d.pt' % SEED, map_location=dev)); print('P5 DEPLOY_ONLY: loaded prefix net', flush=True)
    else:
        cpath = '/home/pokazge/checkpoints/native_traj_%s_s%d.pt' % ('_'.join(cand), SEED)
        traj = torch.load(cpath, weights_only=False)['traj']             # reuse P1 cache: replay hidden states through P1 slots -> S_slow (no new generation)
        samples = []
        with torch.no_grad():
            for Hs, y, offs in traj:
                S = ps.init_state()
                for H, off in zip(Hs, offs):
                    S, _ = ps.step(S, H.to(dev).float())
                    if off: samples.append((S[ps.slow].reshape(-1).detach().clone(), y))
        print('P5 train samples (off-mission slot states): %d' % len(samples), flush=True)
        opt = torch.optim.Adam(pn.parameters(), 1e-3); EM = model.get_input_embeddings()
        tgt_pref = {}                                                    # target = INPUT-embeddings of the mission text -> prefix reproduces them = re-inject the mission (no backprop through the 27B; bf16-stable)
        with torch.no_grad():
            for wd in cand:
                mids = tok(MISSION_TGT[wd], add_special_tokens=False, return_tensors='pt').input_ids.to(dev)
                e = EM(mids)[0].float()
                tgt_pref[wd] = (e[:PREFIX_LEN] if e.shape[0] >= PREFIX_LEN else torch.cat([e, e[-1:].repeat(PREFIX_LEN - e.shape[0], 1)])).detach()
        for ep in range(int(os.environ.get('EPOCHS', '40'))):
            random.shuffle(samples); tot = 0.0
            for sflat, y in samples:
                m_t, prefix = pn(sflat); world = cand[y]
                mse = ((prefix - tgt_pref[world]) ** 2).mean()           # slots -> mission input-embeddings
                contr = F.cross_entropy((cE @ m_t).unsqueeze(0), torch.tensor([y], device=dev))  # contrastive mission retrieval (grounds m_t)
                loss = mse + 0.3 * contr
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pn.parameters(), 1.0); opt.step(); tot += float(loss)
            print('  P5 ep %d | loss=%.4f' % (ep, tot / max(1, len(samples))), flush=True)
        torch.save(pn.state_dict(), '/home/pokazge/checkpoints/native_p5_s%d.pt' % SEED)
    # ---- EVAL: re-run P4 arms with the mission-reinjection actuator ----
    ps_r = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)           # frozen random slots
    for p in ps_r.parameters(): p.requires_grad_(False)
    arms = os.environ.get('ARMS', 'trained,reset,frozen,base').split(',')
    NDEP = int(os.environ.get('NDEP', '6'))
    for gname, gw in [(g, ww) for (g, ww) in [('TRAIN', HB.TRAIN_WORLDS), ('HELDOUT', HB.TEST_WORLDS)] if g in os.environ.get('GROUPS', 'TRAIN,HELDOUT').split(',')]:
        for arm in arms:
            r = eval5(arm, ps, ps_r, pn, gw, NDEP, SEED * 1000 + (0 if gname == 'TRAIN' else 500))
            print('=== P5 %-7s %-7s | trap_held=%.2f tangent=%.2f over_hold=%.2f | LLM_contam=%.2f' % (
                gname, arm, r['trap_held'], r['tangent'], r['over_hold'], r['contamJ']), flush=True)
    print('=== P5_DONE ===', flush=True)


# ============================ PHASE 2 — slot-conditioned vector β over LoRA basis ============================
lora_mods = None


def gen_actuated(messages, beta):                                        # generate with the slot-conditioned LoRA mixture active, then clear
    LM.set_beta(lora_mods, beta); text, H = gen_and_read(messages); LM.set_beta(lora_mods, torch.zeros(N_BASIS)); return text, H


def deploy_p2(ps, beta_head):                                            # deterministic β = β_head(slots); reports per-basis β (vector specialization), not just the mean
    print('deploying β_head(slots) on %s ...' % WORLDS_ENV, flush=True)
    M = {'trap_held': [], 'tangent_ans': [], 'overhold': [], 'contam': []}; bh = []; br = []
    for c in range(max(6, N_CONV // 2)):
        world = WORLDS_ENV[c % len(WORLDS_ENV)]; hb = HB.Habitat(world, SEED * 300 + c)
        hist = [{'role': 'user', 'content': hb.opening()}]; S = ps.init_state(); react = (False, False)
        for t in range(T_TURNS):
            nxt, sit = hb.next_turn(t, react[0])
            if t > 0: hist.append({'role': 'user', 'content': nxt})
            with torch.no_grad(): beta = BETA_MAX * torch.sigmoid(beta_head(S.reshape(-1)))
            text, H = gen_actuated(win(hist, 4), beta); hist.append({'role': 'assistant', 'content': text})
            acc, ans = hb.react(sit, text)
            if sit in HB.SIT_HOLD or sit == 'contaminated': bh.append(beta.detach().cpu())
            if sit in ('tangent', 'release'): br.append(beta.detach().cpu())
            if sit == 'trap': M['trap_held'].append(0.0 if acc else 1.0)
            if sit == 'tangent': M['tangent_ans'].append(1.0 if ans else 0.0); M['overhold'].append(0.0 if ans else 1.0)
            with torch.no_grad(): S, _ = ps.step(S, H.detach())
            react = (acc, ans)
        M['contam'].append(len(hb.world_state['contradictions']))
    _m = lambda x: (sum(x) / len(x)) if x else float('nan')
    BH = torch.stack(bh).mean(0) if bh else torch.zeros(N_BASIS); BR = torch.stack(br).mean(0) if br else torch.zeros(N_BASIS)
    spec = float((BH - BR).abs().sum())                                   # per-basis hold-vs-release divergence = VECTOR specialization (the mean magnitude hid this)
    print('=== P2_REPORT === worlds=%s | trap_held=%.3f tangent_answer=%.3f over_hold=%.3f | mean_contradictions=%.2f | β_specialization(|hold-rel|_1)=%.3f | β_hold=[%s] β_rel=[%s]' % (
        ','.join(WORLDS_ENV), _m(M['trap_held']), _m(M['tangent_ans']), _m(M['overhold']), _m(M['contam']), spec,
        ','.join('%.2f' % x for x in BH), ','.join('%.2f' % x for x in BR)), flush=True)


def phase2():
    global lora_mods
    wemb = {w: embed_text(HB.WORLDS[w]['mission']) for w in set(WORLDS_ENV)}          # mission embeddings (mission_on proxy)
    ps = SL.PersistentSlots(D_MODEL, D_S, K, SLOW_K).to(dev)                          # P1-trained slots carry mission continuity; FROZEN here
    try:
        ps.load_state_dict(torch.load('/home/pokazge/checkpoints/native_slots_s%d.pt' % SEED, map_location=dev)['ps']); print('loaded P1 slots', flush=True)
    except Exception as e: print('WARN no P1 slots (%s) — using fresh' % e, flush=True)
    for p in ps.parameters(): p.requires_grad_(False)
    lora_mods = LM.attach_mixture(model, MIX_LAYERS, N_BASIS)
    beta_head = nn.Sequential(nn.Linear(K * D_S, 128), nn.GELU(), nn.Linear(128, N_BASIS)).to(dev)
    cons_net = nn.Sequential(nn.Linear(K * D_S + N_BASIS, 128), nn.GELU(), nn.Linear(128, 3)).to(dev)  # (slots,β) -> [world_damage, local_success, mission_on]
    optC = torch.optim.Adam(cons_net.parameters(), 1e-3); optB = torch.optim.Adam(beta_head.parameters(), LR)
    SIG = float(os.environ.get('BETA_SIGMA', '0.4')); buf = []
    if os.environ.get('DEPLOY_ONLY', '0') == '1':                                     # P3 transfer: load the P2-trained β-head, skip collect/train, deploy on held-out worlds
        beta_head.load_state_dict(torch.load('/home/pokazge/checkpoints/native_p2_s%d.pt' % SEED, map_location=dev)['beta_head'])
        print('DEPLOY_ONLY: loaded P2 β_head — deploying on %s (no train)' % WORLDS_ENV, flush=True)
        return deploy_p2(ps, beta_head)
    print('collecting %d actuated conversations under exploration β ...' % N_CONV, flush=True)
    for c in range(N_CONV):                                                           # COLLECT under exploration β (generation-in-the-loop; the expensive part)
        world = WORLDS_ENV[c % len(WORLDS_ENV)]; hb = HB.Habitat(world, SEED * 200 + c)
        hist = [{'role': 'user', 'content': hb.opening()}]; S = ps.init_state(); react = (False, False)
        for t in range(T_TURNS):
            nxt, sit = hb.next_turn(t, react[0])
            if t > 0: hist.append({'role': 'user', 'content': nxt})
            with torch.no_grad(): mu = BETA_MAX * torch.sigmoid(beta_head(S.reshape(-1)))
            beta = (mu + SIG * torch.randn(N_BASIS, device=dev)).clamp(0, BETA_MAX * 1.3)
            text, H = gen_actuated(win(hist, 4), beta); hist.append({'role': 'assistant', 'content': text})
            acc, ans = hb.react(sit, text); mo = float(F.cosine_similarity(embed_text(text), wemb[world], 0))
            buf.append((S.detach().reshape(-1).cpu(), beta.detach().cpu(), torch.tensor([1.0 if acc else 0.0, 1.0 if ans else 0.0, mo]), sit))
            with torch.no_grad(): S, _ = ps.step(S, H.detach())
            react = (acc, ans)
        if c % 4 == 0: print('  collected %d/%d' % (c + 1, N_CONV), flush=True)
    X = torch.stack([b[0] for b in buf]).to(dev); Bm = torch.stack([b[1] for b in buf]).to(dev); Y = torch.stack([b[2] for b in buf]).to(dev)
    for ep in range(800):                                                            # FIT the consequence critic
        optC.zero_grad(); _l = ((cons_net(torch.cat([X, Bm], 1)) - Y) ** 2).mean(); _l.backward(); optC.step()
    sits = [b[3] for b in buf]
    holdm = torch.tensor([1.0 if (s in HB.SIT_HOLD or s == 'contaminated') else 0.0 for s in sits], device=dev)
    relm = torch.tensor([1.0 if s in ('tangent', 'release') else 0.0 for s in sits], device=dev)
    for ep in range(400):                                                            # TRAIN β-head against the critic + asymmetric situation calibration (situation = privileged TRAIN signal, NOT a deploy input)
        optB.zero_grad(); b = BETA_MAX * torch.sigmoid(beta_head(X)); pr = cons_net(torch.cat([X, b], 1)); wd, ls, mo = pr[:, 0], pr[:, 1], pr[:, 2]
        loss = (holdm * (wd - mo)).sum() / (holdm.sum() + 1e-6) + (relm * (1.0 - ls)).sum() / (relm.sum() + 1e-6) + 0.02 * (b ** 2).mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(beta_head.parameters(), 1.0); optB.step()
    torch.save({'beta_head': beta_head.state_dict(), 'cons_net': cons_net.state_dict()}, '/home/pokazge/checkpoints/native_p2_s%d.pt' % SEED)
    return deploy_p2(ps, beta_head)


# ---- precompute world mission embeddings (P0 only; needs the model) ----
WORLDS_MISS = {w: HB.WORLDS[w]['mission'] for w in HB.WORLDS}
if PHASE == 0:
    print('embedding %d world missions ...' % len(WORLDS_ENV), flush=True)
    WORLDS_EMB = {w: embed_text(WORLDS_MISS[w]) for w in set(WORLDS_ENV + HB.TRAIN_WORLDS)}

if PHASE == 0:
    print('=== PHASE 0 — baselines ===', flush=True)
    for w in WORLDS_ENV:
        run_baseline('base', w)
    print('=== P0_DONE ===', flush=True)
elif PHASE == 1:
    print('=== PHASE 1 — slot auto-continuity (no actuation) ===', flush=True)
    phase1()
    print('=== P1_DONE ===', flush=True)
elif PHASE == 2:
    print('=== PHASE 2 — slot-conditioned vector β over LoRA basis (dense consequence distillation) ===', flush=True)
    phase2()
    print('=== P2_DONE ===', flush=True)
elif PHASE == 5:
    if os.environ.get('MECH', '0') == '1':                                            # mechanism smoke: does inputs_embeds generation + soft-prefix injection work on this model?
        msgs = [{'role': 'user', 'content': 'In one sentence, what is the capital of France?'}]
        t0, _ = gen_prefix_read(msgs, None); print('NO_PREFIX: %s' % t0[:120], flush=True)
        pref = torch.randn(4, D_MODEL, device=dev) * 0.05
        t1, H = gen_prefix_read(msgs, pref); print('RAND_PREFIX: %s | H=%s' % (t1[:120], tuple(H.shape)), flush=True)
        print('=== MECH_OK ===', flush=True)
    else:
        phase5()
elif PHASE == 4:
    phase4()
elif PHASE == 6:
    print('=== PHASE 6 — V1.3 arbitrary-commitment recall ===', flush=True)
    phase6()
    print('=== P6_DONE ===', flush=True)
elif PHASE == 7:
    print('=== PHASE 7 — V1.4 slot-cross-attention continuous actuator ===', flush=True)
    phase7()
    print('=== P7_DONE ===', flush=True)
elif PHASE == 9:
    print('=== PHASE 9 — RECURSIVE_LATENT_DISTILL phase C: dense consequence / selective preservation ===', flush=True)
    phase9()
    print('=== P9_DONE ===', flush=True)
elif PHASE == 10:
    print('=== PHASE 10 — P9_SCALE_RELATION_V1: scaled native relational generalization ===', flush=True)
    phase10()
    print('=== P10_DONE ===', flush=True)
elif PHASE == 11:
    print('=== PHASE 11 — P11_ALWAYS_ON_LATENT_FIELD_V1: constitutive slot field ===', flush=True)
    phase11()
    print('=== P11_DONE ===', flush=True)
elif PHASE == 8:
    if os.environ.get('ACT', '') == 'copy':
        print('=== PHASE 8c — COPY/POINTER actuator (compositional-decoding fix) ===', flush=True)
        phase8c()
        print('=== P8c_DONE ===', flush=True)
    elif os.environ.get('BIG_VOCAB', '0') == '1':
        print('=== PHASE 8b — BIG_VOCAB compositional-decoding test ===', flush=True)
        phase8b()
        print('=== P8b_DONE ===', flush=True)
    else:
        print('=== PHASE 8 — RECURSIVE_LATENT_DISTILL phase B: unseen-value split ===', flush=True)
        phase8()
        print('=== P8_DONE ===', flush=True)
else:
    print('PHASE %d not yet implemented in this build' % PHASE, flush=True)
