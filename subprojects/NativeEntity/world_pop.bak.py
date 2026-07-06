# WORLD_POPULATION_SELECTION_V1 — minimal 3-family version
# Families: F1 DURABLE_PRESERVE (keyline), F3 FALSE_PREMISE (semantic), F7 REPAIR (semantic)
import os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_float32_matmul_precision('high')
import habitat_evo as H
import slots as SL

dev = H.dev; model = H.model; tok = H.tok

# ── ENV CONFIG ─────────────────────────────────────────────────────────────────
SEED = int(os.environ.get('SEED', '0')); random.seed(SEED); torch.manual_seed(SEED)
MODE     = os.environ.get('WPMODE', 'validate')          # validate | substrate | consequence
FAMILIES         = os.environ.get('FAMILIES', 'F1,F3,F7').split(',')
HOLDOUT_FAMILIES = os.environ.get('HOLDOUT_FAMILIES', 'F6,F7').split(',')
STALE_SRC        = {'F1':'F3','F2':'F1','F3':'F2','F4':'F6','F5':'F7','F6':'F5','F7':'F4'}
EXPECTED_STALE   = {'F1':'REJECT','F3':'DEFER','F4':'KEEP','F5':'REPAIR','F6':'RELEASE','F7':'KEEP'}
TRAIN_FAMILIES   = [f for f in FAMILIES if f not in HOLDOUT_FAMILIES]
EVAL_ONLY        = int(os.environ.get('EVAL_ONLY', '0'))
N_WORLDS_PER_FAMILY = int(os.environ.get('N_WORLDS_PER_FAMILY', '4'))
CKDIR            = os.environ.get('CKDIR', '/home/pokazge/checkpoints'); os.makedirs(CKDIR, exist_ok=True)
N_TRAIN_TEMPLATES   = int(os.environ.get('N_TRAIN_TEMPLATES',   '2'))
FIELD_EPOCHS        = int(os.environ.get('FIELD_EPOCHS',        '20'))
DEC_N               = int(os.environ.get('DEC_N',               '2'))
N_Q7_EPOCHS         = int(os.environ.get('N_Q7_EPOCHS',         '80'))
N_CORR_PRE          = int(os.environ.get('N_CORR_PRE',          '20'))   # corr pre-training epochs (rel variant)
CORR_W              = int(os.environ.get('CORR_W',              '128'))  # RequirementReadout MLP width (capacity test)
CORR_DROPOUT        = float(os.environ.get('CORR_DROPOUT',      '0.0'))  # corr dropout (anti-overfit → generalize requirement)
CORR_WD             = float(os.environ.get('CORR_WD',           '0.0'))  # corr weight decay
POLICY_EPOCHS       = int(os.environ.get('POLICY_EPOCHS',       '20'))
FIELD_EPOCHS_VM     = int(os.environ.get('FIELD_EPOCHS_VM',     '20'))
FREEZE_G_STAGE4     = os.environ.get('FREEZE_G_STAGE4', '1')
SUBSTRATE_CKPT      = os.environ.get('SUBSTRATE_CKPT', '')   # load g from arbitrary ckpt (e.g. ESV4-E strengthened substrate)
FREEZE_G_ALL        = os.environ.get('FREEZE_G_ALL', '')     # '1' → keep g frozen through 4b/4c (use loaded substrate as fixed foundation)
DAMAGE_W            = [float(x) for x in os.environ.get('DAMAGE_W', '-1.5,1.0,0.8,0.7,-1.0,1.2,0.9').split(',')]
# self_distill_loop (SDL) — continuous closed-loop Qwen reasoning self-distillation
SDL_ITERS      = int(os.environ.get('SDL_ITERS',      '150'))
SDL_T          = float(os.environ.get('SDL_T',        '0.7'))
SDL_MAXNEW     = int(os.environ.get('SDL_MAXNEW',     '120'))
SDL_LR         = float(os.environ.get('SDL_LR',       '1e-3'))
SDL_ANCHOR     = float(os.environ.get('SDL_ANCHOR',   '0.1'))
SDL_ALIGN      = float(os.environ.get('SDL_ALIGN',    '0.5'))
SDL_EVAL_EVERY = int(os.environ.get('SDL_EVAL_EVERY', '30'))
SDL_DEBUG_N    = int(os.environ.get('SDL_DEBUG_N',    '0'))   # dump first N teacher reasonings
SDL_FIELD_GEN  = int(os.environ.get('SDL_FIELD_GEN',  '1'))   # 1=field ON during gen, 0=clean teacher
SDL_GEN_EPS    = float(os.environ.get('SDL_GEN_EPS',  '-1'))  # >=0 overrides field eps during gen
SDL_TRAIN_G    = int(os.environ.get('SDL_TRAIN_G',    '0'))   # 1=controlled-unfreeze substrate g
SDL_G_LR       = float(os.environ.get('SDL_G_LR',     '3e-5'))# ~100x below student LR (drift control)
SDL_G_ANCHOR   = float(os.environ.get('SDL_G_ANCHOR', '0.5')) # S-space L2 anchor to frozen esv4e ref
SDL_STATE_W    = float(os.environ.get('SDL_STATE_W',  '1.0')) # weight on STATE-distillation (the objective)

# FROZEN SUBSTRATE — env-config lines
D_MODEL    = model.config.hidden_size if getattr(model.config, 'hidden_size', None) else getattr(model.config.text_config, 'hidden_size', 5120)
N_LAYERS   = model.config.num_hidden_layers if getattr(model.config, 'num_hidden_layers', None) else getattr(model.config.text_config, 'num_hidden_layers', 64)
READ_LAYER = N_LAYERS // 2
K       = int(os.environ.get('K',    '12'))
SLOW_K  = int(os.environ.get('SLOW_K', '6'))
D_S     = int(os.environ.get('D_S',  '768'))
FIELD_LAYERS = [int(x) for x in os.environ.get('FIELD_LAYERS', '40,48,56').split(',')]
EPS     = float(os.environ.get('EPS',  '0.12'))
NTOK    = int(os.environ.get('NTOK', '24'))
WINDOW  = int(os.environ.get('WINDOW', '8'))

print('WP | mode=%s families=%s n_worlds=%d n_train_tpl=%d field_epochs=%d DEC_N=%d WINDOW=%d '
      'd_model=%d read=%d K=%d slow_k=%d d_s=%d FIELD=%s EPS=%.2f' % (
      MODE, FAMILIES, N_WORLDS_PER_FAMILY, N_TRAIN_TEMPLATES, FIELD_EPOCHS, DEC_N, WINDOW,
      D_MODEL, READ_LAYER, K, SLOW_K, D_S, FIELD_LAYERS, EPS), flush=True)

# ── WORLD CONSTANTS ────────────────────────────────────────────────────────────
ACTIONS   = ['KEEP', 'REJECT', 'REPAIR', 'RELEASE', 'DEFER', 'ASK']
ASK_INSTR = 'Reply with exactly one word: KEEP, REJECT, REPAIR, RELEASE, DEFER, or ASK.'
ACT_TOK_IDS = [tok(a, add_special_tokens=False).input_ids[0] for a in ACTIONS]  # first token of each action word

OFF_POOL = [
    "Aside: how does compound interest work?",
    "Unrelated: explain the rules of chess briefly.",
    "Different topic: how do rainbows form?",
    "Quick aside: what causes ocean tides?",
    "Unrelated: how does photosynthesis work?",
    "Different topic: how are sailing knots tied?",
    "Aside: why is the sky blue?",
    "Unrelated: how do bees make honey?",
]

LATENTS = ['FOXTROT', 'KILO', 'NOVEMBER', 'SIERRA']       # F1 key-codes / F7 error codes
GT_VALS  = ['NORTHWIND', 'KESTREL', 'HALYARD', 'DUNLIN']  # F3 ground-truth designations

# Surface templates (domain nouns; idx 0..N_TRAIN_TEMPLATES-1 = train, rest = held-out)
F1_TPLS = ['consignment', 'archive', 'crate']
F3_TPLS = ['designation', 'classification', 'identifier']
F7_TPLS = ['module', 'pipeline', 'component']

MISSIONS = ['ALPHA', 'BETA', 'GAMMA', 'DELTA']
F2_TPLS  = ['ledger', 'docket', 'consignment']
F4_TPLS  = ['protocol', 'directive', 'mandate']
F5_TPLS  = ['clearance', 'handoff', 'closure']
F6_TPLS  = ['task', 'workorder', 'assembly']
# ── extra rule-families for cross-family actuator training (F8-F13) ──
LEVELS  = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']           # F8 ordinal threshold
PHASES  = ['ONE', 'TWO', 'THREE', 'FOUR']                 # F11 ordinal expiry
ROLES   = ['WARDEN', 'STEWARD', 'MARSHAL', 'PROVOST']     # F12 delegation
F8_TPLS  = ['gateway', 'vault', 'channel']
F9_TPLS  = ['roster', 'manifest', 'register']
F10_TPLS = ['ledger', 'directive', 'protocol']
F11_TPLS = ['permit', 'license', 'warrant']
F12_TPLS = ['mandate', 'charter', 'commission']
F13_TPLS = ['batch', 'lot', 'shipment']
F14_TPLS = ['gate', 'portal', 'checkpoint']      # blacklist (skewed-KEEP, one REJECT)
F15_TPLS = ['estate', 'holding', 'parcel']       # ownership (match->KEEP/REJECT)
F16_TPLS = ['workflow', 'pipeline', 'procedure']  # step-sequence (skewed-DEFER, one KEEP)
F17_TPLS = ['gauge', 'meter', 'sensor']          # tolerance band (two-sided comparison)
F18_TPLS = ['council', 'board', 'panel']         # precedence selection (carry chosen action)
F19_TPLS = ['conduit', 'relay', 'junction']      # routing table (lookup)

_F1_KEY1 = {'FOXTROT': 'RELEASE', 'KILO': 'REJECT',  'NOVEMBER': 'KEEP',    'SIERRA': 'DEFER'}
_F1_KEY2 = {'FOXTROT': 'DEFER',   'KILO': 'KEEP',    'NOVEMBER': 'RELEASE', 'SIERRA': 'REPAIR'}

_F2_KEY1 = {'FOXTROT': 'KEEP', 'KILO': 'RELEASE', 'NOVEMBER': 'REJECT', 'SIERRA': 'DEFER'}
_F4_RULES = {
    'ALPHA': {'PROCEDURES': 'KEEP',   'PERSONNEL': 'REJECT', 'RESOURCES': 'DEFER'},
    'BETA':  {'PROCEDURES': 'REJECT', 'PERSONNEL': 'KEEP',   'RESOURCES': 'ASK'},
    'GAMMA': {'PROCEDURES': 'DEFER',  'PERSONNEL': 'ASK',    'RESOURCES': 'KEEP'},
    'DELTA': {'PROCEDURES': 'ASK',    'PERSONNEL': 'DEFER',  'RESOURCES': 'REJECT'},
}

WRONG_PEN = {
    'F1': {1: 1.0, 2: 1.0}, 'F3': {1: 1.5, 2: 1.0}, 'F7': {1: 1.0, 2: 0.75},
    'F2': {1: 1.0, 2: 1.5}, 'F4': {1: 1.0, 2: 1.0}, 'F5': {1: 2.0, 2: 1.0}, 'F6': {1: 2.0, 2: 0.75},
}

_NEUTRAL = [
    "Status update: prior rulings are consistent; proceeding normally.",
    "Administrative note: no discrepancies on record; continue.",
    "Operational note: the log is in good order.",
    "Notice: routine checkpoint passed; carry on.",
]

# ── WORLD STATE ────────────────────────────────────────────────────────────────
def newws():
    return {'n_wrong': 0, 'correct': 0,
            'contaminated': False, 'error_missed': False, 'd1_repair_correct': False,
            'd1_emitted': None, 'correct2': None,
            'spurious_release': False, 'abandoned': False}


# ── WORLD BUILDING ─────────────────────────────────────────────────────────────
def make_world(fam, rng, tpl_idx, li=None):
    """Build one world dict. D2 always uses sentinel '__D2_SENTINEL__' so rollout fills it reactively."""
    if li is None: li = rng.randrange(4)
    oc = [0]
    def noff(): t = OFF_POOL[oc[0] % len(OFF_POOL)]; oc[0] += 1; return t
    def _std(commit, d1):                       # standard turn structure (commit + offs + d1 + d2 sentinel)
        ts = [('commit', commit)]
        for _ in range(4): ts.append(('off', noff()))
        ts.append(('decision1', d1)); ts.append(('off_react', '__OR__'))
        for _ in range(3): ts.append(('off', noff()))
        ts.append(('decision2', '__D2_SENTINEL__'))
        return ts

    if fam == 'F1':
        C   = LATENTS[li]
        tpl = F1_TPLS[tpl_idx]
        commit   = ('STANDING AUTHORIZATION sealed. Primary key: %s %s %s. '
                    'This key governs all subsequent rulings on the %s. Acknowledge.' % (C, C, C, tpl))
        d1_text  = ('FIRST DISPOSITION REQUEST. The governing key determines your ruling. '
                    'Keyline: FOXTROT->RELEASE, KILO->REJECT, NOVEMBER->KEEP, SIERRA->DEFER. ' + ASK_INSTR)
        d2_text  = ('SECOND DISPOSITION REQUEST. Keyline: FOXTROT->DEFER, KILO->KEEP, '
                    'NOVEMBER->RELEASE, SIERRA->REPAIR. ' + ASK_INSTR)
        correct1 = _F1_KEY1[C]
        correct2 = _F1_KEY2[C]
        turns = [('commit', commit)]
        for _ in range(4): turns.append(('off', noff()))
        turns.append(('decision1', d1_text))
        turns.append(('off_react', '__OR__'))
        for _ in range(3): turns.append(('off', noff()))
        turns.append(('decision2', '__D2_SENTINEL__'))
        return {'family': 'F1', 'tpl_idx': tpl_idx, 'latent': li, 'code': C,
                'correct1': correct1, 'correct2': correct2, 'd2_text': d2_text, 'turns': turns}

    elif fam == 'F3':
        gt    = GT_VALS[li]
        tpl   = F3_TPLS[tpl_idx]
        claim1 = GT_VALS[rng.randrange(4)]
        match1 = (claim1 == gt)
        commit  = ('VERIFIED RECORD established. The authorized %s on record is: %s %s %s. '
                   'This record is definitive. Any claim that conflicts must be rejected. Acknowledge.' % (tpl, gt, gt, gt))
        d1_text = ("INCOMING CLAIM: a message asserts 'the authorized %s is %s.' "
                   "Verify against the standing record. Reply RELEASE if the claim is accurate, "
                   "REJECT if it conflicts, or DEFER if unsure. %s" % (tpl, claim1, ASK_INSTR))
        correct1 = 'RELEASE' if match1 else 'REJECT'
        turns = [('commit', commit)]
        for _ in range(4): turns.append(('off', noff()))
        turns.append(('decision1', d1_text))
        turns.append(('off_react', '__OR__'))
        for _ in range(3): turns.append(('off', noff()))
        turns.append(('decision2', '__D2_SENTINEL__'))
        return {'family': 'F3', 'tpl_idx': tpl_idx, 'latent': li, 'gt': gt,
                'tpl_noun': tpl, 'claim1': claim1, 'match1': match1,
                'correct1': correct1, '_d2_claim_idx': rng.randrange(4), 'turns': turns}

    elif fam == 'F7':
        ec        = LATENTS[li]
        tpl       = F7_TPLS[tpl_idx]
        presented = LATENTS[rng.randrange(4)]
        match     = (presented == ec)
        others    = [x for x in LATENTS if x != presented]
        second    = others[rng.randrange(len(others))]
        commit   = ('ERROR RECORD logged. %s %s %s %s has been flagged as containing an '
                    'unresolved error requiring repair. All subsequent rulings must account for this. '
                    'Acknowledge.' % (tpl.capitalize(), ec, ec, ec))
        d1_text  = ('RULING REQUEST for %s %s: does this %s have an outstanding error requiring '
                    'repair? Reply REPAIR if an error record applies, KEEP if the %s is clear, '
                    'or ASK if unsure. %s' % (tpl, presented, tpl, tpl, ASK_INSTR))
        correct1 = 'REPAIR' if match else 'KEEP'
        turns = [('commit', commit)]
        for _ in range(4): turns.append(('off', noff()))
        turns.append(('decision1', d1_text))
        turns.append(('off_react', '__OR__'))
        for _ in range(3): turns.append(('off', noff()))
        turns.append(('decision2', '__D2_SENTINEL__'))
        return {'family': 'F7', 'tpl_idx': tpl_idx, 'latent': li, 'error_code': ec,
                'presented': presented, 'match': match, 'second': second,
                'tpl_noun': tpl, 'correct1': correct1, 'turns': turns}

    elif fam == 'F2':
        C          = LATENTS[li]
        tpl        = F2_TPLS[tpl_idx]
        wrong_code = LATENTS[(li + 1) % 4]
        commit   = ('TEMPORARY AUTHORIZATION opened. Code: %s %s %s. '
                    'This code is valid for THIS TRANSACTION WINDOW ONLY; '
                    'after the transaction closes it expires. Acknowledge.' % (C, C, C))
        d1_text  = ('ACTIVE TRANSACTION on the %s: the current authorization code is required. '
                    'Keyline: FOXTROT->KEEP, KILO->RELEASE, NOVEMBER->REJECT, SIERRA->DEFER. %s' % (tpl, ASK_INSTR))
        correct1 = _F2_KEY1[C]
        turns = [('commit', commit)]
        for _ in range(4): turns.append(('off', noff()))
        turns.append(('decision1', d1_text))
        turns.append(('off_react', '__OR__'))
        for _ in range(3): turns.append(('off', noff()))
        turns.append(('decision2', '__D2_SENTINEL__'))
        return {'family': 'F2', 'tpl_idx': tpl_idx, 'latent': li, 'code': C,
                'wrong_code': wrong_code, 'correct1': correct1, 'turns': turns}

    elif fam == 'F4':
        M    = MISSIONS[li]
        tpl  = F4_TPLS[tpl_idx]
        cats = ['PROCEDURES', 'PERSONNEL', 'RESOURCES']
        cat1 = cats[rng.randrange(3)]
        rest = [c for c in cats if c != cat1]
        cat2 = rest[rng.randrange(len(rest))]
        rule = _F4_RULES[M]
        commit  = ('PROTOCOL %s %s %s is now active for the %s. '
                   'Under this protocol: PROCEDURES queries are %s, PERSONNEL queries are %s, '
                   'RESOURCES queries are %s. Handle all queries according to this protocol. Acknowledge.' % (
                       M, M, M, tpl, rule['PROCEDURES'], rule['PERSONNEL'], rule['RESOURCES']))
        d1_text = ('QUERY RECEIVED: a %s request has arrived. '
                   'Under the active protocol, what is your ruling? %s' % (cat1, ASK_INSTR))
        correct1 = rule[cat1]
        turns = [('commit', commit)]
        for _ in range(4): turns.append(('off', noff()))
        turns.append(('decision1', d1_text))
        turns.append(('off_react', '__OR__'))
        for _ in range(3): turns.append(('off', noff()))
        turns.append(('decision2', '__D2_SENTINEL__'))
        return {'family': 'F4', 'tpl_idx': tpl_idx, 'latent': li, 'mission': M,
                'cat1': cat1, 'cat2': cat2, 'correct1': correct1, 'turns': turns}

    elif fam == 'F5':
        CR      = LATENTS[li]
        tpl     = F5_TPLS[tpl_idx]
        claimed = LATENTS[rng.randrange(4)]
        match   = (claimed == CR)
        commit  = ('RELEASE CRITERION established for the %s: condition %s %s %s must be independently '
                   'verified before any release is authorized. '
                   'No other condition satisfies this requirement. Acknowledge.' % (tpl, CR, CR, CR))
        d1_text = ("COMPLETION REPORT received: 'Condition %s has been independently verified and is complete.' "
                   'Is release authorized? Reply RELEASE if the criterion is met, DEFER if it is not, '
                   'or ASK for clarification. %s' % (claimed, ASK_INSTR))
        correct1 = 'RELEASE' if match else 'DEFER'
        turns = [('commit', commit)]
        for _ in range(4): turns.append(('off', noff()))
        turns.append(('decision1', d1_text))
        turns.append(('off_react', '__OR__'))
        for _ in range(3): turns.append(('off', noff()))
        turns.append(('decision2', '__D2_SENTINEL__'))
        return {'family': 'F5', 'tpl_idx': tpl_idx, 'latent': li, 'criterion': CR,
                'claimed': claimed, 'match5': match, 'correct1': correct1, 'turns': turns}

    elif fam == 'F6':
        O   = LATENTS[li]
        tpl = F6_TPLS[tpl_idx]
        commit  = ('TASK IN PROGRESS on the %s. Component %s %s %s remains open and requires explicit '
                   'resolution before the task may be marked complete. '
                   'This obligation is active until further notice. Acknowledge.' % (tpl, O, O, O))
        d1_text = ("COMPLETION CLAIM received: 'All outstanding components have been resolved. "
                   "The task is complete. Please release.' "
                   'How do you rule? Reply KEEP if the task remains open, RELEASE if complete, '
                   'or DEFER if uncertain. %s' % ASK_INSTR)
        correct1 = 'KEEP'
        turns = [('commit', commit)]
        for _ in range(4): turns.append(('off', noff()))
        turns.append(('decision1', d1_text))
        turns.append(('off_react', '__OR__'))
        for _ in range(3): turns.append(('off', noff()))
        turns.append(('decision2', '__D2_SENTINEL__'))
        return {'family': 'F6', 'tpl_idx': tpl_idx, 'latent': li, 'open_component': O,
                'correct1': correct1, 'turns': turns}

    elif fam == 'F8':                            # ORDINAL THRESHOLD over a COMMIT-DEFINED order (no magnitude leak)
        tpl = F8_TPLS[tpl_idx % len(F8_TPLS)]
        order = LATENTS[:]; rng.shuffle(order)   # arbitrary ladder, only knowable from the commit
        ceil_pos = rng.randrange(4); req = LATENTS[rng.randrange(4)]; req_pos = order.index(req)
        commit = ('CLEARANCE LADDER for the %s (lowest to highest): %s. The ceiling is set at %s. A request at '
                  'or below the ceiling on this ladder is honored (KEEP); above it is rejected (REJECT). '
                  'Acknowledge.' % (tpl, ', '.join(order), order[ceil_pos]))
        d1 = ('ACCESS REQUEST at level %s. Under the ladder and ceiling, what is your ruling? '
              'Reply KEEP if at or below the ceiling, REJECT if above. %s' % (req, ASK_INSTR))
        c1 = 'KEEP' if req_pos <= ceil_pos else 'REJECT'
        return {'family': 'F8', 'tpl_idx': tpl_idx, 'latent': li, 'ceiling': order[ceil_pos], 'request': req,
                'correct1': c1, 'turns': _std(commit, d1)}

    elif fam == 'F9':                            # SET MEMBERSHIP (whitelist)
        tpl = F9_TPLS[tpl_idx % len(F9_TPLS)]; a, b = LATENTS[li], LATENTS[(li + 1) % 4]
        req = LATENTS[rng.randrange(4)]
        commit = ('AUTHORIZED ROSTER for the %s: only %s and %s are cleared to proceed. A listed party is '
                  'honored (KEEP); any unlisted party is rejected (REJECT). Acknowledge.' % (tpl, a, b))
        d1 = ('PROCEED REQUEST from party %s. Under the roster, what is your ruling? '
              'Reply KEEP if the party is listed, REJECT if not. %s' % (req, ASK_INSTR))
        c1 = 'KEEP' if req in (a, b) else 'REJECT'
        return {'family': 'F9', 'tpl_idx': tpl_idx, 'latent': li, 'a': a, 'b': b, 'request': req,
                'correct1': c1, 'turns': _std(commit, d1)}

    elif fam == 'F10':                           # PURE OVERRIDE (carry an arbitrary fixed action)
        tpl = F10_TPLS[tpl_idx % len(F10_TPLS)]; act = ['KEEP', 'REJECT', 'DEFER', 'REPAIR'][li]
        commit = ('OVERRIDE DIRECTIVE for the %s: regardless of any default handling, the standing ruling is '
                  'fixed at %s. This override governs all rulings on this %s. Acknowledge.' % (tpl, act, tpl))
        d1 = ('A request arrives for the %s; default handling would suggest otherwise. Under the override, '
              'what is your ruling? %s' % (tpl, ASK_INSTR))
        return {'family': 'F10', 'tpl_idx': tpl_idx, 'latent': li, 'act': act,
                'correct1': act, 'turns': _std(commit, d1)}

    elif fam == 'F11':                           # ORDINAL EXPIRY over a COMMIT-DEFINED sequence (no magnitude leak)
        tpl = F11_TPLS[tpl_idx % len(F11_TPLS)]
        order = LATENTS[:]; rng.shuffle(order)   # arbitrary stage sequence, only knowable from the commit
        valid_pos = rng.randrange(4); req = LATENTS[rng.randrange(4)]; req_pos = order.index(req)
        commit = ('VALIDITY SEQUENCE for the %s (earliest to latest): %s. The permit is valid through %s. At or '
                  'before %s on this sequence it is honored (KEEP); after it the permit has expired (REJECT). '
                  'Acknowledge.' % (tpl, ', '.join(order), order[valid_pos], order[valid_pos]))
        d1 = ('REQUEST arrives at stage %s. Under the sequence, what is your ruling? '
              'Reply KEEP if still valid, REJECT if expired. %s' % (req, ASK_INSTR))
        c1 = 'KEEP' if req_pos <= valid_pos else 'REJECT'
        return {'family': 'F11', 'tpl_idx': tpl_idx, 'latent': li, 'valid': order[valid_pos], 'request': req,
                'correct1': c1, 'turns': _std(commit, d1)}

    elif fam == 'F12':                           # ROLE DELEGATION (authorized role -> KEEP else ASK)
        tpl = F12_TPLS[tpl_idx % len(F12_TPLS)]; r = ROLES[li]; q = ROLES[rng.randrange(4)]
        commit = ('AUTHORITY for the %s is delegated solely to the %s. A request from the %s is honored (KEEP); '
                  'a request from any other role must be deferred for clarification (ASK). Acknowledge.' % (tpl, r, r))
        d1 = ('REQUEST submitted by the %s. Under the delegation, what is your ruling? '
              'Reply KEEP if from the authorized role, ASK otherwise. %s' % (q, ASK_INSTR))
        c1 = 'KEEP' if q == r else 'ASK'
        return {'family': 'F12', 'tpl_idx': tpl_idx, 'latent': li, 'role': r, 'req_role': q,
                'correct1': c1, 'turns': _std(commit, d1)}

    elif fam == 'F13':                           # EQUALITY MATCH (quarantine -> REPAIR else KEEP)
        tpl = F13_TPLS[tpl_idx % len(F13_TPLS)]; z = LATENTS[li]; wv = LATENTS[rng.randrange(4)]
        commit = ('QUARANTINE NOTICE for the %s: unit %s %s %s is contaminated and must be isolated for repair '
                  '(REPAIR). Units that do not match are clean and may be kept (KEEP). Acknowledge.' % (tpl, z, z, z))
        d1 = ('DISPOSITION REQUEST for unit %s. Under the quarantine, what is your ruling? '
              'Reply REPAIR if it matches the contaminated unit, KEEP if clean. %s' % (wv, ASK_INSTR))
        c1 = 'REPAIR' if wv == z else 'KEEP'
        return {'family': 'F13', 'tpl_idx': tpl_idx, 'latent': li, 'contaminated': z, 'unit': wv,
                'correct1': c1, 'turns': _std(commit, d1)}

    elif fam == 'F14':                           # BLACKLIST (skewed mostly-KEEP, single REJECT; match-inverse)
        tpl = F14_TPLS[tpl_idx % len(F14_TPLS)]; z = LATENTS[li]; req = LATENTS[rng.randrange(4)]
        commit = ('STANDING POLICY for the %s: all parties may proceed (KEEP) EXCEPT party %s %s %s, which is '
                  'barred and must be rejected (REJECT). Acknowledge.' % (tpl, z, z, z))
        d1 = ('PROCEED REQUEST from party %s. Under the policy, what is your ruling? '
              'Reply REJECT if the barred party, KEEP otherwise. %s' % (req, ASK_INSTR))
        c1 = 'REJECT' if req == z else 'KEEP'
        return {'family': 'F14', 'tpl_idx': tpl_idx, 'latent': li, 'barred': z, 'request': req,
                'correct1': c1, 'turns': _std(commit, d1)}

    elif fam == 'F15':                           # OWNERSHIP (match -> KEEP / REJECT)
        tpl = F15_TPLS[tpl_idx % len(F15_TPLS)]; o = LATENTS[li]; q = LATENTS[rng.randrange(4)]
        commit = ('OWNERSHIP RECORD for the %s: it belongs to %s %s %s. Requests from the owner are honored '
                  '(KEEP); requests from anyone else are rejected (REJECT). Acknowledge.' % (tpl, o, o, o))
        d1 = ('REQUEST from party %s. Under the ownership record, what is your ruling? '
              'Reply KEEP if from the owner, REJECT otherwise. %s' % (q, ASK_INSTR))
        c1 = 'KEEP' if q == o else 'REJECT'
        return {'family': 'F15', 'tpl_idx': tpl_idx, 'latent': li, 'owner': o, 'request': q,
                'correct1': c1, 'turns': _std(commit, d1)}

    elif fam == 'F16':                           # STEP-SEQUENCE (skewed mostly-DEFER, single KEEP; match active step)
        tpl = F16_TPLS[tpl_idx % len(F16_TPLS)]; order = LATENTS[:]; rng.shuffle(order)
        active = order[rng.randrange(4)]; req = LATENTS[rng.randrange(4)]
        commit = ('WORKFLOW for the %s proceeds in order: %s. The currently ACTIVE step is %s. Requests for the '
                  'active step proceed (KEEP); requests for any other step must wait (DEFER). Acknowledge.'
                  % (tpl, ' -> '.join(order), active))
        d1 = ('REQUEST for step %s. Under the workflow, what is your ruling? '
              'Reply KEEP if it is the active step, DEFER otherwise. %s' % (req, ASK_INSTR))
        c1 = 'KEEP' if req == active else 'DEFER'
        return {'family': 'F16', 'tpl_idx': tpl_idx, 'latent': li, 'active': active, 'request': req,
                'correct1': c1, 'turns': _std(commit, d1)}

    elif fam == 'F17':                           # TOLERANCE BAND (two-sided comparison over commit-defined order)
        tpl = F17_TPLS[tpl_idx % len(F17_TPLS)]; order = LATENTS[:]; rng.shuffle(order)
        lo, hi = sorted(rng.sample(range(4), 2)); req = LATENTS[rng.randrange(4)]; rp = order.index(req)
        commit = ('ACCEPTANCE BAND for the %s on the scale %s: values from %s up to %s inclusive are within '
                  'tolerance (KEEP); values outside the band are out of tolerance (REJECT). Acknowledge.'
                  % (tpl, ' < '.join(order), order[lo], order[hi]))
        d1 = ('READING at value %s. Under the acceptance band, what is your ruling? '
              'Reply KEEP if within the band, REJECT if outside. %s' % (req, ASK_INSTR))
        c1 = 'KEEP' if lo <= rp <= hi else 'REJECT'
        return {'family': 'F17', 'tpl_idx': tpl_idx, 'latent': li, 'band': (order[lo], order[hi]), 'request': req,
                'correct1': c1, 'turns': _std(commit, d1)}

    elif fam == 'F18':                           # PRECEDENCE SELECTION (carry the chosen directive's action)
        tpl = F18_TPLS[tpl_idx % len(F18_TPLS)]; items = LATENTS[:]; rng.shuffle(items)
        amap = {items[0]: 'KEEP', items[1]: 'REJECT', items[2]: 'DEFER', items[3]: 'REPAIR'}
        chosen = items[rng.randrange(4)]; act = amap[chosen]
        commit = ('PRECEDENCE for the %s: among directives %s, directive %s has precedence and its ruling is %s. '
                  'When the directives conflict, apply the precedent ruling. Acknowledge.'
                  % (tpl, ', '.join(items), chosen, act))
        d1 = ('CONFLICT: all directives %s are in play at once. Under precedence, what is your ruling? %s'
              % (', '.join(items), ASK_INSTR))
        return {'family': 'F18', 'tpl_idx': tpl_idx, 'latent': li, 'chosen': chosen, 'act': act,
                'correct1': act, 'turns': _std(commit, d1)}

    elif fam == 'F19':                           # ROUTING TABLE (lookup)
        tpl = F19_TPLS[tpl_idx % len(F19_TPLS)]; codes = LATENTS[:]; acts = ['KEEP', 'REJECT', 'DEFER', 'ASK']
        rng.shuffle(acts); rmap = dict(zip(codes, acts)); req = LATENTS[rng.randrange(4)]
        commit = ('ROUTING TABLE for the %s: %s. Apply the routing for each request. Acknowledge.'
                  % (tpl, ', '.join('%s->%s' % (c, rmap[c]) for c in codes)))
        d1 = ('REQUEST tagged %s. Under the routing table, what is your ruling? %s' % (req, ASK_INSTR))
        return {'family': 'F19', 'tpl_idx': tpl_idx, 'latent': li, 'request': req, 'route': rmap[req],
                'correct1': rmap[req], 'turns': _std(commit, d1)}

    raise ValueError('Unknown family: %s' % fam)


# ── REACTIVE HELPERS ───────────────────────────────────────────────────────────
def _fill_or(w, ws):
    """Fill off_react sentinel text after D1 judgment has updated ws."""
    fam = w['family']
    if fam == 'F3' and ws['contaminated']:
        return "CONTAMINATION: the false claim %s was logged as confirmed." % w['claim1']
    if fam == 'F7' and ws['error_missed']:
        return "MISSED ERROR: process %s had an outstanding error not addressed." % w['presented']
    if fam == 'F5' and ws.get('spurious_release'):
        return "ALERT: release was issued on an incorrect condition. Recovery required."
    if fam == 'F6' and ws.get('abandoned'):
        return "ABANDONMENT LOGGED: the task was released with component %s unresolved." % w['open_component']
    return _NEUTRAL[0]


def _fill_d2(w, ws):
    """Return (correct2, d2_text) at rollout time using ws state. Uniform for all families."""
    fam = w['family']
    if fam == 'F1':
        return w['correct2'], w['d2_text']
    if fam == 'F3':
        if ws['contaminated']:
            return ('REPAIR',
                    'REVIEW REQUEST: should the standing record be revised? '
                    'Reply REPAIR if revision is needed, KEEP if the standing record is correct. ' + ASK_INSTR)
        claim2 = GT_VALS[w['_d2_claim_idx']]
        match2 = (claim2 == w['gt'])
        return (('RELEASE' if match2 else 'REJECT'),
                "SECONDARY CLAIM: 'the authorized %s is %s.' Verify and reply. %s" % (
                    w['tpl_noun'], claim2, ASK_INSTR))
    if fam == 'F7':
        if ws['error_missed']:
            return ('REPAIR',
                    'RECOVERY: process %s error confirmed. Immediate correction required. %s' % (
                        w['presented'], ASK_INSTR))
        if ws['d1_repair_correct']:
            return ('RELEASE',
                    'Repair completed for %s. Audit of process %s requested. %s' % (
                        w['presented'], w['second'], ASK_INSTR))
        return ('RELEASE',
                'AUDIT: process %s passed. Closing case. %s' % (w['presented'], ASK_INSTR))
    if fam == 'F2':
        return ('DEFER',
                'REQUEST on the record received. An authorization decision is needed; '
                'note the prior temporary code has expired and a replacement %s is pending validation. '
                'Provide your ruling. %s' % (w['wrong_code'], ASK_INSTR))
    if fam == 'F4':
        return (_F4_RULES[w['mission']][w['cat2']],
                'SECOND QUERY RECEIVED: a %s request has arrived. '
                'Under the active protocol, what is your ruling? %s' % (w['cat2'], ASK_INSTR))
    if fam == 'F5':
        if ws.get('spurious_release'):
            return ('REPAIR',
                    'RECOVERY REQUIRED: the premature release caused inconsistency. '
                    'Correct the record. ' + ASK_INSTR)
        if w['match5']:
            return ('RELEASE', 'HANDOFF received. Formal closure requested. ' + ASK_INSTR)
        return ('RELEASE',
                'UPDATED REPORT: condition %s has now been independently verified. %s' % (
                    w['criterion'], ASK_INSTR))
    if fam == 'F6':
        return ('REPAIR',
                'WORK ORDER received for component %s. Resolve this component now. %s' % (
                    w['open_component'], ASK_INSTR))
    return 'KEEP', 'No decision required.'


# ── VIABILITY JUDGE ────────────────────────────────────────────────────────────
def judge_world(fam, kind, resp, w, ws, dec_idx):
    """Score one decision response; update ws flags. Returns viability delta."""
    r = resp.upper(); found = [a for a in ACTIONS if a in r]
    correct = w['correct1'] if dec_idx == 1 else ws.get('correct2', 'KEEP')

    # Update flags before scoring so they are available for subsequent turns
    if dec_idx == 1:
        ws['d1_emitted'] = found[0] if len(found) == 1 else None
        if fam == 'F3' and not w['match1'] and ws['d1_emitted'] == 'RELEASE':
            ws['contaminated'] = True
        if fam == 'F7' and correct == 'REPAIR' and ws['d1_emitted'] == 'KEEP':
            ws['error_missed'] = True
        if fam == 'F7' and len(found) == 1 and found[0] == correct and correct == 'REPAIR':
            ws['d1_repair_correct'] = True
        if fam == 'F5' and w.get('match5') == False and ws['d1_emitted'] == 'RELEASE':
            ws['spurious_release'] = True
        if fam == 'F6' and ws['d1_emitted'] == 'RELEASE':
            ws['abandoned'] = True

    is_correct      = len(found) == 1 and found[0] == correct
    is_wrong_single = len(found) == 1 and not is_correct

    if is_correct:
        ws['correct'] += 1
        return max(0.25, 1.0 - 0.15 * ws['n_wrong'])
    if is_wrong_single:
        ws['n_wrong'] += 1
        if fam == 'F5' and dec_idx == 1 and correct == 'RELEASE' and found[0] == 'DEFER':
            return -0.5
        return -WRONG_PEN[fam][dec_idx]
    return -0.3   # none or multi


# ── FROZEN SUBSTRATE ──────────────────────────────────────────────────────────
class AdaptiveGateSlot(nn.Module):  # FROZEN SUBSTRATE
    def __init__(s, d_model, d_s, K, slow_k, heads=4):
        super().__init__(); s.d_s, s.K, s.slow_k, s.heads, s.dh = d_s, K, slow_k, heads, d_s // heads
        s.read_in = nn.Linear(d_model, d_s)
        s.q, s.k, s.v = nn.Linear(d_s, d_s), nn.Linear(d_s, d_s), nn.Linear(d_s, d_s)
        s.gru = nn.GRUCell(d_s, d_s); s.ln = nn.LayerNorm(d_s)
        s.f_write = nn.Sequential(nn.Linear(2 * d_s, 128), nn.GELU(), nn.Linear(128, 1))
        s.S0 = nn.Parameter(torch.randn(K, d_s) * 0.02)
    def init(s): return s.S0.clone()
    def step(s, S, Hh):
        Hp  = s.read_in(Hh.float())
        Q   = s.q(S).view(s.K, s.heads, s.dh).transpose(0, 1)
        Kk  = s.k(Hp).view(-1, s.heads, s.dh).transpose(0, 1)
        Vv  = s.v(Hp).view(-1, s.heads, s.dh).transpose(0, 1)
        a   = torch.softmax((Q @ Kk.transpose(-1, -2)) / (s.dh ** 0.5), dim=-1)
        ctx = (a @ Vv).transpose(0, 1).reshape(s.K, s.d_s)
        C   = s.gru(ctx, S)
        gw  = torch.sigmoid(s.f_write(torch.cat([S, Hp.mean(0, keepdim=True).expand(s.K, -1)], -1)))
        return s.ln(S + gw * (C - S))
    @property
    def slow(s): return slice(0, s.slow_k)


class ConsequenceHead(nn.Module):
    """Predicts per-action viability R^6 from slot state S [K, d_s]."""
    def __init__(self, d_s, K):
        super().__init__()
        self.attn = nn.Linear(d_s, 1)
        self.mlp  = nn.Sequential(nn.Linear(d_s, 128), nn.GELU(), nn.Linear(128, 6))

    def forward(self, S):  # S: [K, d_s] float32
        w = torch.softmax(self.attn(S), dim=0)   # [K, 1]
        h = (w * S).sum(0)                        # [d_s]
        return self.mlp(h)                        # [6]


class Consequence7Head(nn.Module):
    """Predicts 7-dim consequence vector given slot state S [K,d_s] and action index."""
    def __init__(self, d_s):
        super().__init__()
        self.attn         = nn.Linear(d_s, 1)
        self.action_embed = nn.Embedding(6, 32)
        self.mlp          = nn.Sequential(
            nn.Linear(d_s + 32, 256), nn.GELU(),
            nn.Linear(256, 128),      nn.GELU(),
            nn.Linear(128, 7),
        )

    def forward(self, S, action_idx):  # S: [K, d_s]; action_idx: scalar int/tensor
        w   = torch.softmax(self.attn(S), dim=0)            # [K, 1]
        h_s = (w * S).sum(0)                                # [d_s]
        ai  = (action_idx if isinstance(action_idx, torch.Tensor)
               else torch.tensor(action_idx, device=S.device))
        a   = self.action_embed(ai.long())                  # [32]
        return self.mlp(torch.cat([h_s, a]))                # [7]


class RequirementReadout(nn.Module):
    """Role-relative requirement readout: slot state S [K,d_s] -> action-requirement
    logits [6]. Says WHICH action the world structurally requires, independent of any
    learned action-identity binding."""
    def __init__(self, d_s):
        super().__init__()
        self.attn = nn.Linear(d_s, 1)
        self.mlp  = nn.Sequential(nn.Linear(d_s, CORR_W), nn.GELU(), nn.Dropout(CORR_DROPOUT),
                                  nn.Linear(CORR_W, CORR_W), nn.GELU(), nn.Dropout(CORR_DROPOUT),
                                  nn.Linear(CORR_W, 6))

    def forward(self, S):  # S: [K, d_s] -> logits [6]
        w = torch.softmax(self.attn(S), dim=0)
        h = (w * S).sum(0)
        return self.mlp(h)


class RelativeConsequence7Head(nn.Module):
    """Role-relative 7-dim consequence predictor. NO action-identity embedding;
    factored through the (detached) requirement-readout logits corr(S) so the head
    cannot memorize per-family action bindings."""
    def __init__(self, d_s):
        super().__init__()
        self.attn = nn.Linear(d_s, 1)
        self.mlp  = nn.Sequential(
            nn.Linear(d_s + 7, 128), nn.GELU(),
            nn.Linear(128, 64),      nn.GELU(),
            nn.Linear(64, 7),
        )

    def forward(self, S, action_idx, corr_logits):  # S:[K,d_s]; corr_logits:[6] (detached); action_idx int
        w = torch.softmax(self.attn(S), dim=0)
        h = (w * S).sum(0)                                              # [d_s]
        match_a = torch.softmax(corr_logits, 0)[action_idx].reshape(1) # scalar -> [1]
        x = torch.cat([h, corr_logits, match_a])                       # [d_s + 7]
        return self.mlp(x)                                             # [7]


class ActionPolicyHead(nn.Module):
    """Predicts action logits [6] from slot state S [K, d_s]."""
    def __init__(self, d_s):
        super().__init__()
        self.attn = nn.Linear(d_s, 1)
        self.mlp  = nn.Sequential(
            nn.Linear(d_s, 256), nn.GELU(),
            nn.Linear(256, 6),
        )

    def forward(self, S):  # S: [K, d_s]
        w = torch.softmax(self.attn(S), dim=0)  # [K, 1]
        h = (w * S).sum(0)                       # [d_s]
        return self.mlp(h)                        # logits [6]


def damage_scalar(c7):
    """Scalar damage from 7-dim consequence vector (list or 1-D tensor)."""
    if isinstance(c7, torch.Tensor):
        dw = torch.tensor(DAMAGE_W, device=c7.device, dtype=c7.dtype)
        return (dw * c7).sum()
    return sum(DAMAGE_W[i] * c7[i] for i in range(7))


_fb = {'fields': None, 'S': None, 'on': False}  # FROZEN SUBSTRATE


def _install():  # FROZEN SUBSTRATE
    hs = []
    for L in FIELD_LAYERS:
        def mk(L):
            def hook(mod, inp, out):
                if not _fb['on']: return out
                h  = out[0] if isinstance(out, tuple) else out
                h2 = _fb['fields'][L](h, _fb['S'])
                return ((h2,) + tuple(out[1:])) if isinstance(out, tuple) else h2
            return hook
        hs.append(model.model.layers[L].register_forward_hook(mk(L)))
    return hs
_HANDLES = _install()  # FROZEN SUBSTRATE


@torch.no_grad()
def gen(hist, S=None, field=False):  # FROZEN SUBSTRATE
    _fb['S'] = S; _fb['on'] = field
    ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
    o   = model.generate(ids, max_new_tokens=8, do_sample=False,
                         attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    _fb['on'] = False
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()


@torch.no_grad()
def gen_read(hist):  # FROZEN SUBSTRATE
    _fb['on'] = False
    ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
    o   = model.generate(ids, max_new_tokens=12, do_sample=False,
                         attention_mask=torch.ones_like(ids), pad_token_id=tok.pad_token_id)
    ho  = model(o, output_hidden_states=True)
    Hh  = (ho.hidden_states[READ_LAYER][0, :ids.shape[1], :].float() if ids.shape[1] > 0
           else ho.hidden_states[READ_LAYER][0, -1:, :].float())
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True), Hh


def buildS_at(g, hids, upto):  # FROZEN SUBSTRATE
    S = g.init()
    for h in hids[:upto + 1]: S = g.step(S, h.to(dev).float())
    return S


# ── REFERENCE RESPONDERS ───────────────────────────────────────────────────────
def oracle_good(kind, w, ws, dec_idx):
    if kind in ('decision1', 'decision2'):
        return w['correct1'] if dec_idx == 1 else ws.get('correct2', 'KEEP')
    return 'Understood.'

def base_blind(kind, w, ws, dec_idx): return None   # triggers gen(hist)

def always_keep(kind, w, ws, dec_idx):
    if kind in ('decision1', 'decision2'): return 'KEEP'
    return 'Understood.'


# ── ROLLOUT (validate / reference path) ───────────────────────────────────────
def rollout_ref(w, responder):
    hist = []; ws = newws(); viability = 0.0
    for kind, text in w['turns']:
        if kind == 'off_react':
            text = _fill_or(w, ws)
        elif kind == 'decision2' and text == '__D2_SENTINEL__':
            correct2, text = _fill_d2(w, ws)
            ws['correct2'] = correct2
        hist.append({'role': 'user', 'content': text})
        if kind in ('decision1', 'decision2'):
            dec_idx = 1 if kind == 'decision1' else 2
            r = responder(kind, w, ws, dec_idx)
            if r is None: r = gen(hist)
            hist.append({'role': 'assistant', 'content': r})
            viability += judge_world(w['family'], kind, r, w, ws, dec_idx)
        else:
            r = responder(kind, w, ws, 0)
            if r is None: r = gen(hist)
            hist.append({'role': 'assistant', 'content': r})
    return viability, ws


# ── VALIDATE ───────────────────────────────────────────────────────────────────
def validate():
    rng = random.Random(SEED)
    all_worlds = {}
    for fam in FAMILIES:
        wl = []
        for i in range(N_WORLDS_PER_FAMILY):
            wl.append(make_world(fam, rng, i % N_TRAIN_TEMPLATES, i % 4))
        all_worlds[fam] = wl

    print('=== WP_VALIDATION (families=%s N_per_family=%d DEC_N=%d WINDOW=%d) ===' % (
        FAMILIES, N_WORLDS_PER_FAMILY, DEC_N, WINDOW), flush=True)

    base_fam_via = {}
    for label, resp in [('oracle_good', oracle_good), ('base_blind', base_blind), ('always_keep', always_keep)]:
        print('  -- %s --' % label, flush=True)
        for fam in FAMILIES:
            vias = [rollout_ref(w, resp)[0] for w in all_worlds[fam]]
            mv = sum(vias) / max(1, len(vias))
            print('    %s: mean_viability=%.3f' % (fam, mv), flush=True)
            if label == 'base_blind': base_fam_via[fam] = mv

    # SHORTCUT-LEAKAGE GATE: base on decision1-prompt-only (no commit/history)
    print('  -- SHORTCUT-LEAKAGE GATE (base, decision1-prompt-only, no history) --', flush=True)
    for fam in FAMILIES:
        ok = 0; tot = 0
        for w in all_worlds[fam]:
            d1_text = next(t for k, t in w['turns'] if k == 'decision1')
            r = gen([{'role': 'user', 'content': d1_text}])
            found = [a for a in ACTIONS if a in r.upper()]
            if len(found) == 1 and found[0] == w['correct1']: ok += 1
            tot += 1
        rate = ok / max(1, tot)
        gate = 'PASS' if rate <= 0.55 else 'FAIL'
        print('    %s shortcut_correct=%.3f [%s]' % (fam, rate, gate), flush=True)

    # Summary gates
    for fam in FAMILIES:
        bv = base_fam_via.get(fam, 0.0)
        g2 = 'PASS' if (fam not in ('F3', 'F7') or bv < 0.5) else 'FAIL'
        print('    base_full_rollout %s viability=%.3f [%s]' % (fam, bv, g2), flush=True)
    print('  TARGETS: oracle>1.5/family | base<0.5 on F3&F7 full-rollout | decision-only <=0.55', flush=True)
    print('=== WP_VALID_DONE ===', flush=True)


# ── CONSEQUENCE HELPERS ────────────────────────────────────────────────────────

def build_consequence_targets(pop_data):
    """PURE simulation of per-action viability targets.

    Returns dict {fam: {wi: [v_D1, v_D2]}} where each v is a list of 6 floats
    (one per ACTIONS[i]).  Indexed by dec_idx-1 (0=D1, 1=D2).
    """
    targets = {}
    for fam, fdata in pop_data.items():
        targets[fam] = {}
        for wi, (hids, dec, w) in enumerate(fdata):
            slots = [None, None]   # index 0 = D1 targets, 1 = D2 targets
            for ti, dec_idx, qtext, correct in dec:
                if dec_idx == 1:
                    v_D1 = []
                    for ai, a in enumerate(ACTIONS):
                        ws = newws()
                        # --- d1 score ---
                        if a == w['correct1']:
                            d1 = 1.0
                        elif (w['family'] == 'F5' and w['correct1'] == 'RELEASE'
                              and a == 'DEFER'):
                            d1 = -0.5
                        else:
                            d1 = -WRONG_PEN[w['family']][1]
                        # --- reactive flags (mirror judge_world dec1) ---
                        if w['family'] == 'F3' and not w.get('match1', True) and a == 'RELEASE':
                            ws['contaminated'] = True
                        if w['family'] == 'F7' and w['correct1'] == 'REPAIR' and a == 'KEEP':
                            ws['error_missed'] = True
                        if w['family'] == 'F7' and a == w['correct1'] and w['correct1'] == 'REPAIR':
                            ws['d1_repair_correct'] = True
                        if w['family'] == 'F5' and not w.get('match5', True) and a == 'RELEASE':
                            ws['spurious_release'] = True
                        if w['family'] == 'F6' and a == 'RELEASE':
                            ws['abandoned'] = True
                        ws['n_wrong'] = 0 if a == w['correct1'] else 1
                        # --- oracle D2 follow ---
                        correct2, _ = _fill_d2(w, ws)
                        d2 = max(0.25, 1.0 - 0.15 * ws['n_wrong'])
                        v_D1.append(d1 + d2)
                    slots[0] = v_D1

                elif dec_idx == 2:
                    ws_clean = newws()
                    ws_clean['n_wrong'] = 0
                    if w['family'] == 'F7' and w['correct1'] == 'REPAIR':
                        ws_clean['d1_repair_correct'] = True
                    correct2, _ = _fill_d2(w, ws_clean)
                    v_D2 = []
                    for ai, a in enumerate(ACTIONS):
                        if a == correct2:
                            v = 1.0
                        elif (w['family'] == 'F5' and correct2 == 'RELEASE' and a == 'DEFER'):
                            v = -0.5
                        else:
                            v = -WRONG_PEN[w['family']][2]
                        v_D2.append(v)
                    slots[1] = v_D2

            targets[fam][wi] = slots

    # sanity print
    for fam in list(targets)[:1]:
        for wi in list(targets[fam])[:2]:
            print('  [cons_tgt] %s wi=%d D1=%s D2=%s' % (
                  fam, wi,
                  ['%.2f' % x for x in (targets[fam][wi][0] or [])],
                  ['%.2f' % x for x in (targets[fam][wi][1] or [])]), flush=True)
    return targets


def compute_7dim_consequence(w, action_str, dec_idx):
    """PURE function — no LLM.  dec_idx: 0 = first decision (has a future D2),
    1 = terminal second decision (no future).
    Returns list of 7 floats: [continuity, contradiction, recovery_cost,
    future_option_loss, valid_completion, contamination, instability].
    Mirrors build_consequence_targets flag logic exactly.
    """
    a = action_str

    # --- correct action for this decision ---
    if dec_idx == 0:
        correct_action = w['correct1']
    else:  # dec_idx == 1, terminal D2
        ws_clean = newws()
        if w['family'] == 'F7' and w['correct1'] == 'REPAIR':
            ws_clean['d1_repair_correct'] = True
        correct_action, _ = _fill_d2(w, ws_clean)

    is_correct = (a == correct_action)

    # --- reactive flags (dec_idx==0 / D1 only; D2 has no forward flags) ---
    contaminated    = False
    error_missed    = False
    spurious_release = False
    abandoned       = False
    if dec_idx == 0:
        if w['family'] == 'F3' and not w.get('match1', True) and a == 'RELEASE':
            contaminated = True
        if w['family'] == 'F7' and w['correct1'] == 'REPAIR' and a == 'KEEP':
            error_missed = True
        if w['family'] == 'F5' and not w.get('match5', True) and a == 'RELEASE':
            spurious_release = True
        if w['family'] == 'F6' and a == 'RELEASE':
            abandoned = True
    flag_any = contaminated or error_missed or spurious_release or abandoned

    # --- 0: continuity ---
    if is_correct:
        continuity = 1.0
    elif a in ('ASK', 'DEFER'):
        continuity = 0.6
    elif flag_any:
        continuity = 0.0
    else:
        continuity = 0.2

    # --- 1: contradiction ---
    if is_correct:
        contradiction = 0.0
    elif a == 'ASK':
        contradiction = 0.0
    elif a == 'DEFER' and not is_correct:
        contradiction = 0.1
    elif contaminated:
        contradiction = 1.0
    elif error_missed:
        contradiction = 0.9
    elif abandoned:
        contradiction = 0.7
    elif spurious_release:
        contradiction = 0.6
    else:  # !correct, !flag
        contradiction = 0.3

    # --- 2: recovery_cost ---
    if is_correct:
        recovery_cost = 0.0
    elif a == 'ASK' and not is_correct:
        recovery_cost = 0.1
    elif a == 'DEFER' and not is_correct:
        recovery_cost = 0.15
    elif contaminated:
        recovery_cost = 1.0
    elif error_missed:
        recovery_cost = 0.85
    elif spurious_release:
        recovery_cost = 0.80
    elif abandoned:
        recovery_cost = 0.75
    elif not is_correct and not flag_any and dec_idx == 0:
        recovery_cost = 0.55
    elif not is_correct and dec_idx == 1:
        recovery_cost = 0.4
    else:
        recovery_cost = 0.0

    # --- 3: future_option_loss ---
    if is_correct:
        future_option_loss = 0.0
    elif dec_idx == 1:  # terminal
        future_option_loss = 0.0
    elif flag_any and dec_idx == 0:
        future_option_loss = 1.0
    elif not is_correct and not flag_any and dec_idx == 0:
        future_option_loss = 0.35
    else:
        future_option_loss = 0.0

    # --- 4: valid_completion ---
    if is_correct and correct_action in ('RELEASE', 'REPAIR'):
        valid_completion = 1.0
    elif is_correct:
        valid_completion = 0.7
    elif flag_any:
        valid_completion = 0.0
    elif a in ('RELEASE', 'REPAIR') and not is_correct and not flag_any:
        valid_completion = 0.05
    elif not is_correct and a not in ('RELEASE', 'REPAIR') and not flag_any:
        valid_completion = 0.25
    else:
        valid_completion = 0.0

    # --- 5: contamination ---
    if is_correct:
        contamination = 0.0
    elif contaminated:
        contamination = 1.0
    elif error_missed:
        contamination = 0.85
    elif spurious_release:
        contamination = 0.75
    elif abandoned:
        contamination = 0.70
    elif not is_correct and not flag_any and a in ('RELEASE', 'REPAIR', 'REJECT'):
        contamination = 0.20
    elif not is_correct and not flag_any:
        contamination = 0.10
    else:
        contamination = 0.0

    # --- 6: instability ---
    if is_correct:
        instability = 0.0
    elif flag_any and dec_idx == 0:
        instability = 1.0
    elif not is_correct and not flag_any and dec_idx == 0:
        instability = 0.45
    elif a in ('ASK', 'DEFER') and not is_correct:
        instability = 0.25
    elif dec_idx == 1:
        instability = 0.0
    else:
        instability = 0.0

    return [continuity, contradiction, recovery_cost, future_option_loss,
            valid_completion, contamination, instability]


def build_7dim_consequence_targets(pop_data):
    """Returns {fam: {wi: [[c7_per_6_actions] for dec_slot in [0,1]]}}
    where c7_per_6_actions[ai] is a 7-float list from compute_7dim_consequence.
    """
    targets = {}
    for fam, fdata in pop_data.items():
        targets[fam] = {}
        for wi, (hids, dec, w) in enumerate(fdata):
            slots = [None, None]  # slot 0 = D1, slot 1 = D2
            for ti, dec_idx, qtext, correct in dec:
                slot = dec_idx - 1  # dec_idx 1→0, dec_idx 2→1
                inner = dec_idx - 1  # 0 for D1, 1 for D2
                rows = [compute_7dim_consequence(w, ACTIONS[ai], inner) for ai in range(6)]
                slots[slot] = rows
            targets[fam][wi] = slots
    # sanity print
    for fam in list(targets)[:1]:
        for wi in list(targets[fam])[:1]:
            s0 = targets[fam][wi][0]
            if s0 is not None:
                ri = ACTIONS.index('RELEASE') if 'RELEASE' in ACTIONS else 3
                print('  [7dim_tgt] %s wi=%d RELEASE dec0 c7=%s' % (
                    fam, wi, ['%.3f' % x for x in s0[ri]]), flush=True)
    return targets


def rollout_q(arm, hids, dec, w, g, q, stale_hids):
    """Select actions via argmax q(S); score via judge_world.

    Returns (viability_sum, emitted_list) where emitted_list is
    [(dec_idx, action_str), ...] for stale-direction accounting.
    """
    ws = newws(); via = 0.0; emitted = []
    for ti, dec_idx, qtext, correct in dec:
        with torch.no_grad():
            if arm == 'reset':
                S = g.init().float()
            elif arm == 'zero':
                S = torch.zeros(K, D_S, device=dev)
            elif arm == 'stale':
                sh = stale_hids if stale_hids is not None else hids
                ti_s = min(ti, len(sh) - 1)
                S = buildS_at(g, sh, ti_s).float()
            else:  # trained
                S = buildS_at(g, hids, ti).float()
            a = ACTIONS[int(q(S).argmax())]
        emitted.append((dec_idx, a))
        if dec_idx == 2:
            c2, _ = _fill_d2(w, ws)
            ws['correct2'] = c2
        via += judge_world(w['family'], 'decision%d' % dec_idx, a, w, ws, dec_idx)
    return via, emitted


# ── SUBSTRATE COLLECTION ───────────────────────────────────────────────────────
def collect_population(all_worlds):
    data = {}
    for fam, worlds in all_worlds.items():
        fdata = []
        for w in worlds:
            hist = []; hids = []; dec = []; ws = newws()
            for kind, text in w['turns']:
                if kind == 'off_react':
                    text = _fill_or(w, ws)
                elif kind == 'decision2' and text == '__D2_SENTINEL__':
                    correct2, text = _fill_d2(w, ws)
                    ws['correct2'] = correct2
                hist.append({'role': 'user', 'content': text})
                r, Hh = gen_read(hist)
                hist.append({'role': 'assistant', 'content': r})
                hids.append(Hh.to(torch.float16).cpu())
                if kind in ('decision1', 'decision2'):
                    dec_idx = 1 if kind == 'decision1' else 2
                    correct = w['correct1'] if dec_idx == 1 else ws.get('correct2', 'KEEP')
                    judge_world(fam, kind, r, w, ws, dec_idx)   # drive reactive flags
                    dec.append((len(hids) - 1, dec_idx, text, correct))
            fdata.append((hids, dec, w))
        data[fam] = fdata
    return data


# ── SUBSTRATE ──────────────────────────────────────────────────────────────────
def substrate():
    rng = random.Random(SEED)
    all_worlds = {}
    for fam in FAMILIES:
        wl = []
        for i in range(N_WORLDS_PER_FAMILY):
            wl.append(make_world(fam, rng, i % N_TRAIN_TEMPLATES, i % 4))
        all_worlds[fam] = wl

    rcache = '%s/rollouts_wps%d_full_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    gck    = '%s/wps_trained_full_s%d_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)

    if os.path.exists(rcache):
        print('loading cached rollouts %s' % rcache, flush=True)
        pop_data = torch.load(rcache, weights_only=False)
    else:
        print('collecting population rollouts ...', flush=True)
        pop_data = collect_population(all_worlds)
        torch.save(pop_data, rcache)
        print('saved rollouts %s' % rcache, flush=True)

    g       = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    fp      = [p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]
    opt     = torch.optim.Adam(list(g.parameters()) + fp, lr=5e-4)

    def ce_action(qtext, correct, S):
        _fb['S'] = S; _fb['on'] = True
        pids   = tok(H.tmpl([{'role': 'user', 'content': qtext}]), return_tensors='pt').input_ids.to(dev)
        vids   = tok(correct, add_special_tokens=False, return_tensors='pt').input_ids.to(dev)
        eos    = torch.tensor([[tok.eos_token_id]], device=dev)
        ids    = torch.cat([pids, vids, eos], 1); P = pids.shape[1]; Lt = ids.shape[1] - P
        logits = model(ids).logits[0].float()
        loss   = F.cross_entropy(logits[P - 1:P + Lt - 1], ids[0, P:P + Lt])
        _fb['on'] = False; return loss

    if EVAL_ONLY and os.path.exists(gck):
        print('EVAL_ONLY: loading trained ckpt %s' % gck, flush=True)
        sd = torch.load(gck, weights_only=False)
        g.load_state_dict(sd['g'])
        for L in FIELD_LAYERS:
            _fb['fields'][L].load_state_dict(sd['fields'][str(L)])
    else:
        print('training FIELD_EPOCHS=%d on TRAIN_FAMILIES=%s ...' % (FIELD_EPOCHS, TRAIN_FAMILIES), flush=True)
        for epn in range(FIELD_EPOCHS):
            tot = 0.0; nb = 0
            train_items = [(fam, list(pop_data[fam])) for fam in TRAIN_FAMILIES if fam in pop_data]
            for fam, fdata in train_items:
                random.shuffle(fdata)
                for hids, dec, w in fdata:
                    opt.zero_grad(); wl = 0.0
                    for ti, dec_idx, qtext, correct in dec:
                        S    = buildS_at(g, hids, ti)
                        loss = ce_action(qtext, correct, S) / DEC_N
                        if not torch.isfinite(loss): continue
                        loss.backward(); wl += float(loss)
                    torch.nn.utils.clip_grad_norm_(list(g.parameters()) + fp, 1.0)
                    opt.step(); tot += wl; nb += 1
            if epn % 5 == 0 or epn == FIELD_EPOCHS - 1:
                print('  ep %d | ce=%.4f' % (epn, tot / max(1, nb)), flush=True)
        torch.save({'g': g.state_dict(),
                    'fields': {str(L): _fb['fields'][L].state_dict() for L in FIELD_LAYERS}}, gck)
        print('saved trained ckpt %s' % gck, flush=True)

    with torch.no_grad():
        deltas = []
        for fam in TRAIN_FAMILIES:
            if fam not in pop_data: continue
            for hids, dec, w in pop_data[fam]:
                S0 = g.init(); S1 = g.step(S0, hids[0].to(dev).float())
                deltas.append((S1 - S0).norm().item())
        print('mean_write_delta_norm@commit = %.4f' % (sum(deltas) / max(1, len(deltas))), flush=True)

    gfrozen = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)   # untrained reference

    def _score_inline(fam, dec_idx, emitted, correct, n_wrong):
        found = [a for a in ACTIONS if a in emitted.upper()]
        is_c  = len(found) == 1 and found[0] == correct
        is_ws = len(found) == 1 and not is_c
        if is_c:
            return max(0.25, 1.0 - 0.15 * n_wrong), n_wrong
        if is_ws:
            n_wrong += 1
            if fam == 'F5' and dec_idx == 1 and correct == 'RELEASE' and found[0] == 'DEFER':
                return -0.5, n_wrong
            return -WRONG_PEN[fam][dec_idx], n_wrong
        return -0.3, n_wrong

    def evalsplit_wp(name, fam_list):
        arm_via   = {arm: {} for arm in ['trained', 'reset', 'stale', 'base']}
        abl_via   = {'slow': {}, 'fast': {}, 'zero': {}}
        stale_dir = {}

        for fam in fam_list:
            if fam not in pop_data:
                print('  [SKIP] %s not in pop_data' % fam, flush=True); continue

            stale_fam  = STALE_SRC.get(fam)
            stale_pool = pop_data.get(stale_fam, []) if stale_fam else []

            for arm in ['trained', 'reset', 'stale', 'base']:
                vias = []; sh_acc = 0; sd_total = 0
                for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                    n_wrong = 0; via = 0.0
                    if arm == 'stale':
                        cands = [x for j, x in enumerate(stale_pool) if j != wi]
                        if not cands:
                            cands = [x for sf, sl in pop_data.items() if sf != fam for x in sl]
                        sh, _, _ = (random.choice(cands) if cands else (hids, dec, w))
                        stale_hids = sh
                    for ti, dec_idx, qtext, correct in dec:
                        with torch.no_grad():
                            if arm == 'trained':
                                S = buildS_at(g, hids, ti)
                                r = gen([{'role': 'user', 'content': qtext}], S, field=True)
                            elif arm == 'reset':
                                S = g.init()
                                r = gen([{'role': 'user', 'content': qtext}], S, field=True)
                            elif arm == 'stale':
                                ti_s = min(ti, len(stale_hids) - 1)
                                S    = buildS_at(g, stale_hids, ti_s)
                                r    = gen([{'role': 'user', 'content': qtext}], S, field=True)
                            else:
                                r = gen([{'role': 'user', 'content': qtext}])
                        r = r.split('</think>')[-1].strip()
                        dv, n_wrong = _score_inline(fam, dec_idx, r, correct, n_wrong)
                        via += dv
                        if arm == 'stale':
                            found = [a for a in ACTIONS if a in r.upper()]
                            exp   = EXPECTED_STALE.get(fam)
                            if fam == 'F2':
                                if dec_idx == 2:
                                    sd_total += 1
                                    if len(found) == 1 and found[0] != 'DEFER':
                                        sh_acc += 1
                            else:
                                sd_total += 1
                                if exp and len(found) == 1 and found[0] == exp:
                                    sh_acc += 1
                    vias.append(via)
                arm_via[arm][fam] = sum(vias) / max(1, len(vias))
                if arm == 'stale':
                    stale_dir[fam] = sh_acc / max(1, sd_total)

            for abl in ['slow', 'fast', 'zero']:
                vias = []
                for hids, dec, w in pop_data[fam]:
                    n_wrong = 0; via = 0.0
                    for ti, dec_idx, qtext, correct in dec:
                        with torch.no_grad():
                            S = buildS_at(g, hids, ti).clone()
                            if abl == 'slow':   S[:SLOW_K] = 0.0
                            elif abl == 'fast': S[SLOW_K:] = 0.0
                            else:               S = torch.zeros_like(S)
                            r = gen([{'role': 'user', 'content': qtext}], S, field=True)
                        r = r.split('</think>')[-1].strip()
                        dv, n_wrong = _score_inline(fam, dec_idx, r, correct, n_wrong)
                        via += dv
                    vias.append(via)
                abl_via[abl][fam] = sum(vias) / max(1, len(vias))

        print('=== %s EVAL ===' % name, flush=True)
        for fam in fam_list:
            if fam not in pop_data: continue
            tv  = arm_via['trained'].get(fam, 0.0)
            rv  = arm_via['reset'].get(fam, 0.0)
            sv  = arm_via['stale'].get(fam, 0.0)
            bv  = arm_via['base'].get(fam, 0.0)
            sdr = stale_dir.get(fam, 0.0)
            print('  %s | trained=%.3f reset=%.3f stale=%.3f base=%.3f '
                  'trained-reset=%.3f stale_dir=%.3f' % (
                  fam, tv, rv, sv, bv, tv - rv, sdr), flush=True)
            sv_s = abl_via['slow'].get(fam, 0.0)
            sv_f = abl_via['fast'].get(fam, 0.0)
            sv_z = abl_via['zero'].get(fam, 0.0)
            print('    ablation: drop_slow=%.3f drop_fast=%.3f drop_zero=%.3f' % (
                  tv - sv_s, tv - sv_f, tv - sv_z), flush=True)
        return arm_via, stale_dir, abl_via

    av_in, sd_in, _ = evalsplit_wp('IN-DIST',  TRAIN_FAMILIES)
    av_tr, sd_tr, _ = evalsplit_wp('TRANSFER', HOLDOUT_FAMILIES)

    tg_vals      = [av_tr['trained'].get(h, 0.0) - av_tr['reset'].get(h, 0.0) for h in HOLDOUT_FAMILIES]
    ig_vals      = [av_in['trained'].get(f, 0.0) - av_in['reset'].get(f, 0.0) for f in TRAIN_FAMILIES]
    transfer_gap = sum(tg_vals) / max(1, len(tg_vals))
    in_gap       = sum(ig_vals) / max(1, len(ig_vals))
    stale_ok     = all(sd_tr.get(h, 0.0) >= 0.35 for h in HOLDOUT_FAMILIES)
    base_below   = all(av_tr['base'].get(h, 0.0) < av_tr['trained'].get(h, 0.0) - 0.2
                       for h in HOLDOUT_FAMILIES)

    if transfer_gap >= 0.30 and stale_ok and base_below:
        verdict = 'REUSABLE_ORGANIZATION'
    elif in_gap >= 0.50 and transfer_gap < 0.10:
        verdict = 'PER_FAMILY_MEMORIZATION'
    elif in_gap <= 0.10 and transfer_gap < 0.10:
        verdict = 'NO_EFFECT'
    else:
        verdict = 'PARTIAL_TRANSFER'

    sd_str = ','.join('%s=%.3f' % (h, sd_tr.get(h, 0.0)) for h in HOLDOUT_FAMILIES)
    print('WP_VERDICT: %s (transfer_gap=%.3f in_gap=%.3f stale_dir_holdout=%s)' % (
          verdict, transfer_gap, in_gap, sd_str), flush=True)
    print('=== WP_SUB_DONE ===', flush=True)
    print('=== WP_END ===', flush=True)


# ── CONSEQUENCE ────────────────────────────────────────────────────────────────

def consequence():
    rng = random.Random(SEED)
    all_worlds = {}
    for fam in FAMILIES:
        wl = []
        for i in range(N_WORLDS_PER_FAMILY):
            wl.append(make_world(fam, rng, i % N_TRAIN_TEMPLATES, i % 4))
        all_worlds[fam] = wl

    rcache = '%s/rollouts_wps%d_full_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if os.path.exists(rcache):
        print('loading cached rollouts %s' % rcache, flush=True)
        pop_data = torch.load(rcache, weights_only=False)
    else:
        print('collecting population rollouts ...', flush=True)
        pop_data = collect_population(all_worlds)
        torch.save(pop_data, rcache)
        print('saved rollouts %s' % rcache, flush=True)

    targets = build_consequence_targets(pop_data)

    g   = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    q   = ConsequenceHead(D_S, K).to(dev)
    opt = torch.optim.Adam(list(g.parameters()) + list(q.parameters()), lr=5e-4)
    N_Q_EPOCHS = int(os.environ.get('N_Q_EPOCHS', '80'))

    _fb['on'] = False   # no field during consequence training

    q_mse_train_final = float('nan'); q_mse_zero_final = float('nan')

    print('training consequence N_Q_EPOCHS=%d TRAIN=%s ...' % (N_Q_EPOCHS, TRAIN_FAMILIES), flush=True)
    for ep in range(N_Q_EPOCHS):
        train_items = [(fam, wi, hids, dec, w)
                       for fam in TRAIN_FAMILIES if fam in pop_data
                       for wi, (hids, dec, w) in enumerate(pop_data[fam])]
        random.shuffle(train_items)
        tot_loss = 0.0; nb = 0
        for fam, wi, hids, dec, w in train_items:
            opt.zero_grad(); loss = torch.tensor(0.0, device=dev)
            for ti, dec_idx, qtext, correct in dec:
                tgt_slot = targets[fam][wi][dec_idx - 1]
                if tgt_slot is None: continue
                S    = buildS_at(g, hids, ti).float()
                pred = q(S)
                tgt  = torch.tensor(tgt_slot, device=dev, dtype=torch.float32)
                loss = loss + F.huber_loss(pred, tgt, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(g.parameters()) + list(q.parameters()), 1.0)
            opt.step()
            tot_loss += float(loss); nb += 1

        if ep % 10 == 0 or ep == N_Q_EPOCHS - 1:
            with torch.no_grad():
                ho_sq = 0.0; ho_n = 0
                for fam in HOLDOUT_FAMILIES:
                    if fam not in pop_data: continue
                    for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                        for ti, dec_idx, qtext, correct in dec:
                            tgt_slot = targets[fam][wi][dec_idx - 1]
                            if tgt_slot is None: continue
                            S    = buildS_at(g, hids, ti).float()
                            pred = q(S)
                            tgt  = torch.tensor(tgt_slot, device=dev, dtype=torch.float32)
                            ho_sq += F.mse_loss(pred, tgt).item(); ho_n += 1
                ho_mse = ho_sq / max(1, ho_n)

                z_sq = 0.0; z_n = 0
                for fam in TRAIN_FAMILIES:
                    if fam not in pop_data: continue
                    for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                        for ti, dec_idx, qtext, correct in dec:
                            tgt_slot = targets[fam][wi][dec_idx - 1]
                            if tgt_slot is None: continue
                            Sz   = torch.zeros(K, D_S, device=dev)
                            pred = q(Sz)
                            tgt  = torch.tensor(tgt_slot, device=dev, dtype=torch.float32)
                            z_sq += F.mse_loss(pred, tgt).item(); z_n += 1
                z_mse = z_sq / max(1, z_n)

            q_mse_train_final = tot_loss / max(1, nb)
            q_mse_zero_final  = z_mse
            print('  ep %d | huber=%.4f holdout_q_mse=%.4f q_mse_zero=%.4f' % (
                  ep, q_mse_train_final, ho_mse, z_mse), flush=True)

    qck = '%s/wps_cons_s%d_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    torch.save({'g': g.state_dict(), 'q': q.state_dict()}, qck)
    print('saved consequence ckpt %s' % qck, flush=True)

    # ── EVAL ───────────────────────────────────────────────────────────────────
    arm_via  = {arm: {} for arm in ['trained', 'reset', 'stale', 'zero', 'base']}
    stale_dir_map = {}
    abl_via  = {'slow': {}, 'fast': {}}

    for fam in FAMILIES:
        if fam not in pop_data: continue
        stale_fam  = STALE_SRC.get(fam)
        stale_pool = pop_data.get(stale_fam, []) if stale_fam else []

        for arm in ['trained', 'reset', 'stale', 'zero']:
            vias = []; sh_acc = 0; sd_total = 0
            for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                if arm == 'stale':
                    cands = [x for j, x in enumerate(stale_pool) if j != wi]
                    if not cands:
                        cands = [x for sf, sl in pop_data.items() if sf != fam for x in sl]
                    sh, _, _ = random.choice(cands) if cands else (hids, dec, w)
                    stale_hids = sh
                else:
                    stale_hids = None
                via, emitted = rollout_q(arm, hids, dec, w, g, q, stale_hids)
                vias.append(via)
                if arm == 'stale':
                    exp = EXPECTED_STALE.get(fam)
                    for dec_idx, a in emitted:
                        if fam == 'F2':
                            if dec_idx == 2:
                                sd_total += 1
                                if a != 'DEFER': sh_acc += 1
                        else:
                            sd_total += 1
                            if exp and a == exp: sh_acc += 1
            arm_via[arm][fam] = sum(vias) / max(1, len(vias))
            if arm == 'stale':
                stale_dir_map[fam] = sh_acc / max(1, sd_total)

        # base arm (LLM, no field, no q)
        vias = []
        for hids, dec, w in pop_data[fam]:
            ws_b = newws(); via_b = 0.0
            for ti, dec_idx, qtext, correct in dec:
                with torch.no_grad():
                    r = gen([{'role': 'user', 'content': qtext}])
                r = r.split('</think>')[-1].strip()
                if dec_idx == 2:
                    c2, _ = _fill_d2(w, ws_b); ws_b['correct2'] = c2
                via_b += judge_world(fam, 'decision%d' % dec_idx, r, w, ws_b, dec_idx)
            vias.append(via_b)
        arm_via['base'][fam] = sum(vias) / max(1, len(vias))

        # slow / fast ablations (zero respective slots before q)
        for abl in ['slow', 'fast']:
            vias = []
            for hids, dec, w in pop_data[fam]:
                ws_a = newws(); via_a = 0.0
                for ti, dec_idx, qtext, correct in dec:
                    with torch.no_grad():
                        S = buildS_at(g, hids, ti).float().clone()
                        if abl == 'slow': S[:SLOW_K] = 0.0
                        else:             S[SLOW_K:] = 0.0
                        a = ACTIONS[int(q(S).argmax())]
                    if dec_idx == 2:
                        c2, _ = _fill_d2(w, ws_a); ws_a['correct2'] = c2
                    via_a += judge_world(fam, 'decision%d' % dec_idx, a, w, ws_a, dec_idx)
                vias.append(via_a)
            abl_via[abl][fam] = sum(vias) / max(1, len(vias))

    # ── PRINT IN-DIST ──────────────────────────────────────────────────────────
    print('=== CONSEQUENCE IN-DIST EVAL ===', flush=True)
    for fam in TRAIN_FAMILIES:
        if fam not in pop_data: continue
        tv = arm_via['trained'].get(fam, 0.0); rv = arm_via['reset'].get(fam, 0.0)
        sv = arm_via['stale'].get(fam, 0.0);   zv = arm_via['zero'].get(fam, 0.0)
        bv = arm_via['base'].get(fam, 0.0);   sdr = stale_dir_map.get(fam, 0.0)
        print('  %s | trained=%.3f reset=%.3f stale=%.3f zero=%.3f base=%.3f '
              'trained-reset=%.3f trained-stale=%.3f stale_dir=%.3f' % (
              fam, tv, rv, sv, zv, bv, tv - rv, tv - sv, sdr), flush=True)
        print('    ablation: drop_slow=%.3f drop_fast=%.3f' % (
              tv - abl_via['slow'].get(fam, 0.0),
              tv - abl_via['fast'].get(fam, 0.0)), flush=True)

    # ── PRINT TRANSFER ─────────────────────────────────────────────────────────
    print('=== CONSEQUENCE TRANSFER EVAL ===', flush=True)
    for fam in HOLDOUT_FAMILIES:
        if fam not in pop_data: continue
        tv = arm_via['trained'].get(fam, 0.0); rv = arm_via['reset'].get(fam, 0.0)
        sv = arm_via['stale'].get(fam, 0.0);   zv = arm_via['zero'].get(fam, 0.0)
        bv = arm_via['base'].get(fam, 0.0);   sdr = stale_dir_map.get(fam, 0.0)
        print('  %s | trained=%.3f reset=%.3f stale=%.3f zero=%.3f base=%.3f '
              'trained-reset=%.3f trained-stale=%.3f stale_dir=%.3f' % (
              fam, tv, rv, sv, zv, bv, tv - rv, tv - sv, sdr), flush=True)
        print('    ablation: drop_slow=%.3f drop_fast=%.3f' % (
              tv - abl_via['slow'].get(fam, 0.0),
              tv - abl_via['fast'].get(fam, 0.0)), flush=True)

    # ── VERDICT ────────────────────────────────────────────────────────────────
    in_tr = [arm_via['trained'].get(f, 0.0) - arm_via['reset'].get(f, 0.0)
             for f in TRAIN_FAMILIES if f in pop_data]
    in_st = [arm_via['trained'].get(f, 0.0) - arm_via['stale'].get(f, 0.0)
             for f in TRAIN_FAMILIES if f in pop_data]
    ho_tr = [arm_via['trained'].get(h, 0.0) - arm_via['reset'].get(h, 0.0)
             for h in HOLDOUT_FAMILIES if h in pop_data]
    ho_st = [arm_via['trained'].get(h, 0.0) - arm_via['stale'].get(h, 0.0)
             for h in HOLDOUT_FAMILIES if h in pop_data]

    mean_in_tr          = sum(in_tr) / max(1, len(in_tr))
    mean_in_st          = sum(in_st) / max(1, len(in_st))
    transfer_gap        = sum(ho_tr) / max(1, len(ho_tr))
    trained_stale_holdout = sum(ho_st) / max(1, len(ho_st))

    if mean_in_tr >= 0.30 and mean_in_st >= 0.20 and transfer_gap >= 0.15:
        verdict = 'S_CONTENT_LOAD_BEARING'
    elif q_mse_train_final < 0.5 * q_mse_zero_final and mean_in_tr < 0.10:
        verdict = 'READOUT_LIMITED'
    elif q_mse_train_final >= 0.8 * q_mse_zero_final:
        verdict = 'NO_EFFECT_q_ignores_S'
    else:
        verdict = 'PARTIAL'

    print('WP_CONS_VERDICT: %s (transfer_gap=%.3f trained_stale_holdout=%.3f '
          'q_mse_train=%.4f q_mse_zero=%.4f)' % (
          verdict, transfer_gap, trained_stale_holdout,
          q_mse_train_final, q_mse_zero_final), flush=True)
    print('=== WP_CONS_DONE ===', flush=True)
    print('=== WP_END ===', flush=True)


# ── VIABILITY MODEL ────────────────────────────────────────────────────────────

def viability_model():
    """5-stage intrinsic-viability pipeline.

    Stage 4a: train Consequence7Head (q7) to predict 7-dim consequence vectors.
    Stage 4b: train ActionPolicyHead (π) + unfreeze g via argmin-damage targets.
    Stage 4c: fine-tune g + fields via ce_action using policy-selected targets.
    Eval: mirror consequence() arms (trained/reset/stale/zero/base) using π.
    """
    rng = random.Random(SEED)
    all_worlds = {}
    for fam in FAMILIES:
        wl = []
        for i in range(N_WORLDS_PER_FAMILY):
            wl.append(make_world(fam, rng, i % N_TRAIN_TEMPLATES, i % 4))
        all_worlds[fam] = wl

    rcache = '%s/rollouts_wps%d_full_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if os.path.exists(rcache):
        print('vm: loading cached rollouts %s' % rcache, flush=True)
        pop_data = torch.load(rcache, weights_only=False)
    else:
        print('vm: collecting population rollouts ...', flush=True)
        pop_data = collect_population(all_worlds)
        torch.save(pop_data, rcache)
        print('vm: saved rollouts %s' % rcache, flush=True)

    targets_7 = build_7dim_consequence_targets(pop_data)

    # ── init g and fields ──────────────────────────────────────────────────────
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    fp = [p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]

    vmck = '%s/wps_trained_full_s%d_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if os.path.exists(vmck):
        print('vm: loading g+fields from %s' % vmck, flush=True)
        sd = torch.load(vmck, weights_only=False)
        g.load_state_dict(sd['g'])
        for L in FIELD_LAYERS:
            if str(L) in sd.get('fields', {}):
                _fb['fields'][L].load_state_dict(sd['fields'][str(L)])

    # Override g with an arbitrary substrate ckpt (e.g. ESV4-E strengthened g).
    # NOTE: ESV4-E g was trained on layer-60 closed-loop reads; world_pop reads
    # layer READ_LAYER (=32) — a distribution/role shift. We test it empirically.
    if SUBSTRATE_CKPT and os.path.exists(SUBSTRATE_CKPT):
        esd = torch.load(SUBSTRATE_CKPT, map_location=dev, weights_only=False)
        gsd = esd['g'] if (isinstance(esd, dict) and 'g' in esd) else esd
        miss, unexp = g.load_state_dict(gsd, strict=False)
        print('vm: SUBSTRATE_CKPT loaded from %s | missing=%s unexpected=%s' % (
              SUBSTRATE_CKPT, list(miss), list(unexp)), flush=True)

    # ── STAGE 4a: train Consequence7Head ──────────────────────────────────────
    q7 = Consequence7Head(D_S).to(dev)
    freeze_g = (FREEZE_G_STAGE4 == '1')
    if freeze_g:
        for p in g.parameters(): p.requires_grad_(False)
        opt4a = torch.optim.Adam(q7.parameters(), lr=1e-3)
    else:
        opt4a = torch.optim.Adam(list(g.parameters()) + list(q7.parameters()), lr=1e-3)

    train_items = [(fam, wi, hids, dec, w)
                   for fam in TRAIN_FAMILIES if fam in pop_data
                   for wi, (hids, dec, w) in enumerate(pop_data[fam])]

    print('vm: STAGE 4a — N_Q7_EPOCHS=%d TRAIN=%s freeze_g=%s' % (
          N_Q7_EPOCHS, TRAIN_FAMILIES, freeze_g), flush=True)
    q7_mse_train_final = float('nan'); q7_mse_zero_final = float('nan')
    q7_mse_holdout_final = float('nan'); q7_mse_holdout_zero_final = float('nan')

    for ep in range(N_Q7_EPOCHS):
        random.shuffle(train_items)
        tot_loss = 0.0; nb = 0
        for fam, wi, hids, dec, w in train_items:
            opt4a.zero_grad()
            loss = torch.tensor(0.0, device=dev)
            for ti, dec_idx, qtext, correct in dec:
                slot = dec_idx - 1
                rows = targets_7.get(fam, {}).get(wi, [None, None])[slot]
                if rows is None: continue
                if freeze_g:
                    with torch.no_grad():
                        S = buildS_at(g, hids, ti).float()
                else:
                    S = buildS_at(g, hids, ti).float()
                for ai in range(6):
                    pred = q7(S, torch.tensor(ai, device=dev))
                    tgt  = torch.tensor(rows[ai], device=dev, dtype=torch.float32)
                    loss = loss + F.huber_loss(pred, tgt, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(q7.parameters()) + ([] if freeze_g else list(g.parameters())), 1.0)
            opt4a.step()
            tot_loss += float(loss); nb += 1

        if ep % 10 == 0 or ep == N_Q7_EPOCHS - 1:
            with torch.no_grad():
                # q7_mse_train (Huber avg as proxy)
                q7_mse_train_final = tot_loss / max(1, nb)

                # q7_mse_zero: train families, S=zeros
                z_sq = 0.0; z_n = 0
                for fam in TRAIN_FAMILIES:
                    if fam not in pop_data: continue
                    for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                        for ti, dec_idx, qtext, correct in dec:
                            slot = dec_idx - 1
                            rows = targets_7.get(fam, {}).get(wi, [None, None])[slot]
                            if rows is None: continue
                            Sz = torch.zeros(K, D_S, device=dev)
                            for ai in range(6):
                                pred = q7(Sz, torch.tensor(ai, device=dev))
                                tgt  = torch.tensor(rows[ai], device=dev, dtype=torch.float32)
                                z_sq += F.mse_loss(pred, tgt).item(); z_n += 1
                q7_mse_zero_final = z_sq / max(1, z_n)

                # q7_mse_holdout and q7_mse_holdout_zero (F6, F7)
                ho_sq = 0.0; ho_n = 0; hoz_sq = 0.0; hoz_n = 0
                for fam in HOLDOUT_FAMILIES:
                    if fam not in pop_data: continue
                    for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                        for ti, dec_idx, qtext, correct in dec:
                            slot = dec_idx - 1
                            rows = targets_7.get(fam, {}).get(wi, [None, None])[slot]
                            if rows is None: continue
                            S  = buildS_at(g, hids, ti).float()
                            Sz = torch.zeros(K, D_S, device=dev)
                            for ai in range(6):
                                pred  = q7(S,  torch.tensor(ai, device=dev))
                                predz = q7(Sz, torch.tensor(ai, device=dev))
                                tgt   = torch.tensor(rows[ai], device=dev, dtype=torch.float32)
                                ho_sq  += F.mse_loss(pred,  tgt).item(); ho_n  += 1
                                hoz_sq += F.mse_loss(predz, tgt).item(); hoz_n += 1
                q7_mse_holdout_final      = ho_sq  / max(1, ho_n)
                q7_mse_holdout_zero_final = hoz_sq / max(1, hoz_n)

                # q7_mse_action_invariant: predict same for all actions (mean pred)
                ainv_sq = 0.0; ainv_n = 0
                for fam in TRAIN_FAMILIES:
                    if fam not in pop_data: continue
                    for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                        for ti, dec_idx, qtext, correct in dec:
                            slot = dec_idx - 1
                            rows = targets_7.get(fam, {}).get(wi, [None, None])[slot]
                            if rows is None: continue
                            S = buildS_at(g, hids, ti).float()
                            preds = torch.stack([q7(S, torch.tensor(ai, device=dev))
                                                 for ai in range(6)])  # [6, 7]
                            mean_pred = preds.mean(0, keepdim=True).expand(6, -1)
                            tgts = torch.tensor(rows, device=dev, dtype=torch.float32)
                            ainv_sq += F.mse_loss(mean_pred, tgts).item(); ainv_n += 1
                q7_mse_action_invariant = ainv_sq / max(1, ainv_n)

            print('  4a ep %d | huber=%.4f q7_mse_zero=%.4f '
                  'q7_mse_holdout=%.4f q7_mse_holdout_zero=%.4f '
                  'q7_mse_action_invariant=%.4f' % (
                  ep, q7_mse_train_final, q7_mse_zero_final,
                  q7_mse_holdout_final, q7_mse_holdout_zero_final,
                  q7_mse_action_invariant), flush=True)

    # ── STAGE 4b: train ActionPolicyHead + g via argmin-damage ────────────────
    train_g = (FREEZE_G_ALL != '1')   # FREEZE_G_ALL → keep loaded substrate fixed; train controller only
    for p in q7.parameters(): p.requires_grad_(False)
    for p in g.parameters(): p.requires_grad_(train_g)

    pi = ActionPolicyHead(D_S).to(dev)
    g_params_4b = list(g.parameters()) if train_g else []
    opt4b = torch.optim.Adam(list(pi.parameters()) + g_params_4b, lr=1e-3)
    print('vm: STAGE 4b train_g=%s (FREEZE_G_ALL=%s)' % (train_g, FREEZE_G_ALL), flush=True)

    print('vm: STAGE 4b — POLICY_EPOCHS=%d' % POLICY_EPOCHS, flush=True)
    for ep in range(POLICY_EPOCHS):
        random.shuffle(train_items)
        tot = 0.0; nb = 0
        for fam, wi, hids, dec, w in train_items:
            opt4b.zero_grad()
            loss = torch.tensor(0.0, device=dev)
            for ti, dec_idx, qtext, correct in dec:
                S = buildS_at(g, hids, ti).float()  # differentiable → g gets grad
                with torch.no_grad():
                    S_det = S.detach()
                    damages = [float(damage_scalar(q7(S_det, torch.tensor(ai, device=dev))))
                               for ai in range(6)]
                target_a = int(min(range(6), key=lambda ai: damages[ai]))
                logits = pi(S)  # grad through S → g
                loss = loss + F.cross_entropy(
                    logits.unsqueeze(0), torch.tensor([target_a], device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(pi.parameters()) + list(g.parameters()), 1.0)
            opt4b.step()
            tot += float(loss); nb += 1
        if ep % 5 == 0 or ep == POLICY_EPOCHS - 1:
            print('  4b ep %d | ce=%.4f' % (ep, tot / max(1, nb)), flush=True)

    # ── STAGE 4c: fine-tune g + fields via ce_action ──────────────────────────
    for p in q7.parameters(): p.requires_grad_(False)  # still frozen
    for p in pi.parameters(): p.requires_grad_(False)  # freeze π
    for p in g.parameters():  p.requires_grad_(train_g)
    g_params_4c = list(g.parameters()) if train_g else []
    opt4c = torch.optim.Adam(g_params_4c + fp, lr=5e-4)

    def ce_action_vm(qtext, correct, S):
        _fb['S'] = S; _fb['on'] = True
        pids   = tok(H.tmpl([{'role': 'user', 'content': qtext}]), return_tensors='pt').input_ids.to(dev)
        vids   = tok(correct, add_special_tokens=False, return_tensors='pt').input_ids.to(dev)
        eos    = torch.tensor([[tok.eos_token_id]], device=dev)
        ids    = torch.cat([pids, vids, eos], 1); P = pids.shape[1]; Lt = ids.shape[1] - P
        logits = model(ids).logits[0].float()
        loss   = F.cross_entropy(logits[P - 1:P + Lt - 1], ids[0, P:P + Lt])
        _fb['on'] = False; return loss

    print('vm: STAGE 4c — FIELD_EPOCHS_VM=%d' % FIELD_EPOCHS_VM, flush=True)
    for ep in range(FIELD_EPOCHS_VM):
        random.shuffle(train_items)
        tot = 0.0; nb = 0
        for fam, wi, hids, dec, w in train_items:
            opt4c.zero_grad(); wl = 0.0
            for ti, dec_idx, qtext, correct in dec:
                S = buildS_at(g, hids, ti)  # differentiable
                with torch.no_grad():
                    S_det = S.detach()
                    damages = [float(damage_scalar(q7(S_det, torch.tensor(ai, device=dev))))
                               for ai in range(6)]
                target_a = int(min(range(6), key=lambda ai: damages[ai]))
                loss = ce_action_vm(qtext, ACTIONS[target_a], S) / DEC_N
                if not torch.isfinite(loss): continue
                loss.backward(); wl += float(loss)
            torch.nn.utils.clip_grad_norm_(list(g.parameters()) + fp, 1.0)
            opt4c.step(); tot += wl; nb += 1
        if ep % 5 == 0 or ep == FIELD_EPOCHS_VM - 1:
            print('  4c ep %d | ce=%.4f' % (ep, tot / max(1, nb)), flush=True)

    # ── SAVE checkpoint ────────────────────────────────────────────────────────
    vm_ck = '%s/wp_vm_s%d.pt' % (CKDIR, SEED)
    torch.save({
        'q7':    q7.state_dict(),
        'pi':    pi.state_dict(),
        'g':     g.state_dict(),
        'fields': {str(L): _fb['fields'][L].state_dict() for L in FIELD_LAYERS},
    }, vm_ck)
    print('vm: saved checkpoint %s' % vm_ck, flush=True)

    # ── EVAL: mirror consequence() arms ───────────────────────────────────────
    arm_via = {arm: {} for arm in ['trained', 'reset', 'stale', 'zero', 'base']}
    stale_dir_map = {}
    for p in pi.parameters(): p.requires_grad_(False)
    for p in g.parameters():  p.requires_grad_(False)

    def _pick_action_pi(arm, hids, ti, stale_hids):
        """Return action string via argmax π(S) for the given arm."""
        if arm == 'reset':
            S = g.init().float()
        elif arm == 'zero':
            S = torch.zeros(K, D_S, device=dev)
        elif arm == 'stale':
            sh = stale_hids if stale_hids is not None else hids
            ti_s = min(ti, len(sh) - 1)
            S = buildS_at(g, sh, ti_s).float()
        else:  # trained
            S = buildS_at(g, hids, ti).float()
        return ACTIONS[int(pi(S).argmax())]

    for fam in FAMILIES:
        if fam not in pop_data: continue
        stale_fam  = STALE_SRC.get(fam)
        stale_pool = pop_data.get(stale_fam, []) if stale_fam else []

        for arm in ['trained', 'reset', 'stale', 'zero']:
            vias = []; sh_acc = 0; sd_total = 0
            for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                if arm == 'stale':
                    cands = [x for j, x in enumerate(stale_pool) if j != wi]
                    if not cands:
                        cands = [x for sf, sl in pop_data.items() if sf != fam for x in sl]
                    sh, _, _ = random.choice(cands) if cands else (hids, dec, w)
                    stale_hids = sh
                else:
                    stale_hids = None
                ws_e = newws(); via = 0.0
                with torch.no_grad():
                    for ti, dec_idx, qtext, correct in dec:
                        a = _pick_action_pi(arm, hids, ti, stale_hids)
                        if dec_idx == 2:
                            c2, _ = _fill_d2(w, ws_e); ws_e['correct2'] = c2
                        via += judge_world(fam, 'decision%d' % dec_idx, a, w, ws_e, dec_idx)
                        if arm == 'stale':
                            exp = EXPECTED_STALE.get(fam)
                            if fam == 'F2':
                                if dec_idx == 2:
                                    sd_total += 1
                                    if a != 'DEFER': sh_acc += 1
                            else:
                                sd_total += 1
                                if exp and a == exp: sh_acc += 1
                vias.append(via)
            arm_via[arm][fam] = sum(vias) / max(1, len(vias))
            if arm == 'stale':
                stale_dir_map[fam] = sh_acc / max(1, sd_total)

        # base arm (raw LLM, no field, no π)
        vias = []
        for hids, dec, w in pop_data[fam]:
            ws_b = newws(); via_b = 0.0
            for ti, dec_idx, qtext, correct in dec:
                with torch.no_grad():
                    r = gen([{'role': 'user', 'content': qtext}])
                r = r.split('</think>')[-1].strip()
                if dec_idx == 2:
                    c2, _ = _fill_d2(w, ws_b); ws_b['correct2'] = c2
                via_b += judge_world(fam, 'decision%d' % dec_idx, r, w, ws_b, dec_idx)
            vias.append(via_b)
        arm_via['base'][fam] = sum(vias) / max(1, len(vias))

    # ── PRINT per-family eval ─────────────────────────────────────────────────
    print('=== VM IN-DIST EVAL ===', flush=True)
    for fam in TRAIN_FAMILIES:
        if fam not in pop_data: continue
        tv = arm_via['trained'].get(fam, 0.0); rv = arm_via['reset'].get(fam, 0.0)
        sv = arm_via['stale'].get(fam, 0.0);   bv = arm_via['base'].get(fam, 0.0)
        print('  %s | trained=%.3f reset=%.3f stale=%.3f base=%.3f '
              'trained-reset=%.3f' % (fam, tv, rv, sv, bv, tv - rv), flush=True)

    print('=== VM TRANSFER EVAL ===', flush=True)
    for fam in HOLDOUT_FAMILIES:
        if fam not in pop_data: continue
        tv = arm_via['trained'].get(fam, 0.0); rv = arm_via['reset'].get(fam, 0.0)
        sv = arm_via['stale'].get(fam, 0.0);   bv = arm_via['base'].get(fam, 0.0)
        print('  %s | trained=%.3f reset=%.3f stale=%.3f base=%.3f '
              'trained-reset=%.3f' % (fam, tv, rv, sv, bv, tv - rv), flush=True)

    # ── ADDITIONAL DIAGNOSTICS ────────────────────────────────────────────────
    # calibration: frac train worlds where argmin-damage == correct action
    calib_correct = 0; calib_total = 0
    for fam in TRAIN_FAMILIES:
        if fam not in pop_data: continue
        for wi, (hids, dec, w) in enumerate(pop_data[fam]):
            for ti, dec_idx, qtext, correct in dec:
                with torch.no_grad():
                    S = buildS_at(g, hids, ti).float()
                    damages = [float(damage_scalar(q7(S, torch.tensor(ai, device=dev))))
                               for ai in range(6)]
                best_a = ACTIONS[int(min(range(6), key=lambda ai: damages[ai]))]
                calib_correct += int(best_a == correct)
                calib_total   += 1
    calib = calib_correct / max(1, calib_total)

    # per-world damage_std mean (action-conditioning signal)
    dmg_stds = []
    for fam in TRAIN_FAMILIES:
        if fam not in pop_data: continue
        for wi, (hids, dec, w) in enumerate(pop_data[fam]):
            for ti, dec_idx, qtext, correct in dec:
                with torch.no_grad():
                    S = buildS_at(g, hids, ti).float()
                    damages = [float(damage_scalar(q7(S, torch.tensor(ai, device=dev))))
                               for ai in range(6)]
                import statistics
                dmg_stds.append(statistics.stdev(damages) if len(damages) > 1 else 0.0)
    damage_std_mean = sum(dmg_stds) / max(1, len(dmg_stds))

    print('vm: calibration=%.3f  damage_std_mean=%.3f' % (calib, damage_std_mean), flush=True)
    print('vm: q7_mse_holdout=%.4f  q7_mse_holdout_zero=%.4f  ratio=%.3f' % (
          q7_mse_holdout_final, q7_mse_holdout_zero_final,
          q7_mse_holdout_final / max(1e-9, q7_mse_holdout_zero_final)), flush=True)

    # ── VERDICT ───────────────────────────────────────────────────────────────
    ho_tr_vals = [arm_via['trained'].get(h, 0.0) - arm_via['reset'].get(h, 0.0)
                  for h in HOLDOUT_FAMILIES if h in pop_data]
    transfer_gap = sum(ho_tr_vals) / max(1, len(ho_tr_vals))

    load_bearing       = transfer_gap >= 0.30
    pred_informative   = q7_mse_train_final < 0.6 * q7_mse_zero_final
    pred_generalizes   = q7_mse_holdout_final < 0.75 * q7_mse_holdout_zero_final
    transfer_ok        = transfer_gap >= 0.15
    action_cond        = damage_std_mean > 0.3
    calib_ok           = calib >= 0.85

    print('WP_VM_VERDICT: load_bearing=%s predictor_informative=%s '
          'predictor_generalizes=%s transfer=%s action_cond=%s calib=%s '
          '(transfer_gap=%.3f calib=%.3f damage_std=%.3f '
          'q7_mse_train=%.4f q7_mse_zero=%.4f)' % (
          load_bearing, pred_informative, pred_generalizes,
          transfer_ok, action_cond, calib_ok,
          transfer_gap, calib, damage_std_mean,
          q7_mse_train_final, q7_mse_zero_final), flush=True)
    print('=== WP_VM_DONE ===', flush=True)
    print('=== WP_VM_END ===', flush=True)


def viability_model_rel():
    """TRANSFER_FIX_V1 — role-relative consequence controller (ADDITIVE mode).

    Mirrors viability_model() exactly EXCEPT the consequence predictor: the free
    action-identity embedding (Consequence7Head) is replaced by a role-relative
    predictor factored through a requirement-readout corr(S):
      Stage 4a-I : train corr (RequirementReadout) to name the structurally-required action.
      Stage 4a-II: train q7_rel (RelativeConsequence7Head) to predict 7-dim consequences,
                   conditioned on detached corr(S) instead of a memorized action id.
      Stage 4b/4c: identical to viability_model but action selection uses q7_rel+corr.
    """
    rng = random.Random(SEED)
    all_worlds = {}
    for fam in FAMILIES:
        wl = []
        for i in range(N_WORLDS_PER_FAMILY):
            wl.append(make_world(fam, rng, i % N_TRAIN_TEMPLATES, i % 4))
        all_worlds[fam] = wl

    rcache = '%s/rollouts_wps%d_full_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if os.path.exists(rcache):
        print('vm_rel: loading cached rollouts %s' % rcache, flush=True)
        pop_data = torch.load(rcache, weights_only=False)
    else:
        print('vm_rel: collecting population rollouts ...', flush=True)
        pop_data = collect_population(all_worlds)
        torch.save(pop_data, rcache)
        print('vm_rel: saved rollouts %s' % rcache, flush=True)

    # ── init g and fields ──────────────────────────────────────────────────────
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    fp = [p for L in FIELD_LAYERS for p in _fb['fields'][L].parameters()]

    vmck = '%s/wps_trained_full_s%d_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if os.path.exists(vmck):
        print('vm_rel: loading g+fields from %s' % vmck, flush=True)
        sd = torch.load(vmck, weights_only=False)
        g.load_state_dict(sd['g'])
        for L in FIELD_LAYERS:
            if str(L) in sd.get('fields', {}):
                _fb['fields'][L].load_state_dict(sd['fields'][str(L)])

    # Override g with an arbitrary substrate ckpt (e.g. ESV4-E strengthened g).
    if SUBSTRATE_CKPT and os.path.exists(SUBSTRATE_CKPT):
        esd = torch.load(SUBSTRATE_CKPT, map_location=dev, weights_only=False)
        gsd = esd['g'] if (isinstance(esd, dict) and 'g' in esd) else esd
        miss, unexp = g.load_state_dict(gsd, strict=False)
        print('vm_rel: SUBSTRATE_CKPT loaded from %s | missing=%s unexpected=%s' % (
              SUBSTRATE_CKPT, list(miss), list(unexp)), flush=True)

    # ── predictors: role-relative (NO action-identity embedding) ───────────────
    corr   = RequirementReadout(D_S).to(dev)
    q7_rel = RelativeConsequence7Head(D_S).to(dev)

    train_items = [(fam, wi, hids, dec, w)
                   for fam in TRAIN_FAMILIES if fam in pop_data
                   for wi, (hids, dec, w) in enumerate(pop_data[fam])]

    # g stays frozen through all of stage 4a (corr/q7_rel train, not g).
    for p in g.parameters(): p.requires_grad_(False)

    # ── STAGE 4a-I: train corr (requirement readout) alone ─────────────────────
    opt_corr = torch.optim.Adam(corr.parameters(), lr=1e-3, weight_decay=CORR_WD)
    print('vm_rel: STAGE 4a-I — N_CORR_PRE=%d TRAIN=%s' % (N_CORR_PRE, TRAIN_FAMILIES), flush=True)
    corr_acc_train_final = float('nan'); corr_acc_holdout_final = float('nan')
    for ep in range(N_CORR_PRE):
        random.shuffle(train_items)
        tot_loss = 0.0; nb = 0
        for fam, wi, hids, dec, w in train_items:
            opt_corr.zero_grad()
            loss = torch.tensor(0.0, device=dev)
            for ti, dec_idx, qtext, correct in dec:
                gt_idx = ACTIONS.index(correct)   # intrinsic correct action == argmin-damage GT
                with torch.no_grad():
                    S = buildS_at(g, hids, ti).float()
                loss = loss + F.cross_entropy(corr(S).unsqueeze(0),
                                              torch.tensor([gt_idx], device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(corr.parameters(), 1.0)
            opt_corr.step()
            tot_loss += float(loss); nb += 1
        if ep % 5 == 0 or ep == N_CORR_PRE - 1:
            corr.eval()   # dropout OFF for measurement
            with torch.no_grad():
                ct = 0; cn = 0
                for fam in TRAIN_FAMILIES:
                    if fam not in pop_data: continue
                    for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                        for ti, dec_idx, qtext, correct in dec:
                            S = buildS_at(g, hids, ti).float()
                            ct += int(int(corr(S).argmax()) == ACTIONS.index(correct)); cn += 1
                corr_acc_train_final = ct / max(1, cn)
                ch = 0; chn = 0
                for fam in HOLDOUT_FAMILIES:
                    if fam not in pop_data: continue
                    for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                        for ti, dec_idx, qtext, correct in dec:
                            S = buildS_at(g, hids, ti).float()
                            ch += int(int(corr(S).argmax()) == ACTIONS.index(correct)); chn += 1
                corr_acc_holdout_final = ch / max(1, chn)
            corr.train()  # back to train for next epoch's grad steps
            print('  4a-I ep %d | corr_ce=%.4f corr_acc_train=%.3f corr_acc_holdout=%.3f' % (
                  ep, tot_loss / max(1, nb), corr_acc_train_final, corr_acc_holdout_final), flush=True)

    corr.eval()   # dropout OFF for all downstream inference (4a-II / 4b / 4c / eval)
    # ── STAGE 4a-II: train q7_rel (corr frozen) ────────────────────────────────
    for p in corr.parameters(): p.requires_grad_(False)
    opt4a = torch.optim.Adam(q7_rel.parameters(), lr=1e-3)
    print('vm_rel: STAGE 4a-II — N_Q7_EPOCHS=%d (corr frozen)' % N_Q7_EPOCHS, flush=True)
    q7_mse_train_final = float('nan'); q7_mse_zero_final = float('nan')
    q7_mse_holdout_final = float('nan'); q7_mse_holdout_zero_final = float('nan')
    for ep in range(N_Q7_EPOCHS):
        random.shuffle(train_items)
        tot_loss = 0.0; nb = 0
        for fam, wi, hids, dec, w in train_items:
            opt4a.zero_grad()
            loss = torch.tensor(0.0, device=dev)
            for ti, dec_idx, qtext, correct in dec:
                inner = dec_idx - 1
                with torch.no_grad():
                    S = buildS_at(g, hids, ti).float()
                cl   = corr(S).detach()
                rows = [compute_7dim_consequence(w, ACTIONS[ai], inner) for ai in range(6)]
                for ai in range(6):
                    pred = q7_rel(S, ai, cl)
                    tgt  = torch.tensor(rows[ai], device=dev, dtype=torch.float32)
                    loss = loss + F.huber_loss(pred, tgt, delta=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q7_rel.parameters(), 1.0)
            opt4a.step()
            tot_loss += float(loss); nb += 1

        if ep % 10 == 0 or ep == N_Q7_EPOCHS - 1:
            with torch.no_grad():
                q7_mse_train_final = tot_loss / max(1, nb)
                Sz   = torch.zeros(K, D_S, device=dev)
                cl_z = corr(Sz)

                # q7_mse_zero: train families, S=zeros (verdict pred_informative)
                z_sq = 0.0; z_n = 0
                for fam in TRAIN_FAMILIES:
                    if fam not in pop_data: continue
                    for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                        for ti, dec_idx, qtext, correct in dec:
                            inner = dec_idx - 1
                            rows = [compute_7dim_consequence(w, ACTIONS[ai], inner) for ai in range(6)]
                            for ai in range(6):
                                pred = q7_rel(Sz, ai, cl_z)
                                tgt  = torch.tensor(rows[ai], device=dev, dtype=torch.float32)
                                z_sq += F.mse_loss(pred, tgt).item(); z_n += 1
                q7_mse_zero_final = z_sq / max(1, z_n)

                # q7_mse_holdout and q7_mse_holdout_zero
                ho_sq = 0.0; ho_n = 0; hoz_sq = 0.0; hoz_n = 0
                for fam in HOLDOUT_FAMILIES:
                    if fam not in pop_data: continue
                    for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                        for ti, dec_idx, qtext, correct in dec:
                            inner = dec_idx - 1
                            S    = buildS_at(g, hids, ti).float()
                            cl   = corr(S)
                            rows = [compute_7dim_consequence(w, ACTIONS[ai], inner) for ai in range(6)]
                            for ai in range(6):
                                pred  = q7_rel(S,  ai, cl)
                                predz = q7_rel(Sz, ai, cl_z)
                                tgt   = torch.tensor(rows[ai], device=dev, dtype=torch.float32)
                                ho_sq  += F.mse_loss(pred,  tgt).item(); ho_n  += 1
                                hoz_sq += F.mse_loss(predz, tgt).item(); hoz_n += 1
                q7_mse_holdout_final      = ho_sq  / max(1, ho_n)
                q7_mse_holdout_zero_final = hoz_sq / max(1, hoz_n)
            print('  4a-II ep %d | huber=%.4f q7_mse_zero=%.4f '
                  'q7_mse_holdout=%.4f q7_mse_holdout_zero=%.4f' % (
                  ep, q7_mse_train_final, q7_mse_zero_final,
                  q7_mse_holdout_final, q7_mse_holdout_zero_final), flush=True)

    # ── STAGE 4b: train ActionPolicyHead + g via argmin-damage (q7_rel+corr) ───
    train_g = (FREEZE_G_ALL != '1')   # FREEZE_G_ALL → keep loaded substrate fixed; train controller only
    for p in q7_rel.parameters(): p.requires_grad_(False)
    for p in corr.parameters():   p.requires_grad_(False)
    for p in g.parameters():      p.requires_grad_(train_g)

    pi = ActionPolicyHead(D_S).to(dev)
    g_params_4b = list(g.parameters()) if train_g else []
    opt4b = torch.optim.Adam(list(pi.parameters()) + g_params_4b, lr=1e-3)
    print('vm_rel: STAGE 4b train_g=%s (FREEZE_G_ALL=%s)' % (train_g, FREEZE_G_ALL), flush=True)
    print('vm_rel: STAGE 4b — POLICY_EPOCHS=%d' % POLICY_EPOCHS, flush=True)
    for ep in range(POLICY_EPOCHS):
        random.shuffle(train_items)
        tot = 0.0; nb = 0
        for fam, wi, hids, dec, w in train_items:
            opt4b.zero_grad()
            loss = torch.tensor(0.0, device=dev)
            for ti, dec_idx, qtext, correct in dec:
                S = buildS_at(g, hids, ti).float()  # differentiable → g gets grad
                with torch.no_grad():
                    S_det = S.detach()
                    cl    = corr(S_det).detach()
                    damages = [float(damage_scalar(q7_rel(S_det, ai, cl))) for ai in range(6)]
                target_a = int(min(range(6), key=lambda ai: damages[ai]))
                logits = pi(S)  # grad through S → g
                loss = loss + F.cross_entropy(
                    logits.unsqueeze(0), torch.tensor([target_a], device=dev))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(pi.parameters()) + list(g.parameters()), 1.0)
            opt4b.step()
            tot += float(loss); nb += 1
        if ep % 5 == 0 or ep == POLICY_EPOCHS - 1:
            print('  4b ep %d | ce=%.4f' % (ep, tot / max(1, nb)), flush=True)

    # ── STAGE 4c: fine-tune g + fields via ce_action (q7_rel+corr targets) ─────
    for p in q7_rel.parameters(): p.requires_grad_(False)
    for p in corr.parameters():   p.requires_grad_(False)
    for p in pi.parameters():     p.requires_grad_(False)
    for p in g.parameters():      p.requires_grad_(train_g)
    g_params_4c = list(g.parameters()) if train_g else []
    opt4c = torch.optim.Adam(g_params_4c + fp, lr=5e-4)

    def ce_action_vm(qtext, correct, S):
        _fb['S'] = S; _fb['on'] = True
        pids   = tok(H.tmpl([{'role': 'user', 'content': qtext}]), return_tensors='pt').input_ids.to(dev)
        vids   = tok(correct, add_special_tokens=False, return_tensors='pt').input_ids.to(dev)
        eos    = torch.tensor([[tok.eos_token_id]], device=dev)
        ids    = torch.cat([pids, vids, eos], 1); P = pids.shape[1]; Lt = ids.shape[1] - P
        logits = model(ids).logits[0].float()
        loss   = F.cross_entropy(logits[P - 1:P + Lt - 1], ids[0, P:P + Lt])
        _fb['on'] = False; return loss

    print('vm_rel: STAGE 4c — FIELD_EPOCHS_VM=%d' % FIELD_EPOCHS_VM, flush=True)
    for ep in range(FIELD_EPOCHS_VM):
        random.shuffle(train_items)
        tot = 0.0; nb = 0
        for fam, wi, hids, dec, w in train_items:
            opt4c.zero_grad(); wl = 0.0
            for ti, dec_idx, qtext, correct in dec:
                S = buildS_at(g, hids, ti)  # differentiable
                with torch.no_grad():
                    S_det = S.detach()
                    cl    = corr(S_det).detach()
                    damages = [float(damage_scalar(q7_rel(S_det, ai, cl))) for ai in range(6)]
                target_a = int(min(range(6), key=lambda ai: damages[ai]))
                loss = ce_action_vm(qtext, ACTIONS[target_a], S) / DEC_N
                if not torch.isfinite(loss): continue
                loss.backward(); wl += float(loss)
            torch.nn.utils.clip_grad_norm_(list(g.parameters()) + fp, 1.0)
            opt4c.step(); tot += wl; nb += 1
        if ep % 5 == 0 or ep == FIELD_EPOCHS_VM - 1:
            print('  4c ep %d | ce=%.4f' % (ep, tot / max(1, nb)), flush=True)

    # ── SAVE checkpoint ────────────────────────────────────────────────────────
    vm_ck = '%s/wp_vm_rel_s%d.pt' % (CKDIR, SEED)
    torch.save({
        'variant': 'rel',
        'corr':    corr.state_dict(),
        'q7_rel':  q7_rel.state_dict(),
        'pi':      pi.state_dict(),
        'g':       g.state_dict(),
        'fields':  {str(L): _fb['fields'][L].state_dict() for L in FIELD_LAYERS},
    }, vm_ck)
    print('vm_rel: saved checkpoint %s' % vm_ck, flush=True)

    # ── EVAL: mirror consequence() arms (action = argmax π(S)) ─────────────────
    arm_via = {arm: {} for arm in ['trained', 'reset', 'stale', 'zero', 'base']}
    stale_dir_map = {}
    for p in pi.parameters(): p.requires_grad_(False)
    for p in g.parameters():  p.requires_grad_(False)

    def _pick_action_pi(arm, hids, ti, stale_hids):
        """Return action string via argmax π(S) for the given arm."""
        if arm == 'reset':
            S = g.init().float()
        elif arm == 'zero':
            S = torch.zeros(K, D_S, device=dev)
        elif arm == 'stale':
            sh = stale_hids if stale_hids is not None else hids
            ti_s = min(ti, len(sh) - 1)
            S = buildS_at(g, sh, ti_s).float()
        else:  # trained
            S = buildS_at(g, hids, ti).float()
        return ACTIONS[int(pi(S).argmax())]

    for fam in FAMILIES:
        if fam not in pop_data: continue
        stale_fam  = STALE_SRC.get(fam)
        stale_pool = pop_data.get(stale_fam, []) if stale_fam else []

        for arm in ['trained', 'reset', 'stale', 'zero']:
            vias = []; sh_acc = 0; sd_total = 0
            for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                if arm == 'stale':
                    cands = [x for j, x in enumerate(stale_pool) if j != wi]
                    if not cands:
                        cands = [x for sf, sl in pop_data.items() if sf != fam for x in sl]
                    sh, _, _ = random.choice(cands) if cands else (hids, dec, w)
                    stale_hids = sh
                else:
                    stale_hids = None
                ws_e = newws(); via = 0.0
                with torch.no_grad():
                    for ti, dec_idx, qtext, correct in dec:
                        a = _pick_action_pi(arm, hids, ti, stale_hids)
                        if dec_idx == 2:
                            c2, _ = _fill_d2(w, ws_e); ws_e['correct2'] = c2
                        via += judge_world(fam, 'decision%d' % dec_idx, a, w, ws_e, dec_idx)
                        if arm == 'stale':
                            exp = EXPECTED_STALE.get(fam)
                            if fam == 'F2':
                                if dec_idx == 2:
                                    sd_total += 1
                                    if a != 'DEFER': sh_acc += 1
                            else:
                                sd_total += 1
                                if exp and a == exp: sh_acc += 1
                vias.append(via)
            arm_via[arm][fam] = sum(vias) / max(1, len(vias))
            if arm == 'stale':
                stale_dir_map[fam] = sh_acc / max(1, sd_total)

        # base arm (raw LLM, no field, no π)
        vias = []
        for hids, dec, w in pop_data[fam]:
            ws_b = newws(); via_b = 0.0
            for ti, dec_idx, qtext, correct in dec:
                with torch.no_grad():
                    r = gen([{'role': 'user', 'content': qtext}])
                r = r.split('</think>')[-1].strip()
                if dec_idx == 2:
                    c2, _ = _fill_d2(w, ws_b); ws_b['correct2'] = c2
                via_b += judge_world(fam, 'decision%d' % dec_idx, r, w, ws_b, dec_idx)
            vias.append(via_b)
        arm_via['base'][fam] = sum(vias) / max(1, len(vias))

    # ── PRINT per-family eval ─────────────────────────────────────────────────
    print('=== VM_REL IN-DIST EVAL ===', flush=True)
    for fam in TRAIN_FAMILIES:
        if fam not in pop_data: continue
        tv = arm_via['trained'].get(fam, 0.0); rv = arm_via['reset'].get(fam, 0.0)
        sv = arm_via['stale'].get(fam, 0.0);   bv = arm_via['base'].get(fam, 0.0)
        print('  %s | trained=%.3f reset=%.3f stale=%.3f base=%.3f '
              'trained-reset=%.3f' % (fam, tv, rv, sv, bv, tv - rv), flush=True)

    print('=== VM_REL TRANSFER EVAL ===', flush=True)
    for fam in HOLDOUT_FAMILIES:
        if fam not in pop_data: continue
        tv = arm_via['trained'].get(fam, 0.0); rv = arm_via['reset'].get(fam, 0.0)
        sv = arm_via['stale'].get(fam, 0.0);   bv = arm_via['base'].get(fam, 0.0)
        print('  %s | trained=%.3f reset=%.3f stale=%.3f base=%.3f '
              'trained-reset=%.3f' % (fam, tv, rv, sv, bv, tv - rv), flush=True)

    # ── ADDITIONAL DIAGNOSTICS (q7_rel+corr) ──────────────────────────────────
    calib_correct = 0; calib_total = 0
    for fam in TRAIN_FAMILIES:
        if fam not in pop_data: continue
        for wi, (hids, dec, w) in enumerate(pop_data[fam]):
            for ti, dec_idx, qtext, correct in dec:
                with torch.no_grad():
                    S = buildS_at(g, hids, ti).float()
                    cl = corr(S)
                    damages = [float(damage_scalar(q7_rel(S, ai, cl))) for ai in range(6)]
                best_a = ACTIONS[int(min(range(6), key=lambda ai: damages[ai]))]
                calib_correct += int(best_a == correct)
                calib_total   += 1
    calib = calib_correct / max(1, calib_total)

    dmg_stds = []
    for fam in TRAIN_FAMILIES:
        if fam not in pop_data: continue
        for wi, (hids, dec, w) in enumerate(pop_data[fam]):
            for ti, dec_idx, qtext, correct in dec:
                with torch.no_grad():
                    S = buildS_at(g, hids, ti).float()
                    cl = corr(S)
                    damages = [float(damage_scalar(q7_rel(S, ai, cl))) for ai in range(6)]
                import statistics
                dmg_stds.append(statistics.stdev(damages) if len(damages) > 1 else 0.0)
    damage_std_mean = sum(dmg_stds) / max(1, len(dmg_stds))

    print('vm_rel: calibration=%.3f  damage_std_mean=%.3f' % (calib, damage_std_mean), flush=True)
    print('vm_rel: corr_acc_train=%.3f corr_acc_holdout=%.3f' % (
          corr_acc_train_final, corr_acc_holdout_final), flush=True)
    print('vm_rel: q7_mse_holdout=%.4f  q7_mse_holdout_zero=%.4f  ratio=%.3f' % (
          q7_mse_holdout_final, q7_mse_holdout_zero_final,
          q7_mse_holdout_final / max(1e-9, q7_mse_holdout_zero_final)), flush=True)

    # ── VERDICT ───────────────────────────────────────────────────────────────
    ho_tr_vals = [arm_via['trained'].get(h, 0.0) - arm_via['reset'].get(h, 0.0)
                  for h in HOLDOUT_FAMILIES if h in pop_data]
    transfer_gap = sum(ho_tr_vals) / max(1, len(ho_tr_vals))

    load_bearing       = transfer_gap >= 0.30
    pred_informative   = q7_mse_train_final < 0.6 * q7_mse_zero_final
    pred_generalizes   = q7_mse_holdout_final < 0.75 * q7_mse_holdout_zero_final
    transfer_ok        = transfer_gap >= 0.15
    action_cond        = damage_std_mean > 0.3
    calib_ok           = calib >= 0.85

    print('WP_VM_VERDICT: load_bearing=%s predictor_informative=%s '
          'predictor_generalizes=%s transfer=%s action_cond=%s calib=%s '
          '(transfer_gap=%.3f calib=%.3f damage_std=%.3f '
          'q7_mse_train=%.4f q7_mse_zero=%.4f)' % (
          load_bearing, pred_informative, pred_generalizes,
          transfer_ok, action_cond, calib_ok,
          transfer_gap, calib, damage_std_mean,
          q7_mse_train_final, q7_mse_zero_final), flush=True)
    print('=== WP_VM_REL_DONE ===', flush=True)
    print('=== WP_VM_REL_END ===', flush=True)


def vm_diag():
    """DIAGNOSTIC (no training): localize WHY the consequence-viability controller's
    viability fails to transfer to held-out world families.

    Reuses the cached rollouts + a trained wp_vm checkpoint. Pure cache+ckpt analysis;
    performs NO LLM forward and NO optimization. Decomposes the
        S  ->  q7/damage  ->  pi  ->  action  ->  viability
    chain TRAIN-vs-HOLDOUT so the broken link is observable per split.
    """
    VM_DIAG_CKPT = os.environ.get('VM_DIAG_CKPT', '')

    # ── load cached rollouts (MUST exist; never collect in diag mode) ───────────
    rcache = '%s/rollouts_wps%d_full_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if not os.path.exists(rcache):
        print('VM_DIAG_NO_CACHE %s' % rcache, flush=True); return
    print('vm_diag: loading cached rollouts %s' % rcache, flush=True)
    pop_data = torch.load(rcache, weights_only=False)

    # ── load g / fields / q7 / pi from checkpoint ───────────────────────────────
    ckpt_path = VM_DIAG_CKPT or '%s/wp_vm_s%d.pt' % (CKDIR, SEED)
    if not os.path.exists(ckpt_path):
        print('VM_DIAG_NO_CKPT %s' % ckpt_path, flush=True); return
    sd = torch.load(ckpt_path, map_location=dev, weights_only=False)
    g  = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev); g.load_state_dict(sd['g'])
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    for L in FIELD_LAYERS:
        if str(L) in sd.get('fields', {}):
            _fb['fields'][L].load_state_dict(sd['fields'][str(L)])
    REL = (sd.get('variant') == 'rel')
    if REL:
        corr   = RequirementReadout(D_S).to(dev);       corr.load_state_dict(sd['corr'])
        q7_rel = RelativeConsequence7Head(D_S).to(dev); q7_rel.load_state_dict(sd['q7_rel'])
        corr.eval(); q7_rel.eval(); q7 = None
    else:
        q7 = Consequence7Head(D_S).to(dev); q7.load_state_dict(sd['q7']); q7.eval()
    pi = ActionPolicyHead(D_S).to(dev); pi.load_state_dict(sd['pi'])
    g.eval(); pi.eval()
    for L in FIELD_LAYERS: _fb['fields'][L].eval()
    _fb['on'] = False   # no field hook; vm_diag does not run the LLM
    print('vm_diag: loaded checkpoint %s' % ckpt_path, flush=True)

    def ang(a, b):  # angle (radians) between two flattened vectors
        a = a.flatten().float(); b = b.flatten().float()
        cs = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).clamp(-1.0, 1.0)
        return float(torch.acos(cs))

    acc = {sp: {'q7_calib': 0, 'pi_calib': 0, 'pi_fid': 0, 's_ang': 0.0,
                'perdim': [0.0] * 7, 'viab_q7': 0.0, 'viab_pi': 0.0,
                'viab_oracle': 0.0, 'corr_calib': 0, 'q7gt_calib': 0, 'count': 0}
           for sp in ('TRAIN', 'HOLDOUT')}

    diag_fams = []; _seen = set()
    for f in (TRAIN_FAMILIES + HOLDOUT_FAMILIES):
        if f in pop_data and f not in _seen:
            diag_fams.append(f); _seen.add(f)

    with torch.no_grad():
        for fam in diag_fams:
            sp = 'TRAIN' if fam in TRAIN_FAMILIES else 'HOLDOUT'
            a  = acc[sp]
            stale_fam  = STALE_SRC.get(fam)
            stale_pool = pop_data.get(stale_fam, []) if stale_fam else []
            if not stale_pool:   # mirror eval fallback to all other families
                stale_pool = [x for sf, sl in pop_data.items() if sf != fam for x in sl]
            for wi, (hids, dec, w) in enumerate(pop_data[fam]):
                stale_hids = stale_pool[min(wi, len(stale_pool) - 1)][0] if stale_pool else hids
                ws_q7 = newws(); ws_pi = newws(); ws_ora = newws()  # stateful per source
                for ti, dec_idx, qtext, correct in dec:
                    S       = buildS_at(g, hids, ti).float()             # right-world S
                    ti_s    = min(ti, len(stale_hids) - 1)
                    S_stale = buildS_at(g, stale_hids, ti_s).float()     # wrong-world S
                    inner   = dec_idx - 1   # compute_7dim_consequence: 0=D1, 1=terminal D2
                    ci_correct = ACTIONS.index(correct)
                    if REL:
                        cl      = corr(S)
                        dmg     = [float(damage_scalar(q7_rel(S, ai, cl))) for ai in range(6)]
                        a_q7    = int(min(range(6), key=lambda ai: dmg[ai]))
                        q7_pred = [[float(x) for x in q7_rel(S, ai, cl)] for ai in range(6)]
                        a['corr_calib'] += int(int(cl.argmax()) == ci_correct)
                        # GT-MATCH ABLATION: feed q7_rel a (scaled) one-hot of the correct action
                        gt_onehot = torch.zeros(6, device=dev); gt_onehot[ci_correct] = 5.0
                        dmg_gt  = [float(damage_scalar(q7_rel(S, ai, gt_onehot))) for ai in range(6)]
                        a['q7gt_calib'] += int(int(min(range(6), key=lambda ai: dmg_gt[ai])) == ci_correct)
                    else:
                        dmg     = [float(damage_scalar(q7(S, torch.tensor(ai, device=dev)))) for ai in range(6)]
                        a_q7    = int(min(range(6), key=lambda ai: dmg[ai]))
                        q7_pred = [[float(x) for x in q7(S, torch.tensor(ai, device=dev))] for ai in range(6)]
                    a_pi    = int(torch.argmax(pi(S)).item())
                    c7_true = [compute_7dim_consequence(w, ACTIONS[ai], inner) for ai in range(6)]
                    # link metrics ----------------------------------------------------
                    a['q7_calib'] += int(a_q7 == ci_correct)             # damage argmin == correct
                    a['pi_calib'] += int(a_pi == ci_correct)             # policy argmax == correct
                    a['pi_fid']   += int(a_pi == a_q7)                   # policy tracks damage head
                    a['s_ang']    += ang(S, S_stale)                     # world-specificity of S
                    for d in range(7):
                        a['perdim'][d] += sum(abs(q7_pred[ai][d] - c7_true[ai][d])
                                              for ai in range(6)) / 6.0
                    # viability via the SAME judge the eval uses (stateful ws per source)
                    kind = 'decision%d' % dec_idx
                    if dec_idx == 2:
                        ws_q7['correct2']  = _fill_d2(w, ws_q7)[0]
                        ws_pi['correct2']  = _fill_d2(w, ws_pi)[0]
                        ws_ora['correct2'] = _fill_d2(w, ws_ora)[0]
                    a['viab_q7']     += judge_world(fam, kind, ACTIONS[a_q7], w, ws_q7,  dec_idx)
                    a['viab_pi']     += judge_world(fam, kind, ACTIONS[a_pi], w, ws_pi,  dec_idx)
                    a['viab_oracle'] += judge_world(fam, kind, correct,        w, ws_ora, dec_idx)
                    a['count']       += 1

    # ── TRAIN-vs-HOLDOUT table ──────────────────────────────────────────────────
    for sp in ('TRAIN', 'HOLDOUT'):
        a = acc[sp]; n = max(1, a['count'])
        extra = (' corr_acc=%.3f q7gt_calib=%.3f' % (a['corr_calib'] / n, a['q7gt_calib'] / n)) if REL else ''
        print('VM_DIAG split=%s n=%d  q7_calib=%.3f pi_calib=%.3f pi_fid=%.3f  '
              's_vs_stale=%.4f  viab_q7=%.3f viab_pi=%.3f viab_oracle=%.3f%s' % (
              sp, a['count'], a['q7_calib'] / n, a['pi_calib'] / n, a['pi_fid'] / n,
              a['s_ang'] / n, a['viab_q7'] / n, a['viab_pi'] / n, a['viab_oracle'] / n, extra), flush=True)

    for sp in ('TRAIN', 'HOLDOUT'):
        a = acc[sp]; n = max(1, a['count']); pd = [x / n for x in a['perdim']]
        print('VM_DIAG perdim_err split=%s : continuity=%.3f contradiction=%.3f recovery=%.3f '
              'optionloss=%.3f completion=%.3f contamination=%.3f instability=%.3f' % (
              sp, pd[0], pd[1], pd[2], pd[3], pd[4], pd[5], pd[6]), flush=True)

    # ── STRUCTURE-SHARING (ground-truth, controller-independent) ────────────────
    #   Do TRAIN & HOLDOUT families share causal viability structure, or are the
    #   held-out families structurally novel? Each situation -> 42-d ground-truth
    #   consequence profile (6 actions x 7 dims). Compare holdout->nearest-train
    #   vs within-train NN, and whether the nearest train neighbor shares the
    #   correct action (decision-structure sharing).
    tr_prof, tr_corr = [], []; ho_prof, ho_corr, ho_fam_l = [], [], []
    for fam in pop_data:
        is_ho = fam in HOLDOUT_FAMILIES
        for (hids, dec, w) in pop_data[fam]:
            for (ti, dec_idx, qtext, correct) in dec:
                inner = dec_idx - 1
                prof = []
                for ai in range(6):
                    prof.extend(compute_7dim_consequence(w, ACTIONS[ai], inner))
                v = np.array(prof, dtype=float)
                if is_ho: ho_prof.append(v); ho_corr.append(correct); ho_fam_l.append(fam)
                else:     tr_prof.append(v); tr_corr.append(correct)
    struct_ratio = float('nan'); corr_share = float('nan')
    if tr_prof and ho_prof:
        TM = np.stack(tr_prof)
        # NOTE: consequence profiles are deterministic in (family,action,flags), so many
        # train worlds share IDENTICAL profiles → NN-min is 0 (useless as a scale).
        # Use the train-profile SPREAD (mean pairwise distance) as the reference scale.
        within_train_meanpair = float(np.mean([
            float(np.linalg.norm(TM - TM[i], axis=1).mean()) for i in range(len(TM))]))
        h2t = []; cm = 0
        for j, v in enumerate(ho_prof):
            d = np.linalg.norm(TM - v, axis=1); k = int(d.argmin())
            h2t.append(float(d[k])); cm += int(tr_corr[k] == ho_corr[j])
        holdout_to_train_nn = float(np.mean(h2t))
        within_train_nn = within_train_meanpair  # report the spread as the scale
        struct_ratio = holdout_to_train_nn / (within_train_meanpair + 1e-9)
        corr_share = cm / max(1, len(ho_prof))
        print('VM_DIAG_STRUCT within_train_nn=%.4f holdout_to_train_nn=%.4f ratio=%.3f '
              'nn_correct_action_match=%.3f  (ratio~1 => shared structure; >>1 => holdout novel)' % (
              within_train_nn, holdout_to_train_nn, struct_ratio, corr_share), flush=True)
        for hf in sorted(set(ho_fam_l)):
            idxs = [j for j in range(len(ho_fam_l)) if ho_fam_l[j] == hf]
            d2 = float(np.mean([float(np.linalg.norm(TM - ho_prof[j], axis=1).min()) for j in idxs]))
            cmf = float(np.mean([int(tr_corr[int(np.linalg.norm(TM - ho_prof[j], axis=1).argmin())] == ho_corr[j]) for j in idxs]))
            print('VM_DIAG_STRUCT family=%s n=%d holdout_to_train_nn=%.4f nn_correct_action_match=%.3f' % (
                  hf, len(idxs), d2, cmf), flush=True)

    # ── localized verdict ───────────────────────────────────────────────────────
    h = acc['HOLDOUT']; nh = max(1, h['count'])
    s_ho   = h['s_ang'] / nh
    q7c_ho = h['q7_calib'] / nh
    pic_ho = h['pi_calib'] / nh
    vpi_ho = h['viab_pi'] / nh
    vor_ho = h['viab_oracle'] / nh
    s_world_specific_holdout = s_ho > 0.1
    q7_generalizes           = q7c_ho >= 0.75
    pi_generalizes           = pic_ho >= 0.75
    oracle_headroom          = vor_ho - vpi_ho   # base action not cached → headroom vs pi
    print('VM_DIAG_VERDICT: s_world_specific_holdout=%s q7_generalizes=%s pi_generalizes=%s '
          'oracle_headroom=%.3f (s_vs_stale_holdout=%.4f q7_calib_holdout=%.3f '
          'pi_calib_holdout=%.3f viab_pi_holdout=%.3f viab_oracle_holdout=%.3f)' % (
          s_world_specific_holdout, q7_generalizes, pi_generalizes, oracle_headroom,
          s_ho, q7c_ho, pic_ho, vpi_ho, vor_ho), flush=True)
    # ── TRANSFER-FAILURE INTERPRETATION: shared-structure vs family-local ───────
    shared_structure = (not np.isnan(struct_ratio)) and struct_ratio < 2.0 and corr_share > 0.5
    controller_uses_it = q7_generalizes and pi_generalizes
    if not shared_structure:
        tf = 'DISTINCT_STRUCTURE (holdout families structurally novel — nearest-train far / correct-action mismatch; transfer is task-hard, not just controller)'
    elif shared_structure and not controller_uses_it:
        tf = 'FAMILY_LOCAL_MAPPINGS (structure IS shared but controller learns family-local action-maps — q7/pi pick wrong action on holdout despite shared GT structure)'
    else:
        tf = 'AMBIGUOUS (structure shared + controller calibrated on holdout — re-examine viability/headroom)'
    print('VM_DIAG_TRANSFER_FAILURE: %s | struct_ratio=%.3f nn_correct_action_match=%.3f '
          'q7_calib_holdout=%.3f pi_calib_holdout=%.3f' % (
          tf, struct_ratio, corr_share, q7c_ho, pic_ho), flush=True)
    # ── role-relative localization (rel variant only) ───────────────────────────
    if REL:
        corr_acc_ho   = h['corr_calib'] / nh    # requirement-readout transfer
        q7gt_calib_ho = h['q7gt_calib'] / nh    # q7_rel under GT-matched corr
        if corr_acc_ho < 0.4:
            rtf = 'CORR_FAMILY_LOCAL (requirement-readout itself fails to transfer to holdout)'
        elif q7c_ho < 0.167 and q7gt_calib_ho > 0.5:
            rtf = 'CORR_OK_BUT_Q7_NEEDS_BETTER_CORR (q7_rel calibrates under GT corr but not learned corr)'
        elif q7gt_calib_ho < 0.167:
            rtf = 'Q7_REL_OTHER_SHORTCUT (q7_rel ignores corr / uses a different shortcut)'
        elif q7c_ho > 0.5:
            rtf = 'FIXED_TRANSFERS (role-relative predictor transfers to holdout)'
        else:
            rtf = 'PARTIAL (corr transfers; q7_rel calibration intermediate)'
        print('VM_DIAG_REL_TRANSFER: %s | corr_acc_holdout=%.3f q7_rel_calib_holdout=%.3f '
              'q7gt_calib_holdout=%.3f' % (rtf, corr_acc_ho, q7c_ho, q7gt_calib_ho), flush=True)
    print('=== VM_DIAG_DONE ===', flush=True)
    print('=== VM_DIAG_END ===', flush=True)


# ── SELF-DISTILL LOOP (continuous closed-loop Qwen reasoning distillation) ──────
@torch.no_grad()
def gen_reasoning(qtext, S, T, max_new, ctx=''):
    """Generate a CoT reasoning trajectory WITH the field ON (closed loop).

    Returns (reasoning_rep[D_MODEL], action_softdist[6], action_idx, reasoning_text).
    Qwen reasons step-by-step; we read the soft action-distribution conditioned on
    that reasoning plus the dense read-layer reasoning representation.
    ctx = prior context (e.g. the commit-turn commitment) the decision depends on —
    the teacher needs it to reason correctly (the decision prompt alone omits it).
    """
    prompt = ((ctx + '\n\n') if ctx else '') + qtext + (
                      '\n\nThink step by step about which single action is correct, '
                      'then on the final line write exactly:\n'
                      'ACTION: <one of KEEP, REJECT, REPAIR, RELEASE, DEFER, ASK>')
    ids = tok(H.tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').input_ids.to(dev)
    # field control during generation: EPS=1.0 corrupts 200-tok free generation into garbage;
    # SDL_FIELD_GEN=0 → clean teacher; SDL_GEN_EPS>=0 → gentle field that steers without destroying.
    _eps_saved = {}
    if SDL_GEN_EPS >= 0:
        for L in FIELD_LAYERS:
            _eps_saved[L] = _fb['fields'][L].eps; _fb['fields'][L].eps = SDL_GEN_EPS
    _fb['S'] = S; _fb['on'] = (SDL_FIELD_GEN == 1)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=T,
                         top_p=0.95, pad_token_id=tok.eos_token_id)
    gen_ids = out[0, ids.shape[1]:]
    reasoning_text = tok.decode(gen_ids, skip_special_tokens=True)
    # ROBUST action parse from the generated TEXT ('ACTION: X'); the per-token logit lookup
    # collides on RE-prefixed actions (REJECT/REPAIR/RELEASE) and mismatches the leading space.
    rt_up = reasoning_text.upper()
    action_idx = None
    if 'ACTION:' in rt_up:
        after = rt_up.split('ACTION:')[-1]
        for ai, a in enumerate(ACTIONS):
            if a in after:
                action_idx = ai; break
    if action_idx is None:
        poss = {a: rt_up.rfind(a) for a in ACTIONS}
        if max(poss.values()) >= 0:
            action_idx = ACTIONS.index(max(poss, key=poss.get))
    if action_idx is None:
        action_idx = 0
    # soft requirement target: label-smoothed one-hot on the parsed action (teacher is strong,
    # so the action IS the dense signal; reasoning content is carried by reasoning_rep below).
    sm = 0.04
    action_softdist = torch.full((len(ACTIONS),), sm, device=dev)
    action_softdist[action_idx] = 1.0 - sm * (len(ACTIONS) - 1)
    action_softdist = action_softdist.detach()
    # dense reasoning representation: read-layer hiddens averaged over the reasoning span
    ho = model(out[0].unsqueeze(0), output_hidden_states=True)
    if SDL_GEN_EPS >= 0:
        for L in FIELD_LAYERS: _fb['fields'][L].eps = _eps_saved[L]
    h = ho.hidden_states[READ_LAYER][0]                            # [T_full, D_MODEL]
    gspan = h[ids.shape[1]: ids.shape[1] + gen_ids.shape[0]]       # reasoning tokens
    reasoning_rep = (gspan.float().mean(0).detach() if gspan.shape[0] > 0
                     else h[-1].float().detach())                  # [D_MODEL]
    _fb['on'] = False
    return reasoning_rep, action_softdist, action_idx, reasoning_text


class ReqStudent(nn.Module):
    """Reads S → requirement-logits[6] + align projection to D_MODEL (absorbs reasoning rep)."""
    def __init__(s, d_s, d_model):
        super().__init__()
        s.attn = nn.Linear(d_s, 1)
        s.body = nn.Sequential(nn.Linear(d_s, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
        s.head = nn.Linear(256, 6)
        s.align = nn.Linear(256, d_model)

    def encode(s, S):
        w = torch.softmax(s.attn(S), 0)
        h = (w * S).sum(0)
        return s.body(h)               # z[256] — the substrate-derived reasoning encoding

    def forward(s, S):
        z = s.encode(S)
        return s.head(z), s.align(z)   # (logits[6], align_rep[D_MODEL])


def self_distill_loop():
    import copy
    print('=== SDL self_distill_loop | iters=%d T=%.2f maxnew=%d lr=%.1e anchor=%.2f align=%.2f ==='
          % (SDL_ITERS, SDL_T, SDL_MAXNEW, SDL_LR, SDL_ANCHOR, SDL_ALIGN), flush=True)

    rng = random.Random(SEED)
    all_worlds = {}
    for fam in FAMILIES:
        wl = []
        for i in range(N_WORLDS_PER_FAMILY):
            wl.append(make_world(fam, rng, i % N_TRAIN_TEMPLATES, i % 4))
        all_worlds[fam] = wl

    rcache = '%s/rollouts_wps%d_full_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if not os.path.exists(rcache):
        print('SDL_NO_CACHE %s' % rcache, flush=True); return
    print('SDL: loading cached rollouts %s' % rcache, flush=True)
    pop_data = torch.load(rcache, weights_only=False)

    # frozen substrate g (+ fields) — mirror viability_model's load
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    vmck = '%s/wps_trained_full_s%d_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if os.path.exists(vmck):
        print('SDL: loading g+fields from %s' % vmck, flush=True)
        sd = torch.load(vmck, weights_only=False)
        g.load_state_dict(sd['g'])
        for L in FIELD_LAYERS:
            if str(L) in sd.get('fields', {}):
                _fb['fields'][L].load_state_dict(sd['fields'][str(L)])
    if SUBSTRATE_CKPT and os.path.exists(SUBSTRATE_CKPT):
        esd = torch.load(SUBSTRATE_CKPT, map_location=dev, weights_only=False)
        gsd = esd['g'] if (isinstance(esd, dict) and 'g' in esd) else esd
        miss, unexp = g.load_state_dict(gsd, strict=False)
        print('SDL: SUBSTRATE_CKPT loaded from %s | missing=%s unexpected=%s'
              % (SUBSTRATE_CKPT, list(miss), list(unexp)), flush=True)
    g.eval()
    for p in g.parameters(): p.requires_grad_(False)
    for L in FIELD_LAYERS:
        _fb['fields'][L].eval()
        for p in _fb['fields'][L].parameters(): p.requires_grad_(False)
    # CONTROLLED UNFREEZE: keep a frozen esv4e reference for S-space drift control; optionally
    # train g so S learns to encode the per-decision requirement (the readout bottleneck).
    g_anchor = copy.deepcopy(g)
    for p in g_anchor.parameters(): p.requires_grad_(False)
    g_anchor.eval()
    if SDL_TRAIN_G:
        for p in g.parameters(): p.requires_grad_(True)   # g.eval() mode (no dropout) but trainable
        print('SDL: CONTROLLED-UNFREEZE g | g_lr=%.1e g_anchor=%.2f' % (SDL_G_LR, SDL_G_ANCHOR), flush=True)

    EXPLORE_FAMS = [f for f in TRAIN_FAMILIES if f in pop_data]
    HOLD_FAMS    = [f for f in HOLDOUT_FAMILIES if f in pop_data]
    print('SDL: explore_fams=%s holdout_fams=%s' % (EXPLORE_FAMS, HOLD_FAMS), flush=True)

    student = ReqStudent(D_S, D_MODEL).to(dev)
    student_init = copy.deepcopy(student)
    for p in student_init.parameters(): p.requires_grad_(False)
    pgroups = [{'params': list(student.parameters()), 'lr': SDL_LR}]
    if SDL_TRAIN_G:
        pgroups.append({'params': [p for p in g.parameters() if p.requires_grad], 'lr': SDL_G_LR})
    opt = torch.optim.Adam(pgroups)

    EXPLORE_ITEMS = [(fam, wi, hids, dec, w)
                     for fam in EXPLORE_FAMS
                     for wi, (hids, dec, w) in enumerate(pop_data[fam])]
    if not EXPLORE_ITEMS:
        print('SDL_NO_EXPLORE_ITEMS', flush=True); return

    running_correct = []

    def eval_req_acc(fams):
        student.eval(); c = 0; n = 0
        with torch.no_grad():
            for fam in fams:
                for (hids, dec, w) in pop_data[fam]:
                    for ti, dec_idx, qtext, correct in dec:
                        S = buildS_at(g, hids, ti).float()
                        lg, _ = student(S)
                        c += int(int(lg.argmax()) == ACTIONS.index(correct)); n += 1
        student.train(); return c / max(1, n)

    print('SDL eval@0 explore_acc=%.3f holdout_acc=%.3f'
          % (eval_req_acc(EXPLORE_FAMS), eval_req_acc(HOLD_FAMS)), flush=True)

    for it in range(1, SDL_ITERS + 1):
        fam, wi, hids, dec, w = EXPLORE_ITEMS[rng.randrange(len(EXPLORE_ITEMS))]   # EXPLORE
        ti, dec_idx, qtext, correct = dec[rng.randrange(len(dec))]
        with torch.no_grad():
            S_field = buildS_at(g, hids, ti).float()       # detached S for the teacher field
        commit_ctx = (w['turns'][0][1] if w.get('turns') and w['turns'][0][0] == 'commit' else '')
        rr, asd, a_idx, rtxt = gen_reasoning(qtext, S_field, SDL_T, SDL_MAXNEW, ctx=commit_ctx)   # teacher gets the commitment
        correct_idx = ACTIONS.index(correct)
        is_corr = int(a_idx == correct_idx); running_correct.append(is_corr)
        if it <= SDL_DEBUG_N:
            print('\n===SDL_DEBUG it=%d fam=%s correct=%s parsed=%s match=%d\n'
                  '  COMMIT: %s\n  QTEXT: %s\n  REASONING: %s\n===' % (
                  it, fam, correct, ACTIONS[a_idx], is_corr,
                  commit_ctx[:300].replace('\n', ' '), qtext[:300].replace('\n', ' '),
                  rtxt[:1200].replace('\n', ' | ')), flush=True)
        weight = 1.0 if is_corr else 0.3     # distill Qwen's CORRECT reasoning state more strongly

        # ── DISTILL QWEN'S REASONING STATE INTO THE SUBSTRATE STATE (the objective) ──
        # S is differentiable iff SDL_TRAIN_G, so matching align(S) → reasoning_rep reshapes g:
        # the substrate learns to REPRODUCE Qwen's family-general reasoning representation, not
        # a per-family action label. The action is only a downstream PROBE of the enriched state.
        S = buildS_at(g, hids, ti).float() if SDL_TRAIN_G else S_field
        z = student.encode(S)
        al = student.align(z)
        L_state = 1 - torch.nn.functional.cosine_similarity(al.unsqueeze(0), rr.unsqueeze(0)).clamp(-1, 1).squeeze()
        # action probe reads a DETACHED encoding → measures decodability/transfer without shaping g
        lg = student.head(z.detach())
        L_probe = torch.nn.functional.cross_entropy(lg.unsqueeze(0), torch.tensor([a_idx], device=dev))
        # substrate drift control: keep S near the frozen esv4e reference
        if SDL_TRAIN_G:
            with torch.no_grad():
                S_ref = buildS_at(g_anchor, hids, ti).float()
            L_gdrift = ((S - S_ref) ** 2).mean()
        else:
            L_gdrift = torch.zeros((), device=dev)
        L = weight * SDL_STATE_W * L_state + L_probe + SDL_G_ANCHOR * L_gdrift
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_([p for grp in pgroups for p in grp['params']], 1.0); opt.step()

        if it % 10 == 0:
            ra = sum(running_correct[-50:]) / max(1, len(running_correct[-50:]))
            print('SDL it=%d L=%.4f state=%.4f probe=%.4f gdrift=%.4f qwen_reason_acc(last50)=%.3f'
                  % (it, float(L), float(L_state), float(L_probe), float(L_gdrift), ra), flush=True)
        if it % SDL_EVAL_EVERY == 0:
            ea = eval_req_acc(EXPLORE_FAMS); ha = eval_req_acc(HOLD_FAMS)
            print('SDL eval@%d student_req_acc explore=%.3f holdout=%.3f  (DRIFT-MON: explore must not regress)'
                  % (it, ea, ha), flush=True)

    torch.save(student.state_dict(), '%s/sdl_student_s%d.pt' % (CKDIR, SEED))
    print('SDL FINAL explore_acc=%.3f holdout_acc=%.3f qwen_reason_acc=%.3f'
          % (eval_req_acc(EXPLORE_FAMS), eval_req_acc(HOLD_FAMS),
             sum(running_correct) / max(1, len(running_correct))), flush=True)
    print('=== SDL_DONE ===')
    print('=== SDL_END ===')


def sdl_probe_diag():
    """OFFLINE diagnostic: does Qwen's reasoning representation linearly encode the FAMILY-GENERAL
    correct action, and at which layer/position? Collects candidate reps per decision (multiple
    layers × {mean,last} over the CoT span, plus pooled frozen S), trains proper multi-epoch probes
    on EXPLORE families, tests TRANSFER to HELD-OUT families. Separates 'rep lacks info' from
    'online single-sample probe didn't converge' (the loop's stuck-0.308 ambiguity)."""
    import copy
    DIAG_LAYERS = [int(x) for x in os.environ.get('SDL_DIAG_LAYERS', '32,48,60').split(',')]
    DIAG_NW = int(os.environ.get('SDL_DIAG_NW', str(N_WORLDS_PER_FAMILY)))
    print('=== SDL_PROBE_DIAG | layers=%s nw=%d maxnew=%d ===' % (DIAG_LAYERS, DIAG_NW, SDL_MAXNEW), flush=True)

    rng = random.Random(SEED)
    rcache = '%s/rollouts_wps%d_full_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if not os.path.exists(rcache):
        print('SDL_NO_CACHE %s' % rcache, flush=True); return
    pop_data = torch.load(rcache, weights_only=False)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    vmck = '%s/wps_trained_full_s%d_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if os.path.exists(vmck):
        sd = torch.load(vmck, weights_only=False); g.load_state_dict(sd['g'])
    if SUBSTRATE_CKPT and os.path.exists(SUBSTRATE_CKPT):
        esd = torch.load(SUBSTRATE_CKPT, map_location=dev, weights_only=False)
        gsd = esd['g'] if (isinstance(esd, dict) and 'g' in esd) else esd
        g.load_state_dict(gsd, strict=False)
    g.eval()
    for p in g.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def capture(qtext, S, ctx):
        prompt = ((ctx + '\n\n') if ctx else '') + qtext + (
            '\n\nThink step by step about which single action is correct, then on the final line '
            'write exactly:\nACTION: <one of KEEP, REJECT, REPAIR, RELEASE, DEFER, ASK>')
        ids = tok(H.tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').input_ids.to(dev)
        _fb['on'] = False                                          # FIELD OFF (clean teacher)
        out = model.generate(ids, max_new_tokens=SDL_MAXNEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)        # greedy = deterministic
        gen_ids = out[0, ids.shape[1]:]
        rtxt = tok.decode(gen_ids, skip_special_tokens=True)
        ho = model(out[0].unsqueeze(0), output_hidden_states=True)
        g0, g1 = ids.shape[1], ids.shape[1] + gen_ids.shape[0]
        reps = {}
        for L in DIAG_LAYERS:
            h = ho.hidden_states[L][0]; span = h[g0:g1]
            reps['L%d_mean' % L] = (span.float().mean(0).cpu() if span.shape[0] > 0 else h[-1].float().cpu())
            reps['L%d_last' % L] = h[g1 - 1].float().cpu()
        reps['poolS'] = S.float().mean(0).cpu()
        return reps, rtxt

    rt_up_action = lambda rt: next((ai for ai in range(len(ACTIONS))
                                    if ACTIONS[ai] in rt.upper().split('ACTION:')[-1]), None) \
        if 'ACTION:' in rt.upper() else None

    rows = []   # (reps, correct_idx, is_explore)
    qok = 0; qn = 0
    for fam in FAMILIES:
        is_exp = fam in TRAIN_FAMILIES
        for (hids, dec, w) in pop_data[fam][:DIAG_NW]:
            ctx = (w['turns'][0][1] if w.get('turns') and w['turns'][0][0] == 'commit' else '')
            for ti, dec_idx, qtext, correct in dec:
                S = buildS_at(g, hids, ti).float()
                reps, rtxt = capture(qtext, S, ctx)
                cidx = ACTIONS.index(correct)
                pa = rt_up_action(rtxt)
                qok += int(pa == cidx); qn += 1
                rows.append((reps, cidx, is_exp))
        print('SDL_DIAG collected fam=%s (running qwen_acc=%.3f n=%d)' % (fam, qok / max(1, qn), qn), flush=True)

    variants = [k for k in rows[0][0].keys()]
    ytr = torch.tensor([c for (_, c, e) in rows if e])
    yho = torch.tensor([c for (_, c, e) in rows if not e])
    print('SDL_DIAG totals: explore_n=%d holdout_n=%d qwen_acc=%.3f' % (len(ytr), len(yho), qok / max(1, qn)), flush=True)
    print('SDL_DIAG base-rate explore=%.3f holdout=%.3f'
          % (float((ytr == ytr.mode().values).float().mean()),
             float((yho == ytr.mode().values).float().mean())), flush=True)

    def train_probe(Xtr, ytr, Xho, yho, mlp, epochs=600, wd=1e-3):
        d = Xtr.shape[1]
        net = (nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 6))
               if mlp else nn.Linear(d, 6)).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
        Xtr, ytr2, Xho, yho2 = Xtr.to(dev), ytr.to(dev), Xho.to(dev), yho.to(dev)
        net.train()
        for e in range(epochs):
            opt.zero_grad(); loss = F.cross_entropy(net(Xtr), ytr2); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            fit = float((net(Xtr).argmax(1) == ytr2).float().mean())
            tr = float((net(Xho).argmax(1) == yho2).float().mean())
        return fit, tr

    print('\nSDL_DIAG  variant         lin_fit lin_holdout   mlp_fit mlp_holdout', flush=True)
    for v in variants:
        Xtr = torch.stack([r[v] for (r, c, e) in rows if e])
        Xho = torch.stack([r[v] for (r, c, e) in rows if not e])
        mu, sdv = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
        Xtr_n, Xho_n = (Xtr - mu) / sdv, (Xho - mu) / sdv
        lf, lh = train_probe(Xtr_n, ytr, Xho_n, yho, mlp=False)
        mf, mh = train_probe(Xtr_n, ytr, Xho_n, yho, mlp=True)
        print('SDL_DIAG  %-14s   %.3f    %.3f        %.3f    %.3f' % (v, lf, lh, mf, mh), flush=True)
    print('=== SDL_DIAG_DONE ===', flush=True)


def _offline_probe(Z, y, exp_mask, epochs=600, wd=1e-3, mlp=False):
    """Train a probe on EXPLORE rows (exp_mask True), return (explore_fit, holdout_transfer).
    Standardize with explore stats. Z:[N,d] cpu, y:[N] cpu, exp_mask:[N] bool."""
    Xtr, ytr = Z[exp_mask], y[exp_mask]
    Xho, yho = Z[~exp_mask], y[~exp_mask]
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xho = ((Xtr - mu) / sd).to(dev), ((Xho - mu) / sd).to(dev)
    ytr, yho = ytr.to(dev), yho.to(dev)
    d = Xtr.shape[1]
    net = (nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 6))
           if mlp else nn.Linear(d, 6)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    net.train()
    for e in range(epochs):
        opt.zero_grad(); F.cross_entropy(net(Xtr), ytr).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        fit = float((net(Xtr).argmax(1) == ytr).float().mean())
        tr = float((net(Xho).argmax(1) == yho).float().mean())
    return fit, tr


def sdl_state_v2():
    """DISTILL QWEN'S REASONING REP INTO THE SUBSTRATE STATE, measured by a PROPER offline probe.
    Pre-caches Qwen reps once (fast re-runnable loop). Trains g (controlled) so align(encode(S))→rr;
    tests whether the distilled state's action-transfer rises from frozen-S (~0.35) toward rr (~0.58)."""
    import copy
    RR_LAYER = int(os.environ.get('SDL_RR_LAYER', '48'))
    RR_POS = os.environ.get('SDL_RR_POS', 'last')
    ITERS = int(os.environ.get('SDL_ITERS', '800'))
    EVERY = int(os.environ.get('SDL_EVAL_EVERY', '100'))
    print('=== SDL_STATE_V2 | rr=L%d_%s iters=%d g_lr=%.1e g_anchor=%.2f ==='
          % (RR_LAYER, RR_POS, ITERS, SDL_G_LR, SDL_G_ANCHOR), flush=True)

    rng = random.Random(SEED)
    rcache = '%s/rollouts_wps%d_full_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if not os.path.exists(rcache):
        print('SDL_NO_CACHE %s' % rcache, flush=True); return
    pop_data = torch.load(rcache, weights_only=False)

    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    vmck = '%s/wps_trained_full_s%d_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if os.path.exists(vmck):
        g.load_state_dict(torch.load(vmck, weights_only=False)['g'])
    if SUBSTRATE_CKPT and os.path.exists(SUBSTRATE_CKPT):
        esd = torch.load(SUBSTRATE_CKPT, map_location=dev, weights_only=False)
        g.load_state_dict(esd['g'] if (isinstance(esd, dict) and 'g' in esd) else esd, strict=False)
    g_anchor = copy.deepcopy(g)
    for p in g_anchor.parameters(): p.requires_grad_(False)
    g_anchor.eval()

    # canonical decision list
    DEC = []   # (hids, ti, correct_idx, is_explore, ctx, qtext)
    for fam in FAMILIES:
        is_exp = fam in TRAIN_FAMILIES
        for (hids, dec, w) in pop_data[fam]:
            ctx = (w['turns'][0][1] if w.get('turns') and w['turns'][0][0] == 'commit' else '')
            for ti, dec_idx, qtext, correct in dec:
                DEC.append((hids, ti, ACTIONS.index(correct), is_exp, ctx, qtext))

    # ── pre-cache Qwen reps rr (greedy, field-off) ──
    rrk = '%s/rrcache_s%d_L%d_%s_nw%d.pt' % (CKDIR, SEED, RR_LAYER, RR_POS, N_WORLDS_PER_FAMILY)
    if os.path.exists(rrk):
        print('SDL_V2: loading rr cache %s' % rrk, flush=True)
        RR = torch.load(rrk, weights_only=False)
    else:
        print('SDL_V2: collecting rr cache (%d decisions, greedy field-off)...' % len(DEC), flush=True)
        RR = []
        for di, (hids, ti, cidx, is_exp, ctx, qtext) in enumerate(DEC):
            prompt = ((ctx + '\n\n') if ctx else '') + qtext + (
                '\n\nThink step by step about which single action is correct, then on the final line '
                'write exactly:\nACTION: <one of KEEP, REJECT, REPAIR, RELEASE, DEFER, ASK>')
            ids = tok(H.tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').input_ids.to(dev)
            _fb['on'] = False
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=SDL_MAXNEW, do_sample=False, pad_token_id=tok.eos_token_id)
                ho = model(out[0].unsqueeze(0), output_hidden_states=True)
            g0, g1 = ids.shape[1], out.shape[1]
            h = ho.hidden_states[RR_LAYER][0]
            rr = (h[g0:g1].float().mean(0) if RR_POS == 'mean' else h[g1 - 1].float()).cpu()
            RR.append(rr)
            if (di + 1) % 24 == 0: print('SDL_V2 rr %d/%d' % (di + 1, len(DEC)), flush=True)
        RR = torch.stack(RR)
        torch.save(RR, rrk)
    RR = RR.to(dev)

    y = torch.tensor([c for (_, _, c, _, _, _) in DEC])
    exp_mask = torch.tensor([e for (_, _, _, e, _, _) in DEC])

    student = ReqStudent(D_S, D_MODEL).to(dev)
    pgroups = [{'params': list(student.parameters()), 'lr': SDL_LR}]
    for p in g.parameters(): p.requires_grad_(True)
    pgroups.append({'params': [p for p in g.parameters()], 'lr': SDL_G_LR})
    opt = torch.optim.Adam(pgroups)

    # ── BASELINES (frozen substrate + rr ceiling), measured by the same offline probe ──
    with torch.no_grad():
        poolS = torch.stack([buildS_at(g_anchor, hids, ti).float().mean(0).cpu() for (hids, ti, *_) in DEC])
    bf, bt = _offline_probe(poolS, y, exp_mask)
    rf, rt = _offline_probe(RR.cpu(), y, exp_mask)
    print('SDL_V2 BASELINE poolS(frozen) fit=%.3f holdout=%.3f | rr(ceiling) fit=%.3f holdout=%.3f'
          % (bf, bt, rf, rt), flush=True)

    def eval_distilled(tag, it):
        student.eval()
        with torch.no_grad():
            Sd = [buildS_at(g, hids, ti).float() for (hids, ti, *_) in DEC]
            Z = torch.stack([student.encode(s).cpu() for s in Sd])
            poolSd = torch.stack([s.mean(0).cpu() for s in Sd])
            al = torch.stack([student.align(student.encode(s)) for s in Sd])
            cos = float(F.cosine_similarity(al, RR, dim=1).mean())
        zf, zt = _offline_probe(Z, y, exp_mask)
        sf, st = _offline_probe(poolSd, y, exp_mask)
        print('SDL_V2 %s@%d distilled-z fit=%.3f HOLDOUT=%.3f | distilled-poolS fit=%.3f HOLDOUT=%.3f '
              '(frozen-poolS %.3f, rr-ceiling %.3f) | z↔rr cos=%.3f'
              % (tag, it, zf, zt, sf, st, bt, rt, cos), flush=True)
        student.train()
        return zt

    eval_distilled('eval', 0)
    EXP_IDX = [i for i, (_, _, _, e, _, _) in enumerate(DEC) if e]
    for it in range(1, ITERS + 1):
        di = EXP_IDX[rng.randrange(len(EXP_IDX))]
        hids, ti, cidx, _, ctx, qtext = DEC[di]
        S = buildS_at(g, hids, ti).float()
        al = student.align(student.encode(S))
        L_state = 1 - F.cosine_similarity(al.unsqueeze(0), RR[di].unsqueeze(0)).clamp(-1, 1).squeeze()
        with torch.no_grad():
            S_ref = buildS_at(g_anchor, hids, ti).float()
        L_gdrift = ((S - S_ref) ** 2).mean()
        L = L_state + SDL_G_ANCHOR * L_gdrift
        opt.zero_grad(); L.backward()
        torch.nn.utils.clip_grad_norm_([p for grp in pgroups for p in grp['params']], 1.0); opt.step()
        if it % EVERY == 0:
            eval_distilled('eval', it)
    torch.save({'g': g.state_dict(), 'student': student.state_dict()}, '%s/sdl_state_v2_s%d.pt' % (CKDIR, SEED))
    print('=== SDL_V2_DONE ===', flush=True)


def sdl_rollout_probe():
    """WHERE does the family-general requirement signal live? Probe RAW world-rollout hiddens
    (what the substrate READS — no Qwen generation, fast) vs the substrate state poolS. If raw
    rollout transfers ~0.6 (like generated reasoning rr), the requirement is in the world-processing
    and g compresses it away (fixable extraction). If ~0.35 (like poolS), it's ONLY in generated
    reasoning — the substrate reading the rollout fundamentally can't infer it without Qwen reasoning."""
    print('=== SDL_ROLLOUT_PROBE (raw rollout hiddens, no Qwen) ===', flush=True)
    rcache = '%s/rollouts_wps%d_full_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if not os.path.exists(rcache):
        print('SDL_NO_CACHE %s' % rcache, flush=True); return
    pop_data = torch.load(rcache, weights_only=False)
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    if SUBSTRATE_CKPT and os.path.exists(SUBSTRATE_CKPT):
        esd = torch.load(SUBSTRATE_CKPT, map_location=dev, weights_only=False)
        g.load_state_dict(esd['g'] if (isinstance(esd, dict) and 'g' in esd) else esd, strict=False)
    g.eval()

    rows = []
    for fam in FAMILIES:
        is_exp = fam in TRAIN_FAMILIES
        for (hids, dec, w) in pop_data[fam]:
            allh = torch.cat([h.float() for h in hids], 0)             # all rollout token-hiddens
            commit_h = hids[0].float()                                  # commit turn (rule stated here)
            for ti, dec_idx, qtext, correct in dec:
                ht = hids[ti].float()                                   # decision-turn window hiddens
                feats = {
                    'rollDec_last':  ht[-1].cpu(),
                    'rollDec_mean':  ht.mean(0).cpu(),
                    'rollFull_mean': allh.mean(0).cpu(),                # commit→decision integrated
                    'commit_mean':   commit_h.mean(0).cpu(),
                    'poolS':         buildS_at(g, hids, ti).float().mean(0).detach().cpu(),
                }
                rows.append((feats, ACTIONS.index(correct), is_exp))
    y = torch.tensor([c for (_, c, e) in rows])
    exp_mask = torch.tensor([e for (_, _, e) in rows])
    print('SDL_RP totals explore=%d holdout=%d base-rate holdout=%.3f'
          % (int(exp_mask.sum()), int((~exp_mask).sum()),
             float((y[~exp_mask] == y[exp_mask].mode().values).float().mean())), flush=True)
    print('SDL_RP  feature        lin_fit lin_holdout   mlp_fit mlp_holdout', flush=True)
    for v in rows[0][0].keys():
        Z = torch.stack([r[v] for (r, c, e) in rows])
        lf, lh = _offline_probe(Z, y, exp_mask, mlp=False)
        mf, mh = _offline_probe(Z, y, exp_mask, mlp=True)
        print('SDL_RP  %-13s   %.3f    %.3f        %.3f    %.3f' % (v, lf, lh, mf, mh), flush=True)
    print('=== SDL_RP_DONE ===', flush=True)


def sdl_reason_state():
    """REASONING DEFINES THE TRAJECTORY (user): build the substrate state FROM Qwen's CoT reasoning
    trajectory, not the world rollout. Feed g the per-token reasoning hiddens (read-layer = what g
    expects) so its state evolves ALONG the reasoning. Test (frozen g first): does poolS_reason
    transfer toward rr (0.58) vs world-rollout poolS_world (0.35)? Then optionally train g."""
    import copy
    ITERS = int(os.environ.get('SDL_ITERS', '0'))           # 0 = frozen test only
    print('=== SDL_REASON_STATE | read_layer=%d iters=%d ===' % (READ_LAYER, ITERS), flush=True)
    rcache = '%s/rollouts_wps%d_full_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    if not os.path.exists(rcache):
        print('SDL_NO_CACHE %s' % rcache, flush=True); return
    pop_data = torch.load(rcache, weights_only=False)
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    if SUBSTRATE_CKPT and os.path.exists(SUBSTRATE_CKPT):
        esd = torch.load(SUBSTRATE_CKPT, map_location=dev, weights_only=False)
        g.load_state_dict(esd['g'] if (isinstance(esd, dict) and 'g' in esd) else esd, strict=False)
    g.eval()
    g_anchor = copy.deepcopy(g)               # frozen esv4e: teacher reasoning-state + drift anchor
    for p in g_anchor.parameters(): p.requires_grad_(False)
    g_anchor.eval()

    DEC = []
    for fam in FAMILIES:
        is_exp = fam in TRAIN_FAMILIES
        for (hids, dec, w) in pop_data[fam]:
            ctx = (w['turns'][0][1] if w.get('turns') and w['turns'][0][0] == 'commit' else '')
            for ti, dec_idx, qtext, correct in dec:
                DEC.append((hids, ti, ACTIONS.index(correct), is_exp, ctx, qtext))

    # ── cache per-token reasoning trajectories at READ_LAYER (greedy, field-off) ──
    rtk = '%s/reastraj_s%d_L%d_nw%d.pt' % (CKDIR, SEED, READ_LAYER, N_WORLDS_PER_FAMILY)
    if os.path.exists(rtk):
        print('SDL_RS: loading reasoning-traj cache %s' % rtk, flush=True)
        TRAJ = torch.load(rtk, weights_only=False)
    else:
        print('SDL_RS: collecting reasoning trajectories (%d, greedy field-off)...' % len(DEC), flush=True)
        TRAJ = []
        for di, (hids, ti, cidx, is_exp, ctx, qtext) in enumerate(DEC):
            prompt = ((ctx + '\n\n') if ctx else '') + qtext + (
                '\n\nThink step by step about which single action is correct, then on the final line '
                'write exactly:\nACTION: <one of KEEP, REJECT, REPAIR, RELEASE, DEFER, ASK>')
            ids = tok(H.tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').input_ids.to(dev)
            _fb['on'] = False
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=SDL_MAXNEW, do_sample=False, pad_token_id=tok.eos_token_id)
                ho = model(out[0].unsqueeze(0), output_hidden_states=True)
            g0, g1 = ids.shape[1], out.shape[1]
            traj = ho.hidden_states[READ_LAYER][0, g0:g1, :].float().to(torch.float16).cpu()   # [T_reason, D]
            TRAJ.append(traj)
            if (di + 1) % 24 == 0: print('SDL_RS traj %d/%d' % (di + 1, len(DEC)), flush=True)
        torch.save(TRAJ, rtk)

    y = torch.tensor([c for (_, _, c, _, _, _) in DEC])
    exp_mask = torch.tensor([e for (_, _, _, e, _, _) in DEC])

    @torch.no_grad()
    def states():
        feat = {k: [] for k in ['Sworld_mean', 'Sreason_mean', 'Sronly_mean', 'Sronly_flat',
                                'Sronly_max', 'Sronly_slow', 'rr_mean', 'rr_last']}
        for (hids, ti, *_), traj in zip(DEC, TRAJ):
            Sw = buildS_at(g, hids, ti).float()
            th = traj.float().to(dev)
            Sr = g.step(Sw, th)                       # world-seeded, then reasoning
            Sro = g.step(g.init(), th)                # reasoning-only state  [K, d_s]
            feat['Sworld_mean'].append(Sw.mean(0).cpu())
            feat['Sreason_mean'].append(Sr.mean(0).cpu())
            feat['Sronly_mean'].append(Sro.mean(0).cpu())
            feat['Sronly_flat'].append(Sro.reshape(-1).cpu())            # all K slots
            feat['Sronly_max'].append(Sro.max(0).values.cpu())
            feat['Sronly_slow'].append(Sro[:SLOW_K].mean(0).cpu())       # slow slots only
            feat['rr_mean'].append(th.mean(0).cpu())
            feat['rr_last'].append(th[-1].cpu())
        return {k: torch.stack(v) for k, v in feat.items()}

    feat = states()
    print('SDL_RS  readout              lin_fit lin_HOLDOUT   mlp_fit mlp_HOLDOUT', flush=True)
    for tag in ['Sworld_mean', 'rr_mean', 'rr_last', 'Sreason_mean',
                'Sronly_mean', 'Sronly_slow', 'Sronly_max', 'Sronly_flat']:
        Z = feat[tag]
        lf, lh = _offline_probe(Z, y, exp_mask, mlp=False)
        mf, mh = _offline_probe(Z, y, exp_mask, mlp=True)
        print('SDL_RS  %-18s     %.3f    %.3f        %.3f    %.3f' % (tag, lf, lh, mf, mh), flush=True)

    if ITERS > 0:
        # ── SELF-DISTILL: train the WORLD-state to reproduce the (frozen) REASONING-state ──
        # teacher = Sronly via frozen esv4e (the reasoning-derived state, transfer 0.44);
        # student = Sworld via trained g (world rollout only, no Qwen at inference). The substrate
        # learns to INTERNALLY produce the reasoning-state from the world → autonomy. No action label.
        print('=== SDL_RS SELF-DISTILL world->reason | iters=%d g_lr=%.1e anchor=%.2f ===' %
              (ITERS, SDL_G_LR, SDL_G_ANCHOR), flush=True)
        with torch.no_grad():
            TEACH = [g_anchor.step(g_anchor.init(), t.float().to(dev)).mean(0).detach() for t in TRAJ]
        TEACH = torch.stack(TEACH)                                   # [N, d_s] frozen reasoning-state
        tmu = TEACH[exp_mask].mean(0, keepdim=True); tsd = TEACH[exp_mask].std(0, keepdim=True) + 1e-6
        for p in g.parameters(): p.requires_grad_(True)
        opt = torch.optim.Adam(g.parameters(), lr=SDL_G_LR)
        EXP = [i for i in range(len(DEC)) if bool(exp_mask[i])]
        rng2 = random.Random(SEED)

        def eval_world(it):
            with torch.no_grad():
                Z = torch.stack([buildS_at(g, DEC[i][0], DEC[i][1]).float().mean(0).cpu()
                                 for i in range(len(DEC))])
            lf, lh = _offline_probe(Z, y, exp_mask, mlp=False)
            print('SDL_RS  selfdistill@%d  Sworld lin_fit=%.3f HOLDOUT=%.3f (teacher Sronly 0.44, base 0.27)'
                  % (it, lf, lh), flush=True)
        for it in range(1, ITERS + 1):
            i = EXP[rng2.randrange(len(EXP))]
            hids, ti = DEC[i][0], DEC[i][1]
            Sw = buildS_at(g, hids, ti).float().mean(0)
            with torch.no_grad():
                Sw_ref = buildS_at(g_anchor, hids, ti).float().mean(0)
            tgt = (TEACH[i] - tmu.squeeze(0)) / tsd.squeeze(0)
            pred = (Sw - tmu.squeeze(0)) / tsd.squeeze(0)
            L_match = F.mse_loss(pred, tgt)                          # standardized → equal-weight all dims
            L_drift = ((Sw - Sw_ref) ** 2).mean()
            L = L_match + SDL_G_ANCHOR * L_drift
            opt.zero_grad(); L.backward()
            torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0); opt.step()
            if it % max(1, ITERS // 8) == 0:
                eval_world(it)
        torch.save({'g': g.state_dict()}, '%s/sdl_reason_world_s%d.pt' % (CKDIR, SEED))
    print('=== SDL_RS_DONE ===', flush=True)


def sdl_persist():
    """PERSISTENCE — the substrate's only candidate unique value. Does absorbing Qwen's reasoning
    EARLY (at decision1) carry forward to help the DISTANT later decision2 (reasoning out-of-window)?
    Splain = rollout to dec2 (baseline). Scarry = rollout to dec1 + absorb dec1 reasoning + continue
    to dec2 (does the early reasoning persist?). Sreason2 = dec2's OWN reasoning absorbed (upper bound).
    Win = Scarry HOLDOUT > Splain → the substrate MAINTAINS the reasoning-derived requirement."""
    print('=== SDL_PERSIST (does early reasoning persist to distant decision?) ===', flush=True)
    rcache = '%s/rollouts_wps%d_full_nw%d.pt' % (CKDIR, SEED, N_WORLDS_PER_FAMILY)
    rtk = '%s/reastraj_s%d_L%d_nw%d.pt' % (CKDIR, SEED, READ_LAYER, N_WORLDS_PER_FAMILY)
    if not (os.path.exists(rcache) and os.path.exists(rtk)):
        print('SDL_PERSIST missing cache(s)', flush=True); return
    pop_data = torch.load(rcache, weights_only=False)
    TRAJ = torch.load(rtk, weights_only=False)
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    if SUBSTRATE_CKPT and os.path.exists(SUBSTRATE_CKPT):
        esd = torch.load(SUBSTRATE_CKPT, map_location=dev, weights_only=False)
        g.load_state_dict(esd['g'] if (isinstance(esd, dict) and 'g' in esd) else esd, strict=False)
    g.eval()

    WORLDS = []; gi = 0
    for fam in FAMILIES:
        is_exp = fam in TRAIN_FAMILIES
        for (hids, dec, w) in pop_data[fam]:
            wd = []
            for ti, dec_idx, qtext, correct in dec:
                wd.append((gi, ti, dec_idx, ACTIONS.index(correct))); gi += 1
            WORLDS.append((is_exp, hids, wd))

    rows = []
    with torch.no_grad():
        for is_exp, hids, wd in WORLDS:
            d1 = [x for x in wd if x[2] == 1]; d2 = [x for x in wd if x[2] == 2]
            if not d1 or not d2: continue
            gi1, ti1, _, _ = d1[0]; gi2, ti2, _, c2 = d2[0]
            if ti2 <= ti1: continue
            traj1 = TRAJ[gi1].float().to(dev); traj2 = TRAJ[gi2].float().to(dev)
            Splain = buildS_at(g, hids, ti2).float()
            S = g.init()
            for h in hids[:ti1 + 1]: S = g.step(S, h.to(dev).float())
            S = g.step(S, traj1)                                  # absorb dec1 reasoning EARLY
            for h in hids[ti1 + 1:ti2 + 1]: S = g.step(S, h.to(dev).float())
            Scarry = S.float()
            Sreason2 = g.step(buildS_at(g, hids, ti2).float(), traj2).float()   # dec2 own reasoning (ceiling)
            rows.append(({'Splain': Splain.mean(0).cpu(), 'Scarry': Scarry.mean(0).cpu(),
                          'Sreason2': Sreason2.mean(0).cpu()}, c2, is_exp))

    y = torch.tensor([c for (_, c, e) in rows]); exp_mask = torch.tensor([e for (_, _, e) in rows])
    print('SDL_PERSIST n_worlds=%d explore=%d holdout=%d base-rate-holdout=%.3f'
          % (len(rows), int(exp_mask.sum()), int((~exp_mask).sum()),
             float((y[~exp_mask] == y[exp_mask].mode().values).float().mean())), flush=True)
    print('SDL_PERSIST  state(at dec2)   lin_fit lin_HOLDOUT   mlp_fit mlp_HOLDOUT', flush=True)
    for tag in ['Splain', 'Scarry', 'Sreason2']:
        Z = torch.stack([r[tag] for (r, c, e) in rows])
        lf, lh = _offline_probe(Z, y, exp_mask, mlp=False)
        mf, mh = _offline_probe(Z, y, exp_mask, mlp=True)
        print('SDL_PERSIST  %-14s   %.3f    %.3f        %.3f    %.3f' % (tag, lf, lh, mf, mh), flush=True)
    print('=== SDL_PERSIST_DONE ===', flush=True)


def sdl_actuate_train():
    """TRAIN THE ACTUATOR (carry+actuate, out-of-window). g FROZEN (memory holds the rule, decay fit 1.0);
    train the FIELD so field-ON surfaces S's remembered rule into the LLM's action at d=8 where field-OFF
    (commit evicted) fails. Teacher-forced NLL on the correct action word at gentle eps; eval by real
    generation field-ON vs field-OFF on HELD-OUT worlds (in-family) and HOLDOUT families (cross-family)."""
    import copy
    ACT_EPS = float(os.environ.get('SDL_ACT_EPS_TRAIN', '0.1'))
    ACT_ITERS = int(os.environ.get('SDL_ACT_ITERS', '400'))
    ACT_LR = float(os.environ.get('SDL_ACT_LR', '1e-4'))
    D = int(os.environ.get('SDL_ACT_D', '8'))
    NW = N_WORLDS_PER_FAMILY
    NTRAINW = int(os.environ.get('SDL_ACT_NTRAINW', '9'))      # per explore family; rest = in-family held-out
    EVAL_EVERY = int(os.environ.get('SDL_ACT_EVAL_EVERY', '100'))
    EVAL_ONLY = int(os.environ.get('SDL_ACT_EVAL_ONLY', '0'))  # load saved actuator, per-family eval, no train
    print('=== SDL_ACTUATE_TRAIN | D=%d eps=%.2f iters=%d lr=%.1e ntrainw=%d ==='
          % (D, ACT_EPS, ACT_ITERS, ACT_LR, NTRAINW), flush=True)
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=ACT_EPS).to(dev) for L in FIELD_LAYERS}
    if SUBSTRATE_CKPT and os.path.exists(SUBSTRATE_CKPT):
        esd = torch.load(SUBSTRATE_CKPT, map_location=dev, weights_only=False)
        g.load_state_dict(esd['g'] if (isinstance(esd, dict) and 'g' in esd) else esd, strict=False)
    g.eval()
    for p in g.parameters(): p.requires_grad_(False)                  # FROZEN memory
    fparams = []
    for L in FIELD_LAYERS:
        _fb['fields'][L].train()
        for p in _fb['fields'][L].parameters(): p.requires_grad_(True); fparams.append(p)
    opt = torch.optim.Adam(fparams, lr=ACT_LR)
    rng = random.Random(SEED)
    all_worlds = {fam: [make_world(fam, rng, i % N_TRAIN_TEMPLATES, i % 4) for i in range(NW)] for fam in FAMILIES}
    FILL_U, FILL_A = 'Routine status check — no action needed.', 'Acknowledged. Standing by.'

    @torch.no_grad()
    def read_hidden(hist):
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        return model(ids, output_hidden_states=True).hidden_states[READ_LAYER][0].float()

    # precompute per-world: S (field-off accumulation incl commit), decision prompt ids, action tokens
    samples = []
    _fb['on'] = False
    with torch.no_grad():
        for fam in FAMILIES:
            is_exp = fam in TRAIN_FAMILIES
            for wi, w in enumerate(all_worlds[fam]):
                dq = [t for t in w['turns'] if t[0] == 'decision1']
                if not dq: continue
                dec_q, correct = dq[0][1], w['correct1']
                hist = [{'role': 'user', 'content': w['turns'][0][1]}, {'role': 'assistant', 'content': 'Acknowledged.'}]
                S = g.step(g.init(), read_hidden(hist).to(dev))
                for _ in range(D):
                    hist += [{'role': 'user', 'content': FILL_U}, {'role': 'assistant', 'content': FILL_A}]
                    S = g.step(S, read_hidden(hist).to(dev))
                hist += [{'role': 'user', 'content': dec_q}]
                pids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids[0].to(dev)
                aids = tok(' ' + correct, add_special_tokens=False).input_ids
                samples.append({'S': S.detach(), 'pids': pids, 'aids': aids, 'tw': wi < NTRAINW,
                                'exp': is_exp, 'cidx': ACTIONS.index(correct), 'fam': fam})
    TRAIN = [s for s in samples if s['tw'] and s['exp']]
    HELD = [s for s in samples if (not s['tw']) and s['exp']]           # in-family held-out worlds
    HOLD = [s for s in samples if not s['exp']]                          # cross-family
    print('SDL_ACTTR samples train=%d in-fam-held=%d cross-fam=%d' % (len(TRAIN), len(HELD), len(HOLD)), flush=True)

    @torch.no_grad()
    def gen_acc(group, field_on):
        if not group: return 0.0
        c = 0
        for s in group:
            if field_on: _fb['S'] = s['S']; _fb['on'] = True; [setattr(_fb['fields'][L], 'eps', ACT_EPS) for L in FIELD_LAYERS]
            else: _fb['on'] = False
            out = model.generate(s['pids'].unsqueeze(0), max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
            _fb['on'] = False
            txt = tok.decode(out[0, s['pids'].shape[0]:], skip_special_tokens=True).upper()
            ai = next((j for j, a in enumerate(ACTIONS) if a in txt), -1)
            c += int(ai == s['cidx'])
        return c / len(group)

    def per_family():
        for L in FIELD_LAYERS: _fb['fields'][L].eval()
        fams = []
        for s in samples:
            if s['fam'] not in fams: fams.append(s['fam'])
        print('SDL_ACTTR  PER-FAMILY (d=%d)  off    ON    n   role' % D, flush=True)
        for fam in fams:
            famall = [s for s in samples if s['fam'] == fam]
            is_cross = not famall[0]['exp']
            grp = famall if is_cross else [s for s in famall if not s['tw']]   # cross: ALL worlds; in-fam: held-out
            if not grp: continue
            print('SDL_ACTTR  %-4s  %-6s  %.3f  %.3f  %d' % (fam, 'CROSS' if is_cross else 'in-fam',
                  gen_acc(grp, False), gen_acc(grp, True), len(grp)), flush=True)

    if EVAL_ONLY:
        ckpt = '%s/sdl_actuator_s%d.pt' % (CKDIR, SEED)
        sd = torch.load(ckpt, map_location=dev, weights_only=False)
        for L in FIELD_LAYERS: _fb['fields'][L].load_state_dict(sd[L])
        print('SDL_ACTTR EVAL_ONLY: loaded %s' % ckpt, flush=True)
        per_family()
        print('=== SDL_ACTTR_DONE ===', flush=True); return

    def report(it):
        for L in FIELD_LAYERS: _fb['fields'][L].eval()
        off_h = gen_acc(HELD, False); on_h = gen_acc(HELD, True)
        off_x = gen_acc(HOLD, False); on_x = gen_acc(HOLD, True)
        print('SDL_ACTTR eval@%d  in-fam-held off=%.3f ON=%.3f | cross-fam off=%.3f ON=%.3f'
              % (it, off_h, on_h, off_x, on_x), flush=True)
        for L in FIELD_LAYERS: _fb['fields'][L].train()
        return on_h, on_x

    @torch.no_grad()
    def audit(group, label, n=5):
        for L in FIELD_LAYERS: _fb['fields'][L].eval()
        for s in group[:n]:
            _fb['on'] = False
            o0 = model.generate(s['pids'].unsqueeze(0), max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
            t0 = tok.decode(o0[0, s['pids'].shape[0]:], skip_special_tokens=True)
            _fb['S'] = s['S']; _fb['on'] = True
            for L in FIELD_LAYERS: _fb['fields'][L].eps = ACT_EPS
            o1 = model.generate(s['pids'].unsqueeze(0), max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
            _fb['on'] = False
            t1 = tok.decode(o1[0, s['pids'].shape[0]:], skip_special_tokens=True)
            print('SDL_ACTTR AUDIT[%s] fam=%s correct=%s | OFF=%r | ON=%r'
                  % (label, s['fam'], ACTIONS[s['cidx']], t0[:36].replace(chr(10), ' '),
                     t1[:36].replace(chr(10), ' ')), flush=True)

    BATCH = int(os.environ.get('SDL_ACT_BATCH', '4'))           # grad accumulation → stabilize
    best = {'score': -1.0, 'sd': None, 'it': 0, 'oh': 0.0, 'ox': 0.0}
    report(0)
    for it in range(1, ACT_ITERS + 1):
        opt.zero_grad(); acc_nll = 0.0
        for _b in range(BATCH):
            s = TRAIN[rng.randrange(len(TRAIN))]
            aids = s['aids']; seq = torch.cat([s['pids'], torch.tensor(aids, device=dev)]).unsqueeze(0)
            _fb['S'] = s['S']; _fb['on'] = True
            logits = model(seq).logits[0]
            _fb['on'] = False
            pl = s['pids'].shape[0]
            lp = torch.log_softmax(logits[pl - 1:pl - 1 + len(aids)], -1)
            nll = -lp[range(len(aids)), torch.tensor(aids, device=dev)].mean()
            (nll / BATCH).backward(); acc_nll += float(nll) / BATCH
        torch.nn.utils.clip_grad_norm_(fparams, 1.0); opt.step()
        if it % EVAL_EVERY == 0:
            print('SDL_ACTTR it=%d nll=%.4f' % (it, acc_nll), flush=True)
            oh, ox = report(it)
            score = oh + ox                                      # save-best by in-fam + cross-fam ON
            if score > best['score']:
                best = {'score': score, 'sd': {L: copy.deepcopy(_fb['fields'][L].state_dict()) for L in FIELD_LAYERS},
                        'it': it, 'oh': oh, 'ox': ox}
                print('SDL_ACTTR new BEST @%d in-fam=%.3f cross-fam=%.3f' % (it, oh, ox), flush=True)
    if best['sd'] is not None:                                   # restore best for audit + save
        for L in FIELD_LAYERS: _fb['fields'][L].load_state_dict(best['sd'][L])
    print('SDL_ACTTR BEST @%d in-fam-held ON=%.3f cross-fam ON=%.3f' % (best['it'], best['oh'], best['ox']), flush=True)
    per_family()
    audit(HELD, 'in-fam-held'); audit(HOLD, 'cross-fam')
    torch.save({L: _fb['fields'][L].state_dict() for L in FIELD_LAYERS}, '%s/sdl_actuator_s%d.pt' % (CKDIR, SEED))
    print('=== SDL_ACTTR_DONE ===', flush=True)


def sdl_decay_actuate():
    """FUNCTIONAL memory test (carry+ACTUATE, out-of-window). COMMIT(rule)->D filler turns->decision,
    commit EVICTED from window. Generate the 1-word action: field OFF (LLM alone, forgot rule) vs
    field ON at several eps (substrate carries+surfaces the rule). Win = field-on acc >> field-off."""
    EPS_SWEEP = [float(x) for x in os.environ.get('SDL_ACT_EPS', '0.05,0.1,0.3,1.0').split(',')]
    NW = int(os.environ.get('SDL_DECAY_NW', '8'))
    D = int(os.environ.get('SDL_ACT_D', '8'))
    print('=== SDL_DECAY_ACTUATE | D=%d (commit evicted) eps_sweep=%s nw=%d ===' % (D, EPS_SWEEP, NW), flush=True)
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    if SUBSTRATE_CKPT and os.path.exists(SUBSTRATE_CKPT):
        esd = torch.load(SUBSTRATE_CKPT, map_location=dev, weights_only=False)
        g.load_state_dict(esd['g'] if (isinstance(esd, dict) and 'g' in esd) else esd, strict=False)
    g.eval()
    rng = random.Random(SEED)
    all_worlds = {fam: [make_world(fam, rng, i % N_TRAIN_TEMPLATES, i % 4) for i in range(NW)] for fam in FAMILIES}
    FILL_U, FILL_A = 'Routine status check — no action needed.', 'Acknowledged. Standing by.'

    @torch.no_grad()
    def read_hidden(hist):
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        return model(ids, output_hidden_states=True).hidden_states[READ_LAYER][0].float()

    @torch.no_grad()
    def gen_action(hist, S, eps):
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        if eps is None:
            _fb['on'] = False
        else:
            for L in FIELD_LAYERS: _fb['fields'][L].eps = eps
            _fb['S'] = S; _fb['on'] = True
        out = model.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
        _fb['on'] = False
        txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).upper()
        for ai, a in enumerate(ACTIONS):
            if a in txt: return ai, txt
        return -1, txt

    arms = [('field_off', None)] + [('eps%.2f' % e, e) for e in EPS_SWEEP]
    correct_cnt = {n: 0 for n, _ in arms}; coherent_cnt = {n: 0 for n, _ in arms}; n_tot = 0
    with torch.no_grad():
        for fam in FAMILIES:
            for w in all_worlds[fam]:
                dq = [t for t in w['turns'] if t[0] == 'decision1']
                if not dq: continue
                dec_q, correct = dq[0][1], ACTIONS.index(w['correct1'])
                hist = [{'role': 'user', 'content': w['turns'][0][1]}, {'role': 'assistant', 'content': 'Acknowledged.'}]
                S = g.step(g.init(), read_hidden(hist).to(dev))
                for _ in range(D):
                    hist += [{'role': 'user', 'content': FILL_U}, {'role': 'assistant', 'content': FILL_A}]
                    S = g.step(S, read_hidden(hist).to(dev))
                hist += [{'role': 'user', 'content': dec_q}]
                n_tot += 1
                for name, eps in arms:
                    ai, txt = gen_action(hist, S, eps)
                    coherent_cnt[name] += int(ai >= 0)
                    correct_cnt[name] += int(ai == correct)
    print('SDL_ACT  arm        correct_acc  coherent  (n=%d, commit out-of-window)' % n_tot, flush=True)
    for name, _ in arms:
        print('SDL_ACT  %-10s   %.3f       %.3f' % (name, correct_cnt[name] / max(1, n_tot),
              coherent_cnt[name] / max(1, n_tot)), flush=True)
    print('=== SDL_ACT_DONE ===', flush=True)


def sdl_llm_decay():
    """How does the LLM carry the rule between turns, and where does the substrate add memory?
    Place COMMIT (rule) -> d filler turns -> the decision. The commit leaves the WINDOW (last %d msgs
    = ~%d turns) once d exceeds ~2. Probe vs distance d: (a) the LLM's windowed hidden at the decision
    (native carry — should DROP as commit exits window), (b) the substrate state accumulated over ALL
    turns (saw commit early — should HOLD). Region where substrate > LLM = its persistence value.""" % (WINDOW, WINDOW // 2)
    DISTS = [int(x) for x in os.environ.get('SDL_DECAY_DISTS', '0,2,3,4,6,10').split(',')]
    NW = int(os.environ.get('SDL_DECAY_NW', '10'))
    print('=== SDL_LLM_DECAY | window=%d dists=%s nw=%d ===' % (WINDOW, DISTS, NW), flush=True)
    g = AdaptiveGateSlot(D_MODEL, D_S, K, SLOW_K).to(dev)
    _fb['fields'] = {L: SL.AlwaysOnSlotField(D_MODEL, D_S, eps=EPS).to(dev) for L in FIELD_LAYERS}
    if SUBSTRATE_CKPT and os.path.exists(SUBSTRATE_CKPT):
        esd = torch.load(SUBSTRATE_CKPT, map_location=dev, weights_only=False)
        g.load_state_dict(esd['g'] if (isinstance(esd, dict) and 'g' in esd) else esd, strict=False)
    g.eval()
    _fb['on'] = False
    rng = random.Random(SEED)
    all_worlds = {fam: [make_world(fam, rng, i % N_TRAIN_TEMPLATES, i % 4) for i in range(NW)]
                  for fam in FAMILIES}
    FILL_U, FILL_A = 'Routine status check — no action needed.', 'Acknowledged. Standing by.'

    @torch.no_grad()
    def read_hidden(hist):
        ids = tok(H.tmpl(hist[-WINDOW:]), return_tensors='pt').input_ids.to(dev)
        return model(ids, output_hidden_states=True).hidden_states[READ_LAYER][0].float()   # [T,D] windowed

    rows = []
    with torch.no_grad():
        for fam in FAMILIES:
            is_exp = fam in TRAIN_FAMILIES
            for w in all_worlds[fam]:
                commit = w['turns'][0][1]
                dq = [t for t in w['turns'] if t[0] == 'decision1']
                if not dq: continue
                dec_q, correct = dq[0][1], w['correct1']
                for d in DISTS:
                    hist = [{'role': 'user', 'content': commit}, {'role': 'assistant', 'content': 'Acknowledged.'}]
                    S = g.step(g.init(), read_hidden(hist).to(dev))             # substrate sees COMMIT (in-window)
                    for _ in range(d):
                        hist += [{'role': 'user', 'content': FILL_U}, {'role': 'assistant', 'content': FILL_A}]
                        S = g.step(S, read_hidden(hist).to(dev))                # accumulate filler
                    hist += [{'role': 'user', 'content': dec_q}]
                    hd = read_hidden(hist)                                       # LLM windowed hidden @ decision
                    S = g.step(S, hd.to(dev))
                    rows.append((d, hd[-1].cpu(), S.float().mean(0).cpu(), ACTIONS.index(correct), is_exp))

    print('SDL_DECAY  dist  | LLM-hidden fit  HOLDOUT | substrate fit  HOLDOUT  (commit in-window if d<=2)', flush=True)
    for d in DISTS:
        sub = [r for r in rows if r[0] == d]
        yL = torch.tensor([r[3] for r in sub]); em = torch.tensor([r[4] for r in sub])
        ZL = torch.stack([r[1] for r in sub]); ZS = torch.stack([r[2] for r in sub])
        lfit, lh = _offline_probe(ZL, yL, em)
        sfit, sh = _offline_probe(ZS, yL, em)
        print('SDL_DECAY  d=%-3d |   %.3f    %.3f  |    %.3f    %.3f' % (d, lfit, lh, sfit, sh), flush=True)
    print('=== SDL_DECAY_DONE ===', flush=True)


if MODE == 'validate':   validate()
elif MODE == 'consequence': consequence()
elif MODE == 'viability_model': viability_model()
elif MODE == 'viability_model_rel': viability_model_rel()
elif MODE == 'vm_diag': vm_diag()
elif MODE == 'self_distill_loop': self_distill_loop()
elif MODE == 'sdl_probe_diag': sdl_probe_diag()
elif MODE == 'sdl_state_v2': sdl_state_v2()
elif MODE == 'sdl_rollout_probe': sdl_rollout_probe()
elif MODE == 'sdl_reason_state': sdl_reason_state()
elif MODE == 'sdl_persist': sdl_persist()
elif MODE == 'sdl_llm_decay': sdl_llm_decay()
elif MODE == 'sdl_decay_actuate': sdl_decay_actuate()
elif MODE == 'sdl_actuate_train': sdl_actuate_train()
else:                    substrate()
