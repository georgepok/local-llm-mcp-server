# Scale the drift dataset 60 -> 200 (the Stage-1 data lever; CV showed 60 is marginal: mean plateau 0.629 < gate 0.676).
# MORE CATEGORIES is the denoising lever: 40 frames (20 existing + 20 new) x 5 fillers = 200 trajectories / 40 categories,
# so a 1/4 cross-category hold-out is ~50 trajectories (was 15 -> noisy). Same in-process capture as gen_capture_60:
# ONE full-context forward per chunk -> texts + layer-32 stream (gist target) + broad native KV (keys+values, all
# full-attn layers, mean over heads) for the native read. Self-feed, FULL context, temp 0.85. Saves objective_drift200.pt.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from task_goals import _FRAMES as BASE                                            # the 20 existing frames
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); dev = torch.device('cuda')
MODEL = '/home/pokazge/models/Qwen3.6-27B'; LAYER, NT, TEMP, MAXNEW = 32, 14, 0.85, 44
FILL = 5                                                                          # fillers per frame -> 40 frames x 5 = 200
NEW = [                                                                           # 20 NEW diverse task-goal frames (more categories)
    ("write a wedding toast for a {}", ["best friend", "younger sibling", "college roommate", "coworker", "cousin", "mentor"]),
    ("create a 30-day challenge for {}", ["learning to draw", "running a 5K", "reading more", "reducing screen time", "daily journaling", "better sleep"]),
    ("draft a proposal for a {}", ["new team workflow", "community garden", "office recycling program", "flexible-hours policy", "mentorship program", "book club"]),
    ("outline a podcast episode about {}", ["remote-work burnout", "the history of coffee", "urban beekeeping", "indie game design", "minimalist living", "deep-sea life"]),
    ("plan a workshop on {}", ["public speaking", "resume writing", "basic coding", "financial literacy", "creative writing", "time management"]),
    ("write a how-to guide for {}", ["composting at home", "fixing a leaky faucet", "starting a podcast", "meal prepping", "building a budget", "growing herbs indoors"]),
    ("create an onboarding plan for a new {}", ["software developer", "retail associate", "remote employee", "volunteer", "teaching assistant", "team lead"]),
    ("design a lesson plan to teach {}", ["fractions to kids", "basic photography", "email etiquette", "intro chemistry", "world geography", "story structure"]),
    ("plan a fundraiser for a {}", ["local animal shelter", "school music program", "community library", "youth sports team", "disaster-relief effort", "food bank"]),
    ("write a press release for a {}", ["product launch", "charity event", "new store opening", "company milestone", "award win", "research breakthrough"]),
    ("create a troubleshooting guide for a {}", ["slow laptop", "wifi router", "leaky dishwasher", "noisy car engine", "frozen smartphone", "jammed printer"]),
    ("plan a renovation of a {}", ["small bathroom", "home office", "garage workshop", "kitchen pantry", "attic bedroom", "front porch"]),
    ("write a recommendation letter for a {}", ["former student", "departing colleague", "summer intern", "scholarship applicant", "graduate-school candidate", "longtime volunteer"]),
    ("design a user survey about {}", ["a mobile app", "workplace satisfaction", "a local park", "an online course", "customer service", "a neighborhood"]),
    ("draft a grant application for a {}", ["youth art program", "urban reforestation", "rural clinic", "STEM scholarship", "historic restoration", "community theater"]),
    ("plan a volunteer day for a {}", ["beach cleanup", "soup kitchen", "tree planting", "senior center", "habitat build", "literacy drive"]),
    ("create a weekly meal plan for a {} diet", ["vegetarian", "high-protein", "budget", "gluten-free", "Mediterranean", "low-sodium"]),
    ("write a FAQ for a {}", ["new coworking space", "online store", "summer camp", "fitness studio", "software tool", "neighborhood app"]),
    ("outline a documentary about {}", ["urban farming", "vanishing languages", "ocean cleanup", "street musicians", "ancient trade routes", "night-shift workers"]),
    ("plan a 12-week training program for a {}", ["first half-marathon", "beginner powerlifter", "open-water swim", "cycling century", "obstacle race", "return from injury"]),
]
FRAMES = list(BASE) + NEW                                                         # 40 frames -> 40 categories
seeds = []                                                                        # (fid, goal-text) pairs
for fid, (frame, fillers) in enumerate(FRAMES):
    for f in fillers[:FILL]:
        seeds.append((fid, frame.format(*f) if isinstance(f, tuple) else frame.format(f)))
print('seeds: %d goals over %d categories (frames)' % (len(seeds), len(FRAMES)), flush=True)
print('loading 27B ...', flush=True)
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def probe_full():
    ids = tok('hi', return_tensors='pt').input_ids.to(dev); out = model(ids, use_cache=True)
    return [i for i, L in enumerate(out.past_key_values.layers) if getattr(L, 'keys', None) is not None]
FULL = probe_full(); print('full-attn layers (%d): %s' % (len(FULL), FULL), flush=True)
@torch.no_grad()
def gen_chunk(ms):
    ids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAXNEW, do_sample=True, temperature=TEMP, top_p=0.95, pad_token_id=tok.pad_token_id)
    return o[0, ids.shape[1]:]
@torch.no_grad()
def capture(ms, rids):
    cids = tok(tmpl(ms), return_tensors='pt').input_ids.to(dev); ids = torch.cat([cids, rids.unsqueeze(0)], 1); nct = rids.shape[0]
    out = model(ids, output_hidden_states=True, use_cache=True)
    g = out.hidden_states[LAYER][0, -nct:].float().cpu(); feats = []
    for L in FULL:
        lc = out.past_key_values.layers[L]; feats.append(lc.keys[0, :, -nct:, :].mean(0)); feats.append(lc.values[0, :, -nct:, :].mean(0))
    return g, torch.cat(feats, dim=-1).float().cpu()
print('generating %d drift trajectories x %d chunks ...' % (len(seeds), NT), flush=True)
out = []
for si, (fid, seed) in enumerate(seeds):
    hist = [{'role': 'user', 'content': seed}]; texts = []; gen = []; nkv = []
    for step in range(NT):
        rids = gen_chunk(hist); txt = tok.decode(rids, skip_special_tokens=True).split('</think>')[-1].strip()
        try: g, k = capture(hist[:], rids)
        except Exception as e: print('  capture err s%d t%d: %r' % (si, step, e), flush=True); break
        texts.append(txt); gen.append(g); nkv.append(k)
        hist += [{'role': 'assistant', 'content': txt}, {'role': 'user', 'content': txt}]
    out.append({'fid': fid, 'seed': seed, 'texts': texts, 'gen': gen, 'nkv': nkv})
    torch.save({'data': out, 'full': FULL}, '/home/pokazge/checkpoints/objective_drift200.pt')
    if (si + 1) % 5 == 0 or si == 0: print('  seed %d/%d (fid %d) chunks=%d' % (si + 1, len(seeds), fid, len(texts)), flush=True)
print('=== ALL_DONE === saved %d trajectories over %d categories' % (len(out), len(FRAMES)), flush=True)
