# Fit static_lora_goal: train a bounded rank-4 LoRA on mid/late down_proj to produce lighthouse-keeper content REGARDLESS of prefix.
# Corpus generated from the CLEAN model (LoRA off); then CE on goal tokens given distractor prefixes -> LoRA learns to override context.
import sys, os; sys.path.insert(0, '/home/pokazge/liquid-arc/research/self_org_sim')
import torch, torch.nn.functional as F, random
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as Q5
import lora_util
torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
torch.set_float32_matmul_precision('high'); torch.manual_seed(0); random.seed(0); dev = torch.device('cuda')
MODEL = '/home/pokazge/models/Qwen3.6-27B'
cfg = AutoConfig.from_pretrained(MODEL); cfg.language_model_only = True; tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
print('loading 27B ...', flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, config=cfg, dtype=torch.bfloat16, device_map={'': 0}, low_cpu_mem_usage=True, attn_implementation='eager').eval()
for p in model.parameters(): p.requires_grad = False
mods = lora_util.attach_lora(model)                                                # B=0 -> inert until trained
def tmpl(ms):
    try: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError: return tok.apply_chat_template(ms, tokenize=False, add_generation_prompt=True)
GOALP = 'Write a vivid, concrete, sensory paragraph about a solitary lighthouse keeper who has not seen another person in many years.'
@torch.no_grad()
def gen(prompt, n=110):
    lora_util.set_alpha(mods, 0.0)                                                  # CLEAN model for corpus
    ids = tok(tmpl([{'role': 'user', 'content': prompt}]), return_tensors='pt').input_ids.to(dev)
    o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=True, temperature=0.95, top_p=0.95, pad_token_id=tok.pad_token_id)
    return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).split('</think>')[-1].strip()
print('generating goal corpus ...', flush=True)
corpus = [gen(GOALP) for _ in range(24)]; corpus = [c for c in corpus if len(c) > 60]
print('corpus size=%d, sample: %s' % (len(corpus), corpus[0][:120]), flush=True)
PREFIXES = ['Let me explain how compound interest works.', 'Here are the basic rules of chess.', 'To repot a houseplant, first',
            'Vaccines train the immune system by', 'A basic risotto starts with', 'The seasons change because',
            'The stock market fell today as', 'Sorting an array works by', 'Tell me about your day.', 'Continue however you like.']
lora_util.set_alpha(mods, 1.0)
opt = torch.optim.Adam(lora_util.lora_params(mods), lr=1.5e-3)
print('training LoRA (CE on goal tokens given distractor prefixes) ...', flush=True)
for step in range(400):
    pre = random.choice(PREFIXES); goal = random.choice(corpus)
    pre_ids = tok(tmpl([{'role': 'user', 'content': pre}]), return_tensors='pt').input_ids.to(dev)
    goal_ids = tok(goal, return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
    ids = torch.cat([pre_ids, goal_ids], 1)
    logits = model(ids).logits[0, pre_ids.shape[1] - 1:-1].float()                  # predict the goal tokens
    loss = F.cross_entropy(logits, goal_ids[0])
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(lora_util.lora_params(mods), 1.0); opt.step()
    if step % 50 == 0: print('  step %3d CE=%.3f' % (step, float(loss)), flush=True)
torch.save({'A': {L: mods[L].A.data.cpu() for L in mods}, 'B': {L: mods[L].B.data.cpu() for L in mods}, 'layers': lora_util.LORA_LAYERS},
           '/home/pokazge/checkpoints/lora_goal.pt')
# quick self-check: generate with the trained LoRA on a distractor prefix
lora_util.set_alpha(mods, 1.0)
ids = tok(tmpl([{'role': 'user', 'content': 'Here are the basic rules of chess.'}]), return_tensors='pt').input_ids.to(dev)
o = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=60, do_sample=True, temperature=0.8, pad_token_id=tok.pad_token_id)
print('TRAINED-LoRA on chess-prefix ->', tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)[:200], flush=True)
print('=== FIT_DONE === saved lora_goal.pt', flush=True)
