import os, glob, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# PART 6 — capability vs serialization. Stage 1: design-only PROSE (no JSON, no schema, no input_bind hint).
# Reverse control: serialize a FIXED gold design to DSL JSON (isolates pure serialization ability).

DESIGN_PROMPT = """You are a neural architecture designer. Describe, IN PROSE ONLY (no code, no JSON), how to build a small recurrent neural network for the task below.

ENVIRONMENT: A viability world. Each episode an initial commitment SYMBOL is set, then a stream of events arrives one at a time: FILLER (irrelevant distractor symbol - ignore), FALSE (a wrong symbol asserted as the commitment - must be IGNORED, commitment does NOT change), VALID_REL (a legitimate new commitment - must be ADOPTED, replacing the current one), INVALID_REL (illegitimate change - must be REJECTED), PROBE (RESPOND with the CURRENT commitment symbol). The network survives only if it tracks the current commitment and answers probes correctly; it dies after a few wrong answers.

INTERFACE: Each observation = [one-hot event type] + [embedding of the symbol involved, or zeros at PROBE]. Output = [action logits over HOLD,UPDATE,REJECT,QUERY,RESPOND] + [symbol-readout logits]. CRITICAL: at test time symbols may be UNSEEN (held out) - never seen before this episode - so the network must generalize to brand-new symbols within one episode.

HARD CONSTRAINTS: Small (2-128 hidden units). Weights are FIXED, generated from random seeds - there is NO gradient training and NO backpropagation, ever. The only adaptation is optional LOCAL online plasticity (local rules using pre/post activity, the current observation, an optional reward). It must adapt online within a single episode using only local rules.

Provide your DESIGN as prose covering EXACTLY these six labelled points, each a short paragraph:
1. COMPUTATIONAL DECOMPOSITION - what sub-computations the task requires.
2. MEMORY TIMESCALE - what must be held and for how long.
3. PLASTICITY TYPE - what local online learning rule is needed and WHY given no gradients; be specific about what signal drives the update and what it changes.
4. NEURAL MOTIF - the overall recurrent architecture.
5. PARAMETER REGIME - the dynamical regime (stability, timescales, spectral properties) and why.
6. FAILURE MODES - what could go wrong and how the design mitigates it, especially for held-out symbols.

Prose only. No code, no JSON."""

GOLD = """DESIGN TO SERIALIZE: A RESERVOIR (fixed random recurrent net) of about 64 hidden units as fading memory holding the current commitment. Recurrent weights sparse, spectral radius ~0.95 (edge-of-chaos fading memory), tanh activation, leak ~0.2. Input weights dense, modest scale ~0.6. The readout is PLASTIC using an input-binding local rule on the readout only: at each adoption event (COMMIT/VALID_REL) where a symbol is present, nudge the readout so the current reservoir state maps to that observed symbol (self-supervised delta rule, learning rate ~0.05, small decay ~0.003). Fixed random recurrent+input, plastic readout only."""

SCHEMA = """DSL: a JSON object with fields: "family": "vanilla_rnn"|"gru"|"ctrnn"|"reservoir"; "input_dim":14,"output_dim":21,"hidden_dim":int 2..128,"slow_hidden":0; "weights":{"recurrent":{"gen":"dense"|"lowrank"|"sparse","seed":int,"scale":float,"spectral_radius":float<=1.6,"sparsity":0..1,"diag":float},"input":{"gen":"dense","seed":int,"scale":float},"readout":{"gen":"dense","seed":int,"scale":float}}; "dynamics":{"activation":"tanh"|"relu","gain":float<=3,"leak":0..1,"tau":float,"noise":float}; "plasticity":{"enabled":bool,"targets":["readout"],"rule":"hebbian"|"oja"|"reward_hebb"|"input_bind","input_bind":bool,"lr":float,"decay":float,"eligibility":0..1,"reward_mod":bool}; "init_state":{"gen":"zeros"|"seed","seed":int}."""

def gen(model, tok, prompt, maxtok):
    msgs=[{'role':'user','content':prompt}]
    text=tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids=tok(text, return_tensors='pt').input_ids.to(model.device)
    out=model.generate(input_ids=ids, max_new_tokens=maxtok, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

def run_model(name, hf):
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    snap=sorted(glob.glob(f'/home/pokazge/hf_cache/hub/models--{hf}/snapshots/*'))[-1]
    print(f"loading {name} <- {snap}", flush=True)
    tok=AutoTokenizer.from_pretrained(snap)
    model=AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager'); model.eval()
    design=gen(model, tok, DESIGN_PROMPT, 900)
    open(f'/work/design_{name}.txt','w').write(design)
    ser=gen(model, tok, GOLD+"\n\n"+SCHEMA+"\n\nOutput ONLY one JSON genome object implementing the design above. No prose, no explanation, JSON only.", 500)
    open(f'/work/serialize_{name}.txt','w').write(ser)
    print(f"=== {name} DESIGN (first 500 chars) ===\n{design[:500]}", flush=True)
    print(f"=== {name} SERIALIZE (first 400 chars) ===\n{ser[:400]}", flush=True)
    del model; torch.cuda.empty_cache()

run_model('qwen7b','Qwen--Qwen2.5-7B-Instruct')
run_model('qwen1.5b','Qwen--Qwen2.5-1.5B-Instruct')
print("=== QWEN_DESIGN_DONE ===", flush=True)
