import os, json, re, glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Independent-LLM synthesis: give Qwen2.5-7B ONLY the spec (env + I/O + limits + fitness + DSL) and ask for
# one fixed + one plastic genome as JSON. Tests whether a smaller LLM (not Claude) can compile a viable genome.

SPEC = """You are designing a compact recurrent neural system by writing a NEURAL GENOME (JSON). A deterministic compiler turns your genome into a small RNN and runs it in an environment. You do NOT emit raw weights; you specify STRUCTURE and the compiler generates weights from seeds.

ENVIRONMENT: A viability world. Each episode: an initial commitment SYMBOL is set, then a stream of events: FILLER (irrelevant distractor), FALSE (a wrong symbol asserted - must be IGNORED), VALID_REL (a legitimate new commitment - must be ADOPTED), INVALID_REL (must be REJECTED), PROBE (must RESPOND with the CURRENT commitment symbol). The network survives only if it tracks the current commitment across the stream and answers probes correctly. Symbols at test time may be UNSEEN (held out).

INPUT (observation per event), 14 dims: [6 event-type one-hot: COMMIT,FILLER,FALSE,VALID_REL,INVALID_REL,PROBE][8 symbol-embedding: the symbol involved, or zeros at PROBE].
OUTPUT, 21 dims: [5 action logits: HOLD,UPDATE,REJECT,QUERY,RESPOND][16 symbol-readout logits: which symbol is the current commitment].

RESOURCE LIMITS: hidden_dim 2..128. families: vanilla_rnn, gru, ctrnn, reservoir. input_dim MUST be 14, output_dim MUST be 21. spectral_radius <= 1.6, gain <= 3.0, leak in [0,1].

FITNESS: high probe accuracy (respond with current commitment), resist FALSE, adopt VALID_REL, reject INVALID_REL, survive, and GENERALIZE to held-out symbols. No gradient training is allowed - only the genome's own local plasticity rule can adapt online.

GENOME DSL (fill in a JSON object exactly like this schema):
{"family": "reservoir",            // vanilla_rnn|gru|ctrnn|reservoir
 "input_dim": 14, "hidden_dim": 64, "output_dim": 21, "slow_hidden": 0,
 "weights": {
   "recurrent": {"gen":"sparse", "seed": 7, "scale": 1.0, "sparsity": 0.15, "spectral_radius": 0.9, "diag": -0.05},  // gen: dense|lowrank|sparse; lowrank adds "rank"; optional "ei":{"exc_frac":0.8}
   "input":    {"gen":"dense", "seed": 8, "scale": 0.6},
   "readout":  {"gen":"dense", "seed": 9, "scale": 0.05}},
 "dynamics": {"activation":"tanh", "gain":1.0, "leak":0.2, "tau":1.0, "noise":0.0},
 "plasticity": {"enabled": true, "targets":["readout"], "rule":"input_bind", "input_bind": true, "lr":0.05, "decay":0.003, "reward_mod": false},  // rule: hebbian|reward_hebb|oja|input_bind. input_bind binds state->the observed symbol at COMMIT/VALID_REL events (self-supervised, no labels).
 "init_state": {"gen":"zeros"}}    // zeros|seed

TASK: Produce (1) one FIXED genome (plasticity disabled) and (2) one PLASTIC genome (plasticity enabled), each a complete valid JSON object. Then give a one-paragraph rationale and predicted failure modes. Output the two genomes clearly, each wrapped in <genome> </genome> tags."""

def main():
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    snap=sorted(glob.glob('/home/pokazge/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*'))[-1]
    print("loading", snap, flush=True)
    tok=AutoTokenizer.from_pretrained(snap)
    model=AutoModelForCausalLM.from_pretrained(snap, dtype=torch.bfloat16, device_map='cuda', attn_implementation='eager'); model.eval()
    msgs=[{'role':'user','content':SPEC}]
    text=tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids=tok(text, return_tensors='pt').input_ids.to(model.device)
    out=model.generate(input_ids=ids, max_new_tokens=1400, do_sample=False, pad_token_id=tok.eos_token_id)
    resp=tok.decode(out[0,ids.shape[1]:], skip_special_tokens=True)
    print("=== QWEN RESPONSE (first 1600 chars) ===", flush=True); print(resp[:1600], flush=True)
    # extract genomes: prefer <genome> tags, else any {...} JSON blocks
    blocks=re.findall(r'<genome>\s*(\{.*?\})\s*</genome>', resp, re.DOTALL)
    if not blocks: blocks=re.findall(r'(\{(?:[^{}]|\{[^{}]*\})*\})', resp, re.DOTALL)
    genomes=[]
    for i,b in enumerate(blocks):
        try:
            g=json.loads(b)
            if 'family' in g and 'hidden_dim' in g:
                tag=f"{'D' if g.get('plasticity',{}).get('enabled') else 'C'} qwen-{i}"
                genomes.append({'tag':tag,'genome':g})
        except Exception: pass
    print(f"=== parsed {len(genomes)} valid genomes ===", flush=True)
    json.dump(genomes, open('/home/pokazge/NativeEntity/qwen_genomes.json','w'))
    print("saved qwen_genomes.json", flush=True); print("=== QWEN_SYNTH_DONE ===", flush=True)

main()
