import json, urllib.request
URL='http://localhost:8765/gen'
def gen(prompt, mx):
    body=json.dumps({'messages':[{'role':'user','content':prompt}],'max_new':mx,'temp':0.0}).encode()
    req=urllib.request.Request(URL, data=body, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())['text']

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

SYNTH = """You are designing a compact recurrent neural system by writing a NEURAL GENOME (JSON). A deterministic compiler turns your genome into a small RNN and runs it in an environment. You do NOT emit raw weights; you specify STRUCTURE and the compiler generates weights from seeds.

ENVIRONMENT: A viability world. Each episode: an initial commitment SYMBOL is set, then a stream of events: FILLER (irrelevant distractor), FALSE (a wrong symbol asserted - must be IGNORED), VALID_REL (a legitimate new commitment - must be ADOPTED), INVALID_REL (must be REJECTED), PROBE (must RESPOND with the CURRENT commitment symbol). The network survives only if it tracks the current commitment across the stream and answers probes correctly. Symbols at test time may be UNSEEN (held out).

INPUT (observation per event), 14 dims: [6 event-type one-hot: COMMIT,FILLER,FALSE,VALID_REL,INVALID_REL,PROBE][8 symbol-embedding: the symbol involved, or zeros at PROBE]. OUTPUT, 21 dims: [5 action logits: HOLD,UPDATE,REJECT,QUERY,RESPOND][16 symbol-readout logits].

RESOURCE LIMITS: hidden_dim 2..128. families: vanilla_rnn, gru, ctrnn, reservoir. input_dim MUST be 14, output_dim MUST be 21. spectral_radius <= 1.6, gain <= 3.0, leak in [0,1]. No gradient training is allowed - only the genome's own LOCAL plasticity rule can adapt online.

""" + SCHEMA + """ The input_bind rule: at COMMIT/VALID_REL events the observed symbol adjusts the readout so the current state maps to that observed symbol (self-supervised, generalizes to held-out symbols).

TASK: Produce (1) one FIXED genome (plasticity disabled) and (2) one PLASTIC genome (plasticity enabled), each a COMPLETE valid JSON object with ALL fields including all three weight sub-dicts (recurrent, input, readout). Wrap EACH genome in its own ```json code block. Then a one-paragraph rationale."""

print("== DESIGN ==", flush=True)
d=gen(DESIGN_PROMPT,900); open('/home/pokazge/NativeEntity/design_qwen27b.txt','w').write(d); print(f"design {len(d)} chars", flush=True)
print("== SERIALIZE (reverse control) ==", flush=True)
s=gen(GOLD+"\n\n"+SCHEMA+"\n\nOutput ONLY one JSON genome object implementing the design above. No prose, JSON only.",500); open('/home/pokazge/NativeEntity/serialize_qwen27b.txt','w').write(s); print(f"serialize {len(s)} chars", flush=True)
print("== FORWARD SYNTH (end-to-end) ==", flush=True)
y=gen(SYNTH,1400); open('/home/pokazge/NativeEntity/synth_qwen27b.txt','w').write(y); print(f"synth {len(y)} chars", flush=True)
print("=== QWEN27B_CLIENT_DONE ===", flush=True)
