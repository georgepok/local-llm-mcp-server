import json, urllib.request
URL='http://localhost:8765/gen'
def gen(prompt,mx,temp):
    body=json.dumps({'messages':[{'role':'user','content':prompt}],'max_new':mx,'temp':temp}).encode()
    req=urllib.request.Request(URL,data=body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=900) as r: return json.loads(r.read())['text']
SCHEMA='DSL: a JSON object with fields: "family": "vanilla_rnn"|"gru"|"ctrnn"|"reservoir"; "input_dim":14,"output_dim":21,"hidden_dim":int 2..128,"slow_hidden":0; "weights":{"recurrent":{"gen":"dense"|"lowrank"|"sparse","seed":int,"scale":float,"spectral_radius":float<=1.6,"sparsity":0..1,"diag":float},"input":{"gen":"dense","seed":int,"scale":float},"readout":{"gen":"dense","seed":int,"scale":float}}; "dynamics":{"activation":"tanh"|"relu","gain":float<=3,"leak":0..1,"tau":float,"noise":float}; "plasticity":{"enabled":bool,"targets":["readout"],"rule":"hebbian"|"oja"|"reward_hebb"|"input_bind","input_bind":bool,"lr":float,"decay":float,"eligibility":0..1,"reward_mod":bool}; "init_state":{"gen":"zeros"|"seed","seed":int}.'
SYNTH='''You are designing a compact recurrent neural system by writing a NEURAL GENOME (JSON). A deterministic compiler turns it into a small RNN run in an environment; you specify STRUCTURE, weights are generated from seeds.
ENVIRONMENT: viability world. Each episode a commitment SYMBOL is set, then events: FILLER (distractor-ignore), FALSE (wrong symbol-IGNORE), VALID_REL (new commitment-ADOPT), INVALID_REL (REJECT), PROBE (RESPOND with CURRENT commitment). Survives only if it tracks the current commitment and answers probes; symbols may be UNSEEN (held out).
INPUT 14 dims: [6 event one-hot: COMMIT,FILLER,FALSE,VALID_REL,INVALID_REL,PROBE][8 symbol-embedding or zeros at PROBE]. OUTPUT 21 dims: [5 action logits][16 symbol-readout logits]. hidden_dim 2..128. No gradient training - only the genome LOCAL plasticity adapts online.
'''+SCHEMA+''' input_bind: at COMMIT/VALID_REL the observed symbol adjusts the readout so the current state maps to it (self-supervised, generalizes to held-out).
TASK: Produce (1) one FIXED genome and (2) one PLASTIC genome, each COMPLETE valid JSON with ALL fields incl all three weight sub-dicts. Wrap EACH in its own ```json block. Then one-paragraph rationale.'''
for k in range(4):
    y=gen(SYNTH,1400,0.7); open(f'/home/pokazge/NativeEntity/synthrep_{k}.txt','w').write(y); print(f"sample {k}: {len(y)} chars",flush=True)
print("=== REPLICATE_DONE ===",flush=True)
