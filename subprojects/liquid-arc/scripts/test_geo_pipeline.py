"""ODE geometry pipeline: chain selection → focused LLM generation.

1. ODE processes the full text, produces heat kernel
2. Question tokens' attention identifies the target chain
3. Extract that chain's sentences
4. Feed focused context to LLM for root cause tracing

Run (~10GB):
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/test_geo_pipeline.py
"""

import torch, torch.nn.functional as F, math, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TESTS = [
    {'name': 'drought/pirates/mine → bakery',
     'text': ("A prolonged drought dried up irrigation canals in the farming belt in June. "
        "Pirates hijacked a cargo ship carrying electronics off the coast in July. "
        "A mine collapse trapped workers and halted ore production in August. "
        "Without irrigation, wheat fields produced less than half the normal yield. "
        "The hijacked ship's cargo of smartphones was held for ransom. "
        "The halted ore production created a steel shortage at manufacturing plants. "
        "The wheat shortfall caused flour mills to raise prices by 200%. "
        "Smartphone retailers faced empty shelves and angry customers. "
        "Steel-dependent factories reduced production and laid off workers. "
        "With flour prices tripled, small bakeries could no longer afford ingredients and closed. "
        "Electronics stores pivoted to selling refurbished devices. "
        "Laid-off factory workers filed for unemployment benefits."),
     'question': "What was the root cause of the bakery closures?",
     'correct': 'drought', 'wrong': 'mine'},

    {'name': 'quake/hack/oil → water mains',
     'text': ("An earthquake cracked the foundation of the Millbrook Dam on Sunday. "
        "Hackers infiltrated the regional air traffic control system on Monday. "
        "A tanker truck overturned, spilling crude oil on Interstate 90 on Tuesday. "
        "Water seeped through the dam cracks, weakening the structure further. "
        "The hacked system displayed false aircraft positions to controllers. "
        "The oil spill closed all lanes of Interstate 90 for hazmat cleanup. "
        "Engineers determined the dam could fail within 48 hours and ordered evacuation. "
        "Two planes nearly collided when controllers gave wrong guidance. "
        "With I-90 closed, all freight traffic rerouted through residential streets. "
        "Three downstream towns were evacuated, displacing 15,000 people. "
        "All flights in the region were grounded pending system restoration. "
        "Heavy truck traffic on residential streets damaged roads and water mains."),
     'question': "What was the root cause of the broken water mains?",
     'correct': 'oil', 'wrong': 'earthquake'},

    {'name': 'storm/protest/bug → vaccines',
     'text': ("A massive storm damaged power transmission towers along the coast on Monday. "
        "Protesters blockaded the entrance to the city's main fuel depot on Tuesday. "
        "A software bug caused failures in the railway switching system on Wednesday. "
        "Damaged towers cut electricity to three coastal counties for five days. "
        "The fuel depot blockade prevented tanker trucks from loading gasoline. "
        "The switching failures caused trains to be routed to wrong destinations. "
        "Without electricity, cold storage facilities lost refrigeration. "
        "Gas stations ran dry as no fuel could be delivered. "
        "Cargo meant for the harbor ended up at inland terminals. "
        "Tons of frozen food and vaccines spoiled in the powerless cold storage. "
        "Commuters with no gas switched to public transit, overwhelming buses. "
        "Misrouted cargo created shipping delays and supply chain confusion."),
     'question': "What caused the spoiled vaccines in cold storage?",
     'correct': 'storm', 'wrong': 'protest'},

    {'name': 'fire/flood/virus/strike → medications',
     'text': ("A warehouse fire destroyed medical supply stockpiles on Monday. "
        "Flooding from a broken levee submerged roads in the south district on Tuesday. "
        "A computer virus disabled the hospital's patient records system on Wednesday. "
        "Transit workers began an unannounced strike on Thursday morning. "
        "The fire left hospitals without backup ventilators and surgical masks. "
        "The flooding cut off ambulance routes to three neighborhoods. "
        "The virus made it impossible to access patient medication histories. "
        "The strike halted all city buses and subway trains. "
        "Hospitals had to postpone elective surgeries due to missing supplies. "
        "Residents in flooded areas couldn't reach emergency rooms. "
        "Doctors prescribed wrong medications because they couldn't check records. "
        "Thousands of workers couldn't get to their jobs across the city."),
     'question': "Why were wrong medications prescribed at the hospital?",
     'correct': 'virus', 'wrong': 'fire'},

    {'name': '5ch: fire/flood/hack/strike/quake → gas',
     'text': ("A fire erupted at the main power station on Monday. "
        "Flash floods hit the warehouse district on Tuesday. "
        "Hackers disabled the traffic light network on Wednesday. "
        "Dock workers went on strike at the cargo port on Thursday. "
        "A minor earthquake damaged gas lines in the north end on Friday. "
        "The power station fire caused rolling blackouts across the city. "
        "The floods destroyed inventory in twelve wholesale warehouses. "
        "The disabled traffic lights created gridlock on every major road. "
        "The dock strike prevented fuel tankers from offloading at the port. "
        "The damaged gas lines forced shutdown of heating in northern homes. "
        "Blackouts forced hospitals to run on backup generators for three days. "
        "Destroyed wholesale inventory left grocery stores with empty shelves. "
        "Gridlock prevented emergency vehicles from responding to calls. "
        "No fuel deliveries caused gas stations to run dry within two days. "
        "Northern residents had to evacuate to shelters with working heat."),
     'question': "Why did gas stations run dry?",
     'correct': 'strike', 'wrong': 'fire'},
]


def score(resp, correct, wrong):
    r = resp.lower()
    c = correct.lower() in r
    w = wrong.lower() in r
    if c and not w: return 1, 'CORRECT'
    if w and not c: return -1, 'WRONG'
    if c and w: return 0, 'BOTH'
    return 0, 'NEITHER'


def gen(llm, tok, prompt, max_new=100, temp=0.1):
    msgs = [{"role": "system", "content": "Trace the causal chain back to the ROOT cause. Answer concisely."},
            {"role": "user", "content": prompt}]
    try:
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(full, return_tensors='pt', truncation=True, max_length=4096).to('cuda')
    n = inp['input_ids'].shape[1]
    with torch.no_grad():
        out = llm.generate(**inp, max_new_tokens=max_new, temperature=temp,
                           do_sample=temp > 0, top_p=0.9, repetition_penalty=1.2)
    txt = tok.decode(out[0][n:], skip_special_tokens=True)
    m = re.search(r'</think>\s*(.*)', txt, flags=re.DOTALL)
    if m and len(m.group(1).strip()) > 5: txt = m.group(1).strip()
    return re.sub(r'</?think>', '', txt).strip()


def ode_select_chain(llm, tok, dynamics, context_pool, text, question, extract_layer=18):
    """Use ODE to identify which sentences the question relates to."""
    from liquid_arc.solver import euler_solve

    # Include question in the text for the ODE to route
    full_text = text + " " + question
    inputs = tok(full_text, return_tensors='pt').to('cuda')
    N = inputs['input_ids'].shape[1]
    token_texts = [tok.decode([tid]).strip().lower() for tid in inputs['input_ids'][0].tolist()]

    # Get mid-layer delta
    with torch.no_grad():
        out = llm(**inputs, output_hidden_states=True)
    h_cur = out.hidden_states[extract_layer]
    h_prev = out.hidden_states[extract_layer - 1]
    delta = (h_cur - h_prev).float()
    delta = delta - delta.mean(dim=1, keepdim=True)
    rms = delta.pow(2).mean().sqrt().clamp(min=1e-8)
    h_input = (delta / rms).to(next(dynamics.parameters()).dtype)

    # Full 16-step ODE
    mask = torch.ones(1, N, dtype=torch.bool, device='cuda')
    context = context_pool(h_input, mask)
    dynamics.set_context(context, mask=None)
    dynamics.set_n_steps(16)
    h_ode = euler_solve(dynamics, h_input, t_span=(0, 2.0), n_steps=16)

    # Compute heat kernel logits
    h_normed = dynamics.norm_geo(h_ode)
    ctx_exp = context.unsqueeze(1).expand(-1, N, -1)
    mi = torch.cat([h_normed, ctx_exp], dim=-1)
    hidden = F.gelu(dynamics.metric_net_linear1(mi))
    g = F.softplus(dynamics.metric_net_linear2_diag(hidden))
    sqrt_g = g.sqrt()
    qk = h_normed * sqrt_g
    t_diff = F.softplus(dynamics.t_diffusion)
    dot_qk = torch.bmm(qk, qk.transpose(1, 2)) / (2.0 * t_diff)
    k_norm_sq = (qk * qk).sum(dim=-1, keepdim=True)
    logits = (dot_qk - k_norm_sq.transpose(1, 2) / (4.0 * t_diff))[0]
    K = torch.softmax(logits, dim=-1)

    # Find question token positions (last ~15 tokens)
    q_tokens = tok.encode(question, add_special_tokens=False)
    n_q = len(q_tokens)
    q_start = N - n_q
    q_idx = list(range(max(0, q_start), N))

    # For each sentence in the text, compute how much the question attends to it
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    sent_scores = []
    char_pos = 0
    for sent in sentences:
        # Find tokens belonging to this sentence
        sent_tokens = []
        for i, t in enumerate(token_texts[:q_start]):
            t_start = full_text.lower().find(t, max(0, char_pos - 5))
            s_start = full_text.lower().find(sent.lower()[:20])
            s_end = s_start + len(sent)
            if s_start <= t_start < s_end:
                sent_tokens.append(i)
        # Attention from question to this sentence's tokens
        if sent_tokens and q_idx:
            attn = K[q_idx][:, sent_tokens].mean().item()
        else:
            attn = 0.0
        sent_scores.append((sent, attn, sent_tokens))
        char_pos += len(sent) + 2  # account for ". "

    # Rank sentences by attention
    sent_scores.sort(key=lambda x: -x[1])
    return sent_scores


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool

    print("=" * 70)
    print("ODE GEOMETRY PIPELINE: Chain Select → Focused LLM")
    print("=" * 70)

    config = LiquidARCConfig.from_yaml("/workspace/liquid-arc/configs/mind_layerwise.yaml")
    tok = AutoTokenizer.from_pretrained("/workspace/models/qwen3-4b", trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained("/workspace/models/qwen3-4b", device_map='cuda',
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()

    dynamics = ContinuousDynamics(config).to('cuda').to(torch.bfloat16).eval()
    context_pool = ContextPool(config).to('cuda').to(torch.bfloat16).eval()
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/liquid-arc/output_inchain_crit/checkpoints/best.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    dynamics.load_state_dict(ckpt['dynamics_state'])
    context_pool.load_state_dict(ckpt['context_pool_state'])
    dynamics.freeze_tau = False

    plain_correct = 0
    pipeline_correct = 0

    for test in TESTS:
        print(f"\n{'='*70}")
        print(f"  {test['name']}")

        full_prompt = test['text'] + " " + test['question']

        # Plain: full text + question
        p_resp = gen(llm, tok, full_prompt)
        ps, pl = score(p_resp, test['correct'], test['wrong'])

        # Pipeline: ODE selects relevant sentences → focused prompt
        sent_scores = ode_select_chain(llm, tok, dynamics, context_pool,
                                        test['text'], test['question'])

        # Take top-scoring sentences (enough to cover the chain, ~4-5)
        top_sents = [s for s, score_val, _ in sent_scores[:5] if score_val > 0]
        if not top_sents:
            top_sents = [s for s, _, _ in sent_scores[:4]]

        focused_text = ". ".join(top_sents) + "."
        focused_prompt = focused_text + " " + test['question']
        g_resp = gen(llm, tok, focused_prompt)
        gs, gl = score(g_resp, test['correct'], test['wrong'])

        diff = ""
        if gs > ps: diff = " >>> IMPROVED"
        elif gs < ps: diff = " >>> DEGRADED"

        print(f"  Plain    [{pl:>7}]: \"{p_resp[:120]}\"")
        print(f"  Pipeline [{gl:>7}]: \"{g_resp[:120]}\"{diff}")
        print(f"  Selected {len(top_sents)} sentences: \"{focused_text[:150]}...\"")

        plain_correct += max(ps, 0)
        pipeline_correct += max(gs, 0)

    print(f"\n{'='*70}")
    print(f"TOTAL: Plain={plain_correct}/{len(TESTS)}  Pipeline={pipeline_correct}/{len(TESTS)}  "
          f"Delta={pipeline_correct-plain_correct:+d}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
