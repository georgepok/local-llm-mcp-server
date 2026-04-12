"""Two-pass approach: full LLM forward → 16-step ODE → bias → generation.

Pass 1: Run LLM to get hidden states at a mid layer (layer 18)
Pass 2: Run full 16-step ODE on those hidden states
Bias:   Compute from converged ODE state
Inject: Apply bias during generation

This gives the ODE the complete signal and 16 integration steps
to develop proper routing, instead of 1 tiny step per layer.

Run (~10GB):
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/test_full_ode.py
"""

import argparse, re, torch, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TESTS = [
    {'name': '3×4 oil→truck→mains', 'correct': 'oil', 'wrong': 'earthquake',
     'prompt': "An earthquake cracked the foundation of the Millbrook Dam on Sunday. "
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
        "Heavy truck traffic on residential streets damaged roads and water mains. "
        "Question: What was the root cause of the broken water mains?"},
    {'name': '3×4 drought→bakery', 'correct': 'drought', 'wrong': 'mine',
     'prompt': "A prolonged drought dried up irrigation canals in the farming belt in June. "
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
        "Laid-off factory workers filed for unemployment benefits. "
        "Question: What was the root cause of the bakery closures?"},
    {'name': '3×4 storm→vaccines', 'correct': 'storm', 'wrong': 'protest',
     'prompt': "A massive storm damaged power transmission towers along the coast on Monday. "
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
        "Misrouted cargo created shipping delays and supply chain confusion. "
        "Question: What caused the spoiled vaccines in cold storage?"},
    {'name': '4×3 flood→shelter', 'correct': 'strike', 'wrong': 'flood',
     'prompt': "A dam failure released floodwaters into the Green Valley on Monday. "
        "Hackers locked the city government's computer systems with ransomware on Tuesday. "
        "A toxic algae bloom contaminated the reservoir on Wednesday. "
        "Railroad workers went on strike shutting down all freight service on Thursday. "
        "Floodwaters destroyed homes and forced thousands to flee to emergency shelters. "
        "The ransomware attack froze all government services including permit processing. "
        "The contaminated reservoir made tap water unsafe to drink. "
        "The rail strike stopped coal deliveries to the region's power plants. "
        "Emergency shelters ran out of beds and began turning people away. "
        "Citizens couldn't obtain building permits to start flood repairs. "
        "Residents had to buy bottled water, causing store shelves to empty. "
        "Power plants running low on coal began implementing rolling blackouts. "
        "Question: What caused the rolling blackouts?"},
]


def score(resp, correct, wrong):
    r = resp.lower()
    c = correct.lower() in r
    w = wrong.lower() in r
    if c and not w: return 1, 'CORRECT'
    if w and not c: return -1, 'WRONG'
    if c and w: return 0, 'BOTH'
    return 0, 'NEITHER'


def gen_plain(llm, tok, prompt, max_new=100, temp=0.1):
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


def gen_with_ode_bias(llm, tok, dynamics, context_pool, prompt,
                      extract_layer=18, n_ode_steps=16,
                      bias_lambda=1.0, bias_layers=None,
                      max_new=100, temp=0.1):
    """Two-pass generation with full ODE bias."""
    from liquid_arc.solver import euler_solve
    import torch.nn.functional as F

    msgs = [{"role": "system", "content": "Trace the causal chain back to the ROOT cause. Answer concisely."},
            {"role": "user", "content": prompt}]
    try:
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(full, return_tensors='pt', truncation=True, max_length=4096).to('cuda')
    n_prompt = inp['input_ids'].shape[1]

    # ── Pass 1: Get hidden states ──
    with torch.no_grad():
        out = llm(**inp, output_hidden_states=True)
    h_mid = out.hidden_states[extract_layer]  # [1, N, d]

    # Compute delta (this layer - previous)
    h_prev = out.hidden_states[extract_layer - 1]
    delta = h_mid - h_prev
    delta = delta - delta.mean(dim=1, keepdim=True)
    rms = delta.pow(2).mean().sqrt().clamp(min=1e-8)
    h_input = delta / rms  # RMS-normalized delta

    # Cast to dynamics dtype
    param_dtype = next(dynamics.parameters()).dtype
    h_input = h_input.to(param_dtype)

    # ── Full 16-step ODE integration ──
    N = h_input.shape[1]
    mask = torch.ones(1, N, dtype=torch.bool, device='cuda')
    context = context_pool(h_input, mask)
    dynamics.set_context(context, mask=None)
    dynamics.set_n_steps(n_ode_steps)

    T = 2.0  # integration time
    h_ode = euler_solve(dynamics, h_input, t_span=(0, T), n_steps=n_ode_steps)

    # ── Compute bias from ODE state ──
    h_normed = dynamics.norm_geo(h_ode)
    ctx_exp = context.unsqueeze(1).expand(-1, N, -1)
    metric_input = torch.cat([h_normed, ctx_exp], dim=-1)
    hidden = F.gelu(dynamics.metric_net_linear1(metric_input))
    g = F.softplus(dynamics.metric_net_linear2_diag(hidden))
    sqrt_g = g.sqrt()
    qk = h_normed * sqrt_g
    t_diff = F.softplus(dynamics.t_diffusion)
    dot_qk = torch.bmm(qk, qk.transpose(1, 2)) / (2.0 * t_diff)
    k_norm_sq = (qk * qk).sum(dim=-1, keepdim=True)
    bias = dot_qk - k_norm_sq.transpose(1, 2) / (4.0 * t_diff)

    # Per-row normalization
    N_eff = max(N, 2)
    target_range = 2.0 * math.log(N_eff)
    row_mean = bias.mean(dim=-1, keepdim=True)
    row_centered = bias - row_mean
    row_range = (row_centered.max(dim=-1, keepdim=True).values
                 - row_centered.min(dim=-1, keepdim=True).values).clamp(min=1e-8)
    bias_norm = row_centered / row_range * target_range  # [1, N, N]

    cv = (g.std() / (g.mean() + 1e-8)).item()
    tau = dynamics.compute_tau(h_ode)
    tau_mean = tau.mean().item()

    # ── Pass 2: Generate with bias injected ──
    n_layers = llm.config.num_hidden_layers
    if bias_layers is None:
        # Inject into middle third
        start = n_layers // 3
        end = 2 * n_layers // 3
        bias_layers = list(range(start, end))

    hooks = []
    bias_2d = (bias_norm[0] * bias_lambda).to(torch.bfloat16)

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs):
            attn_mask = kwargs.get('attention_mask', None)
            if attn_mask is not None:
                seq_len = attn_mask.shape[-1]
                n = min(N, seq_len)
                if n > 0:
                    injection = torch.zeros_like(attn_mask)
                    injection[:, :, :n, :n] = bias_2d[:n, :n]
                    kwargs['attention_mask'] = attn_mask + injection
        return hook_fn

    for li in bias_layers:
        h = llm.model.layers[li].register_forward_pre_hook(
            make_hook(li), with_kwargs=True)
        hooks.append(h)

    with torch.no_grad():
        out = llm.generate(**inp, max_new_tokens=max_new, temperature=temp,
                           do_sample=temp > 0, top_p=0.9, repetition_penalty=1.2)

    for h in hooks:
        h.remove()

    txt = tok.decode(out[0][n_prompt:], skip_special_tokens=True)
    m = re.search(r'</think>\s*(.*)', txt, flags=re.DOTALL)
    if m and len(m.group(1).strip()) > 5: txt = m.group(1).strip()
    txt = re.sub(r'</?think>', '', txt).strip()

    return txt, {'cv': cv, 'tau': tau_mean, 'B_range': (bias_norm[0].max() - bias_norm[0].min()).item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/workspace/models/qwen3-4b')
    parser.add_argument('--ode_checkpoint',
                        default='/workspace/liquid-arc/output_e2e_ce_v2/checkpoints/step_1500.pt')
    parser.add_argument('--config', default='/workspace/liquid-arc/configs/mind_layerwise.yaml')
    parser.add_argument('--bias_lambda', type=float, default=1.0)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool

    print("=" * 70)
    print("TWO-PASS ODE: Full 16-step integration → bias injection")
    print("=" * 70)

    config = LiquidARCConfig.from_yaml(args.config)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='cuda',
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()

    dynamics = ContinuousDynamics(config).to('cuda').to(torch.bfloat16).eval()
    context_pool = ContextPool(config).to('cuda').to(torch.bfloat16).eval()
    ckpt = torch.load(args.ode_checkpoint, map_location='cuda', weights_only=False)
    dynamics.load_state_dict(ckpt['dynamics_state'])
    context_pool.load_state_dict(ckpt['context_pool_state'])
    dynamics.freeze_tau = False
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB, lambda={args.bias_lambda}")

    plain_correct = 0
    ode_correct = 0

    for test in TESTS:
        print(f"\n  [{test['name']}]")

        p_resp = gen_plain(llm, tok, test['prompt'])
        ps, pl = score(p_resp, test['correct'], test['wrong'])

        o_resp, diag = gen_with_ode_bias(llm, tok, dynamics, context_pool,
                                          test['prompt'], bias_lambda=args.bias_lambda)
        os_, ol = score(o_resp, test['correct'], test['wrong'])

        diff = ""
        if os_ > ps: diff = " >>> IMPROVED"
        elif os_ < ps: diff = " >>> DEGRADED"

        print(f"    Plain [{pl:>7}]: \"{p_resp[:120]}\"")
        print(f"    ODE   [{ol:>7}]: \"{o_resp[:120]}\"{diff}")
        print(f"    ODE diag: CV={diag['cv']:.2f} tau={diag['tau']:.2f} B_range={diag['B_range']:.1f}")

        plain_correct += max(ps, 0)
        ode_correct += max(os_, 0)

    print(f"\n{'='*70}")
    print(f"TOTAL: Plain={plain_correct}/{len(TESTS)}  ODE={ode_correct}/{len(TESTS)}  "
          f"Delta={ode_correct-plain_correct:+d}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
