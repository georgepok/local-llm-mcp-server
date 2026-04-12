"""Test ODE as standalone geometry controller.

Instead of injecting bias back into LLM, analyze the bias matrix directly:
1. Does it cluster tokens by causal chain?
2. Can we extract chain structure from the geometry?
3. Can we use the extracted structure to help the LLM?

Run (~10GB):
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/test_ode_geometry.py
"""

import torch, torch.nn.functional as F, math, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TESTS = [
    {'name': 'drought/pirates/mine → bakery',
     'prompt': ("A prolonged drought dried up irrigation canals in the farming belt in June. "
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
     'chains': {
         'A_drought': ['drought', 'dried', 'irrigation', 'wheat', 'flour', 'shortfall', 'prices', 'bakeries', 'ingredients'],
         'B_pirates': ['pirates', 'hijacked', 'cargo', 'electronics', 'smartphones', 'ransom', 'retailers', 'shelves'],
         'C_mine': ['mine', 'collapse', 'ore', 'steel', 'shortage', 'factories', 'laid', 'unemployment'],
     },
     'question': "What was the root cause of the bakery closures?",
     'target_chain': 'A_drought',
    },
    {'name': 'quake/hack/oil → water mains',
     'prompt': ("An earthquake cracked the foundation of the Millbrook Dam on Sunday. "
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
     'chains': {
         'A_quake': ['earthquake', 'dam', 'cracks', 'seeped', 'evacuation', 'evacuated', 'displacing'],
         'B_hack': ['hackers', 'hacked', 'aircraft', 'controllers', 'planes', 'collided', 'flights', 'grounded'],
         'C_oil': ['tanker', 'oil', 'spill', 'interstate', 'hazmat', 'freight', 'rerouted', 'residential', 'truck', 'mains'],
     },
     'question': "What was the root cause of the broken water mains?",
     'target_chain': 'C_oil',
    },
]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.solver import euler_solve

    print("=" * 70)
    print("ODE AS GEOMETRY CONTROLLER — Chain Discovery")
    print("=" * 70)

    config = LiquidARCConfig.from_yaml("/workspace/liquid-arc/configs/mind_layerwise.yaml")
    tok = AutoTokenizer.from_pretrained("/workspace/models/qwen3-4b", trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained("/workspace/models/qwen3-4b", device_map='cuda',
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()

    dynamics = ContinuousDynamics(config).to('cuda').to(torch.bfloat16).eval()
    context_pool = ContextPool(config).to('cuda').to(torch.bfloat16).eval()
    ckpt = torch.load("/workspace/liquid-arc/output_e2e_ce_v2/checkpoints/step_1500.pt",
        map_location='cuda', weights_only=False)
    dynamics.load_state_dict(ckpt['dynamics_state'])
    context_pool.load_state_dict(ckpt['context_pool_state'])
    dynamics.freeze_tau = False

    for test in TESTS:
        print(f"\n{'='*70}")
        print(f"  {test['name']}")
        print(f"{'='*70}")

        inputs = tok(test['prompt'], return_tensors='pt').to('cuda')
        N = inputs['input_ids'].shape[1]
        token_texts = [tok.decode([tid]).strip().lower() for tid in inputs['input_ids'][0].tolist()]

        # Map tokens to chains
        token_chain = {}
        for i, t in enumerate(token_texts):
            for chain_name, keywords in test['chains'].items():
                if any(k in t for k in keywords):
                    token_chain[i] = chain_name
                    break

        # Get hidden states, extract mid-layer delta
        with torch.no_grad():
            out = llm(**inputs, output_hidden_states=True)
        h18 = out.hidden_states[18]
        h17 = out.hidden_states[17]
        delta = h18 - h17
        delta = delta - delta.mean(dim=1, keepdim=True)
        rms = delta.pow(2).mean().sqrt().clamp(min=1e-8)
        h_input = (delta / rms).to(next(dynamics.parameters()).dtype)

        # Full 16-step ODE
        mask = torch.ones(1, N, dtype=torch.bool, device='cuda')
        context = context_pool(h_input, mask)
        dynamics.set_context(context, mask=None)
        dynamics.set_n_steps(16)
        h_ode = euler_solve(dynamics, h_input, t_span=(0, 2.0), n_steps=16)

        # Compute bias matrix from ODE state
        h_normed = dynamics.norm_geo(h_ode)
        param_dtype = next(dynamics.parameters()).dtype
        ctx = context.unsqueeze(1).expand(-1, N, -1)
        mi = torch.cat([h_normed, ctx], dim=-1)
        hidden = F.gelu(dynamics.metric_net_linear1(mi))
        g = F.softplus(dynamics.metric_net_linear2_diag(hidden))
        sqrt_g = g.sqrt()
        qk = h_normed * sqrt_g
        t_diff = F.softplus(dynamics.t_diffusion)
        dot_qk = torch.bmm(qk, qk.transpose(1, 2)) / (2.0 * t_diff)
        k_norm_sq = (qk * qk).sum(dim=-1, keepdim=True)
        bias = dot_qk - k_norm_sq.transpose(1, 2) / (4.0 * t_diff)
        B = bias[0].float()  # [N, N]

        # Convert to heat kernel (attention weights)
        K = torch.softmax(B, dim=-1)  # [N, N]

        # ── Analysis 1: Within-chain vs cross-chain attention ──
        chain_names = list(test['chains'].keys())
        print(f"\n  Chain attention matrix (avg attention between chain pairs):")
        print(f"  {'':>12}", end='')
        for cn in chain_names:
            print(f"  {cn:>12}", end='')
        print()

        for cn_i in chain_names:
            idx_i = [k for k, v in token_chain.items() if v == cn_i]
            print(f"  {cn_i:>12}", end='')
            for cn_j in chain_names:
                idx_j = [k for k, v in token_chain.items() if v == cn_j]
                if idx_i and idx_j:
                    attn = K[idx_i][:, idx_j].mean().item()
                else:
                    attn = 0
                marker = ' *' if cn_i == cn_j else '  '
                print(f"  {attn:>10.4f}{marker}", end='')
            print()

        # ── Analysis 2: Can the geometry identify the target chain? ──
        # For the question "what caused X?", X tokens should attend most to target chain
        q_text = test['question'].lower()
        # Find the effect token (last noun before "?")
        effect_words = ['bakery', 'closures', 'mains', 'water']
        effect_idx = [i for i, t in enumerate(token_texts) if any(e in t for e in effect_words)]

        if effect_idx:
            # Which chain do effect tokens attend to most?
            print(f"\n  Effect tokens: {[(i, token_texts[i]) for i in effect_idx]}")
            print(f"  Effect → chain attention:")
            for cn in chain_names:
                chain_idx = [k for k, v in token_chain.items() if v == cn]
                if chain_idx:
                    attn = K[effect_idx][:, chain_idx].mean().item()
                    is_target = ' ← TARGET' if cn == test['target_chain'] else ''
                    print(f"    → {cn}: {attn:.4f}{is_target}")

        # ── Analysis 3: Chain-internal flow direction ──
        # Within the target chain, does attention flow from effect → root?
        target_chain = test['target_chain']
        target_idx = sorted([k for k, v in token_chain.items() if v == target_chain])
        if len(target_idx) >= 4:
            early = target_idx[:len(target_idx)//2]  # root cause end
            late = target_idx[len(target_idx)//2:]    # effect end
            # Does late attend to early (effect → root cause)?
            late_to_early = K[late][:, early].mean().item()
            early_to_late = K[early][:, late].mean().item()
            print(f"\n  Within target chain ({target_chain}):")
            print(f"    Effect→Root attention: {late_to_early:.4f}")
            print(f"    Root→Effect attention: {early_to_late:.4f}")
            if late_to_early > early_to_late:
                print(f"    >>> Geometry traces BACKWARD (effect→root) — correct for causal tracing")
            else:
                print(f"    >>> Geometry traces FORWARD (root→effect)")

        # ── Analysis 4: Use geometry to construct structured prompt ──
        # Rank chains by how much the effect attends to them
        # Then construct a prompt hint: "The answer is in the chain involving: X, Y, Z"
        if effect_idx:
            chain_scores = {}
            for cn in chain_names:
                chain_idx_list = [k for k, v in token_chain.items() if v == cn]
                if chain_idx_list:
                    chain_scores[cn] = K[effect_idx][:, chain_idx_list].mean().item()
            ranked = sorted(chain_scores.items(), key=lambda x: -x[1])
            print(f"\n  Chain ranking by geometric attention from effect:")
            for rank, (cn, score) in enumerate(ranked):
                is_target = ' ← CORRECT' if cn == test['target_chain'] else ''
                print(f"    {rank+1}. {cn}: {score:.4f}{is_target}")

            # Did geometry correctly identify the target chain as #1?
            if ranked[0][0] == test['target_chain']:
                print(f"  >>> GEOMETRY CORRECTLY IDENTIFIES TARGET CHAIN")
            else:
                print(f"  >>> Geometry picked wrong chain ({ranked[0][0]})")

    print(f"\n{'='*70}")
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")


if __name__ == '__main__':
    main()
