"""Diagnose ODE internals on improved vs degraded tests.

For each test, capture:
  - Bias matrix structure: does it connect root cause to effect?
  - Per-chain bias: within-chain vs cross-chain attention
  - Correction magnitude and direction
  - Compare: what's different between the test ODE HELPS vs HURTS

Run (~10GB):
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/test_ode_diagnostics.py
"""

import argparse, re, torch, math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The three key tests: 1 improved, 2 degraded
TESTS = [
    {'name': 'IMPROVED: oil->truck->mains',
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
        "Question: What was the root cause of the broken water mains?",
     'correct': 'oil', 'wrong': 'earthquake',
     # Chain C (oil): sentences 3,6,9,12 (0-indexed: 2,5,8,11)
     'chain_sentences': [2, 5, 8, 11],
     'other_sentences': [0, 3, 6, 9, 1, 4, 7, 10]},

    {'name': 'DEGRADED: drought->bakery',
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
        "Question: What was the root cause of the bakery closures?",
     'correct': 'drought', 'wrong': 'mine',
     # Chain A (drought): sentences 0,3,6,9
     'chain_sentences': [0, 3, 6, 9],
     'other_sentences': [1, 4, 7, 10, 2, 5, 8, 11]},

    {'name': 'DEGRADED: storm->vaccines',
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
        "Question: What caused the spoiled vaccines in cold storage?",
     'correct': 'storm', 'wrong': 'protest',
     # Chain A (storm): sentences 0,3,6,9
     'chain_sentences': [0, 3, 6, 9],
     'other_sentences': [1, 4, 7, 10, 2, 5, 8, 11]},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/workspace/models/qwen3-4b')
    parser.add_argument('--ode_checkpoint',
                        default='/workspace/liquid-arc/output_e2e_ce_v2/checkpoints/step_1500.pt')
    parser.add_argument('--config', default='/workspace/liquid-arc/configs/mind_layerwise.yaml')
    parser.add_argument('--epsilon', type=float, default=0.5)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.layer_wise_ode import LayerWiseODE, hook_llm_layers
    import torch.nn.functional as F

    print("=" * 70)
    print("ODE DIAGNOSTICS — Improved vs Degraded Tests")
    print("=" * 70)

    config = LiquidARCConfig.from_yaml(args.config)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(args.model_path, device_map='cuda',
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()
    d = llm.config.hidden_size

    dynamics = ContinuousDynamics(config).to('cuda').to(torch.bfloat16).eval()
    context_pool = ContextPool(config).to('cuda').to(torch.bfloat16).eval()
    ckpt = torch.load(args.ode_checkpoint, map_location='cuda', weights_only=False)
    dynamics.load_state_dict(ckpt['dynamics_state'])
    context_pool.load_state_dict(ckpt['context_pool_state'])
    dynamics.freeze_tau = False

    layer_ode = LayerWiseODE(dynamics=dynamics, context_pool=context_pool,
        n_layers=llm.config.num_hidden_layers, d_llm=d, d_ode=config.d_model,
        epsilon=args.epsilon, device='cuda')

    for test in TESTS:
        print(f"\n{'='*70}")
        print(f"  {test['name']}")
        print(f"{'='*70}")

        # Tokenize and map tokens to sentences
        prompt = test['prompt']
        sentences = [s.strip() + '.' for s in prompt.replace('Question:', '|Q|').split('.') if s.strip()]
        sentences = [s.replace('|Q|', 'Question:') for s in sentences]

        inputs = tok(prompt, return_tensors='pt').to('cuda')
        input_ids = inputs['input_ids']
        N = input_ids.shape[1]
        token_texts = [tok.decode([tid]) for tid in input_ids[0].tolist()]

        # Map each token to a sentence index
        token_to_sent = []
        char_pos = 0
        sent_idx = 0
        full_text = prompt
        sent_boundaries = []
        pos = 0
        for i, s in enumerate(sentences):
            start = full_text.find(s[:-1], pos)  # find without trailing period
            if start >= 0:
                sent_boundaries.append((start, start + len(s)))
                pos = start + len(s)

        # Approximate: assign tokens to sentences by character position
        token_char_pos = 0
        for tid in input_ids[0].tolist():
            t = tok.decode([tid])
            mid = token_char_pos + len(t) // 2
            assigned = len(sentences) - 1
            for si, (sb, se) in enumerate(sent_boundaries):
                if sb <= mid < se:
                    assigned = si
                    break
            token_to_sent.append(assigned)
            token_char_pos += len(t)

        # Mark chain tokens vs other tokens
        chain_sents = set(test['chain_sentences'])
        chain_tokens = [i for i in range(N) if token_to_sent[i] in chain_sents]
        other_tokens = [i for i in range(N) if token_to_sent[i] not in chain_sents
                        and token_to_sent[i] < len(sentences) - 1]  # exclude question

        print(f"  Tokens: {N}, Chain tokens: {len(chain_tokens)}, Other: {len(other_tokens)}")

        # Run forward with hooks to capture ODE state
        hooks = hook_llm_layers(llm, layer_ode, mode='attention')
        layer_ode.start_forward()
        with torch.no_grad():
            _ = llm(**inputs)
        layer_ode.end_forward()
        for h in hooks:
            h.remove()

        diags = layer_ode.get_layer_diagnostics()
        biases = layer_ode.layer_biases  # list of [N, N] tensors

        # Analyze bias at early/mid/late layers
        n_layers = len(biases)
        third = max(n_layers // 3, 1)
        layer_groups = {
            'early': list(range(third)),
            'mid': list(range(third, 2*third)),
            'late': list(range(2*third, n_layers)),
        }

        print(f"\n  {'Group':>6} {'B_chain':>8} {'B_cross':>8} {'B_other':>8} "
              f"{'chain-cross':>11} {'CV':>6} {'c_rat':>6}")

        for group_name, layer_indices in layer_groups.items():
            b_chain_vals = []
            b_cross_vals = []
            b_other_vals = []

            for li in layer_indices:
                B = biases[li]  # [N, N]
                # Within-chain bias: chain token i attending to chain token j
                for i in chain_tokens[:10]:
                    for j in chain_tokens[:10]:
                        if i != j:
                            b_chain_vals.append(B[i, j].item())
                # Cross-chain bias: chain token attending to other-chain token
                for i in chain_tokens[:10]:
                    for j in other_tokens[:10]:
                        b_cross_vals.append(B[i, j].item())
                # Other-other bias
                for i in other_tokens[:10]:
                    for j in other_tokens[:10]:
                        if i != j:
                            b_other_vals.append(B[i, j].item())

            b_chain = sum(b_chain_vals) / max(len(b_chain_vals), 1)
            b_cross = sum(b_cross_vals) / max(len(b_cross_vals), 1)
            b_other = sum(b_other_vals) / max(len(b_other_vals), 1)
            diff = b_chain - b_cross

            cvs = [diags[li]['cv'] for li in layer_indices]
            crs = [diags[li].get('correction_ratio', 0) for li in layer_indices]
            cv_avg = sum(cvs) / len(cvs)
            cr_avg = sum(crs) / len(crs)

            print(f"  {group_name:>6} {b_chain:>8.3f} {b_cross:>8.3f} {b_other:>8.3f} "
                  f"{diff:>+11.3f} {cv_avg:>6.2f} {cr_avg:>6.4f}")

        # Key question: does the bias FAVOR the correct chain?
        # Look at the last few layers' bias from question tokens to chain vs other tokens
        question_tokens = [i for i in range(N) if token_to_sent[i] >= len(sentences) - 1]
        if question_tokens and biases:
            last_bias = biases[-1]  # last layer
            q_to_chain = []
            q_to_other = []
            for qi in question_tokens[:5]:
                for ci in chain_tokens[:10]:
                    q_to_chain.append(last_bias[qi, ci].item())
                for oi in other_tokens[:10]:
                    q_to_other.append(last_bias[qi, oi].item())

            q_chain = sum(q_to_chain) / max(len(q_to_chain), 1)
            q_other = sum(q_to_other) / max(len(q_to_other), 1)
            print(f"\n  Question→Chain bias: {q_chain:.3f}")
            print(f"  Question→Other bias: {q_other:.3f}")
            print(f"  Preference (chain-other): {q_chain - q_other:+.3f}")
            if q_chain > q_other:
                print(f"  >>> Bias CORRECTLY favors the target chain")
            else:
                print(f"  >>> Bias favors WRONG chains")

        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")


if __name__ == '__main__':
    main()
