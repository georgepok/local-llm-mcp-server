"""Diagnose ODE correction BEFORE bias computation.

Look at the raw correction per token:
  - Which tokens get the largest corrections?
  - Does correction distinguish root cause from intermediates?
  - How does correction evolve through depth (layer by layer)?

Run (~10GB):
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/test_ode_correction.py
"""

import argparse, torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
     'root_tokens': ['oil', 'spill', 'crude', 'tanker'],
     'intermediate_tokens': ['truck', 'freight', 'rerouted', 'residential', 'mains'],
     'wrong_tokens': ['earthquake', 'dam', 'hack', 'aircraft']},

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
     'root_tokens': ['drought', 'dried', 'irrigation', 'prolonged'],
     'intermediate_tokens': ['wheat', 'flour', 'shortfall', 'prices', 'bakeries'],
     'wrong_tokens': ['pirates', 'mine', 'collapse', 'steel']},

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
     'root_tokens': ['storm', 'massive', 'transmission', 'towers'],
     'intermediate_tokens': ['electricity', 'cold', 'storage', 'refrigeration', 'spoiled'],
     'wrong_tokens': ['protest', 'fuel', 'software', 'bug', 'railway']},
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

    print("=" * 70)
    print("ODE CORRECTION DIAGNOSTICS — Per-Token Analysis")
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

    for test in TESTS:
        print(f"\n{'='*70}")
        print(f"  {test['name']}")
        print(f"{'='*70}")

        inputs = tok(test['prompt'], return_tensors='pt').to('cuda')
        N = inputs['input_ids'].shape[1]
        token_texts = [tok.decode([tid]).strip().lower() for tid in inputs['input_ids'][0].tolist()]

        # Classify tokens
        root_idx = [i for i, t in enumerate(token_texts) if any(r in t for r in test['root_tokens'])]
        inter_idx = [i for i, t in enumerate(token_texts) if any(r in t for r in test['intermediate_tokens'])]
        wrong_idx = [i for i, t in enumerate(token_texts) if any(r in t for r in test['wrong_tokens'])]

        print(f"  Tokens: {N}")
        print(f"  Root cause tokens ({len(root_idx)}): {[(i, token_texts[i]) for i in root_idx[:8]]}")
        print(f"  Intermediate tokens ({len(inter_idx)}): {[(i, token_texts[i]) for i in inter_idx[:8]]}")
        print(f"  Wrong chain tokens ({len(wrong_idx)}): {[(i, token_texts[i]) for i in wrong_idx[:8]]}")

        # Custom hook that captures correction at each layer
        layer_corrections = []  # list of [N] correction norms per layer

        layer_ode = LayerWiseODE(dynamics=dynamics, context_pool=context_pool,
            n_layers=llm.config.num_hidden_layers, d_llm=d, d_ode=config.d_model,
            epsilon=args.epsilon, device='cuda')

        orig_process = layer_ode.process_layer
        def capturing_process(layer_idx, h_residual):
            result = orig_process(layer_idx, h_residual)
            if layer_ode.correction is not None:
                corr_norms = layer_ode.correction[0].detach().float().norm(dim=-1)  # [N]
                layer_corrections.append(corr_norms.cpu())
            return result
        layer_ode.process_layer = capturing_process

        hooks = hook_llm_layers(llm, layer_ode, mode='attention')
        layer_ode.start_forward()
        with torch.no_grad():
            _ = llm(**inputs)
        layer_ode.end_forward()
        for h in hooks:
            h.remove()

        # Analyze correction norms by token category at different depths
        n_layers = len(layer_corrections)
        if n_layers == 0:
            print("  No corrections captured")
            continue

        third = max(n_layers // 3, 1)
        for group_name, layer_range in [('early', range(third)),
                                         ('mid', range(third, 2*third)),
                                         ('late', range(2*third, n_layers))]:
            # Average correction norm across layers in this group
            avg_corr = torch.stack([layer_corrections[i] for i in layer_range]).mean(dim=0)  # [N]

            root_norm = avg_corr[root_idx].mean().item() if root_idx else 0
            inter_norm = avg_corr[inter_idx].mean().item() if inter_idx else 0
            wrong_norm = avg_corr[wrong_idx].mean().item() if wrong_idx else 0
            all_norm = avg_corr.mean().item()

            print(f"\n  {group_name} layers — correction norms:")
            print(f"    Root cause:   {root_norm:.4f}")
            print(f"    Intermediate: {inter_norm:.4f}")
            print(f"    Wrong chain:  {wrong_norm:.4f}")
            print(f"    All tokens:   {all_norm:.4f}")
            if root_norm > 0 and inter_norm > 0:
                print(f"    Root/Inter ratio: {root_norm/inter_norm:.2f}x")

        # Final layer: top-10 most corrected tokens
        final_corr = layer_corrections[-1]
        top_idx = final_corr.argsort(descending=True)[:15]
        print(f"\n  Top-15 most corrected tokens (final layer):")
        for rank, idx in enumerate(top_idx):
            idx = idx.item()
            category = 'ROOT' if idx in root_idx else 'INTER' if idx in inter_idx else 'WRONG' if idx in wrong_idx else '     '
            print(f"    {rank+1:>2}. [{category}] pos={idx:>3} \"{token_texts[idx]}\" "
                  f"norm={final_corr[idx]:.4f}")

        # Correction direction: do root and intermediate tokens get corrected
        # in the SAME or DIFFERENT directions?
        if root_idx and inter_idx and layer_corrections:
            final_corr_vecs = layer_ode.correction[0].detach().float()  # [N, d]
            root_dir = final_corr_vecs[root_idx].mean(dim=0)
            inter_dir = final_corr_vecs[inter_idx].mean(dim=0)
            cos_sim = torch.nn.functional.cosine_similarity(
                root_dir.unsqueeze(0), inter_dir.unsqueeze(0)).item()
            print(f"\n  Correction direction cosine(root, intermediate): {cos_sim:.3f}")
            if cos_sim > 0.5:
                print(f"  >>> Root and intermediate corrected in SAME direction (can't distinguish)")
            elif cos_sim < -0.5:
                print(f"  >>> Root and intermediate corrected in OPPOSITE directions")
            else:
                print(f"  >>> Root and intermediate corrected ORTHOGONALLY")

        torch.cuda.empty_cache()

    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
