"""Check what signal the ODE gets from the residual stream.

Does the residual stream differentiate root cause from intermediate tokens?
If yes: ODE/MetricNet is failing to use the signal.
If no: ODE can't possibly distinguish them — the input is the bottleneck.

Also: what do the hidden states look like at the 6 hook layers?
"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TESTS = [
    {'name': 'drought->bakery (DEGRADED)',
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
     'intermediate_tokens': ['wheat', 'flour', 'shortfall', 'prices', 'bakeries', 'ingredients'],
     'wrong_tokens': ['pirates', 'mine', 'collapse', 'steel', 'electronics']},
]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Qwen3-4B...")
    tok = AutoTokenizer.from_pretrained("/workspace/models/qwen3-4b", trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained("/workspace/models/qwen3-4b", device_map='cuda',
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()
    n_layers = llm.config.num_hidden_layers

    for test in TESTS:
        print(f"\n{'='*70}")
        print(f"  {test['name']}")
        print(f"{'='*70}")

        inputs = tok(test['prompt'], return_tensors='pt').to('cuda')
        N = inputs['input_ids'].shape[1]
        token_texts = [tok.decode([tid]).strip().lower() for tid in inputs['input_ids'][0].tolist()]

        root_idx = [i for i, t in enumerate(token_texts) if any(r in t for r in test['root_tokens'])]
        inter_idx = [i for i, t in enumerate(token_texts) if any(r in t for r in test['intermediate_tokens'])]
        wrong_idx = [i for i, t in enumerate(token_texts) if any(r in t for r in test['wrong_tokens'])]

        print(f"  Root:  {[(i, token_texts[i]) for i in root_idx]}")
        print(f"  Inter: {[(i, token_texts[i]) for i in inter_idx]}")
        print(f"  Wrong: {[(i, token_texts[i]) for i in wrong_idx[:6]]}")

        # Get all hidden states
        with torch.no_grad():
            out = llm(**inputs, output_hidden_states=True)
        hs = out.hidden_states  # tuple of [1, N, d] for each layer + embed

        # Analyze at hook layers (every 6th: 5, 11, 17, 23, 29, 35)
        hook_layers = list(range(5, n_layers, 6))
        print(f"\n  Analyzing layers: {hook_layers}")

        print(f"\n  {'Layer':>5} {'root_norm':>10} {'inter_norm':>10} {'wrong_norm':>10} "
              f"{'r-i cos':>8} {'r-w cos':>8} {'i-w cos':>8}")

        for li in hook_layers:
            h = hs[li][0].float()  # [N, d]

            # Per-category norms
            root_norm = h[root_idx].norm(dim=-1).mean().item() if root_idx else 0
            inter_norm = h[inter_idx].norm(dim=-1).mean().item() if inter_idx else 0
            wrong_norm = h[wrong_idx].norm(dim=-1).mean().item() if wrong_idx else 0

            # Per-category mean directions + cosine similarity
            root_mean = h[root_idx].mean(dim=0) if root_idx else torch.zeros(h.shape[1])
            inter_mean = h[inter_idx].mean(dim=0) if inter_idx else torch.zeros(h.shape[1])
            wrong_mean = h[wrong_idx].mean(dim=0) if wrong_idx else torch.zeros(h.shape[1])

            ri_cos = torch.nn.functional.cosine_similarity(
                root_mean.unsqueeze(0), inter_mean.unsqueeze(0)).item()
            rw_cos = torch.nn.functional.cosine_similarity(
                root_mean.unsqueeze(0), wrong_mean.unsqueeze(0)).item()
            iw_cos = torch.nn.functional.cosine_similarity(
                inter_mean.unsqueeze(0), wrong_mean.unsqueeze(0)).item()

            print(f"  {li:>5} {root_norm:>10.1f} {inter_norm:>10.1f} {wrong_norm:>10.1f} "
                  f"{ri_cos:>8.3f} {rw_cos:>8.3f} {iw_cos:>8.3f}")

        # Check: can a simple linear probe distinguish root from intermediate?
        # Use last layer hidden states
        h_last = hs[-1][0].float()
        if root_idx and inter_idx:
            root_vecs = h_last[root_idx]  # [n_root, d]
            inter_vecs = h_last[inter_idx]  # [n_inter, d]

            # Compute within-class vs between-class distance
            root_centroid = root_vecs.mean(dim=0)
            inter_centroid = inter_vecs.mean(dim=0)
            centroid_dist = (root_centroid - inter_centroid).norm().item()

            root_spread = (root_vecs - root_centroid).norm(dim=-1).mean().item()
            inter_spread = (inter_vecs - inter_centroid).norm(dim=-1).mean().item()

            print(f"\n  Last layer separability:")
            print(f"    Root centroid <-> Inter centroid distance: {centroid_dist:.1f}")
            print(f"    Root within-class spread: {root_spread:.1f}")
            print(f"    Inter within-class spread: {inter_spread:.1f}")
            ratio = centroid_dist / (root_spread + inter_spread + 1e-8)
            print(f"    Separation ratio (dist/spread): {ratio:.3f}")
            if ratio > 1.0:
                print(f"    >>> Root and intermediate ARE separable in residual stream")
            elif ratio > 0.3:
                print(f"    >>> Partially separable (signal exists but weak)")
            else:
                print(f"    >>> NOT separable — residual stream doesn't distinguish them")

        # Also check: within the target chain, does position-in-chain matter?
        # Root is early in chain, intermediate is late
        if root_idx and inter_idx:
            # Average position in sequence
            root_pos = sum(root_idx) / len(root_idx)
            inter_pos = sum(inter_idx) / len(inter_idx)
            print(f"\n  Positional info:")
            print(f"    Root avg position: {root_pos:.0f}/{N}")
            print(f"    Inter avg position: {inter_pos:.0f}/{N}")
            print(f"    Gap: {inter_pos - root_pos:.0f} tokens")

    print(f"\nGPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")


if __name__ == '__main__':
    main()
