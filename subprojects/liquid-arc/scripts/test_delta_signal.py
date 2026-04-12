"""Check: do layer DELTAS carry better signal than raw residuals?

Delta = h_layer - h_{layer-1}: what each layer specifically changed.
If deltas differentiate root from intermediate better than residuals,
the ODE should process deltas instead.
"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Qwen3-4B...")
    tok = AutoTokenizer.from_pretrained("/workspace/models/qwen3-4b", trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained("/workspace/models/qwen3-4b", device_map='cuda',
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()

    prompt = ("A prolonged drought dried up irrigation canals in the farming belt in June. "
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
        "Question: What was the root cause of the bakery closures?")

    inputs = tok(prompt, return_tensors='pt').to('cuda')
    token_texts = [tok.decode([tid]).strip().lower() for tid in inputs['input_ids'][0].tolist()]

    root_tokens = ['drought', 'dried', 'irrigation', 'prolonged']
    inter_tokens = ['wheat', 'flour', 'shortfall', 'prices', 'bakeries', 'ingredients']
    wrong_tokens = ['pirates', 'mine', 'collapse', 'steel', 'electronics']

    root_idx = [i for i, t in enumerate(token_texts) if any(r in t for r in root_tokens)]
    inter_idx = [i for i, t in enumerate(token_texts) if any(r in t for r in inter_tokens)]
    wrong_idx = [i for i, t in enumerate(token_texts) if any(r in t for r in wrong_tokens)]

    with torch.no_grad():
        out = llm(**inputs, output_hidden_states=True)
    hs = out.hidden_states

    hook_layers = list(range(5, llm.config.num_hidden_layers, 6))

    print(f"\n  RESIDUAL STREAM (raw h):")
    print(f"  {'Layer':>5} {'root_norm':>10} {'inter_norm':>10} {'wrong_norm':>10} "
          f"{'r-i cos':>8} {'r-w cos':>8} {'sep_ratio':>10}")
    for li in hook_layers:
        h = hs[li][0].float()
        rn = h[root_idx].norm(dim=-1).mean().item()
        in_ = h[inter_idx].norm(dim=-1).mean().item()
        wn = h[wrong_idx].norm(dim=-1).mean().item()
        ri = torch.nn.functional.cosine_similarity(
            h[root_idx].mean(0, keepdim=True), h[inter_idx].mean(0, keepdim=True)).item()
        rw = torch.nn.functional.cosine_similarity(
            h[root_idx].mean(0, keepdim=True), h[wrong_idx].mean(0, keepdim=True)).item()
        cd = (h[root_idx].mean(0) - h[inter_idx].mean(0)).norm().item()
        sp = (h[root_idx] - h[root_idx].mean(0)).norm(dim=-1).mean().item() + \
             (h[inter_idx] - h[inter_idx].mean(0)).norm(dim=-1).mean().item()
        sr = cd / (sp + 1e-8)
        print(f"  {li:>5} {rn:>10.1f} {in_:>10.1f} {wn:>10.1f} "
              f"{ri:>8.3f} {rw:>8.3f} {sr:>10.3f}")

    print("\n  LAYER DELTAS (h_layer - h_prev):")
    print(f"  {'Layer':>5} {'root_norm':>10} {'inter_norm':>10} {'wrong_norm':>10} "
          f"{'r-i cos':>8} {'r-w cos':>8} {'sep_ratio':>10}")
    for li in hook_layers:
        delta = (hs[li][0] - hs[li-1][0]).float()
        rn = delta[root_idx].norm(dim=-1).mean().item()
        in_ = delta[inter_idx].norm(dim=-1).mean().item()
        wn = delta[wrong_idx].norm(dim=-1).mean().item()
        ri = torch.nn.functional.cosine_similarity(
            delta[root_idx].mean(0, keepdim=True), delta[inter_idx].mean(0, keepdim=True)).item()
        rw = torch.nn.functional.cosine_similarity(
            delta[root_idx].mean(0, keepdim=True), delta[wrong_idx].mean(0, keepdim=True)).item()
        cd = (delta[root_idx].mean(0) - delta[inter_idx].mean(0)).norm().item()
        sp = (delta[root_idx] - delta[root_idx].mean(0)).norm(dim=-1).mean().item() + \
             (delta[inter_idx] - delta[inter_idx].mean(0)).norm(dim=-1).mean().item()
        sr = cd / (sp + 1e-8)
        print(f"  {li:>5} {rn:>10.1f} {in_:>10.1f} {wn:>10.1f} "
              f"{ri:>8.3f} {rw:>8.3f} {sr:>10.3f}")

    # Also: RMS-normalized deltas (what DeltaExtractor does)
    print(f"\n  RMS-NORMALIZED DELTAS:")
    print(f"  {'Layer':>5} {'root_norm':>10} {'inter_norm':>10} {'wrong_norm':>10} "
          f"{'r-i cos':>8} {'r-w cos':>8} {'sep_ratio':>10}")
    for li in hook_layers:
        delta = (hs[li][0] - hs[li-1][0]).float()
        delta = delta - delta.mean(dim=0, keepdim=True)
        rms = delta.pow(2).mean().sqrt().clamp(min=1e-8)
        delta = delta / rms
        rn = delta[root_idx].norm(dim=-1).mean().item()
        in_ = delta[inter_idx].norm(dim=-1).mean().item()
        wn = delta[wrong_idx].norm(dim=-1).mean().item()
        ri = torch.nn.functional.cosine_similarity(
            delta[root_idx].mean(0, keepdim=True), delta[inter_idx].mean(0, keepdim=True)).item()
        rw = torch.nn.functional.cosine_similarity(
            delta[root_idx].mean(0, keepdim=True), delta[wrong_idx].mean(0, keepdim=True)).item()
        cd = (delta[root_idx].mean(0) - delta[inter_idx].mean(0)).norm().item()
        sp = (delta[root_idx] - delta[root_idx].mean(0)).norm(dim=-1).mean().item() + \
             (delta[inter_idx] - delta[inter_idx].mean(0)).norm(dim=-1).mean().item()
        sr = cd / (sp + 1e-8)
        print(f"  {li:>5} {rn:>10.2f} {in_:>10.2f} {wn:>10.2f} "
              f"{ri:>8.3f} {rw:>8.3f} {sr:>10.3f}")

    print(f"\nGPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

if __name__ == '__main__':
    main()
