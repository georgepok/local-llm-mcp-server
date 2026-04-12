"""Verify Qwen3-30B-A3B loads correctly for LiquidARC coupling.

Checks:
1. Model loads in bf16 within memory budget
2. output_hidden_states returns 49 states (embed + 48 layers)
3. All 48 layers have hookable self_attn (Qwen3MoeAttention)
4. Generation produces coherent text
5. Delta extraction from middle layer (layer 24) works
6. Forward hook injection into attention works

Run in fgn-train container on DGX Spark:
  python3 /workspace/liquid-arc/scripts/verify_qwen3_30b.py
"""

import torch
import time

MODEL_PATH = '/workspace/models/Qwen3-30B-A3B'

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("Qwen3-30B-A3B Verification for LiquidARC")
    print("=" * 60)

    # 1. Load model
    print("\n[1] Loading model...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  GPU memory: {mem_gb:.1f} GB")
    print(f"  Model type: {type(model).__name__}")
    print(f"  hidden_size: {model.config.hidden_size}")
    print(f"  num_layers: {model.config.num_hidden_layers}")
    assert model.config.hidden_size == 2048, f"Expected 2048, got {model.config.hidden_size}"
    assert model.config.num_hidden_layers == 48, f"Expected 48, got {model.config.num_hidden_layers}"
    print("  PASS: dimensions correct")

    # 2. Hidden states
    print("\n[2] Testing output_hidden_states...")
    inputs = tok("The bridge collapsed on March 1st due to heavy flooding.", return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states
    print(f"  Hidden states count: {len(hs)} (expected 49)")
    print(f"  Layer 0 shape: {hs[0].shape}")
    print(f"  Layer 24 (mid) shape: {hs[24].shape}")
    print(f"  Layer 48 (last) shape: {hs[48].shape}")
    assert len(hs) == 49, f"Expected 49 hidden states, got {len(hs)}"
    assert hs[24].shape[-1] == 2048, f"Expected d=2048, got {hs[24].shape[-1]}"
    print("  PASS: hidden states work")

    # 3. Hookable attention layers
    print("\n[3] Checking hookable attention layers...")
    lm = model.model
    hookable_count = 0
    for i, layer in enumerate(lm.layers):
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            attn_type = type(attn).__name__
            if i < 3 or i == 24 or i == 47:
                print(f"  Layer {i}: {attn_type}")
                # Check for q_proj, k_proj, v_proj
                has_qkv = all(hasattr(attn, p) for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
                print(f"    Has Q/K/V/O: {has_qkv}")
            hookable_count += 1
    print(f"  Hookable attention layers: {hookable_count}/48")
    assert hookable_count == 48, f"Expected 48 hookable layers, got {hookable_count}"
    print("  PASS: all layers hookable")

    # 4. Generation test
    print("\n[4] Testing generation...")
    messages = [
        {"role": "user", "content": "What is 2+2? Answer in one word."}
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    gen_inputs = tok(text, return_tensors='pt').to('cuda')

    t0 = time.time()
    with torch.no_grad():
        gen_out = model.generate(
            **gen_inputs,
            max_new_tokens=20,
            do_sample=False,
        )
    gen_time = time.time() - t0
    response = tok.decode(gen_out[0][gen_inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Response: \"{response}\"")
    print(f"  Generation time: {gen_time:.2f}s")
    assert len(response.strip()) > 0, "Empty response!"
    print("  PASS: generation works")

    # 5. Delta extraction test
    print("\n[5] Testing delta extraction from layer 24...")
    test_text = "The bridge collapsed on March 1st. Trucks were rerouted through downtown."
    test_inputs = tok(test_text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        test_out = model(**test_inputs, output_hidden_states=True)

    h = test_out.hidden_states[24]  # [1, N, 2048]
    N = h.shape[1]

    # Compute deltas
    h_prev = torch.cat([h[:, :1, :], h[:, :-1, :]], dim=1)
    delta_h = h - h_prev  # [1, N, 2048]

    # RMS normalization
    delta_h = delta_h.float()
    delta_h = delta_h - delta_h.mean(dim=1, keepdim=True)
    rms = delta_h.pow(2).mean().sqrt().clamp(min=1e-8)
    delta_h = delta_h / rms

    # Check heavy-tailed distribution
    norms = delta_h[0].norm(dim=-1)
    p50 = norms.median().item()
    p90 = norms.quantile(0.9).item()
    ratio = p90 / max(p50, 1e-8)

    token_texts = [tok.decode([tid]) for tid in test_inputs.input_ids[0].tolist()]
    print(f"  Tokens: {N}")
    print(f"  Delta norms — p50: {p50:.3f}, p90: {p90:.3f}, p90/p50: {ratio:.1f}")
    print(f"  Top-5 delta tokens: ", end="")
    top_idx = norms.argsort(descending=True)[:5]
    for idx in top_idx:
        print(f"\"{token_texts[idx]}\"({norms[idx]:.2f}) ", end="")
    print()

    if ratio > 2.0:
        print("  PASS: heavy-tailed delta distribution (content > function words)")
    else:
        print(f"  WARN: ratio {ratio:.1f} — may need RMS norm check")

    # 6. Hook injection test
    print("\n[6] Testing attention hook injection...")
    hook_fired = [0]

    def test_hook(module, args, kwargs):
        hook_fired[0] += 1

    # Hook middle third (layers 16-31)
    hooks = []
    for i in range(16, 32):
        h = lm.layers[i].self_attn.register_forward_pre_hook(test_hook, with_kwargs=True)
        hooks.append(h)

    with torch.no_grad():
        _ = model(**test_inputs)

    for h in hooks:
        h.remove()

    print(f"  Hooks fired: {hook_fired[0]} times (expected {16 * 1} = 16)")
    assert hook_fired[0] == 16, f"Expected 16 hook fires, got {hook_fired[0]}"
    print("  PASS: hooks work")

    # 7. Multi-turn reasoning test
    print("\n[7] Testing multi-turn reasoning quality...")
    messages = [
        {"role": "user", "content": (
            "A bridge collapsed on March 1st. Because of this, trucks were rerouted "
            "through downtown. The downtown rerouting caused supply disruptions at "
            "the warehouse. What was the root cause of the supply disruptions?"
        )}
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    gen_inputs = tok(text, return_tensors='pt').to('cuda')

    with torch.no_grad():
        gen_out = model.generate(
            **gen_inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    response = tok.decode(gen_out[0][gen_inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  Response: \"{response[:200]}\"")

    if "bridge" in response.lower():
        print("  PASS: correctly identified root cause (bridge collapse)")
    else:
        print("  WARN: root cause not explicitly mentioned — check response quality")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Model: Qwen3-30B-A3B ({model.config.num_hidden_layers} layers, d={model.config.hidden_size})")
    print(f"  GPU memory: {mem_gb:.1f} GB model, {peak_mem:.1f} GB peak")
    print(f"  Headroom for ODE: {130.7 - peak_mem:.1f} GB")
    print(f"  Hidden states: {len(hs)} layers, all hookable")
    print(f"  Delta extraction: layer 24, d=2048")
    print(f"  Bias injection: layers 16-31 (middle third)")
    print(f"  Generation: working")
    print("=" * 60)


if __name__ == '__main__':
    main()
