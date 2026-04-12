"""Verify Layer-Wise ODE co-processing with Qwen3-4B.

Phase 1 proof of concept:
  1. Load Qwen3-4B + trained ContinuousDynamics
  2. Hook all 36 layers with LayerWiseODE
  3. Verify ODE processes at every layer
  4. Measure: CV per layer, D² per layer, bias range per layer
  5. Generate with layer-wise co-processing
  6. Run causal chain test
  7. Compare: plain generation vs layer-wise generation

Run in fgn-train container on DGX Spark:
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/verify_layerwise.py
"""

import argparse
import time
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/workspace/models/qwen3-4b')
    parser.add_argument('--ode_checkpoint', type=str,
                        default='/workspace/liquid-arc/output_2048/checkpoints/best.pt')
    parser.add_argument('--config', type=str,
                        default='/workspace/liquid-arc/configs/mind_layerwise.yaml')
    parser.add_argument('--skip_generate', action='store_true')
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.layer_wise_ode import LayerWiseODE, LayerWiseBridge, hook_llm_layers

    print("=" * 70)
    print("LAYER-WISE ODE CO-PROCESSING — Phase 1 Verification")
    print("=" * 70)

    # ── 1. Load config ──
    print("\n[1] Loading config...")
    config = LiquidARCConfig.from_yaml(args.config)
    print(f"  d_model={config.d_model}, sensory_alpha={config.sensory_alpha}")

    # ── 2. Load Qwen3-4B ──
    print("\n[2] Loading Qwen3-4B...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map='cuda',
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()
    load_time = time.time() - t0
    n_layers = llm.config.num_hidden_layers
    d_llm = llm.config.hidden_size
    mem_gb = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded in {load_time:.1f}s, {mem_gb:.1f} GB")
    print(f"  {n_layers} layers, d={d_llm}")

    # ── 3. Load ODE dynamics ──
    print("\n[3] Loading ODE dynamics...")
    dynamics = ContinuousDynamics(config).to('cuda').to(torch.bfloat16)
    context_pool = ContextPool(config).to('cuda').to(torch.bfloat16)

    ckpt = torch.load(args.ode_checkpoint, map_location='cuda', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    cleaned = {k.replace("_orig_mod.", "").replace('metric_net_linear2.', 'metric_net_linear2_diag.'): v
               for k, v in state_dict.items()}
    dyn_keys = {k: v for k, v in cleaned.items()
                if k.startswith('dynamics.') or k.startswith('context_pool.')}
    holder = nn.ModuleDict({'dynamics': dynamics, 'context_pool': context_pool})
    missing, unexpected = holder.load_state_dict(dyn_keys, strict=False)
    print(f"  Loaded {len(dyn_keys)} keys ({len(missing)} missing)")
    dynamics.eval()
    dynamics.freeze_tau = False

    # ── 4. Create LayerWiseODE ──
    print("\n[4] Creating LayerWiseODE...")
    d_ode = config.d_model
    print(f"  LLM d={d_llm}, ODE d={d_ode}" +
          (f" (projection needed)" if d_llm != d_ode else " (native match)"))
    layer_ode = LayerWiseODE(
        dynamics=dynamics, context_pool=context_pool,
        n_layers=n_layers, d_llm=d_llm, d_ode=d_ode, device='cuda')
    print(f"  {n_layers} layers")

    # ── 5. Hook test: forward pass ──
    print("\n[5] Testing forward pass with hooks...")
    hooks = hook_llm_layers(llm, layer_ode)
    print(f"  Registered {len(hooks)} hooks")

    test_text = "The bridge collapsed on March 1st. Trucks were rerouted through downtown."
    inputs = tokenizer(test_text, return_tensors='pt').to('cuda')
    n_tokens = inputs['input_ids'].shape[1]

    layer_ode.start_forward()
    t0 = time.time()
    with torch.no_grad():
        _ = llm(**inputs)
    layer_ode.end_forward()
    fwd_time = time.time() - t0

    diag = layer_ode.get_layer_diagnostics()
    summary = layer_ode.get_summary()
    print(f"  Forward: {fwd_time:.2f}s, {summary['n_layers']}/{n_layers} layers processed")
    assert summary['n_layers'] == n_layers, \
        f"Expected {n_layers} layers, got {summary['n_layers']}"
    print(f"  PASS: all {n_layers} layers processed")

    # ── 6. Per-layer diagnostics ──
    print(f"\n[6] Per-layer diagnostics ({n_tokens} tokens):")
    print(f"  {'Layer':>6} {'CV':>8} {'tau':>8} {'B_range':>10} {'B_std':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    for d in diag:
        idx = d['layer_idx']
        if idx < 3 or idx == n_layers // 2 or idx >= n_layers - 2 or idx % 6 == 0:
            print(f"  {idx:>6} {d['cv']:>8.3f} {d['tau_mean']:>8.3f} "
                  f"{d['B_range']:>10.2f} {d['B_std']:>8.3f}")

    print(f"\n  Depth summary:")
    print(f"    CV:      early={summary['cv_early']:.3f}  mid={summary['cv_mid']:.3f}  "
          f"late={summary['cv_late']:.3f}")
    print(f"    B_range: early={summary['B_range_early']:.2f}  "
          f"late={summary['B_range_late']:.2f}")
    print(f"    tau: {summary['tau_mean']:.3f}")

    # Remove test hooks
    for h in hooks:
        h.remove()

    # ── 7. Generation tests ──
    if not args.skip_generate:
        print("\n[7] Generation tests...")

        bridge = LayerWiseBridge(llm=llm, tokenizer=tokenizer, layer_ode=layer_ode)

        # 7a: Simple
        print("\n  7a. Simple question:")
        result = bridge.generate("What is 2+2? Answer in one word.",
                                 max_new_tokens=30, temperature=0.1)
        print(f"    Response: \"{result['response']}\"")

        # 7b: Causal chain (THE key test from spec)
        print("\n  7b. Causal chain reasoning:")
        result = bridge.generate(
            "A bridge collapsed on March 1st. Because of this, trucks were rerouted "
            "through downtown. The downtown rerouting caused supply disruptions at "
            "the warehouse. What was the root cause of the supply disruptions?",
            max_new_tokens=100, temperature=0.7)
        response = result['response']
        print(f"    Response: \"{response[:200]}\"")
        gen_summary = result['diagnostics']
        if gen_summary:
            print(f"    CV: early={gen_summary.get('cv_early', 0):.3f} "
                  f"mid={gen_summary.get('cv_mid', 0):.3f} "
                  f"late={gen_summary.get('cv_late', 0):.3f}")
        if 'bridge' in response.lower():
            print(f"    PASS: root cause identified")
        else:
            print(f"    WARN: root cause not mentioned")

        # 7c: Plain generation for comparison
        print("\n  7c. Plain generation (no ODE):")
        layer_ode._active = False
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer directly."},
            {"role": "user", "content": (
                "A bridge collapsed on March 1st. Because of this, trucks were rerouted "
                "through downtown. The downtown rerouting caused supply disruptions at "
                "the warehouse. What was the root cause of the supply disruptions?"
            )},
        ]
        try:
            plain_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            plain_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

        plain_inputs = tokenizer(plain_prompt, return_tensors='pt',
                                 truncation=True, max_length=2048).to('cuda')
        with torch.no_grad():
            plain_out = llm.generate(
                **plain_inputs, max_new_tokens=100,
                temperature=0.7, do_sample=True, top_p=0.9,
                repetition_penalty=1.2)
        plain_text = tokenizer.decode(
            plain_out[0][plain_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True)
        print(f"    Plain: \"{plain_text[:200]}\"")

        bridge.remove_hooks()
    else:
        print("\n[7] Skipped (--skip_generate)")

    # ── 8. Resources ──
    print(f"\n[8] Resources:")
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  GPU peak: {peak_mem:.1f} GB")
    print(f"  Forward overhead: {fwd_time:.2f}s ({fwd_time/max(n_tokens,1)*1000:.1f} ms/tok)")

    print("\n" + "=" * 70)
    print(f"Layer-Wise ODE: {n_layers} layers, d={d_llm}, "
          f"step {ckpt.get('step', '?')}")
    print("=" * 70)


if __name__ == '__main__':
    main()
