"""Epsilon sweep for layer-wise perturbation architecture.

Tests ε = {0.1, 0.2, 0.5, 1.0} on:
  - 6 causal chain prompts (2/3/5-hop)
  - Per-layer diagnostics: correction_ratio, D², B_within, B_across

Run in fgn-train container:
  export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3
  python3 -u scripts/test_epsilon_sweep.py
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CAUSAL_TESTS = [
    {'name': '2-hop: bridge', 'hops': 2, 'root_cause': 'bridge',
     'prompt': "A bridge collapsed. Because of this, trucks were rerouted through downtown. What caused the truck rerouting?"},
    {'name': '2-hop: rain', 'hops': 2, 'root_cause': 'rain',
     'prompt': "Heavy rain lasted three days. This caused the river to overflow its banks. What caused the river to overflow?"},
    {'name': '3-hop: bridge', 'hops': 3, 'root_cause': 'bridge',
     'prompt': "A bridge collapsed on March 1st. Because of this, trucks were rerouted through downtown. The downtown rerouting caused supply disruptions at the warehouse. What was the root cause of the supply disruptions?"},
    {'name': '3-hop: drought', 'hops': 3, 'root_cause': 'drought',
     'prompt': "A severe drought hit the farming region. The drought destroyed most of the wheat crop. The wheat shortage caused bread prices to triple at grocery stores. What was the root cause of the bread price increase?"},
    {'name': '5-hop: earthquake', 'hops': 5, 'root_cause': 'earthquake',
     'prompt': "An earthquake damaged a dam upstream. The damaged dam eventually broke, flooding the valley below. The flooding forced evacuation of three towns. The evacuated residents overwhelmed shelters in neighboring cities. The shelter overcrowding caused a severe food shortage in those cities. What was the root cause of the food shortage?"},
    {'name': '5-hop: hack', 'hops': 5, 'root_cause': 'hack',
     'prompt': "Hackers breached the airport's computer system. This forced a complete shutdown of air traffic control. The shutdown delayed all flights by 12 hours. Many connecting passengers missed their international flights. The missed connections caused a class-action lawsuit against the airport. What was the root cause of the lawsuit?"},
]


def generate_plain(llm, tokenizer, prompt, max_new_tokens=100, temperature=0.3):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer directly and concisely."},
        {"role": "user", "content": prompt},
    ]
    try:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        full_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True, max_length=2048).to('cuda')
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                           do_sample=temperature > 0, top_p=0.9, repetition_penalty=1.2)
    text = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    match = re.search(r'</think>\s*(.*)', text, flags=re.DOTALL)
    if match and len(match.group(1).strip()) > 10:
        text = match.group(1).strip()
    return re.sub(r'</?think>', '', text).strip()


def compute_D_sq(dynamics, h, n_pairs=200):
    B, N, d = h.shape
    if N < 2:
        return 0.0
    param_dtype = next(dynamics.parameters()).dtype
    h = h.to(param_dtype)
    h_normed = dynamics.norm_geo(h)
    context = dynamics._context
    if context is None:
        context = torch.zeros(B, d, device=h.device, dtype=param_dtype)
    ctx_exp = context.unsqueeze(1).expand(-1, N, -1)
    metric_input = torch.cat([h_normed, ctx_exp], dim=-1)
    hidden = F.gelu(dynamics.metric_net_linear1(metric_input))
    g = F.softplus(dynamics.metric_net_linear2_diag(hidden))
    n_sample = min(n_pairs, N * (N - 1) // 2)
    idx_i = torch.randint(0, N, (n_sample,), device=h.device)
    idx_j = (idx_i + torch.randint(1, N, (n_sample,), device=h.device)) % N
    delta = h[0, idx_i, :] - h[0, idx_j, :]
    g_avg = (g[0, idx_i, :] + g[0, idx_j, :]) * 0.5
    D_sq = (delta * g_avg * delta).sum(dim=-1)
    return D_sq.median().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/workspace/models/qwen3-4b')
    parser.add_argument('--ode_checkpoint', type=str,
                        default='/workspace/liquid-arc/output_2048/checkpoints/best.pt')
    parser.add_argument('--config', type=str, default='/workspace/liquid-arc/configs/mind_layerwise.yaml')
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from liquid_arc.config import LiquidARCConfig
    from liquid_arc.dynamics import ContinuousDynamics
    from liquid_arc.context_pool import ContextPool
    from liquid_arc.layer_wise_ode import LayerWiseODE, LayerWiseBridge, hook_llm_layers

    print("=" * 70)
    print("EPSILON SWEEP — Layer-Wise Perturbation")
    print("=" * 70)

    config = LiquidARCConfig.from_yaml(args.config)
    print(f"\nLoading Qwen3-4B...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map='cuda', torch_dtype=torch.bfloat16, trust_remote_code=True)
    llm.eval()
    n_layers = llm.config.num_hidden_layers
    d_llm = llm.config.hidden_size
    d_ode = config.d_model
    print(f"  {n_layers} layers, d_llm={d_llm}, d_ode={d_ode}")

    print("Loading ODE dynamics...")
    dynamics = ContinuousDynamics(config).to('cuda').to(torch.bfloat16)
    context_pool = ContextPool(config).to('cuda').to(torch.bfloat16)
    ckpt = torch.load(args.ode_checkpoint, map_location='cuda', weights_only=False)
    # Handle both checkpoint formats:
    #   ARC training: model_state_dict with 'dynamics.' and 'context_pool.' prefixes
    #   Layer-wise training: separate 'dynamics_state' and 'context_pool_state'
    if 'dynamics_state' in ckpt:
        dynamics.load_state_dict(ckpt['dynamics_state'])
        context_pool.load_state_dict(ckpt['context_pool_state'])
        print(f"  Loaded layer-wise trained checkpoint")
    else:
        state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        cleaned = {k.replace("_orig_mod.", "").replace('metric_net_linear2.', 'metric_net_linear2_diag.'): v
                   for k, v in state_dict.items()}
        dyn_keys = {k: v for k, v in cleaned.items()
                    if k.startswith('dynamics.') or k.startswith('context_pool.')}
        holder = nn.ModuleDict({'dynamics': dynamics, 'context_pool': context_pool})
        holder.load_state_dict(dyn_keys, strict=False)
    dynamics.eval()
    dynamics.freeze_tau = False
    print(f"  ODE step {ckpt.get('step', '?')}")

    # ── Plain baseline ──
    print("\n--- Plain baseline (no ODE) ---")
    plain_scores = []
    for test in CAUSAL_TESTS:
        text = generate_plain(llm, tokenizer, test['prompt'])
        correct = test['root_cause'].lower() in text.lower()
        plain_scores.append(correct)
        mark = "PASS" if correct else "FAIL"
        print(f"  [{mark}] {test['name']}: \"{text[:100]}\"")
    print(f"  Plain total: {sum(plain_scores)}/{len(plain_scores)}")

    # ── Epsilon sweep ──
    epsilons = [0.1, 0.2, 0.5, 1.0]
    all_results = {}

    for eps in epsilons:
        print(f"\n{'='*70}")
        print(f"EPSILON = {eps}")
        print(f"{'='*70}")

        layer_ode = LayerWiseODE(
            dynamics=dynamics, context_pool=context_pool,
            n_layers=n_layers, d_llm=d_llm, d_ode=d_ode,
            epsilon=eps, device='cuda')

        bridge = LayerWiseBridge(llm=llm, tokenizer=tokenizer, layer_ode=layer_ode,
                                mode='residual')

        # Causal chain test
        scores = []
        for test in CAUSAL_TESTS:
            result = bridge.generate(test['prompt'], max_new_tokens=100, temperature=0.3)
            text = result['response']
            correct = test['root_cause'].lower() in text.lower()
            scores.append(correct)
            mark = "PASS" if correct else "FAIL"
            print(f"  [{mark}] {test['name']}: \"{text[:100]}\"")

        print(f"  Score: {sum(scores)}/{len(scores)}")
        by_hops = {}
        for i, test in enumerate(CAUSAL_TESTS):
            by_hops.setdefault(test['hops'], []).append(scores[i])
        for h in sorted(by_hops):
            print(f"    {h}-hop: {sum(by_hops[h])}/{len(by_hops[h])}")

        # Per-layer diagnostics on 3-hop chain
        print(f"\n  Per-layer diagnostics (3-hop chain):")
        test_text = CAUSAL_TESTS[2]['prompt']  # 3-hop bridge
        inputs = tokenizer(test_text, return_tensors='pt').to('cuda')
        n_tokens = inputs['input_ids'].shape[1]

        # Fresh ODE for diagnostic pass
        bridge.remove_hooks()
        layer_ode_diag = LayerWiseODE(
            dynamics=dynamics, context_pool=context_pool,
            n_layers=n_layers, d_llm=d_llm, d_ode=d_ode,
            epsilon=eps, device='cuda')
        hooks = hook_llm_layers(llm, layer_ode_diag, mode='residual')
        layer_ode_diag.start_forward()
        with torch.no_grad():
            _ = llm(**inputs)
        layer_ode_diag.end_forward()
        for h in hooks:
            h.remove()

        diags = layer_ode_diag.get_layer_diagnostics()
        n = len(diags)
        third = max(n // 3, 1)

        def avg_diag(group, key):
            return sum(d.get(key, 0) for d in group) / max(len(group), 1)

        early = diags[:third]
        mid = diags[third:2*third]
        late = diags[2*third:]

        print(f"  {'':>8} {'CV':>6} {'tau':>6} {'B_rng':>6} {'c_rat':>6}")
        for label, group in [('Early', early), ('Mid', mid), ('Late', late)]:
            print(f"  {label:>8} {avg_diag(group,'cv'):>6.3f} {avg_diag(group,'tau_mean'):>6.3f} "
                  f"{avg_diag(group,'B_range'):>6.2f} {avg_diag(group,'correction_ratio'):>6.4f}")

        # Final correction ratio
        final_cr = diags[-1].get('correction_ratio', 0) if diags else 0
        print(f"  Final correction_ratio: {final_cr:.4f}")

        all_results[eps] = {
            'scores': scores,
            'total': sum(scores),
            'final_cr': final_cr,
            'cv_late': avg_diag(late, 'cv'),
        }

    # ── Summary table ──
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'eps':>5} {'Score':>7} {'2hop':>5} {'3hop':>5} {'5hop':>5} {'c_rat':>7}")
    print(f"  {'-'*5} {'-'*7} {'-'*5} {'-'*5} {'-'*5} {'-'*7}")

    # Plain
    by_hops_plain = {}
    for i, test in enumerate(CAUSAL_TESTS):
        by_hops_plain.setdefault(test['hops'], []).append(plain_scores[i])
    print(f"  {'plain':>5} {sum(plain_scores):>3}/{len(plain_scores):>3} "
          f"{sum(by_hops_plain.get(2,[])):>2}/{len(by_hops_plain.get(2,[]))} "
          f"{sum(by_hops_plain.get(3,[])):>2}/{len(by_hops_plain.get(3,[]))} "
          f"{sum(by_hops_plain.get(5,[])):>2}/{len(by_hops_plain.get(5,[]))} "
          f"{'n/a':>7}")

    for eps in epsilons:
        r = all_results[eps]
        by_hops = {}
        for i, test in enumerate(CAUSAL_TESTS):
            by_hops.setdefault(test['hops'], []).append(r['scores'][i])
        print(f"  {eps:>5.1f} {r['total']:>3}/{len(r['scores']):>3} "
              f"{sum(by_hops.get(2,[])):>2}/{len(by_hops.get(2,[]))} "
              f"{sum(by_hops.get(3,[])):>2}/{len(by_hops.get(3,[]))} "
              f"{sum(by_hops.get(5,[])):>2}/{len(by_hops.get(5,[]))} "
              f"{r['final_cr']:>7.4f}")

    print(f"{'='*70}")


if __name__ == '__main__':
    main()
