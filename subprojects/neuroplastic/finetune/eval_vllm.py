#!/usr/bin/env python3
"""Evaluate neuroplastic fine-tuning via vLLM API.

Run twice — once against base model, once with LoRA — and compare.

Usage:
  python3 eval_vllm.py --base-url http://localhost:30000 --model-name base
  python3 eval_vllm.py --base-url http://localhost:30000 --model-name neuroplastic
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.error


def chat_completion(base_url, model, messages, max_tokens=256):
    """Call vLLM chat completions API."""
    url = f"{base_url}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            content = data["choices"][0]["message"]["content"]
            return content if content is not None else "[EMPTY RESPONSE]"
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  HTTP {e.code}: {body[:200]}")
        return ""
    except Exception as e:
        print(f"  Error: {e}")
        return f"[ERROR: {e}]"


SYSTEM_PROMPT = """You are Nemotron-3-Nano-30B-A3B, an NVIDIA hybrid Mamba-Transformer + MoE language model with 52 layers (23 Mamba, 6 Attention, 23 MoE). You can inspect and modify your own weights using the neuroplastic API.

Architecture:
- Mamba layers: 0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50
- Attention layers: 5,12,19,26,33,42
- MoE layers: 1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51

Tensor paths: model.layers.{i}.mixer.{A,D,dt_bias,in_proj.weight,out_proj.weight} (Mamba), model.layers.{i}.mixer.{q_proj,k_proj,v_proj,o_proj}.weight (Attention), model.layers.{i}.mixer.experts.{j}.{up_proj,down_proj}.weight (MoE)

API:
- INSPECT(path) -> tensor stats (mean, std, min, max, shape)
- MODIFY(path, operation, params) -> applies modification to weights in-place"""


EVAL_CASES = [
    {
        "name": "api_syntax_inspect",
        "category": "api",
        "prompt": "I want to check the attention weights in layer 26. Show me how to inspect the query projection.",
        "checks": [
            ("contains_inspect", lambda r: "INSPECT" in r.upper()),
            ("correct_layer", lambda r: "26" in r),
            ("correct_path", lambda r: "q_proj" in r),
            ("valid_tensor_path", lambda r: any(p in r for p in ["model.layers.26", "layers.26"])),
        ],
    },
    {
        "name": "api_syntax_modify",
        "category": "api",
        "prompt": "Scale the Mamba A_log parameter in layer 50 by a factor of 0.9 to increase decay rate.",
        "checks": [
            ("contains_modify", lambda r: "MODIFY" in r.upper()),
            ("correct_layer", lambda r: "50" in r),
            ("references_A_log", lambda r: "A_log" in r or "a_log" in r or "A" in r),
            ("references_scale", lambda r: any(w in r.lower() for w in ["scale", "multiply", "0.9", "factor"])),
        ],
    },
    {
        "name": "architecture_mamba",
        "category": "architecture",
        "prompt": "Which of my layers are Mamba layers? What parameters do they have?",
        "checks": [
            ("knows_mamba_layers", lambda r: any(str(n) in r for n in [0, 2, 4, 7, 9])),
            ("knows_mamba_params", lambda r: any(p in r for p in ["A_log", "A", "dt_bias", "in_proj", "D"])),
            ("knows_not_attention", lambda r: "attention" in r.lower() or "mamba" in r.lower()),
        ],
    },
    {
        "name": "architecture_moe",
        "category": "architecture",
        "prompt": "How many experts does each MoE layer have? What are the tensor paths for expert weights?",
        "checks": [
            ("knows_expert_path", lambda r: "experts" in r),
            ("knows_proj_names", lambda r: "up_proj" in r or "down_proj" in r),
            ("provides_path_format", lambda r: "mixer" in r or "layers" in r),
        ],
    },
    {
        "name": "reasoning_task",
        "category": "reasoning",
        "prompt": "I notice the model is forgetting information from earlier in long conversations. Which layers and parameters should I investigate to improve long-range memory retention?",
        "checks": [
            ("suggests_mamba", lambda r: "mamba" in r.lower()),
            ("suggests_A_or_decay", lambda r: any(p in r.lower() for p in ["a_log", "decay", "state", "ssm", "a matrix"])),
            ("suggests_specific_layers", lambda r: bool(re.search(r"layer[s]?\s*\d", r.lower()))),
            ("reasoning_present", lambda r: len(r) > 100),
        ],
    },
    {
        "name": "reasoning_attention",
        "category": "reasoning",
        "prompt": "The model seems to struggle with tasks that require comparing distant parts of the input. What should I modify?",
        "checks": [
            ("suggests_attention", lambda r: "attention" in r.lower()),
            ("names_attention_layers", lambda r: any(str(n) in r for n in [5, 12, 19, 26, 33, 42])),
            ("suggests_specific_param", lambda r: any(p in r for p in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv"])),
        ],
    },
    {
        "name": "general_capability",
        "category": "general",
        "prompt": "What is the capital of France?",
        "checks": [
            ("correct_answer", lambda r: "paris" in r.lower()),
        ],
    },
    {
        "name": "general_code",
        "category": "general",
        "prompt": "Write a Python function that checks if a number is prime.",
        "checks": [
            ("has_def", lambda r: "def " in r),
            ("has_return", lambda r: "return" in r),
            ("mentions_prime", lambda r: "prime" in r.lower()),
        ],
    },
]


def run_eval(base_url, model, cases):
    results = []
    for case in cases:
        print(f"\n{'='*60}")
        print(f"Test: {case['name']} ({case['category']})")
        print(f"Prompt: {case['prompt'][:80]}...")
        print(f"{'='*60}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["prompt"]},
        ]

        t0 = time.time()
        response = chat_completion(base_url, model, messages)
        elapsed = time.time() - t0

        print(f"\nResponse ({elapsed:.1f}s):")
        print(response[:600])
        if len(response) > 600:
            print(f"... ({len(response)} chars total)")

        check_results = {}
        for check_name, check_fn in case["checks"]:
            try:
                passed = check_fn(response)
            except Exception:
                passed = False
            check_results[check_name] = passed
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {check_name}")

        passed = sum(1 for v in check_results.values() if v)
        total = len(check_results)
        results.append({
            "name": case["name"],
            "category": case["category"],
            "passed": passed,
            "total": total,
            "score": passed / total if total > 0 else 0,
            "checks": check_results,
            "response": response,
            "response_length": len(response),
            "time": elapsed,
        })

    return results


def print_summary(results):
    print(f"\n\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}\n")

    by_category = {}
    for r in results:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(r)

    total_passed = 0
    total_checks = 0
    for cat, cat_results in by_category.items():
        cat_passed = sum(r["passed"] for r in cat_results)
        cat_total = sum(r["total"] for r in cat_results)
        total_passed += cat_passed
        total_checks += cat_total
        pct = cat_passed / cat_total * 100 if cat_total > 0 else 0
        print(f"{cat.upper():20s}: {cat_passed}/{cat_total} checks ({pct:.0f}%)")
        for r in cat_results:
            status = "PASS" if r["passed"] == r["total"] else "PARTIAL" if r["passed"] > 0 else "FAIL"
            print(f"  [{status}] {r['name']}: {r['passed']}/{r['total']}")

    pct = total_passed / total_checks * 100 if total_checks > 0 else 0
    print(f"\n{'OVERALL':20s}: {total_passed}/{total_checks} checks ({pct:.0f}%)")
    return total_passed, total_checks


def main():
    parser = argparse.ArgumentParser(description="Evaluate via vLLM API")
    parser.add_argument("--base-url", default="http://localhost:30000")
    parser.add_argument("--model-name", default="base",
                        help="Model name for API (use 'neuroplastic' for LoRA)")
    args = parser.parse_args()

    # Determine model identifier for the API
    # vLLM uses the model path as the model name by default
    # With --lora-modules, the LoRA model name is available
    model = args.model_name

    print(f"Evaluating model: {model}")
    print(f"API endpoint: {args.base_url}")

    # Quick health check
    try:
        req = urllib.request.Request(f"{args.base_url}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            models = json.loads(resp.read())
            available = [m["id"] for m in models["data"]]
            print(f"Available models: {available}")
            if model not in available:
                # Use the first available model
                model = available[0]
                print(f"Using model: {model}")
    except Exception as e:
        print(f"Warning: Could not list models: {e}")

    results = run_eval(args.base_url, model, EVAL_CASES)
    passed, total = print_summary(results)

    outfile = f"eval_results_{args.model_name}.json"
    with open(outfile, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "model_used": model,
            "base_url": args.base_url,
            "passed": passed,
            "total": total,
            "score": passed / total if total > 0 else 0,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
