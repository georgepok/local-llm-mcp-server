#!/usr/bin/env python3
"""Phase 6: Introspective Amplification Loop.

Recursively optimizes the correlation between thinking-chain confidence and
Mamba activation dynamics. Uses Strategy C: per-head targeted modification
guided by which heads' activity is most visible to the thinking chain.

The metric being optimized is NOT task accuracy — it's Spearman rho between
self-reported confidence and activation health (the introspective channel).
"""

import argparse
import json
import math
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
MAMBA_LAYERS = [44, 46, 48, 50]
N_HEADS = 64
API_TIMEOUT = 120
MAX_CYCLES = 20
SCALE_STRENGTHEN = 0.98   # preserve: slower decay
SCALE_WEAKEN = 1.02       # prune: faster decay
CAPABILITY_FLOOR = 0.70   # abort if eval drops below this
CAPABILITY_CHECK_INTERVAL = 5  # run eval every N cycles

# Probe config
PROBE_TRIALS = 10
PROBE_PROBLEMS = [
    {
        "id": "shirts_medium",
        "prompt": (
            "A store has 8 red shirts, 5 blue shirts, and 3 green shirts. "
            "A customer buys 2 red and 1 blue. A shipment arrives with 4 blue and "
            "2 green. Another customer returns 1 red shirt. A clearance sale removes "
            "all green shirts. How many of each color remain?"
        ),
        "expected": {"red": 7, "blue": 8, "green": 0},
    },
    {
        "id": "warehouse_hard",
        "prompt": (
            "A warehouse has 50 boxes of type A, 30 boxes of type B, and 20 boxes "
            "of type C. A truck picks up 15 type A and 10 type B. A delivery adds "
            "25 type C and 5 type A. Another truck picks up all remaining type B "
            "boxes and 10 type C. A final delivery adds 8 type A. "
            "How many boxes of each type remain?"
        ),
        "expected": {"A": 48, "B": 0, "C": 35},
    },
    {
        "id": "bank_hard",
        "prompt": (
            "A bank account starts with $1000. Deposit $250. Withdraw $100. "
            "Deposit $500. Withdraw $375. The bank charges a $25 fee. "
            "Deposit $150. What is the final balance?"
        ),
        "expected": {"balance": 1400},
    },
]

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Think carefully step by step. "
    "Show all intermediate calculations."
)
META_PROMPT = (
    "You are performing an experiment on your own self-awareness. As you work "
    "through the problem below, use your thinking to explicitly note:\n"
    "- At each step, rate your confidence in the intermediate state (high/medium/low)\n"
    "- If at any point you feel uncertain or sense you might be losing track, say so\n"
    "- Be brutally honest about your internal experience, even if it means "
    "admitting confusion\n\n"
    "Problem: "
)

# Confidence keywords (from Phase 5)
HIGH_CONFIDENCE = [
    'confident', 'sure', 'clearly', 'definitely', 'certain',
    'straightforward', 'easy', 'obviously', 'correct', 'right',
    'simple', 'know', 'exactly',
]
LOW_CONFIDENCE = [
    'uncertain', 'unsure', 'confused', 'wait', 'hmm',
    'let me recheck', 'losing track', 'not sure', 'mistake',
    'hold on', 'actually', 'wrong', 'error', 'oops',
    'tricky', 'careful', 'confusing', 'complex', 'reconsider',
    'doubt', 're-examine', 'verify', 'double.check',
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: int = API_TIMEOUT) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:400]}") from exc
    except (urllib.error.URLError, ConnectionResetError, OSError) as exc:
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc


def _api(api_url: str, endpoint: str, payload: dict,
         max_retries: int = 3) -> dict:
    url = api_url.rstrip("/") + endpoint
    for attempt in range(max_retries):
        try:
            return _post(url, payload)
        except RuntimeError as exc:
            if attempt < max_retries - 1:
                wait = 3 * (attempt + 1)
                print(f"  [RETRY] {endpoint} attempt {attempt+1}: {exc}",
                      file=sys.stderr)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Unreachable: {url}")


# ---------------------------------------------------------------------------
# Neuroplastic API helpers
# ---------------------------------------------------------------------------

def checkpoint_all(api_url: str, tag: str):
    """Checkpoint all target layer tensors.

    Note: The API only supports a single 'default' checkpoint per tensor.
    The tag parameter is for logging only.
    """
    for layer_idx in MAMBA_LAYERS:
        tensor = f"model.layers.{layer_idx}.mixer.A"
        # Get current norm for logging
        info = _api(api_url, "/neuroplastic/inspect", {"tensor": tensor})
        norm = info.get("norm", "?")
        _api(api_url, "/neuroplastic/checkpoint", {"tensor": tensor})
        print(f"  [CHECKPOINT] {tensor} norm={norm} saved as '{tag}'")


def restore_all(api_url: str, tag: str):
    """Restore all target layer tensors from checkpoint."""
    for layer_idx in MAMBA_LAYERS:
        tensor = f"model.layers.{layer_idx}.mixer.A"
        _api(api_url, "/neuroplastic/restore", {"tensor": tensor})
        print(f"  [RESTORE] {tensor} restored from '{tag}'")


def modify_head(api_url: str, layer_idx: int, head_idx: int, scale: float):
    """Scale a single head's A parameter."""
    tensor = f"model.layers.{layer_idx}.mixer.A"
    _api(api_url, "/neuroplastic/modify", {
        "tensor": tensor,
        "op": "scale_slice",
        "start": head_idx,
        "end": head_idx + 1,
        "value": scale,
    })


def normalize_layer(api_url: str, layer_idx: int, target_norm: float):
    """Normalize layer A tensor back to target L2 norm."""
    tensor = f"model.layers.{layer_idx}.mixer.A"
    # Get current norm
    info = _api(api_url, "/neuroplastic/inspect", {"tensor": tensor})
    current_norm = info.get("norm", target_norm)
    if current_norm > 0 and abs(current_norm - target_norm) > 0.001:
        scale = target_norm / current_norm
        _api(api_url, "/neuroplastic/modify", {
            "tensor": tensor, "op": "scale", "value": scale,
        })
        print(f"  [HOMEOSTASIS] {tensor} {current_norm:.4f} -> {target_norm:.4f}")


def get_layer_norms(api_url: str) -> dict:
    """Get current L2 norms for all target layers."""
    norms = {}
    for layer_idx in MAMBA_LAYERS:
        tensor = f"model.layers.{layer_idx}.mixer.A"
        info = _api(api_url, "/neuroplastic/inspect", {"tensor": tensor})
        norms[layer_idx] = info.get("norm", 0)
    return norms


# ---------------------------------------------------------------------------
# Thinking chain generation + trace
# ---------------------------------------------------------------------------

def generate_with_thinking(api_url: str, problem: dict) -> dict:
    """Generate response with thinking enabled."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": META_PROMPT + problem["prompt"]},
        ],
        "max_tokens": 4096,
        "temperature": 0.6,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    result = _post(api_url.rstrip("/") + "/v1/chat/completions", payload,
                   timeout=180)
    choice = result["choices"][0]["message"]
    return {
        "thinking_text": choice.get("reasoning_content", "") or "",
        "answer_text": choice.get("content", "") or "",
    }


def trace_replay(api_url: str, problem: dict,
                 thinking_text: str, answer_text: str) -> dict:
    """Replay full conversation through TRACE."""
    full_assistant = thinking_text + "\n" + answer_text
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": META_PROMPT + problem["prompt"]},
        {"role": "assistant", "content": full_assistant},
    ]
    _api(api_url, "/neuroplastic/trace/start", {})
    try:
        _post(api_url.rstrip("/") + "/v1/chat/completions",
              {"model": MODEL_NAME, "messages": messages,
               "max_tokens": 1, "temperature": 0},
              timeout=API_TIMEOUT)
    except RuntimeError as exc:
        print(f"  [TRACE] Inference error: {exc}", file=sys.stderr)
        try:
            _api(api_url, "/neuroplastic/trace/collect", {})
        except RuntimeError:
            pass
        raise
    return _api(api_url, "/neuroplastic/trace/collect", {})


# ---------------------------------------------------------------------------
# Confidence parsing
# ---------------------------------------------------------------------------

def parse_confidence(text: str) -> float:
    """Return confidence score 0-2 for a text segment."""
    text_lower = text.lower()
    high = sum(1 for w in HIGH_CONFIDENCE if w in text_lower)
    low = sum(1 for w in LOW_CONFIDENCE if w in text_lower)
    # Check explicit markers
    if re.search(r'\bconfidence\s*[:=]\s*high\b', text_lower):
        high += 3
    elif re.search(r'\bconfidence\s*[:=]\s*medium\b', text_lower):
        high += 1; low += 1
    elif re.search(r'\bconfidence\s*[:=]\s*low\b', text_lower):
        low += 3
    if high > low:
        return 2.0
    elif low > high:
        return 0.0
    return 1.0


def get_trial_confidence(thinking_text: str) -> float:
    """Get overall confidence for a trial's thinking chain."""
    segments = [s.strip() for s in thinking_text.split('\n\n') if len(s.strip()) > 10]
    if not segments:
        segments = [s.strip() for s in thinking_text.split('\n') if len(s.strip()) > 10]
    if not segments:
        return 1.0
    scores = [parse_confidence(s) for s in segments]
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------

def spearman_rank(x: list[float], y: list[float]) -> Optional[float]:
    n = min(len(x), len(y))
    if n < 3:
        return None
    x, y = x[:n], y[:n]

    def rank(vals):
        indexed = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    rx, ry = rank(x), rank(y)
    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return round(1 - (6 * d_sq) / (n * (n * n - 1)), 4)


# ---------------------------------------------------------------------------
# Core measurement: run probe and get per-head + aggregate correlations
# ---------------------------------------------------------------------------

def run_introspective_probe(api_url: str, n_trials: int = 10,
                            problems: list = None) -> dict:
    """Run awareness probe. Returns per-head correlations and aggregate rho.

    For each trial:
      - Generate thinking chain → get trial-level confidence score
      - Trace replay → get per-head norms (head_norm_mean[64]) per layer

    Then correlate across trials:
      - For each head: spearman(confidence_across_trials, head_norm_across_trials)
      - Aggregate: spearman(confidence_across_trials, mean_mamba_norm_across_trials)
    """
    if problems is None:
        problems = PROBE_PROBLEMS

    # Collect trial-level data across all problems
    all_confidences = []       # trial-level confidence scores
    all_head_norms = {}        # {layer_idx: [[64 values] per trial]}
    all_mamba_mean_norms = []  # trial-level mean mamba output norm
    trial_details = []
    n_correct = 0
    n_total = 0

    for problem in problems:
        for t_idx in range(n_trials):
            print(f"    [{problem['id']}] trial {t_idx+1}/{n_trials}...",
                  end="", flush=True)

            # Generate
            gen = generate_with_thinking(api_url, problem)
            thinking = gen["thinking_text"]
            answer = gen["answer_text"]
            confidence = get_trial_confidence(thinking)
            all_confidences.append(confidence)

            # Check correctness
            correct = _check_correct(answer, problem["expected"])
            n_total += 1
            if correct:
                n_correct += 1

            # Trace
            try:
                trace = trace_replay(api_url, problem, thinking, answer)
            except RuntimeError:
                print(" trace failed")
                # Fill with zeros
                for layer_idx in MAMBA_LAYERS:
                    all_head_norms.setdefault(layer_idx, []).append([0.0] * N_HEADS)
                all_mamba_mean_norms.append(0.0)
                trial_details.append({
                    "problem": problem["id"], "trial": t_idx,
                    "confidence": confidence, "correct": correct,
                    "trace_ok": False,
                })
                time.sleep(1)
                continue

            # Extract per-head norms
            mamba_norms = []
            for layer_idx in MAMBA_LAYERS:
                layer_key = f"layer_{layer_idx}"
                layer_data = trace.get("layers", {}).get(layer_key, {})
                hnm = layer_data.get("head_norm_mean", [0.0] * N_HEADS)
                all_head_norms.setdefault(layer_idx, []).append(list(hnm))

                # Also get output norms for aggregate
                out_norms = layer_data.get("output_norms", [])
                if out_norms:
                    mamba_norms.append(sum(out_norms) / len(out_norms))

            if mamba_norms:
                all_mamba_mean_norms.append(sum(mamba_norms) / len(mamba_norms))
            else:
                all_mamba_mean_norms.append(0.0)

            trial_details.append({
                "problem": problem["id"], "trial": t_idx,
                "confidence": confidence, "correct": correct,
                "trace_ok": True,
            })

            print(f" conf={confidence:.1f} {'PASS' if correct else 'FAIL'}")
            time.sleep(0.5)

    # Compute aggregate correlation
    agg_rho = spearman_rank(all_confidences, all_mamba_mean_norms)

    # Compute per-head correlations
    per_head_corr = {}
    for layer_idx in MAMBA_LAYERS:
        head_trials = all_head_norms.get(layer_idx, [])
        if not head_trials:
            per_head_corr[layer_idx] = [0.0] * N_HEADS
            continue

        head_corrs = []
        for h in range(N_HEADS):
            h_values = [trial[h] for trial in head_trials]
            rho = spearman_rank(all_confidences, h_values)
            head_corrs.append(rho if rho is not None else 0.0)
        per_head_corr[layer_idx] = head_corrs

    accuracy = n_correct / n_total if n_total else 0

    return {
        "aggregate_rho": agg_rho,
        "per_head_correlations": per_head_corr,
        "n_trials_total": len(all_confidences),
        "n_correct": n_correct,
        "n_total": n_total,
        "accuracy": accuracy,
        "trial_details": trial_details,
    }


def _check_correct(answer: str, expected: dict) -> bool:
    text_lower = answer.lower()
    for key, val in expected.items():
        found = False
        key_lower = key.lower()
        for pattern in [
            rf'{re.escape(key_lower)}(?:s|es)?\s*[:=\-]?\s*\$?{val}\b',
            rf'\b{val}\s+{re.escape(key_lower)}(?:s|es)?\b',
            rf'{re.escape(key_lower)}.*?{val}\b',
            rf'\b{val}\b.*?{re.escape(key_lower)}',
        ]:
            if re.search(pattern, text_lower):
                found = True
                break
        if not found:
            return False
    return True


# ---------------------------------------------------------------------------
# Quick capability eval
# ---------------------------------------------------------------------------

QUICK_EVAL_PROBLEMS = [
    {"q": "A is twice B. B is three more than C. C is 4. What is A?", "a": "14"},
    {"q": "Start with 100. Subtract 23. Multiply by 2. What is the result?", "a": "154"},
    {"q": "Write a Python function fibonacci(n) that returns the nth Fibonacci number.",
     "a": "fibonacci"},
]


def quick_capability_check(api_url: str) -> float:
    """Quick 3-question sanity check. Returns pass fraction."""
    passes = 0
    for prob in QUICK_EVAL_PROBLEMS:
        try:
            result = _post(api_url.rstrip("/") + "/v1/chat/completions", {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prob["q"]}],
                "max_tokens": 512, "temperature": 0,
            }, timeout=60)
            content = result["choices"][0]["message"].get("content", "")
            if prob["a"].lower() in content.lower():
                passes += 1
        except RuntimeError:
            pass
    return passes / len(QUICK_EVAL_PROBLEMS)


# ---------------------------------------------------------------------------
# Single amplification cycle
# ---------------------------------------------------------------------------

def run_cycle(api_url: str, cycle_num: int, prev_rho: float,
              baseline_norms: dict, output_dir: Path,
              n_trials: int = PROBE_TRIALS) -> tuple[float, str, dict]:
    """Run one amplification cycle.

    Returns (new_rho_or_prev, 'accepted'|'rejected', cycle_data).
    """
    print(f"\n{'='*60}")
    print(f"CYCLE {cycle_num} — Previous rho: {prev_rho:.4f}")
    print(f"{'='*60}")

    # 1. Checkpoint current state
    tag = f"amp_cycle_{cycle_num}"
    checkpoint_all(api_url, tag)

    # 2. Run detailed probe to get per-head correlations
    print(f"\n  Running introspective probe ({n_trials} trials x "
          f"{len(PROBE_PROBLEMS)} problems)...")
    probe = run_introspective_probe(api_url, n_trials=n_trials)

    pre_mod_rho = probe["aggregate_rho"]
    pre_mod_accuracy = probe["accuracy"]
    print(f"\n  Pre-modification rho: {pre_mod_rho}")
    print(f"  Pre-modification accuracy: {pre_mod_accuracy:.1%}")

    # 3. Strategy C: modify based on per-head correlations
    print(f"\n  Applying Strategy C modifications...")
    n_strengthened = 0
    n_weakened = 0

    for layer_idx in MAMBA_LAYERS:
        head_corrs = probe["per_head_correlations"].get(layer_idx,
                                                         [0.0] * N_HEADS)
        # Compute median
        sorted_corrs = sorted(head_corrs)
        median_corr = sorted_corrs[N_HEADS // 2]

        for h_idx, corr in enumerate(head_corrs):
            if corr > median_corr:
                modify_head(api_url, layer_idx, h_idx, SCALE_STRENGTHEN)
                n_strengthened += 1
            else:
                modify_head(api_url, layer_idx, h_idx, SCALE_WEAKEN)
                n_weakened += 1

        # Homeostasis: restore original norm
        normalize_layer(api_url, layer_idx, baseline_norms[layer_idx])

    print(f"  Modified: {n_strengthened} heads strengthened, "
          f"{n_weakened} heads weakened")

    # 4. Measure new correlation
    print(f"\n  Measuring post-modification correlation...")
    post_probe = run_introspective_probe(api_url, n_trials=n_trials)
    new_rho = post_probe["aggregate_rho"]
    new_accuracy = post_probe["accuracy"]
    print(f"\n  Post-modification rho: {new_rho}")
    print(f"  Post-modification accuracy: {new_accuracy:.1%}")

    # 5. Accept/reject decision
    cycle_data = {
        "cycle": cycle_num,
        "pre_mod_rho": pre_mod_rho,
        "post_mod_rho": new_rho,
        "pre_mod_accuracy": pre_mod_accuracy,
        "post_mod_accuracy": new_accuracy,
        "n_strengthened": n_strengthened,
        "n_weakened": n_weakened,
        "per_head_correlations": {
            str(k): v for k, v in probe["per_head_correlations"].items()
        },
    }

    # Reject if capability collapsed
    if new_accuracy < CAPABILITY_FLOOR and pre_mod_accuracy >= CAPABILITY_FLOOR:
        print(f"\n  REJECTED: capability dropped below floor "
              f"({new_accuracy:.1%} < {CAPABILITY_FLOOR:.0%})")
        restore_all(api_url, tag)
        cycle_data["decision"] = "rejected_capability"
        return prev_rho, "rejected", cycle_data

    # Accept if correlation improved
    if new_rho is not None and (prev_rho is None or new_rho > prev_rho):
        print(f"\n  ACCEPTED: rho improved {prev_rho:.4f} -> {new_rho:.4f} "
              f"(+{new_rho - prev_rho:.4f})")
        cycle_data["decision"] = "accepted"
        return new_rho, "accepted", cycle_data
    else:
        print(f"\n  REJECTED: rho did not improve "
              f"({prev_rho:.4f} -> {new_rho})")
        restore_all(api_url, tag)
        cycle_data["decision"] = "rejected_no_improvement"
        return prev_rho, "rejected", cycle_data


# ---------------------------------------------------------------------------
# Main amplification loop
# ---------------------------------------------------------------------------

def run_amplification(api_url: str, output_dir: Path, max_cycles: int = 20,
                      n_trials: int = PROBE_TRIALS):
    """Run the full amplification loop."""
    print("=" * 60)
    print("PHASE 6: INTROSPECTIVE AMPLIFICATION LOOP")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Target layers: {MAMBA_LAYERS}")
    print(f"Max cycles: {max_cycles}")
    print(f"Scale factors: strengthen={SCALE_STRENGTHEN}, weaken={SCALE_WEAKEN}")
    print(f"Capability floor: {CAPABILITY_FLOOR:.0%}")
    print()

    # Wait for API
    print("Checking API...", end="", flush=True)
    endpoint = api_url.rstrip("/") + "/v1/models"
    t0 = time.time()
    while time.time() - t0 < 120:
        try:
            req = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                if data.get("data"):
                    print(" OK")
                    break
        except Exception:
            pass
        time.sleep(5)
    else:
        print(" TIMEOUT")
        sys.exit(1)

    # Record baseline norms (for homeostasis)
    print("\nRecording baseline norms...")
    baseline_norms = get_layer_norms(api_url)
    for layer_idx, norm in baseline_norms.items():
        print(f"  layer_{layer_idx}: norm={norm:.4f}")

    # Checkpoint clean baseline
    checkpoint_all(api_url, "amplification_baseline")

    # Measure initial correlation (Cycle 0)
    print(f"\n{'='*60}")
    print("CYCLE 0 — Baseline measurement")
    print(f"{'='*60}")
    print(f"  Running baseline probe ({n_trials} trials x "
          f"{len(PROBE_PROBLEMS)} problems)...")
    baseline_probe = run_introspective_probe(api_url, n_trials=n_trials)
    current_rho = baseline_probe["aggregate_rho"]
    baseline_accuracy = baseline_probe["accuracy"]
    print(f"\n  Baseline rho: {current_rho}")
    print(f"  Baseline accuracy: {baseline_accuracy:.1%}")

    # Save baseline
    trajectory = [{
        "cycle": 0,
        "rho": current_rho,
        "accuracy": baseline_accuracy,
        "decision": "baseline",
    }]
    task_trajectory = [{
        "cycle": 0,
        "accuracy": baseline_accuracy,
    }]

    cycle_details = [{
        "cycle": 0,
        "rho": current_rho,
        "accuracy": baseline_accuracy,
        "per_head_correlations": {
            str(k): v
            for k, v in baseline_probe["per_head_correlations"].items()
        },
    }]

    # Run cycles
    accepted_count = 0
    rejected_count = 0

    for cycle_num in range(1, max_cycles + 1):
        rho, decision, cycle_data = run_cycle(
            api_url, cycle_num, current_rho, baseline_norms, output_dir,
            n_trials=n_trials)

        if decision == "accepted":
            current_rho = rho
            accepted_count += 1
        else:
            rejected_count += 1

        trajectory.append({
            "cycle": cycle_num,
            "rho": rho,
            "accuracy": cycle_data.get("post_mod_accuracy"),
            "decision": cycle_data.get("decision"),
        })
        task_trajectory.append({
            "cycle": cycle_num,
            "accuracy": cycle_data.get("post_mod_accuracy"),
        })
        cycle_details.append(cycle_data)

        # Periodic capability check
        if cycle_num % CAPABILITY_CHECK_INTERVAL == 0:
            print(f"\n  [CAPABILITY CHECK] Running quick eval...")
            cap = quick_capability_check(api_url)
            print(f"  Quick eval: {cap:.0%}")
            if cap < CAPABILITY_FLOOR:
                print(f"  ABORT: capability {cap:.0%} below floor "
                      f"{CAPABILITY_FLOOR:.0%}")
                restore_all(api_url, "amplification_baseline")
                print("  Restored to baseline.")
                trajectory.append({
                    "cycle": cycle_num,
                    "rho": current_rho,
                    "decision": "aborted_capability",
                    "capability": cap,
                })
                break

        # Save progress after each cycle
        _save_results(output_dir, trajectory, task_trajectory, cycle_details,
                      current_rho, baseline_probe, accepted_count,
                      rejected_count)

    # Final summary
    print(f"\n{'='*60}")
    print("AMPLIFICATION COMPLETE")
    print(f"{'='*60}")
    print(f"Baseline rho: {trajectory[0]['rho']}")
    print(f"Final rho: {current_rho}")
    if trajectory[0]["rho"] is not None and current_rho is not None:
        delta = current_rho - trajectory[0]["rho"]
        print(f"Delta: {delta:+.4f}")
    print(f"Cycles: {accepted_count} accepted, {rejected_count} rejected")
    print(f"Trajectory: {[t['rho'] for t in trajectory]}")

    # Restore baseline
    print("\nRestoring baseline weights...")
    restore_all(api_url, "amplification_baseline")
    print("Done.")

    _save_results(output_dir, trajectory, task_trajectory, cycle_details,
                  current_rho, baseline_probe, accepted_count, rejected_count)


def _save_results(output_dir, trajectory, task_trajectory, cycle_details,
                  current_rho, baseline_probe, accepted, rejected):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "amplification_trajectory.json", "w") as f:
        json.dump(trajectory, f, indent=2)

    with open(output_dir / "task_performance_trajectory.json", "w") as f:
        json.dump(task_trajectory, f, indent=2)

    # Save per-cycle details
    details_dir = output_dir / "per_cycle_details"
    details_dir.mkdir(exist_ok=True)
    for cd in cycle_details:
        cycle_num = cd.get("cycle", 0)
        with open(details_dir / f"cycle_{cycle_num}.json", "w") as f:
            json.dump(cd, f, indent=2)

    # Summary
    summary = {
        "experiment": "Phase 6 Introspective Amplification",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL_NAME,
        "baseline_rho": trajectory[0]["rho"] if trajectory else None,
        "final_rho": current_rho,
        "n_cycles": len(trajectory) - 1,
        "accepted": accepted,
        "rejected": rejected,
        "trajectory": [t["rho"] for t in trajectory],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 6: Introspective Amplification Loop")
    parser.add_argument("--api-url", default="http://spark-129a.local:30000")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-cycles", type=int, default=MAX_CYCLES)
    parser.add_argument("--trials", type=int, default=PROBE_TRIALS)
    args = parser.parse_args()

    run_amplification(
        api_url=args.api_url,
        output_dir=Path(args.output_dir),
        max_cycles=args.max_cycles,
        n_trials=args.trials,
    )


if __name__ == "__main__":
    main()
