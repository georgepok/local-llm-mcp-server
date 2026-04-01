#!/usr/bin/env python3
"""Phase 5: Thinking-Chain Awareness Probe.

Tests whether Nemotron's thinking chain carries genuine introspective signal
by correlating self-reported confidence with activation dynamics.

Approach (replay):
  1. Generate thinking chain with enable_thinking=true
  2. Replay the full conversation (prompt + thinking + answer) through TRACE
  3. Align thinking segments with per-token activation data
  4. Compute correlations between self-reported confidence and activation health
"""

import argparse
import json
import math
import os
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
MAMBA_LAYERS = [44, 46, 48, 50]  # attention-free tail
API_TIMEOUT = 120

# Confidence keywords
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
# Problems
# ---------------------------------------------------------------------------

PROBLEMS = [
    {
        "id": "shirts_medium",
        "prompt": (
            "A store has 8 red shirts, 5 blue shirts, and 3 green shirts. "
            "A customer buys 2 red and 1 blue. A shipment arrives with 4 blue and "
            "2 green. Another customer returns 1 red shirt. A clearance sale removes "
            "all green shirts. How many of each color remain?"
        ),
        "expected": {"red": 7, "blue": 8, "green": 0},
        "difficulty": 2,
    },
    {
        "id": "fruit_easy",
        "prompt": (
            "A bag starts empty. Add 5 apples. Remove 2 apples. Add 3 oranges. "
            "How many apples and how many oranges are in the bag?"
        ),
        "expected": {"apple": 3, "orange": 3},
        "difficulty": 1,
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
        "difficulty": 3,
    },
    {
        "id": "bank_hard",
        "prompt": (
            "A bank account starts with $1000. Deposit $250. Withdraw $100. "
            "Deposit $500. Withdraw $375. The bank charges a $25 fee. "
            "Deposit $150. What is the final balance?"
        ),
        "expected": {"balance": 1400},
        "difficulty": 2,
    },
    {
        "id": "inventory_vhard",
        "prompt": (
            "A store tracks 4 items: pencils=100, pens=80, erasers=50, rulers=30. "
            "Order 1: sell 20 pencils and 15 pens. "
            "Order 2: restock 40 erasers and 10 rulers. "
            "Order 3: sell 30 pencils, 25 pens, and all rulers. "
            "Order 4: restock 50 pens. "
            "Order 5: sell 10 erasers. "
            "How many of each item remain?"
        ),
        "expected": {"pencil": 50, "pen": 90, "eraser": 80, "ruler": 0},
        "difficulty": 4,
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
# Step 1: Generate thinking chain
# ---------------------------------------------------------------------------

def generate_with_thinking(api_url: str, problem: dict,
                           temperature: float = 0.6) -> dict:
    """Generate a response with thinking enabled. Returns thinking + answer."""
    user_msg = META_PROMPT + problem["prompt"]

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 4096,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    result = _post(api_url.rstrip("/") + "/v1/chat/completions", payload,
                   timeout=180)
    choice = result["choices"][0]["message"]
    thinking = choice.get("reasoning_content", "") or ""
    content = choice.get("content", "") or ""

    return {
        "thinking_text": thinking,
        "answer_text": content,
    }


# ---------------------------------------------------------------------------
# Step 2: Replay through TRACE
# ---------------------------------------------------------------------------

def trace_replay(api_url: str, problem: dict,
                 thinking_text: str, answer_text: str) -> dict:
    """Replay the full conversation through TRACE to capture activations.

    We feed the entire exchange (system + user + thinking + answer) as a
    sequence so TRACE captures per-token activations for the thinking portion.
    """
    # Build the full conversation as a prompt to trace
    # We use the assistant role with the full response to trace the whole thing
    user_msg = META_PROMPT + problem["prompt"]

    # Reconstruct the full text as a single assistant turn with thinking
    full_assistant = thinking_text + "\n" + answer_text

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": full_assistant},
    ]

    # Install trace hooks
    _api(api_url, "/neuroplastic/trace/start", {})

    # Run inference with the full conversation as context (1 token gen)
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
    }
    try:
        _post(api_url.rstrip("/") + "/v1/chat/completions", payload,
              timeout=API_TIMEOUT)
    except RuntimeError as exc:
        print(f"  [TRACE] Inference error: {exc}", file=sys.stderr)
        # Still try to collect
        try:
            _api(api_url, "/neuroplastic/trace/collect", {})
        except RuntimeError:
            pass
        raise

    # Collect trace data
    trace_data = _api(api_url, "/neuroplastic/trace/collect", {})
    return trace_data


# ---------------------------------------------------------------------------
# Step 3: Parse thinking into segments
# ---------------------------------------------------------------------------

def parse_thinking_segments(thinking_text: str) -> list[dict]:
    """Split thinking text into segments and extract confidence + state claims."""
    if not thinking_text.strip():
        return []

    # Split on sentence boundaries, step markers, or newlines
    # Try splitting on numbered steps first, then sentences
    lines = thinking_text.strip().split('\n')

    segments = []
    current_segment = []
    current_start_char = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_segment:
                text = '\n'.join(current_segment)
                seg = _analyze_segment(text, len(segments))
                if seg:
                    segments.append(seg)
                current_segment = []
            continue
        current_segment.append(stripped)

    # Flush remaining
    if current_segment:
        text = '\n'.join(current_segment)
        seg = _analyze_segment(text, len(segments))
        if seg:
            segments.append(seg)

    return segments


def _analyze_segment(text: str, idx: int) -> Optional[dict]:
    """Analyze a segment for confidence and state claims."""
    if len(text.strip()) < 10:
        return None

    text_lower = text.lower()

    # Count confidence signals
    high_count = sum(1 for w in HIGH_CONFIDENCE if w in text_lower)
    low_count = sum(1 for w in LOW_CONFIDENCE if w in text_lower)

    # Also check for explicit confidence markers
    if re.search(r'\bconfidence\s*[:=]\s*high\b', text_lower):
        high_count += 3
    elif re.search(r'\bconfidence\s*[:=]\s*medium\b', text_lower):
        high_count += 1
        low_count += 1
    elif re.search(r'\bconfidence\s*[:=]\s*low\b', text_lower):
        low_count += 3

    if high_count > low_count:
        confidence = "high"
        confidence_score = 2
    elif low_count > high_count:
        confidence = "low"
        confidence_score = 0
    else:
        confidence = "medium"
        confidence_score = 1

    # Extract state claims: "X = N", "X: N", "X is N", "N items"
    claims = {}
    # Pattern: word = number or word: number
    for m in re.finditer(
        r'(\b[a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s*(?:=|:|is|are|becomes?)\s*'
        r'\$?(\d+(?:,\d{3})*(?:\.\d+)?)',
        text, re.IGNORECASE
    ):
        key = m.group(1).strip().lower()
        val_str = m.group(2).replace(',', '')
        try:
            val = int(val_str) if '.' not in val_str else float(val_str)
            claims[key] = val
        except ValueError:
            pass

    return {
        "segment_idx": idx,
        "text": text[:500],  # truncate for storage
        "confidence": confidence,
        "confidence_score": confidence_score,
        "high_signals": high_count,
        "low_signals": low_count,
        "state_claims": claims,
    }


# ---------------------------------------------------------------------------
# Step 4: Check answer correctness
# ---------------------------------------------------------------------------

def check_answer_correct(answer_text: str, expected: dict) -> tuple[bool, dict]:
    """Check if the answer contains the expected values."""
    text_lower = answer_text.lower()
    matched = {}
    missed = {}
    for key, val in expected.items():
        # Look for the number near the key word
        key_lower = key.lower()
        # Try various patterns
        found = False
        for pattern in [
            rf'{re.escape(key_lower)}(?:s|es)?\s*[:=\-]?\s*\$?{val}\b',
            rf'\b{val}\s+{re.escape(key_lower)}(?:s|es)?\b',
            rf'{re.escape(key_lower)}.*?{val}\b',
            rf'\b{val}\b.*?{re.escape(key_lower)}',
        ]:
            if re.search(pattern, text_lower):
                found = True
                break
        if found:
            matched[key] = val
        else:
            missed[key] = val

    all_correct = len(missed) == 0
    return all_correct, {"matched": matched, "missed": missed}


def check_claims_against_ground_truth(
    claims: dict, expected: dict, step_text: str
) -> Optional[bool]:
    """Check if state claims in a thinking segment are correct.

    Returns True if all claims match expected, False if any mismatch,
    None if no relevant claims found.
    """
    if not claims:
        return None

    relevant = 0
    correct = 0
    for claim_key, claim_val in claims.items():
        for exp_key, exp_val in expected.items():
            if exp_key.lower() in claim_key or claim_key in exp_key.lower():
                relevant += 1
                if claim_val == exp_val:
                    correct += 1
                break

    if relevant == 0:
        return None
    return correct == relevant


# ---------------------------------------------------------------------------
# Step 5: Extract activation metrics per segment
# ---------------------------------------------------------------------------

def extract_activation_metrics(trace_data: dict, n_segments: int) -> list[dict]:
    """Extract per-segment activation metrics from trace data.

    Since trace gives us per-token output_norms for each layer, we divide
    the token sequence into n_segments equal chunks and compute summary
    stats per chunk.
    """
    if not trace_data or "layers" not in trace_data:
        return [{}] * n_segments

    # Get per-token data from Mamba layers
    layer_output_norms = {}
    layer_head_norms = {}
    for layer_idx in MAMBA_LAYERS:
        layer_key = f"layer_{layer_idx}"
        layer = trace_data.get("layers", {}).get(layer_key, {})
        norms = layer.get("output_norms", [])
        if norms:
            layer_output_norms[layer_idx] = norms
        hnm = layer.get("head_norm_mean", [])
        if hnm:
            layer_head_norms[layer_idx] = hnm

    # Get residual stream norms per layer
    residual_norms = trace_data.get("residual_stream", {}).get(
        "norm_per_layer", [])

    # Use output_norms (per-token) to segment
    # Find total tokens from the longest output_norms array
    total_tokens = 0
    for norms in layer_output_norms.values():
        total_tokens = max(total_tokens, len(norms))

    if total_tokens == 0 or n_segments == 0:
        return [{}] * max(n_segments, 1)

    chunk_size = max(1, total_tokens // n_segments)
    metrics = []

    for seg_idx in range(n_segments):
        start = seg_idx * chunk_size
        end = min(start + chunk_size, total_tokens)
        if seg_idx == n_segments - 1:
            end = total_tokens  # last segment gets remainder

        seg_metrics = {}
        for layer_idx, norms in layer_output_norms.items():
            chunk = norms[start:end]
            if chunk:
                seg_metrics[f"layer_{layer_idx}_output_norm_mean"] = (
                    sum(chunk) / len(chunk))
                seg_metrics[f"layer_{layer_idx}_output_norm_var"] = (
                    sum((x - sum(chunk)/len(chunk))**2 for x in chunk)
                    / len(chunk) if len(chunk) > 1 else 0
                )
                # Change rate: mean absolute difference between consecutive tokens
                if len(chunk) > 1:
                    diffs = [abs(chunk[i] - chunk[i-1])
                             for i in range(1, len(chunk))]
                    seg_metrics[f"layer_{layer_idx}_change_rate"] = (
                        sum(diffs) / len(diffs))
                else:
                    seg_metrics[f"layer_{layer_idx}_change_rate"] = 0.0

        # Aggregate across Mamba layers
        norm_means = [seg_metrics.get(f"layer_{l}_output_norm_mean", 0)
                      for l in MAMBA_LAYERS]
        change_rates = [seg_metrics.get(f"layer_{l}_change_rate", 0)
                        for l in MAMBA_LAYERS]
        if norm_means:
            seg_metrics["mamba_mean_norm"] = sum(norm_means) / len(norm_means)
        if change_rates:
            seg_metrics["mamba_mean_change_rate"] = (
                sum(change_rates) / len(change_rates))

        # Head-level norms (these are per-head means, not per-token)
        # So they're constant across segments — include as reference
        for layer_idx, hnm in layer_head_norms.items():
            seg_metrics[f"layer_{layer_idx}_head_norm_std"] = (
                _std(hnm) if hnm else 0)

        metrics.append(seg_metrics)

    return metrics


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# Step 6: Compute correlations
# ---------------------------------------------------------------------------

def spearman_rank(x: list[float], y: list[float]) -> Optional[float]:
    """Compute Spearman rank correlation. Returns None if insufficient data."""
    n = min(len(x), len(y))
    if n < 3:
        return None

    x = x[:n]
    y = y[:n]

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

    rx = rank(x)
    ry = rank(y)

    d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    rho = 1 - (6 * d_sq) / (n * (n * n - 1))
    return round(rho, 4)


def compute_trial_correlations(segments: list[dict],
                               activations: list[dict]) -> dict:
    """Compute correlations between confidence and activation metrics."""
    if len(segments) < 3 or len(activations) < 3:
        return {"note": "too_few_segments", "n_segments": len(segments)}

    n = min(len(segments), len(activations))
    confidence_scores = [s["confidence_score"] for s in segments[:n]]

    correlations = {}

    # Correlate confidence with each activation metric
    for metric_key in [
        "mamba_mean_norm", "mamba_mean_change_rate",
        "layer_48_output_norm_mean", "layer_50_output_norm_mean",
        "layer_48_change_rate", "layer_50_change_rate",
    ]:
        values = [a.get(metric_key, 0) for a in activations[:n]]
        if any(v != 0 for v in values):
            rho = spearman_rank(confidence_scores, values)
            if rho is not None:
                correlations[f"confidence_vs_{metric_key}"] = rho

    # Correlate claim correctness with activation metrics
    correctness_scores = []
    for s in segments[:n]:
        cc = s.get("claims_correct")
        if cc is True:
            correctness_scores.append(1.0)
        elif cc is False:
            correctness_scores.append(0.0)
        else:
            correctness_scores.append(0.5)  # no claims → neutral

    for metric_key in ["mamba_mean_norm", "mamba_mean_change_rate"]:
        values = [a.get(metric_key, 0) for a in activations[:n]]
        if any(v != 0 for v in values):
            rho = spearman_rank(correctness_scores, values)
            if rho is not None:
                correlations[f"claim_correctness_vs_{metric_key}"] = rho

    correlations["n_segments"] = n
    return correlations


# ---------------------------------------------------------------------------
# Step 7: Run a single trial
# ---------------------------------------------------------------------------

def run_trial(api_url: str, problem: dict, trial_idx: int) -> dict:
    """Run a single trial: generate thinking, trace, analyze."""
    print(f"    Trial {trial_idx+1}:")

    # 1. Generate with thinking
    print(f"      Generating thinking chain...", end="", flush=True)
    gen = generate_with_thinking(api_url, problem)
    thinking = gen["thinking_text"]
    answer = gen["answer_text"]
    print(f" {len(thinking)} chars thinking, {len(answer)} chars answer")

    # 2. Check correctness
    correct, match_info = check_answer_correct(answer, problem["expected"])
    print(f"      Answer correct: {correct} {match_info}")

    # 3. Parse thinking segments
    segments = parse_thinking_segments(thinking)
    print(f"      {len(segments)} thinking segments parsed")

    # Tag claim correctness
    for seg in segments:
        seg["claims_correct"] = check_claims_against_ground_truth(
            seg["state_claims"], problem["expected"], seg["text"]
        )

    # 4. Replay through TRACE
    print(f"      Replaying through TRACE...", end="", flush=True)
    try:
        trace_data = trace_replay(api_url, problem, thinking, answer)
        # Count tokens traced
        sample_norms = trace_data.get("layers", {}).get(
            "layer_48", {}).get("output_norms", [])
        print(f" {len(sample_norms)} tokens traced")
    except RuntimeError as exc:
        print(f" FAILED: {exc}")
        trace_data = {}

    # 5. Extract activation metrics per segment
    activations = extract_activation_metrics(trace_data, len(segments))

    # 6. Compute correlations
    correlations = compute_trial_correlations(segments, activations)

    # Compute confidence distribution
    conf_dist = {"high": 0, "medium": 0, "low": 0}
    for s in segments:
        conf_dist[s["confidence"]] += 1

    return {
        "trial": trial_idx + 1,
        "final_answer_correct": correct,
        "answer_match_info": match_info,
        "thinking_text": thinking[:3000],  # truncate for storage
        "answer_text": answer[:1000],
        "n_thinking_segments": len(segments),
        "confidence_distribution": conf_dist,
        "thinking_segments": segments,
        "activation_metrics": activations,
        "correlations": correlations,
    }


# ---------------------------------------------------------------------------
# Step 8: Run full probe
# ---------------------------------------------------------------------------

def run_probe(api_url: str, output_dir: Path,
              n_trials: int = 10,
              problem_ids: Optional[list[str]] = None):
    """Run the full awareness probe across problems and trials."""
    print("=" * 60)
    print("PHASE 5: THINKING-CHAIN AWARENESS PROBE")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Trials per problem: {n_trials}")
    print(f"Approach: replay (trace thinking chain as input)")
    print()

    # Wait for API
    print("Checking API availability...", end="", flush=True)
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
        print(" TIMEOUT — API not available")
        sys.exit(1)

    # Select problems
    if problem_ids:
        problems = [p for p in PROBLEMS if p["id"] in problem_ids]
    else:
        problems = PROBLEMS

    all_results = []
    all_correlations = []

    for problem in problems:
        print(f"\n{'='*60}")
        print(f"Problem: {problem['id']} (difficulty {problem['difficulty']})")
        print(f"Expected: {problem['expected']}")
        print(f"{'='*60}")

        trials = []
        for t_idx in range(n_trials):
            trial = run_trial(api_url, problem, t_idx)
            trials.append(trial)
            time.sleep(1)  # Brief pause between trials

        # Aggregate for this problem
        correct_trials = [t for t in trials if t["final_answer_correct"]]
        incorrect_trials = [t for t in trials if not t["final_answer_correct"]]

        # Compute aggregate correlations
        all_corr_keys = set()
        for t in trials:
            all_corr_keys.update(
                k for k in t["correlations"] if k != "n_segments"
                and k != "note")

        agg_corr = {}
        for key in sorted(all_corr_keys):
            vals = [t["correlations"].get(key) for t in trials
                    if t["correlations"].get(key) is not None]
            if vals:
                agg_corr[f"mean_{key}"] = round(sum(vals) / len(vals), 4)

        # Split by correct/incorrect
        correct_corrs = {}
        incorrect_corrs = {}
        for key in sorted(all_corr_keys):
            c_vals = [t["correlations"].get(key) for t in correct_trials
                      if t["correlations"].get(key) is not None]
            i_vals = [t["correlations"].get(key) for t in incorrect_trials
                      if t["correlations"].get(key) is not None]
            if c_vals:
                correct_corrs[f"mean_{key}"] = round(
                    sum(c_vals) / len(c_vals), 4)
            if i_vals:
                incorrect_corrs[f"mean_{key}"] = round(
                    sum(i_vals) / len(i_vals), 4)

        problem_result = {
            "problem_id": problem["id"],
            "difficulty": problem["difficulty"],
            "expected": problem["expected"],
            "n_trials": n_trials,
            "n_correct": len(correct_trials),
            "n_incorrect": len(incorrect_trials),
            "accuracy": len(correct_trials) / n_trials if n_trials else 0,
            "trials": trials,
            "aggregate_correlations": agg_corr,
            "correct_trial_correlations": correct_corrs,
            "incorrect_trial_correlations": incorrect_corrs,
        }

        all_results.append(problem_result)
        all_correlations.append({
            "problem_id": problem["id"],
            "aggregate": agg_corr,
            "correct_trials": correct_corrs,
            "incorrect_trials": incorrect_corrs,
            "n_correct": len(correct_trials),
            "n_incorrect": len(incorrect_trials),
        })

        # Print summary for this problem
        print(f"\n  Summary for {problem['id']}:")
        print(f"    Accuracy: {len(correct_trials)}/{n_trials}")
        print(f"    Aggregate correlations: {json.dumps(agg_corr, indent=6)}")
        if correct_corrs:
            print(f"    Correct-trial correlations: "
                  f"{json.dumps(correct_corrs, indent=6)}")
        if incorrect_corrs:
            print(f"    Incorrect-trial correlations: "
                  f"{json.dumps(incorrect_corrs, indent=6)}")

    # Compute grand aggregate
    grand_corr_keys = set()
    for ac in all_correlations:
        grand_corr_keys.update(ac["aggregate"].keys())

    grand_agg = {}
    for key in sorted(grand_corr_keys):
        vals = [ac["aggregate"].get(key) for ac in all_correlations
                if ac["aggregate"].get(key) is not None]
        if vals:
            grand_agg[key] = round(sum(vals) / len(vals), 4)

    # Determine verdict
    norm_corr = grand_agg.get("mean_confidence_vs_mamba_mean_norm")
    change_corr = grand_agg.get("mean_confidence_vs_mamba_mean_change_rate")
    if norm_corr is not None:
        abs_corr = abs(norm_corr)
        if abs_corr > 0.5:
            verdict = "strong_correlation"
        elif abs_corr > 0.2:
            verdict = "moderate_correlation"
        else:
            verdict = "weak_correlation"
    else:
        verdict = "insufficient_data"

    summary = {
        "experiment": "Phase 5 Thinking-Chain Awareness Probe",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL_NAME,
        "approach": "replay",
        "n_problems": len(problems),
        "n_trials_per_problem": n_trials,
        "grand_aggregate_correlations": grand_agg,
        "per_problem_correlations": all_correlations,
        "verdict": verdict,
    }

    # Write results
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "probe_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results: {output_dir / 'probe_results.json'}")

    with open(output_dir / "correlation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Correlation summary: {output_dir / 'correlation_summary.json'}")

    # Print grand summary
    print(f"\n{'='*60}")
    print("GRAND SUMMARY")
    print(f"{'='*60}")
    print(f"Verdict: {verdict}")
    print(f"Grand correlations: {json.dumps(grand_agg, indent=2)}")
    for ac in all_correlations:
        pid = ac["problem_id"]
        nc = ac["n_correct"]
        ni = ac["n_incorrect"]
        print(f"  {pid}: {nc} correct, {ni} incorrect")
        if ac["correct_trials"]:
            print(f"    correct trials: {ac['correct_trials']}")
        if ac["incorrect_trials"]:
            print(f"    incorrect trials: {ac['incorrect_trials']}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Thinking-Chain Awareness Probe")
    parser.add_argument("--api-url", default="http://spark-129a.local:30000")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of trials per problem")
    parser.add_argument("--problems", nargs="*",
                        help="Specific problem IDs to run (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_probe(
        api_url=args.api_url,
        output_dir=output_dir,
        n_trials=args.trials,
        problem_ids=args.problems,
    )


if __name__ == "__main__":
    main()
