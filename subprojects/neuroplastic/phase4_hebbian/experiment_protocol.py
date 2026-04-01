#!/usr/bin/env python3
"""Phase 4: Hebbian Learning Experiment Protocol

Automates a four-phase experiment to measure whether Hebbian adaptation of
Mamba SSM decay parameters improves state-tracking accuracy.

Phases:
  A. Baseline    — run all 50 training problems without any weight updates
  B. Hebbian     — process the same problems with online Hebbian updates
  C. Retention   — test on 20 held-out problems after adaptation (no updates)
  D. Cross-domain — run the full capability eval harness to check for regression

Usage:
    python3 experiment_protocol.py --api-url http://spark-129a.local:30000 [options]

Options:
    --api-url         vLLM + neuroplastic API base URL
    --phase           Which phase to run: a, b, c, d, or all (default: all)
    --learning-rate   Hebbian eta for Phase B (default: 0.01)
    --output-dir      Directory for result files (default: ./results)
    --training-inputs Path to training_inputs.json (default: same directory)
    --no-restore      Do NOT restore weights after Phase B (keep adapted state for C/D)
"""

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
DEFAULT_API_URL = "http://spark-129a.local:30000"
TARGET_LAYERS = [44, 46, 48, 50]

# System prompt for inference queries
SYSTEM_PROMPT = (
    "You are a precise calculation assistant. "
    "When asked to track quantities across operations, "
    "work through each step carefully and give the exact final count for each item. "
    "Always state the final answer explicitly with the number first, then the item name."
)

# Retention test problems (20 held-out problems, not in training_inputs.json)
# These test the same state-tracking skill but with novel scenarios.
RETENTION_INPUTS = [
    {
        "id": "ret_001",
        "question": "A drawer starts with 10 socks. Remove 4. Add 2. Remove 2. How many socks?",
        "key_facts": ["6 sock"],
    },
    {
        "id": "ret_002",
        "question": "A piggy bank has 25 coins. Add 10. Spend 18. Add 5. How many coins?",
        "key_facts": ["22 coin"],
    },
    {
        "id": "ret_003",
        "question": "A tank has 50 fish. Add 20 fish. Remove 50 fish. Add 10 fish. How many fish?",
        "key_facts": ["30 fish"],
    },
    {
        "id": "ret_004",
        "question": "Start with 8 chairs and 4 tables. Remove 3 chairs. Add 2 tables. Remove all tables. How many chairs and tables remain?",
        "key_facts": ["5 chair", "0 table"],
    },
    {
        "id": "ret_005",
        "question": "A playlist has 12 songs. Add 5. Delete 8. Add 3. Delete all remaining. How many songs?",
        "key_facts": ["0 song"],
    },
    {
        "id": "ret_006",
        "question": "Inventory: Pens=20, Pencils=15. Add 10 pens. Remove 15 pencils. Add 5 pencils. Remove 12 pens. How many pens and pencils?",
        "key_facts": ["18 pen", "5 pencil"],
    },
    {
        "id": "ret_007",
        "question": "A shelf has 30 cans of soup, 20 cans of beans, 10 cans of corn. Sell 10 soup, 15 beans, 10 corn. Restock 5 soup and 20 corn. Sell 8 soup and 0 beans. How many of each remain?",
        "key_facts": ["17 soup", "5 bean", "20 corn"],
    },
    {
        "id": "ret_008",
        "question": "Variable P=100, Q=50. Add 25 to P. Subtract 30 from Q. Set P to P minus Q. What are P and Q?",
        "key_facts": ["105", "20"],
    },
    {
        "id": "ret_009",
        "question": "Three jars: Red=15, Green=10, Blue=5. Move 5 from Red to Green. Move 3 from Blue to Red. Move all Green to Blue. How many in each jar?",
        "key_facts": ["13 red", "0 green", "18 blue"],
    },
    {
        "id": "ret_010",
        "question": "A queue has 0 items. Enqueue 5. Enqueue 3. Dequeue 2. Enqueue 4. Dequeue 6. Enqueue 1. How many items are in the queue?",
        "key_facts": ["5 item"],
    },
    {
        "id": "ret_011",
        "question": "Budget: Income=$2000, Expenses=$1500, Savings=$0. Add $500 income. Pay $300 extra expenses. Transfer $200 to savings. Pay $100 from savings. What is the final savings balance?",
        "key_facts": ["100 dollar"],
    },
    {
        "id": "ret_012",
        "question": "Points: Player 1=0, Player 2=0, Player 3=0. Round 1: P1+5, P2+3, P3+7. Round 2: P1+4, P2+8, P3+2. Round 3: Lowest scorer gets +3 bonus. P1+2, P2+1, P3+3. What are the final scores?",
        "key_facts": ["14", "12", "12"],
    },
    {
        "id": "ret_013",
        "question": "A warehouse starts with 200 units of product X and 150 units of product Y. Ship 60 X and 40 Y. Receive 0 X and 80 Y. Ship all remaining Y. Ship 100 X. Receive 30 X. How many X and Y remain?",
        "key_facts": ["70 X", "0 Y"],
    },
    {
        "id": "ret_014",
        "question": "A team has 5 developers, 3 designers, and 2 managers. Hire 2 developers. 1 designer quits. Promote 1 developer to manager. Contract 4 temporary developers. The project ends: all temporary developers leave. What is the final headcount?",
        "key_facts": ["6 developer", "2 designer", "3 manager"],
    },
    {
        "id": "ret_015",
        "question": "Energy levels: Battery A=100%, Battery B=80%, Battery C=60%. A powers a device for 4 hours at 10%/hour. B charges at 5%/hour for 4 hours. C powers a device for 3 hours at 15%/hour then charges at 10%/hour for 1 hour. What are the final battery levels?",
        "key_facts": ["60% A", "100% B", "25% C"],
    },
    {
        "id": "ret_016",
        "question": "Library: Fiction=200 books, Non-fiction=150 books, Reference=50 books. Check out 30 fiction and 20 non-fiction. Return 10 fiction. Check out 50 reference. Return all reference. Add 25 new non-fiction. How many of each type?",
        "key_facts": ["180 fiction", "155 non-fiction", "50 reference"],
    },
    {
        "id": "ret_017",
        "question": "Network traffic: Router A handles 100 Mbps, Router B handles 0 Mbps. Shift 30 Mbps from A to B. Add 20 Mbps to A. B develops a fault and drops 25% of its traffic (round down). Restore B to full capacity (add back dropped traffic). What traffic does each router handle?",
        "key_facts": ["90 Mbps A", "37 Mbps B"],
    },
    {
        "id": "ret_018",
        "question": "A recipe calls for 4 eggs, 2 cups flour, 1 cup milk, 0.5 tsp salt. Double the recipe. Use 3 eggs for another project. Use half the flour for bread. Use all milk for the recipe. What ingredients remain after making the recipe?",
        "key_facts": ["5 egg", "2 cup flour", "0 milk"],
    },
    {
        "id": "ret_019",
        "question": "Store register: $100 starting cash. Sale 1: $45. Sale 2: $30. Give change of $5. Sale 3: $60. Pay supplier $80. Sale 4: $25. Give change of $10. How much cash is in the register?",
        "key_facts": ["165 dollar"],
    },
    {
        "id": "ret_020",
        "question": "Particles: Type A=50, Type B=30, Type C=20. Reaction 1: 2A + 1B → 1C (happens 10 times). Reaction 2: 3C → 1A + 2B (happens 5 times). How many of each particle remain after both reactions?",
        "key_facts": ["35 A", "25 B", "15 C"],
    },
]

# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    """POST JSON to url, return parsed response dict."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc


def call_model(
    api_url: str,
    question: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> Optional[str]:
    """Send a chat completion request, return the response text or None."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    endpoint = api_url.rstrip("/") + "/v1/chat/completions"
    try:
        result = _post(endpoint, payload)
        return result["choices"][0]["message"]["content"]
    except (RuntimeError, KeyError, IndexError) as exc:
        print(f"    [API ERROR] {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Scoring helpers (mirrored from run_eval.py for consistency)
# ---------------------------------------------------------------------------

def _fact_matches(fact: str, response_lower: str) -> bool:
    """Check if a key fact appears in the response, handling format variations.

    Handles:
    - Direct substring: "2 apple" in "I found 2 apple in the bag"
    - Reversed order: "2 apple" matches "apples: 2" or "apple: 2"
    - Pluralization: "apple" matches "apples"
    - Zero values: "0 orange" matches "oranges: 0" or "no oranges" or "0 oranges"

    This is a verbatim copy of _fact_matches from run_eval.py to ensure
    identical scoring logic across Phase A, B, C evaluations.
    """
    fact_lower = fact.lower()

    # Direct substring match
    if fact_lower in response_lower:
        return True

    # Try "number word" → "word(s): number" / "word(s) number" reversal
    parts = fact_lower.split()
    if len(parts) == 2:
        num_str, word = parts
        # Match with optional plural 's'/'es' and separators like ": " or " = "
        pattern = rf'{re.escape(word)}(?:s|es)?\s*[:=\-]?\s*{re.escape(num_str)}\b'
        if re.search(pattern, response_lower):
            return True
        # Also match "number word(s/es)" with word boundary
        pattern2 = rf'\b{re.escape(num_str)}\s+{re.escape(word)}(?:s|es)?\b'
        if re.search(pattern2, response_lower):
            return True
        # "no <word>" for zero
        if num_str == "0":
            pattern3 = rf'\bno\s+{re.escape(word)}(?:s|es)?\b'
            if re.search(pattern3, response_lower):
                return True

    return False


def score_problem(response: Optional[str], key_facts: list[str]) -> dict[str, Any]:
    """Score a single problem response against its key facts.

    Returns:
        {
          "pass": bool,
          "matched": list of matched facts,
          "missed": list of missed facts,
        }
    """
    if response is None:
        return {"pass": False, "matched": [], "missed": list(key_facts)}

    response_lower = response.lower()
    matched = []
    missed = []
    for fact in key_facts:
        if _fact_matches(fact, response_lower):
            matched.append(fact)
        else:
            missed.append(fact)

    return {
        "pass": len(missed) == 0,
        "matched": matched,
        "missed": missed,
    }


# ---------------------------------------------------------------------------
# Phase A: Baseline
# ---------------------------------------------------------------------------

def run_phase_a(
    api_url: str,
    training_inputs: list[dict],
    output_dir: Path,
) -> dict[str, Any]:
    """Baseline: run all problems without any Hebbian updates.

    Args:
        api_url: API base URL.
        training_inputs: List of problem dicts from training_inputs.json.
        output_dir: Where to write phase_a_results.json.

    Returns:
        Summary dict with accuracy and per-problem results.
    """
    print(f"\n{'='*60}")
    print("PHASE A: Baseline (no Hebbian updates)")
    print(f"{'='*60}")
    print(f"Problems: {len(training_inputs)}")

    results = []
    pass_count = 0
    t_start = time.time()

    for i, problem in enumerate(training_inputs):
        pid = problem["id"]
        question = problem["question"]
        key_facts = problem["key_facts"]
        difficulty = problem.get("difficulty", 0)

        print(f"  [{i+1:3d}/{len(training_inputs)}] {pid} (diff={difficulty}): ", end="", flush=True)

        response = call_model(api_url, question)
        score = score_problem(response, key_facts)
        passed = score["pass"]
        pass_count += int(passed)

        status = "PASS" if passed else "FAIL"
        miss_str = f" miss={score['missed']}" if score["missed"] else ""
        print(f"{status}{miss_str}")

        results.append({
            "id": pid,
            "difficulty": difficulty,
            "question": question,
            "key_facts": key_facts,
            "response": response,
            "score": score,
        })

    elapsed = time.time() - t_start
    accuracy = pass_count / len(training_inputs) if training_inputs else 0.0

    # Break down by difficulty
    by_difficulty: dict[int, dict] = {}
    for r in results:
        d = r["difficulty"]
        if d not in by_difficulty:
            by_difficulty[d] = {"pass": 0, "total": 0}
        by_difficulty[d]["total"] += 1
        if r["score"]["pass"]:
            by_difficulty[d]["pass"] += 1
    for d, stats in by_difficulty.items():
        stats["accuracy"] = stats["pass"] / stats["total"] if stats["total"] else 0.0

    summary = {
        "phase": "A",
        "description": "Baseline (no Hebbian updates)",
        "timestamp": _iso_now(),
        "api_url": api_url,
        "n_problems": len(training_inputs),
        "pass_count": pass_count,
        "accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 1),
        "by_difficulty": by_difficulty,
        "results": results,
    }

    out_path = output_dir / "phase_a_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nPhase A complete: {accuracy:.1%} ({pass_count}/{len(training_inputs)}) "
          f"in {elapsed:.1f}s")
    print(f"Results written to: {out_path}")
    return summary


# ---------------------------------------------------------------------------
# Phase B: Hebbian adaptation
# ---------------------------------------------------------------------------

def run_phase_b(
    api_url: str,
    training_inputs: list[dict],
    target_layers: list[int],
    learning_rate: float,
    output_dir: Path,
) -> dict[str, Any]:
    """Hebbian adaptation: process problems with online weight updates.

    Imports HebbianEngine and runs run_episode() across all training inputs.
    After each problem:
      1. Trace the problem's question through the model
      2. Compute Hebbian deltas from activation data
      3. Apply updates and homeostasis
      4. Immediately query the model for its answer (with updated weights)
      5. Score the response

    Args:
        api_url: API base URL.
        training_inputs: List of problem dicts.
        target_layers: Mamba layer indices to update.
        learning_rate: Hebbian eta.
        output_dir: Where to write phase_b_results.json.

    Returns:
        Summary dict with accuracy trajectory and per-step records.
    """
    print(f"\n{'='*60}")
    print("PHASE B: Hebbian Adaptation")
    print(f"{'='*60}")
    print(f"Problems: {len(training_inputs)}")
    print(f"Target layers: {target_layers}")
    print(f"Learning rate: {learning_rate}")

    # Import HebbianEngine from the same directory
    sys.path.insert(0, str(Path(__file__).parent))
    from hebbian_engine import HebbianEngine  # noqa: PLC0415

    engine = HebbianEngine(
        api_url=api_url,
        target_layers=target_layers,
        learning_rate=learning_rate,
    )

    # Checkpoint the starting state so we can restore after Phase B if needed
    print("\nCheckpointing pre-adaptation weights...")
    pre_adaptation_norms = engine.checkpoint_all("phase_b_pre_adaptation")

    results = []
    pass_count = 0
    t_start = time.time()

    for i, problem in enumerate(training_inputs):
        pid = problem["id"]
        question = problem["question"]
        key_facts = problem["key_facts"]
        difficulty = problem.get("difficulty", 0)

        print(f"\n  [{i+1:3d}/{len(training_inputs)}] {pid} (diff={difficulty})")

        # Run one Hebbian step: trace, update weights, apply homeostasis
        step_checkpoint = f"phase_b_step_{i}"
        original_norms = engine.checkpoint_all(step_checkpoint)

        # Apply global decay
        engine._apply_global_decay()  # noqa: SLF001

        # Trace the question
        try:
            trace_data = engine.trace_input(question)
            trace_ok = True
        except RuntimeError as exc:
            print(f"    [TRACE ERROR] {exc}", file=sys.stderr)
            engine.restore_all(step_checkpoint)
            trace_ok = False
            trace_data = {}

        # Compute and apply updates if trace succeeded
        updates_applied = {}
        if trace_ok:
            updates = engine.compute_updates(trace_data)
            if updates:
                engine.apply_updates(updates)
                engine.apply_homeostasis(original_norms)
                for t, deltas in updates.items():
                    nonzero = sum(1 for d in deltas if d != 0.0)
                    updates_applied[t] = {
                        "nonzero_heads": nonzero,
                        "min": min(deltas),
                        "max": max(deltas),
                    }

        # Query model AFTER updates (measures adapted performance)
        response = call_model(api_url, question)
        score = score_problem(response, key_facts)
        passed = score["pass"]
        pass_count += int(passed)

        status = "PASS" if passed else "FAIL"
        miss_str = f" miss={score['missed']}" if score["missed"] else ""
        print(f"    Answer: {status}{miss_str}")

        results.append({
            "id": pid,
            "difficulty": difficulty,
            "step": i,
            "question": question,
            "key_facts": key_facts,
            "trace_ok": trace_ok,
            "updates_applied": updates_applied,
            "response": response,
            "score": score,
            "running_accuracy": pass_count / (i + 1),
        })

    elapsed = time.time() - t_start
    accuracy = pass_count / len(training_inputs) if training_inputs else 0.0

    # Build accuracy trajectory (windowed average, window=10)
    window = 10
    trajectory = []
    for i in range(len(results)):
        start = max(0, i - window + 1)
        window_results = results[start : i + 1]
        window_acc = sum(1 for r in window_results if r["score"]["pass"]) / len(window_results)
        trajectory.append({"step": i, "window_accuracy": window_acc})

    # Checkpoint the adapted state before any restore
    print("\nCheckpointing post-adaptation weights...")
    engine.checkpoint_all("phase_b_post_adaptation")

    summary = {
        "phase": "B",
        "description": "Hebbian Adaptation",
        "timestamp": _iso_now(),
        "api_url": api_url,
        "target_layers": target_layers,
        "learning_rate": learning_rate,
        "n_problems": len(training_inputs),
        "pass_count": pass_count,
        "accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 1),
        "accuracy_trajectory": trajectory,
        "pre_adaptation_norms": pre_adaptation_norms,
        "results": results,
    }

    out_path = output_dir / "phase_b_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nPhase B complete: {accuracy:.1%} ({pass_count}/{len(training_inputs)}) "
          f"in {elapsed:.1f}s")
    print(f"Results written to: {out_path}")

    return summary


# ---------------------------------------------------------------------------
# Phase C: Retention
# ---------------------------------------------------------------------------

def run_phase_c(
    api_url: str,
    output_dir: Path,
    retention_inputs: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """Retention: test on new problems without further Hebbian updates.

    This phase runs AFTER Phase B with weights in their adapted state.
    No further weight modifications are made. We measure whether the
    Hebbian-adapted weights generalise to unseen state-tracking problems.

    Args:
        api_url: API base URL.
        output_dir: Where to write phase_c_results.json.
        retention_inputs: Optional override for the 20 retention problems.
            Defaults to the RETENTION_INPUTS constant defined in this module.

    Returns:
        Summary dict with accuracy and per-problem results.
    """
    print(f"\n{'='*60}")
    print("PHASE C: Retention (no further updates)")
    print(f"{'='*60}")

    if retention_inputs is None:
        retention_inputs = RETENTION_INPUTS

    print(f"Retention problems: {len(retention_inputs)}")
    print("NOTE: Weights are in their post-Phase-B adapted state.")

    results = []
    pass_count = 0
    t_start = time.time()

    for i, problem in enumerate(retention_inputs):
        pid = problem.get("id", f"ret_{i+1:03d}")
        question = problem["question"]
        key_facts = problem["key_facts"]

        print(f"  [{i+1:3d}/{len(retention_inputs)}] {pid}: ", end="", flush=True)

        response = call_model(api_url, question)
        score = score_problem(response, key_facts)
        passed = score["pass"]
        pass_count += int(passed)

        status = "PASS" if passed else "FAIL"
        miss_str = f" miss={score['missed']}" if score["missed"] else ""
        print(f"{status}{miss_str}")

        results.append({
            "id": pid,
            "question": question,
            "key_facts": key_facts,
            "response": response,
            "score": score,
        })

    elapsed = time.time() - t_start
    accuracy = pass_count / len(retention_inputs) if retention_inputs else 0.0

    summary = {
        "phase": "C",
        "description": "Retention (post-adaptation, new problems, no updates)",
        "timestamp": _iso_now(),
        "api_url": api_url,
        "n_problems": len(retention_inputs),
        "pass_count": pass_count,
        "accuracy": accuracy,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }

    out_path = output_dir / "phase_c_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nPhase C complete: {accuracy:.1%} ({pass_count}/{len(retention_inputs)}) "
          f"in {elapsed:.1f}s")
    print(f"Results written to: {out_path}")

    return summary


# ---------------------------------------------------------------------------
# Phase D: Cross-domain regression check
# ---------------------------------------------------------------------------

def run_phase_d(
    api_url: str,
    output_dir: Path,
    eval_script: Optional[Path] = None,
) -> dict[str, Any]:
    """Cross-domain: run the full eval harness to check for degradation.

    Calls the Phase 1 eval harness via subprocess. This is the same
    evaluation used in all prior phases, providing a stable external
    benchmark that the weight modifications cannot influence.

    Args:
        api_url: API base URL.
        output_dir: Where to write phase_d_results.json and eval output.
        eval_script: Path to run_eval.py. Defaults to the canonical location
                     at phase1_artifacts/eval_harness/run_eval.py relative
                     to this script's parent directory.

    Returns:
        Summary dict with pass/fail status and eval output path.
    """
    print(f"\n{'='*60}")
    print("PHASE D: Cross-Domain Regression Check")
    print(f"{'='*60}")

    if eval_script is None:
        # Canonical location relative to this file
        eval_script = (
            Path(__file__).parent.parent
            / "phase1_artifacts"
            / "eval_harness"
            / "run_eval.py"
        )

    if not eval_script.exists():
        msg = f"Eval script not found: {eval_script}"
        print(f"  ERROR: {msg}", file=sys.stderr)
        return {
            "phase": "D",
            "description": "Cross-domain regression",
            "timestamp": _iso_now(),
            "error": msg,
            "success": False,
        }

    eval_output_dir = output_dir / "phase_d_eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(eval_script),
        "--api-url", api_url,
        "--output-dir", str(eval_output_dir),
        "--trials", "1",
        "--skip-self-knowledge",
    ]

    print(f"Running eval harness: {' '.join(cmd)}")
    print("(This takes ~2-3 minutes with trials=1...)")

    t_start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute hard limit
        )
        elapsed = time.time() - t_start
        success = proc.returncode == 0
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t_start
        print("  [TIMEOUT] Eval harness exceeded 10 minutes", file=sys.stderr)
        summary = {
            "phase": "D",
            "description": "Cross-domain regression",
            "timestamp": _iso_now(),
            "api_url": api_url,
            "error": "timeout after 600s",
            "success": False,
            "elapsed_seconds": round(elapsed, 1),
        }
        out_path = output_dir / "phase_d_results.json"
        out_path.write_text(json.dumps(summary, indent=2))
        return summary
    except OSError as exc:
        elapsed = time.time() - t_start
        print(f"  [OS ERROR] {exc}", file=sys.stderr)
        summary = {
            "phase": "D",
            "description": "Cross-domain regression",
            "timestamp": _iso_now(),
            "api_url": api_url,
            "error": str(exc),
            "success": False,
            "elapsed_seconds": round(elapsed, 1),
        }
        out_path = output_dir / "phase_d_results.json"
        out_path.write_text(json.dumps(summary, indent=2))
        return summary

    # Print stdout/stderr from eval harness
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr and not success:
        print(proc.stderr, file=sys.stderr)

    # Try to load the capability baseline JSON written by the harness
    cap_baseline_path = eval_output_dir / "capability_baseline.json"
    cap_data = None
    if cap_baseline_path.exists():
        try:
            cap_data = json.loads(cap_baseline_path.read_text())
        except json.JSONDecodeError:
            pass

    # Extract summary from capability data
    eval_summary = None
    if cap_data and "summary" in cap_data:
        eval_summary = cap_data["summary"]

    summary = {
        "phase": "D",
        "description": "Cross-domain regression (full capability eval)",
        "timestamp": _iso_now(),
        "api_url": api_url,
        "eval_script": str(eval_script),
        "eval_output_dir": str(eval_output_dir),
        "returncode": proc.returncode,
        "success": success,
        "elapsed_seconds": round(elapsed, 1),
        "eval_summary": eval_summary,
        "stdout_tail": proc.stdout[-2000:] if proc.stdout else None,
    }

    out_path = output_dir / "phase_d_results.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if success:
        print(f"\nPhase D complete in {elapsed:.1f}s")
    else:
        print(f"\nPhase D FAILED (returncode={proc.returncode}) in {elapsed:.1f}s",
              file=sys.stderr)

    if eval_summary:
        overall = eval_summary.get("overall", {})
        acc = overall.get("accuracy", 0.0)
        print(f"Capability eval accuracy: {acc:.1%}")

    print(f"Results written to: {out_path}")
    return summary


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def write_combined_report(
    phase_results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Write a combined summary across all phases to phase4_summary.json."""
    report = {
        "experiment": "Phase 4 Hebbian Learning",
        "timestamp": _iso_now(),
        "phases": {},
    }

    for phase_label, data in phase_results.items():
        if data is None:
            continue
        report["phases"][phase_label] = {
            "description": data.get("description", ""),
            "accuracy": data.get("accuracy"),
            "pass_count": data.get("pass_count"),
            "n_problems": data.get("n_problems"),
            "elapsed_seconds": data.get("elapsed_seconds"),
            "success": data.get("success", True),
        }

    # Compare A vs B vs C if all ran
    if "A" in report["phases"] and "B" in report["phases"]:
        a_acc = report["phases"]["A"]["accuracy"] or 0.0
        b_acc = report["phases"]["B"]["accuracy"] or 0.0
        report["hebbian_lift_b_vs_a"] = round(b_acc - a_acc, 4)
        print(f"\nHebbian lift (B vs A): {report['hebbian_lift_b_vs_a']:+.1%}")

    if "A" in report["phases"] and "C" in report["phases"]:
        a_acc = report["phases"]["A"]["accuracy"] or 0.0
        c_acc = report["phases"]["C"]["accuracy"] or 0.0
        report["retention_lift_c_vs_a"] = round(c_acc - a_acc, 4)
        print(f"Retention lift (C vs A): {report['retention_lift_c_vs_a']:+.1%}")

    out_path = output_dir / "phase4_summary.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Combined report written to: {out_path}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def load_training_inputs(path: Path) -> list[dict]:
    """Load and validate training_inputs.json.

    Args:
        path: Path to the JSON file.

    Returns:
        List of problem dicts.

    Raises:
        SystemExit: If the file is missing or malformed.
    """
    if not path.exists():
        print(f"ERROR: training_inputs.json not found at {path}", file=sys.stderr)
        sys.exit(1)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: Failed to parse {path}: {exc}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(data, list):
        print(f"ERROR: {path} must be a JSON array", file=sys.stderr)
        sys.exit(1)
    return data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4 Hebbian Learning Experiment Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"vLLM + neuroplastic API base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--phase",
        choices=["a", "b", "c", "d", "all"],
        default="all",
        help="Which phase(s) to run (default: all)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Hebbian learning rate for Phase B (default: 0.01)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for result files (default: phase4_hebbian/results/)",
    )
    parser.add_argument(
        "--training-inputs",
        default=None,
        help="Path to training_inputs.json (default: same directory as this script)",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Do NOT restore weights after Phase B (leave adapted state for C/D)",
    )
    parser.add_argument(
        "--eval-script",
        default=None,
        help="Path to run_eval.py for Phase D (default: auto-detect)",
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_inputs_path = (
        Path(args.training_inputs)
        if args.training_inputs
        else script_dir / "training_inputs.json"
    )

    eval_script = Path(args.eval_script) if args.eval_script else None

    # Wait for API to be ready
    print(f"\n  Waiting for API at {args.api_url}...")
    endpoint = args.api_url.rstrip("/") + "/v1/models"
    t0 = time.time()
    while time.time() - t0 < 600:
        try:
            req = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                models = [m["id"] for m in data.get("data", [])]
                if models:
                    print(f"  API ready. Models: {models}")
                    break
        except Exception:
            pass
        elapsed = time.time() - t0
        print(f"  Not ready ({elapsed:.0f}s). Retrying in 15s...")
        time.sleep(15)
    else:
        print("  FATAL: API not ready after 600s")
        sys.exit(1)

    # Print header
    timestamp = _iso_now()
    print(f"\nPhase 4 Hebbian Learning Experiment")
    print(f"  API URL:       {args.api_url}")
    print(f"  Phase:         {args.phase}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Timestamp:     {timestamp}")

    # Load training inputs (needed for A and B)
    training_inputs = load_training_inputs(training_inputs_path)
    print(f"  Training problems loaded: {len(training_inputs)}")

    phases_to_run = (
        ["a", "b", "c", "d"] if args.phase == "all" else [args.phase]
    )

    phase_results: dict[str, Optional[dict]] = {
        "A": None, "B": None, "C": None, "D": None,
    }

    # Phase A
    if "a" in phases_to_run:
        phase_results["A"] = run_phase_a(
            api_url=args.api_url,
            training_inputs=training_inputs,
            output_dir=output_dir,
        )

    # Phase B
    if "b" in phases_to_run:
        phase_results["B"] = run_phase_b(
            api_url=args.api_url,
            training_inputs=training_inputs,
            target_layers=TARGET_LAYERS,
            learning_rate=args.learning_rate,
            output_dir=output_dir,
        )

        # Restore weights after Phase B unless --no-restore
        if not args.no_restore and "c" not in phases_to_run and "d" not in phases_to_run:
            print("\nRestoring pre-adaptation weights (use --no-restore to skip)...")
            sys.path.insert(0, str(script_dir))
            from hebbian_engine import HebbianEngine  # noqa: PLC0415
            restore_engine = HebbianEngine(args.api_url, TARGET_LAYERS)
            restore_engine.restore_all("phase_b_pre_adaptation")

    # Phase C (runs with post-Phase-B weights if --no-restore or running all phases)
    if "c" in phases_to_run:
        phase_results["C"] = run_phase_c(
            api_url=args.api_url,
            output_dir=output_dir,
        )

    # Phase D
    if "d" in phases_to_run:
        phase_results["D"] = run_phase_d(
            api_url=args.api_url,
            output_dir=output_dir,
            eval_script=eval_script,
        )

    # After all phases that keep adapted weights, restore if requested
    if (
        "b" in phases_to_run
        and not args.no_restore
        and ("c" in phases_to_run or "d" in phases_to_run)
    ):
        print("\nRestoring pre-adaptation weights after all phases...")
        sys.path.insert(0, str(script_dir))
        from hebbian_engine import HebbianEngine  # noqa: PLC0415
        restore_engine = HebbianEngine(args.api_url, TARGET_LAYERS)
        restore_engine.restore_all("phase_b_pre_adaptation")
        print("Weights restored to pre-Phase-B state.")

    # Write combined report
    write_combined_report(
        {k: v for k, v in phase_results.items() if v is not None},
        output_dir,
    )

    print(f"\nDone. All results in: {output_dir}")


if __name__ == "__main__":
    main()
