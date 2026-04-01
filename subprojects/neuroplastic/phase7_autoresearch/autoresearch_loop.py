#!/usr/bin/env python3
"""Phase 7: Autoresearch-Style Self-Modification at Scale.

Nemotron proposes its own weight modifications, eval accepts/rejects them,
and the model learns from the history to propose better modifications.

Target: 100+ cycles overnight. ~2 min per cycle.

Monitoring:
  results/live_status.json  — real-time status (updated every cycle)
  results/cycle_log.jsonl   — per-cycle log (append-only)
  results/best_config.json  — current best modification stack
  autoresearch.log          — stdout
"""

import argparse
import json
import re
import signal
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

from fast_eval import run_fast_eval, check_key_facts, MODEL_NAME

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_TIMEOUT = 120

# Architecture map
MAMBA_LAYERS = [0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50]
ATTENTION_LAYERS = [5,12,19,26,33,42]
MOE_LAYERS = [1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51]

# All available tensor paths
MAMBA_TENSORS = ["mixer.A", "mixer.D", "mixer.dt_bias", "mixer.out_proj.weight",
                 "mixer.in_proj.weight", "mixer.conv1d.weight"]
ATTENTION_TENSORS = ["mixer.qkv_proj.weight", "mixer.o_proj.weight"]
MOE_TENSORS = ["mixer.gate.weight", "mixer.gate.e_score_correction_bias"]

# Neuroplastic API operations
AVAILABLE_OPS = [
    "scale", "add", "scale_slice", "add_slice", "zero_heads",
    "scale_rows", "scale_cols", "lerp", "clamp", "add_noise", "normalize"
]

# Phase 7 expanded search sensitivity findings (top tensors)
SENSITIVITY_MAP = {
    "model.layers.33.mixer.qkv_proj.weight": 0.625,
    "model.layers.16.mixer.dt_bias": 0.250,
    "model.layers.32.mixer.D": 0.250,
    "model.layers.32.mixer.out_proj.weight": 0.250,
    "model.layers.48.mixer.A": 0.250,
    "model.layers.19.mixer.qkv_proj.weight": 0.250,
    "model.layers.33.mixer.o_proj.weight": 0.250,
    "model.layers.15.mixer.gate.weight": 0.250,
    "model.layers.27.mixer.gate.e_score_correction_bias": 0.250,
    "model.layers.38.mixer.gate.e_score_correction_bias": 0.250,
}

# Context management
MAX_HISTORY_TURNS = 5  # Keep last N turns in full
MAX_CONTEXT_TOKENS_APPROX = 20000  # Rough limit before compression


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
        raise RuntimeError(f"HTTP {exc.code}: {body[:400]}") from exc
    except (urllib.error.URLError, ConnectionResetError, OSError) as exc:
        raise RuntimeError(f"Network error: {exc}") from exc


def _api(api_url: str, endpoint: str, payload: dict,
         max_retries: int = 3) -> dict:
    url = api_url.rstrip("/") + endpoint
    for attempt in range(max_retries):
        try:
            return _post(url, payload)
        except RuntimeError as exc:
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
            else:
                raise
    raise RuntimeError(f"Unreachable: {url}")


# ---------------------------------------------------------------------------
# Neuroplastic API wrappers
# ---------------------------------------------------------------------------

def inspect_tensor(api_url: str, tensor: str) -> dict:
    return _api(api_url, "/neuroplastic/inspect", {"tensor": tensor})


def checkpoint_tensor(api_url: str, tensor: str):
    _api(api_url, "/neuroplastic/checkpoint", {"tensor": tensor})


def restore_tensor(api_url: str, tensor: str):
    _api(api_url, "/neuroplastic/restore", {"tensor": tensor})


def modify_tensor(api_url: str, tensor: str, op: str, **kwargs) -> dict:
    payload = {"tensor": tensor, "op": op}
    payload.update(kwargs)
    return _api(api_url, "/neuroplastic/modify", payload)


def list_tensors(api_url: str, filter_str: str = "") -> dict:
    return _api(api_url, "/neuroplastic/list", {"filter": filter_str})


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

def _detect_stuck_tensor(compressed_history: list, window: int = 5) -> str | None:
    """If the last N history entries all target the same tensor, return it."""
    recent = [h for h in compressed_history[-window:]
              if "no MODIFY" not in h and "failed" not in h]
    if len(recent) < 3:
        return None
    tensors = []
    for h in recent:
        # Extract tensor name from "Cycle N: model.layers.X.Y ..."
        parts = h.split(": ", 1)
        if len(parts) > 1:
            tensor = parts[1].split(" ")[0]
            tensors.append(tensor)
    if len(set(tensors)) == 1 and tensors:
        return tensors[0]
    return None


def build_system_prompt(current_score: int, total: int,
                        cycle: int, accepted_mods: list,
                        compressed_history: list) -> str:
    """Build the system prompt for Nemotron's modification proposal."""

    mod_summary = "None yet." if not accepted_mods else "\n".join(
        f"  Cycle {m['cycle']}: {m['tensor']} {m['op']} "
        f"{m.get('params',{})} → {m['score_before']}→{m['score_after']}"
        for m in accepted_mods[-15:]
    )

    history_str = ""
    if compressed_history:
        history_str = "\n".join(compressed_history[-20:])

    # Detect if stuck on one tensor
    stuck_tensor = _detect_stuck_tensor(compressed_history)
    diversify_msg = ""
    if stuck_tensor:
        diversify_msg = (
            f"\n**IMPORTANT: You have been repeatedly modifying {stuck_tensor} "
            f"without improvement. You MUST try a DIFFERENT tensor this time. "
            f"Explore: other Mamba layers (A, D, dt_bias), attention layer 42 "
            f"(o_proj.weight), MoE gate weights, or try different operations "
            f"like add_noise or scale_slice on new layers.**\n"
        )

    prompt = f"""You are modifying your own neural network weights to improve state-tracking accuracy.

Score: {current_score}/{total}. Cycle: {cycle}.
Kept modifications so far: {mod_summary}

ARCHITECTURE: 52 layers. Mamba SSM layers: {MAMBA_LAYERS}. Attention layers: {ATTENTION_LAYERS}. MoE layers: {MOE_LAYERS}.
Tensor suffixes: Mamba: mixer.A, mixer.D, mixer.dt_bias, mixer.out_proj.weight. Attention: mixer.qkv_proj.weight, mixer.o_proj.weight. MoE: mixer.gate.weight.

KEY FINDINGS: Layer 33 qkv_proj is most sensitive (0.625). Layer 48 mixer.A scaling to 0.85-0.95 improved tracking. Attention layers 33,42 are bottlenecks. Small changes (0.95-1.05) accumulate safely. Mamba layers 44,46,48,50 control late-stage state dynamics. MoE gate correction biases (layers 27,38,49) also showed sensitivity.

OPERATIONS: scale, add, scale_slice (start,end,value), add_slice, add_noise (scale,seed), scale_rows (indices,value), scale_cols, normalize (target_norm), clamp (min,max), lerp (alpha).
{diversify_msg}
PROPOSE exactly ONE modification using this XML format. You MUST include this tag:
<MODIFY tensor="model.layers.48.mixer.A" op="scale" value="0.95"/>

Briefly explain your reasoning, then output the MODIFY tag.

{f"Recent history: {history_str}" if history_str else ""}"""
    return prompt


# ---------------------------------------------------------------------------
# Parse Nemotron's response for modification proposals
# ---------------------------------------------------------------------------

def parse_modification(response: str) -> dict | None:
    """Extract MODIFY tag from Nemotron's response."""
    # Match <MODIFY ...> or <MODIFY .../>
    pattern = r'<MODIFY\s+([^>]+?)(?:/>|>.*?</MODIFY>|>)'
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None

    attrs_str = match.group(1)

    # Parse attributes
    attrs = {}
    for m in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', attrs_str):
        key, val = m.group(1), m.group(2)
        attrs[key] = val

    if "tensor" not in attrs or "op" not in attrs:
        return None

    # Convert numeric values
    result = {"tensor": attrs["tensor"], "op": attrs["op"]}
    for key in ["value", "scale", "alpha", "min", "max", "target_norm",
                 "start", "end", "seed"]:
        if key in attrs:
            try:
                result[key] = float(attrs[key])
                if result[key] == int(result[key]) and key in ("start", "end", "seed"):
                    result[key] = int(result[key])
            except ValueError:
                result[key] = attrs[key]

    # Parse indices (JSON array)
    if "indices" in attrs:
        try:
            result["indices"] = json.loads(attrs["indices"])
        except json.JSONDecodeError:
            result["indices"] = attrs["indices"]

    return result


def parse_inspect(response: str) -> str | None:
    """Extract INSPECT tag from response."""
    match = re.search(r'<INSPECT\s+tensor="([^"]+)"', response)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Ask Nemotron for a proposal
# ---------------------------------------------------------------------------

def get_proposal(api_url: str, system_prompt: str,
                 recent_messages: list) -> tuple[str, str]:
    """Send conversation to Nemotron, get response with modification proposal.

    Returns (full_response, thinking_text).
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_messages)
    messages.append({"role": "user",
                     "content": "Propose your next modification."})

    resp = _post(api_url.rstrip("/") + "/v1/chat/completions", {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.7,  # Some creativity in proposals
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=120)

    choice = resp["choices"][0]["message"]
    content = choice.get("content") or ""
    thinking = ""

    # Check reasoning_content field first (vLLM thinking models)
    if choice.get("reasoning_content"):
        thinking = choice["reasoning_content"]

    # Extract thinking from <think> tags if present in content
    if content and "<think>" in content:
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            if not thinking:
                thinking = think_match.group(1)
            content = content[think_match.end():].strip()

    return content, thinking


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

class Monitor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / "cycle_log.jsonl"
        self.status_path = output_dir / "live_status.json"
        self.start_time = time.time()

    def log_cycle(self, data: dict):
        entry = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - self.start_time, 1),
            **data,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def update_status(self, status: dict):
        status["last_updated"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status["elapsed_s"] = round(time.time() - self.start_time, 1)
        elapsed = time.time() - self.start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        status["elapsed_human"] = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
        with open(self.status_path, "w") as f:
            json.dump(status, f, indent=2)

    def save(self, filename: str, data):
        with open(self.output_dir / filename, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Autoresearch Loop
# ---------------------------------------------------------------------------

def run_autoresearch(api_url: str, output_dir: Path,
                     max_cycles: int = 200,
                     problems_path: str = "eval_problems.json"):
    """Main autoresearch loop: Nemotron proposes, we eval, accept/reject."""

    monitor = Monitor(output_dir)
    shutdown = {"requested": False}

    def handle_signal(signum, frame):
        print(f"\n[SIGNAL] Shutdown requested (signal {signum})")
        shutdown["requested"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Load existing state if resuming
    state_path = output_dir / "autoresearch_state.json"
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        print(f"Resuming from cycle {state['cycle']}, "
              f"score {state['best_score']}/{state['total']}")
    else:
        state = {
            "cycle": 0,
            "best_score": 0,
            "current_score": 0,
            "total": 20,
            "accepted_mods": [],
            "compressed_history": [],
            "checkpointed_tensors": [],
            "total_accepted": 0,
            "total_rejected": 0,
        }

    # Wait for API
    print("Checking API...", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < 120:
        try:
            req = urllib.request.Request(
                api_url.rstrip("/") + "/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=10):
                print(" OK")
                break
        except Exception:
            time.sleep(5)
    else:
        print(" TIMEOUT")
        sys.exit(1)

    # Baseline eval
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    baseline = run_fast_eval(api_url, problems_path, verbose=True)
    state["current_score"] = baseline["score"]
    if state["cycle"] == 0:
        state["best_score"] = baseline["score"]
    state["total"] = baseline["total"]

    print(f"\nBaseline: {baseline['score']}/{baseline['total']} "
          f"({baseline['accuracy']:.0%})")
    for cat, info in baseline["per_category"].items():
        print(f"  {cat}: {info['passed']}/{info['total']}")

    monitor.log_cycle({
        "event": "baseline",
        "score": baseline["score"],
        "total": baseline["total"],
        "per_category": baseline["per_category"],
    })

    # Warn if baseline is too high
    if baseline["accuracy"] >= 0.95:
        print("\n[WARNING] Baseline accuracy >= 95%. Eval problems may be "
              "too easy — limited room for improvement.")
    elif baseline["accuracy"] <= 0.30:
        print("\n[WARNING] Baseline accuracy <= 30%. Eval problems may be "
              "too hard — model can't improve from here.")

    # Recent conversation messages for context
    recent_messages = []

    # Main loop
    print(f"\n{'='*70}")
    print(f"AUTORESEARCH LOOP — Target: {max_cycles} cycles")
    print(f"{'='*70}")

    start_cycle = state["cycle"] + 1

    for cycle in range(start_cycle, max_cycles + 1):
        if shutdown["requested"]:
            print(f"\n[SHUTDOWN] Saving state at cycle {cycle-1}")
            break

        cycle_start = time.time()
        state["cycle"] = cycle

        print(f"\n--- Cycle {cycle}/{max_cycles} "
              f"(score: {state['current_score']}/{state['total']}, "
              f"best: {state['best_score']}) ---")

        # Build system prompt with current state
        system_prompt = build_system_prompt(
            state["current_score"], state["total"],
            cycle, state["accepted_mods"],
            state["compressed_history"],
        )

        # Get proposal from Nemotron
        print("  Requesting proposal...", end="", flush=True)
        try:
            response, thinking = get_proposal(
                api_url, system_prompt, recent_messages[-MAX_HISTORY_TURNS*2:])
            print(f" got {len(response)} chars")
        except Exception as exc:
            print(f" FAILED: {exc}")
            state["compressed_history"].append(
                f"Cycle {cycle}: proposal request failed ({exc})")
            monitor.log_cycle({
                "event": "cycle",
                "cycle": cycle,
                "error": str(exc)[:200],
            })
            continue

        # Parse modification from response, or from thinking if response is empty
        mod = parse_modification(response)
        if not mod and thinking:
            mod = parse_modification(thinking)
        inspect_req = parse_inspect(response) if not mod else None
        if not inspect_req and not mod and thinking:
            inspect_req = parse_inspect(thinking)

        if inspect_req:
            # Handle inspect request
            print(f"  Inspecting: {inspect_req}")
            try:
                info = inspect_tensor(api_url, inspect_req)
                inspect_result = json.dumps(info, indent=2)
                recent_messages.append({"role": "assistant", "content": response})
                recent_messages.append({"role": "user",
                    "content": f"Inspection result:\n{inspect_result}\n\n"
                               f"Now propose a modification."})
                state["compressed_history"].append(
                    f"Cycle {cycle}: inspected {inspect_req}")
                monitor.log_cycle({
                    "event": "inspect",
                    "cycle": cycle,
                    "tensor": inspect_req,
                })
            except Exception as exc:
                recent_messages.append({"role": "assistant", "content": response})
                recent_messages.append({"role": "user",
                    "content": f"Inspection failed: {exc}\nPropose a modification."})
            continue

        if not mod:
            print(f"  No MODIFY tag found in response. Clearing context...")
            # Clear recent messages to avoid context pollution from failed attempts
            recent_messages.clear()
            state["compressed_history"].append(
                f"Cycle {cycle}: no MODIFY tag in response")
            monitor.log_cycle({
                "event": "no_tag",
                "cycle": cycle,
                "response_preview": response[:200],
            })
            continue

        tensor_name = mod["tensor"]
        op = mod["op"]
        mod_params = {k: v for k, v in mod.items()
                      if k not in ("tensor", "op")}

        print(f"  Proposal: {tensor_name} op={op} {mod_params}")

        # Extract reasoning summary from thinking chain
        reasoning_summary = ""
        if thinking:
            # Take first 200 chars of thinking as summary
            reasoning_summary = thinking[:200].replace("\n", " ").strip()

        # Checkpoint the tensor
        print(f"  Checkpointing {tensor_name}...", end="", flush=True)
        try:
            checkpoint_tensor(api_url, tensor_name)
            if tensor_name not in state["checkpointed_tensors"]:
                state["checkpointed_tensors"].append(tensor_name)
            print(" OK")
        except RuntimeError as exc:
            print(f" FAILED: {exc}")
            state["compressed_history"].append(
                f"Cycle {cycle}: checkpoint failed for {tensor_name}")
            recent_messages.append({"role": "assistant", "content": response})
            recent_messages.append({"role": "user",
                "content": f"Checkpoint failed for {tensor_name}: {exc}. "
                           f"Try a different tensor."})
            monitor.log_cycle({
                "event": "checkpoint_fail",
                "cycle": cycle,
                "tensor": tensor_name,
                "error": str(exc)[:200],
            })
            continue

        # Apply modification
        print(f"  Applying {op}...", end="", flush=True)
        try:
            modify_result = modify_tensor(api_url, tensor_name, op, **mod_params)
            print(" OK")
        except RuntimeError as exc:
            print(f" FAILED: {exc}")
            restore_tensor(api_url, tensor_name)
            state["compressed_history"].append(
                f"Cycle {cycle}: modify failed {tensor_name} {op}")
            recent_messages.append({"role": "assistant", "content": response})
            recent_messages.append({"role": "user",
                "content": f"Modification failed: {exc}. "
                           f"Try a different approach."})
            monitor.log_cycle({
                "event": "modify_fail",
                "cycle": cycle,
                "tensor": tensor_name,
                "op": op,
                "error": str(exc)[:200],
            })
            continue

        # Eval
        print(f"  Running eval...", flush=True)
        eval_result = run_fast_eval(api_url, problems_path, verbose=False)
        new_score = eval_result["score"]

        score_before = state["current_score"]
        delta = new_score - score_before

        # Accept/reject
        if new_score >= score_before:
            decision = "KEPT"
            state["current_score"] = new_score
            state["total_accepted"] += 1
            if new_score > state["best_score"]:
                state["best_score"] = new_score
            # Save accepted modification
            state["accepted_mods"].append({
                "cycle": cycle,
                "tensor": tensor_name,
                "op": op,
                "params": mod_params,
                "score_before": score_before,
                "score_after": new_score,
                "reasoning": reasoning_summary[:100],
            })
            # Re-checkpoint with new state (so future restores go to this)
            checkpoint_tensor(api_url, tensor_name)
            print(f"  >>> KEPT: {score_before}→{new_score} (+{delta})")
        else:
            decision = "REJECTED"
            state["total_rejected"] += 1
            restore_tensor(api_url, tensor_name)
            print(f"  <<< REJECTED: {score_before}→{new_score} ({delta})")

        cycle_elapsed = time.time() - cycle_start

        # Compress this cycle into history
        params_str = " ".join(f"{k}={v}" for k, v in mod_params.items())
        history_line = (
            f"Cycle {cycle}: {tensor_name} {op} {params_str} "
            f"→ {score_before}→{new_score} {decision} "
            f"({cycle_elapsed:.0f}s)"
        )
        state["compressed_history"].append(history_line)

        # Update conversation context
        feedback = (
            f"Modification applied: {tensor_name} {op} {params_str}\n"
            f"Score: {score_before} → {new_score} ({decision})\n"
            f"Per-category: " +
            ", ".join(f"{cat}: {info['passed']}/{info['total']}"
                      for cat, info in eval_result["per_category"].items())
        )
        recent_messages.append({"role": "assistant", "content": response})
        recent_messages.append({"role": "user", "content": feedback})

        # Trim conversation to last N turns
        if len(recent_messages) > MAX_HISTORY_TURNS * 2:
            recent_messages = recent_messages[-(MAX_HISTORY_TURNS * 2):]

        # Log
        monitor.log_cycle({
            "event": "cycle",
            "cycle": cycle,
            "tensor": tensor_name,
            "op": op,
            "params": mod_params,
            "score_before": score_before,
            "score_after": new_score,
            "decision": decision,
            "per_category": eval_result["per_category"],
            "elapsed_ms": round(cycle_elapsed * 1000),
            "reasoning": reasoning_summary[:100],
        })

        monitor.update_status({
            "phase": "autoresearch",
            "cycle": cycle,
            "max_cycles": max_cycles,
            "current_score": state["current_score"],
            "best_score": state["best_score"],
            "total_problems": state["total"],
            "accuracy": state["current_score"] / state["total"],
            "total_accepted": state["total_accepted"],
            "total_rejected": state["total_rejected"],
            "acceptance_rate": (state["total_accepted"] /
                               (state["total_accepted"] + state["total_rejected"])
                               if (state["total_accepted"] + state["total_rejected"]) > 0
                               else 0),
            "last_decision": decision,
            "last_tensor": tensor_name,
            "last_op": op,
            "last_delta": delta,
            "accepted_mods_count": len(state["accepted_mods"]),
            "recent_accepted": [
                {"cycle": m["cycle"], "tensor": m["tensor"],
                 "op": m["op"], "delta": m["score_after"] - m["score_before"]}
                for m in state["accepted_mods"][-10:]
            ],
        })

        # Save state for resume
        if cycle % 5 == 0:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
            monitor.save("best_config.json", {
                "best_score": state["best_score"],
                "current_score": state["current_score"],
                "total": state["total"],
                "accepted_modifications": state["accepted_mods"],
            })

        # Regression check every 10 cycles
        if cycle % 10 == 0:
            print(f"\n  [REGRESSION CHECK] Running full eval (3 trials)...")
            # Run eval 3 times for stability
            scores = []
            for trial in range(3):
                r = run_fast_eval(api_url, problems_path)
                scores.append(r["score"])
            avg_score = sum(scores) / len(scores)
            print(f"  Regression check: scores={scores}, avg={avg_score:.1f}")

            reg_data = {
                "cycle": cycle,
                "scores": scores,
                "avg_score": avg_score,
                "best_score": state["best_score"],
            }
            reg_path = output_dir / "regression_checks" / f"cycle_{cycle:04d}.json"
            with open(reg_path, "w") as f:
                json.dump(reg_data, f, indent=2)

            # Hard stop if degraded badly
            if avg_score < state["total"] * 0.3:  # Below 30%
                print(f"\n  [ABORT] Regression detected: avg {avg_score:.1f} "
                      f"< {state['total'] * 0.3:.0f}")
                # Restore all checkpointed tensors
                for tensor in state["checkpointed_tensors"]:
                    try:
                        restore_tensor(api_url, tensor)
                    except RuntimeError:
                        pass
                break

    # Final save
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    monitor.save("best_config.json", {
        "best_score": state["best_score"],
        "current_score": state["current_score"],
        "total": state["total"],
        "accepted_modifications": state["accepted_mods"],
    })

    # Summary
    print(f"\n{'='*70}")
    print("AUTORESEARCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Cycles: {state['cycle']}")
    print(f"  Best score: {state['best_score']}/{state['total']} "
          f"({state['best_score']/state['total']:.0%})")
    print(f"  Current score: {state['current_score']}/{state['total']}")
    print(f"  Accepted: {state['total_accepted']}, "
          f"Rejected: {state['total_rejected']}")
    if state["total_accepted"] + state["total_rejected"] > 0:
        rate = state["total_accepted"] / (state["total_accepted"] + state["total_rejected"])
        print(f"  Acceptance rate: {rate:.1%}")

    # Save session summary
    summary = {
        "cycles_completed": state["cycle"],
        "best_score": state["best_score"],
        "baseline_score": baseline["score"],
        "total_problems": state["total"],
        "total_accepted": state["total_accepted"],
        "total_rejected": state["total_rejected"],
        "accepted_modifications": state["accepted_mods"],
        "compressed_history": state["compressed_history"],
    }
    ts = time.strftime("%Y%m%dT%H%M%S")
    monitor.save(f"session_summaries/session_{ts}.json", summary)

    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Autoresearch-Style Self-Modification Loop")
    parser.add_argument("--api-url", default="http://spark-129a.local:30000")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-cycles", type=int, default=200,
                        help="Maximum number of cycles (default: 200)")
    parser.add_argument("--problems", default="eval_problems.json",
                        help="Path to eval problems JSON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    run_autoresearch(
        api_url=args.api_url,
        output_dir=output_dir,
        max_cycles=args.max_cycles,
        problems_path=args.problems,
    )


if __name__ == "__main__":
    main()
