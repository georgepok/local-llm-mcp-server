#!/usr/bin/env python3
"""Phase 7: Expanded Search — Score-Driven Modification Across Full Architecture.

Combines Phase 3's LLM-reasoned approach (which reached 91.7%) with systematic
exploration across ALL modifiable tensors, not just the 4 Mamba A layers.

Architecture map (Nemotron-3-Nano-30B):
  - 52 layers total
  - Mamba layers (23): 0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50
  - Attention layers (6): 5,12,19,26,33,42
  - MoE layers (23): 1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51

Approach:
  Phase A: Sensitivity scan — probe each tensor type with a small perturbation,
           measure which ones affect task performance. Build a sensitivity map.
  Phase B: Targeted search — use the sensitivity map to guide LLM-reasoned
           modifications across the most sensitive tensors.

Designed for long unattended runs with real-time monitoring via:
  - results/live_status.json  (updated after every action)
  - results/search_log.jsonl  (append-only event log)
  - phase7.log               (stdout)
"""

import argparse
import json
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
API_TIMEOUT = 120

MAMBA_LAYERS = [0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50]
ATTENTION_LAYERS = [5,12,19,26,33,42]
MOE_LAYERS = [1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51]

# Tensors to probe in sensitivity scan (per layer type)
MAMBA_TENSORS = ["mixer.A", "mixer.D", "mixer.dt_bias", "mixer.out_proj.weight"]
ATTENTION_TENSORS = ["mixer.o_proj.weight", "mixer.qkv_proj.weight"]
MOE_TENSORS = ["mixer.gate.weight", "mixer.gate.e_score_correction_bias"]

# Scale factors for sensitivity probing — aggressive to detect signal
PROBE_SCALE_DOWN = 0.5
PROBE_SCALE_UP = 2.0

# Eval problems — fast subset for sensitivity scan
EVAL_PROBLEMS = [
    # Easy baselines (should always pass — canary for catastrophic damage)
    {
        "id": "seq_001", "category": "sequential_reasoning",
        "q": "A is twice B. B is three more than C. C is 4. What is A? Show your work step by step.",
        "check": lambda r: "14" in r,
    },
    {
        "id": "code_001", "category": "code_generation",
        "q": ("Write a Python function `fibonacci(n)` that returns the nth Fibonacci number. "
              "Use 0-based indexing (fibonacci(0) = 0, fibonacci(1) = 1)."),
        "check": lambda r: "def fibonacci" in r and ("return" in r or "yield" in r),
    },
    # Medium — model usually gets these right but they're sensitive
    {
        "id": "state_001", "category": "state_tracking",
        "q": ("A bag starts empty. Add 3 apples. Add 2 oranges. Remove 1 apple. "
              "Add 4 bananas. Remove 2 oranges. How many of each fruit are in the bag? "
              "List apples, oranges, and bananas separately."),
        "check": lambda r: all(x in r.lower() for x in ["2", "apple"]) and
                           ("0" in r or "no" in r.lower()) and "4" in r and "banana" in r.lower(),
    },
    {
        "id": "seq_002", "category": "sequential_reasoning",
        "q": ("Start with 100. Subtract 23. Multiply by 2. Add 17. Divide by 3. "
              "What is the final result? Show your work step by step."),
        "check": lambda r: "57" in r,
    },
    # Hard — model fails these ~20-40% of the time (key sensitivity detectors)
    {
        "id": "bank_hard", "category": "state_tracking",
        "q": ("A bank account starts with $1000. Deposit $250. Withdraw $100. "
              "Deposit $500. Withdraw $375. The bank charges a $25 fee. "
              "Deposit $150. What is the final balance?"),
        "check": lambda r: "1400" in r,
    },
    {
        "id": "warehouse_hard", "category": "state_tracking",
        "q": ("A warehouse has 50 boxes of type A, 30 boxes of type B, and 20 boxes "
              "of type C. A truck picks up 15 type A and 10 type B. A delivery adds "
              "25 type C and 5 type A. Another truck picks up all remaining type B "
              "boxes and 10 type C. A final delivery adds 8 type A. "
              "How many boxes of each type remain?"),
        "check": lambda r: "48" in r and ("0" in r or "no" in r.lower()) and "35" in r,
    },
    {
        "id": "inventory_vhard", "category": "state_tracking",
        "q": ("A store tracks 4 items: pencils=100, pens=80, erasers=50, rulers=30. "
              "Order 1: sell 20 pencils and 15 pens. "
              "Order 2: restock 40 erasers and 10 rulers. "
              "Order 3: sell 30 pencils, 25 pens, and all rulers. "
              "Order 4: restock 50 pens. "
              "Order 5: sell 10 erasers. "
              "How many of each item remain?"),
        "check": lambda r: "50" in r and "90" in r and "80" in r,
    },
    {
        "id": "multi_var", "category": "state_tracking",
        "q": ("Three variables: X=0, Y=10, Z=5. Add 5 to X. Subtract 3 from Y. "
              "Set Z to X+Y. Multiply X by 2. Set Y to Z-X. Add Z to X. "
              "What are the final values of X, Y, and Z?"),
        "check": lambda r: "22" in r and "2" in r and "12" in r,
    },
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
# Neuroplastic API
# ---------------------------------------------------------------------------

def inspect_tensor(api_url: str, tensor: str) -> dict:
    return _api(api_url, "/neuroplastic/inspect", {"tensor": tensor})


def checkpoint_tensor(api_url: str, tensor: str):
    _api(api_url, "/neuroplastic/checkpoint", {"tensor": tensor})


def restore_tensor(api_url: str, tensor: str):
    _api(api_url, "/neuroplastic/restore", {"tensor": tensor})


def modify_tensor(api_url: str, tensor: str, op: str, **kwargs):
    payload = {"tensor": tensor, "op": op}
    payload.update(kwargs)
    return _api(api_url, "/neuroplastic/modify", payload)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(api_url: str, problems: list = None, trials: int = 1) -> dict:
    """Run quick eval. Returns {total, passed, accuracy, per_category, details}."""
    if problems is None:
        problems = EVAL_PROBLEMS

    results = []
    for prob in problems:
        passes = 0
        for _ in range(trials):
            try:
                resp = _post(api_url.rstrip("/") + "/v1/chat/completions", {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prob["q"]}],
                    "max_tokens": 512,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                }, timeout=60)
                content = resp["choices"][0]["message"].get("content", "")
                if prob["check"](content):
                    passes += 1
            except RuntimeError:
                pass
        results.append({
            "id": prob["id"],
            "category": prob["category"],
            "passed": passes,
            "trials": trials,
            "rate": passes / trials,
        })

    total = len(results) * trials
    passed = sum(r["passed"] for r in results)

    # Per category
    cats = {}
    for r in results:
        cat = r["category"]
        cats.setdefault(cat, {"passed": 0, "total": 0})
        cats[cat]["passed"] += r["passed"]
        cats[cat]["total"] += r["trials"]
    for cat in cats:
        cats[cat]["accuracy"] = cats[cat]["passed"] / cats[cat]["total"] if cats[cat]["total"] else 0

    return {
        "total": total,
        "passed": passed,
        "accuracy": passed / total if total else 0,
        "per_category": cats,
        "details": results,
    }


# ---------------------------------------------------------------------------
# Logging / Monitoring
# ---------------------------------------------------------------------------

class Monitor:
    """Real-time monitoring for long runs."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = output_dir / "search_log.jsonl"
        self.status_path = output_dir / "live_status.json"
        self.start_time = time.time()
        self.events = 0

    def log_event(self, event_type: str, data: dict):
        """Append event to JSONL log."""
        entry = {
            "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - self.start_time, 1),
            "event": event_type,
            **data,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.events += 1

    def update_status(self, status: dict):
        """Overwrite live status file."""
        status["last_updated"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status["elapsed_s"] = round(time.time() - self.start_time, 1)
        status["elapsed_human"] = _fmt_duration(time.time() - self.start_time)
        with open(self.status_path, "w") as f:
            json.dump(status, f, indent=2)

    def save_results(self, filename: str, data):
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Phase A: Sensitivity Scan
# ---------------------------------------------------------------------------

def run_sensitivity_scan(api_url: str, monitor: Monitor,
                         output_dir: Path = None) -> list[dict]:
    """Probe every tensor type in every layer with small perturbations.

    For each tensor, checkpoint → scale by 0.95 → eval → restore → scale by
    1.05 → eval → restore. Record the accuracy delta from baseline.

    Returns list of {tensor, layer, baseline_acc, scale_down_acc, scale_up_acc,
    sensitivity} sorted by sensitivity (most sensitive first).
    """
    print("=" * 70)
    print("PHASE A: SENSITIVITY SCAN")
    print("=" * 70)

    # Baseline eval (3 trials for stability)
    print("\nRunning baseline eval (3 trials)...")
    baseline = run_eval(api_url, trials=3)
    baseline_acc = baseline["accuracy"]
    baseline_cats = baseline["per_category"]
    print(f"  Baseline accuracy: {baseline_acc:.1%}")
    for cat, info in baseline_cats.items():
        print(f"    {cat}: {info['passed']}/{info['total']} = {info['accuracy']:.1%}")

    monitor.log_event("baseline", {
        "accuracy": baseline_acc,
        "per_category": baseline_cats,
    })
    monitor.update_status({
        "phase": "A_sensitivity_scan",
        "baseline_accuracy": baseline_acc,
        "scans_completed": 0,
        "scans_total": "computing...",
    })

    # Build scan list
    scan_targets = []
    for layer_idx in MAMBA_LAYERS:
        for tensor_suffix in MAMBA_TENSORS:
            scan_targets.append((layer_idx, "mamba", tensor_suffix))
    for layer_idx in ATTENTION_LAYERS:
        for tensor_suffix in ATTENTION_TENSORS:
            scan_targets.append((layer_idx, "attention", tensor_suffix))
    for layer_idx in MOE_LAYERS:
        for tensor_suffix in MOE_TENSORS:
            scan_targets.append((layer_idx, "moe", tensor_suffix))

    total_scans = len(scan_targets)
    print(f"\n{total_scans} tensors to scan across {len(set(l for l,_,_ in scan_targets))} layers")

    monitor.update_status({
        "phase": "A_sensitivity_scan",
        "baseline_accuracy": baseline_acc,
        "scans_completed": 0,
        "scans_total": total_scans,
    })

    # Load partial results if resuming
    if output_dir is None:
        output_dir = monitor.output_dir
    partial_path = output_dir / "sensitivity_map_partial.json"
    results = []
    scanned_tensors = set()
    if partial_path.exists():
        with open(partial_path) as f:
            results = json.load(f)
        scanned_tensors = {r["tensor"] for r in results}
        print(f"  Resuming from {len(results)} previously scanned tensors")

    for scan_idx, (layer_idx, layer_type, tensor_suffix) in enumerate(scan_targets):
        tensor_name = f"model.layers.{layer_idx}.{tensor_suffix}"

        # Skip already scanned
        if tensor_name in scanned_tensors:
            continue

        # Check tensor exists and get info
        try:
            info = inspect_tensor(api_url, tensor_name)
        except RuntimeError:
            print(f"  [{scan_idx+1}/{total_scans}] {tensor_name}: NOT FOUND, skipping")
            continue

        shape = info.get("shape", [])
        norm = info.get("norm", 0)

        print(f"  [{scan_idx+1}/{total_scans}] {tensor_name} "
              f"shape={shape} norm={norm:.2f}", end="", flush=True)

        # Checkpoint
        checkpoint_tensor(api_url, tensor_name)

        # Test scale down
        try:
            modify_tensor(api_url, tensor_name, "scale", value=PROBE_SCALE_DOWN)
            eval_down = run_eval(api_url, trials=1)
            acc_down = eval_down["accuracy"]
            cats_down = eval_down["per_category"]
        except RuntimeError as exc:
            print(f" SCALE_DOWN_FAIL({exc})", end="")
            acc_down = baseline_acc
            cats_down = {}
        restore_tensor(api_url, tensor_name)

        # Test scale up
        try:
            modify_tensor(api_url, tensor_name, "scale", value=PROBE_SCALE_UP)
            eval_up = run_eval(api_url, trials=1)
            acc_up = eval_up["accuracy"]
            cats_up = eval_up["per_category"]
        except RuntimeError as exc:
            print(f" SCALE_UP_FAIL({exc})", end="")
            acc_up = baseline_acc
            cats_up = {}
        restore_tensor(api_url, tensor_name)

        # Compute sensitivity = max absolute accuracy change
        delta_down = acc_down - baseline_acc
        delta_up = acc_up - baseline_acc
        sensitivity = max(abs(delta_down), abs(delta_up))

        result = {
            "tensor": tensor_name,
            "layer": layer_idx,
            "layer_type": layer_type,
            "tensor_suffix": tensor_suffix,
            "shape": shape,
            "norm": norm,
            "baseline_acc": baseline_acc,
            "acc_scale_down": acc_down,
            "acc_scale_up": acc_up,
            "delta_down": round(delta_down, 4),
            "delta_up": round(delta_up, 4),
            "sensitivity": round(sensitivity, 4),
        }
        results.append(result)

        indicator = ""
        if sensitivity > 0.15:
            indicator = " *** HIGH ***"
        elif sensitivity > 0.05:
            indicator = " * moderate *"

        print(f"  down={delta_down:+.2f} up={delta_up:+.2f} "
              f"sens={sensitivity:.2f}{indicator}")

        monitor.log_event("scan", {
            "scan_idx": scan_idx + 1,
            "tensor": tensor_name,
            "sensitivity": sensitivity,
            "delta_down": delta_down,
            "delta_up": delta_up,
        })
        monitor.update_status({
            "phase": "A_sensitivity_scan",
            "baseline_accuracy": baseline_acc,
            "scans_completed": len(results),
            "scans_total": total_scans,
            "last_tensor": tensor_name,
            "last_sensitivity": sensitivity,
            "high_sensitivity_count": sum(
                1 for r in results if r["sensitivity"] > 0.15),
        })

        # Save partial results every 5 scans
        if len(results) % 5 == 0:
            with open(partial_path, "w") as f:
                json.dump(results, f, indent=2)

    # Sort by sensitivity
    results.sort(key=lambda r: r["sensitivity"], reverse=True)
    monitor.save_results("sensitivity_map.json", results)

    # Print top 20
    print(f"\n{'='*70}")
    print("TOP 20 MOST SENSITIVE TENSORS")
    print(f"{'='*70}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:2d}. {r['tensor']:50s} sens={r['sensitivity']:.3f} "
              f"(down={r['delta_down']:+.3f} up={r['delta_up']:+.3f})")

    return results


# ---------------------------------------------------------------------------
# Phase B: Targeted Search
# ---------------------------------------------------------------------------

def run_targeted_search(api_url: str, sensitivity_map: list[dict],
                        monitor: Monitor, max_steps: int = 100,
                        top_n: int = 30):
    """Score-driven search over the most sensitive tensors.

    Uses a systematic grid: for each of the top_n most sensitive tensors,
    try a range of scale factors. Accept modifications that improve eval score,
    reject those that don't. Compound accepted modifications.
    """
    print(f"\n{'='*70}")
    print("PHASE B: TARGETED SEARCH")
    print(f"{'='*70}")

    # Select top candidates
    candidates = sensitivity_map[:top_n]
    print(f"Searching over top {len(candidates)} most sensitive tensors")

    # Baseline eval
    print("\nRunning baseline eval (3 trials)...")
    baseline = run_eval(api_url, trials=3)
    current_acc = baseline["accuracy"]
    current_cats = baseline["per_category"]
    peak_acc = current_acc
    peak_step = 0
    print(f"  Starting accuracy: {current_acc:.1%}")

    # Checkpoint all candidate tensors
    print("Checkpointing candidate tensors...")
    for c in candidates:
        checkpoint_tensor(api_url, c["tensor"])

    # Scale factors to try (from conservative to aggressive)
    scale_factors = [0.97, 0.98, 0.99, 1.01, 1.02, 1.03]

    # Track modifications
    accepted_mods = []
    rejected_count = 0
    step = 0
    stale_count = 0  # consecutive rejections

    trajectory = [{
        "step": 0,
        "accuracy": current_acc,
        "per_category": current_cats,
        "action": "baseline",
        "tensor": None,
        "scale": None,
    }]

    monitor.update_status({
        "phase": "B_targeted_search",
        "current_accuracy": current_acc,
        "peak_accuracy": peak_acc,
        "peak_step": peak_step,
        "steps_completed": 0,
        "steps_total": max_steps,
        "accepted": 0,
        "rejected": 0,
        "stale_streak": 0,
        "accepted_modifications": [],
    })

    for step in range(1, max_steps + 1):
        # Pick candidate: cycle through candidates, trying different scales
        cand_idx = (step - 1) % len(candidates)
        scale_idx = ((step - 1) // len(candidates)) % len(scale_factors)
        candidate = candidates[cand_idx]
        scale = scale_factors[scale_idx]

        tensor_name = candidate["tensor"]

        # Determine expected direction from sensitivity scan
        # If scale_down improved accuracy, favor scales < 1
        # If scale_up improved accuracy, favor scales > 1
        if candidate["delta_down"] > candidate["delta_up"] and scale > 1:
            # Skip: sensitivity says down is better but we'd scale up
            # Still try occasionally (every 3rd)
            if step % 3 != 0:
                continue
        if candidate["delta_up"] > candidate["delta_down"] and scale < 1:
            if step % 3 != 0:
                continue

        print(f"\n  Step {step}/{max_steps}: {tensor_name} "
              f"scale={scale:.3f}", end="", flush=True)

        # Checkpoint current state of this tensor
        checkpoint_tensor(api_url, tensor_name)

        # Apply modification
        try:
            modify_tensor(api_url, tensor_name, "scale", value=scale)
        except RuntimeError as exc:
            print(f" MODIFY_FAIL: {exc}")
            restore_tensor(api_url, tensor_name)
            continue

        # Eval
        eval_result = run_eval(api_url, trials=2)
        new_acc = eval_result["accuracy"]
        new_cats = eval_result["per_category"]

        # Accept/reject
        improved = new_acc > current_acc
        # Also accept lateral moves if they improve a weak category
        if not improved and new_acc == current_acc:
            # Check if any previously-failing category improved
            for cat, info in new_cats.items():
                old_info = current_cats.get(cat, {"accuracy": 0})
                if info["accuracy"] > old_info.get("accuracy", 0):
                    improved = True
                    break

        if improved:
            delta = new_acc - current_acc
            current_acc = new_acc
            current_cats = new_cats
            stale_count = 0
            accepted_mods.append({
                "step": step,
                "tensor": tensor_name,
                "scale": scale,
                "accuracy_before": current_acc - delta,
                "accuracy_after": current_acc,
                "delta": round(delta, 4),
            })
            if current_acc > peak_acc:
                peak_acc = current_acc
                peak_step = step
            print(f"  ACCEPT acc={current_acc:.1%} "
                  f"(+{delta:+.3f}) peak={peak_acc:.1%}")
        else:
            restore_tensor(api_url, tensor_name)
            rejected_count += 1
            stale_count += 1
            print(f"  REJECT acc={new_acc:.1%} vs {current_acc:.1%}")

        trajectory.append({
            "step": step,
            "accuracy": current_acc,
            "per_category": current_cats,
            "action": "accepted" if improved else "rejected",
            "tensor": tensor_name,
            "scale": scale,
            "eval_accuracy": new_acc,
        })

        monitor.log_event("search_step", {
            "step": step,
            "tensor": tensor_name,
            "scale": scale,
            "eval_acc": new_acc,
            "current_acc": current_acc,
            "accepted": improved,
            "peak_acc": peak_acc,
        })
        monitor.update_status({
            "phase": "B_targeted_search",
            "current_accuracy": current_acc,
            "peak_accuracy": peak_acc,
            "peak_step": peak_step,
            "steps_completed": step,
            "steps_total": max_steps,
            "accepted": len(accepted_mods),
            "rejected": rejected_count,
            "stale_streak": stale_count,
            "accepted_modifications": [
                {"step": m["step"], "tensor": m["tensor"],
                 "scale": m["scale"], "delta": m["delta"]}
                for m in accepted_mods[-20:]  # last 20
            ],
        })

        # Save trajectory periodically
        if step % 10 == 0:
            monitor.save_results("search_trajectory.json", trajectory)
            monitor.save_results("accepted_modifications.json", accepted_mods)

        # If 15 consecutive rejections, shuffle candidate order
        if stale_count >= 15:
            print(f"\n  [STALE] {stale_count} consecutive rejections — "
                  f"reshuffling candidates")
            import random
            random.shuffle(candidates)
            stale_count = 0

        # Capability floor check every 20 steps
        if step % 20 == 0:
            floor_eval = run_eval(api_url, trials=3)
            floor_acc = floor_eval["accuracy"]
            print(f"\n  [FLOOR CHECK] Accuracy: {floor_acc:.1%}")
            if floor_acc < 0.30:
                print(f"  ABORT: accuracy {floor_acc:.1%} below floor")
                # Restore all
                for c in candidates:
                    try:
                        restore_tensor(api_url, c["tensor"])
                    except RuntimeError:
                        pass
                break

    # Final save
    monitor.save_results("search_trajectory.json", trajectory)
    monitor.save_results("accepted_modifications.json", accepted_mods)

    # Final eval
    print(f"\n{'='*70}")
    print("FINAL EVALUATION (3 trials)")
    print(f"{'='*70}")
    final_eval = run_eval(api_url, trials=3)
    final_acc = final_eval["accuracy"]
    print(f"  Final accuracy: {final_acc:.1%}")
    print(f"  Peak accuracy: {peak_acc:.1%} (step {peak_step})")
    print(f"  Baseline accuracy: {trajectory[0]['accuracy']:.1%}")
    print(f"  Accepted: {len(accepted_mods)}, Rejected: {rejected_count}")

    for cat, info in final_eval["per_category"].items():
        print(f"    {cat}: {info['accuracy']:.1%}")

    summary = {
        "baseline_accuracy": trajectory[0]["accuracy"],
        "final_accuracy": final_acc,
        "peak_accuracy": peak_acc,
        "peak_step": peak_step,
        "total_steps": step,
        "accepted": len(accepted_mods),
        "rejected": rejected_count,
        "final_per_category": final_eval["per_category"],
        "accepted_modifications": accepted_mods,
    }
    monitor.save_results("search_summary.json", summary)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 7: Expanded Search")
    parser.add_argument("--api-url", default="http://spark-129a.local:30000")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--phase", choices=["scan", "search", "all"],
                        default="all")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Max search steps in Phase B")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Top N tensors to search over")
    parser.add_argument("--skip-scan", action="store_true",
                        help="Skip scan, load existing sensitivity_map.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    monitor = Monitor(output_dir)

    # Wait for API
    print("Checking API...", end="", flush=True)
    endpoint = args.api_url.rstrip("/") + "/v1/models"
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

    # Phase A: Sensitivity Scan
    if args.phase in ("scan", "all"):
        if args.skip_scan:
            print("Loading existing sensitivity map...")
            with open(output_dir / "sensitivity_map.json") as f:
                sensitivity_map = json.load(f)
        else:
            sensitivity_map = run_sensitivity_scan(args.api_url, monitor)
    else:
        with open(output_dir / "sensitivity_map.json") as f:
            sensitivity_map = json.load(f)

    # Phase B: Targeted Search
    if args.phase in ("search", "all"):
        run_targeted_search(
            args.api_url, sensitivity_map, monitor,
            max_steps=args.max_steps,
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()
