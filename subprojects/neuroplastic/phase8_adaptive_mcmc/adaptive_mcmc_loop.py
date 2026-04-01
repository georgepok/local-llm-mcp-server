#!/usr/bin/env python3
"""Phase 8: Adaptive MCMC Self-Modification Loop.

A Python sampler (not the model) selects which tensor/operation/magnitude to
try. Accept/reject history is tracked and fed back to bias future proposals
toward combinations that have worked before.

No model involvement in proposal selection — pure statistical adaptation.

Monitoring:
  results/live_status.json     — real-time status (updated every cycle)
  results/cycle_log.jsonl      — per-cycle log (append-only)
  results/sampler_state.json   — sampler checkpoint (saved every 10 cycles)
  results/sampler_evolution.json — distribution snapshots at cycles 50/100/200
"""

import argparse
import json
import random
import signal
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ---------------------------------------------------------------------------
# Import fast_eval from phase7
# ---------------------------------------------------------------------------

_PHASE7_DIR = Path(__file__).parent.parent / "phase7_autoresearch"
sys.path.insert(0, str(_PHASE7_DIR))

try:
    from fast_eval import run_fast_eval
except ImportError as exc:
    raise ImportError(
        f"Cannot import fast_eval from {_PHASE7_DIR}. "
        f"Ensure phase7_autoresearch/fast_eval.py exists."
    ) from exc

from adaptive_sampler import AdaptiveMCMCSampler

# ---------------------------------------------------------------------------
# Candidate pools
# ---------------------------------------------------------------------------

CANDIDATE_TENSORS = []

# Mamba SSM layers
for _layer in [0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25, 28, 30, 32, 35, 37, 39, 41, 44, 46, 48, 50]:
    CANDIDATE_TENSORS.append(f"model.layers.{_layer}.mixer.A")
    CANDIDATE_TENSORS.append(f"model.layers.{_layer}.mixer.D")
    CANDIDATE_TENSORS.append(f"model.layers.{_layer}.mixer.dt_bias")

# Attention layers
for _layer in [5, 12, 19, 26, 33, 42]:
    CANDIDATE_TENSORS.append(f"model.layers.{_layer}.mixer.qkv_proj.weight")
    CANDIDATE_TENSORS.append(f"model.layers.{_layer}.mixer.o_proj.weight")

# MoE layers (subset)
for _layer in [1, 13, 27, 38, 45, 49, 51]:
    CANDIDATE_TENSORS.append(f"model.layers.{_layer}.mixer.gate.weight")

CANDIDATE_OPS = ["scale", "add", "scale_slice", "add_noise"]

# ---------------------------------------------------------------------------
# HTTP helpers (same pattern as phase7)
# ---------------------------------------------------------------------------

API_TIMEOUT = 120


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
    raise RuntimeError(f"Unreachable: {url}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Neuroplastic API wrappers
# ---------------------------------------------------------------------------

def inspect_tensor(api_url: str, tensor: str) -> dict:
    return _api(api_url, "/neuroplastic/inspect", {"tensor": tensor})


def checkpoint_tensor(api_url: str, tensor: str, name: str = "default") -> dict:
    return _api(api_url, "/neuroplastic/checkpoint", {"tensor": tensor, "name": name})


def restore_tensor(api_url: str, tensor: str, name: str = "default") -> dict:
    return _api(api_url, "/neuroplastic/restore", {"tensor": tensor, "name": name})


def modify_tensor(api_url: str, tensor: str, op: str, **kwargs) -> dict:
    payload = {"tensor": tensor, "op": op}
    payload.update(kwargs)
    return _api(api_url, "/neuroplastic/modify", payload)


# ---------------------------------------------------------------------------
# Build modification params from sampled values
# ---------------------------------------------------------------------------

def build_modify_params(op: str, magnitude: float) -> dict:
    """Convert (op, magnitude) into the kwargs dict for modify_tensor."""
    if op == "scale":
        return {"value": magnitude}
    elif op == "add":
        return {"value": magnitude}
    elif op == "scale_slice":
        n_heads = 64
        block_size = random.choice([8, 16, 32])
        start = random.randint(0, n_heads - block_size)
        end = start + block_size
        return {"start": start, "end": end, "value": magnitude}
    elif op == "add_noise":
        return {"scale": magnitude, "seed": random.randint(0, 2**31 - 1)}
    else:
        return {"value": magnitude}


# ---------------------------------------------------------------------------
# Monitoring helpers
# ---------------------------------------------------------------------------

class Monitor:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "regression_checks").mkdir(exist_ok=True)
        (self.output_dir / "session_summaries").mkdir(exist_ok=True)
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
        status["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status["elapsed_s"] = round(time.time() - self.start_time, 1)
        elapsed = time.time() - self.start_time
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        status["elapsed_human"] = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
        with open(self.status_path, "w") as f:
            json.dump(status, f, indent=2)

    def save(self, filename: str, data: dict):
        path = self.output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# API readiness check
# ---------------------------------------------------------------------------

def wait_for_api(api_url: str, timeout_s: int = 120):
    print("Checking API...", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(
                api_url.rstrip("/") + "/v1/models", method="GET")
            with urllib.request.urlopen(req, timeout=10):
                print(" OK")
                return
        except Exception:
            time.sleep(5)
    print(" TIMEOUT")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_adaptive_mcmc(
    api_url: str,
    output_dir: Path,
    max_cycles: int = 200,
    problems_path: str = "eval_problems.json",
    resume: bool = False,
) -> dict:
    """Adaptive MCMC self-modification loop.

    The sampler, not the model, selects tensor/op/magnitude. Accept/reject
    history updates the sampler's probability distributions.
    """
    monitor = Monitor(output_dir)
    shutdown = {"requested": False}

    def _handle_signal(signum, frame):
        print(f"\n[SIGNAL] Shutdown requested (signal {signum})")
        shutdown["requested"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # ------------------------------------------------------------------
    # Sampler state — load or fresh
    # ------------------------------------------------------------------
    sampler_path = output_dir / "sampler_state.json"
    loop_state_path = output_dir / "loop_state.json"
    evolution_snapshots = {}  # cycle -> sampler stats snapshot

    if resume and sampler_path.exists() and loop_state_path.exists():
        sampler = AdaptiveMCMCSampler.load(str(sampler_path))
        with open(loop_state_path) as f:
            state = json.load(f)
        print(f"Resumed from cycle {state['cycle']}, "
              f"score {state['current_score']}/{state['total']}, "
              f"sampler history {sampler.get_stats()['n_history']} entries")
        # Reload any partial evolution snapshots
        evol_path = output_dir / "sampler_evolution.json"
        if evol_path.exists():
            with open(evol_path) as f:
                evolution_snapshots = json.load(f)
    else:
        sampler = AdaptiveMCMCSampler(temperature=2.0)
        state = {
            "cycle": 0,
            "baseline_score": 0,
            "current_score": 0,
            "best_score": 0,
            "total": 20,
            "total_accepted": 0,
            "total_rejected": 0,
            "accepted_mods": [],
            "checkpointed_tensors": [],
        }

    # ------------------------------------------------------------------
    # API readiness
    # ------------------------------------------------------------------
    wait_for_api(api_url)

    # ------------------------------------------------------------------
    # Baseline eval
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    baseline = run_fast_eval(api_url, problems_path, verbose=True)
    state["total"] = baseline["total"]

    if state["cycle"] == 0:
        state["baseline_score"] = baseline["score"]
        state["current_score"] = baseline["score"]
        state["best_score"] = baseline["score"]

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

    if baseline["accuracy"] >= 0.95:
        print("\n[WARNING] Baseline >= 95% — limited room for improvement.")
    elif baseline["accuracy"] <= 0.30:
        print("\n[WARNING] Baseline <= 30% — model may be unable to improve.")

    # ------------------------------------------------------------------
    # Main MCMC loop
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"ADAPTIVE MCMC LOOP — Target: {max_cycles} cycles")
    print(f"{'='*70}")

    start_cycle = state["cycle"] + 1

    for cycle in range(start_cycle, max_cycles + 1):
        if shutdown["requested"]:
            print(f"\n[SHUTDOWN] Saving state at cycle {cycle - 1}")
            break

        cycle_start = time.time()
        state["cycle"] = cycle

        # Temperature annealing — 2.0 → 1.0 over the run
        sampler.decay_temperature(cycle, total_cycles=max_cycles)

        print(f"\n--- Cycle {cycle}/{max_cycles} "
              f"(score: {state['current_score']}/{state['total']}, "
              f"best: {state['best_score']}, "
              f"T={sampler.temperature:.3f}) ---")

        # ------------------------------------------------------------------
        # Sampler proposes tensor / op / magnitude
        # ------------------------------------------------------------------
        tensor = sampler.sample_tensor(CANDIDATE_TENSORS)
        op = sampler.sample_op(CANDIDATE_OPS)
        magnitude = sampler.sample_magnitude(tensor, op)
        mod_params = build_modify_params(op, magnitude)

        print(f"  Sampler proposal: {tensor}  op={op}  magnitude={magnitude:.5g}  "
              f"params={mod_params}")

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        print(f"  Checkpointing {tensor}...", end="", flush=True)
        try:
            checkpoint_tensor(api_url, tensor)
            if tensor not in state["checkpointed_tensors"]:
                state["checkpointed_tensors"].append(tensor)
            print(" OK")
        except RuntimeError as exc:
            print(f" FAILED: {exc}")
            sampler.update(tensor, op, magnitude, accepted=False, score_delta=0)
            monitor.log_cycle({
                "event": "checkpoint_fail",
                "cycle": cycle,
                "tensor": tensor,
                "op": op,
                "error": str(exc)[:200],
            })
            continue

        # ------------------------------------------------------------------
        # Apply modification
        # ------------------------------------------------------------------
        print(f"  Applying {op}...", end="", flush=True)
        try:
            modify_tensor(api_url, tensor, op, **mod_params)
            print(" OK")
        except RuntimeError as exc:
            print(f" FAILED: {exc}")
            try:
                restore_tensor(api_url, tensor)
            except RuntimeError:
                pass
            sampler.update(tensor, op, magnitude, accepted=False, score_delta=0)
            monitor.log_cycle({
                "event": "modify_fail",
                "cycle": cycle,
                "tensor": tensor,
                "op": op,
                "params": mod_params,
                "error": str(exc)[:200],
            })
            continue

        # ------------------------------------------------------------------
        # Eval
        # ------------------------------------------------------------------
        print(f"  Running eval...", flush=True)
        eval_result = run_fast_eval(api_url, problems_path, verbose=False)
        new_score = eval_result["score"]
        score_before = state["current_score"]
        delta = new_score - score_before

        # ------------------------------------------------------------------
        # Accept / reject (ties accepted)
        # ------------------------------------------------------------------
        if new_score >= score_before:
            decision = "KEPT"
            state["current_score"] = new_score
            state["total_accepted"] += 1
            if new_score > state["best_score"]:
                state["best_score"] = new_score
            state["accepted_mods"].append({
                "cycle": cycle,
                "tensor": tensor,
                "op": op,
                "params": mod_params,
                "magnitude": magnitude,
                "score_before": score_before,
                "score_after": new_score,
            })
            # Re-checkpoint so future restores return to this improved state
            try:
                checkpoint_tensor(api_url, tensor)
            except RuntimeError:
                pass
            print(f"  >>> KEPT: {score_before}→{new_score} (+{delta})")
        else:
            decision = "REJECTED"
            state["total_rejected"] += 1
            try:
                restore_tensor(api_url, tensor)
            except RuntimeError as exc:
                print(f"  [WARN] Restore failed: {exc}")
            print(f"  <<< REJECTED: {score_before}→{new_score} ({delta})")

        accepted = decision == "KEPT"

        # ------------------------------------------------------------------
        # Update sampler
        # ------------------------------------------------------------------
        sampler.update(tensor, op, magnitude, accepted=accepted, score_delta=delta)

        cycle_elapsed = time.time() - cycle_start

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        monitor.log_cycle({
            "event": "cycle",
            "cycle": cycle,
            "tensor": tensor,
            "op": op,
            "magnitude": magnitude,
            "params": mod_params,
            "score_before": score_before,
            "score_after": new_score,
            "decision": decision,
            "per_category": eval_result["per_category"],
            "elapsed_ms": round(cycle_elapsed * 1000),
            "sampler_temperature": sampler.temperature,
        })

        n_tried = state["total_accepted"] + state["total_rejected"]
        acceptance_rate = state["total_accepted"] / n_tried if n_tried > 0 else 0.0

        monitor.update_status({
            "phase": "adaptive_mcmc",
            "cycle": cycle,
            "max_cycles": max_cycles,
            "current_score": state["current_score"],
            "best_score": state["best_score"],
            "baseline_score": state["baseline_score"],
            "total_problems": state["total"],
            "accuracy": state["current_score"] / state["total"],
            "total_accepted": state["total_accepted"],
            "total_rejected": state["total_rejected"],
            "acceptance_rate": acceptance_rate,
            "sampler_temperature": sampler.temperature,
            "last_decision": decision,
            "last_tensor": tensor,
            "last_op": op,
            "last_magnitude": magnitude,
            "last_delta": delta,
            "accepted_mods_count": len(state["accepted_mods"]),
            "top_tensors": sorted(
                sampler.tensor_scores.items(),
                key=lambda x: x[1], reverse=True
            )[:5],
            "op_scores": dict(sampler.op_scores),
            "recent_accepted": [
                {
                    "cycle": m["cycle"],
                    "tensor": m["tensor"],
                    "op": m["op"],
                    "delta": m["score_after"] - m["score_before"],
                }
                for m in state["accepted_mods"][-10:]
            ],
        })

        # ------------------------------------------------------------------
        # Save sampler state every 10 cycles
        # ------------------------------------------------------------------
        if cycle % 10 == 0:
            sampler.save(str(sampler_path))
            with open(loop_state_path, "w") as f:
                json.dump(state, f, indent=2)
            print(f"  [CHECKPOINT] Sampler state saved at cycle {cycle}")

        # ------------------------------------------------------------------
        # Evolution snapshot at cycles 50, 100, 200 (and max_cycles)
        # ------------------------------------------------------------------
        snapshot_cycles = {50, 100, 200, max_cycles}
        if cycle in snapshot_cycles:
            evolution_snapshots[str(cycle)] = sampler.get_stats()
            monitor.save("sampler_evolution.json", evolution_snapshots)
            print(f"  [SNAPSHOT] Sampler distribution recorded at cycle {cycle}")

        # ------------------------------------------------------------------
        # Regression check every 20 cycles
        # ------------------------------------------------------------------
        if cycle % 20 == 0:
            print(f"\n  [REGRESSION CHECK] Running 3-trial eval at cycle {cycle}...")
            scores = []
            for trial in range(3):
                r = run_fast_eval(api_url, problems_path)
                scores.append(r["score"])
            avg_score = sum(scores) / len(scores)
            warn_threshold = state["baseline_score"] * 0.70
            status_tag = "OK"
            if avg_score < warn_threshold:
                status_tag = "WARN"
                print(f"  [REGRESSION WARN] avg={avg_score:.1f} < "
                      f"70% of baseline ({warn_threshold:.1f})")
            else:
                print(f"  Regression check: scores={scores}, avg={avg_score:.1f}  [{status_tag}]")

            reg_data = {
                "cycle": cycle,
                "scores": scores,
                "avg_score": avg_score,
                "baseline_score": state["baseline_score"],
                "warn_threshold": warn_threshold,
                "status": status_tag,
            }
            reg_path = output_dir / "regression_checks" / f"cycle_{cycle:04d}.json"
            with open(reg_path, "w") as f:
                json.dump(reg_data, f, indent=2)

            monitor.log_cycle({
                "event": "regression_check",
                "cycle": cycle,
                "scores": scores,
                "avg_score": avg_score,
                "status": status_tag,
            })

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    sampler.save(str(sampler_path))
    with open(loop_state_path, "w") as f:
        json.dump(state, f, indent=2)

    # Ensure final snapshot is recorded
    evolution_snapshots[str(state["cycle"])] = sampler.get_stats()
    monitor.save("sampler_evolution.json", evolution_snapshots)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_tried = state["total_accepted"] + state["total_rejected"]
    acceptance_rate = state["total_accepted"] / n_tried if n_tried > 0 else 0.0

    print(f"\n{'='*70}")
    print("ADAPTIVE MCMC COMPLETE")
    print(f"{'='*70}")
    print(f"  Cycles:           {state['cycle']}")
    print(f"  Baseline score:   {state['baseline_score']}/{state['total']} "
          f"({state['baseline_score']/state['total']:.0%})")
    print(f"  Best score:       {state['best_score']}/{state['total']} "
          f"({state['best_score']/state['total']:.0%})")
    print(f"  Current score:    {state['current_score']}/{state['total']}")
    print(f"  Accepted:         {state['total_accepted']}")
    print(f"  Rejected:         {state['total_rejected']}")
    print(f"  Acceptance rate:  {acceptance_rate:.1%}")
    print(f"\nTop tensors by sampler score:")
    top_tensors = sorted(sampler.tensor_scores.items(), key=lambda x: x[1], reverse=True)
    for t, s in top_tensors[:10]:
        print(f"  {s:+.2f}  {t}")
    print(f"\nOp scores: " +
          "  ".join(f"{op}: {s:+.2f}" for op, s in sorted(sampler.op_scores.items())))

    ts = time.strftime("%Y%m%dT%H%M%S")
    session_summary = {
        "cycles_completed": state["cycle"],
        "baseline_score": state["baseline_score"],
        "best_score": state["best_score"],
        "current_score": state["current_score"],
        "total_problems": state["total"],
        "total_accepted": state["total_accepted"],
        "total_rejected": state["total_rejected"],
        "acceptance_rate": acceptance_rate,
        "sampler_final_stats": sampler.get_stats(),
        "accepted_modifications": state["accepted_mods"],
    }
    monitor.save(f"session_summaries/session_{ts}.json", session_summary)

    return state


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 8: Adaptive MCMC Self-Modification Loop")
    parser.add_argument(
        "--api-url", default="http://localhost:30000",
        help="vLLM / neuroplastic API base URL (default: http://localhost:30000)")
    parser.add_argument(
        "--max-cycles", type=int, default=200,
        help="Maximum number of MCMC cycles (default: 200)")
    parser.add_argument(
        "--output-dir", default="results",
        help="Directory for logs and checkpoints (default: ./results)")
    parser.add_argument(
        "--problems", default="eval_problems.json",
        help="Path to eval_problems.json (default: eval_problems.json)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved sampler state and loop state in --output-dir")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    run_adaptive_mcmc(
        api_url=args.api_url,
        output_dir=output_dir,
        max_cycles=args.max_cycles,
        problems_path=args.problems,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
