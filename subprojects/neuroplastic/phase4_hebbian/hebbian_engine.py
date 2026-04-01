#!/usr/bin/env python3
"""Phase 4: Hebbian Learning Engine

Implements online Hebbian updates to Mamba SSM decay parameters (mixer.A) using
the neuroplastic API's trace infrastructure. Activation patterns during inference
drive per-head weight updates: active heads get slower decay (longer memory),
dormant heads get faster decay (pruned pathways). Homeostatic regulation prevents
runaway strengthening via L2 norm normalization and a BCM-like adaptive threshold.

Architecture overview:
  1. trace_input()    — install hooks, trigger inference, collect activation data
  2. compute_updates() — per-head Hebbian deltas from per_head_magnitudes
  3. apply_updates()   — issue add_slice operations to the API
  4. apply_homeostasis() — normalize back to original L2 norm after updates
  5. run_episode()    — sequence all above steps across a list of inputs

Hebbian rule for Mamba A (large negative decay parameter):
  mixer.A values are negative (mean ≈ -171).
  - More positive (less negative) A → slower decay → longer memory
  - More negative A → faster decay → shorter memory

  For each head h:
    relative_activity = activity_h / max_activity_across_heads
    If activity_h > median:
      delta = +eta * relative_activity   (less negative → slower decay)
    Else:
      delta = -eta * relative_activity   (more negative → faster decay)

  BCM threshold: delta is attenuated when activity_h is near the running mean,
  sharpening the active/dormant distinction over time.

Usage:
  engine = HebbianEngine("http://spark-129a.local:30000", [44, 46, 48, 50])
  results = engine.run_episode(["Start with 5. Add 3. Total?", ...])
"""

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

# Small universal decay applied before each Hebbian update (slight forgetting).
# Scales all target A tensors by this factor before per-head updates are computed.
GLOBAL_DECAY_FACTOR = 0.999

# BCM threshold: attenuation is applied when activity is within this fraction
# of the running mean. Sharpens discrimination between active and dormant heads.
BCM_DEAD_ZONE = 0.1  # 10% of running mean = dead zone (no update)

# Number of elements in mixer.A for these Mamba layers (64 heads per layer)
HEADS_PER_LAYER = 64

# Request timeout for API calls (seconds)
API_TIMEOUT = 120

# Inference trigger: we generate 1 token at temperature 0 to minimise side effects.
# enable_thinking=false avoids the Nemotron reasoning expansion problem.
INFERENCE_PAYLOAD = {
    "model": MODEL_NAME,
    "max_tokens": 1,
    "temperature": 0.0,
    "chat_template_kwargs": {"enable_thinking": False},
}


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _post(url: str, payload: dict, timeout: int = API_TIMEOUT) -> dict:
    """POST JSON payload to url and return parsed JSON response."""
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
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:400]}") from exc
    except (urllib.error.URLError, ConnectionResetError, OSError) as exc:
        raise RuntimeError(f"Network error calling {url}: {exc}") from exc


def _neuroplastic(api_url: str, endpoint: str, payload: dict,
                   max_retries: int = 3) -> dict:
    """Call a neuroplastic API endpoint with retry on connection errors."""
    url = api_url.rstrip("/") + endpoint
    for attempt in range(max_retries):
        try:
            return _post(url, payload)
        except RuntimeError as exc:
            if attempt < max_retries - 1 and ("Connection" in str(exc) or "Network" in str(exc)):
                import time
                wait = 2 * (attempt + 1)
                print(f"  [RETRY] {endpoint} failed ({exc}), attempt {attempt + 1}, "
                      f"waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Unreachable: {url}")


# ---------------------------------------------------------------------------
# HebbianEngine
# ---------------------------------------------------------------------------

class HebbianEngine:
    """Online Hebbian learning on Mamba decay parameters via the neuroplastic API.

    Args:
        api_url: Base URL of the vLLM + neuroplastic API.
        target_layers: List of Mamba layer indices to update (e.g. [44, 46, 48, 50]).
        learning_rate: Hebbian step size (eta). Controls per-update delta magnitude.
            Typical range: 0.001 – 0.05. Default 0.01 gives ~0.06% shift on peak head.
    """

    def __init__(
        self,
        api_url: str,
        target_layers: list[int],
        learning_rate: float = 0.01,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.target_layers = list(target_layers)
        self.learning_rate = learning_rate

        # Running mean of per-head activity for BCM-like threshold adaptation.
        # Dict: layer_index -> list[float] of length HEADS_PER_LAYER.
        # Initialised lazily on first trace.
        self._running_mean: dict[int, list[float]] = {}

        # Trajectory log: list of episode records populated by run_episode().
        self.trajectory: list[dict] = []

    # ------------------------------------------------------------------
    # Tensor name helpers
    # ------------------------------------------------------------------

    def _tensor_name(self, layer: int) -> str:
        """Return the vLLM runtime tensor name for mixer.A at layer."""
        return f"model.layers.{layer}.mixer.A"

    # ------------------------------------------------------------------
    # trace_input
    # ------------------------------------------------------------------

    def trace_input(self, input_text: str) -> dict:
        """Run an activation trace for input_text.

        Steps:
          1. POST /neuroplastic/trace/start  — install forward hooks
          2. POST /v1/chat/completions       — trigger one inference pass
          3. POST /neuroplastic/trace/collect — gather hook data, remove hooks

        Args:
            input_text: The prompt to trace through the model.

        Returns:
            The parsed trace data dict from /neuroplastic/trace/collect.
            Structure:
              {
                "layers": {
                  "layer_44": {
                    "type": "mamba",
                    "head_norm_mean": [float]*64,  # per-head mean activation
                    "change_rate": [float per token],
                    ...
                  }, ...
                },
                "residual_stream": {...}
              }

        Raises:
            RuntimeError: On API or network failure after retries.
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Step 1: install hooks
                _neuroplastic(self.api_url, "/neuroplastic/trace/start", {})

                # Step 2: trigger inference (1 token, deterministic)
                inference_payload = dict(INFERENCE_PAYLOAD)
                inference_payload["messages"] = [
                    {"role": "user", "content": input_text}
                ]
                try:
                    _post(
                        self.api_url + "/v1/chat/completions",
                        inference_payload,
                        timeout=API_TIMEOUT,
                    )
                except RuntimeError as exc:
                    # Try to collect anyway to remove hooks
                    print(f"  [TRACE] Inference error (cleanup): {exc}",
                          file=sys.stderr)
                    try:
                        _neuroplastic(self.api_url, "/neuroplastic/trace/collect", {})
                    except RuntimeError:
                        pass
                    raise

                # Step 3: collect trace data
                trace_result = _neuroplastic(
                    self.api_url, "/neuroplastic/trace/collect", {}
                )
                return trace_result

            except RuntimeError as exc:
                if attempt < max_retries - 1:
                    import time
                    wait = 5 * (attempt + 1)
                    print(f"  [TRACE] Attempt {attempt + 1} failed: {exc}. "
                          f"Retrying in {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                else:
                    raise

    # ------------------------------------------------------------------
    # compute_updates
    # ------------------------------------------------------------------

    def compute_updates(self, trace_data: dict) -> dict[str, list[float]]:
        """Compute per-head Hebbian update deltas from trace activations.

        For each target layer:
          - Extract per_head_magnitudes from the trace (shape: [tokens][heads])
          - Compute mean activity per head across all tokens
          - Identify active (> median) vs dormant (< median) heads
          - Apply BCM-like attenuation near the running mean
          - Assign delta = +eta * relative_activity  (active, less negative)
                     or  -eta * relative_activity  (dormant, more negative)

        Args:
            trace_data: Dict returned by trace_input(). Must contain a "layers"
                        key with per-Mamba-layer activation data.

        Returns:
            Dict mapping tensor name -> list[float] of per-head deltas.
            Length of each list equals HEADS_PER_LAYER (64).

        Note:
            If trace data for a layer is missing or malformed, that layer is
            skipped with a warning and omitted from the returned dict.
        """
        updates: dict[str, list[float]] = {}
        layers_data = trace_data.get("layers", {})

        for layer_idx in self.target_layers:
            # The trace keys use layer index with "layer_" prefix
            layer_key = f"layer_{layer_idx}"
            layer_data = layers_data.get(layer_key)

            if layer_data is None:
                print(f"  [HEBBIAN] layer_{layer_idx} not in trace data — skipping",
                      file=sys.stderr)
                continue

            if layer_data.get("type") != "mamba":
                print(
                    f"  [HEBBIAN] layer_{layer_idx} type={layer_data.get('type')!r} "
                    f"(expected 'mamba') — skipping",
                    file=sys.stderr,
                )
                continue

            # The trace returns head_norm_mean: a flat [64] array of mean
            # per-head output norms across all tokens. This is the activity signal.
            head_norm_mean = layer_data.get("head_norm_mean")
            if not head_norm_mean:
                print(
                    f"  [HEBBIAN] layer_{layer_idx} missing head_norm_mean — skipping",
                    file=sys.stderr,
                )
                continue

            head_activity = list(head_norm_mean)
            n_heads = len(head_activity)

            # Layer-level dynamism (change_rate mean, if available)
            change_rate_data = layer_data.get("change_rate", [])
            if change_rate_data:
                _change_rate = sum(change_rate_data) / len(change_rate_data)
            else:
                _change_rate = 0.0  # unused in current rule but kept for future use

            # Statistics for this layer's activations
            max_activity = max(head_activity) if head_activity else 1.0
            if max_activity == 0.0:
                max_activity = 1.0  # avoid division by zero

            sorted_activities = sorted(head_activity)
            mid = len(sorted_activities) // 2
            if len(sorted_activities) % 2 == 0:
                median_activity = (sorted_activities[mid - 1] + sorted_activities[mid]) / 2.0
            else:
                median_activity = sorted_activities[mid]

            # Initialise or retrieve running mean for BCM threshold
            if layer_idx not in self._running_mean:
                self._running_mean[layer_idx] = list(head_activity)
            else:
                # Exponential moving average with alpha=0.1
                alpha = 0.1
                prev = self._running_mean[layer_idx]
                self._running_mean[layer_idx] = [
                    alpha * head_activity[h] + (1.0 - alpha) * prev[h]
                    for h in range(min(n_heads, len(prev)))
                ]

            running_mean = self._running_mean[layer_idx]

            # Compute per-head deltas
            deltas = []
            for h in range(n_heads):
                activity_h = head_activity[h]
                relative_activity = activity_h / max_activity

                # BCM dead zone: attenuate when activity is near running mean
                running_h = running_mean[h] if h < len(running_mean) else median_activity
                bcm_distance = abs(activity_h - running_h)
                bcm_scale = running_h * BCM_DEAD_ZONE
                if bcm_scale > 0 and bcm_distance < bcm_scale:
                    # Smoothly attenuate within dead zone
                    attenuation = bcm_distance / bcm_scale
                else:
                    attenuation = 1.0

                if activity_h > median_activity:
                    # Active head: make A less negative (slower decay, longer memory)
                    delta = +self.learning_rate * relative_activity * attenuation
                else:
                    # Dormant head: make A more negative (faster decay, prune)
                    delta = -self.learning_rate * relative_activity * attenuation

                deltas.append(delta)

            tensor_name = self._tensor_name(layer_idx)
            updates[tensor_name] = deltas

        return updates

    # ------------------------------------------------------------------
    # apply_updates
    # ------------------------------------------------------------------

    def apply_updates(self, updates: dict[str, list[float]]) -> None:
        """Apply per-head Hebbian deltas to target tensors via add_slice.

        Sends one POST /neuroplastic/modify per head that has a non-zero delta.
        Each operation uses add_slice to modify exactly one element (one head).

        Args:
            updates: Dict from compute_updates() mapping tensor name to
                     list of per-head float deltas.
        """
        for tensor_name, deltas in updates.items():
            n_applied = 0
            for h, delta in enumerate(deltas):
                if delta == 0.0:
                    continue
                payload = {
                    "tensor": tensor_name,
                    "op": "add_slice",
                    "params": {
                        "start": h,
                        "end": h + 1,
                        "value": delta,
                    },
                }
                try:
                    _neuroplastic(self.api_url, "/neuroplastic/modify", payload)
                    n_applied += 1
                except RuntimeError as exc:
                    print(
                        f"  [HEBBIAN] apply_updates failed for {tensor_name}[{h}]: {exc}",
                        file=sys.stderr,
                    )
            if n_applied:
                print(f"  [HEBBIAN] Applied {n_applied} updates to {tensor_name}")

    # ------------------------------------------------------------------
    # apply_homeostasis
    # ------------------------------------------------------------------

    def apply_homeostasis(self, original_norms: dict[str, float]) -> None:
        """Normalise each target tensor back to its pre-episode L2 norm.

        Uses the neuroplastic 'normalize' operation which rescales the tensor
        to the specified target L2 norm without changing its direction.

        Args:
            original_norms: Dict mapping tensor name to original L2 norm float.
                            Typically returned by checkpoint_all().
        """
        for tensor_name, target_norm in original_norms.items():
            if target_norm <= 0.0:
                print(
                    f"  [HOMEOSTASIS] Skipping {tensor_name}: "
                    f"original norm={target_norm:.4f} (zero or negative)",
                    file=sys.stderr,
                )
                continue
            payload = {
                "tensor": tensor_name,
                "op": "normalize",
                "params": {"target_norm": target_norm},
            }
            try:
                _neuroplastic(self.api_url, "/neuroplastic/modify", payload)
                print(
                    f"  [HOMEOSTASIS] Normalized {tensor_name} -> L2={target_norm:.4f}"
                )
            except RuntimeError as exc:
                print(
                    f"  [HOMEOSTASIS] normalize failed for {tensor_name}: {exc}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # _get_tensor_norm
    # ------------------------------------------------------------------

    def _get_tensor_norm(self, tensor_name: str) -> float:
        """Inspect a tensor and return its current L2 norm.

        Returns:
            The norm as a float, or 0.0 if the inspect call fails.
        """
        payload = {"tensor": tensor_name, "per_head": False}
        try:
            result = _neuroplastic(self.api_url, "/neuroplastic/inspect", payload)
            return float(result.get("norm", 0.0))
        except RuntimeError as exc:
            print(
                f"  [ENGINE] _get_tensor_norm failed for {tensor_name}: {exc}",
                file=sys.stderr,
            )
            return 0.0

    # ------------------------------------------------------------------
    # checkpoint_all
    # ------------------------------------------------------------------

    def checkpoint_all(self, name: str) -> dict[str, float]:
        """Checkpoint all target layer tensors and record their L2 norms.

        Args:
            name: Checkpoint name passed to the API (e.g. "hebbian_episode_0").

        Returns:
            Dict mapping tensor_name -> L2 norm at time of checkpoint.
            Used later as the homeostasis target.
        """
        norms: dict[str, float] = {}
        for layer_idx in self.target_layers:
            tensor_name = self._tensor_name(layer_idx)
            norm = self._get_tensor_norm(tensor_name)
            norms[tensor_name] = norm

            payload = {
                "tensor": tensor_name,
                "name": name,
            }
            try:
                _neuroplastic(self.api_url, "/neuroplastic/checkpoint", payload)
                print(
                    f"  [CHECKPOINT] {tensor_name} norm={norm:.4f} saved as '{name}'"
                )
            except RuntimeError as exc:
                print(
                    f"  [CHECKPOINT] Failed for {tensor_name}: {exc}",
                    file=sys.stderr,
                )

        return norms

    # ------------------------------------------------------------------
    # restore_all
    # ------------------------------------------------------------------

    def restore_all(self, name: str) -> None:
        """Restore all target layer tensors from a named checkpoint.

        Args:
            name: Checkpoint name to restore from.
        """
        for layer_idx in self.target_layers:
            tensor_name = self._tensor_name(layer_idx)
            payload = {
                "tensor": tensor_name,
                "name": name,
            }
            try:
                _neuroplastic(self.api_url, "/neuroplastic/restore", payload)
                print(f"  [RESTORE] {tensor_name} restored from '{name}'")
            except RuntimeError as exc:
                print(
                    f"  [RESTORE] Failed for {tensor_name}: {exc}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # _apply_global_decay
    # ------------------------------------------------------------------

    def _apply_global_decay(self) -> None:
        """Apply a small universal scale to all target tensors (slight forgetting).

        mixer.A is negative, so scale_slice by 0.999 makes all values slightly
        less extreme (moves them toward zero), which means slightly slower decay
        overall. This prevents runaway strengthening across episodes.
        """
        for layer_idx in self.target_layers:
            tensor_name = self._tensor_name(layer_idx)
            payload = {
                "tensor": tensor_name,
                "op": "scale_slice",
                "params": {
                    "start": 0,
                    "end": HEADS_PER_LAYER,
                    "value": GLOBAL_DECAY_FACTOR,
                },
            }
            try:
                _neuroplastic(self.api_url, "/neuroplastic/modify", payload)
            except RuntimeError as exc:
                print(
                    f"  [DECAY] scale failed for {tensor_name}: {exc}",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # run_episode
    # ------------------------------------------------------------------

    def run_episode(
        self,
        inputs: list[str],
        record_trajectory: bool = True,
    ) -> list[dict]:
        """Process a sequence of inputs with Hebbian updates between each.

        Protocol for each input:
          1. Checkpoint all targets (record original norms)
          2. Apply global decay (GLOBAL_DECAY_FACTOR) before Hebbian updates
          3. Trace the input
          4. Compute per-head deltas from trace activations
          5. Apply Hebbian deltas via add_slice
          6. Apply homeostasis (normalize to original norms)
          7. Log the step record

        Args:
            inputs: Ordered list of prompt strings to process.
            record_trajectory: If True, append each step record to self.trajectory.

        Returns:
            List of step record dicts, one per input. Each record contains:
              - "input_idx": int
              - "input_text": str
              - "timestamp": ISO timestamp string
              - "trace_ok": bool (True if trace succeeded)
              - "layers_updated": list of layer indices that received updates
              - "update_summary": dict of tensor_name -> {min, max, mean, nonzero}
              - "original_norms": dict of tensor_name -> float
        """
        step_records = []

        print(f"\n[HebbianEngine] Starting episode: {len(inputs)} inputs, "
              f"layers={self.target_layers}, eta={self.learning_rate}")

        for idx, input_text in enumerate(inputs):
            print(f"\n  Step {idx + 1}/{len(inputs)}: {input_text[:80]!r}")
            t0 = time.time()

            step_record: dict = {
                "input_idx": idx,
                "input_text": input_text,
                "timestamp": _iso_now(),
                "trace_ok": False,
                "layers_updated": [],
                "update_summary": {},
                "original_norms": {},
            }

            # 1. Checkpoint and record norms
            checkpoint_name = f"episode_step_{idx}"
            original_norms = self.checkpoint_all(checkpoint_name)
            step_record["original_norms"] = original_norms

            # 2. Global decay before per-head update
            self._apply_global_decay()

            # 3. Trace
            try:
                trace_data = self.trace_input(input_text)
                step_record["trace_ok"] = True
            except RuntimeError as exc:
                print(f"  [EPISODE] Trace failed at step {idx}: {exc}", file=sys.stderr)
                # Restore to last checkpoint and continue
                self.restore_all(checkpoint_name)
                if record_trajectory:
                    self.trajectory.append(step_record)
                step_records.append(step_record)
                continue

            # 4. Compute updates
            updates = self.compute_updates(trace_data)

            if not updates:
                print("  [EPISODE] No updates computed (empty trace or no mamba layers)")
                if record_trajectory:
                    self.trajectory.append(step_record)
                step_records.append(step_record)
                continue

            # 5. Apply Hebbian deltas
            self.apply_updates(updates)

            # 6. Apply homeostasis
            self.apply_homeostasis(original_norms)

            # 7. Build summary for logging
            layers_updated = []
            update_summary = {}
            for tensor_name, deltas in updates.items():
                nonzero = sum(1 for d in deltas if d != 0.0)
                if nonzero > 0:
                    layers_updated.append(tensor_name)
                    update_summary[tensor_name] = {
                        "min": min(deltas),
                        "max": max(deltas),
                        "mean": sum(deltas) / len(deltas) if deltas else 0.0,
                        "nonzero_heads": nonzero,
                    }

            step_record["layers_updated"] = layers_updated
            step_record["update_summary"] = update_summary
            step_record["elapsed_seconds"] = round(time.time() - t0, 2)

            print(
                f"  [EPISODE] Step {idx + 1} done in {step_record['elapsed_seconds']}s. "
                f"Updated {len(layers_updated)} tensors."
            )

            if record_trajectory:
                self.trajectory.append(step_record)
            step_records.append(step_record)

        print(f"\n[HebbianEngine] Episode complete. {len(step_records)} steps processed.")
        return step_records


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    """Return current UTC time as an ISO 8601 string."""
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Smoke test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HebbianEngine smoke test — traces a single input and reports deltas."
    )
    parser.add_argument(
        "--api-url",
        default="http://spark-129a.local:30000",
        help="vLLM + neuroplastic API base URL",
    )
    parser.add_argument(
        "--input",
        default="A bag starts with 5 apples. Remove 2. Add 3. How many apples are there?",
        help="Input text to trace",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Hebbian learning rate (eta)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute deltas but do not apply them",
    )
    args = parser.parse_args()

    engine = HebbianEngine(
        api_url=args.api_url,
        target_layers=[44, 46, 48, 50],
        learning_rate=args.learning_rate,
    )

    print("Tracing input...")
    try:
        trace_data = engine.trace_input(args.input)
    except RuntimeError as e:
        print(f"Trace failed: {e}", file=sys.stderr)
        sys.exit(1)

    print("Computing updates...")
    updates = engine.compute_updates(trace_data)

    for tensor_name, deltas in updates.items():
        nonzero = sum(1 for d in deltas if d != 0.0)
        print(f"\n{tensor_name}:")
        print(f"  Non-zero deltas: {nonzero}/{len(deltas)}")
        print(f"  Delta range: [{min(deltas):.6f}, {max(deltas):.6f}]")
        print(f"  Delta mean:  {sum(deltas)/len(deltas):.6f}")
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        print(f"  Positive (slower decay): {pos}, Negative (faster decay): {neg}")

    if not args.dry_run:
        print("\nApplying updates...")
        engine.apply_updates(updates)
        print("Done.")
    else:
        print("\n[dry-run] Updates NOT applied.")
