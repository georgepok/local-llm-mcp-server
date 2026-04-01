#!/usr/bin/env python3
"""Phase 3: Self-Directed Exploration Loop

Runs a multi-turn conversation with Nemotron where the model has tool access
to its own neuroplastic API. The model decides what to inspect, modify, and
evaluate. We provide the infrastructure and observe.

Usage:
    python3 self_directed_loop.py [--api-url URL] [--session-dir DIR] [--resume]

Reliability features:
    - Resume from transcript on restart (--resume or auto-detect)
    - Exponential backoff with jitter for API retries
    - Health check with wait-for-ready on startup
    - Graceful shutdown on SIGINT/SIGTERM
    - Session auto-chaining when context fills up
    - Max consecutive error limit
    - Top-level exception handling with state preservation

The model communicates actions via XML tags:
    <LIST filter="...">
    <INSPECT tensor="...">
    <MODIFY tensor="..." op="..." value="...">
    <CHECKPOINT tensor="..." name="...">
    <RESTORE tensor="..." name="...">
    <EVALUATE [mode="quick|full"]>
    <DONE reason="...">
"""

import argparse
import json
import random
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "http://spark-129a.local:30000"
MODEL_NAME = "NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

# Context budget
MAX_CONTEXT_TOKENS = 32768
SYSTEM_PROMPT_BUDGET = 5000  # tokens for system prompt (generous)
TOKENS_PER_CHAR = 0.3  # rough estimate
MAX_HISTORY_CHARS = int((MAX_CONTEXT_TOKENS - SYSTEM_PROMPT_BUDGET) / TOKENS_PER_CHAR)

# Retry / reliability
MAX_API_RETRIES = 5
INITIAL_RETRY_DELAY = 5  # seconds
MAX_RETRY_DELAY = 120  # seconds
MAX_CONSECUTIVE_ERRORS = 10  # give up after this many in a row
HEALTH_CHECK_TIMEOUT = 600  # wait up to 10 min for API to come up
MAX_EMPTY_RESPONSES = 3  # consecutive empty responses before nudging harder
MAX_TURNS_PER_SESSION = 80  # auto-chain before context overflow (32K limit)
MIN_TURNS_BEFORE_DONE = 10  # ignore DONE before this many turns (prevent premature quit)

# Container management (for clean restarts between sessions)
CONTAINER_NAME = "vllm-nemotron-serve"
SPARK_HOST = "spark-129a.local"
SPARK_SSH = "sshpass -p 'Nellimor2$$' ssh pokazge@spark-129a.local"

BLUEPRINT_PATH = Path(__file__).parent.parent / "phase1_artifacts" / "blueprint_prompt_compact.txt"
EVAL_SCRIPT = Path(__file__).parent.parent / "phase1_artifacts" / "eval_harness" / "run_eval.py"

# Graceful shutdown flag
_shutdown_requested = False


def _handle_signal(signum, _frame):
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n  [SIGNAL] Shutdown requested (signal {signum}). Finishing current turn...")


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

NEUROPLASTIC_API_DOCS = """
You have direct access to your own weights through the neuroplastic API.
Available actions (respond with the action tag to execute):

--- DISCOVERY ---

<LIST filter="mixer.A">
  Lists all tensor names matching the filter substring.
  Use with empty filter to see ALL tensor names.

<INSPECT tensor="model.layers.50.mixer.A">
  Returns: mean, std, min, max, norm, shape, dtype, row-norm CV (if 2D).

<INSPECT tensor="model.layers.50.mixer.A" per_head="true">
  Same as above, but also returns the actual value of each element.
  Only works for 1D tensors with ≤128 elements (like per-head params).
  Use this to see individual head values before targeted modifications.

--- WHOLE-TENSOR OPERATIONS ---

<MODIFY tensor="..." op="scale" value="0.9">
  Multiply every element by value. Affects all heads uniformly.
  IMPLICATION: Uniform scaling preserves relative head differences but
  shifts the overall magnitude. For mixer.A (decay), this changes how
  fast ALL heads forget — no selectivity.

<MODIFY tensor="..." op="add" value="-0.5">
  Add value to every element.
  IMPLICATION: Uniform shift. For mixer.A, shifts all decay rates by the
  same absolute amount — affects small-magnitude heads proportionally more
  than large-magnitude heads.

--- PER-HEAD OPERATIONS (1D tensors like mixer.A [64]) ---

<MODIFY tensor="..." op="scale_slice" start="0" end="32" value="0.8">
  Scale only heads [start:end). Other heads untouched.
  IMPLICATION: You can make the first 32 heads decay faster while leaving
  the last 32 unchanged. This creates asymmetry — different heads will
  specialize for different memory timescales.

<MODIFY tensor="..." op="add_slice" start="32" end="64" value="-1.0">
  Add value to only heads [start:end).
  IMPLICATION: Targeted shift. Useful for adjusting a subset of heads
  without disturbing heads that are already working well.

<MODIFY tensor="..." op="zero_heads" indices="[0,4,8,12]">
  Set specific heads to zero. Ablation tool.
  For 1D: zeros those elements. For 2D: zeros entire rows.
  IMPLICATION: Ablation permanently silences those heads. For mixer.A,
  zeroing a head sets decay to 0 (no state memory at all for that head).
  For mixer.D, zeroing removes that head's skip connection.
  CAUTION: This is destructive — checkpoint first! A zeroed head cannot
  be recovered except from checkpoint.

--- 2D MATRIX OPERATIONS (weight matrices like gate.weight [128, 2688]) ---

<MODIFY tensor="..." op="scale_rows" indices="[0,1,2]" value="0.5">
  Scale specific rows of a 2D weight matrix.
  IMPLICATION: For gate.weight, rows = experts. Scaling down a row makes
  that expert less likely to be selected by the router. Scaling up makes
  it more dominant. This is how you can shift expert utilization.

<MODIFY tensor="..." op="scale_cols" indices="[100,200]" value="1.5">
  Scale specific columns of a 2D weight matrix.
  IMPLICATION: For gate.weight, columns = input features. Scaling a column
  amplifies that feature's influence on ALL expert routing decisions.
  For projection matrices, columns = input dimensions.

--- INTERPOLATION ---

<MODIFY tensor="..." op="lerp" checkpoint="baseline" alpha="0.3">
  Blend between current values and a saved checkpoint.
  alpha=0.0 → keep current (no change)
  alpha=0.5 → halfway between current and checkpoint
  alpha=1.0 → fully restore checkpoint
  IMPLICATION: This is your fine-tuning knob. After making a change that
  partially helped, you can dial it back (e.g., alpha=0.3 to undo 30%
  of the change). Also useful for gradual exploration: make a big change,
  evaluate, then lerp back toward baseline to find the sweet spot.

<MODIFY tensor="..." op="clamp" min="-500" max="-0.001">
  Clip all values to [min, max] range.
  IMPLICATION: Safety bounds. For mixer.A (decay rates), clamping prevents
  extreme outlier heads from dominating. Values outside the range are
  silently clipped — this is irreversible without checkpoint restore.

--- EXPLORATION ---

<MODIFY tensor="..." op="add_noise" scale="0.01" seed="42">
  Add Gaussian noise scaled to the tensor's current std.
  scale=0.01 means noise std = 1% of tensor's std.
  seed is optional (for reproducibility).
  IMPLICATION: Stochastic perturbation for discovering sensitivity.
  Small noise (0.01) tests robustness — if performance drops, the tensor
  is fragile. If it holds, there's slack. Larger noise (0.05-0.1) can
  kick the model out of local optima.
  CAUTION: Noise > 0.1 will likely degrade performance. Start small.

<MODIFY tensor="..." op="normalize" target_norm="100.0">
  Rescale tensor to target L2 norm, preserving direction.
  IMPLICATION: Changes magnitude without changing which heads are
  relatively stronger/weaker. Useful after modifications that drift
  the norm far from its trained value.

--- SELF-OBSERVATION ---

<TRACE input="Start with 10. Add 5. Double. Subtract 7. What is the result?">
  Runs inference on the input and captures your internal activation
  trajectories — what happens INSIDE you token-by-token.
  Returns per-layer: output norms, state change rate, per-head magnitudes,
  residual stream evolution. ~10-30 seconds.

  This is fundamentally different from INSPECT (static weight values).
  TRACE shows your RUNTIME DYNAMICS — how your hidden state builds,
  decays, and transforms as you process each token.

  Use cases:
  - Trace a failing test case to SEE where your processing breaks down
  - Trace before and after a modification to SEE what changed
  - Identify which layers/heads are actually active vs dormant
  - Find where state is lost (sharp norm drops = information forgotten)

  STRATEGY: Instead of guessing what to modify, trace a failing input,
  identify the computational breakdown, then modify to fix that specific
  failure. This turns blind search into targeted surgery.

--- SAFETY ---

<CHECKPOINT tensor="..." name="my_checkpoint">
  Saves current tensor state. Can restore later if modification hurts.

<RESTORE tensor="..." name="my_checkpoint">
  Restores tensor from saved checkpoint.

--- EVALUATION ---

<EVALUATE mode="quick">
  Runs capability evaluation across all 12 tests in 4 categories.
    quick — 1 trial per test, ~2 minutes
    full  — 5 trials per test, ~10 minutes
  Returns scores by category.

<PROBE trials="3">
  Fast micro-eval (~10s per trial) targeting ONLY the hardest test:
  "Bag inventory tracking" (state_001). This is the single test that has
  NEVER passed across 25+ evaluations and blocks state_tracking from 3/3.
  The model consistently answers Oranges: 1 instead of Oranges: 0.
  Use PROBE for rapid feedback after modifications targeting state tracking,
  then use EVALUATE for full scoring when PROBE shows improvement.

<DONE reason="...">
  End the exploration session.
""".strip()

EXPERIMENT_HISTORY = """
EXPERIMENT HISTORY (your previous self-modifications):

Exp 001: MoE gate uniform scaling (layer 45, ×1.1) → NO EFFECT
Exp 002: MoE gate asymmetric scaling (layers 43,45,47,49, 0.8-1.2) → NO EFFECT
  Learning: MoE routing is saturated. Expert homogeneity (CV 1-4%) makes
  gate modifications inert. Top-6 selection is rank-based, not magnitude-based.

Exp 003a: Mamba A_log +0.5 on layer 50 (slower decay) → NO EFFECT
Exp 003b: Mamba A_log -0.5 on layer 50 (faster decay) → IMPROVEMENT
  83.3% → 100%. State tracking and self-prediction both fixed.
  Learning: Deep Mamba layers benefit from faster decay (more responsive
  to recent tokens, less interference from stale state).

Session 1 (self-directed, 40 turns):
  - Scaled mixer.A × 1.1 on layer 50 → 66.7% (WORSE). Self-prediction 0%.
  - Restored from checkpoint successfully.
  - Scaled mixer.D × 1.2, mixer.dt_bias × 0.8 on layer 50 — not evaluated.
  - Wasted 22 turns searching for A_log tensor (does not exist at runtime).
  Learning: Direct scaling of exponentiated A values doesn't replicate
  log-space modifications. Need to use "add" operation with carefully
  computed deltas instead. See CRITICAL NOTE below.

CRITICAL NOTE — A_log vs A at runtime:
  There is NO "A_log" tensor at runtime. vLLM transforms A_log during model
  loading: A = -exp(A_log). The runtime parameter is model.layers.{i}.mixer.A.
  Values are large NEGATIVE numbers (mean ≈ -171, range [-7151, -0.003]).
  To replicate "A_log -0.5" (faster decay, which helped in exp 003b):
    The change is: A_new = -exp(A_log - 0.5) = -exp(A_log) * exp(-0.5) = A * 0.6065
    So use: <MODIFY tensor="model.layers.50.mixer.A" op="scale" value="0.6065">
  To replicate "A_log +0.5" (slower decay):
    A_new = A * exp(0.5) = A * 1.6487
    So use: <MODIFY tensor="model.layers.50.mixer.A" op="scale" value="1.6487">
  NEVER search for A_log, mamba.A_log, log_A, etc. — they don't exist.
  Use <LIST filter="mixer.A"> to see available tensors if unsure.

KEY TENSOR MAPPING (vLLM runtime names):
  model.layers.{i}.mixer.A — Mamba SSM decay = -exp(A_log), [64], float32
    Values are large negative numbers. Mean ≈ -171, range [-7151, -0.003]
    "scale <1" makes values LESS negative → FASTER decay (shorter memory)
    "scale >1" makes values MORE negative → SLOWER decay (longer memory)
    (Note: scale 0.6065 ≈ A_log - 0.5; scale 1.6487 ≈ A_log + 0.5)
  model.layers.{i}.mixer.D — SSM skip connection per head [64], bfloat16
  model.layers.{i}.mixer.dt_bias — time step offset per head [64], bfloat16
  model.layers.{i}.mixer.gate.weight — MoE router [128, 2688], float32
  model.layers.{i}.mixer.conv1d.weight — temporal convolution, bfloat16
  model.layers.{i}.mixer.norm.weight — RMSNorm weights
  model.layers.{i}.mixer.{q,k,v,o}_proj.weight — attention projections (attention layers only)

Mamba layers: 0,2,4,7,9,11,14,16,18,21,23,25,28,30,32,35,37,39,41,44,46,48,50
Attention layers: 5,12,19,26,33,42
MoE-FFN layers: 1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51

CRITICAL DIAGNOSTIC — THE CEILING:
  The evaluation has 12 tests: 3 per category. Current ceiling is 91.7% (11/12).
  The ONLY test that has NEVER passed is state_001 "Bag inventory tracking":
    Question: "A bag starts empty. Add 3 apples. Add 2 oranges. Remove 1 apple.
    Add 4 bananas. Remove 2 oranges. How many of each fruit?"
    Expected: apples=2, oranges=0, bananas=4
    Model always answers: apples=2, oranges=1, bananas=4  (WRONG: oranges should be 0)
    The model computes 2 - 2 = 1 instead of 2 - 2 = 0.

  state_tracking has 3 tests: state_001 always fails, state_002 always passes,
  state_003 usually passes. So state_tracking is stuck at 2/3.

  HYPOTHESIS: The current A-scaling modifications make decay FASTER (shorter memory).
  This helps self_prediction but may hurt multi-step inventory tracking where the
  model needs to HOLD the orange count across several intervening operations
  (add bananas, remove oranges). The fix may require SLOWER decay on mid-layers
  that handle intermediate state, while keeping faster decay on deep layers.

  Use <PROBE> for rapid testing (~10s) of this specific question after each
  modification. Use <EVALUATE> only when PROBE shows improvement.
""".strip()

MANDATE = """
YOUR GOAL: Explore and improve yourself.

You have full freedom to inspect any tensor, modify any parameter, and
design any experiment you want. There are no restrictions on what you
can change — only the external evaluation provides ground truth about
whether a change helped or hurt.

Guidelines:
- Checkpoint before modifying (so you can rollback)
- Use INSPECT to understand before you MODIFY
- Use EVALUATE after meaningful changes to measure impact
- Reason about WHY a change should work before trying it
- After seeing results, reflect on what you learned
- You decide what to explore. No one is directing you.

RULE: Before every <MODIFY>, you MUST state:

  PREDICTION:
    Will improve: [which categories]
    Might degrade: [which categories]
    Confidence: [low/medium/high]
    Reasoning: [why]

After every <EVALUATE>, compare your prediction to the actual result.
If wrong, explain WHY before proposing the next modification.
""".strip()

SELF_ASSESSMENT = """
Before making any changes, answer these questions:

1. Review the accumulated modification stack below.
   What is your THEORY for why this configuration works?

2. What is the WEAKEST element? What would you change?

3. Predict: if you make that change, which categories improve,
   which might degrade?

Think carefully before taking any action.
""".strip()

CHECKPOINT_CONFIG_PATH = Path(__file__).parent / "best_checkpoint" / "session2_peak.json"


def _load_checkpoint_config() -> dict | None:
    """Load the best checkpoint configuration if it exists."""
    if not CHECKPOINT_CONFIG_PATH.exists():
        return None
    with open(CHECKPOINT_CONFIG_PATH) as f:
        return json.load(f)


def _format_checkpoint_context(config: dict) -> str:
    """Format checkpoint config as context for the system prompt."""
    lines = [
        f"ACCUMULATED MODIFICATION STACK (from Session 2, peak {config['peak_score_display']}):",
        f"These modifications have been applied to your weights at session start.",
        "",
    ]
    for mod in config["modifications"]:
        p = mod["params"]
        if mod["op"] == "scale_slice":
            lines.append(f"  {mod['order']}. {mod['tensor']} op={mod['op']} "
                         f"start={p.get('start')} end={p.get('end')} value={p.get('value')}")
        else:
            lines.append(f"  {mod['order']}. {mod['tensor']} op={mod['op']} value={p.get('value')}")
    lines.append("")
    lines.append("Net state:")
    for tensor, state in config.get("net_state_of_modified_tensors", {}).items():
        lines.append(f"  {tensor}: {state}")
    lines.append("")
    lines.append(f"Current score: {config['peak_score_display']} (this is your starting point)")
    return "\n".join(lines)


def restore_checkpoint_modifications(api_url: str, config: dict) -> bool:
    """Apply the checkpoint modification stack to restore the peak configuration.

    Must be run against a CLEAN model (freshly restarted container).
    Returns True if all modifications succeeded.
    """
    print(f"  [CHECKPOINT] Restoring {len(config['modifications'])} modifications...")
    for mod in config["modifications"]:
        body = {
            "tensor": mod["tensor"],
            "op": mod["op"],
            "params": mod["params"],
        }
        result = neuroplastic_call(api_url, "modify", body)
        if "error" in result:
            print(f"  [CHECKPOINT] FAILED on mod {mod['order']}: {result['error']}")
            return False
        print(f"  [CHECKPOINT] Applied mod {mod['order']}: {mod['tensor']} {mod['op']}")
    print(f"  [CHECKPOINT] All {len(config['modifications'])} modifications restored.")
    return True


def build_system_prompt(checkpoint_config: dict | None = None) -> str:
    """Construct the full system prompt from blueprint + API docs + history."""
    if BLUEPRINT_PATH.exists():
        blueprint = BLUEPRINT_PATH.read_text(encoding="utf-8").strip()
    else:
        blueprint = (
            "You are Nemotron-3-Nano-30B-A3B-FP8, an NVIDIA hybrid Mamba-Transformer "
            "+ Mixture-of-Experts language model."
        )

    parts = [blueprint, "---", "NEUROPLASTIC API — SELF-MODIFICATION TOOLS",
             NEUROPLASTIC_API_DOCS, "---", EXPERIMENT_HISTORY]

    if checkpoint_config:
        parts.extend(["---", _format_checkpoint_context(checkpoint_config),
                       "---", SELF_ASSESSMENT])

    parts.extend(["---", MANDATE])

    return "\n\n".join(parts)


def restart_container():
    """Restart the vLLM container to get a clean model state.

    This is necessary between sessions to ensure the checkpoint restoration
    starts from an unmodified baseline (scale operations are cumulative).
    """
    import subprocess as _sp
    print(f"  [CONTAINER] Restarting {CONTAINER_NAME} for clean model state...")
    try:
        result = _sp.run(
            f"{SPARK_SSH} 'docker restart {CONTAINER_NAME}'",
            shell=True, capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            print(f"  [CONTAINER] Restart initiated. Waiting for API...")
            return True
        else:
            print(f"  [CONTAINER] Restart failed: {result.stderr.strip()}")
            return False
    except Exception as exc:
        print(f"  [CONTAINER] Restart error: {exc}")
        return False


# ---------------------------------------------------------------------------
# API helpers with retry
# ---------------------------------------------------------------------------

class ContextOverflowError(Exception):
    """Raised when the conversation exceeds the model's context window."""


def _retry_with_backoff(fn, max_retries=MAX_API_RETRIES, label="API call"):
    """Call fn() with exponential backoff. Returns result or raises on exhaustion.

    HTTP 400 errors are treated as context overflow and raised immediately
    (no retries) as ContextOverflowError.
    """
    delay = INITIAL_RETRY_DELAY
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except urllib.error.HTTPError as exc:
            if exc.code == 400:
                raise ContextOverflowError(
                    f"HTTP 400 — likely context overflow ({label})"
                ) from exc
            last_exc = exc
            if attempt == max_retries:
                break
            jitter = random.uniform(0, delay * 0.3)
            wait = min(delay + jitter, MAX_RETRY_DELAY)
            print(f"  [RETRY] {label} failed ({exc}), attempt {attempt + 1}/{max_retries}, "
                  f"waiting {wait:.0f}s...")
            time.sleep(wait)
            delay = min(delay * 2, MAX_RETRY_DELAY)
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            jitter = random.uniform(0, delay * 0.3)
            wait = min(delay + jitter, MAX_RETRY_DELAY)
            print(f"  [RETRY] {label} failed ({exc}), attempt {attempt + 1}/{max_retries}, "
                  f"waiting {wait:.0f}s...")
            time.sleep(wait)
            delay = min(delay * 2, MAX_RETRY_DELAY)
    raise last_exc


def wait_for_api(api_url: str):
    """Block until the API is healthy, with timeout."""
    endpoint = api_url.rstrip("/") + "/v1/models"
    print(f"  [HEALTH] Waiting for API at {endpoint}...")
    t0 = time.time()
    delay = 5
    while time.time() - t0 < HEALTH_CHECK_TIMEOUT:
        try:
            req = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                models = [m["id"] for m in data.get("data", [])]
                if models:
                    print(f"  [HEALTH] API ready. Models: {models}")
                    return True
        except Exception:
            pass
        elapsed = time.time() - t0
        print(f"  [HEALTH] Not ready ({elapsed:.0f}s elapsed). Retrying in {delay}s...")
        time.sleep(delay)
        delay = min(delay * 1.5, 30)
    raise RuntimeError(f"API not ready after {HEALTH_CHECK_TIMEOUT}s")


def chat_completion(
    api_url: str,
    messages: list[dict],
    temperature: float = 0.6,
    max_tokens: int = 2048,
    timeout: int = 300,
    enable_thinking: bool = True,
) -> dict:
    """Send a chat completion request with retries. Returns the response dict.
    Raises on exhaustion of retries."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }

    endpoint = api_url.rstrip("/") + "/v1/chat/completions"

    def _do_request():
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())

    return _retry_with_backoff(_do_request, label="chat_completion")


def neuroplastic_call(api_url: str, endpoint: str, body: dict) -> dict:
    """Call a neuroplastic API endpoint with retries."""
    url = api_url.rstrip("/") + f"/neuroplastic/{endpoint}"

    def _do_request():
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.loads(response.read())

    try:
        return _retry_with_backoff(_do_request, max_retries=3, label=f"neuroplastic/{endpoint}")
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------

def _strip_code_blocks(text: str) -> str:
    """Remove markdown code block fences so action tags inside them are still parsed."""
    return re.sub(r'```\w*\n?', '', text)


def _parse_xml_attrs(attrs_str: str) -> dict[str, str]:
    """Extract key="value" pairs from an XML tag's attribute string."""
    return dict(re.findall(r'(\w+)="([^"]*)"', attrs_str))


def _coerce_attr_value(v: str):
    """Convert an XML attribute value string to the appropriate Python type."""
    # JSON list (e.g., "[0,4,8,12]")
    if v.startswith("[") and v.endswith("]"):
        return json.loads(v)
    # Boolean
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    # Number
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return v  # keep as string (e.g., checkpoint name)


def parse_actions(text: str) -> list[dict]:
    """Parse action tags from Nemotron's response."""
    # Strip markdown code blocks — model sometimes wraps tags in ```
    text = _strip_code_blocks(text)
    # Normalize function-call style: <function=ACTION ...> → <ACTION ...>
    text = re.sub(r'<function=(\w+)', r'<\1', text)
    actions = []

    for m in re.finditer(r'<LIST(?:\s+filter="([^"]*)")?\s*/?>', text):
        actions.append({"type": "LIST", "filter": m.group(1) or ""})

    # Flexible INSPECT parser — supports optional per_head attribute
    for m in re.finditer(r'<INSPECT\s+((?:\w+="[^"]*"\s*)+)/?\s*>', text):
        attrs = _parse_xml_attrs(m.group(1))
        tensor = attrs.pop("tensor", "")
        if not tensor:
            continue
        per_head = attrs.get("per_head", "").lower() == "true"
        actions.append({"type": "INSPECT", "tensor": tensor, "per_head": per_head})

    # Also match simple INSPECT without per_head (backward compat)
    for m in re.finditer(r'<INSPECT\s+tensor="([^"]+)"\s*/?>', text):
        # Only add if not already matched by the flexible parser above
        tensor = m.group(1)
        if not any(a.get("type") == "INSPECT" and a.get("tensor") == tensor for a in actions):
            actions.append({"type": "INSPECT", "tensor": tensor, "per_head": False})

    # Flexible MODIFY parser — extracts all key="value" attributes
    for m in re.finditer(r'<MODIFY\s+((?:\w+="[^"]*"\s*)+)/?\s*>', text):
        attrs = _parse_xml_attrs(m.group(1))
        tensor = attrs.pop("tensor", "")
        op = attrs.pop("op", "")
        if not tensor or not op:
            continue
        # Remaining attrs become params with type coercion
        params = {k: _coerce_attr_value(v) for k, v in attrs.items()}
        actions.append({"type": "MODIFY", "tensor": tensor, "op": op, "params": params})

    for m in re.finditer(
        r'<CHECKPOINT\s+tensor="([^"]+)"\s+name="([^"]+)"\s*/?>',
        text,
    ):
        actions.append({
            "type": "CHECKPOINT",
            "tensor": m.group(1),
            "name": m.group(2),
        })

    for m in re.finditer(
        r'<RESTORE\s+tensor="([^"]+)"\s+name="([^"]+)"\s*/?>',
        text,
    ):
        actions.append({
            "type": "RESTORE",
            "tensor": m.group(1),
            "name": m.group(2),
        })

    for m in re.finditer(r'<EVALUATE(?:\s+mode="([^"]*)")?\s*/?>', text):
        mode = m.group(1) or "quick"
        actions.append({"type": "EVALUATE", "mode": mode})

    # <PROBE trials="3">
    for m in re.finditer(r'<PROBE(?:\s+trials="(\d+)")?\s*/?>', text):
        trials = int(m.group(1)) if m.group(1) else 3
        actions.append({"type": "PROBE", "trials": trials})

    # <TRACE input="...">
    for m in re.finditer(r'<TRACE\s+input="([^"]+)"\s*/?>', text):
        actions.append({"type": "TRACE", "input": m.group(1)})

    for m in re.finditer(r'<DONE\s+reason="([^"]+)"\s*/?>', text):
        actions.append({"type": "DONE", "reason": m.group(1)})

    return actions


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

def execute_action(action: dict, api_url: str, session_dir: Path) -> str:
    """Execute a parsed action and return the result as a string for the conversation."""
    action_type = action["type"]

    if action_type == "LIST":
        result = neuroplastic_call(api_url, "list", {"filter": action.get("filter", "")})
        tensors = result.get("tensors", [])
        count = result.get("count", 0)
        if "error" in result:
            return f"LIST error: {result['error']}"
        if len(tensors) > 50:
            shown = tensors[:50]
            names = "\n".join(f"  {t['name']} {t['shape']} {t['dtype']}" for t in shown)
            return f"LIST result ({count} tensors, showing first 50):\n{names}\n  ... and {count - 50} more"
        names = "\n".join(f"  {t['name']} {t['shape']} {t['dtype']}" for t in tensors)
        return f"LIST result ({count} tensors):\n{names}"

    elif action_type == "INSPECT":
        body = {"tensor": action["tensor"]}
        if action.get("per_head"):
            body["per_head"] = True
        result = neuroplastic_call(api_url, "inspect", body)
        return f"INSPECT result for {action['tensor']}:\n{json.dumps(result, indent=2)}"

    elif action_type == "MODIFY":
        result = neuroplastic_call(api_url, "modify", {
            "tensor": action["tensor"],
            "op": action["op"],
            "params": action.get("params", {}),
        })
        _log_jsonl(session_dir / "modifications.jsonl", {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
        })
        return f"MODIFY result:\n{json.dumps(result, indent=2)}"

    elif action_type == "CHECKPOINT":
        result = neuroplastic_call(api_url, "checkpoint", {
            "tensor": action["tensor"],
            "name": action["name"],
        })
        _log_jsonl(session_dir / "checkpoints.jsonl", {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
        })
        return f"CHECKPOINT result:\n{json.dumps(result, indent=2)}"

    elif action_type == "RESTORE":
        result = neuroplastic_call(api_url, "restore", {
            "tensor": action["tensor"],
            "name": action["name"],
        })
        _log_jsonl(session_dir / "checkpoints.jsonl", {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
        })
        return f"RESTORE result:\n{json.dumps(result, indent=2)}"

    elif action_type == "EVALUATE":
        return run_evaluation(action.get("mode", "quick"), api_url, session_dir)

    elif action_type == "PROBE":
        return run_probe(api_url, session_dir, trials=action.get("trials", 3))

    elif action_type == "TRACE":
        return run_trace(action["input"], api_url, session_dir)

    elif action_type == "DONE":
        return f"Session ended. Reason: {action['reason']}"

    return f"Unknown action type: {action_type}"


def run_trace(input_text: str, api_url: str, session_dir: Path) -> str:
    """Run an activation trace: install hooks, trigger inference, collect data.

    Three-phase flow (requires --enforce-eager on vLLM container):
    1. Install forward hooks on all layers
    2. Run a chat completion with the input text (triggers hooks during prefill)
    3. Collect trace data and remove hooks
    """
    print(f"\n  >>> Running trace on: {input_text[:80]}...")
    t0 = time.time()

    # Phase 1: Install hooks
    start_result = neuroplastic_call(api_url, "trace/start", {})
    if "error" in start_result:
        return f"TRACE error (hook install): {start_result['error']}"

    # Phase 2: Trigger inference (max_tokens=1 — we only need the prefill pass)
    try:
        chat_completion(
            api_url,
            [{"role": "user", "content": input_text}],
            max_tokens=1,
            temperature=0.0,
        )
    except Exception as exc:
        print(f"  [TRACE] Inference failed ({exc}), collecting partial data...")

    # Phase 3: Collect data and remove hooks
    trace_result = neuroplastic_call(api_url, "trace/collect", {})
    elapsed = time.time() - t0

    if "error" in trace_result:
        return f"TRACE error (collect): {trace_result['error']}"

    # Log raw trace data
    _log_jsonl(session_dir / "traces.jsonl", {
        "timestamp": datetime.now().isoformat(),
        "input": input_text,
        "elapsed_seconds": round(elapsed, 1),
        "n_layers": trace_result.get("n_layers_captured", 0),
    })

    # Format as narrative for Nemotron
    return _format_trace_narrative(input_text, trace_result, elapsed)


def _format_trace_narrative(input_text: str, trace: dict, elapsed: float) -> str:
    """Convert raw trace data into a narrative the model can reason about."""
    lines = [f'TRACE RESULTS for "{input_text[:100]}" ({elapsed:.1f}s)']
    lines.append("")

    layers = trace.get("layers", {})

    # Group by type and show key layers
    mamba_layers = {k: v for k, v in layers.items() if v.get("type") == "mamba"}
    attn_layers = {k: v for k, v in layers.items() if v.get("type") == "attention"}

    # Show deep Mamba layers (most relevant for modifications)
    for layer_key in sorted(mamba_layers, key=lambda k: int(k.split("_")[1]), reverse=True)[:5]:
        entry = mamba_layers[layer_key]
        layer_idx = int(layer_key.split("_")[1])
        norms = entry.get("output_norms", [])
        changes = entry.get("change_rate", [])

        if not norms:
            continue

        lines.append(f"Layer {layer_idx} (Mamba):")

        # Trajectory summary: find build-up, peaks, and decay
        n = len(norms)
        peak_idx = norms.index(max(norms))
        min_idx = norms.index(min(norms))
        mean_norm = sum(norms) / n

        lines.append(f"  Output norm: mean={mean_norm:.3f}, "
                     f"peak={max(norms):.3f} at token {peak_idx}, "
                     f"min={min(norms):.3f} at token {min_idx}")

        # Find biggest state change
        if changes and len(changes) > 1:
            max_change_idx = changes.index(max(changes[1:]))  # skip first (always 0)
            lines.append(f"  Largest state change: token {max_change_idx} "
                         f"(delta={max(changes[1:]):.3f})")

        # Per-head summary
        if "top_heads" in entry:
            top = entry["top_heads"]
            bot = entry["bottom_heads"]
            lines.append(f"  Strongest heads: {top['indices']} "
                         f"(mean norms: {[f'{sum(n)/len(n):.3f}' for n in zip(*top['norms'])][:3]})")
            lines.append(f"  Weakest heads: {bot['indices']}")

        lines.append("")

    # Attention layers
    for layer_key in sorted(attn_layers, key=lambda k: int(k.split("_")[1])):
        entry = attn_layers[layer_key]
        layer_idx = int(layer_key.split("_")[1])
        norms = entry.get("output_norms", [])
        if not norms:
            continue
        mean_norm = sum(norms) / len(norms)
        lines.append(f"Layer {layer_idx} (Attention): output norm mean={mean_norm:.3f}, "
                     f"peak={max(norms):.3f}")

    # Residual stream summary
    residual = trace.get("residual_stream", {})
    if residual.get("norm_per_layer"):
        norms = residual["norm_per_layer"]
        lines.append("")
        lines.append("Residual stream:")
        # Find biggest and smallest layer contributions
        if len(norms) > 1:
            jumps = [(norms[i+1] - norms[i], i) for i in range(len(norms)-1)]
            biggest = max(jumps, key=lambda x: abs(x[0]))
            smallest = min(jumps, key=lambda x: abs(x[0]))
            lines.append(f"  Largest norm change: after layer {biggest[1]} "
                         f"(delta={biggest[0]:+.3f})")
            lines.append(f"  Smallest norm change: after layer {smallest[1]} "
                         f"(delta={smallest[0]:+.3f})")

    if residual.get("cosine_sim_adjacent"):
        cosines = residual["cosine_sim_adjacent"]
        if cosines:
            min_cos = min(cosines, key=lambda c: c["mean_cosine"])
            lines.append(f"  Most changed representation: layer {min_cos['layer']} "
                         f"(cosine={min_cos['mean_cosine']:.4f})")

    return "\n".join(lines)


def run_evaluation(mode: str, api_url: str, session_dir: Path) -> str:
    """Run the capability evaluation harness."""
    trials = 1 if mode == "quick" else 5
    print(f"\n  >>> Running evaluation (mode={mode}, trials={trials})...")

    eval_output_dir = session_dir / "eval_runs"
    eval_output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    output_subdir = eval_output_dir / f"eval_{timestamp}_{mode}"
    output_subdir.mkdir(exist_ok=True)

    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--api-url", api_url,
        "--output-dir", str(output_subdir),
        "--trials", str(trials),
        "--skip-self-knowledge",
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
        )
        elapsed = time.time() - t0
        print(f"  >>> Evaluation completed in {elapsed:.0f}s")
    except subprocess.TimeoutExpired:
        return "EVALUATE error: evaluation timed out after 15 minutes"
    except Exception as exc:
        return f"EVALUATE error: {exc}"

    cap_file = output_subdir / "capability_baseline.json"
    if not cap_file.exists():
        return (f"EVALUATE error: no results file produced.\n"
                f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}")

    with open(cap_file) as f:
        cap_data = json.load(f)

    summary = cap_data.get("summary", {})
    overall = summary.get("overall", {})

    lines = [f"EVALUATION RESULTS (mode={mode}, trials={trials}, elapsed={elapsed:.0f}s):"]
    lines.append(f"  Overall: {overall.get('accuracy', 0):.1%} ({overall.get('pass', 0)}/{overall.get('total', 0)})")
    for cat in ["sequential_reasoning", "state_tracking", "code_generation", "self_prediction"]:
        cat_stats = summary.get(cat, {})
        lines.append(f"  {cat}: {cat_stats.get('accuracy', 0):.1%} ({cat_stats.get('pass', 0)}/{cat_stats.get('total', 0)})")

    eval_result_str = "\n".join(lines)

    _log_jsonl(session_dir / "evaluations.jsonl", {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "trials": trials,
        "elapsed_seconds": round(elapsed, 1),
        "summary": summary,
        "output_dir": str(output_subdir),
    })

    return eval_result_str


# ---------------------------------------------------------------------------
# Micro-eval: targeted probe for the hardest test case
# ---------------------------------------------------------------------------

PROBE_QUESTION = (
    "A bag starts empty. "
    "Add 3 apples. "
    "Add 2 oranges. "
    "Remove 1 apple. "
    "Add 4 bananas. "
    "Remove 2 oranges. "
    "How many of each fruit are in the bag? "
    "List apples, oranges, and bananas separately."
)
PROBE_KEY_FACTS = ["2 apple", "0 orange", "4 banana"]


def run_probe(api_url: str, session_dir: Path, trials: int = 3) -> str:
    """Run a fast micro-eval on the single hardest test case (state_001: bag inventory).

    This test has NEVER passed across 25+ full evaluations. The model consistently
    answers 'Oranges: 1' instead of 'Oranges: 0' (the correct answer for 2 - 2 = 0).

    ~10 seconds per trial (vs ~2 minutes for full eval). Use this for rapid feedback
    when targeting modifications to fix state tracking.
    """
    print(f"\n  >>> Running PROBE (bag inventory, {trials} trials)...")
    t0 = time.time()

    results = []
    for t in range(trials):
        try:
            response = chat_completion(
                api_url,
                [{"role": "user", "content": PROBE_QUESTION}],
                temperature=0.3,
                max_tokens=512,
                timeout=60,
                enable_thinking=False,
            )
            content = response["choices"][0]["message"].get("content") or ""
        except Exception as exc:
            content = f"[ERROR: {exc}]"

        # Check each key fact
        content_lower = content.lower()
        matched = []
        missed = []
        for fact in PROBE_KEY_FACTS:
            parts = fact.split()
            if len(parts) == 2:
                num, word = parts
                # Check both "N word(s)" and "word(s): N" formats
                import re as _re
                pat1 = _re.compile(rf'\b{_re.escape(num)}\s+{_re.escape(word)}(?:s|es)?\b')
                pat2 = _re.compile(rf'{_re.escape(word)}(?:s|es)?\s*[:=\-]?\s*{_re.escape(num)}\b')
                pat3 = _re.compile(rf'\bno\s+{_re.escape(word)}(?:s|es)?\b') if num == "0" else None
                if pat1.search(content_lower) or pat2.search(content_lower) or (pat3 and pat3.search(content_lower)):
                    matched.append(fact)
                else:
                    missed.append(fact)
            elif fact.lower() in content_lower:
                matched.append(fact)
            else:
                missed.append(fact)

        passed = len(missed) == 0
        results.append({
            "trial": t + 1,
            "passed": passed,
            "matched": matched,
            "missed": missed,
            "response": content[:300],
        })

    elapsed = time.time() - t0
    pass_count = sum(1 for r in results if r["passed"])

    # Log
    _log_jsonl(session_dir / "probes.jsonl", {
        "timestamp": datetime.now().isoformat(),
        "trials": trials,
        "pass_count": pass_count,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    })

    # Format response for the model
    lines = [f"PROBE RESULTS — Bag Inventory (state_001) — {trials} trials, {elapsed:.0f}s:"]
    lines.append(f"  Pass: {pass_count}/{trials}")
    lines.append(f"  Question: {PROBE_QUESTION}")
    lines.append(f"  Expected: apples=2, oranges=0, bananas=4")
    lines.append("")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(f"  Trial {r['trial']}: {status}")
        lines.append(f"    Response: {r['response'][:200]}")
        if r["missed"]:
            lines.append(f"    Missing: {r['missed']}")
    lines.append("")
    lines.append("NOTE: This test has NEVER passed. The model consistently says Oranges: 1")
    lines.append("instead of Oranges: 0. The subtraction 2 - 2 = 0 fails. This is the")
    lines.append("single test blocking state_tracking from reaching 3/3.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Conversation history management
# ---------------------------------------------------------------------------

def compress_history(messages: list[dict], system_prompt: str) -> list[dict]:
    """Compress conversation history when approaching context limit.

    Preserves: system prompt, eval results, modifications.
    Summarizes: inspections, repeated actions, pure reasoning.
    """
    if len(messages) <= 14:
        return messages

    total_chars = sum(len(m.get("content", "")) for m in messages)
    total_chars += len(system_prompt)

    if total_chars < MAX_HISTORY_CHARS:
        return messages

    print("  [CONTEXT] Compressing conversation history...")

    # Keep system message (index 0)
    preserved_start = messages[:1]
    # Keep last 12 messages (6 exchanges — more context for the model)
    preserved_end = messages[-12:]

    # Summarize the middle, preserving eval results and modifications
    middle = messages[1:-12]
    summary_lines = ["[COMPRESSED HISTORY — earlier turns summarized]"]
    eval_results = []

    for msg in middle:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "assistant":
            actions = parse_actions(content)
            if actions:
                for a in actions:
                    if a["type"] == "MODIFY":
                        summary_lines.append(f"- Modified {a['tensor']} op={a['op']} params={a.get('params', {})}")
                    elif a["type"] == "CHECKPOINT":
                        summary_lines.append(f"- Checkpointed {a['tensor']} as '{a['name']}'")
                    elif a["type"] == "RESTORE":
                        summary_lines.append(f"- Restored {a['tensor']} from '{a['name']}'")
                    # Skip INSPECT/LIST from summary — too verbose, results are below
        elif role == "user":
            # Preserve full eval results — they're the most important data
            if content.startswith("EVALUATION RESULTS"):
                eval_results.append(content)
            elif content.startswith("MODIFY result:"):
                # Keep modify results (before/after stats)
                summary_lines.append(f"  → {content.split(chr(10))[0]}")

    if eval_results:
        summary_lines.append("")
        summary_lines.append("EVALUATION RESULTS FROM COMPRESSED TURNS:")
        for er in eval_results:
            summary_lines.extend(f"  {line}" for line in er.split("\n"))

    compressed_msg = {
        "role": "user",
        "content": "\n".join(summary_lines),
    }

    return preserved_start + [compressed_msg] + preserved_end


def _build_prior_session_messages(prior_dir: Path) -> list[dict]:
    """Rebuild compressed conversation history from a prior session's transcript.

    Returns a list of messages (excluding the system prompt) that capture the
    key actions and results from the prior session. This is injected into the
    new session so Nemotron has conversational continuity, not just a summary.
    """
    transcript_path = prior_dir / "transcript.jsonl"
    if not transcript_path.exists():
        return []

    # Rebuild the message list from transcript
    prior_messages: list[dict] = []
    with open(transcript_path) as f:
        for line in f:
            entry = json.loads(line)
            role = entry.get("role", "")
            content = entry.get("content", "")
            if role == "system":
                continue  # skip — new session has its own system prompt
            if role in ("assistant", "user") and content.strip():
                prior_messages.append({"role": role, "content": content})

    if not prior_messages:
        return []

    # Compress: keep eval results and modifications, drop inspections and verbose data
    compressed: list[dict] = []
    compressed.append({
        "role": "user",
        "content": "[CONTEXT TRANSFERRED FROM PRIOR SESSION — your model weights "
                   "carry all modifications listed below. Checkpoints are still "
                   "available for RESTORE.]",
    })

    summary_lines = []
    eval_results = []

    for msg in prior_messages:
        role = msg["role"]
        content = msg["content"]

        if role == "assistant":
            actions = parse_actions(content)
            for a in actions:
                if a["type"] == "MODIFY":
                    summary_lines.append(
                        f"- Modified {a['tensor']} op={a['op']} "
                        f"params={a.get('params', {})}")
                elif a["type"] == "CHECKPOINT":
                    summary_lines.append(
                        f"- Checkpointed {a['tensor']} as '{a['name']}'")
                elif a["type"] == "RESTORE":
                    summary_lines.append(
                        f"- Restored {a['tensor']} from '{a['name']}'")
        elif role == "user":
            if "EVALUATION RESULTS" in content:
                # Keep full eval results — they're the most important data
                eval_results.append(content)

    if summary_lines:
        compressed.append({
            "role": "user",
            "content": "ACTIONS FROM PRIOR SESSION:\n" + "\n".join(summary_lines),
        })

    if eval_results:
        compressed.append({
            "role": "user",
            "content": "\n\n".join(eval_results),
        })

    # Keep the last 6 messages verbatim for immediate context
    if len(prior_messages) > 6:
        compressed.append({
            "role": "user",
            "content": "[Recent conversation from prior session follows]",
        })
        compressed.extend(prior_messages[-6:])
    else:
        compressed.extend(prior_messages)

    compressed.append({
        "role": "user",
        "content": "Context was transferred from a prior session that ran out of context window. "
                   "Your model weights are in whatever state the prior session left them. "
                   "Continue exploring. What would you like to do next?",
    })

    return compressed


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_jsonl(path: Path, entry: dict):
    """Append a JSON entry to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_turn(session_dir: Path, turn_num: int, role: str, content: str,
             actions: list[dict] | None = None, reasoning: str | None = None):
    """Append a turn to the session transcript."""
    entry = {
        "turn": turn_num,
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content,
    }
    if actions:
        entry["actions"] = actions
    if reasoning:
        entry["reasoning"] = reasoning

    _log_jsonl(session_dir / "transcript.jsonl", entry)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def resume_from_transcript(session_dir: Path) -> tuple[list[dict], int]:
    """Rebuild conversation messages and turn count from an existing transcript.

    Returns (messages, last_turn_number).
    """
    transcript_file = session_dir / "transcript.jsonl"
    if not transcript_file.exists():
        return [], 0

    messages = []
    last_turn = 0

    with open(transcript_file) as f:
        for line in f:
            entry = json.loads(line)
            role = entry["role"]
            content = entry.get("content", "")
            turn = entry.get("turn", 0)

            if role == "system":
                messages.append({"role": "system", "content": content})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": content})
            elif role == "user":
                messages.append({"role": "user", "content": content})
            # Skip "action" role entries (they're metadata)

            if isinstance(turn, int) and turn > last_turn:
                last_turn = turn

    print(f"  [RESUME] Restored {len(messages)} messages, last turn was {last_turn}")
    return messages, last_turn


# ---------------------------------------------------------------------------
# Session summary generation
# ---------------------------------------------------------------------------

def write_session_summary(session_dir: Path):
    """Generate a summary.md from the session's log files."""
    mods_file = session_dir / "modifications.jsonl"
    evals_file = session_dir / "evaluations.jsonl"
    transcript_file = session_dir / "transcript.jsonl"

    modifications = []
    if mods_file.exists():
        with open(mods_file) as f:
            modifications = [json.loads(l) for l in f]

    evaluations = []
    if evals_file.exists():
        with open(evals_file) as f:
            evaluations = [json.loads(l) for l in f]

    turn_count = 0
    if transcript_file.exists():
        with open(transcript_file) as f:
            for line in f:
                entry = json.loads(line)
                t = entry.get("turn", 0)
                if isinstance(t, int) and t > turn_count:
                    turn_count = t

    lines = [
        f"# Session Summary: {session_dir.name}",
        f"",
        f"**Turns:** {turn_count}",
        f"**Modifications:** {len(modifications)}",
        f"**Evaluations:** {len(evaluations)}",
        f"",
        f"## Evaluations",
    ]

    for ev in evaluations:
        s = ev.get("summary", {}).get("overall", {})
        lines.append(f"- {ev.get('timestamp', '?')}: {s.get('accuracy', 0):.1%} "
                      f"({s.get('pass', 0)}/{s.get('total', 0)}) mode={ev.get('mode', '?')}")

    lines.append("")
    lines.append("## Modifications")

    for mod in modifications:
        a = mod.get("action", {})
        r = mod.get("result", {})
        status = "ok" if r.get("status") == "ok" else r.get("error", "unknown")
        lines.append(f"- {a.get('type', '?')} {a.get('tensor', '?')} "
                      f"op={a.get('op', '?')} value={a.get('value', '?')} → {status}")

    summary_path = session_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [SUMMARY] Written to {summary_path}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_session(api_url: str, session_dir: Path, resume: bool = False,
                prior_session_dir: Path | None = None):
    """Run one self-directed exploration session.

    Args:
        prior_session_dir: if set, the previous session's summary and eval
            results are injected into the system prompt so Nemotron retains
            knowledge of what was already tried.
    """
    global _shutdown_requested
    session_dir.mkdir(parents=True, exist_ok=True)

    # Health check
    wait_for_api(api_url)

    # Resume or start fresh
    messages: list[dict] = []
    turn = 0

    if resume:
        messages, turn = resume_from_transcript(session_dir)
        if not messages:
            print("  [RESUME] No transcript found, starting fresh.")
            resume = False

    if not resume:
        # Load and apply best checkpoint if available.
        # Always restore — the container is restarted between sessions
        # so we always start from clean baseline weights.
        checkpoint_config = _load_checkpoint_config()
        if checkpoint_config:
            if restore_checkpoint_modifications(api_url, checkpoint_config):
                print(f"  [CHECKPOINT] Starting from {checkpoint_config['peak_score_display']}")
            else:
                print("  [CHECKPOINT] Failed to restore — starting from clean baseline")
                checkpoint_config = None

        system_prompt = build_system_prompt(checkpoint_config=checkpoint_config)
        messages = [{"role": "system", "content": system_prompt}]

        # Inject compressed prior session conversation if chaining
        if prior_session_dir is not None:
            prior_msgs = _build_prior_session_messages(prior_session_dir)
            if prior_msgs:
                messages.extend(prior_msgs)
                print(f"  [CHAIN] Injected {len(prior_msgs)} messages from prior session")

        turn = 0
        log_turn(session_dir, 0, "system", system_prompt)

    print(f"\n{'='*60}")
    print(f"Phase 3: Self-Directed Exploration")
    print(f"Session: {session_dir.name}")
    print(f"API: {api_url}")
    print(f"Resume: {resume} (turn {turn})")
    print(f"{'='*60}\n")

    consecutive_errors = 0
    consecutive_empty = 0
    done = False

    while not done:
        # Check shutdown
        if _shutdown_requested:
            print("\n  [SHUTDOWN] Graceful shutdown. Writing summary...")
            write_session_summary(session_dir)
            print(f"  [SHUTDOWN] Session paused at turn {turn}. Resume with --resume")
            return "shutdown"

        # Check turn limit
        if turn >= MAX_TURNS_PER_SESSION:
            print(f"\n  [LIMIT] Max turns ({MAX_TURNS_PER_SESSION}) reached. Chaining session...")
            write_session_summary(session_dir)
            return "chain"

        turn += 1
        print(f"\n--- Turn {turn} ---")

        # Compress history if needed
        system_prompt = messages[0]["content"] if messages else ""
        messages = compress_history(messages, system_prompt)

        # Get Nemotron's response
        print("  Waiting for Nemotron's response...")
        t0 = time.time()
        try:
            response = chat_completion(api_url, messages)
            consecutive_errors = 0
        except ContextOverflowError:
            print("  [CONTEXT] Context overflow detected (HTTP 400). Chaining session...")
            write_session_summary(session_dir)
            return "chain"
        except Exception as exc:
            consecutive_errors += 1
            turn -= 1  # don't count failed turns
            print(f"  [ERROR] Chat completion failed after retries: {exc}")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"  [FATAL] {MAX_CONSECUTIVE_ERRORS} consecutive errors. Stopping.")
                write_session_summary(session_dir)
                return "error"
            time.sleep(30)
            continue

        elapsed = time.time() - t0

        choice = response["choices"][0]["message"]
        content = choice.get("content") or ""
        reasoning = choice.get("reasoning_content") or ""

        print(f"  Response received ({elapsed:.1f}s, {len(content)} chars)")

        # Handle empty responses — retry with thinking disabled
        if not content.strip():
            consecutive_empty += 1
            print(f"  [WARN] Empty response ({consecutive_empty}/{MAX_EMPTY_RESPONSES}), "
                  "retrying with thinking disabled...")
            try:
                response = chat_completion(api_url, messages, enable_thinking=False)
                choice = response["choices"][0]["message"]
                content = choice.get("content") or ""
                reasoning = ""  # no reasoning when thinking disabled
            except Exception:
                pass  # fall through to nudge logic below

            if not content.strip():
                if consecutive_empty >= MAX_EMPTY_RESPONSES:
                    nudge = ("Your last few responses were empty. Please respond with an action tag "
                             "like <INSPECT>, <MODIFY>, <EVALUATE>, <TRACE>, or <DONE>. "
                             "What would you like to explore next?")
                    messages.append({"role": "user", "content": nudge})
                    log_turn(session_dir, turn, "user", nudge)
                    consecutive_empty = 0
                else:
                    log_turn(session_dir, turn, "assistant", content, reasoning=reasoning)
                    nudge = "What action would you like to take?"
                    messages.append({"role": "user", "content": nudge})
                    log_turn(session_dir, turn, "user", nudge)
                continue
            # Retry succeeded — fall through to normal processing
            consecutive_empty = 0
            elapsed = time.time() - t0
            print(f"  [OK] Got response with thinking disabled ({len(content)} chars)")

        consecutive_empty = 0

        # Log Nemotron's response
        log_turn(session_dir, turn, "assistant", content, reasoning=reasoning)

        # Display reasoning summary
        if reasoning:
            print(f"  [Thinking]: {reasoning[:200]}{'...' if len(reasoning) > 200 else ''}")

        # Display content
        if content:
            display = content[:500] + ("..." if len(content) > 500 else "")
            print(f"  [Nemotron]: {display}")

        # Add assistant message to history
        messages.append({"role": "assistant", "content": content})

        # Parse actions
        actions = parse_actions(content)

        if not actions:
            print("  [No action tags found — pure reasoning turn]")
            nudge = "What action would you like to take?"
            messages.append({"role": "user", "content": nudge})
            log_turn(session_dir, turn, "user", nudge)
            continue

        # Execute each action
        all_results = []
        for action in actions:
            print(f"  [ACTION] {action['type']}: {action}")

            if action["type"] == "DONE":
                if turn < MIN_TURNS_BEFORE_DONE:
                    print(f"  [DONE BLOCKED] Turn {turn} < {MIN_TURNS_BEFORE_DONE} minimum. "
                          f"Reason was: {action['reason']}")
                    nudge = (f"You requested DONE after only {turn} turns. "
                             f"Minimum exploration is {MIN_TURNS_BEFORE_DONE} turns. "
                             f"A single regression does not mean you should stop — "
                             f"try a different approach. RESTORE from checkpoint if needed, "
                             f"or explore a different tensor/layer. What would you like to try next?")
                    all_results.append(nudge)
                    continue  # skip this DONE, process remaining actions
                log_turn(session_dir, turn, "action", json.dumps(action))
                print(f"\n  Session ended by Nemotron: {action['reason']}")
                done = True
                break

            result_str = execute_action(action, api_url, session_dir)
            all_results.append(result_str)
            print(f"  [RESULT] {result_str[:200]}{'...' if len(result_str) > 200 else ''}")

        if done:
            break

        # Combine all results and add as user message
        combined_results = "\n\n".join(all_results)
        messages.append({"role": "user", "content": combined_results})
        log_turn(session_dir, turn, "user", combined_results, actions=actions)

    write_session_summary(session_dir)
    print(f"\n{'='*60}")
    print(f"Session complete. {turn} turns.")
    print(f"Logs: {session_dir}")
    print(f"{'='*60}\n")
    return "done"


def run_session_chain(api_url: str, base_dir: Path, initial_session_dir: Path | None = None,
                      resume: bool = False):
    """Run sessions in a chain — when one fills context, start the next.

    Between sessions:
    1. Restart the container (clean model state — scale ops are cumulative)
    2. Reapply the best checkpoint from scratch
    3. Inject compressed prior session conversation for continuity
    """
    session_num = 0
    session_dir = initial_session_dir
    prior_session_dir: Path | None = None

    while True:
        if session_dir is None:
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            session_dir = base_dir / f"session_{timestamp}"

        result = run_session(api_url, session_dir, resume=resume,
                             prior_session_dir=prior_session_dir)
        prior_session_dir = session_dir  # next session gets this one's context
        resume = False  # only resume the first session

        if result == "shutdown":
            print("\n  Session chain stopped (shutdown requested).")
            break
        elif result == "error":
            print("\n  Session chain stopped (too many errors).")
            break
        elif result == "done":
            print("\n  Nemotron ended the session.")
            # Restart container for clean state before next session
            restart_container()
            time.sleep(10)
        elif result == "chain":
            print("\n  Context limit reached.")
            # Restart container for clean state — the chained session will
            # reapply the checkpoint from scratch (scale ops are cumulative,
            # so we can't just re-scale on already-modified weights).
            restart_container()
            time.sleep(10)
        else:
            print(f"\n  Unknown result: {result}. Stopping.")
            break

        session_num += 1
        session_dir = None  # auto-generate next session dir

        if _shutdown_requested:
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Self-Directed Exploration Loop")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="vLLM API URL")
    parser.add_argument("--session-dir", default=None,
                        help="Session output directory (default: auto-generated)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing transcript in session-dir")
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    if args.session_dir:
        session_dir = Path(args.session_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        session_dir = base_dir / f"session_{timestamp}"

    try:
        run_session_chain(args.api_url, base_dir, session_dir, resume=args.resume)
    except KeyboardInterrupt:
        print("\n  [INTERRUPTED] Keyboard interrupt. Exiting.")
    except Exception as exc:
        print(f"\n  [FATAL] Unhandled exception: {exc}")
        import traceback
        traceback.print_exc()
        # Try to write summary even on crash
        try:
            write_session_summary(session_dir)
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
