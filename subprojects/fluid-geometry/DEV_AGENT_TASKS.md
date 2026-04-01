# Geometric Engine v2 — Validation Tasks

## Context
- **Date**: 2026-02-08
- **Target**: spark-129a.local:30000
- **Container**: vllm-nemotron-serve
- **Model**: NVIDIA-Nemotron-3-Nano-30B-A3B-FP8
- **Engine File**: `/workspace/fluid_geometry.py` (mounted from `/home/pokazge/models/fluid_geometry.py`)
- **State File**: `/workspace/engine_state/geometric_engine_state.json`
- **Reference Docs**: `GEOMETRIC_ENGINE_SPEC.md`, `IMPLEMENTATION.md`, `TEST_RESULTS.md`

## Priority
These tasks are ordered by dependency. Complete them in sequence.

---

## Task 1: Verify Think Token IDs

**Why**: Engine resolved `<think>=12` and `</think>=13`. These IDs seem low for model-specific reasoning tokens. Need confirmation they aren't mapping to unrelated control tokens.

**Action**:
```bash
docker exec vllm-nemotron-serve python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/workspace/model', trust_remote_code=True)
print('ID 12 decodes to:', repr(tok.decode([12])))
print('ID 13 decodes to:', repr(tok.decode([13])))
print()
# Also verify reverse lookup
for name in ['<think>', '</think>']:
    ids = tok.encode(name, add_special_tokens=False)
    print(f'{name} encodes to: {ids}')
"
```

**Pass criteria**: ID 12 decodes to `<think>`, ID 13 decodes to `</think>`. Reverse lookup matches.

**If FAIL**: Find correct token IDs and update the token resolution logic in `FluidGeometryLogitsProcessor.__init__()`.

---

## Task 2: Convergence Test — Push 10K+ Tokens

**Why**: Engine is at ~500 tokens (C≈5%). Need to reach operational confidence (C>0.63 at 10K tokens) and observe whether κ_ref stabilizes, baseline_perplexity is reasonable, and confidence_override remains near 1.0.

**Action**: Send 100 diverse prompts to the model. Use varied types to exercise different entropy/curvature regimes.

```bash
#!/bin/bash
# convergence_test.sh — run from any machine that can reach spark-129a:30000
API="http://spark-129a.local:30000/v1/chat/completions"
MODEL="NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

PROMPTS=(
  "What is 17 times 23?"
  "Explain photosynthesis in three sentences."
  "Write a haiku about rain."
  "What are the pros and cons of nuclear energy?"
  "Write a Python function that reverses a linked list."
  "Translate 'good morning' into Japanese, Korean, and Mandarin."
  "If a train leaves Chicago at 9am going 60mph and another leaves NYC at 10am going 80mph, when do they meet?"
  "Summarize the plot of Hamlet in 50 words."
  "What is the difference between TCP and UDP?"
  "Write a short story about a robot that learns to paint."
  "Explain quantum entanglement to a 10 year old."
  "List 5 common logical fallacies with examples."
  "Write SQL to find the top 10 customers by total spend."
  "What causes inflation?"
  "Compare and contrast REST and GraphQL."
  "Write a limerick about a programmer."
  "Explain the halting problem."
  "What is the chain rule in calculus?"
  "Design a database schema for a library system."
  "Why do we dream?"
  "Write a bash one-liner that finds all files larger than 100MB."
  "Explain how a transformer neural network works."
  "What is the trolley problem and what are the main ethical positions?"
  "Write a recursive Fibonacci function in Rust."
  "Describe the water cycle."
  "What is gerrymandering?"
  "Explain P vs NP."
  "Write a poem about silence."
  "How does a refrigerator work?"
  "What are design patterns in software engineering? Name three."
  "Explain the difference between correlation and causation."
  "Write a function in JavaScript that debounces another function."
  "What is the Fermi paradox?"
  "Describe three sorting algorithms and their time complexity."
  "What is cognitive dissonance?"
  "Write a regex that matches email addresses."
  "Explain supply and demand."
  "What is the significance of Euler's identity?"
  "Write a Python class for a binary search tree."
  "What causes the seasons?"
  "Explain the concept of entropy in thermodynamics."
  "Write a short dialogue between a cat and a dog."
  "What is the Turing test?"
  "Describe how HTTPS works."
  "What is the overview effect?"
  "Write a function that checks if a string is a palindrome."
  "Explain the prisoner's dilemma."
  "What is dark matter?"
  "Write a Dockerfile for a Python Flask application."
  "Explain the difference between machine learning and deep learning."
  "What is the Sapir-Whorf hypothesis?"
  "Write pseudocode for Dijkstra's algorithm."
  "What causes tides?"
  "Explain what a monad is in functional programming."
  "What is the Dunning-Kruger effect?"
  "Write a Python generator that yields prime numbers."
  "Describe how GPS works."
  "What is game theory?"
  "Write a CSS animation for a bouncing ball."
  "Explain the double-slit experiment."
  "What is the tragedy of the commons?"
  "Write a simple neural network in NumPy."
  "What is CRISPR and how does it work?"
  "Explain map-reduce."
  "What is the Mandelbrot set?"
  "Write a Python decorator that caches function results."
  "What is the greenhouse effect?"
  "Explain eventual consistency."
  "What is the Ship of Theseus?"
  "Write a function that flattens a nested array."
  "How do vaccines work?"
  "Explain the CAP theorem."
  "What is synesthesia?"
  "Write an implementation of a simple hash map."
  "What is the Doppler effect?"
  "Explain backpropagation."
  "What is the Baader-Meinhof phenomenon?"
  "Write a Python script that reads a CSV and computes column averages."
  "What is the heat death of the universe?"
  "Explain how public key cryptography works."
  "What is the bystander effect?"
  "Write a state machine in Python for a traffic light."
  "What is plate tectonics?"
  "Explain the difference between concurrency and parallelism."
  "What is the Streisand effect?"
  "Write a function that computes the edit distance between two strings."
  "How does the immune system work?"
  "Explain what a bloom filter is."
  "What is the mere exposure effect?"
  "Write a simple HTTP server in Python."
  "What is the Coriolis effect?"
  "Explain the observer pattern."
  "What is the Zeigarnik effect?"
  "Write a Python function that generates permutations."
  "How do black holes form?"
  "Explain eventual consistency vs strong consistency."
  "What is the anchoring bias?"
  "Write a binary search implementation."
  "How does natural selection work?"
  "Count to 20."
)

echo "Starting convergence test: ${#PROMPTS[@]} prompts"
echo "================================"

for i in "${!PROMPTS[@]}"; do
  PROMPT="${PROMPTS[$i]}"
  echo "[$((i+1))/${#PROMPTS[@]}] $PROMPT"
  
  RESPONSE=$(curl -s "$API" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
      \"max_tokens\": 500
    }")
  
  TOKENS=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('total_tokens','?'))" 2>/dev/null)
  FINISH=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0].get('finish_reason','?'))" 2>/dev/null)
  
  echo "  tokens=$TOKENS finish=$FINISH"
  sleep 0.5
done

echo ""
echo "================================"
echo "Convergence test complete. Now inspect state:"
echo ""
docker exec vllm-nemotron-serve cat /workspace/engine_state/geometric_engine_state.json 2>/dev/null || echo "State file not yet written"
```

**Pass criteria**:
- All 100 prompts return valid responses (no 500 errors, no empty content)
- State file exists after test completes
- `t_global` > 10000
- `kappa_ref` has moved from initial 1.0 to a stable value
- `baseline_perplexity` > 0
- `confidence_override` > 0.9 (stability monitor has not triggered major pullback)

**Record**: Save the state file contents and the full test log. We need this data.

---

## Task 3: Inspect Converged State

**Why**: After Task 2, we need to evaluate the state the engine converged to.

**Action**:
```bash
# Pull state
docker exec vllm-nemotron-serve cat /workspace/engine_state/geometric_engine_state.json | python3 -m json.tool

# Check confidence level
docker exec vllm-nemotron-serve python3 -c "
import json, math
with open('/workspace/engine_state/geometric_engine_state.json') as f:
    s = json.load(f)
C = 1 - math.exp(-s['t_global'] / 10000)
C_eff = C * s['confidence_override']
print(f't_global:            {s[\"t_global\"]}')
print(f'kappa_ref:           {s[\"kappa_ref\"]:.4f}')
print(f'kappa_running_mean:  {s[\"kappa_running_mean\"]:.4f}')
print(f'kappa_running_var:   {s[\"kappa_running_var\"]:.4f}')
print(f'baseline_perplexity: {s[\"baseline_perplexity\"]:.2f}')
print(f'baseline_count:      {s[\"baseline_count\"]}')
print(f'confidence_override: {s[\"confidence_override\"]:.4f}')
print(f'confidence (C):      {C:.4f}')
print(f'effective (C_eff):   {C_eff:.4f}')
print()
print(f'At current C_eff={C_eff:.3f}:')
print(f'  If kappa/kappa_ref = +1.0 → T = {1.0 * (1 + C_eff * 1.0):.3f}')
print(f'  If kappa/kappa_ref = -1.0 → T = {1.0 * (1 + C_eff * -1.0):.3f}')
print(f'  If kappa/kappa_ref = +3.0 → T = {1.0 * (1 + C_eff * 3.0):.3f}')
print(f'  Max think bias:    {C_eff * 15.0:.2f} logits')
"
```

**What to look for**:
- `kappa_ref` should be in range 0.1–2.0 (if it's still near 1.0, EMA rate may be too slow)
- `baseline_perplexity` should be plausible for this model (likely 5–30 range)
- `confidence_override` near 1.0 means stability monitor is happy; below 0.8 means something triggered pullback
- Temperature range at operational confidence should be meaningful but not extreme (T between 0.7–1.5 for typical κ values)

---

## Task 4: Add Diagnostic Logging

**Why**: We cannot evaluate whether the engine is doing anything useful without seeing its per-token decisions. Current implementation has no observable output during generation. We need a lightweight diagnostic trace.

**Action**: Add a logging mode to `fluid_geometry.py` that writes per-token diagnostics to a file when a flag is set.

**Location**: In `GeometricRequestProcessor.__call__()`, after step 6 (apply biases), before step 7 (update calibrator).

**Add this code block** (approximately after line 480 in current file):

```python
# Diagnostic trace (enable via environment variable)
if os.environ.get('FG_TRACE', '0') == '1':
    trace_line = (
        f"t={self.calibrator.state.t_global}"
        f" H={phase.H:.3f}"
        f" dH={phase.delta_H:.3f}"
        f" d2H={phase.delta2_H:.3f}"
        f" k={phase.kappa:.4f}"
        f" T={phase.T_applied:.3f}"
        f" C={self.calibrator.get_confidence():.3f}"
        f" co={self.calibrator.state.confidence_override:.3f}"
    )
    with open('/workspace/engine_state/trace.log', 'a') as tf:
        tf.write(trace_line + '\n')
```

**Then test with tracing enabled**:
```bash
# Stop container
docker stop vllm-nemotron-serve

# Restart with trace flag
# (same docker run command as IMPLEMENTATION.md but add -e FG_TRACE=1)
docker run -d \
  --name vllm-nemotron-serve \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 30000:30000 \
  -e FG_TRACE=1 \
  -v /home/pokazge/models/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8:/workspace/model \
  -v /home/pokazge/models/nano_v3_reasoning_parser.py:/workspace/nano_v3_reasoning_parser.py \
  -v /home/pokazge/models/fluid_geometry.py:/workspace/fluid_geometry.py \
  -v /home/pokazge/models/engine_state:/workspace/engine_state \
  nvcr.io/nvidia/vllm:26.01-py3 \
  python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 30000 \
    --model /workspace/model \
    --served-model-name NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 \
    --trust-remote-code \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --enable-prefix-caching \
    --reasoning-parser-plugin /workspace/nano_v3_reasoning_parser.py \
    --reasoning-parser nano_v3 \
    --logits-processors fluid_geometry:FluidGeometryLogitsProcessor

# After startup, send a few test prompts, then inspect:
docker exec vllm-nemotron-serve head -100 /workspace/engine_state/trace.log
```

**Pass criteria**: Trace file contains one line per generated token showing H, κ, T values. H values should be in range 0–12 (log₂ of vocab size ≈ 18 for 256K vocab, but effective entropy is much lower). κ values should cluster around 0 with occasional excursions. T should be near 1.0 with small deviations proportional to C_eff.

---

## Task 5: Baseline Comparison (A/B Test)

**Why**: Need evidence that the engine at operational confidence either helps, hurts, or is neutral. This is the core validation the spec called for.

**Prerequisite**: Tasks 2 and 4 complete. Engine at C > 0.6.

**Action**: Run the same 20 prompts twice — once with engine active (normal), once with engine effectively disabled.

**Method to disable without rebuilding**: Set `confidence_override` to 0 in the state file:

```bash
# Save current state
docker exec vllm-nemotron-serve cp /workspace/engine_state/geometric_engine_state.json /workspace/engine_state/geometric_engine_state.json.backup

# Create disabled state (confidence_override = 0)
docker exec vllm-nemotron-serve python3 -c "
import json
with open('/workspace/engine_state/geometric_engine_state.json') as f:
    s = json.load(f)
s['confidence_override'] = 0.0
with open('/workspace/engine_state/geometric_engine_state.json', 'w') as f:
    json.dump(s, f)
print('Set confidence_override to 0 (engine disabled)')
"

# Restart container to pick up the state
docker restart vllm-nemotron-serve
sleep 300  # Wait for model load

# Run 20 test prompts, save responses as baseline_responses.json
# (use a subset of Task 2 prompts)

# Restore active state
docker exec vllm-nemotron-serve cp /workspace/engine_state/geometric_engine_state.json.backup /workspace/engine_state/geometric_engine_state.json
docker restart vllm-nemotron-serve
sleep 300

# Run same 20 prompts, save responses as active_responses.json
```

**Compare**:
- Total tokens generated (does engine cause more/fewer tokens?)
- Response completeness (does engine cause more `length` finishes vs `stop`?)
- Subjective quality (read both responses side by side for 5 prompts)
- If trace is enabled: compare H/κ/T distributions between runs

**Pass criteria**: Active engine responses are at least as good as disabled. Any quality difference should be explainable by the temperature/bias mechanics.

---

## Task 6: Thread Safety Check

**Why**: Calibrator uses `threading.Lock()`. vLLM may use asyncio, multiprocessing, or thread pools. Wrong concurrency model means the lock does nothing.

**Action**:
```bash
docker exec vllm-nemotron-serve python3 -c "
import vllm
import inspect
# Check if vLLM uses multiprocessing workers
print('vLLM version:', vllm.__version__)

# Check the server's worker model
import importlib
mod = importlib.import_module('vllm.entrypoints.openai.api_server')
src = inspect.getsource(mod)
if 'multiprocessing' in src:
    print('WARNING: Server uses multiprocessing — threading.Lock is insufficient')
elif 'ThreadPool' in src or 'thread' in src.lower():
    print('Server uses threads — threading.Lock is appropriate')
else:
    print('Server likely uses asyncio — threading.Lock is safe (single-threaded)')
"
```

**Also check**: With `--max-num-seqs 8`, send 4 concurrent requests and verify state file is not corrupted:

```bash
for i in 1 2 3 4; do
  curl -s http://spark-129a.local:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"NVIDIA-Nemotron-3-Nano-30B-A3B-FP8","messages":[{"role":"user","content":"Write a 200 word essay about the number '$i'."}],"max_tokens":400}' &
done
wait
echo "All requests complete"
docker exec vllm-nemotron-serve python3 -c "import json; json.load(open('/workspace/engine_state/geometric_engine_state.json')); print('State file valid JSON')"
```

**Pass criteria**: No JSON parse errors on state file after concurrent requests. No container errors in logs.

---

## Reporting

After completing all tasks, create `VALIDATION_RESULTS.md` in this directory with:

1. Token ID verification result
2. Converged state file contents (full JSON)
3. Computed confidence and temperature ranges
4. Trace log excerpt (first 100 lines)
5. A/B comparison summary
6. Thread safety finding
7. Any anomalies, errors, or unexpected behavior

---

## File Locations Summary

| File | Machine | Path |
|------|---------|------|
| Engine source | spark-129a | `/home/pokazge/models/fluid_geometry.py` |
| Engine (container) | container | `/workspace/fluid_geometry.py` |
| State file | spark-129a | `/home/pokazge/models/engine_state/geometric_engine_state.json` |
| State (container) | container | `/workspace/engine_state/geometric_engine_state.json` |
| Trace log | container | `/workspace/engine_state/trace.log` |
| This file | local | `DEV_AGENT_TASKS.md` |
