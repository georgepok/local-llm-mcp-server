# Best Checkpoint: Session 2 Peak (91.7%)

This directory records the modification stack that produced the highest score observed
across all self-directed sessions: **91.7% (11/12)** on the quick eval harness, achieved
during `session_20260311T010244` at 01:50 UTC.

## What is stored here

- `session2_peak.json` — the canonical record of all 10 modifications (with tensor stats,
  timestamps, and per-eval scores) that together produced the 91.7% peak.

There is no saved weight file here. The modifications are lightweight scalar operations
(scale, scale_slice) applied in-place to the live model held in the vLLM serving process.
To restore the peak configuration you must re-apply them from a clean baseline.

## Prerequisites

The self-directed loop loads the model into vLLM at startup. You need the serving process
running before you can apply modifications:

```
# On DGX Spark
ssh pokazge@spark-129a.local
# start vLLM container / service as normal (see deploy.sh in fluid-geometry)
```

Then from this repo, ensure the neuroplastic API server is reachable at the configured
address (default: http://spark-129a.local:30000).

## How to restore the peak configuration

The 10 modifications must be applied **in order** to an unmodified (baseline) model.
Each modification is cumulative — they build on one another. Applying them out of order
or to an already-modified model will produce a different (and likely worse) result.

### Option A: Manual API calls

Send each modification as a POST request to the neuroplastic modify endpoint. Using the
exact values from `session2_peak.json`:

```bash
# Example for modification 1
curl -X POST http://spark-129a.local:30000/neuroplastic/modify \
  -H 'Content-Type: application/json' \
  -d '{"tensor": "model.layers.50.mixer.A", "op": "scale", "params": {"value": 0.6065}}'
```

Repeat for all 10 entries in `modifications` array order (order 1 through 10).

### Option B: Python replay script

```python
import json, urllib.request

MODS_FILE = "session2_peak.json"
API_BASE  = "http://spark-129a.local:30000"

with open(MODS_FILE) as f:
    data = json.load(f)

for mod in data["modifications"]:
    payload = json.dumps({
        "tensor": mod["tensor"],
        "op":     mod["op"],
        "params": mod["params"],
    }).encode()
    req = urllib.request.Request(
        f"{API_BASE}/neuroplastic/modify",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    print(f"[{mod['order']:02d}] {mod['tensor']} {mod['op']} -> {result['status']}")
```

### Option C: Resume via self_directed_loop.py with a seeded transcript

Not recommended — the loop is designed for exploration, not replay.

## The 10 modifications at a glance

| Order | Tensor                              | Op          | Value(s)                        | Eval after |
|------:|-------------------------------------|-------------|----------------------------------|------------|
|     1 | model.layers.50.mixer.A             | scale       | 0.6065                          | 66.7%      |
|     2 | model.layers.48.mixer.A             | scale       | 0.6065                          | 66.7%      |
|     3 | model.layers.46.mixer.D             | scale       | 1.5                             | 75.0%      |
|     4 | model.layers.42.mixer.o_proj.weight | scale       | 1.2                             | 83.3%      |
|     5 | model.layers.33.mixer.o_proj.weight | scale       | 1.2                             | (batched)  |
|     6 | model.layers.46.mixer.A             | scale       | 0.6065                          | 83.3%      |
|     7 | model.layers.46.mixer.A             | scale       | 0.6065                          | (batched)  |
|     8 | model.layers.46.mixer.A             | scale_slice | start=0, end=32, value=0.4      | 83.3%      |
|     9 | model.layers.44.mixer.A             | scale       | 0.6065                          | 83.3%      |
|    10 | model.layers.50.mixer.A             | scale       | 0.5                             | **91.7%**  |

"Batched" means no eval was run immediately after that mod; the score shown is from the
next eval which included it. Full timestamps and before/after norms are in the JSON.

## Net effect on each modified tensor

- **layers.50.mixer.A**: 0.6065 x 0.5 = **0.3033x** baseline (mods 1 and 10)
- **layers.48.mixer.A**: **0.6065x** baseline (mod 2)
- **layers.46.mixer.D**: **1.5x** baseline (mod 3, amplified)
- **layers.42.mixer.o_proj.weight**: **1.2x** baseline (mod 4)
- **layers.33.mixer.o_proj.weight**: **1.2x** baseline (mod 5)
- **layers.46.mixer.A**: 0.6065 x 0.6065 = 0.368x for heads[32:64]; 0.368 x 0.4 = **0.147x** for heads[0:32] (mods 6, 7, 8)
- **layers.44.mixer.A**: **0.6065x** baseline (mod 9)

## What the peak revealed

The winning gain came from `self_prediction` jumping from 0/3 to 3/3. The category that
never improved was `state_tracking` (stuck at 2/3 throughout).

The peak is fragile: applying mod 11 (`layers.46.mixer.D scale_slice start=0 end=32
value=1.2`) immediately dropped the score back to 83.3%, and mod 12 (a third scale of
`layers.46.mixer.A`) dropped it further to 66.7% (back to baseline).

The critical insight is that layer 50's mixer.A (the final attention layer in the upper
third of the network) is highly sensitive: reducing it to ~30% of baseline was the single
modification that unlocked the peak. The earlier mods (particularly mod 4, the o_proj
scale at layer 42) appear to be a prerequisite — without the 83.3% platform they
established, the layer-50 reduction alone may not have been sufficient.

## Verification

After applying all 10 modifications, run the quick eval to confirm:

```bash
python3 /path/to/neuroplastic/phase1_artifacts/eval_harness/run_eval.py --mode quick
```

Expected result: 11/12 correct (91.7%), with self_prediction at 3/3.
