"""Stage 1 — autonomous architecture extension.

Qwen80 adds a new retrieval mode `temporal` to
`liquid_arc.graph_rag.decoupled.graph_db.KnowledgeGraphDB`. The loop
runs a harness that scores shape correctness + ordering + benchmark
lift over a recency-only baseline + no-regression on existing modes.

Reward components (total ∈ [0, 1]):
  0.25 shape_ok         — returns list[dict] with required keys
  0.25 ordering_correct — constructed test where expected ranking is known
  0.30 beats_baseline   — top-k recall vs recency-only, averaged over 5 queries
  0.20 no_regression    — existing metric + graph modes still work

If total reward >= target (0.90), the loop terminates with success.

Invocation (on Spark, once Qwen80 is serving):
    python scripts/selfint_stage1_temporal.py \
        --vllm_url http://172.17.0.1:30000/v1 \
        --model Qwen3-Next-80B-A3B-Instruct-FP8 \
        --out_dir /workspace/liquid-arc/shared/outbox/selfint_stage1_temporal
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


_CODE_RX = re.compile(r"```(?:python)?\s*\n(.*?)```", flags=re.DOTALL)


def llm_generate(vllm_url: str, model: str, messages: List[Dict[str, str]],
                 max_tokens: int = 5000, temperature: float = 0.1) -> str:
    r = requests.post(
        f"{vllm_url.rstrip('/')}/chat/completions",
        json={"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature,
               "chat_template_kwargs": {"enable_thinking": False}},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def extract_code(text: str) -> Optional[str]:
    m = _CODE_RX.search(text)
    if m:
        return m.group(1)
    if "```python" in text:
        body = text.split("```python", 1)[1]
        return body.rsplit("\n", 1)[0] if "\n" in body else body
    return None


# ----------------------------------------------------------------------
# Task — contract + reference + instructions for Qwen
# ----------------------------------------------------------------------

TASK_PROMPT = textwrap.dedent('''\
    Write ONE standalone top-level Python function that will be
    monkey-patched as a new method on the existing `KnowledgeGraphDB`
    class:

        def query_temporal(self, query_nodes, k):
            ...

    It must NOT replace any existing method — it's a NEW capability.

    Score each candidate node `n` (n not in query_nodes, n in self.G):

      temporal_score = 1 - (|n.last_seen - mean(query.last_seen)| / max_delta)
          where max_delta = max over all candidates |node.last_seen - q_mean|
          fall back to 1.0 when max_delta == 0

      causal_score = 1.0 / (1.0 + shortest_path_distance)
          computed on `self.G.to_undirected()`
          if `n` is unreachable from every query node, causal_score = 0.0

      combined_score = 0.5 * temporal_score + 0.5 * causal_score

    Return list of dicts sorted by combined_score DESCENDING, top-k:
      {"id", "type", "role",
       "temporal_score", "causal_score", "combined_score",
       "source": "temporal"}

    Read node attributes via `self.G.nodes[nid]["last_seen"]`,
    `self.G.nodes[nid]["type"]`, `self.G.nodes[nid]["role"]`.

    `self.G` is a `networkx.DiGraph`. Use `import networkx as nx`
    inside the file.

    Output ONE ```python``` fenced block. Do NOT write a class. Do NOT
    write test code. Only the single function + its imports.
''')


# ----------------------------------------------------------------------
# Harness scaffolding — data, baseline, reward
# ----------------------------------------------------------------------


HARNESS_TEMPLATE = '''\
import json, os, random, time, sys
sys.path.insert(0, "/workspace/liquid-arc")

# ---- 1) Monkey-patch KnowledgeGraphDB with the candidate function ----
from liquid_arc.graph_rag.decoupled import graph_db as gdb_module
from liquid_arc.graph_rag.decoupled.graph_db import KnowledgeGraphDB

CANDIDATE_CODE = """
{candidate_code}
"""

# Exec candidate into a scratch namespace inheriting the module's imports.
_ns = dict(gdb_module.__dict__)
exec(CANDIDATE_CODE, _ns)

# Patch the function onto the class as a new method.
assert "query_temporal" in _ns, "candidate did not define query_temporal"
KnowledgeGraphDB.query_temporal = _ns["query_temporal"]

# ---- 2) Build test graph ----
# 30 nodes across 3 causal chains, with timestamps spread over a week.
random.seed(42)
db = KnowledgeGraphDB("/tmp/stage1_db.json")
db.clear()
NOW = 1_700_000_000.0
HOUR = 3600.0
frag_nodes, frag_edges = [], []

# Chain A: events separated by 1 hour, starting NOW
for i in range(10):
    frag_nodes.append({{"id": f"A_{{i}}", "type": "event", "role":
                       "root" if i==0 else ("terminal" if i==9 else "intermediate")}})
    if i > 0:
        frag_edges.append({{"src": f"A_{{i-1}}", "dst": f"A_{{i}}", "type": "causes"}})

# Chain B: events separated by 1 day, starting NOW - 3 days
for i in range(10):
    frag_nodes.append({{"id": f"B_{{i}}", "type": "state", "role":
                       "root" if i==0 else ("terminal" if i==9 else "intermediate")}})
    if i > 0:
        frag_edges.append({{"src": f"B_{{i-1}}", "dst": f"B_{{i}}", "type": "precedes"}})

# Chain C: events separated by 6 hours, starting NOW - 1 day
for i in range(10):
    frag_nodes.append({{"id": f"C_{{i}}", "type": "consequence", "role":
                       "root" if i==0 else ("terminal" if i==9 else "intermediate")}})
    if i > 0:
        frag_edges.append({{"src": f"C_{{i-1}}", "dst": f"C_{{i}}", "type": "enables"}})

db.add_fragment({{"nodes": frag_nodes, "edges": frag_edges}}, autosave=False)

# Rewrite last_seen timestamps to our controlled values
for i in range(10):
    db.G.nodes[f"A_{{i}}"]["last_seen"] = NOW - (9 - i) * HOUR              # 0..9h ago
    db.G.nodes[f"B_{{i}}"]["last_seen"] = NOW - 3 * 24 * HOUR - (9 - i) * 24 * HOUR  # 3-12 days ago
    db.G.nodes[f"C_{{i}}"]["last_seen"] = NOW - 24 * HOUR - (9 - i) * 6 * HOUR       # 1-3.25 days ago

# ---- 3) Shape check ----
result_shape = {{"shape_ok": False, "ordering_correct": 0.0,
                 "beats_baseline": 0.0, "no_regression": 0.0,
                 "errors": []}}

try:
    out = db.query_temporal(["A_0"], k=5)
    assert isinstance(out, list), "query_temporal must return list"
    assert len(out) <= 5, "must respect k"
    if out:
        first = out[0]
        assert isinstance(first, dict), "each item must be dict"
        required = {{"id", "type", "role", "temporal_score",
                     "causal_score", "combined_score", "source"}}
        missing = required - set(first.keys())
        assert not missing, f"missing keys: {{missing}}"
        assert first["source"] == "temporal", "source must be 'temporal'"
        # Check sorted by combined_score descending
        scores = [r["combined_score"] for r in out]
        assert scores == sorted(scores, reverse=True), "must sort by combined_score desc"
    result_shape["shape_ok"] = True
except Exception as e:
    result_shape["errors"].append(f"shape: {{e}}")

# ---- 4) Ordering correctness ----
# For query [A_0], expected ordering (top-5) should include close-in-chain
# or temporally-close nodes. A_1..A_4 should dominate (causal near + temporal close).
try:
    out = db.query_temporal(["A_0"], k=5)
    returned_ids = [r["id"] for r in out]
    # Expected: at least 3 of top-5 should be in chain A (most temporally close
    # AND causally closest).
    chain_A_in_top5 = sum(1 for nid in returned_ids if nid.startswith("A_"))
    result_shape["ordering_correct"] = min(1.0, chain_A_in_top5 / 3)
except Exception as e:
    result_shape["errors"].append(f"ordering: {{e}}")

# ---- 5) Beats recency-only baseline ----
# A time-sensitive query: query for A_5 should find temporally-close nodes.
# The baseline sorts only by |last_seen - query.last_seen|, ignoring chain.
# Temporal mode (with causal component) should do BETTER on chain membership.
def recency_baseline(query_nodes, k):
    q_ls = [db.G.nodes[q]["last_seen"] for q in query_nodes if q in db.G]
    if not q_ls:
        return []
    q_mean = sum(q_ls) / len(q_ls)
    scored = []
    for nid, data in db.G.nodes(data=True):
        if nid in query_nodes:
            continue
        d = abs(data["last_seen"] - q_mean)
        scored.append((nid, -d))
    scored.sort(key=lambda x: -x[1])
    return [nid for nid, _ in scored[:k]]

try:
    # Evaluate how many top-5 results are in the SAME CHAIN as the query.
    # Temporal mode should pick its chain; recency might pick cross-chain
    # nodes that happen to have close timestamps.
    wins = 0
    total = 0
    for seed in ("A_5", "B_5", "C_5", "A_0", "C_0"):
        t_out = db.query_temporal([seed], k=5)
        t_ids = [r["id"] for r in t_out]
        r_ids = recency_baseline([seed], k=5)
        chain = seed[0]
        t_hits = sum(1 for n in t_ids if n.startswith(chain))
        r_hits = sum(1 for n in r_ids if n.startswith(chain))
        total += 1
        if t_hits > r_hits:
            wins += 1
        elif t_hits == r_hits:
            wins += 0.5
    result_shape["beats_baseline"] = wins / total
except Exception as e:
    result_shape["errors"].append(f"baseline: {{e}}")

# ---- 6) No regression on existing KnowledgeGraphDB methods ----
try:
    # Sanity: existing graph-DB methods still function after monkey-patch.
    reach = db.get_reachable("A_0", max_hops=10)
    assert isinstance(reach, set), "get_reachable must return set"
    assert "A_1" in reach and "A_5" in reach, (
        f"get_reachable regressed — expected A_1 and A_5 in reach, got {{sorted(reach)[:10]}}")
    sub = db.extract_subgraph(["A_0", "A_1", "A_2"], max_nodes=10)
    assert isinstance(sub, dict) and "nodes" in sub and "edges" in sub
    chain = db.trace_causal_chain("A_9")
    assert chain.get("root") == "A_0", (
        f"trace_causal_chain regressed — expected root A_0, got {{chain.get('root')}}")
    result_shape["no_regression"] = 1.0
except Exception as e:
    result_shape["errors"].append(f"regression: {{e}}")

# ---- 7) Compute reward ----
reward = (0.25 * (1.0 if result_shape["shape_ok"] else 0.0)
          + 0.25 * result_shape["ordering_correct"]
          + 0.30 * result_shape["beats_baseline"]
          + 0.20 * result_shape["no_regression"])

result = {{**result_shape, "reward": reward}}
print("RESULT_JSON: " + json.dumps(result))
'''


# ----------------------------------------------------------------------
# Loop
# ----------------------------------------------------------------------


def build_messages(current_attempt_code: Optional[str],
                   previous_error: Optional[str],
                   reward_breakdown: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    system = textwrap.dedent('''\
        You write Python functions that extend an existing class at
        runtime via monkey-patching. Output exactly ONE fenced
        ```python``` block containing the standalone functions below.
        Do NOT write a class. Do NOT redefine existing methods other
        than `query_relevant_patched`. Do NOT include any test/harness.
    ''') + "\n\n" + TASK_PROMPT

    user = "Write both functions."
    if previous_error:
        user += (f"\n\nPrevious attempt failed with:\n{previous_error[-1500:]}\n"
                 f"\nReward breakdown: {json.dumps(reward_breakdown, default=str) if reward_breakdown else '(none)'}\n"
                 f"\nFix the failure and output the full class again.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


def run_code_capture(code: str, out_dir: Path, iteration: int,
                      timeout_s: int = 120) -> Dict[str, Any]:
    script = out_dir / f"iter_{iteration:02d}.py"
    with open(script, "w") as f:
        f.write(code)
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "/workspace/liquid-arc")
    try:
        proc = subprocess.run(
            ["python", str(script)], env=env,
            cwd="/workspace/liquid-arc",
            capture_output=True, text=True, timeout=timeout_s,
        )
        stdout, stderr, ec = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\n[TIMEOUT]"
        ec = -1
    result_json = None
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT_JSON: "):
            try:
                result_json = json.loads(line[len("RESULT_JSON: "):])
            except Exception:
                pass
    return {"stdout": stdout, "stderr": stderr, "exit_code": ec,
            "result_json": result_json}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm_url", default="http://172.17.0.1:30000/v1")
    p.add_argument("--model", default="Qwen3-Next-80B-A3B-Instruct-FP8")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_rounds", type=int, default=10)
    p.add_argument("--target_reward", type=float, default=0.90)
    p.add_argument("--patience", type=int, default=4)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace: List[Dict[str, Any]] = []
    best_reward = -1.0
    best_code: Optional[str] = None
    best_result: Optional[Dict[str, Any]] = None
    previous_error: Optional[str] = None
    previous_breakdown: Optional[Dict[str, Any]] = None
    stagnation = 0

    for rnd in range(args.max_rounds):
        print(f"\n===== round {rnd+1}/{args.max_rounds} =====", flush=True)
        messages = build_messages(best_code, previous_error, previous_breakdown)
        t0 = time.time()
        try:
            response = llm_generate(args.vllm_url, args.model, messages,
                                     max_tokens=6000, temperature=0.1)
        except Exception as exc:
            print(f"[llm-error] {exc}", flush=True)
            trace.append({"round": rnd, "llm_error": str(exc)})
            continue
        gen_s = time.time() - t0
        candidate = extract_code(response)
        if not candidate:
            print(f"[no-code-block] {gen_s:.1f}s", flush=True)
            previous_error = "Response had no ```python``` fenced block."
            trace.append({"round": rnd, "gen_s": gen_s, "no_code": True})
            stagnation += 1
            if stagnation >= args.patience:
                break
            continue

        # Assemble a standalone script that patches the candidate into
        # the graph_db module and runs the harness.
        harness = HARNESS_TEMPLATE.format(candidate_code=candidate)
        run = run_code_capture(harness, out_dir, rnd, timeout_s=180)
        reward = 0.0
        components = {}
        if run["result_json"] is not None:
            reward = run["result_json"].get("reward", 0.0)
            components = run["result_json"]
        trace.append({
            "round": rnd, "gen_s": gen_s,
            "exit_code": run["exit_code"],
            "reward": reward,
            "components": {k: v for k, v in components.items()
                            if k in ("shape_ok", "ordering_correct",
                                     "beats_baseline", "no_regression",
                                     "errors")},
            "stderr_tail": (run["stderr"] or "")[-500:],
        })
        print(f"  reward={reward:.3f}  "
              f"shape={components.get('shape_ok')}  "
              f"ord={components.get('ordering_correct', 0):.2f}  "
              f"beat={components.get('beats_baseline', 0):.2f}  "
              f"regr={components.get('no_regression', 0):.2f}  "
              f"{gen_s:.1f}s", flush=True)
        if components.get("errors"):
            print(f"  errors: {components['errors']}", flush=True)

        if reward > best_reward + 1e-6:
            best_reward = reward
            best_code = candidate
            best_result = components
            stagnation = 0
            with open(out_dir / "best_candidate.py", "w") as f:
                f.write(candidate)
        else:
            stagnation += 1

        if reward >= args.target_reward:
            print(f"  target hit — stopping", flush=True)
            break
        if stagnation >= args.patience:
            print(f"  patience exhausted — stopping", flush=True)
            break

        # Prepare feedback for next round
        if run["result_json"] is None:
            previous_error = ("Harness didn't produce RESULT_JSON. Most likely "
                               "your code raised before reaching the harness "
                               "scoring. STDERR tail:\n"
                               + (run["stderr"] or "")[-1500:])
        else:
            err_list = components.get("errors") or []
            previous_error = (f"Reward {reward:.3f}; components = "
                               f"shape_ok={components.get('shape_ok')} "
                               f"ordering_correct={components.get('ordering_correct')} "
                               f"beats_baseline={components.get('beats_baseline')} "
                               f"no_regression={components.get('no_regression')}. "
                               f"Errors: {err_list}")
        previous_breakdown = components

    summary = {
        "status": "success" if best_reward >= args.target_reward else "partial",
        "best_reward": best_reward,
        "best_result": best_result,
        "rounds": len(trace),
        "trace": trace,
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n=== STAGE-1 SUMMARY ===")
    print(f"  best reward: {best_reward:.3f}  status: {summary['status']}")
    print(f"  rounds: {len(trace)}")
    sys.exit(0 if summary["status"] == "success" else 1)


if __name__ == "__main__":
    main()
