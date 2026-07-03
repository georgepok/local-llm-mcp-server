"""Verify the four correction safeguards:
  (1) Cap — correcting the same example >3 times returns an error.
  (2) Smaller LR — per-step delta is modest compared to old 1e-4 behavior.
  (3) Replay diversity — replay_per_correction stats are returned.
  (4) Health check — runs and reports variance.
"""

import asyncio
import json
import os

from fastmcp import Client


URL = os.environ.get("GRAPH_MCP_URL", "http://192.168.1.184:8420/sse")


def graph_example(tag: str):
    """Produce N graphs that share topology but use different node labels,
    so the cap-check (which hashes the whole example) sees them as distinct."""
    return {
        "nodes": [
            {"id": f"{tag}_scope_a", "type": "role", "role": "scope"},
            {"id": f"{tag}_scope_b", "type": "role", "role": "scope"},
            {"id": f"{tag}_cred", "type": "credential", "role": "intermediate"},
            {"id": f"{tag}_req", "type": "requirement", "role": "intermediate"},
            {"id": f"{tag}_prereq_a", "type": "prerequisite", "role": "terminal"},
            {"id": f"{tag}_prereq_b", "type": "prerequisite", "role": "terminal"},
        ],
        "edges": [
            {"src": f"{tag}_scope_a", "dst": f"{tag}_cred", "type": "requires"},
            {"src": f"{tag}_scope_b", "dst": f"{tag}_cred", "type": "requires"},
            {"src": f"{tag}_cred", "dst": f"{tag}_req", "type": "requires"},
            {"src": f"{tag}_req", "dst": f"{tag}_prereq_a", "type": "requires",
             "scope": f"{tag}_scope_a"},
            {"src": f"{tag}_req", "dst": f"{tag}_prereq_b", "type": "requires",
             "scope": f"{tag}_scope_b"},
        ],
    }


def query_example(tag: str):
    return {"type": "implication_check",
            "premise": f"{tag}_req",
            "conclusion": f"{tag}_prereq_a",
            "context_scope": f"{tag}_scope_a"}


def _parse(r):
    return json.loads(r.data) if hasattr(r, 'data') else json.loads(r.content[0].text)


async def run():
    client = Client(URL)
    async with client:
        g0 = graph_example("A")
        q0 = query_example("A")

        print("── TEST 1: cap enforcement ──────────────────────────")
        for i in range(5):
            r = await client.call_tool("correct_answer", {
                "graph_json": json.dumps(g0),
                "query_json": json.dumps(q0),
                "correct_answer_json": json.dumps({"valid": True}),
            })
            d = _parse(r)
            if d.get('error') == 'correction_cap_reached':
                print(f"  step {i+1}: CAP HIT — cap={d['cap']}, "
                      f"count_on_this_example={d['corrections_on_this_example']}")
            else:
                print(f"  step {i+1}: learned={d.get('learned')}  "
                      f"on_this_example={d.get('corrections_on_this_example')}  "
                      f"replay={d.get('replay', {}).get('n_replayed')}  "
                      f"health_collapsed={d.get('health', {}).get('collapsed')}  "
                      f"rolled_back={d.get('rolled_back')}")

        print("\n── TEST 2: diverse examples (mix of True/False) keep learning ─")
        # Teach on 6 DIFFERENT graphs, alternating labels so health check has
        # label diversity to measure against.
        cases = [("B", True), ("C", False), ("D", True), ("E", False), ("F2", True), ("G2", False)]
        for tag, want in cases:
            r = await client.call_tool("correct_answer", {
                "graph_json": json.dumps(graph_example(tag)),
                "query_json": json.dumps(query_example(tag)),
                "correct_answer_json": json.dumps({"valid": want}),
            })
            d = _parse(r)
            h = d.get('health', {})
            print(f"  tag={tag} want={want}: before={d.get('before_loss', 0):.4f}  "
                  f"after={d.get('after_loss', 0):.4f}  "
                  f"replay_n={d.get('replay', {}).get('n_replayed')}  "
                  f"out_var={h.get('output_variance')}  "
                  f"lab_var={h.get('label_variance')}  "
                  f"corr={h.get('pred_label_correlation')}  "
                  f"collapsed={h.get('collapsed')}  "
                  f"rolled_back={d.get('rolled_back')}")

        print("\n── TEST 3: new LR produces gentler per-step movement ─")
        # Fresh example; compare before/after — at lr=1e-5 the first-step
        # delta should be <<1 order of magnitude (old lr=1e-4 produced 0.62→0.87)
        r = await client.call_tool("correct_answer", {
            "graph_json": json.dumps(graph_example("F")),
            "query_json": json.dumps(query_example("F")),
            "correct_answer_json": json.dumps({"valid": True}),
        })
        d = _parse(r)
        bl = d.get('before_loss', 0.0)
        al = d.get('after_loss', 0.0)
        delta = bl - al
        print(f"  before_loss={bl:.4f}  after_loss={al:.4f}  Δ={delta:+.4f}")
        print(f"  (at lr=1e-5, Δ should be much smaller than at lr=1e-4;"
              f" expect ≲ 0.05 per step)")


if __name__ == "__main__":
    asyncio.run(run())
