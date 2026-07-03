"""Verify the correct_answer MCP tool ― full round-trip.

1. Call analyze_graph on the crypto_exam/linear_algebra/senior_eng case.
2. Observe head's valid_prob (expected: biased toward false — wrong).
3. Call correct_answer with {"valid": true}.
4. Call analyze_graph again — expect the head's valid_prob to have moved
   toward the correct answer.
5. Repeat correction several times — valid_prob should converge to >0.5.
"""

import asyncio
import json
import os

from fastmcp import Client


URL = os.environ.get("GRAPH_MCP_URL", "http://192.168.1.184:8420/sse")


GRAPH = {
    "nodes": [
        {"id": "senior_engineer", "type": "role", "role": "scope"},
        {"id": "junior_developer", "type": "role", "role": "scope"},
        {"id": "security_cert", "type": "credential", "role": "intermediate"},
        {"id": "network_exam", "type": "requirement", "role": "intermediate"},
        {"id": "crypto_exam", "type": "requirement", "role": "intermediate"},
        {"id": "linear_algebra", "type": "prerequisite", "role": "terminal"},
        {"id": "alt_pathway", "type": "prerequisite", "role": "terminal"},
    ],
    "edges": [
        {"src": "senior_engineer", "dst": "security_cert", "type": "requires"},
        {"src": "junior_developer", "dst": "security_cert", "type": "requires"},
        {"src": "security_cert", "dst": "network_exam", "type": "requires"},
        {"src": "security_cert", "dst": "crypto_exam", "type": "requires"},
        {"src": "crypto_exam", "dst": "linear_algebra", "type": "requires",
         "scope": "senior_engineer"},
        {"src": "crypto_exam", "dst": "alt_pathway", "type": "requires",
         "scope": "junior_developer"},
    ],
}
QUERY = {"type": "implication_check",
         "premise": "crypto_exam", "conclusion": "linear_algebra",
         "context_scope": "senior_engineer"}
CORRECT = {"valid": True}


def _parse(r):
    return json.loads(r.data) if hasattr(r, 'data') else json.loads(r.content[0].text)


async def run():
    client = Client(URL)
    async with client:
        # 1. Baseline
        r = await client.call_tool("analyze_graph", {
            "graph_json": json.dumps(GRAPH),
            "query_json": json.dumps(QUERY)})
        baseline = _parse(r)
        print("── BEFORE corrections ──────────────────────────────")
        print(f"  authoritative valid: {baseline['valid']}")
        print(f"  head_valid_prob:     {baseline['head_valid_prob']:.4f}")
        print(f"  head_says_valid:     {baseline['head_says_valid']}")

        # 2-5. Iterative corrections
        print("\n── APPLYING CORRECTIONS ────────────────────────────")
        for step in range(8):
            c = await client.call_tool("correct_answer", {
                "graph_json": json.dumps(GRAPH),
                "query_json": json.dumps(QUERY),
                "correct_answer_json": json.dumps(CORRECT)})
            cd = _parse(c)
            print(f"  step {step + 1}: before_loss={cd['before_loss']:.4f} "
                  f"→ after_loss={cd['after_loss']:.4f}  "
                  f"pred_valid_prob={cd['current_prediction']['valid_prob']:.4f}  "
                  f"corrections_total={cd['corrections_applied']}")

        # 6. Final check
        print("\n── AFTER 8 corrections ────────────────────────────")
        r = await client.call_tool("analyze_graph", {
            "graph_json": json.dumps(GRAPH),
            "query_json": json.dumps(QUERY)})
        after = _parse(r)
        print(f"  authoritative valid: {after['valid']}")
        print(f"  head_valid_prob:     {after['head_valid_prob']:.4f}")
        print(f"  head_says_valid:     {after['head_says_valid']}")
        delta = after['head_valid_prob'] - baseline['head_valid_prob']
        print(f"\n  DELTA head_valid_prob: {delta:+.4f} "
              f"({'learning' if delta > 0.1 else 'no meaningful change'})")


if __name__ == "__main__":
    asyncio.run(run())
