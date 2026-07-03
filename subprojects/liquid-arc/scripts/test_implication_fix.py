"""Verify the implication_check hybrid fix on the reported bug case."""

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


async def run():
    client = Client(URL)
    async with client:
        # Case 1: senior scope, crypto → linear_algebra  (expect valid: TRUE)
        q1 = {"type": "implication_check",
              "premise": "crypto_exam", "conclusion": "linear_algebra",
              "context_scope": "senior_engineer"}
        r1 = await client.call_tool(
            "analyze_graph",
            {"graph_json": json.dumps(GRAPH), "query_json": json.dumps(q1)})
        d1 = json.loads(r1.data) if hasattr(r1, 'data') else json.loads(r1.content[0].text)

        # Case 2: junior scope, crypto → linear_algebra  (expect valid: FALSE)
        q2 = {"type": "implication_check",
              "premise": "crypto_exam", "conclusion": "linear_algebra",
              "context_scope": "junior_developer"}
        r2 = await client.call_tool(
            "analyze_graph",
            {"graph_json": json.dumps(GRAPH), "query_json": json.dumps(q2)})
        d2 = json.loads(r2.data) if hasattr(r2, 'data') else json.loads(r2.content[0].text)

        # Case 3: junior scope, crypto → alt_pathway  (expect valid: TRUE)
        q3 = {"type": "implication_check",
              "premise": "crypto_exam", "conclusion": "alt_pathway",
              "context_scope": "junior_developer"}
        r3 = await client.call_tool(
            "analyze_graph",
            {"graph_json": json.dumps(GRAPH), "query_json": json.dumps(q3)})
        d3 = json.loads(r3.data) if hasattr(r3, 'data') else json.loads(r3.content[0].text)

        # Case 4: senior scope, crypto → alt_pathway (expect valid: FALSE)
        q4 = {"type": "implication_check",
              "premise": "crypto_exam", "conclusion": "alt_pathway",
              "context_scope": "senior_engineer"}
        r4 = await client.call_tool(
            "analyze_graph",
            {"graph_json": json.dumps(GRAPH), "query_json": json.dumps(q4)})
        d4 = json.loads(r4.data) if hasattr(r4, 'data') else json.loads(r4.content[0].text)

        def verdict(label, expected, result):
            got = result.get('valid')
            head = result.get('head_valid_prob')
            mark = 'PASS' if got == expected else 'FAIL'
            print(f"  {mark} {label:<55} expected={expected} "
                  f"got_authoritative={got} head_prob={head:.3f}")

        print("─" * 80)
        print("IMPLICATION CHECK — post-fix verification")
        print("─" * 80)
        verdict("senior / crypto ⊨ linear_algebra", True, d1)
        verdict("junior / crypto ⊨ linear_algebra", False, d2)
        verdict("junior / crypto ⊨ alt_pathway", True, d3)
        verdict("senior / crypto ⊨ alt_pathway", False, d4)


if __name__ == "__main__":
    asyncio.run(run())
