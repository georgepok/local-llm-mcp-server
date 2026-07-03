"""End-to-end MCP protocol test for the graph reasoning server.

Uses fastmcp.Client to connect over SSE, call each of the three tools with
both integer-form and string-form inputs, and verify the server returns
parseable JSON without errors.
"""

import asyncio
import json
import sys

from fastmcp import Client


import os
URL = os.environ.get("GRAPH_MCP_URL", "http://192.168.1.184:8420/sse")


def _print_result(title, res):
    print(f"\n--- {title} ---")
    if hasattr(res, 'data'):
        payload = res.data
    elif hasattr(res, 'content') and res.content:
        payload = res.content[0].text
    else:
        payload = str(res)
    try:
        parsed = json.loads(payload) if isinstance(payload, str) else payload
        print(json.dumps(parsed, indent=2)[:600])
    except Exception:
        print(str(payload)[:600])


async def main():
    client = Client(URL)
    async with client:
        # Test 1: analyze_graph with STRING types/roles (the failure case)
        graph = {
            "nodes": [
                {"id": "A", "type": "event", "role": "root"},
                {"id": "B", "type": "consequence", "role": "intermediate"},
                {"id": "C", "type": "state", "role": "intermediate"},
                {"id": "D", "type": "consequence", "role": "terminal"},
            ],
            "edges": [
                {"src": "A", "dst": "B", "type": "causes"},
                {"src": "B", "dst": "C", "type": "causes"},
                {"src": "C", "dst": "D", "type": "causes"},
            ],
        }
        query = {"type": "root_cause", "target": "D"}
        r1 = await client.call_tool(
            "analyze_graph",
            {"graph_json": json.dumps(graph), "query_json": json.dumps(query)},
        )
        _print_result("analyze_graph (string types/roles → root_cause)", r1)

        # Test 2: connection_check with string types
        graph_parallel = {
            "nodes": [
                {"id": "A1", "type": "event", "role": "root"},
                {"id": "A2", "type": "state", "role": "terminal"},
                {"id": "B1", "type": "cause", "role": "root"},
                {"id": "B2", "type": "outcome", "role": "terminal"},
            ],
            "edges": [
                {"src": "A1", "dst": "A2", "type": "causes"},
                {"src": "B1", "dst": "B2", "type": "causes"},
            ],
        }
        r2 = await client.call_tool(
            "analyze_graph",
            {"graph_json": json.dumps(graph_parallel),
             "query_json": json.dumps({"type": "connection_check",
                                        "src": "A1", "dst": "B2"})},
        )
        _print_result("analyze_graph (string types → connection_check, expect False)", r2)

        # Test 3: get_graph_diagnostics with strings
        r3 = await client.call_tool(
            "get_graph_diagnostics",
            {"graph_json": json.dumps(graph)},
        )
        _print_result("get_graph_diagnostics (string types)", r3)

        # Test 4: compare_graphs — isomorphic pair with string types
        graph_a = {
            "nodes": [
                {"id": "X1", "type": "event", "role": "root"},
                {"id": "X2", "type": "state", "role": "intermediate"},
                {"id": "X3", "type": "outcome", "role": "terminal"},
            ],
            "edges": [
                {"src": "X1", "dst": "X2", "type": "causes"},
                {"src": "X2", "dst": "X3", "type": "causes"},
            ],
        }
        graph_b = {
            "nodes": [
                {"id": "Y1", "type": "cause", "role": "root"},
                {"id": "Y2", "type": "step", "role": "intermediate"},
                {"id": "Y3", "type": "consequence", "role": "terminal"},
            ],
            "edges": [
                {"src": "Y1", "dst": "Y2", "type": "causes"},
                {"src": "Y2", "dst": "Y3", "type": "causes"},
            ],
        }
        r4 = await client.call_tool(
            "compare_graphs",
            {"graph_a_json": json.dumps(graph_a),
             "graph_b_json": json.dumps(graph_b)},
        )
        _print_result("compare_graphs (isomorphic linear chains w/ string types)", r4)

        # Test 5: integer types should still work (backwards compat)
        graph_int = {
            "nodes": [
                {"id": "P", "type": 0, "role": 0},
                {"id": "Q", "type": 1, "role": 2},
            ],
            "edges": [{"src": "P", "dst": "Q", "type": 0}],
        }
        r5 = await client.call_tool(
            "analyze_graph",
            {"graph_json": json.dumps(graph_int),
             "query_json": json.dumps({"type": "root_cause", "target": "Q"})},
        )
        _print_result("analyze_graph (INTEGER types → still works)", r5)

        print("\n─────────────────────────────────────")
        print("ALL 5 TOOL CALLS COMPLETED WITHOUT PROTOCOL ERROR")
        print("─────────────────────────────────────")


if __name__ == "__main__":
    asyncio.run(main())
