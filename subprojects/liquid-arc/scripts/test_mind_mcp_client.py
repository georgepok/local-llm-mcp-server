"""Test LiquidARC Mind MCP server using a full MCP client.

Connects via SSE transport, lists tools, and exercises each one.
"""
import asyncio
import json
from fastmcp import Client


SERVER_URL = "http://spark-129a.local:8420/sse"


async def call_tool(client, name: str, args: dict) -> dict:
    """Call an MCP tool and return parsed result."""
    result = await client.call_tool(name, args)
    # CallToolResult has .content list of TextContent/etc
    text = result.content[0].text if result.content else "{}"
    return json.loads(text)


async def main():
    print(f"Connecting to {SERVER_URL}...")
    async with Client(SERVER_URL) as client:
        # 1. List available tools
        print("\n=== Tool Discovery ===")
        tools = await client.list_tools()
        for t in tools:
            print(f"  {t.name}: {t.description[:80]}")
        tool_names = {t.name for t in tools}
        assert 'observe_event' in tool_names, "Missing observe_event tool"
        assert 'get_context' in tool_names, "Missing get_context tool"
        assert 'get_diagnostics' in tool_names, "Missing get_diagnostics tool"
        assert 'provide_feedback' in tool_names, "Missing provide_feedback tool"
        assert 'signal_goal' in tool_names, "Missing signal_goal tool"
        assert 'reset' in tool_names, "Missing reset tool"
        print(f"  All 6 tools discovered OK")

        # 2. Reset to clean state
        print("\n=== Reset ===")
        r = await call_tool(client, 'reset', {})
        print(f"  {r}")

        # 3. Observe a conversation
        print("\n=== Observe Events ===")
        r1 = await call_tool(client, 'observe_event', {
            'event_type': 'user_message',
            'content': 'Can you help me understand how neural networks learn?',
        })
        print(f"  Event 1 (user): pred_err={r1['prediction_error']:.3f}, "
              f"cv={r1['cv']:.3f}, h_norm={r1['h_norm']:.3f}")

        r2 = await call_tool(client, 'observe_event', {
            'event_type': 'assistant_message',
            'content': 'Neural networks learn through backpropagation, '
                       'adjusting weights to minimize a loss function.',
        })
        print(f"  Event 2 (assistant): pred_err={r2['prediction_error']:.3f}, "
              f"cv={r2['cv']:.3f}")

        r3 = await call_tool(client, 'observe_event', {
            'event_type': 'user_message',
            'content': 'What about gradient descent specifically?',
        })
        print(f"  Event 3 (user): pred_err={r3['prediction_error']:.3f}, "
              f"cv={r3['cv']:.3f}")

        # 4. Get context — relevance-scored events
        print("\n=== Get Context ===")
        ctx = await call_tool(client, 'get_context', {})
        print(f"  Status: {ctx['status']}, Events: {ctx['n_events']}")
        print(f"  Focus indices: {ctx['focus_indices']}")
        for item in ctx['context']:
            print(f"    [{item['type']:15s}] rel={item['relevance']:.3f} | "
                  f"{item['preview'][:55]}")

        # 5. Signal a goal
        print("\n=== Signal Goal ===")
        gr = await call_tool(client, 'signal_goal', {
            'goal_text': 'Help user understand deep learning fundamentals',
            'priority': 0.9,
        })
        print(f"  Goal: pred_err={gr['prediction_error']:.3f}, "
              f"events={gr['events_in_context']}")

        # 6. Get diagnostics
        print("\n=== Diagnostics ===")
        diag = await call_tool(client, 'get_diagnostics', {})
        print(f"  Status: {diag['status']}")
        print(f"  CV: {diag['metric_cv']:.3f}")
        print(f"  Tau: mean={diag['tau_mean']:.3f}, std={diag['tau_std']:.3f}")
        print(f"  Beta: mean={diag['beta_mean']:.3f}, std={diag['beta_std']:.3f}")
        print(f"  Events: {diag['events_in_context']} in context, "
              f"{diag['event_count_total']} total")

        # 7. Topic shift — does prediction error spike?
        print("\n=== Topic Shift ===")
        r_shift = await call_tool(client, 'observe_event', {
            'event_type': 'user_message',
            'content': 'Actually, forget ML. Tell me about the history '
                       'of the Roman Empire and Julius Caesar.',
        })
        print(f"  Topic shift: pred_err={r_shift['prediction_error']:.3f} "
              f"(prev was {r3['prediction_error']:.3f})")

        # 8. Build longer conversation
        print("\n=== Extended Conversation ===")
        msgs = [
            ('user_message', 'When was the Roman Empire founded?'),
            ('assistant_message', 'Rome was traditionally founded in 753 BC.'),
            ('user_message', 'What about the fall of Rome?'),
            ('assistant_message', 'The Western Empire fell in 476 AD.'),
            ('tool_result', '{"search": "fall of rome", "results": ["476 AD"]}'),
            ('user_message', 'What were the main causes?'),
        ]
        for etype, content in msgs:
            r = await call_tool(client, 'observe_event', {
                'event_type': etype,
                'content': content,
            })
        print(f"  After {r['events_in_context']} events: "
              f"pred_err={r['prediction_error']:.3f}, cv={r['cv']:.3f}")

        # Get final context with all events
        ctx_final = await call_tool(client, 'get_context', {})
        print(f"  Context has {ctx_final['n_events']} events")
        print(f"  Top 3 by relevance:")
        for item in ctx_final['context'][:3]:
            print(f"    [{item['type']:15s}] rel={item['relevance']:.3f} | "
                  f"{item['preview'][:55]}")

        # 9. Final diagnostics
        print("\n=== Final Diagnostics ===")
        diag_final = await call_tool(client, 'get_diagnostics', {})
        print(f"  CV: {diag_final['metric_cv']:.3f}")
        print(f"  h_norm: {diag_final['h_norm']:.3f}")
        print(f"  Events: {diag_final['event_count_total']} total")

        # 10. Reset and verify clean state
        print("\n=== Final Reset ===")
        await call_tool(client, 'reset', {})
        ctx_empty = await call_tool(client, 'get_context', {})
        assert ctx_empty['status'] == 'no_state', f"Expected no_state, got {ctx_empty['status']}"
        print(f"  Clean reset confirmed: {ctx_empty['status']}")

        print("\n=== ALL MCP CLIENT TESTS PASSED ===")


if __name__ == '__main__':
    asyncio.run(main())
