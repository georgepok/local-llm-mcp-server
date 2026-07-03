"""Query the live MCP server for its tool list."""

import asyncio
import os

from fastmcp import Client


URL = os.environ.get("GRAPH_MCP_URL", "http://192.168.1.184:8420/sse")


async def run():
    client = Client(URL)
    async with client:
        tools = await client.list_tools()
        print(f"Server reports {len(tools)} tools:")
        for t in tools:
            print(f"  • {t.name}")
            desc = (t.description or "").strip().split('\n')[0][:100]
            print(f"      {desc}")


if __name__ == "__main__":
    asyncio.run(run())
