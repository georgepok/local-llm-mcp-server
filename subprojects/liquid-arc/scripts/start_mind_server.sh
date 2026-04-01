#!/bin/bash
# Install dependencies and start MCP server
pip install sentence-transformers fastmcp --break-system-packages -q 2>/dev/null
exec python -u -m liquid_arc.mcp_serve "$@"
