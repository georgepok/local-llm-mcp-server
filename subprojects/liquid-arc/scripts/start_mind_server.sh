#!/bin/bash
# Install dependencies and start MCP server
# sentence-transformers: legacy encoder (optional with --use_ode_encoder)
# transformers: tokenizer for ODE encoder
pip install sentence-transformers fastmcp transformers --break-system-packages -q 2>/dev/null
export PYTHONPATH=/workspace/liquid-arc:/workspace/fgn-v3:${PYTHONPATH:-}
exec python -u -m liquid_arc.mcp_serve "$@"
