#!/bin/bash

# Start Local LLM MCP Server in LOCAL mode (stdio)
# This mode is used by Claude Desktop for local access

set -e

cd "$(dirname "$0")/.."

echo "🚀 Starting Local LLM MCP Server (Local Mode - stdio)"
echo "=================================================="
echo ""
echo "This server runs in stdio mode for Claude Desktop."
echo "To stop: Press Ctrl+C or run ./scripts/stop.sh"
echo ""

# Check if dist exists
if [ ! -d "dist" ]; then
  echo "❌ Build directory not found. Running build..."
  npm run build
fi

# Start in stdio mode (default)
node dist/index.js
