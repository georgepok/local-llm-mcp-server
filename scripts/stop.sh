#!/bin/bash

# Stop Local LLM MCP Server instances

echo "🛑 Stopping Local LLM MCP Server..."
echo "===================================="
echo ""

# Find node processes running the MCP server
PIDS=$(pgrep -f "node.*dist/index.js" || true)

if [ -z "$PIDS" ]; then
  echo "ℹ️  No running server instances found."
  exit 0
fi

echo "Found running server process(es):"
echo "$PIDS" | while read pid; do
  echo "  PID: $pid"
done
echo ""

# Kill processes
echo "Stopping server(s)..."
echo "$PIDS" | while read pid; do
  kill -SIGINT "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null
  echo "  ✓ Stopped PID: $pid"
done

echo ""
echo "✅ Server stopped successfully"
