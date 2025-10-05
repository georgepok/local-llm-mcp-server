#!/bin/bash

# Start Local LLM MCP Server in REMOTE mode (HTTP/SSE)
# This mode allows access from other devices on your network

set -e

cd "$(dirname "$0")/.."

# Default configuration
PORT=${PORT:-3000}
HOST=${HOST:-0.0.0.0}

echo "🌐 Starting Local LLM MCP Server (Remote Mode - HTTP/SSE)"
echo "========================================================="
echo ""
echo "Server will be accessible on your network at:"
echo "  http://YOUR_IP:$PORT"
echo ""
echo "Configuration:"
echo "  Port: $PORT"
echo "  Host: $HOST"
echo ""
echo "To use a different port: PORT=8080 ./scripts/start-remote.sh"
echo "To stop: Press Ctrl+C or run ./scripts/stop.sh"
echo ""

# Check if dist exists
if [ ! -d "dist" ]; then
  echo "❌ Build directory not found. Running build..."
  npm run build
fi

# Find local IP address
if command -v ifconfig &> /dev/null; then
  LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)
  if [ -n "$LOCAL_IP" ]; then
    echo "💡 Your local IP address appears to be: $LOCAL_IP"
    echo "   Access from network: http://$LOCAL_IP:$PORT"
    echo ""
  fi
fi

# Start in HTTP mode
MCP_TRANSPORT=http PORT=$PORT HOST=$HOST node dist/index.js
