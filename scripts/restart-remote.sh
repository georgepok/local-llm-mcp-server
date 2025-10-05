#!/bin/bash

# Restart Local LLM MCP Server in REMOTE mode

set -e

cd "$(dirname "$0")"

echo "🔄 Restarting Local LLM MCP Server (Remote Mode)..."
echo "===================================================="
echo ""

# Stop existing instances
./stop.sh

# Wait a moment
sleep 1

# Start in remote mode
./start-remote.sh
