#!/bin/bash

# Start Local LLM MCP Server in DUAL mode
# Runs both stdio (for Claude Desktop) and HTTP/HTTPS (for network access) simultaneously

set -e

cd "$(dirname "$0")/.."

# Default configuration
PORT=${PORT:-3000}
HOST=${HOST:-0.0.0.0}
USE_HTTPS=${USE_HTTPS:-false}

echo "🔄 Starting Local LLM MCP Server (Dual Mode)"
echo "============================================"
echo ""

if [ "$USE_HTTPS" = "true" ]; then
  CERT_PATH=${HTTPS_CERT_PATH:-"./certs/cert.pem"}
  KEY_PATH=${HTTPS_KEY_PATH:-"./certs/key.pem"}

  echo "Mode: Stdio + HTTPS"
  echo "Configuration:"
  echo "  Port: $PORT"
  echo "  Host: $HOST"
  echo "  Certificate: $CERT_PATH"
  echo "  Key: $KEY_PATH"
  echo ""

  # Check certificates
  if [ ! -f "$CERT_PATH" ] || [ ! -f "$KEY_PATH" ]; then
    echo "❌ Certificate or key file not found"
    echo "Run: ./scripts/start-https.sh for certificate generation instructions"
    exit 1
  fi
else
  echo "Mode: Stdio + HTTP"
  echo "Configuration:"
  echo "  Port: $PORT"
  echo "  Host: $HOST"
  echo ""
fi

# Check if dist exists
if [ ! -d "dist" ]; then
  echo "❌ Build directory not found. Running build..."
  npm run build
fi

echo "This server will be available:"
echo "  - Via stdio for Claude Desktop (local)"
if [ "$USE_HTTPS" = "true" ]; then
  echo "  - Via HTTPS on port $PORT (network)"
else
  echo "  - Via HTTP on port $PORT (network)"
fi
echo ""

# Find local IP address
if command -v ifconfig &> /dev/null; then
  LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -n 1)
  if [ -n "$LOCAL_IP" ]; then
    echo "💡 Your local IP address appears to be: $LOCAL_IP"
    if [ "$USE_HTTPS" = "true" ]; then
      echo "   Network access: https://$LOCAL_IP:$PORT"
    else
      echo "   Network access: http://$LOCAL_IP:$PORT"
    fi
    echo ""
  fi
fi

echo "To stop: Press Ctrl+C or run ./scripts/stop.sh"
echo ""

# Start in dual mode
if [ "$USE_HTTPS" = "true" ]; then
  MCP_DUAL_MODE=true MCP_HTTPS=true PORT=$PORT HOST=$HOST \
    HTTPS_CERT_PATH=$CERT_PATH HTTPS_KEY_PATH=$KEY_PATH \
    node dist/index.js
else
  MCP_DUAL_MODE=true PORT=$PORT HOST=$HOST node dist/index.js
fi
