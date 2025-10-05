#!/bin/bash

# Start Local LLM MCP Server in HTTPS mode
# Requires SSL certificate and key files

set -e

cd "$(dirname "$0")/.."

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -h|--host)
      HOST="$2"
      shift 2
      ;;
    --cert)
      CERT_PATH="$2"
      shift 2
      ;;
    --key)
      KEY_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--port PORT] [--host HOST] [--cert CERT_PATH] [--key KEY_PATH]"
      exit 1
      ;;
  esac
done

# Default configuration
PORT=${PORT:-3000}
HOST=${HOST:-0.0.0.0}
CERT_PATH=${HTTPS_CERT_PATH:-${CERT_PATH:-"./certs/cert.pem"}}
KEY_PATH=${HTTPS_KEY_PATH:-${KEY_PATH:-"./certs/key.pem"}}

echo "🔒 Starting Local LLM MCP Server (HTTPS Mode)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Port: $PORT"
echo "  Host: $HOST"
echo "  Certificate: $CERT_PATH"
echo "  Key: $KEY_PATH"
echo ""

# Check if certificate files exist
if [ ! -f "$CERT_PATH" ]; then
  echo "❌ Certificate file not found: $CERT_PATH"
  echo ""
  echo "To generate self-signed certificates for development:"
  echo "  mkdir -p certs"
  echo "  openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes"
  echo ""
  exit 1
fi

if [ ! -f "$KEY_PATH" ]; then
  echo "❌ Key file not found: $KEY_PATH"
  exit 1
fi

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
    echo "   Access from network: https://$LOCAL_IP:$PORT"
    echo ""
  fi
fi

echo "To stop: Press Ctrl+C or run ./scripts/stop.sh"
echo ""

# Start in HTTPS mode
MCP_TRANSPORT=https PORT=$PORT HOST=$HOST \
  HTTPS_CERT_PATH=$CERT_PATH HTTPS_KEY_PATH=$KEY_PATH \
  node dist/index.js
