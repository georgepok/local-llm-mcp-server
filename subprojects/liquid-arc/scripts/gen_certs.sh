#!/bin/bash
# Generate self-signed SSL certs for LiquidARC Mind MCP server
# Usage: bash scripts/gen_certs.sh [output_dir]

CERT_DIR="${1:-/workspace/liquid-arc/certs}"
mkdir -p "$CERT_DIR"

openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$CERT_DIR/mind_key.pem" \
    -out "$CERT_DIR/mind_cert.pem" \
    -days 365 \
    -subj "/CN=spark-129a.local/O=LiquidARC Mind" \
    -addext "subjectAltName=DNS:spark-129a.local,DNS:localhost,IP:192.168.1.184"

echo "Certs generated:"
echo "  Certificate: $CERT_DIR/mind_cert.pem"
echo "  Private key: $CERT_DIR/mind_key.pem"
echo ""
echo "Usage:"
echo "  --ssl_certfile $CERT_DIR/mind_cert.pem --ssl_keyfile $CERT_DIR/mind_key.pem --https_port 8421"
