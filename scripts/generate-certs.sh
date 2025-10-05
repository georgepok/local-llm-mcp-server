#!/bin/bash

# Generate self-signed SSL certificates for HTTPS mode
# For development and testing purposes only

set -e

cd "$(dirname "$0")/.."

CERT_DIR="certs"
CERT_FILE="$CERT_DIR/cert.pem"
KEY_FILE="$CERT_DIR/key.pem"

echo "🔐 Generating Self-Signed SSL Certificates"
echo "=========================================="
echo ""

# Create certs directory if it doesn't exist
if [ ! -d "$CERT_DIR" ]; then
  echo "Creating certificates directory: $CERT_DIR"
  mkdir -p "$CERT_DIR"
fi

# Check if certificates already exist
if [ -f "$CERT_FILE" ] && [ -f "$KEY_FILE" ]; then
  echo "⚠️  Certificates already exist:"
  echo "   Certificate: $CERT_FILE"
  echo "   Key: $KEY_FILE"
  echo ""
  read -p "Overwrite existing certificates? (y/N) " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
  fi
fi

echo ""
echo "Generating RSA 4096-bit self-signed certificate..."
echo "Valid for 365 days"
echo ""
echo "You will be prompted for certificate details."
echo "You can press Enter to use defaults for most fields."
echo ""

# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 \
  -keyout "$KEY_FILE" \
  -out "$CERT_FILE" \
  -days 365 \
  -nodes \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

echo ""
echo "✅ Certificates generated successfully!"
echo ""
echo "Files created:"
echo "  Certificate: $CERT_FILE"
echo "  Key: $KEY_FILE"
echo ""
echo "⚠️  IMPORTANT SECURITY NOTES:"
echo "  - These are SELF-SIGNED certificates for DEVELOPMENT only"
echo "  - Browsers will show security warnings"
echo "  - NOT suitable for production use"
echo "  - For production, use certificates from a trusted CA (Let's Encrypt, etc.)"
echo ""
echo "To start HTTPS server:"
echo "  ./scripts/start-https.sh"
echo ""
echo "To use custom certificate paths:"
echo "  HTTPS_CERT_PATH=/path/to/cert.pem HTTPS_KEY_PATH=/path/to/key.pem ./scripts/start-https.sh"
echo ""
