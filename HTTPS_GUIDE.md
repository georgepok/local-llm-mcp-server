# HTTPS and Dual Mode Guide

Complete guide for running the Local LLM MCP Server with HTTPS encryption and dual transport mode.

## Quick Reference

```bash
# Generate self-signed certificates (development)
npm run generate:certs

# Start HTTPS server
npm run start:https

# Start dual mode (stdio + HTTP)
npm run start:dual

# Start dual mode with HTTPS
USE_HTTPS=true npm run start:dual
```

---

## Table of Contents

1. [Transport Modes](#transport-modes)
2. [HTTPS Setup](#https-setup)
3. [Dual Mode](#dual-mode)
4. [Certificate Management](#certificate-management)
5. [Production Deployment](#production-deployment)
6. [Troubleshooting](#troubleshooting)

---

## Transport Modes

The server supports three transport modes:

### 1. Stdio Mode (Default)
**Purpose**: Local Claude Desktop integration
**Usage**: `npm start` or `npm run start:local`
**Protocol**: Standard input/output (IPC)

### 2. HTTP Mode
**Purpose**: Remote network access without encryption
**Usage**: `npm run start:remote`
**Protocol**: HTTP (unencrypted)

### 3. HTTPS Mode
**Purpose**: Secure remote network access
**Usage**: `npm run start:https`
**Protocol**: HTTPS (TLS/SSL encrypted)

### 4. Dual Mode
**Purpose**: Both local (stdio) and remote (HTTP/HTTPS) simultaneously
**Usage**: `npm run start:dual`
**Protocols**: Stdio + HTTP/HTTPS

---

## HTTPS Setup

### Step 1: Generate Certificates

For **development/testing** (self-signed):

```bash
npm run generate:certs
```

This creates:
- `certs/cert.pem` - SSL certificate
- `certs/key.pem` - Private key

For **production**, use certificates from a trusted CA (see [Production Deployment](#production-deployment)).

### Step 2: Start HTTPS Server

```bash
npm run start:https
```

Or with custom paths:

```bash
HTTPS_CERT_PATH=/path/to/cert.pem \
HTTPS_KEY_PATH=/path/to/key.pem \
npm run start:https
```

### Step 3: Access via HTTPS

```bash
# From another device
curl -k https://192.168.1.100:3000/health

# The -k flag allows self-signed certificates (development only)
```

**Browser access**: Navigate to `https://YOUR_IP:3000`

⚠️ **Note**: Self-signed certificates will show security warnings in browsers. This is expected for development.

---

## Dual Mode

Dual mode runs **both** stdio and HTTP/HTTPS transports simultaneously, allowing:
- Claude Desktop to connect via stdio (local)
- Other devices to connect via HTTP/HTTPS (network)

### HTTP + Stdio (Default Dual Mode)

```bash
npm run start:dual
```

This starts:
- Stdio transport for Claude Desktop
- HTTP server on port 3000 for network access

### HTTPS + Stdio

```bash
USE_HTTPS=true npm run start:dual
```

Or with custom configuration:

```bash
USE_HTTPS=true \
PORT=8443 \
HTTPS_CERT_PATH=/path/to/cert.pem \
HTTPS_KEY_PATH=/path/to/key.pem \
npm run start:dual
```

### Example Output (Dual Mode)

```
🔄 Starting Local LLM MCP Server (Dual Mode)
============================================

Mode: Stdio + HTTPS
Configuration:
  Port: 3000
  Host: 0.0.0.0
  Certificate: ./certs/cert.pem
  Key: ./certs/key.pem

This server will be available:
  - Via stdio for Claude Desktop (local)
  - Via HTTPS on port 3000 (network)

💡 Your local IP address appears to be: 192.168.1.100
   Network access: https://192.168.1.100:3000

[HTTPS] Starting MCP server on 0.0.0.0:3000...
[HTTPS] MCP Server listening on https://0.0.0.0:3000
[Dual] Also starting stdio transport for Claude Desktop...
[Dual] Both transports active: stdio + HTTPS
```

---

## Certificate Management

### Self-Signed Certificates (Development)

**Generate:**
```bash
npm run generate:certs
```

**Validity**: 365 days

**Security**:
- ⚠️ For development/testing ONLY
- Browsers will show warnings
- Not suitable for production

**Manual generation:**
```bash
mkdir -p certs
openssl req -x509 -newkey rsa:4096 \
  -keyout certs/key.pem \
  -out certs/cert.pem \
  -days 365 \
  -nodes \
  -subj "/C=US/ST=State/L=City/O=Org/CN=localhost"
```

### Trusted Certificates (Production)

For production, use certificates from a Certificate Authority:

#### Option 1: Let's Encrypt (Free)

```bash
# Install certbot
sudo apt-get install certbot  # Ubuntu/Debian
brew install certbot          # macOS

# Get certificate (requires domain name)
sudo certbot certonly --standalone -d yourdomain.com

# Certificates location:
# /etc/letsencrypt/live/yourdomain.com/fullchain.pem (cert)
# /etc/letsencrypt/live/yourdomain.com/privkey.pem (key)
```

Start server:
```bash
HTTPS_CERT_PATH=/etc/letsencrypt/live/yourdomain.com/fullchain.pem \
HTTPS_KEY_PATH=/etc/letsencrypt/live/yourdomain.com/privkey.pem \
npm run start:https
```

#### Option 2: Commercial CA

Purchase from providers like:
- DigiCert
- Sectigo
- GlobalSign

Follow provider's instructions for certificate generation and installation.

### Certificate Renewal

**Let's Encrypt** (auto-renewal):
```bash
# Test renewal
sudo certbot renew --dry-run

# Setup auto-renewal (cron)
sudo crontab -e
# Add: 0 0 * * * certbot renew --quiet
```

After renewal, restart the server to load new certificates.

---

## Production Deployment

### Security Checklist

- [ ] Use trusted CA certificates (not self-signed)
- [ ] Set strong firewall rules
- [ ] Use non-standard port (not 3000)
- [ ] Enable authentication (see REMOTE_ACCESS_PLAN.md)
- [ ] Use environment variables for sensitive data
- [ ] Enable rate limiting
- [ ] Set up monitoring/logging
- [ ] Regular certificate renewal
- [ ] Disable CORS or restrict origins

### Recommended Configuration

```bash
# .env file (never commit to git!)
PORT=8443
HOST=0.0.0.0
MCP_TRANSPORT=https
HTTPS_CERT_PATH=/etc/letsencrypt/live/yourdomain.com/fullchain.pem
HTTPS_KEY_PATH=/etc/letsencrypt/live/yourdomain.com/privkey.pem
```

Start with:
```bash
source .env && npm run start:https
```

### Firewall Configuration

**Allow HTTPS port** (example: 8443):

```bash
# Ubuntu/Debian (ufw)
sudo ufw allow 8443/tcp

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=8443/tcp
sudo firewall-cmd --reload

# macOS
# Configure via System Preferences > Security & Privacy > Firewall
```

### Reverse Proxy (Recommended)

Use nginx or Caddy for:
- Automatic HTTPS management
- Load balancing
- Additional security headers

**Example nginx config:**

```nginx
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

---

## Environment Variables Reference

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MCP_TRANSPORT` | Transport mode | `stdio` | `http`, `https`, `stdio` |
| `MCP_DUAL_MODE` | Enable dual mode | `false` | `true` |
| `MCP_HTTPS` | Force HTTPS in dual mode | `false` | `true` |
| `PORT` | Server port | `3000` | `8443` |
| `HOST` | Bind address | `0.0.0.0` | `192.168.1.100` |
| `HTTPS_CERT_PATH` | SSL certificate path | `./certs/cert.pem` | `/etc/ssl/cert.pem` |
| `HTTPS_KEY_PATH` | SSL key path | `./certs/key.pem` | `/etc/ssl/key.pem` |

---

## CLI Flags Reference

| Flag | Description | Equivalent Env Var |
|------|-------------|--------------------|
| `--http` | Start in HTTP mode | `MCP_TRANSPORT=http` |
| `--https` | Start in HTTPS mode | `MCP_TRANSPORT=https` |
| `--dual` | Enable dual mode | `MCP_DUAL_MODE=true` |

**Examples:**

```bash
# HTTP mode
node dist/index.js --http

# HTTPS mode
HTTPS_CERT_PATH=./certs/cert.pem HTTPS_KEY_PATH=./certs/key.pem \
  node dist/index.js --https

# Dual mode
node dist/index.js --dual
```

---

## Troubleshooting

### Certificate File Not Found

**Error**: `Certificate file not found: ./certs/cert.pem`

**Solution**:
```bash
# Generate certificates
npm run generate:certs

# Or specify custom path
HTTPS_CERT_PATH=/path/to/cert.pem npm run start:https
```

### Browser Shows Security Warning

**Issue**: "Your connection is not private" or similar

**Cause**: Self-signed certificate

**Solutions**:
1. **Development**: Click "Advanced" → "Proceed anyway" (varies by browser)
2. **Production**: Use trusted CA certificate (Let's Encrypt, etc.)
3. **Testing**: Use curl with `-k` flag: `curl -k https://...`

### Port Already in Use

**Error**: `EADDRINUSE: address already in use :::3000`

**Solution**:
```bash
# Stop existing server
npm run stop

# Or use different port
PORT=8443 npm run start:https
```

### Cannot Connect from Network

**Check 1**: Firewall allows the port
```bash
# macOS
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# Linux
sudo ufw status
```

**Check 2**: Server bound to correct interface
```bash
# Look for 0.0.0.0, not 127.0.0.1 in server output
[HTTPS] MCP Server listening on https://0.0.0.0:3000
```

**Check 3**: Test locally first
```bash
curl -k https://localhost:3000/health
```

### SSL Handshake Failed

**Error**: `SSL routines:ssl3_get_record:wrong version number`

**Causes**:
1. Trying to connect via HTTP to HTTPS server
2. Corrupted certificate files
3. Certificate/key mismatch

**Solutions**:
```bash
# Use correct protocol (https://)
curl https://... (not http://...)

# Regenerate certificates
rm -rf certs/
npm run generate:certs

# Verify certificate
openssl x509 -in certs/cert.pem -text -noout
```

---

## Usage Examples

### Example 1: Development with HTTPS

```bash
# Generate certs
npm run generate:certs

# Start HTTPS server
npm run start:https

# Test from another device
curl -k https://192.168.1.100:3000/health
```

### Example 2: Claude Desktop + Network Access

```bash
# Start dual mode (both transports)
npm run start:dual

# Claude Desktop connects via stdio automatically
# Network clients access via http://YOUR_IP:3000
```

### Example 3: Secure Dual Mode

```bash
# Generate certs
npm run generate:certs

# Start dual mode with HTTPS
USE_HTTPS=true npm run start:dual

# Claude Desktop: stdio
# Network clients: https://YOUR_IP:3000
```

### Example 4: Production with Let's Encrypt

```bash
# Get certificate
sudo certbot certonly --standalone -d api.yourdomain.com

# Start server
HTTPS_CERT_PATH=/etc/letsencrypt/live/api.yourdomain.com/fullchain.pem \
HTTPS_KEY_PATH=/etc/letsencrypt/live/api.yourdomain.com/privkey.pem \
PORT=443 \
npm run start:https
```

---

## Next Steps

- **Authentication**: See [REMOTE_ACCESS_PLAN.md](REMOTE_ACCESS_PLAN.md) for security implementation
- **Network Access**: See [NETWORK_USAGE.md](NETWORK_USAGE.md) for network configuration
- **Scripts**: See [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) for all available scripts
- **API**: See [API.md](API.md) for API reference

---

**Security Reminder:**
- Self-signed certificates: Development ONLY
- Production: Always use trusted CA certificates
- Keep private keys secure (never commit to git)
- Regular certificate renewal
- Monitor access logs
