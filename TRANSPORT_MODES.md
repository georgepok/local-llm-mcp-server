# Transport Modes - Quick Reference

The Local LLM MCP Server supports four transport modes for maximum flexibility.

## Transport Modes

| Mode | Protocol | Use Case | Command |
|------|----------|----------|---------|
| **Stdio** | Standard I/O | Claude Desktop (local) | `npm start` or `npm run start:local` |
| **HTTP** | HTTP/SSE | Network access (unencrypted) | `npm run start:remote` |
| **HTTPS** | HTTPS/SSE | Network access (encrypted) | `npm run start:https` |
| **Dual** | Stdio + HTTP/HTTPS | Both local and remote | `npm run start:dual` |

---

## Quick Start Examples

### 1. Local Only (Claude Desktop)
```bash
npm run start:local
```
**Access**: Claude Desktop only (stdio)

### 2. Network Access (HTTP)
```bash
npm run start:remote
```
**Access**: `http://YOUR_IP:3000`

### 3. Secure Network Access (HTTPS)
```bash
# First time: generate certificates
npm run generate:certs

# Start HTTPS server
npm run start:https
```
**Access**: `https://YOUR_IP:3000`

### 4. Both Local and Remote (Dual Mode)
```bash
npm run start:dual
```
**Access**:
- Claude Desktop (stdio)
- Network clients at `http://YOUR_IP:3000`

### 5. Secure Dual Mode (HTTPS + Stdio)
```bash
# Generate certs if needed
npm run generate:certs

# Start dual mode with HTTPS
USE_HTTPS=true npm run start:dual
```
**Access**:
- Claude Desktop (stdio)
- Network clients at `https://YOUR_IP:3000`

---

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `MCP_TRANSPORT` | `stdio`, `http`, `https` | Transport mode |
| `MCP_DUAL_MODE` | `true`, `false` | Enable dual mode |
| `MCP_HTTPS` | `true`, `false` | Force HTTPS in dual mode |
| `PORT` | Number | Server port (default: 3000) |
| `HOST` | IP address | Bind address (default: 0.0.0.0) |
| `HTTPS_CERT_PATH` | File path | SSL certificate |
| `HTTPS_KEY_PATH` | File path | SSL private key |

---

## CLI Flags

| Flag | Description |
|------|-------------|
| `--http` | Start in HTTP mode |
| `--https` | Start in HTTPS mode |
| `--dual` | Enable dual mode |

**Examples:**
```bash
node dist/index.js --http
node dist/index.js --https
node dist/index.js --dual
```

---

## When to Use Each Mode

### Stdio Mode
✅ Use when:
- Running with Claude Desktop
- Local-only access needed
- No network access required

❌ Don't use when:
- Need to access from other devices
- Want network API access

### HTTP Mode
✅ Use when:
- Accessing from local network
- Development/testing
- Trusted network environment

❌ Don't use when:
- Sensitive data transmission
- Public internet exposure
- Production deployment

### HTTPS Mode
✅ Use when:
- Secure network access needed
- Production deployment
- Sensitive data transmission
- Public internet access

❌ Don't use when:
- Only local access needed (use stdio)
- Can't manage certificates

### Dual Mode
✅ Use when:
- Need both Claude Desktop AND network access
- Want single server instance
- Development with multiple clients

❌ Don't use when:
- Only one access method needed
- Want separate configs for each

---

## Configuration Examples

### Development: Dual Mode with HTTPS
```bash
# Generate self-signed certs (one time)
npm run generate:certs

# Start dual mode
USE_HTTPS=true npm run start:dual
```

**Result:**
- Claude Desktop connects via stdio
- Other devices access via `https://YOUR_IP:3000`
- Both transports active simultaneously

### Production: HTTPS Only
```bash
# Use Let's Encrypt certificates
HTTPS_CERT_PATH=/etc/letsencrypt/live/yourdomain.com/fullchain.pem \
HTTPS_KEY_PATH=/etc/letsencrypt/live/yourdomain.com/privkey.pem \
PORT=443 \
npm run start:https
```

**Result:**
- Secure network access only
- Trusted certificates
- No browser warnings

### Testing: HTTP for Quick Network Access
```bash
PORT=3000 npm run start:remote
```

**Result:**
- Fast setup, no certificates needed
- Local network access
- Good for quick testing

---

## Port Configuration

| Mode | Default Port | Recommended Production Port |
|------|--------------|----------------------------|
| Stdio | N/A | N/A |
| HTTP | 3000 | 8080 or custom |
| HTTPS | 3000 | 443 (standard HTTPS) |
| Dual | 3000 | 443 for HTTPS, stdio uses no port |

**Custom port examples:**
```bash
# HTTP on port 8080
PORT=8080 npm run start:remote

# HTTPS on standard port 443 (requires sudo)
sudo PORT=443 npm run start:https

# Dual mode on custom port
PORT=8443 npm run start:dual
```

---

## Troubleshooting

### "Port already in use"
```bash
# Stop existing server
npm run stop

# Or use different port
PORT=3001 npm run start:remote
```

### "Certificate file not found" (HTTPS)
```bash
# Generate certificates
npm run generate:certs

# Or specify custom path
HTTPS_CERT_PATH=/path/to/cert.pem npm run start:https
```

### "Cannot connect from network"
```bash
# Check server is bound to 0.0.0.0 (not 127.0.0.1)
# Server output should show:
# [HTTP] MCP Server listening on http://0.0.0.0:3000

# Check firewall allows the port
sudo ufw allow 3000/tcp  # Linux
# macOS: System Preferences > Security & Privacy > Firewall
```

### Browser shows "Connection not secure" (HTTPS)
This is normal for self-signed certificates in development.

**Development**: Click "Advanced" → "Proceed"

**Production**: Use trusted CA certificate (Let's Encrypt, etc.)

---

## Documentation Links

- **[HTTPS_GUIDE.md](HTTPS_GUIDE.md)** - Complete HTTPS and dual mode guide
- **[SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)** - All available scripts
- **[NETWORK_USAGE.md](NETWORK_USAGE.md)** - Network configuration
- **[README.md](README.md)** - Main documentation

---

## Summary

**Most Common Scenarios:**

1. **Claude Desktop only**: `npm run start:local`
2. **Network testing**: `npm run start:remote`
3. **Secure network**: `npm run generate:certs && npm run start:https`
4. **Both local + remote**: `npm run start:dual`
5. **Production**: Use HTTPS with trusted certificates

Choose the mode that fits your use case!
