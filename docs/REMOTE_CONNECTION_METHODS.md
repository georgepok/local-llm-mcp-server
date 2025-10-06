# Remote Connection Methods Summary

Complete comparison of all methods for connecting Claude Desktop to this MCP server remotely.

## Overview

There are **three methods** to connect Claude Desktop to a remote MCP server:

1. **Custom Connector UI** - Production (requires paid plan)
2. **mcp-remote Proxy** - Development/Testing (works with any plan)
3. **Direct stdio** - Local only (no remote capability)

## Method Comparison

| Feature | Custom Connector UI | mcp-remote Proxy | Direct stdio |
|---------|-------------------|------------------|--------------|
| **Remote Access** | ✅ Yes | ✅ Yes | ❌ Local only |
| **Self-signed Certs** | ❌ No | ✅ Yes | N/A |
| **Localhost** | ❌ No | ✅ Yes | ✅ Yes |
| **Authentication** | ✅ OAuth built-in | ⚠️ Limited | N/A |
| **Plan Required** | Pro/Max/Team/Enterprise | Any plan | Any plan |
| **Setup Complexity** | Easy (UI) | Medium (JSON) | Easy (JSON) |
| **Production Ready** | ✅ Yes | ⚠️ Testing only | ✅ Yes |
| **Network Access** | ✅ Yes | ✅ Yes | ❌ No |
| **Certificate Validation** | ✅ Required | Optional | N/A |

---

## Method 1: Custom Connector UI

### When to Use

✅ **Best for:**
- Production deployments
- Public servers with valid SSL
- Claude Pro/Max/Team/Enterprise users
- Maximum security

❌ **Not suitable for:**
- Self-signed certificates
- localhost development
- Testing environments
- Free plan users

### Configuration

**Via Claude Desktop UI:**
1. Settings > Connectors
2. Add custom connector
3. Enter URL: `https://your-domain.com/sse`
4. Optional: Add OAuth credentials
5. Click Add

**No config file needed!**

### Requirements

- ✅ Valid SSL certificate (Let's Encrypt, etc.)
- ✅ Publicly accessible URL
- ✅ Pro/Max/Team/Enterprise plan
- ✅ Proper OAuth setup (if using auth)

### Example Setup

```bash
# Server with valid SSL
PORT=443 \
MCP_TRANSPORT=https \
HTTPS_CERT_PATH=/etc/letsencrypt/live/domain.com/fullchain.pem \
HTTPS_KEY_PATH=/etc/letsencrypt/live/domain.com/privkey.pem \
node dist/index.js
```

Then add in Claude Desktop UI: `https://your-domain.com/sse`

### Documentation
- [Official Guide](https://support.claude.com/en/articles/11175166)
- [Building Connectors](https://support.claude.com/en/articles/11503834)

---

## Method 2: mcp-remote Proxy

### When to Use

✅ **Best for:**
- Development and testing
- Self-signed certificates
- localhost connections
- Home network access
- Quick prototyping

❌ **Not suitable for:**
- Production environments
- Internet-facing servers
- High-security requirements

### How It Works

```
Claude Desktop (stdio) ← → mcp-remote (proxy) ← → MCP Server (HTTPS/SSE)
```

The `mcp-remote` npm package:
- Accepts stdio from Claude Desktop
- Connects to remote SSE endpoint
- Proxies all MCP messages
- Handles certificate validation

### Configuration

**File:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "local-llm-remote": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://localhost:3010/sse"
      ],
      "env": {
        "NODE_TLS_REJECT_UNAUTHORIZED": "0"
      }
    }
  }
}
```

### Variations

#### Local Network (with IP):
```json
{
  "command": "npx",
  "args": ["-y", "mcp-remote", "https://192.168.1.100:3010/sse"],
  "env": {"NODE_TLS_REJECT_UNAUTHORIZED": "0"}
}
```

#### Production with Valid Cert:
```json
{
  "command": "npx",
  "args": ["-y", "mcp-remote", "https://your-domain.com/sse"]
}
```

#### With HTTP Proxy:
```json
{
  "command": "npx",
  "args": ["-y", "mcp-remote", "https://server.com/sse", "--enable-proxy"],
  "env": {
    "HTTPS_PROXY": "http://127.0.0.1:3128",
    "NO_PROXY": "localhost,127.0.0.1"
  }
}
```

### Testing

Before adding to config, test the connection:

```bash
NODE_TLS_REJECT_UNAUTHORIZED=0 npx -y mcp-remote https://localhost:3010/sse
```

Expected output:
```
[12345] Connected to remote server using SSEClientTransport
[12345] Proxy established successfully
[12345] Press Ctrl+C to exit
```

### Documentation
- [npm Package](https://www.npmjs.com/package/mcp-remote)
- [Our Guide](CLAUDE_DESKTOP_REMOTE.md)
- [Quick Start](../REMOTE_QUICKSTART.md)
- [Examples](../claude_desktop_config_examples.json)

---

## Method 3: Direct stdio (Local Only)

### When to Use

✅ **Best for:**
- Local development
- Maximum performance
- No network complexity
- Standard Claude Desktop integration

❌ **Cannot:**
- Access from remote devices
- Connect over network
- Use from different machines

### Configuration

**File:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "local-llm": {
      "command": "node",
      "args": ["/absolute/path/to/local-llm-mcp-server/dist/index.js"],
      "env": {
        "NODE_ENV": "production"
      }
    }
  }
}
```

### No Server Startup Needed

The server starts automatically when Claude Desktop launches.

---

## Decision Matrix

### Choose Custom Connector UI if:
- ✅ You have Pro/Max/Team/Enterprise plan
- ✅ You have a valid SSL certificate
- ✅ Server is publicly accessible
- ✅ You need production-grade security
- ✅ You want simple UI configuration

### Choose mcp-remote Proxy if:
- ✅ You're developing/testing
- ✅ Using self-signed certificates
- ✅ Connecting to localhost
- ✅ On your home network
- ✅ Any Claude plan (including free)
- ✅ Need quick setup

### Choose Direct stdio if:
- ✅ Running on same machine
- ✅ No remote access needed
- ✅ Want maximum performance
- ✅ Standard local setup

---

## Security Comparison

### Custom Connector UI
- ✅ Certificate validation enforced
- ✅ OAuth authentication supported
- ✅ Production-grade security
- ✅ Anthropic-managed
- ❌ Must trust server provider

### mcp-remote Proxy
- ⚠️ Can disable cert validation
- ⚠️ Limited authentication
- ⚠️ Development use only
- ✅ Full control over connection
- ⚠️ Environment variable for unsafe TLS

### Direct stdio
- ✅ No network exposure
- ✅ Maximum security (local only)
- ✅ No certificate concerns
- ❌ No remote access

---

## Example Scenarios

### Scenario 1: Development on Laptop

**Best Method:** mcp-remote Proxy

```bash
# Terminal 1
npm run start:https

# Terminal 2 - Edit config
{
  "mcpServers": {
    "local-llm-remote": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://localhost:3010/sse"],
      "env": {"NODE_TLS_REJECT_UNAUTHORIZED": "0"}
    }
  }
}

# Restart Claude Desktop
```

### Scenario 2: Access from iPad on Home Network

**Best Method:** mcp-remote Proxy

```bash
# On server machine:
npm run start:https

# Find IP: ifconfig | grep "inet "
# Example: 192.168.1.100

# On iPad - Install Claude Desktop
# Edit config to use:
https://192.168.1.100:3010/sse
```

### Scenario 3: Production Cloud Server

**Best Method:** Custom Connector UI

```bash
# Server setup
certbot certonly --standalone -d api.example.com

PORT=443 \
MCP_TRANSPORT=https \
HTTPS_CERT_PATH=/etc/letsencrypt/live/api.example.com/fullchain.pem \
HTTPS_KEY_PATH=/etc/letsencrypt/live/api.example.com/privkey.pem \
node dist/index.js

# In Claude Desktop:
# Settings > Connectors > Add custom connector
# URL: https://api.example.com/sse
```

### Scenario 4: Same Machine, No Network

**Best Method:** Direct stdio

```bash
# Edit claude_desktop_config.json
{
  "mcpServers": {
    "local-llm": {
      "command": "node",
      "args": ["/Users/you/local-llm-mcp-server/dist/index.js"]
    }
  }
}

# Just restart Claude Desktop - done!
```

---

## Troubleshooting by Method

### Custom Connector UI Issues

**"Connection failed"**
- Check server is publicly accessible
- Verify SSL certificate is valid
- Test with curl: `curl https://your-domain.com/sse`

**"Certificate error"**
- Cannot use self-signed certs
- Use Let's Encrypt or other CA
- Verify cert hasn't expired

### mcp-remote Proxy Issues

**"Cannot POST /sse"**
- Normal! Proxy tries POST first, then GET
- Look for "Connected to remote server using SSEClientTransport"

**"Connection refused"**
- Server not running: `npm run start:https`
- Wrong port or IP
- Firewall blocking

**"Certificate error" (even with NODE_TLS_REJECT_UNAUTHORIZED)**
- Cert variable not in correct env section
- Try testing with command line first
- Check config JSON syntax

### Direct stdio Issues

**"Command not found"**
- Wrong path to dist/index.js
- Use absolute path, not relative
- Check file exists: `ls -la /path/to/dist/index.js`

**Server won't start**
- Check Claude Desktop logs: Settings > Developer
- LM Studio not running
- Port conflict

---

## Migration Paths

### From stdio to mcp-remote Proxy
```json
// Before (stdio)
{
  "local-llm": {
    "command": "node",
    "args": ["/path/to/dist/index.js"]
  }
}

// After (mcp-remote)
{
  "local-llm-stdio": {
    "command": "node",
    "args": ["/path/to/dist/index.js"]
  },
  "local-llm-remote": {
    "command": "npx",
    "args": ["-y", "mcp-remote", "https://localhost:3010/sse"],
    "env": {"NODE_TLS_REJECT_UNAUTHORIZED": "0"}
  }
}
```

### From mcp-remote to Custom Connector UI
1. Get valid SSL certificate
2. Deploy to public server
3. Remove from claude_desktop_config.json
4. Add via Settings > Connectors UI
5. Remove NODE_TLS_REJECT_UNAUTHORIZED

---

## Quick Reference

| Scenario | Method | Config Location | Server Command |
|----------|--------|----------------|----------------|
| Local dev | stdio | config.json | Auto-starts |
| Localhost testing | mcp-remote | config.json | `npm run start:https` |
| Home network | mcp-remote | config.json | `npm run start:https` |
| Production | UI Connector | UI Settings | `node dist/index.js` (with valid cert) |

---

## Documentation Index

- **Quick Start:** [REMOTE_QUICKSTART.md](../REMOTE_QUICKSTART.md)
- **Complete Guide:** [CLAUDE_DESKTOP_REMOTE.md](CLAUDE_DESKTOP_REMOTE.md)
- **Config Examples:** [claude_desktop_config_examples.json](../claude_desktop_config_examples.json)
- **HTTPS Setup:** [HTTPS_GUIDE.md](../HTTPS_GUIDE.md)
- **Network Usage:** [NETWORK_USAGE.md](../NETWORK_USAGE.md)
- **Spec Validation:** [MCP_SPEC_VALIDATION.md](MCP_SPEC_VALIDATION.md)

---

**Summary:** For development/testing, use **mcp-remote proxy**. For production, use **Custom Connector UI**. For local-only, use **direct stdio**.
