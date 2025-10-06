# Connecting Claude Desktop to Remote MCP Server

Guide to connecting Claude Desktop to this MCP server over the network using custom connectors.

## Overview

Claude Desktop supports two methods for connecting to remote MCP servers:

1. **Custom Connector UI** (Recommended for production)
2. **mcp-remote Proxy** (Works with self-signed certificates)

## Method 1: Custom Connector UI (Production)

### Prerequisites
- Claude Desktop with Pro, Max, Team, or Enterprise plan
- Trusted SSL certificate (not self-signed)
- Publicly accessible server URL

### Setup Steps

1. **Navigate to Settings**
   - Open Claude Desktop
   - Go to `Settings > Connectors`

2. **Add Custom Connector**
   - Click "Add custom connector" at the bottom
   - Enter your server URL: `https://your-server.com/sse`
   - Click "Add"

3. **Advanced Settings** (Optional)
   - Click "Advanced settings" to add:
     - OAuth Client ID (if using authentication)
     - OAuth Client Secret (if using authentication)

### Important Notes

- ⚠️ **Self-signed certificates are NOT supported** in the UI
- ✅ Only use trusted MCP servers
- ✅ Review permissions before connecting
- ⚠️ SSE transport may be deprecated in future (use Streamable HTTP when available)

### Authentication

For authenticated servers:
- OAuth callback URL: `https://claude.ai/api/mcp/auth_callback`
- Supports Dynamic Client Registration (DCR)
- Can specify custom client ID/secret

---

## Method 2: mcp-remote Proxy (Development/Testing)

**Use this method when:**
- Using self-signed certificates (development)
- Testing on localhost
- Server not publicly accessible
- Need to work around certificate validation

### How It Works

```
Claude Desktop (stdio) ← → mcp-remote (proxy) ← → Your HTTPS Server (SSE)
```

The `mcp-remote` package acts as a bridge:
- Exposes stdio interface to Claude Desktop
- Connects to remote SSE server
- Handles certificate validation
- Proxies all MCP messages

### Installation

No installation needed! Use `npx` to run directly.

### Configuration

#### For localhost HTTPS with self-signed cert:

**File:** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

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

#### For remote server with valid certificate:

```json
{
  "mcpServers": {
    "local-llm-remote": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://192.168.1.100:3010/sse"
      ]
    }
  }
}
```

#### With HTTP proxy:

```json
{
  "mcpServers": {
    "local-llm-remote": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://your-server.com/sse",
        "--enable-proxy"
      ],
      "env": {
        "HTTPS_PROXY": "http://127.0.0.1:3128",
        "NO_PROXY": "localhost,127.0.0.1"
      }
    }
  }
}
```

### Testing Connection

Before adding to Claude Desktop config, test the connection:

```bash
# With self-signed cert
NODE_TLS_REJECT_UNAUTHORIZED=0 npx -y mcp-remote https://localhost:3010/sse

# With valid cert
npx -y mcp-remote https://your-server.com/sse
```

**Expected output:**
```
[12345] Connecting to remote server: https://localhost:3010/sse
[12345] Using transport strategy: http-first
[12345] Received error: Error POSTing to endpoint (HTTP 404)
[12345] Recursively reconnecting for reason: falling-back-to-alternate-transport
[12345] Using transport strategy: sse-only
[12345] Connected to remote server using SSEClientTransport
[12345] Local STDIO server running
[12345] Proxy established successfully
[12345] Press Ctrl+C to exit
```

The "HTTP 404" error is expected - `mcp-remote` tries POST first, then falls back to SSE GET (which succeeds).

### Restart Claude Desktop

After modifying `claude_desktop_config.json`:
1. Quit Claude Desktop completely
2. Restart Claude Desktop
3. The server should appear in the MCP tools list

### Verify Connection

In Claude Desktop, check:
1. Settings > Developer > View Logs
2. Look for connection messages
3. Try using a tool (e.g., "list available local models")

---

## Comparison

| Feature | Custom Connector UI | mcp-remote Proxy |
|---------|-------------------|------------------|
| Setup | Easy (UI-based) | Medium (JSON config) |
| Self-signed certs | ❌ Not supported | ✅ Supported |
| Authentication | ✅ OAuth built-in | ⚠️ Limited |
| Localhost | ❌ Not supported | ✅ Supported |
| Production use | ✅ Recommended | ⚠️ Testing only |
| Plan required | Pro/Max/Team/Enterprise | Any |

---

## Server Setup

### Start Server for Remote Access

#### Local network (HTTP):
```bash
npm run start:remote
# Server at http://0.0.0.0:3000/sse
```

#### Local network (HTTPS with self-signed cert):
```bash
npm run start:https
# Server at https://0.0.0.0:3010/sse
```

#### Find your local IP:
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
# Use this IP: http://192.168.1.100:3000/sse
```

#### Production (HTTPS with valid cert):
```bash
PORT=3010 \
MCP_TRANSPORT=https \
HTTPS_CERT_PATH=/path/to/cert.pem \
HTTPS_KEY_PATH=/path/to/key.pem \
node dist/index.js
```

---

## Troubleshooting

### Connection fails with "SSL: CERTIFICATE_VERIFY_FAILED"

**Problem:** Self-signed certificate rejected

**Solution:** Use `mcp-remote` proxy with `NODE_TLS_REJECT_UNAUTHORIZED=0`:
```json
{
  "env": {
    "NODE_TLS_REJECT_UNAUTHORIZED": "0"
  }
}
```

### mcp-remote shows "Cannot POST /sse"

**This is normal!** The proxy tries POST first (for Streamable HTTP), then falls back to GET (for SSE). Look for:
```
Recursively reconnecting for reason: falling-back-to-alternate-transport
Connected to remote server using SSEClientTransport
```

### Claude Desktop doesn't show tools

1. Check server logs: `tail -f /tmp/mcp-server.log`
2. Verify server is running: `curl -k https://localhost:3010/health`
3. Test with mcp-remote directly (see Testing Connection above)
4. Restart Claude Desktop completely
5. Check Claude Desktop logs: Settings > Developer > View Logs

### "Connection refused" error

- Ensure server is running
- Check firewall allows connections
- Verify port is correct
- Test with curl first: `curl -k https://localhost:3010/sse`

### Tools appear but calls fail

- Check LM Studio is running
- Verify models are loaded in LM Studio
- Check server logs for errors

---

## Security Considerations

### Development/Testing (mcp-remote)

✅ **Safe for:**
- Local development
- Testing on localhost
- Home network testing

⚠️ **Not recommended for:**
- Production deployments
- Untrusted networks
- Internet-facing servers

### Production (Custom Connector UI)

✅ **Requirements:**
- Trusted SSL certificate (from CA like Let's Encrypt)
- Secure server configuration
- Authentication enabled
- Regular security updates

⚠️ **Never:**
- Use self-signed certs in production
- Expose without authentication
- Connect to untrusted servers
- Disable certificate validation in production

---

## Advanced Configuration

### Tool Filtering

Filter out specific tools with `mcp-remote`:

```json
{
  "command": "npx",
  "args": [
    "-y",
    "mcp-remote",
    "https://localhost:3010/sse",
    "--ignore-tool",
    "tool_pattern*"
  ]
}
```

### Multiple Remote Servers

You can connect to multiple servers:

```json
{
  "mcpServers": {
    "local-llm-remote": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://localhost:3010/sse"]
    },
    "other-server": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://other-server.com/sse"]
    }
  }
}
```

### Debugging

Enable verbose logging:

```bash
# Test with debug output
DEBUG=* npx -y mcp-remote https://localhost:3010/sse
```

---

## Example Workflows

### Scenario 1: Development on Localhost

```bash
# 1. Start HTTPS server
npm run start:https

# 2. Add to claude_desktop_config.json:
{
  "mcpServers": {
    "local-llm-dev": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://localhost:3010/sse"],
      "env": {"NODE_TLS_REJECT_UNAUTHORIZED": "0"}
    }
  }
}

# 3. Restart Claude Desktop
# 4. Use tools in conversations
```

### Scenario 2: Production Deployment

```bash
# 1. Get SSL certificate from Let's Encrypt
certbot certonly --standalone -d your-domain.com

# 2. Start server with valid cert
PORT=443 \
MCP_TRANSPORT=https \
HTTPS_CERT_PATH=/etc/letsencrypt/live/your-domain.com/fullchain.pem \
HTTPS_KEY_PATH=/etc/letsencrypt/live/your-domain.com/privkey.pem \
node dist/index.js

# 3. In Claude Desktop UI:
#    Settings > Connectors > Add custom connector
#    URL: https://your-domain.com/sse
```

### Scenario 3: Home Network Access

```bash
# 1. Find your local IP
ifconfig | grep "inet " | grep -v 127.0.0.1
# Example: 192.168.1.100

# 2. Start server
npm run start:https

# 3. From another device, add to config:
{
  "mcpServers": {
    "local-llm-network": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://192.168.1.100:3010/sse"],
      "env": {"NODE_TLS_REJECT_UNAUTHORIZED": "0"}
    }
  }
}
```

---

## References

- **mcp-remote package:** https://www.npmjs.com/package/mcp-remote
- **Claude custom connectors:** https://support.claude.com/en/articles/11175166
- **MCP specification:** https://modelcontextprotocol.io
- **This server's documentation:** See [HTTPS_GUIDE.md](HTTPS_GUIDE.md)

---

## Summary

**For Development:**
Use `mcp-remote` proxy with `claude_desktop_config.json` - supports self-signed certs and localhost.

**For Production:**
Use Custom Connector UI in Claude Desktop settings - requires valid SSL certificate.

**Quick Start (Development):**
```bash
# 1. Start server
npm run start:https

# 2. Add to config
{
  "local-llm-remote": {
    "command": "npx",
    "args": ["-y", "mcp-remote", "https://localhost:3010/sse"],
    "env": {"NODE_TLS_REJECT_UNAUTHORIZED": "0"}
  }
}

# 3. Restart Claude Desktop
```

That's it! 🚀
