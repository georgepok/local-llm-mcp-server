# Remote Network Access Guide

This guide shows you how to access the Local LLM MCP Server from other devices on your network.

## Overview

The server supports two transport modes:

1. **Stdio Mode** (Default) - Local access only, used by Claude Desktop
2. **HTTP Mode** (New) - Network accessible, for remote clients

## Quick Start - HTTP Mode

### 1. Start Server in HTTP Mode

```bash
# Default: localhost:3000
MCP_TRANSPORT=http npm start

# Or specify custom port/host
PORT=8080 HOST=0.0.0.0 MCP_TRANSPORT=http npm start
```

### 2. Access from Network

The server will be accessible at:
```
http://YOUR_MACHINE_IP:3000
```

**Find your IP address:**
```bash
# macOS/Linux
ifconfig | grep "inet " | grep -v 127.0.0.1

# Example output: 192.168.1.100
```

### 3. Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Server information |
| `/health` | GET | Health check |
| `/sse` | GET | Server-Sent Events stream |
| `/message` | POST | Send JSON-RPC messages |

## Usage Examples

### Health Check

```bash
curl http://192.168.1.100:3000/health
```

Response:
```json
{
  "status": "ok",
  "transport": "http",
  "timestamp": "2025-10-05T16:03:56.804Z"
}
```

### Server Info

```bash
curl http://192.168.1.100:3000/
```

Response:
```json
{
  "server": "local-llm-mcp-server",
  "version": "1.0.0",
  "transport": "http",
  "endpoints": {
    "health": "/health",
    "sse": "/sse",
    "message": "/message"
  }
}
```

### SSE Connection (Server-Sent Events)

```bash
curl -N http://192.168.1.100:3000/sse
```

Receives:
```
data: {"type":"connected"}
```

## Configuration

### Environment Variables

```bash
# Transport mode
MCP_TRANSPORT=http          # Enable HTTP mode (default: stdio)

# Network settings
PORT=3000                    # Port to listen on (default: 3000)
HOST=0.0.0.0                # Host to bind to (default: 0.0.0.0)
                            # 0.0.0.0 = all interfaces
                            # 127.0.0.1 = localhost only
```

### Command Line

```bash
# Using --http flag
node dist/index.js --http

# Or environment variable
MCP_TRANSPORT=http node dist/index.js
```

## Network Scenarios

### Scenario 1: Access from Another Computer

**Server Machine (192.168.1.100):**
```bash
PORT=3000 MCP_TRANSPORT=http npm start
```

**Client Machine:**
```bash
curl http://192.168.1.100:3000/health
```

### Scenario 2: Access from Mobile Device

**Server:**
```bash
MCP_TRANSPORT=http npm start
```

**Mobile Browser:**
```
http://192.168.1.100:3000
```

### Scenario 3: Different Port (Avoid Conflicts)

```bash
PORT=8080 MCP_TRANSPORT=http npm start
```

Access at: `http://192.168.1.100:8080`

## Firewall Configuration

### macOS

Allow incoming connections on port 3000:

```bash
# Check if firewall is enabled
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# Add node to firewall exceptions (one-time)
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/node
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/local/bin/node
```

### Router Port Forwarding (Optional)

If you want to access from outside your local network:

1. Log into your router admin panel
2. Find "Port Forwarding" or "Virtual Server" settings
3. Forward external port → Internal IP:3000
4. Access via your public IP

**⚠️ Security Warning:** This exposes your server to the internet. Since there's no authentication in this simplified version, only do this on a trusted network or add authentication.

## Testing Your Setup

### 1. Test Local Access

```bash
curl http://localhost:3000/health
```

Expected: `{"status":"ok",...}`

### 2. Test Network Access

From another device:
```bash
curl http://YOUR_SERVER_IP:3000/health
```

### 3. Test SSE Stream

```bash
curl -N http://YOUR_SERVER_IP:3000/sse
```

Should receive: `data: {"type":"connected"}`

## Troubleshooting

### Server Won't Start

**Error: `Address already in use`**
```bash
# Check what's using the port
lsof -i :3000

# Use a different port
PORT=3001 MCP_TRANSPORT=http npm start
```

### Can't Connect from Network

**Check 1: Server is listening on all interfaces**
```bash
# Server output should show:
# [HTTP] MCP Server listening on http://0.0.0.0:3000
#                                        ^^^^^^^^
#                                    (not 127.0.0.1)
```

**Check 2: Firewall is not blocking**
```bash
# Temporarily disable macOS firewall to test
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off

# Re-enable after testing
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
```

**Check 3: Correct IP address**
```bash
# Make sure you're using the server's local network IP, not:
# - 127.0.0.1 (localhost only)
# - Public IP (unless port forwarded)
```

### Connection Timeouts

**Check network connectivity:**
```bash
# From client, ping the server
ping 192.168.1.100

# If ping fails, check:
# - Same network/subnet
# - WiFi isolation disabled (common on guest networks)
# - VPN not interfering
```

## Preserving Local Claude Desktop Access

**Important:** The default mode is still stdio for local Claude Desktop.

```bash
# Local Claude Desktop (stdio mode - default)
npm start

# Remote access (HTTP mode - explicit)
MCP_TRANSPORT=http npm start
```

You can run both simultaneously on the same machine:
- Claude Desktop: Uses stdio mode (default when started by Claude)
- Remote clients: Start another instance with HTTP mode on different port

## Example: Running Both Modes

**Terminal 1 - For Claude Desktop:**
```bash
# This runs in stdio mode automatically when Claude starts it
# No manual start needed
```

**Terminal 2 - For Network Access:**
```bash
PORT=3001 MCP_TRANSPORT=http npm start
```

Now you have:
- Claude Desktop → stdio (automatic)
- Network access → http://localhost:3001

## Network Performance

**Expected latency:**
- Local network: 1-10ms
- Same WiFi: 5-20ms
- Ethernet: 1-5ms

**Factors affecting performance:**
- Network congestion
- WiFi signal strength
- Number of active clients
- LM Studio model size and speed

## Security Considerations

**Current Implementation:** No authentication (suitable for home networks)

**Recommendations for home network:**
- ✅ Keep server on private network only
- ✅ Don't port forward to internet
- ✅ Use firewall to restrict access
- ✅ Monitor connected clients

**For production use:** See `REMOTE_ACCESS_PLAN.md` for authentication implementation.

## Client Examples

### JavaScript/Node.js

```javascript
// Health check
const response = await fetch('http://192.168.1.100:3000/health');
const data = await response.json();
console.log(data);

// SSE connection
const eventSource = new EventSource('http://192.168.1.100:3000/sse');
eventSource.onmessage = (event) => {
  console.log('Received:', JSON.parse(event.data));
};
```

### Python

```python
import requests

# Health check
response = requests.get('http://192.168.1.100:3000/health')
print(response.json())

# SSE stream
import sseclient
response = requests.get('http://192.168.1.100:3000/sse', stream=True)
client = sseclient.SSEClient(response)
for event in client.events():
    print(event.data)
```

### cURL

```bash
# Simple GET
curl http://192.168.1.100:3000/health

# POST JSON-RPC message
curl -X POST http://192.168.1.100:3000/message \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {}
  }'
```

## Advanced Configuration

### Custom Host Binding

```bash
# Listen only on specific interface
HOST=192.168.1.100 MCP_TRANSPORT=http npm start

# Listen on all IPv4 interfaces
HOST=0.0.0.0 MCP_TRANSPORT=http npm start

# Localhost only (no network access)
HOST=127.0.0.1 MCP_TRANSPORT=http npm start
```

### Multiple Ports

Run multiple instances for different purposes:

```bash
# Instance 1: Public access
PORT=3000 MCP_TRANSPORT=http npm start

# Instance 2: Admin access (different terminal)
PORT=3001 MCP_TRANSPORT=http npm start
```

## Monitoring

### Check Active Connections

Server logs will show:
```
[HTTP] Client connected via SSE
[HTTP] Client disconnected from SSE
```

### View Server Status

```bash
curl http://localhost:3000/health
```

## Next Steps

- **Add Authentication**: See `REMOTE_ACCESS_PLAN.md` for secure implementation
- **Add HTTPS**: For encrypted communication
- **Add Rate Limiting**: Prevent abuse
- **Add Monitoring**: Track usage and performance

---

**Quick Reference:**

```bash
# Start HTTP mode
MCP_TRANSPORT=http npm start

# Custom port
PORT=8080 MCP_TRANSPORT=http npm start

# Test from network
curl http://YOUR_IP:3000/health
```
