# Testing MCP Server with curl

Guide to testing the MCP server HTTPS transport using curl.

## Quick Test Scripts

We provide ready-to-use test scripts:

```bash
# Simple health checks
./test-mcp-simple.sh 3010

# Full MCP protocol test
./test-mcp-curl.sh 3010
```

## Manual curl Commands

### 1. Health Check
```bash
curl -k https://localhost:3010/health
```

**Response:**
```json
{
  "status": "ok",
  "transport": "http",
  "timestamp": "2025-10-05T23:10:53.355Z"
}
```

### 2. Server Info
```bash
curl -k https://localhost:3010/
```

**Response:**
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

### 3. SSE Connection (Get Session ID)
```bash
curl -k -N https://localhost:3010/sse
```

**Response:**
```
event: endpoint
data: /message?sessionId=c2b0762f-80fd-4974-a03b-7e1452ceb6d2
```

Extract the session ID: `c2b0762f-80fd-4974-a03b-7e1452ceb6d2`

### 4. Send MCP Messages

#### Initialize
```bash
SESSION_ID="c2b0762f-80fd-4974-a03b-7e1452ceb6d2"

curl -k -X POST "https://localhost:3010/message?sessionId=$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": {
        "name": "curl-client",
        "version": "1.0.0"
      }
    }
  }'
```

**Response:** `Accepted` (actual response comes via SSE)

#### List Tools
```bash
curl -k -X POST "https://localhost:3010/message?sessionId=$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list",
    "params": {}
  }'
```

#### List Resources
```bash
curl -k -X POST "https://localhost:3010/message?sessionId=$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "resources/list",
    "params": {}
  }'
```

#### Read Resource
```bash
curl -k -X POST "https://localhost:3010/message?sessionId=$SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "resources/read",
    "params": {
      "uri": "local://models"
    }
  }'
```

## Understanding the Flow

### MCP over SSE/HTTP Protocol:

1. **Client connects to `/sse`**
   - Server creates SSE connection
   - Server sends session ID via SSE event
   - Connection stays open for responses

2. **Client sends messages to `/message?sessionId=xxx`**
   - POST JSON-RPC messages
   - Server routes to correct MCP session
   - Immediate HTTP response: "Accepted"

3. **Server sends responses via SSE**
   - JSON-RPC responses stream over SSE
   - Client must listen to SSE stream for results

### Why "Accepted" Response?

The `/message` endpoint returns "Accepted" because:
- MCP uses bidirectional communication
- Request is queued for processing
- Actual response comes via SSE stream
- This is normal MCP SSE behavior

## Server Logs Confirm Success

When you run the curl test, check `/tmp/mcp-server.log`:

```
[HTTP] GET /sse
[HTTP] Client connecting via SSE
[HTTP] Client connected via SSE (session: c2b0762f-80fd-4974-a03b-7e1452ceb6d2)
[HTTP] POST /message
[HTTP] Received message: {"jsonrpc":"2.0","id":1,"method":"initialize"...
[HTTP] POST /message
[HTTP] Received message: {"jsonrpc":"2.0","id":2,"method":"tools/list"...
```

This confirms:
✅ SSE connection established
✅ Session created
✅ Messages received
✅ MCP protocol working

## Full Example Session

```bash
# 1. Start server
PORT=3010 MCP_TRANSPORT=https \
  HTTPS_CERT_PATH=./certs/cert.pem \
  HTTPS_KEY_PATH=./certs/key.pem \
  node dist/index.js > /tmp/mcp-server.log 2>&1 &

# 2. Test health
curl -k https://localhost:3010/health

# 3. Get session (in one terminal)
curl -k -N https://localhost:3010/sse
# Note the sessionId from output

# 4. Send messages (in another terminal)
SESSION="your-session-id-here"
curl -k -X POST "https://localhost:3010/message?sessionId=$SESSION" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}'

# 5. Check server logs
tail -f /tmp/mcp-server.log
```

## Automated Testing

Use the provided scripts for automated testing:

```bash
# Full protocol test
./test-mcp-curl.sh 3010

# Simple endpoint test
./test-mcp-simple.sh 3010
```

Both scripts:
- Start SSE connection
- Extract session ID
- Send MCP messages
- Show server is working

## Notes

- `-k` flag: Allows self-signed certificates
- `-N` flag: Disables buffering for SSE
- Responses come via SSE, not HTTP response body
- Keep SSE connection open while sending messages
- Session IDs are unique per connection

## Production Use

For production:
1. Use trusted certificates (remove `-k`)
2. Add authentication
3. Use proper MCP client library (not curl)
4. Handle SSE reconnection
5. Parse SSE events properly

The curl examples are for testing and demonstration only.
