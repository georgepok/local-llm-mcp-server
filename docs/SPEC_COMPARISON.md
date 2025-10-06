# MCP SSE Transport: Specification vs Implementation

Side-by-side comparison of the official MCP SDK specification and our implementation.

## 1. SSE Connection Establishment

### Specification (from SDK)
```javascript
// node_modules/@modelcontextprotocol/sdk/dist/esm/server/sse.js:52-76
async start() {
  if (this._sseResponse) {
    throw new Error("SSEServerTransport already started!");
  }

  this.res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache, no-transform",
    Connection: "keep-alive",
  });

  // Send the endpoint event
  const endpointUrl = new URL(this._endpoint, dummyBase);
  endpointUrl.searchParams.set('sessionId', this._sessionId);
  const relativeUrlWithSession = endpointUrl.pathname + endpointUrl.search + endpointUrl.hash;

  this.res.write(`event: endpoint\ndata: ${relativeUrlWithSession}\n\n`);

  this._sseResponse = this.res;

  this.res.on("close", () => {
    this._sseResponse = undefined;
    this.onclose?.call(this);
  });
}
```

### Our Implementation
```typescript
// src/http-transport.ts:70-89
this.app.get('/sse', async (req, res) => {
  console.log('[HTTP] Client connecting via SSE');

  // Create SSE transport for this client
  const transport = new SSEServerTransport('/message', res);

  // Store transport by session ID
  this.transports.set(transport.sessionId, transport);

  // Connect the MCP server to this transport
  // NOTE: connect() calls start() automatically (per SDK line 54)
  await this.mcpServer.connect(transport);

  console.log(`[HTTP] Client connected via SSE (session: ${transport.sessionId})`);

  // Clean up on disconnect
  transport.onclose = () => {
    console.log(`[HTTP] Client disconnected from SSE (session: ${transport.sessionId})`);
    this.transports.delete(transport.sessionId);
  };
});
```

**Analysis:** ✅ **COMPLIANT**
- We use the SDK's `SSEServerTransport` class directly
- `mcpServer.connect()` calls `transport.start()` automatically
- All headers and event format handled by SDK
- Session cleanup properly implemented

---

## 2. Message POST Handling

### Specification (from SDK)
```javascript
// node_modules/@modelcontextprotocol/sdk/dist/esm/server/sse.js:82-122
async handlePostMessage(req, res, parsedBody) {
  if (!this._sseResponse) {
    const message = "SSE connection not established";
    res.writeHead(500).end(message);
    throw new Error(message);
  }

  // Validate request headers for DNS rebinding protection
  const validationError = this.validateRequestHeaders(req);
  if (validationError) {
    res.writeHead(403).end(validationError);
    this.onerror?.call(this, new Error(validationError));
    return;
  }

  let body;
  try {
    const ct = contentType.parse(req.headers["content-type"] ?? "");
    if (ct.type !== "application/json") {
      throw new Error(`Unsupported content-type: ${ct.type}`);
    }

    body = parsedBody ?? await getRawBody(req, {
      limit: MAXIMUM_MESSAGE_SIZE,
      encoding: ct.parameters.charset ?? "utf-8",
    });
  } catch (error) {
    res.writeHead(400).end(String(error));
    this.onerror?.call(this, error);
    return;
  }

  try {
    await this.handleMessage(
      typeof body === 'string' ? JSON.parse(body) : body,
      { requestInfo, authInfo }
    );
  } catch (_e) {
    res.writeHead(400).end(`Invalid message: ${body}`);
    return;
  }

  res.writeHead(202).end("Accepted");
}
```

### Our Implementation
```typescript
// src/http-transport.ts:92-141
this.app.post('/message', async (req, res) => {
  try {
    console.log('[HTTP] Received message:', JSON.stringify(req.body).substring(0, 100));

    // Extract session ID from query parameter
    const sessionId = req.query.sessionId as string;

    if (!sessionId) {
      res.status(400).json({
        jsonrpc: '2.0',
        id: null,
        error: {
          code: -32600,
          message: 'Invalid Request: sessionId required',
        },
      });
      return;
    }

    // Find the transport for this session
    const transport = this.transports.get(sessionId);

    if (!transport) {
      res.status(404).json({
        jsonrpc: '2.0',
        id: null,
        error: {
          code: -32001,
          message: 'Session not found',
        },
      });
      return;
    }

    // Forward the message to the transport
    // SDK handles all validation and sends "202 Accepted"
    await transport.handlePostMessage(req, res, req.body);

  } catch (error) {
    console.error('[HTTP] Error handling message:', error);
    res.status(500).json({
      jsonrpc: '2.0',
      id: req.body?.id || null,
      error: {
        code: -32603,
        message: 'Internal error',
        data: error instanceof Error ? error.message : 'Unknown error',
      },
    });
  }
});
```

**Analysis:** ✅ **COMPLIANT**
- We delegate all validation to SDK's `handlePostMessage()`
- SDK handles Content-Type validation
- SDK handles JSON parsing and validation
- SDK returns HTTP 202 with "Accepted" body
- We only add session routing logic (not part of SDK)

---

## 3. Sending Responses via SSE

### Specification (from SDK)
```javascript
// node_modules/@modelcontextprotocol/sdk/dist/esm/server/sse.js:144-149
async send(message) {
  if (!this._sseResponse) {
    throw new Error("Not connected");
  }

  this._sseResponse.write(`event: message\ndata: ${JSON.stringify(message)}\n\n`);
}
```

### Our Implementation
```typescript
// We don't implement send() - the SDK handles it!
// The MCP Server class calls transport.send() automatically
// when generating responses to client requests.

// src/http-transport.ts:29-36
constructor(
  private config: HttpTransportConfig,
  mcpServer: Server
) {
  this.mcpServer = mcpServer;
  this.app = express();
  this.setupMiddleware();
  this.setupRoutes();
}
```

**Analysis:** ✅ **COMPLIANT**
- SDK's `SSEServerTransport.send()` handles all response formatting
- MCP `Server` class calls `transport.send()` when needed
- We don't need to implement anything - it's automatic!

---

## 4. Session Management

### Specification (from SDK)
```javascript
// node_modules/@modelcontextprotocol/sdk/dist/esm/server/sse.js:16-21
constructor(_endpoint, res, options) {
  this._endpoint = _endpoint;
  this.res = res;
  this._sessionId = randomUUID();  // Generate unique session ID
  this._options = options || { enableDnsRebindingProtection: false };
}

// node_modules/@modelcontextprotocol/sdk/dist/esm/server/sse.js:155-158
get sessionId() {
  return this._sessionId;
}
```

### Our Implementation
```typescript
// src/http-transport.ts:27
private transports: Map<string, SSEServerTransport> = new Map();

// src/http-transport.ts:74-77
const transport = new SSEServerTransport('/message', res);
this.transports.set(transport.sessionId, transport);

// src/http-transport.ts:85-88
transport.onclose = () => {
  console.log(`[HTTP] Client disconnected from SSE (session: ${transport.sessionId})`);
  this.transports.delete(transport.sessionId);
};

// src/http-transport.ts:112
const transport = this.transports.get(sessionId);
```

**Analysis:** ✅ **COMPLIANT**
- SDK generates UUID session IDs automatically
- We store transports in a Map for routing
- We clean up sessions on disconnect
- Session routing is our addition (required for multi-client support)

---

## 5. Protocol Flow

### Specification Flow
```
1. Client: GET /sse
2. Server: 200 OK + SSE headers
3. Server: event: endpoint\ndata: /message?sessionId=xxx\n\n
4. Client: POST /message?sessionId=xxx + JSON-RPC
5. Server: 202 Accepted + "Accepted"
6. Server: event: message\ndata: {JSON-RPC response}\n\n
```

### Our Implementation Flow
```
1. Client: GET /sse
2. HttpTransport: new SSEServerTransport('/message', res)
3. HttpTransport: await mcpServer.connect(transport)
   ↓
   SDK calls transport.start()
   ↓
   SDK sends 200 + headers + endpoint event
4. Client: POST /message?sessionId=xxx
5. HttpTransport: Find transport by sessionId
6. HttpTransport: await transport.handlePostMessage(req, res, body)
   ↓
   SDK validates, parses, routes to MCP server
   ↓
   SDK returns 202 Accepted
   ↓
   MCP server processes request
   ↓
   MCP server calls transport.send(response)
   ↓
   SDK sends: event: message\ndata: {...}\n\n
```

**Analysis:** ✅ **COMPLIANT**
- Exact match with SDK specification
- All protocol details handled by SDK
- We only add session routing layer

---

## Key Findings

### What the SDK Handles Automatically

1. ✅ SSE connection headers (Content-Type, Cache-Control, Connection)
2. ✅ Endpoint event generation with session ID
3. ✅ UUID session ID generation
4. ✅ POST request validation (Content-Type, JSON parsing)
5. ✅ HTTP 202 "Accepted" response
6. ✅ SSE message event formatting
7. ✅ JSON-RPC message validation
8. ✅ Connection lifecycle management

### What We Implement

1. 📋 Express.js HTTP server setup
2. 📋 Multiple session routing (Map-based)
3. 📋 Session cleanup on disconnect
4. 📋 HTTPS support (certificates)
5. 📋 CORS configuration
6. 📋 Request logging
7. 📋 Error handling for missing sessions

### Specification Compliance Matrix

| Feature | Required by Spec | Implemented | How |
|---------|------------------|-------------|-----|
| SSE Headers | ✅ | ✅ | SDK automatic |
| Endpoint Event | ✅ | ✅ | SDK automatic |
| UUID Session ID | ✅ | ✅ | SDK automatic |
| HTTP 202 Response | ✅ | ✅ | SDK automatic |
| SSE Message Events | ✅ | ✅ | SDK automatic |
| JSON-RPC Format | ✅ | ✅ | SDK automatic |
| Session Routing | ❌ (not in spec) | ✅ | Our addition |
| Multi-Client | ❌ (not in spec) | ✅ | Our addition |
| HTTPS Support | ❌ (not in spec) | ✅ | Our addition |

**Legend:**
- ✅ = Required and implemented
- ❌ = Not required (optional enhancement)

---

## Conclusion

Our implementation achieves **100% specification compliance** by:

1. Using the official `SSEServerTransport` from `@modelcontextprotocol/sdk`
2. Letting the SDK handle all protocol-level details
3. Only adding necessary routing logic for multi-client support
4. Following the exact flow specified in SDK source code

**No changes needed** - the implementation is fully compliant with the MCP SSE Transport Specification.

---

**References:**
- SDK Source: `node_modules/@modelcontextprotocol/sdk/dist/esm/server/sse.js`
- Our Implementation: `src/http-transport.ts`
- Protocol Version: `2024-11-05`
