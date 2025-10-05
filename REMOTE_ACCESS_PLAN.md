# Remote Access Implementation Plan

## Executive Summary

This document outlines the plan to make the Local LLM MCP Server accessible remotely over the network while maintaining backward compatibility with local stdio-based access.

## Current Architecture Analysis

### Existing Implementation
- **Transport**: stdio (Standard Input/Output)
- **Protocol**: JSON-RPC over stdio
- **Client**: Claude Desktop (local process)
- **Communication**: Parent process spawns server as child process
- **Security**: Implicit (same machine, process isolation)

### Current Flow
```
Claude Desktop (Parent Process)
    ↓ spawn child process
Local LLM MCP Server (Child Process)
    ↓ stdio communication
    ↓ JSON-RPC messages
LM Studio (localhost:1234)
```

## MCP Transport Options (2025)

### 1. **Stdio Transport** (Current - Local Only)
- ✅ Already implemented
- ✅ Zero network exposure
- ✅ Process isolation
- ❌ Cannot access remotely
- ❌ Single client only
- **Use Case**: Local Claude Desktop integration

### 2. **Streamable HTTP Transport** (New Standard - March 2025)
- ✅ Multi-client support
- ✅ Network accessible
- ✅ Supports SSE for streaming
- ✅ HTTP standard security options
- ❌ Requires server process management
- **Use Case**: Remote MCP clients, web applications

### 3. **HTTP+SSE Transport** (Legacy - Pre-2025)
- ⚠️ Being phased out
- ✅ Remote access capable
- ❌ Not recommended for new implementations
- **Status**: Deprecated as of March 26, 2025

## Security Requirements for Remote Access

### Critical Security Considerations

1. **Authentication**
   - API key/token authentication
   - JWT tokens for session management
   - OAuth2 for enterprise integrations

2. **Authorization**
   - Role-based access control (RBAC)
   - Per-tool/resource permissions
   - Model access restrictions

3. **Transport Security**
   - TLS/SSL encryption (HTTPS)
   - Certificate management
   - Secure WebSocket (WSS) for SSE

4. **Data Privacy**
   - End-to-end encryption for sensitive data
   - Request/response encryption
   - Audit logging

5. **Network Security**
   - IP allowlisting/denylisting
   - Rate limiting
   - DDoS protection
   - CORS configuration

6. **LM Studio Access**
   - Firewall rules (LM Studio should NOT be exposed)
   - Server acts as secure proxy
   - Model access control

## Proposed Architecture

### Option 1: Dual Transport Server (Recommended)

Support both stdio (local) and HTTP (remote) transports in the same codebase.

```
┌─────────────────────────────────────────────────────────┐
│                  MCP Server (Node.js)                   │
│                                                         │
│  ┌──────────────┐              ┌──────────────────┐   │
│  │ Stdio        │              │ HTTP/SSE         │   │
│  │ Transport    │              │ Transport        │   │
│  │ (Local)      │              │ (Remote)         │   │
│  └──────┬───────┘              └────────┬─────────┘   │
│         │                               │             │
│         └────────┬──────────────────────┘             │
│                  │                                     │
│         ┌────────▼──────────┐                         │
│         │   Core MCP Logic  │                         │
│         │   - Tools         │                         │
│         │   - Resources     │                         │
│         │   - Prompts       │                         │
│         └────────┬──────────┘                         │
│                  │                                     │
│         ┌────────▼──────────┐                         │
│         │  LM Studio Client │                         │
│         └────────┬──────────┘                         │
└──────────────────┼─────────────────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │   LM Studio        │
         │ (localhost:1234)   │
         └────────────────────┘

Local Client                    Remote Client
(Claude Desktop)                (Web Browser, API)
     │                               │
     │ stdio                         │ HTTPS
     ▼                               ▼
   Stdio Transport              HTTP Transport
```

**Advantages:**
- ✅ Preserves existing local functionality
- ✅ Adds remote capability
- ✅ Single codebase
- ✅ Shared business logic
- ✅ Flexible deployment

**Implementation:**
```typescript
// Detect mode from environment or CLI args
if (process.argv.includes('--http')) {
  // HTTP/SSE transport
  const httpTransport = new HttpServerTransport(config);
  await server.connect(httpTransport);
} else {
  // stdio transport (default)
  const stdioTransport = new StdioServerTransport();
  await server.connect(stdioTransport);
}
```

### Option 2: Proxy Architecture

Use existing `mcp-proxy` package to expose stdio server via HTTP/SSE.

```
Remote Client                     Local Network
     │                                 │
     │ HTTPS                           │
     ▼                                 ▼
┌──────────────┐              ┌──────────────┐
│  MCP Proxy   │◄─────────────│ MCP Server   │
│  (HTTP/SSE)  │   stdio      │   (stdio)    │
└──────────────┘              └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │  LM Studio   │
                              └──────────────┘
```

**Advantages:**
- ✅ No code changes to existing server
- ✅ Uses proven mcp-proxy package
- ✅ Quick implementation
- ✅ Separate concerns (proxy vs server)

**Disadvantages:**
- ❌ Extra component to manage
- ❌ Additional latency
- ❌ Two processes instead of one

**Implementation:**
```bash
# Install mcp-proxy
npm install -g mcp-proxy

# Run server with proxy
mcp-proxy --port 3000 \
  --command "node /path/to/local-llm-mcp-server/dist/index.js" \
  --token "your-secret-token" \
  --ssl-cert /path/to/cert.pem \
  --ssl-key /path/to/key.pem
```

### Option 3: Separate HTTP Server

Create a new HTTP-specific server that wraps the existing MCP server.

```
┌─────────────────────────────────┐
│   HTTP Server (Express/Fastify) │
│   - Authentication              │
│   - CORS                        │
│   - Rate Limiting               │
│   └──────┬──────────────────────┘
│          │
│   ┌──────▼──────────┐
│   │  MCP HTTP       │
│   │  Transport      │
│   │  Adapter        │
│   └──────┬──────────┘
│          │
│   ┌──────▼──────────┐
│   │  Existing MCP   │
│   │  Server Logic   │
│   └──────┬──────────┘
└──────────┼───────────────────────┘
           │
    ┌──────▼──────┐
    │  LM Studio  │
    └─────────────┘
```

**Advantages:**
- ✅ Full control over HTTP layer
- ✅ Custom security implementation
- ✅ Better integration with web frameworks
- ✅ Advanced features (caching, monitoring)

**Disadvantages:**
- ❌ More code to write
- ❌ More maintenance
- ❌ Duplicates some MCP SDK functionality

## Implementation Plan

### Phase 1: Dual Transport Support (Recommended)

#### 1.1 Add HTTP Transport Dependencies
```bash
npm install express cors helmet express-rate-limit jsonwebtoken
npm install --save-dev @types/express @types/cors @types/jsonwebtoken
```

#### 1.2 Create HTTP Transport Module
```typescript
// src/http-transport.ts
import express from 'express';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';

export class HttpTransportServer {
  private app: express.Application;
  private transport: SSEServerTransport;

  constructor(private config: HttpConfig) {
    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();
  }

  private setupMiddleware() {
    // CORS, helmet, rate limiting, auth
  }

  private setupRoutes() {
    // MCP SSE endpoint
    this.app.post('/mcp/sse', async (req, res) => {
      // Handle SSE connection
    });
  }

  async start() {
    this.app.listen(this.config.port);
  }
}
```

#### 1.3 Update Main Server
```typescript
// src/index.ts
async run(): Promise<void> {
  await this.lmStudio.initialize();

  if (process.env.MCP_TRANSPORT === 'http' || process.argv.includes('--http')) {
    // HTTP mode
    const httpConfig = {
      port: parseInt(process.env.PORT || '3000'),
      apiKey: process.env.API_KEY,
      enableSSL: process.env.ENABLE_SSL === 'true',
    };
    const httpServer = new HttpTransportServer(httpConfig);
    await httpServer.start();
    await this.server.connect(httpServer.getTransport());
  } else {
    // Stdio mode (default)
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
  }
}
```

#### 1.4 Add Security Layer
```typescript
// src/middleware/auth.ts
export function requireAuth(req, res, next) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token || !validateToken(token)) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
}

// src/middleware/rate-limit.ts
import rateLimit from 'express-rate-limit';

export const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
});
```

### Phase 2: Configuration & Documentation

#### 2.1 Environment Variables
```bash
# .env.example
# Transport mode
MCP_TRANSPORT=stdio  # or 'http'

# HTTP mode settings
PORT=3000
API_KEY=your-secret-key-here
ENABLE_SSL=true
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem

# CORS settings
CORS_ORIGINS=https://example.com,https://app.example.com

# Rate limiting
RATE_LIMIT_WINDOW_MS=900000  # 15 minutes
RATE_LIMIT_MAX_REQUESTS=100
```

#### 2.2 Update README.md
Add remote access documentation:
- HTTP server setup
- SSL certificate generation
- API authentication
- Client configuration examples

#### 2.3 Update API.md
Document HTTP endpoints:
- `POST /mcp/sse` - SSE connection endpoint
- Authentication headers
- Error responses

### Phase 3: Testing & Security

#### 3.1 Security Testing
- [ ] Test authentication bypass attempts
- [ ] Verify SSL/TLS configuration
- [ ] Test rate limiting
- [ ] Validate CORS policies
- [ ] Test with malformed requests

#### 3.2 Integration Testing
- [ ] Test stdio mode (existing)
- [ ] Test HTTP mode (new)
- [ ] Test concurrent clients
- [ ] Test failover scenarios

#### 3.3 Load Testing
- [ ] Benchmark HTTP vs stdio
- [ ] Test with multiple concurrent connections
- [ ] Monitor memory usage
- [ ] Test LM Studio under load

### Phase 4: Deployment Options

#### 4.1 Docker Deployment
```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY dist ./dist

ENV MCP_TRANSPORT=http
ENV PORT=3000

EXPOSE 3000
CMD ["node", "dist/index.js"]
```

#### 4.2 Reverse Proxy (nginx)
```nginx
server {
    listen 443 ssl http2;
    server_name mcp.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location /mcp/ {
        proxy_pass http://localhost:3000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

#### 4.3 Cloud Deployment
- AWS: EC2 + Application Load Balancer
- GCP: Cloud Run or GKE
- Azure: App Service or AKS

## Security Best Practices

### 1. Authentication
```typescript
// JWT-based authentication
const token = jwt.sign(
  { userId, permissions: ['read', 'write'] },
  process.env.JWT_SECRET,
  { expiresIn: '1h' }
);
```

### 2. Authorization
```typescript
// Per-tool authorization
function canAccessTool(user: User, toolName: string): boolean {
  return user.permissions.includes(`tool:${toolName}`);
}
```

### 3. Audit Logging
```typescript
// Log all requests
logger.info('MCP Request', {
  userId: req.user.id,
  tool: req.body.tool,
  timestamp: new Date(),
  ip: req.ip,
});
```

### 4. Input Validation
```typescript
// Validate all inputs
const RequestSchema = z.object({
  method: z.string(),
  params: z.object({}).passthrough(),
});
```

## Network Architecture Examples

### Home Network Setup
```
Internet
    ↓
Router (Port Forward 443 → 3000)
    ↓
MCP Server (HTTPS on port 3000)
    ↓
LM Studio (localhost:1234)
```

### Corporate Network Setup
```
Internet
    ↓
VPN Gateway
    ↓
Internal Network
    ↓
MCP Server (Internal HTTPS)
    ↓
LM Studio (localhost:1234)
```

### Cloud Deployment
```
Internet
    ↓
Cloud Load Balancer (TLS termination)
    ↓
MCP Server Container (port 3000)
    ↓
VPN/Private Network
    ↓
LM Studio (On-premise or cloud VM)
```

## Compatibility Matrix

| Transport | Local Access | Remote Access | Multi-Client | Streaming | Security |
|-----------|-------------|---------------|--------------|-----------|----------|
| stdio     | ✅ Yes      | ❌ No         | ❌ No        | ✅ Yes    | Implicit |
| HTTP/SSE  | ✅ Yes      | ✅ Yes        | ✅ Yes       | ✅ Yes    | Explicit |

## Migration Path

### For Existing Local Users
No changes needed. Server defaults to stdio mode for backward compatibility.

### For Remote Access Users
1. Set `MCP_TRANSPORT=http` environment variable
2. Configure API key authentication
3. Set up SSL certificates
4. Update firewall rules
5. Configure client with HTTP endpoint

## Performance Considerations

### Stdio (Local)
- **Latency**: ~1-5ms (IPC overhead)
- **Throughput**: Very high (same machine)
- **Overhead**: Minimal

### HTTP/SSE (Remote)
- **Latency**: ~10-100ms (network + serialization)
- **Throughput**: Depends on network
- **Overhead**: HTTP headers, TLS handshake

## Recommended Approach

**Phase 1 (Immediate)**: Implement Option 1 (Dual Transport)
- Maintains backward compatibility
- Single codebase
- Flexible deployment
- Industry-standard approach

**Phase 2 (Future)**: Add advanced features
- WebSocket transport for lower latency
- GraphQL API for complex queries
- gRPC for high-performance scenarios

## Next Steps

1. ✅ Document architecture and plan
2. ⏳ Implement HTTP transport module
3. ⏳ Add authentication middleware
4. ⏳ Create security configuration
5. ⏳ Write integration tests
6. ⏳ Update documentation
7. ⏳ Deploy and test remotely

## Questions & Decisions Needed

1. **Authentication Method**: API Key, JWT, or OAuth2?
   - **Recommendation**: Start with API Key, add JWT for sessions

2. **Default Port**: 3000, 8080, or custom?
   - **Recommendation**: 3000 (common for Node.js apps)

3. **SSL/TLS**: Required or optional?
   - **Recommendation**: Required for production, optional for development

4. **Multi-tenancy**: Support multiple users/organizations?
   - **Recommendation**: Phase 2 feature

5. **Rate Limiting**: Per IP, per user, or per API key?
   - **Recommendation**: Per API key for authenticated requests

---

**Status**: Planning Complete ✅
**Next Action**: Begin Phase 1 Implementation
