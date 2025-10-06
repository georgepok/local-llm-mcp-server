# Migration to Streamable HTTP

Guide for migrating from SSE transport to Streamable HTTP transport (MCP 2025-03-26).

## Summary of Changes

### What Changed

**SSE Transport (Deprecated)**
- Protocol: `2024-11-05`
- Endpoint: `/sse` (GET) + `/message` (POST)
- Session management: Query parameter `?sessionId=xxx`
- Responses: SSE events

**Streamable HTTP (Current)**
- Protocol: `2025-03-26`
- Endpoint: `/mcp` (unified for GET/POST/DELETE)
- Session management: Header `Mcp-Session-Id`
- Responses: SSE events (same format)
- Better spec compliance and future-proofing

### Why This Change?

1. **Official Recommendation**: SSE transport deprecated in MCP spec 2025-03-26
2. **Better Protocol**: Streamable HTTP is the official replacement
3. **Improved Features**:
   - Session resumability support
   - Better error handling
   - Cleaner API design
   - Support for DELETE requests (session termination)

## Migration Steps

### For Server Operators

**No action required** - The server automatically uses the new protocol!

Old behavior:
```bash
# Started with /sse and /message endpoints
npm run start:https
```

New behavior:
```bash
# Starts with /mcp endpoint (all methods)
npm run start:https
```

### For Client Developers

#### Old Client Code (SSE Transport)

```typescript
// 1. Connect to SSE
const sseResponse = await fetch('https://localhost:3010/sse');
const sessionId = extractFromSSE(sseResponse);

// 2. Send message
await fetch(`https://localhost:3010/message?sessionId=${sessionId}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(message)
});
```

#### New Client Code (Streamable HTTP)

```typescript
// 1. Send request with protocol version
const response = await fetch('https://localhost:3010/mcp', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json, text/event-stream',
    'Mcp-Protocol-Version': '2025-03-26'
  },
  body: JSON.stringify(initMessage)
});

// 2. Extract session from header
const sessionId = response.headers.get('mcp-session-id');

// 3. Use session in subsequent requests
await fetch('https://localhost:3010/mcp', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json, text/event-stream',
    'Mcp-Protocol-Version': '2025-03-26',
    'Mcp-Session-Id': sessionId  // Header instead of query param
  },
  body: JSON.stringify(nextMessage)
});
```

### Required Header Changes

| Feature | Old (SSE) | New (Streamable HTTP) |
|---------|-----------|----------------------|
| Content Type | `application/json` | `application/json` |
| Accept | Optional | `application/json, text/event-stream` (required) |
| Protocol Version | Not used | `Mcp-Protocol-Version: 2025-03-26` (required) |
| Session ID | Query param `?sessionId=xxx` | Header `Mcp-Session-Id: xxx` |

### Endpoint Changes

| Operation | Old (SSE) | New (Streamable HTTP) |
|-----------|-----------|----------------------|
| Initialize | POST `/message?sessionId=new` | POST `/mcp` |
| Send Message | POST `/message?sessionId=xxx` | POST `/mcp` with `Mcp-Session-Id` header |
| Get SSE Stream | GET `/sse` | GET `/mcp` with `Accept: text/event-stream` |
| Close Session | Not supported | DELETE `/mcp` with `Mcp-Session-Id` header |

## Testing

### Test Old SSE Transport (No longer works)

```bash
# This will fail - endpoint removed
curl -k https://localhost:3010/sse
# Error: 404 Not Found
```

### Test New Streamable HTTP

```bash
# Run automated test
npm run test:streamable

# Manual test
curl -k -X POST "https://localhost:3010/mcp" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Protocol-Version: 2025-03-26" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}'
```

## Response Format

**Both transports use the same SSE response format:**

```
event: message
data: {"jsonrpc":"2.0","result":{...},"id":1}

```

Parsing is identical:
```javascript
const lines = sseData.split('\n');
for (const line of lines) {
  if (line.startsWith('data: ')) {
    const jsonData = line.substring(6);
    const response = JSON.parse(jsonData);
    // Handle response
  }
}
```

## Backwards Compatibility

### Client Compatibility

**Old clients using SSE transport will NOT work** with the new server.

Reason: The `/sse` and `/message` endpoints have been removed.

**Migration required** for all clients to use `/mcp` endpoint.

### Protocol Version Negotiation

The server supports protocol version `2025-03-26` only.

Clients must send:
```
Mcp-Protocol-Version: 2025-03-26
```

## Error Handling

### New Error Responses

**Missing Accept Header**
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32000,
    "message": "Not Acceptable: Client must accept both application/json and text/event-stream"
  },
  "id": null
}
```

**Missing Protocol Version**
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32000,
    "message": "Protocol version mismatch"
  },
  "id": null
}
```

**Invalid Session ID**
```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32000,
    "message": "Invalid session ID"
  },
  "id": null
}
```

## Updated Documentation

### Files Updated

- ✅ `src/http-transport.ts` - New StreamableHTTPServerTransport
- ✅ `test-streamable-http.js` - Test suite for new protocol
- ✅ `package.json` - Added `test:streamable` script
- ⚠️ Deprecated test files (SSE-based):
  - `test-sse-responses.js` - No longer works
  - `validate-mcp-spec.js` - Tests old SSE spec
  - `test-mcp-curl.sh` - Uses old endpoints
  - `test-mcp-simple.sh` - Uses old endpoints

### Files to Update (Manual Action Required)

- `docs/CURL_TESTING.md` - Update curl examples for `/mcp` endpoint
- `docs/MCP_SPEC_VALIDATION.md` - Update for 2025-03-26 protocol
- `docs/CLAUDE_DESKTOP_REMOTE.md` - Update `mcp-remote` usage
- `claude_desktop_config_examples.json` - Update URL examples
- `REMOTE_QUICKSTART.md` - Update endpoints

## Migration Checklist

### For This Server

- [x] Replace SSEServerTransport with StreamableHTTPServerTransport
- [x] Change endpoint from `/sse` + `/message` to `/mcp`
- [x] Update session management (query param → header)
- [x] Add protocol version handling
- [x] Create test suite for Streamable HTTP
- [ ] Update all documentation
- [ ] Deprecate old test files
- [ ] Update curl examples
- [ ] Update client connection guides

### For Client Developers

- [ ] Update endpoint URLs to `/mcp`
- [ ] Add required headers (`Accept`, `Mcp-Protocol-Version`)
- [ ] Change session ID from query param to header
- [ ] Update SSE parsing (format unchanged)
- [ ] Test with new server
- [ ] Update error handling for new error codes

## Rollback Plan

**There is no rollback** - SSE transport is deprecated by MCP specification.

If you need the old behavior temporarily:
1. Check out the commit before this migration
2. Build and deploy that version
3. Plan migration as soon as possible

**Do not use SSE transport in production** - it is no longer maintained by the MCP specification.

## Support

### Getting Help

**If you encounter issues:**

1. Check logs: `tail -f /tmp/mcp-server.log`
2. Test health: `curl -k https://localhost:3010/health`
3. Run tests: `npm run test:streamable`
4. Review protocol: [MCP Specification 2025-03-26](https://modelcontextprotocol.io)

### Common Issues

**"Not Acceptable" error**
- Add `Accept: application/json, text/event-stream` header

**"Protocol version mismatch"**
- Add `Mcp-Protocol-Version: 2025-03-26` header

**"404 Not Found" on /sse**
- Endpoint removed - use `/mcp` instead

**mcp-remote not working**
- `mcp-remote` may need updates for Streamable HTTP
- Check for latest version: `npm info mcp-remote`

## Timeline

- **Before**: SSE Transport (Protocol 2024-11-05)
- **Now**: Streamable HTTP (Protocol 2025-03-26)
- **Future**: Continued support for Streamable HTTP only

## Summary

**Key Takeaways:**

1. ✅ Server updated to Streamable HTTP (2025-03-26)
2. ❌ SSE Transport removed (deprecated by spec)
3. 🔄 All clients must migrate to new endpoints/headers
4. ✅ Same SSE response format (parsing unchanged)
5. 📚 Documentation updates in progress

**Next Steps:**

1. Test with new client code
2. Update any custom integrations
3. Review updated documentation
4. Report issues if encountered

---

**Protocol Evolution:**
- `2024-11-05`: SSE Transport introduced
- `2025-03-26`: **Streamable HTTP replaces SSE** ← We are here
- Future: Streamable HTTP is the standard going forward
