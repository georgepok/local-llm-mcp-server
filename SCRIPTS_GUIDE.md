# Server Management Scripts

Convenient scripts for starting and stopping the Local LLM MCP Server.

## Quick Reference

```bash
# Start server in LOCAL mode (stdio - for Claude Desktop)
npm run start:local
# or
./scripts/start-local.sh

# Start server in REMOTE mode (HTTP - for network access)
npm run start:remote
# or
./scripts/start-remote.sh

# Stop all running server instances
npm run stop
# or
./scripts/stop.sh

# Restart in remote mode
npm run restart:remote
# or
./scripts/restart-remote.sh
```

## Scripts Overview

### `start-local.sh` - Local Mode (stdio)

**Purpose**: Start the server in stdio mode for local Claude Desktop access.

**Usage**:
```bash
./scripts/start-local.sh
# or
npm run start:local
```

**Behavior**:
- Runs server in stdio transport mode (default)
- Used by Claude Desktop for local IPC communication
- Automatically builds if `dist/` directory doesn't exist
- Outputs connection status to console

**When to use**:
- Running server for Claude Desktop
- Local-only access needed
- Testing stdio transport

---

### `start-remote.sh` - Remote Mode (HTTP/SSE)

**Purpose**: Start the server in HTTP mode for network access from other devices.

**Usage**:
```bash
./scripts/start-remote.sh

# Custom port
PORT=8080 ./scripts/start-remote.sh

# Custom host and port
PORT=8080 HOST=192.168.1.100 ./scripts/start-remote.sh

# Using npm
npm run start:remote
```

**Environment Variables**:
- `PORT` - Port to listen on (default: 3000)
- `HOST` - Host to bind to (default: 0.0.0.0)

**Behavior**:
- Runs server in HTTP transport mode
- Binds to all network interfaces (0.0.0.0)
- Automatically detects and displays your local IP address
- Shows access URLs for network clients
- Automatically builds if needed

**When to use**:
- Accessing server from other devices on your network
- Mobile device access
- Testing remote connections
- Multiple client connections

**Example Output**:
```
🌐 Starting Local LLM MCP Server (Remote Mode - HTTP/SSE)
=========================================================

Server will be accessible on your network at:
  http://YOUR_IP:3000

Configuration:
  Port: 3000
  Host: 0.0.0.0

To use a different port: PORT=8080 ./scripts/start-remote.sh
To stop: Press Ctrl+C or run ./scripts/stop.sh

💡 Your local IP address appears to be: 192.168.1.100
   Access from network: http://192.168.1.100:3000

[HTTP] MCP Server listening on http://0.0.0.0:3000
MCP server ready for remote connections
```

---

### `stop.sh` - Stop Server

**Purpose**: Stop all running MCP server instances.

**Usage**:
```bash
./scripts/stop.sh
# or
npm run stop
```

**Behavior**:
- Finds all running `node dist/index.js` processes
- Sends SIGINT (graceful shutdown) to each process
- Falls back to SIGKILL if SIGINT fails
- Reports which processes were stopped

**Example Output**:
```
🛑 Stopping Local LLM MCP Server...
====================================

Found running server process(es):
  PID: 12345

Stopping server(s)...
  ✓ Stopped PID: 12345

✅ Server stopped successfully
```

---

### `restart-remote.sh` - Restart Remote Server

**Purpose**: Stop and restart the server in HTTP mode (convenience script).

**Usage**:
```bash
./scripts/restart-remote.sh

# With custom port
PORT=8080 ./scripts/restart-remote.sh

# Using npm
npm run restart:remote
```

**Behavior**:
- Calls `stop.sh` to stop running instances
- Waits 1 second
- Calls `start-remote.sh` to start in HTTP mode

**When to use**:
- Reloading configuration changes
- Applying code updates
- Switching from local to remote mode

---

## Common Scenarios

### Scenario 1: Development - Local Testing

```bash
# Terminal 1: Run tests
npm run test:regression

# Terminal 2: Start local server for Claude Desktop
npm run start:local
```

### Scenario 2: Network Access - Multiple Devices

```bash
# Start server for network access
npm run start:remote

# Access from another device
curl http://192.168.1.100:3000/health
```

### Scenario 3: Port Conflict

```bash
# If port 3000 is in use
PORT=8080 npm run start:remote

# Or use the script directly
PORT=8080 ./scripts/start-remote.sh
```

### Scenario 4: Both Local and Remote Access

```bash
# Terminal 1: Claude Desktop (stdio mode) - started automatically by Claude

# Terminal 2: Network access (HTTP mode) on different port
PORT=3001 npm run start:remote
```

### Scenario 5: Quick Restart After Code Changes

```bash
# Build and restart
npm run build && npm run restart:remote
```

---

## Troubleshooting

### Script Permission Denied

If you get "Permission denied" error:

```bash
# Make scripts executable
chmod +x scripts/*.sh
```

### Port Already in Use

**Error**: `Address already in use`

**Solution**:
```bash
# Option 1: Stop the server using the port
npm run stop

# Option 2: Use a different port
PORT=3001 npm run start:remote

# Option 3: Find and kill the process manually
lsof -i :3000
kill <PID>
```

### Server Not Accessible from Network

**Check 1**: Verify server is listening on 0.0.0.0, not 127.0.0.1
```bash
# Look for this in output:
# [HTTP] MCP Server listening on http://0.0.0.0:3000
#                                        ^^^^^^^^ (should be 0.0.0.0)
```

**Check 2**: Find your correct local IP
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**Check 3**: Test locally first
```bash
curl http://localhost:3000/health
```

**Check 4**: Check firewall settings (see NETWORK_USAGE.md)

### Script Not Found

If npm commands don't work:

```bash
# Use scripts directly
./scripts/start-remote.sh

# Or run from project root
cd /path/to/local-llm-mcp-server
./scripts/start-remote.sh
```

---

## Integration with npm

All scripts are available as npm commands for convenience:

| npm Command | Script | Mode |
|-------------|--------|------|
| `npm start` | Direct node execution | Default (stdio) |
| `npm run start:local` | `./scripts/start-local.sh` | Local (stdio) |
| `npm run start:remote` | `./scripts/start-remote.sh` | Remote (HTTP) |
| `npm run stop` | `./scripts/stop.sh` | Stop all |
| `npm run restart:remote` | `./scripts/restart-remote.sh` | Restart HTTP |
| `npm run build` | TypeScript compilation | Build only |
| `npm run test:regression` | `node test-regression.js` | Run tests |

---

## Script Locations

```
scripts/
├── start-local.sh      # Start in stdio mode (Claude Desktop)
├── start-remote.sh     # Start in HTTP mode (network access)
├── stop.sh             # Stop all servers
└── restart-remote.sh   # Restart in HTTP mode
```

All scripts are executable and can be run directly:
```bash
./scripts/start-local.sh
./scripts/start-remote.sh
./scripts/stop.sh
./scripts/restart-remote.sh
```

---

## Notes

- **Claude Desktop**: When running via Claude Desktop's config, the server automatically starts in stdio mode. No manual start needed.

- **Multiple Instances**: You can run both local (stdio) and remote (HTTP) modes simultaneously on different ports.

- **Auto-build**: All start scripts automatically build if `dist/` doesn't exist.

- **Graceful Shutdown**: Press Ctrl+C or use `npm run stop` for clean shutdown with proper cleanup.

- **Environment Variables**: All scripts support standard environment variables (`PORT`, `HOST`, `MCP_TRANSPORT`).

---

## See Also

- [README.md](README.md) - Main documentation
- [NETWORK_USAGE.md](NETWORK_USAGE.md) - Network access guide
- [API.md](API.md) - API reference
- [EXAMPLES.md](EXAMPLES.md) - Usage examples
