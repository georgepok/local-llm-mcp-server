# Remote Connection Quick Start

Connect Claude Desktop to this MCP server over your network in 3 steps.

**⚠️ Important:** This server now uses **Streamable HTTP (MCP 2025-03-26)** instead of the deprecated SSE transport.
**Note:** `mcp-remote` proxy may need updates to support Streamable HTTP. Check compatibility.

## Prerequisites

- Claude Desktop installed
- Node.js 18+ installed
- This MCP server built: `npm run build`
- LM Studio running with a model loaded

## Quick Start

### Step 1: Start the Server

```bash
npm run start:https
```

You should see:
```
[HTTPS] MCP Server listening on https://0.0.0.0:3010
[HTTPS] MCP endpoint: https://0.0.0.0:3010/mcp
[HTTPS] Protocol: Streamable HTTP (2025-03-26)
```

### Step 2: Add to Claude Desktop Config

**macOS:** Open `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** Open `%APPDATA%\Claude\claude_desktop_config.json`

Add this configuration:

```json
{
  "mcpServers": {
    "local-llm-remote": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://localhost:3010/mcp"
      ],
      "env": {
        "NODE_TLS_REJECT_UNAUTHORIZED": "0"
      }
    }
  }
}
```

Save the file.

### Step 3: Restart Claude Desktop

1. Quit Claude Desktop completely
2. Restart Claude Desktop
3. Check that tools are available (they'll appear in conversations)

## Verify It's Working

In Claude Desktop, ask:
```
List the available local models
```

You should get a response showing models loaded in LM Studio!

## From Another Computer

### Find Your IP Address

```bash
# macOS/Linux
ifconfig | grep "inet " | grep -v 127.0.0.1

# Windows
ipconfig
```

Example: `192.168.1.100`

### Update the Configuration

Change `localhost` to your IP:

```json
{
  "mcpServers": {
    "local-llm-remote": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-remote",
        "https://192.168.1.100:3010/mcp"
      ],
      "env": {
        "NODE_TLS_REJECT_UNAUTHORIZED": "0"
      }
    }
  }
}
```

## Troubleshooting

### "Connection refused"

**Check server is running:**
```bash
curl -k https://localhost:3010/health
```

Should return: `{"status":"ok"}`

**Check firewall:**
- macOS: System Settings > Network > Firewall > Allow Node.js
- Windows: Windows Defender > Allow Node.js through firewall

### "Cannot POST /mcp" error

**This is normal!** `mcp-remote` tries POST first, then falls back to SSE GET.

Look for this message (it means success):
```
Connected to remote server using SSEClientTransport
Proxy established successfully
```

### Tools don't appear

1. **Check server logs:**
   ```bash
   tail -f /tmp/mcp-server.log
   ```

2. **Check Claude Desktop logs:**
   - Settings > Developer > View Logs

3. **Completely restart Claude Desktop**
   - Quit from menu bar
   - Reopen

4. **Test mcp-remote directly:**
   ```bash
   NODE_TLS_REJECT_UNAUTHORIZED=0 npx -y mcp-remote https://localhost:3010/mcp
   ```
   Should show: "Proxy established successfully"

### Tools fail when called

**Check LM Studio:**
- Is LM Studio running?
- Is a model loaded?
- Is the server started? (Server tab > Start Server)

**Check LM Studio connection:**
```bash
curl http://localhost:1234/v1/models
```

Should return a list of models.

## Next Steps

- **See full documentation:** [CLAUDE_DESKTOP_REMOTE.md](docs/CLAUDE_DESKTOP_REMOTE.md)
- **Production setup:** [HTTPS_GUIDE.md](HTTPS_GUIDE.md)
- **Configuration examples:** [claude_desktop_config_examples.json](claude_desktop_config_examples.json)

## Common Use Cases

### Development on localhost
✅ Use as-is: `https://localhost:3010/mcp`

### Home network access
✅ Replace `localhost` with your IP: `https://192.168.1.100:3010/mcp`

### Production with valid SSL
✅ Use your domain: `https://your-domain.com/mcp`
⚠️ Remove `NODE_TLS_REJECT_UNAUTHORIZED=0`

## Security Notes

⚠️ **Development only:**
- `NODE_TLS_REJECT_UNAUTHORIZED=0` disables certificate validation
- Only use on trusted networks
- Never use in production

✅ **For production:**
- Get a valid SSL certificate (Let's Encrypt)
- Remove `NODE_TLS_REJECT_UNAUTHORIZED`
- Add authentication
- Use firewall rules

## Support

Having issues? Check:
1. Server logs: `tail -f /tmp/mcp-server.log`
2. Test connection: `curl -k https://localhost:3010/health`
3. Verify MCP spec compliance: `npm run test:spec`
4. Full documentation: [CLAUDE_DESKTOP_REMOTE.md](docs/CLAUDE_DESKTOP_REMOTE.md)

---

**That's it!** You should now have Claude Desktop connected to your local LLM server remotely. 🎉
