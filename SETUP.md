# Setup Instructions for Claude Desktop

This guide will help you set up the Local LLM MCP Server to work with Claude Desktop on your machine.

## Prerequisites ✅

1. **LM Studio** - Make sure LM Studio is installed and running on port 1234
2. **Node.js 18+** - Required for running the MCP server
3. **Claude Desktop** - The host application that will use the MCP server

## Step 1: Verify LM Studio Setup

1. Open LM Studio
2. Load a model (e.g., Llama 3.1, Code Llama, etc.)
3. Go to the "Local Server" tab
4. Start the server (should show it's running on `http://localhost:1234`)
5. Test that it's working by visiting: `http://localhost:1234/v1/models`

## Step 2: Build the MCP Server

```bash
cd /Users/george/Documents/GitHub/local-llm-mcp-server

# Install dependencies (if not already done)
npm install

# Build the project
npm run build

# Test the server (optional)
node test-server.js
```

## Step 3: Configure Claude Desktop

### Find your Claude Desktop config file:

**macOS:**
```bash
open ~/Library/Application\ Support/Claude/
```

**Linux:**
```bash
mkdir -p ~/.config/claude
```

**Windows:**
```bash
%APPDATA%\Claude\
```

### Create or edit `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "local-llm": {
      "command": "node",
      "args": ["/Users/george/Documents/GitHub/local-llm-mcp-server/dist/index.js"],
      "env": {
        "NODE_ENV": "production"
      }
    }
  }
}
```

**Important:** Make sure the path in `args` matches your actual installation directory.

## Step 4: Restart Claude Desktop

1. Completely quit Claude Desktop
2. Restart Claude Desktop
3. Look for MCP server connection indicators in the interface

## Step 5: Test the Integration

Once Claude Desktop is running with the MCP server, you should be able to use commands like:

### Test Local Reasoning
```
Use the local_reasoning tool to analyze this business scenario locally: "A company is considering expanding into international markets but has limited resources."
```

### Test Privacy Analysis
```
Use the private_analysis tool to perform sentiment analysis on this text: "I'm excited about the new product launch but concerned about the timeline."
```

### Test Secure Rewriting
```
Use the secure_rewrite tool to rewrite this email in a professional tone while protecting privacy: "Hey John, can you check the customer data for account #12345? The customer at john.smith@email.com called about billing issues."
```

## Troubleshooting

### MCP Server Not Connecting

1. **Check the path:** Ensure the path in `claude_desktop_config.json` is correct
2. **Check permissions:** Make sure the MCP server files are executable
3. **Check LM Studio:** Verify LM Studio is running and accessible at `localhost:1234`
4. **Check logs:** Look at Claude Desktop logs for error messages

### Test MCP Server Manually

```bash
# Test the server directly
cd /Users/george/Documents/GitHub/local-llm-mcp-server
node dist/index.js
```

The server should start and wait for JSON-RPC messages on stdin.

### LM Studio Connection Issues

```bash
# Test LM Studio API directly
curl http://localhost:1234/v1/models

# Should return a list of available models
```

### Common Configuration Issues

1. **Wrong path in config:** Double-check the absolute path to `dist/index.js`
2. **Node.js version:** Ensure you're using Node.js 18 or higher
3. **Missing dependencies:** Run `npm install` again if needed
4. **Build issues:** Run `npm run build` to ensure TypeScript compilation succeeded

## Configuration Options

### Custom Configuration
Create a `config.json` file in the project root to customize:

```json
{
  "lmStudio": {
    "baseUrl": "http://localhost:1234/v1",
    "defaultModel": "your-model-name",
    "timeout": 30000
  },
  "privacy": {
    "defaultLevel": "moderate"
  }
}
```

### Environment Variables

You can also set configuration via environment variables:

```json
{
  "mcpServers": {
    "local-llm": {
      "command": "node",
      "args": ["/Users/george/Documents/GitHub/local-llm-mcp-server/dist/index.js"],
      "env": {
        "NODE_ENV": "production",
        "LLM_SERVER_PORT": "1234",
        "LLM_SERVER_HOST": "localhost",
        "PRIVACY_LEVEL": "moderate"
      }
    }
  }
}
```

## Verification

### Check MCP Server Status
Use the resource endpoint to verify everything is working:

```
Read the local://status resource to check if the MCP server and LM Studio are properly connected.
```

### Check Available Models
```
Read the local://models resource to see what models are available in LM Studio.
```

### Check Server Configuration
```
Read the local://config resource to see the current server capabilities and settings.
```

## Next Steps

Once everything is working:

1. **Explore the tools:** Try different analysis types and privacy levels
2. **Use prompt templates:** Explore pre-built templates for common tasks
3. **Customize settings:** Adjust the configuration for your specific needs
4. **Review privacy settings:** Ensure privacy levels meet your requirements

## Support

If you encounter issues:

1. Check the logs in Claude Desktop
2. Verify LM Studio is running and accessible
3. Test the MCP server manually using the test script
4. Ensure all paths and permissions are correct

The Local LLM MCP Server provides powerful privacy-preserving AI capabilities by keeping your sensitive data local while still benefiting from advanced language model processing.