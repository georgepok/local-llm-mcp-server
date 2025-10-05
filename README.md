# Local LLM MCP Server

A Model Context Protocol (MCP) server that bridges local LLMs running in LM Studio with Claude Desktop and other MCP clients. Keep your sensitive data private by running AI tasks locally while seamlessly integrating with cloud-based AI assistants.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js Version](https://img.shields.io/badge/node-%3E%3D18.0.0-brightgreen)](https://nodejs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue)](https://www.typescriptlang.org/)

## 🌟 Features

### 🔒 Privacy-First Design
- **Local Processing**: All sensitive data stays on your machine
- **No Cloud Exposure**: Private analysis, code review, and content processing happens locally
- **Privacy Levels**: Configurable privacy protection (strict, moderate, minimal)
- **No Telemetry**: Zero usage tracking or data collection

### 🤖 Dynamic Multi-Model Support
- **Auto-Discovery**: Automatically detects all models loaded in LM Studio
- **Flexible Selection**: Use different models for different tasks
- **Runtime Switching**: Change default models during your session
- **Per-Request Override**: Specify model for individual requests
- **Smart Initialization**: First available model auto-selected as default

### 🛠️ Comprehensive Tool Suite

**Local Reasoning** - General-purpose AI tasks with complete privacy
- Complex problem solving and multi-step reasoning
- Question answering and task planning
- Context-aware responses

**Private Analysis** - 7 analysis types for sensitive content
- Sentiment Analysis (domain-aware)
- Entity Extraction (people, orgs, locations, domain-specific)
- Content Classification
- Summarization with key points
- Privacy Scanning (PII, GDPR compliance)
- Security Auditing (vulnerabilities, misconfigurations)

**Secure Rewriting** - Transform text while maintaining privacy
- Style adaptation (formal, casual, professional)
- Sensitive information removal
- Privacy-preserving transformations

**Code Analysis** - Local code review and security
- Security vulnerability detection
- Code quality assessment
- Bug detection and optimization suggestions

**Template Completion** - Intelligent form and document filling

### 🎯 Domain-Specific Intelligence
Specialized analysis for:
- **Medical**: Healthcare context, HIPAA compliance, clinical terminology
- **Legal**: Legal terminology, regulatory compliance, confidentiality
- **Financial**: Financial regulations, market analysis, data protection
- **Technical**: Software development, engineering contexts
- **Academic**: Scholarly research, methodology, citations

## 🚀 Quick Start

### Prerequisites

1. **Node.js 18+**
   ```bash
   node --version  # Should be >= 18.0.0
   ```

2. **LM Studio**
   - Download from [lmstudio.ai](https://lmstudio.ai)
   - Load at least one model (e.g., Llama 3.2, Qwen, Mistral)
   - Start the local server (Server tab → Start Server)
   - Default URL: `http://localhost:1234`

### Installation

```bash
git clone https://github.com/yourusername/local-llm-mcp-server.git
cd local-llm-mcp-server
npm install
npm run build
```

### Configure Claude Desktop

Edit your Claude Desktop config file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add this configuration:
```json
{
  "mcpServers": {
    "local-llm": {
      "command": "node",
      "args": ["/absolute/path/to/local-llm-mcp-server/dist/index.js"]
    }
  }
}
```

**Important**: Use the absolute path to your installation.

### Start Using

1. **Restart Claude Desktop** - The server starts automatically
2. **Discover Models** - Read resource `local://models` to see available models
3. **Try It** - Ask Claude to use the `local_reasoning` tool with a simple prompt

The server automatically:
- Discovers all models loaded in LM Studio
- Sets the first model as default
- Provides full capability documentation via `local://capabilities`

### Convenience Scripts

For easier server management, use the included scripts:

```bash
# Start in local mode (stdio - for Claude Desktop)
npm run start:local

# Start in remote mode (HTTP - for network access)
npm run start:remote

# Stop all running servers
npm run stop

# Restart in remote mode
npm run restart:remote
```

See [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) for detailed usage.

## 🌐 Remote Network Access

Access the server from other devices on your home network!

```bash
# Start in HTTP mode for network access
MCP_TRANSPORT=http npm start

# Or use CLI flag
npm start -- --http

# Access from any device on your network
curl http://YOUR_MACHINE_IP:3000/health
```

**Quick Start:**
```bash
# Find your IP address
ifconfig | grep "inet " | grep -v 127.0.0.1

# Start server (default port 3000)
MCP_TRANSPORT=http npm start

# Access from browser or API client
http://192.168.1.100:3000
```

**Available endpoints:**
- `/` - Server information
- `/health` - Health check
- `/sse` - Server-Sent Events stream
- `/message` - JSON-RPC messages

**See [NETWORK_USAGE.md](NETWORK_USAGE.md) for complete guide** including:
- Firewall configuration
- Client examples (JavaScript, Python, cURL)
- Troubleshooting
- Multiple device scenarios

**Dual Mode Support:**
- **Local Mode** (default): stdio transport for Claude Desktop
- **Remote Mode**: HTTP/SSE transport for network access
- Both modes can run simultaneously on different ports!

## 📋 Available Tools

### Core Tools

#### `local_reasoning`
Use the local LLM for specialized reasoning tasks while keeping data private.

```typescript
// Example usage in Claude
await tools.local_reasoning({
  prompt: "Analyze the logical flow of this argument...",
  system_prompt: "You are a critical thinking expert",
  model_params: {
    temperature: 0.7,
    max_tokens: 1500
  }
});
```

#### `private_analysis`
Analyze sensitive content locally without cloud exposure.

```typescript
await tools.private_analysis({
  content: "Confidential business document...",
  analysis_type: "sentiment", // or "entities", "classification", etc.
  domain: "financial"
});
```

#### `secure_rewrite`
Rewrite or transform text locally for privacy.

```typescript
await tools.secure_rewrite({
  content: "Original text with sensitive info...",
  style: "professional",
  privacy_level: "strict"
});
```

#### `code_analysis`
Analyze code locally for security, quality, or documentation.

```typescript
await tools.code_analysis({
  code: "function processUserData(input) { ... }",
  language: "javascript",
  analysis_focus: "security"
});
```

#### `template_completion`
Complete templates or forms using the local LLM.

```typescript
await tools.template_completion({
  template: "Dear [NAME], Thank you for [ACTION]...",
  context: "Customer submitted a support ticket about billing",
  format: "email"
});
```

## 📚 Resources

### Available Resources

- `local://models` - List of available models in LM Studio
- `local://status` - Current status of the local LLM server
- `local://config` - Server configuration and capabilities

### Example Resource Usage

```typescript
// Get available models
const models = await resources.read("local://models");

// Check server status
const status = await resources.read("local://status");

// View configuration
const config = await resources.read("local://config");
```

## 🎯 Prompt Templates

### Using Pre-built Templates

```typescript
// Privacy analysis template
await prompts.get("privacy-analysis", {
  content: "Document to analyze...",
  regulation: "GDPR"
});

// Code security review template
await prompts.get("code-security-review", {
  code: "const user = req.body.user;",
  language: "javascript",
  security_focus: "injection vulnerabilities"
});

// Meeting summary template
await prompts.get("meeting-summary", {
  meeting_content: "Meeting transcript...",
  participants: "Alice, Bob, Charlie",
  focus_areas: "action items, decisions"
});
```

### Available Templates

- **Privacy & Security**: `privacy-analysis`, `secure-rewrite`
- **Code Analysis**: `code-security-review`, `code-optimization`
- **Business**: `meeting-summary`, `email-draft`, `risk-assessment`
- **Research**: `research-synthesis`, `literature-review`
- **Content**: `content-adaptation`, `technical-documentation`

## ⚙️ Configuration

### Model Configuration

Configure different models for different capabilities:

```json
{
  "models": {
    "reasoning": {
      "name": "Reasoning Model",
      "capabilities": ["reasoning", "analysis", "problem-solving"],
      "defaultParams": {
        "temperature": 0.7,
        "max_tokens": 2000
      }
    },
    "privacy": {
      "name": "Privacy Model",
      "capabilities": ["privacy", "anonymization", "security"],
      "defaultParams": {
        "temperature": 0.2,
        "max_tokens": 1500
      }
    }
  }
}
```

### Privacy Settings

```json
{
  "privacy": {
    "defaultLevel": "moderate",
    "enableLogging": false,
    "logRetentionDays": 7
  }
}
```

### Performance Tuning

```json
{
  "performance": {
    "cacheEnabled": true,
    "cacheTTL": 3600,
    "maxConcurrentRequests": 5,
    "requestTimeout": 60000
  }
}
```

## 🔒 Privacy Levels

### Strict
- Never expose personal names, addresses, phone numbers, emails
- Generalize all specific locations and dates
- Remove all identifying information
- Use placeholders for sensitive data

### Moderate
- Protect personal identifiable information
- Generalize specific details when appropriate
- Maintain readability while ensuring privacy
- Remove sensitive financial or health data

### Minimal
- Protect obvious sensitive information (SSNs, credit cards)
- Remove personal contact information
- Maintain the natural flow of the text

## 🛠️ Development

### Project Structure

```
src/
├── index.ts              # Main MCP server
├── types.ts              # TypeScript type definitions
├── lm-studio-client.ts   # LM Studio API client
├── privacy-tools.ts      # Privacy-preserving functions
├── analysis-tools.ts     # Content analysis tools
├── prompt-templates.ts   # Pre-built prompt templates
└── config.ts            # Configuration management
```

### Building

```bash
# TypeScript compilation
npm run build

# Type checking
npm run typecheck

# Development with auto-reload
npm run dev
```

### Adding Custom Tools

1. Define the tool in `index.ts`:

```typescript
this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
  // ... existing tools
  case 'my-custom-tool':
    result = await this.handleCustomTool(args);
    break;
});
```

2. Implement the handler:

```typescript
private async handleCustomTool(args: any): Promise<MCPResponse> {
  // Your custom logic here
  const response = await this.lmStudio.generateResponse(
    args.prompt,
    args.system_prompt
  );

  return {
    content: [{ type: 'text', text: response }]
  };
}
```

## 🔍 Troubleshooting

### Common Issues

**LM Studio Connection Issues**
- Ensure LM Studio is running and the server is started
- Check that the base URL matches your LM Studio configuration
- Verify the model is loaded and available

**Performance Issues**
- Adjust the `maxConcurrentRequests` setting
- Increase `requestTimeout` for complex requests
- Consider using a more powerful local model

**Privacy Concerns**
- Review and adjust privacy level settings
- Enable strict privacy mode for sensitive data
- Disable logging if handling confidential information

### Debug Mode

Set environment variable for verbose logging:

```bash
DEBUG=mcp:* npm start
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

- Create an issue for bug reports
- Start a discussion for feature requests
- Check the documentation for common questions

---

**Note**: This server is designed to work with local LLMs for privacy-sensitive tasks. Always review the privacy settings and ensure they meet your requirements before processing confidential data.