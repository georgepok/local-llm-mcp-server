# API Documentation

Complete reference for the Local LLM MCP Server tools, resources, and prompts.

## Table of Contents

- [Tools](#tools)
- [Resources](#resources)
- [Prompts](#prompts)
- [Model Management](#model-management)
- [Error Handling](#error-handling)

---

## Tools

### `local_reasoning`

Use local LLM for general-purpose reasoning and question answering.

**Parameters:**
- `prompt` (string, required): The question or task to process
- `system_prompt` (string, optional): System context or role definition
- `model_params` (object, optional): Model parameters
  - `temperature` (number, 0-2): Randomness (default: 0.7)
  - `max_tokens` (number, 1-8192): Maximum response length
  - `top_p` (number, 0-1): Nucleus sampling
  - `frequency_penalty` (number, -2 to 2): Penalize repeated tokens
  - `presence_penalty` (number, -2 to 2): Penalize new topics
- `model` (string, optional): Specific model name (overrides default)

**Example:**
```json
{
  "prompt": "Explain quantum entanglement in simple terms",
  "system_prompt": "You are a physics teacher explaining concepts to beginners",
  "model_params": {
    "temperature": 0.7,
    "max_tokens": 500
  },
  "model": "qwen3-30b-a3b-deepseek-distill-instruct-2507"
}
```

**Response:**
```
{
  "content": [
    {
      "type": "text",
      "text": "Quantum entanglement is a phenomenon where..."
    }
  ]
}
```

---

### `private_analysis`

Analyze content locally without cloud exposure.

**Parameters:**
- `content` (string, required): Text to analyze
- `analysis_type` (enum, required): Type of analysis
  - `sentiment`: Detect emotions and sentiment
  - `entities`: Extract named entities
  - `classification`: Categorize content
  - `summary`: Generate summary with key points
  - `key_points`: Extract main points and insights
  - `privacy_scan`: Detect PII and privacy issues
  - `security_audit`: Security vulnerability assessment
- `domain` (enum, optional): Domain context (default: "general")
  - `general`, `medical`, `legal`, `financial`, `technical`, `academic`

**Example - Sentiment Analysis:**
```json
{
  "content": "The new policy changes are concerning but necessary for growth",
  "analysis_type": "sentiment",
  "domain": "financial"
}
```

**Response:**
```json
{
  "sentiment": "neutral",
  "confidence": 0.75,
  "emotions": ["concern", "optimism"],
  "reasoning": "Mixed sentiment with concern balanced by positive outlook"
}
```

**Example - Entity Extraction:**
```json
{
  "content": "Dr. Smith from Mayo Clinic prescribed medication X on January 15, 2024",
  "analysis_type": "entities",
  "domain": "medical"
}
```

**Response:**
```json
{
  "people": ["Dr. Smith"],
  "organizations": ["Mayo Clinic"],
  "locations": [],
  "dates": ["January 15, 2024"],
  "domainSpecific": {
    "medications": ["medication X"],
    "medical_conditions": [],
    "procedures": []
  }
}
```

**Example - Privacy Scan:**
```json
{
  "content": "Contact John Doe at john.doe@email.com or 555-123-4567",
  "analysis_type": "privacy_scan"
}
```

**Response:**
```json
{
  "issues": [
    {
      "type": "email",
      "severity": "medium",
      "description": "Email address detected",
      "location": "john.doe@email.com"
    },
    {
      "type": "phone",
      "severity": "medium",
      "description": "Phone number detected",
      "location": "555-123-4567"
    }
  ],
  "overallRisk": "medium",
  "recommendations": [
    "Remove or mask email addresses",
    "Remove or mask phone numbers"
  ]
}
```

---

### `secure_rewrite`

Rewrite text with privacy protection and style transformation.

**Parameters:**
- `content` (string, required): Original text to rewrite
- `style` (string, required): Target style (e.g., "formal", "casual", "professional")
- `privacy_level` (enum, optional): Privacy protection level (default: "moderate")
  - `strict`: Maximum privacy, remove all PII
  - `moderate`: Balance privacy and readability
  - `minimal`: Basic PII protection only

**Example:**
```json
{
  "content": "Hey, call me at 555-1234 to discuss the contract",
  "style": "professional",
  "privacy_level": "strict"
}
```

**Response:**
```
"Please contact me via provided communication channels to discuss the contractual agreement."
```

---

### `code_analysis`

Analyze code locally for security, quality, or documentation.

**Parameters:**
- `code` (string, required): Source code to analyze
- `language` (string, required): Programming language (e.g., "python", "javascript", "java")
- `analysis_focus` (enum, required): Focus area
  - `security`: Vulnerability and security issues
  - `quality`: Code quality and best practices
  - `documentation`: Documentation needs
  - `optimization`: Performance optimization
  - `bugs`: Potential bugs and errors

**Example:**
```json
{
  "code": "def login(username, password):\n    query = f\"SELECT * FROM users WHERE name='{username}' AND pwd='{password}'\"\n    return db.execute(query)",
  "language": "python",
  "analysis_focus": "security"
}
```

**Response:**
```json
{
  "analysis": "Critical SQL injection vulnerability detected",
  "issues": [
    {
      "line": 2,
      "severity": "high",
      "description": "SQL Injection vulnerability - user input directly in query",
      "suggestion": "Use parameterized queries or ORM to prevent SQL injection"
    },
    {
      "line": 2,
      "severity": "high",
      "description": "Plaintext password in query",
      "suggestion": "Passwords should be hashed before comparison"
    }
  ],
  "metrics": {
    "linesOfCode": 3,
    "functionsFound": 1
  },
  "recommendations": [
    "Implement input validation and sanitization",
    "Use parameterized queries to prevent SQL injection",
    "Hash passwords before storage and comparison"
  ]
}
```

---

### `template_completion`

Complete templates with intelligent placeholder filling.

**Parameters:**
- `template` (string, required): Template with `[PLACEHOLDER]` markers
- `context` (string, required): Context information for completion
- `format` (string, optional): Output format requirements

**Example:**
```json
{
  "template": "Dear [CUSTOMER_NAME],\n\nThank you for your [REQUEST_TYPE] regarding [SUBJECT].\n\nNext steps:\n1. [STEP_1]\n2. [STEP_2]\n3. [STEP_3]\n\nBest regards,\n[AGENT_NAME]",
  "context": "Customer John Smith requested a premium account upgrade for advanced analytics features"
}
```

**Response:**
```
"Dear John Smith,

Thank you for your upgrade request regarding premium analytics features.

Next steps:
1. Verify your payment method
2. Explore your new advanced features
3. Contact us if you have any questions

Best regards,
Customer Service Team"
```

---

### `set_default_model`

Change the default model for the current session.

**Parameters:**
- `model` (string, required): Model name from available models

**Example:**
```json
{
  "model": "qwen3-30b-a3b-deepseek-distill-instruct-2507"
}
```

**Response:**
```json
{
  "success": true,
  "defaultModel": "qwen3-30b-a3b-deepseek-distill-instruct-2507",
  "previousDefault": "openai/gpt-oss-20b",
  "message": "Default model set to \"qwen3-30b-a3b-deepseek-distill-instruct-2507\""
}
```

**Error Response (invalid model):**
```json
{
  "error": "Model \"invalid-model\" not found",
  "availableModels": [
    "openai/gpt-oss-20b",
    "qwen3-30b-a3b-deepseek-distill-instruct-2507"
  ],
  "currentDefault": "openai/gpt-oss-20b",
  "message": "Model \"invalid-model\" is not available. Choose from the available models listed above."
}
```

---

## Resources

### `local://models`

List all models loaded in LM Studio with current default.

**Response:**
```json
{
  "models": [
    "openai/gpt-oss-20b",
    "qwen3-30b-a3b-deepseek-distill-instruct-2507",
    "text-embedding-nomic-embed-text-v1.5"
  ],
  "count": 3,
  "defaultModel": "openai/gpt-oss-20b"
}
```

---

### `local://status`

Check if LM Studio server is online/offline.

**Response:**
```json
{
  "status": "online",
  "timestamp": "2025-01-05T02:22:09.412Z"
}
```

---

### `local://config`

View server capabilities, domains, privacy levels, and analysis types.

**Response:**
```json
{
  "server": "local-llm-mcp-server",
  "version": "1.0.0",
  "capabilities": [
    "reasoning",
    "analysis",
    "rewriting",
    "code_analysis",
    "templates"
  ],
  "privacy_levels": ["strict", "moderate", "minimal"],
  "supported_domains": [
    "general",
    "medical",
    "legal",
    "financial",
    "technical",
    "academic"
  ],
  "analysis_types": [
    "sentiment",
    "entities",
    "classification",
    "summary",
    "key_points",
    "privacy_scan",
    "security_audit"
  ]
}
```

---

### `local://capabilities`

Comprehensive documentation including tool examples and workflow guide.

**Response:** (Abbreviated example)
```json
{
  "modelManagement": {
    "availableModels": ["openai/gpt-oss-20b", "qwen3-30b..."],
    "defaultModel": "openai/gpt-oss-20b",
    "howToUse": {
      "readModels": "Read resource: local://models",
      "setDefault": "Use tool: set_default_model with {\"model\": \"model-name\"}",
      "useSpecificModel": "Add \"model\" parameter to any tool call",
      "useDefaultModel": "Omit \"model\" parameter to use the default model"
    }
  },
  "tools": {
    "local_reasoning": {
      "purpose": "General-purpose reasoning and question answering",
      "parameters": {...},
      "example": "{\"prompt\": \"Explain quantum computing\", \"model\": \"openai/gpt-oss-20b\"}"
    }
    // ... other tools
  },
  "workflow": {
    "step1": "Read local://models to see available models",
    "step2": "Optionally set a default model using set_default_model tool",
    "step3": "Use any tool - it will use default model unless you specify \"model\" parameter",
    "step4": "Override default by passing \"model\" parameter in any tool call"
  }
}
```

---

## Prompts

Pre-built prompt templates for common tasks.

### Available Prompts

1. **`privacy-analysis`** - Analyze content for privacy issues
2. **`secure-rewrite`** - Rewrite with privacy protection
3. **`document-analysis`** - Comprehensive document analysis
4. **`risk-assessment`** - Risk analysis with mitigation
5. **`code-security-review`** - Security-focused code review
6. **`code-optimization`** - Code optimization suggestions
7. **`meeting-summary`** - Meeting summaries with action items
8. **`email-draft`** - Professional email drafting
9. **`research-synthesis`** - Synthesize research sources
10. **`literature-review`** - Systematic literature review
11. **`content-adaptation`** - Adapt content for different audiences
12. **`technical-documentation`** - Generate technical docs

---

## Model Management

### Workflow

1. **Discover Models**
   ```
   Read resource: local://models
   ```

2. **View Current Default**
   ```json
   // In response from local://models
   {
     "defaultModel": "openai/gpt-oss-20b"
   }
   ```

3. **Change Default**
   ```json
   {
     "tool": "set_default_model",
     "arguments": {
       "model": "qwen3-30b-a3b-deepseek-distill-instruct-2507"
     }
   }
   ```

4. **Use Default Model**
   ```json
   {
     "tool": "local_reasoning",
     "arguments": {
       "prompt": "Your question here"
       // No model specified - uses default
     }
   }
   ```

5. **Override for Specific Request**
   ```json
   {
     "tool": "local_reasoning",
     "arguments": {
       "prompt": "Your question here",
       "model": "openai/gpt-oss-20b"  // Overrides default
     }
   }
   ```

---

## Error Handling

### Common Errors

**Model Not Found:**
```json
{
  "error": "Model \"invalid-model\" not found",
  "availableModels": ["model1", "model2"],
  "currentDefault": "model1",
  "message": "Model \"invalid-model\" is not available..."
}
```

**No Default Model Set:**
```
"Error: No model specified and no default model set. Use setDefaultModel() or pass a model parameter."
```

**LM Studio Connection Failed:**
```
"Error: Failed to generate response: Failed to connect to LM Studio"
```

**Timeout:**
```
"Error: Request timed out after 600s. Performance suggestions: Reduce max_tokens to 1000 or less, Try using a faster/smaller model..."
```

### Best Practices

1. **Always check `local://models` first** to see available models
2. **Set a default model** for your session to avoid specifying it every time
3. **Use appropriate model parameters** based on your task
4. **Handle errors gracefully** - check for `isError: true` in responses
5. **Monitor performance** - adjust `max_tokens` and `temperature` as needed
