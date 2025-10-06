# Testing Guide - Version 2.0.0

Quick reference for testing all new features added in v2.0.0.

---

## 🔧 Setup

1. **Build the server:**
   ```bash
   npm run build
   ```

2. **Start the server:**
   ```bash
   npm run start
   # OR for remote access:
   npm run start:remote
   ```

3. **Ensure LM Studio is running** with at least one model loaded.

---

## ✅ Test Checklist

### 1. Protocol Handshake ✅

**What:** Server now properly responds to initialize requests with capabilities.

**Test in Claude:**
```
The server should auto-connect when you open Claude Desktop.
Check that no initialization errors appear in the logs.
```

**Expected:** Silent success - server connects automatically.

---

### 2. Resource Access ✅

**What:** Read model metadata and server info via MCP resources.

**Test Resource: `local://models`**
```
Read resource: local://models
```

**Expected Response:**
```json
{
  "models": [
    {
      "id": "openai/gpt-oss-20b",
      "name": "Openai Gpt Oss 20b",
      "type": "llm",
      "provider": "OpenAI",
      "parameters": "20B",
      "capabilities": ["chat", "reasoning", "completion"],
      "contextWindow": 8192,
      "isDefault": true,
      "status": "ready"
    }
  ],
  "llmModels": ["openai/gpt-oss-20b", "qwen3-30b-..."],
  "embeddingModels": ["text-embedding-nomic-..."],
  "defaultModel": "openai/gpt-oss-20b",
  "count": 3
}
```

**Test Resource: `local://status`**
```
Read resource: local://status
```

**Expected Response:**
```json
{
  "status": "online",
  "timestamp": "2025-10-06T..."
}
```

**Test Resource: `local://config`**
```
Read resource: local://config
```

**Expected Response:**
```json
{
  "server": "local-llm-mcp-server",
  "version": "2.0.0",
  "capabilities": ["reasoning", "analysis", "rewriting", "code_analysis", "templates"],
  "privacy_levels": ["strict", "moderate", "minimal"],
  "supported_domains": ["general", "medical", "legal", "financial", "technical", "academic"],
  "analysis_types": ["sentiment", "entities", "classification", "summary", "key_points", "privacy_scan", "security_audit"]
}
```

**Test Resource: `local://capabilities`**
```
Read resource: local://capabilities
```

**Expected:** Detailed JSON with examples for each tool.

---

### 3. Discovery Tools ✅

**What:** New tools to list and inspect models without errors.

**Test: List All Models**
```
Use tool: list_models with {}
```

**Expected Response:**
```json
{
  "models": [
    {
      "id": "openai/gpt-oss-20b",
      "name": "Openai Gpt Oss 20b",
      "type": "llm",
      "provider": "OpenAI",
      "parameters": "20B",
      "capabilities": ["chat", "reasoning", "completion"],
      "contextWindow": 8192,
      "isDefault": true,
      "status": "ready",
      "metadata": { ... }
    }
  ],
  "defaultModel": "openai/gpt-oss-20b",
  "summary": {
    "total": 3,
    "llmCount": 2,
    "embeddingCount": 1
  }
}
```

**Test: List Only LLM Models**
```
Use tool: list_models with {"type": "llm"}
```

**Expected:** Only LLM models returned, no embedding models.

**Test: List Only Embedding Models**
```
Use tool: list_models with {"type": "embedding"}
```

**Expected:** Only embedding models returned.

**Test: Get Model Info**
```
Use tool: get_model_info with {"model": "openai/gpt-oss-20b"}
```

**Expected Response:**
```json
{
  "model": {
    "id": "openai/gpt-oss-20b",
    "displayName": "GPT OSS 20B",
    "provider": "OpenAI",
    "architecture": "GPT-style Transformer",
    "parameters": "20B",
    "type": "llm",
    "capabilities": ["chat", "reasoning", "completion", "code-generation"],
    "contextWindow": 8192,
    "maxTokens": 4096,
    "isDefault": true,
    "status": "ready",
    "metadata": {
      "created": 1234567890,
      "ownedBy": "openai"
    },
    "usage": {
      "purpose": "Text generation, reasoning, and completion tasks",
      "bestFor": ["Chat", "Analysis", "Code generation", "Reasoning"]
    }
  }
}
```

**Test: Invalid Model (Error Handling)**
```
Use tool: get_model_info with {"model": "invalid-model-name"}
```

**Expected Response:**
```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Model 'invalid-model-name' not found",
    "details": {
      "requestedModel": "invalid-model-name",
      "suggestion": "Use list_models tool to see available models"
    },
    "availableModels": ["openai/gpt-oss-20b", "qwen3-30b-...", "text-embedding-..."]
  }
}
```

---

### 4. Model Self-Reporting Accuracy ✅

**What:** Models now report their actual specifications, not GPT-4.

**Test: Ask Model About Itself**
```
Use tool: local_reasoning with {
  "prompt": "What are your model specifications? Include your model name, parameter count, provider, and context window.",
  "model": "openai/gpt-oss-20b"
}
```

**Expected Response (from model):**
```
I am GPT OSS 20B, a 20 billion parameter GPT-style language model.
My specifications:
- Model Name: GPT OSS 20B
- Provider: OpenAI (Open Source)
- Architecture: GPT-style Transformer
- Parameters: 20 billion
- Context Window: 8,192 tokens
- Capabilities: chat, reasoning, completion
```

**❌ SHOULD NOT SAY:** "I am GPT-4 Turbo" or "I have 12 billion parameters"

**Test: Ask Qwen Model**
```
Use tool: local_reasoning with {
  "prompt": "What model are you?",
  "model": "qwen3-30b-a3b-deepseek-distill-instruct-2507"
}
```

**Expected:** Model correctly identifies as Qwen3 30B, not GPT-4.

---

### 5. Enhanced Privacy Analysis ✅

**What:** Privacy scan now detects actual PII with regex + LLM.

**Test: Basic PII Detection**
```
Use tool: private_analysis with {
  "content": "Contact me at john.doe@example.com or call 555-123-4567",
  "analysis_type": "privacy_scan"
}
```

**Expected Response:**
```json
{
  "issues": [
    {
      "type": "PII - Email Address",
      "severity": "medium",
      "description": "Found 1 email address(es): john.doe@example.com",
      "location": "Content"
    },
    {
      "type": "PII - Phone Number",
      "severity": "medium",
      "description": "Found 1 potential phone number(s)",
      "location": "Content"
    }
  ],
  "overallRisk": "medium",
  "recommendations": [
    "Review all detected PII",
    "Consider redacting or anonymizing sensitive information",
    "Ensure GDPR compliance if handling EU data"
  ]
}
```

**Test: Credit Card Detection**
```
Use tool: private_analysis with {
  "content": "My credit card is 4532-1234-5678-9010",
  "analysis_type": "privacy_scan"
}
```

**Expected:** High severity issue for credit card detected.

**Test: SSN Detection**
```
Use tool: private_analysis with {
  "content": "SSN: 123-45-6789",
  "analysis_type": "privacy_scan"
}
```

**Expected:** High severity issue for SSN detected.

**Test: No PII**
```
Use tool: private_analysis with {
  "content": "The weather is nice today",
  "analysis_type": "privacy_scan"
}
```

**Expected:**
```json
{
  "issues": [],
  "overallRisk": "low",
  "recommendations": [ ... ]
}
```

---

## 🧪 Integration Tests

### Test 1: Full Workflow

1. **List available models:**
   ```
   Use tool: list_models with {"type": "llm"}
   ```

2. **Get info about first model:**
   ```
   Use tool: get_model_info with {"model": "<id-from-step-1>"}
   ```

3. **Ask model about itself:**
   ```
   Use tool: local_reasoning with {
     "prompt": "What are you?",
     "model": "<id-from-step-1>"
   }
   ```

4. **Verify accuracy:** Model response should match metadata from step 2.

### Test 2: Privacy Scan Workflow

1. **Scan content with PII:**
   ```
   Use tool: private_analysis with {
     "content": "John Smith (john@example.com) called from 555-1234 regarding account #123-45-6789",
     "analysis_type": "privacy_scan"
   }
   ```

2. **Verify detections:**
   - ✅ Email detected
   - ✅ Phone detected
   - ✅ SSN detected (if format matches)
   - ✅ Overall risk should be HIGH

3. **Get recommendations** from the response.

### Test 3: Error Handling

1. **Try invalid model in set_default_model:**
   ```
   Use tool: set_default_model with {"model": "nonexistent-model"}
   ```

2. **Verify structured error:**
   - ✅ Contains `error` object
   - ✅ Contains `error.code`
   - ✅ Contains `error.message`
   - ✅ Contains `error.details`
   - ✅ Contains `availableModels` array

---

## 📊 Success Criteria

### ✅ All tests pass if:

1. **Resources are accessible** and return valid JSON
2. **Discovery tools work** without requiring error-based discovery
3. **Models report accurate information** about themselves
4. **Privacy scan detects PII** using both regex and LLM
5. **Errors are structured** with codes and details
6. **Server initializes** without errors

### ❌ Failure indicators:

- ❌ "Resource not found" errors
- ❌ Models claiming to be GPT-4/GPT-4 Turbo
- ❌ Privacy scan returns empty issues for obvious PII
- ❌ Errors are plain strings without structure
- ❌ Tools require triggering errors to discover models

---

## 🔍 Troubleshooting

### Issue: Resources return "Resource not found"

**Solution:** Ensure you're using the built version:
```bash
npm run build
npm run start
```

### Issue: Models still claim to be GPT-4

**Solution:** Clear any cached system prompts:
1. Restart LM Studio
2. Rebuild the server: `npm run build`
3. Restart the MCP server

### Issue: Privacy scan finds nothing

**Solution:**
1. Check LM Studio is running
2. Verify the model is loaded
3. Check server logs for errors

### Issue: Tools show "local-llm-remote:" prefix

**Note:** This is normal - the MCP client adds this prefix for namespacing. The actual tool names in the code are correct.

---

## 📝 Test Results Template

Use this template to record your test results:

```
## Test Results - v2.0.0

**Date:** 2025-10-06
**Tester:** [Your Name]
**Environment:** [Local/Remote]

### Resources
- [ ] local://models - PASS/FAIL
- [ ] local://status - PASS/FAIL
- [ ] local://config - PASS/FAIL
- [ ] local://capabilities - PASS/FAIL

### Discovery Tools
- [ ] list_models (all) - PASS/FAIL
- [ ] list_models (llm) - PASS/FAIL
- [ ] list_models (embedding) - PASS/FAIL
- [ ] get_model_info (valid) - PASS/FAIL
- [ ] get_model_info (invalid) - PASS/FAIL

### Model Accuracy
- [ ] GPT OSS 20B introspection - PASS/FAIL
- [ ] Qwen3 introspection - PASS/FAIL

### Privacy Analysis
- [ ] Email detection - PASS/FAIL
- [ ] Phone detection - PASS/FAIL
- [ ] SSN detection - PASS/FAIL
- [ ] Credit card detection - PASS/FAIL
- [ ] No PII baseline - PASS/FAIL

### Error Handling
- [ ] Structured error codes - PASS/FAIL
- [ ] Error details object - PASS/FAIL
- [ ] Available models in error - PASS/FAIL

**Overall Result:** PASS/FAIL
**Notes:** [Any observations]
```

---

**Happy Testing! 🎉**

If you find any issues, please report them on GitHub with:
1. The exact tool/resource that failed
2. The input you used
3. The expected vs actual output
4. Server logs (if available)
