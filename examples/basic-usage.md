# Basic Usage Examples

This document provides practical examples of using the Local LLM MCP Server with different host applications.

## Setup

1. Ensure LM Studio is running with a model loaded on port 1234
2. Build and start the MCP server:
   ```bash
   npm run build
   npm start
   ```

## Claude Desktop Integration

### Configuration

Add to your Claude Desktop MCP configuration file (`~/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "local-llm": {
      "command": "node",
      "args": ["/path/to/local-llm-mcp-server/dist/index.js"],
      "env": {
        "NODE_ENV": "production"
      }
    }
  }
}
```

### Example Conversations

#### Privacy-Preserving Document Analysis

**User**: I have a confidential business plan that I need analyzed for sentiment and key insights, but I can't send it to cloud services.

**Claude**: I can help you analyze your confidential business plan using our local LLM server to ensure your data stays private. Let me use the private analysis tool.

```
Tool: private_analysis
Arguments:
{
  "content": "[Your confidential business plan content here]",
  "analysis_type": "sentiment",
  "domain": "financial"
}
```

The analysis shows a predominantly positive sentiment (0.82 confidence) with strong enthusiasm around market opportunities and growth projections...

#### Secure Code Review

**User**: Can you review this authentication function for security issues? It handles sensitive user data.

**Claude**: I'll analyze your authentication function locally to ensure sensitive code doesn't leave your system.

```
Tool: code_analysis
Arguments:
{
  "code": "function authenticateUser(username, password) {\n  const query = `SELECT * FROM users WHERE username = '${username}' AND password = '${password}'`;\n  return db.query(query);\n}",
  "language": "javascript",
  "analysis_focus": "security"
}
```

The analysis reveals several critical security vulnerabilities:
1. **SQL Injection** (Critical): Direct string interpolation allows SQL injection attacks
2. **Plain Text Passwords** (High): Passwords should be hashed
3. **Missing Input Validation** (Medium): No validation on username/password inputs...

#### Privacy-Aware Content Rewriting

**User**: I need to rewrite this customer email to remove personal information but keep it professional.

**Claude**: I'll rewrite the email locally to protect customer privacy while maintaining professionalism.

```
Tool: secure_rewrite
Arguments:
{
  "content": "Dear John Smith at ABC Corp, Thank you for calling about your account #12345. We've resolved the billing issue with your credit card ending in 4567...",
  "style": "professional",
  "privacy_level": "strict"
}
```

Result: "Dear [Customer Name] at [Company], Thank you for contacting us about your account. We've resolved the billing issue with your payment method..."

## Advanced Examples

### Multi-Step Analysis Workflow

```typescript
// Step 1: Privacy scan
const privacyResult = await tools.private_analysis({
  content: documentContent,
  analysis_type: "privacy_scan"
});

// Step 2: If safe, proceed with analysis
if (privacyResult.overallRisk === "low") {
  const analysis = await tools.private_analysis({
    content: documentContent,
    analysis_type: "key_points",
    domain: "legal"
  });
}

// Step 3: Generate summary with privacy protection
const summary = await tools.secure_rewrite({
  content: analysis.summary,
  style: "executive summary",
  privacy_level: "moderate"
});
```

### Template-Based Email Generation

```typescript
// Using prompt template for professional email
const emailPrompt = await prompts.get("email-draft", {
  purpose: "follow up on project status",
  recipient: "project team",
  tone: "professional",
  key_points: "milestone completion, next steps, resource needs"
});

// Generate email using local LLM
const email = await tools.local_reasoning({
  prompt: emailPrompt.messages[0].content.text,
  model_params: {
    temperature: 0.6,
    max_tokens: 800
  }
});
```

### Code Documentation Generation

```typescript
// Generate documentation for complex code
const docs = await tools.template_completion({
  template: `# {{function_name}} Function

## Description
{{description}}

## Parameters
{{parameters}}

## Returns
{{returns}}

## Example Usage
{{example}}

## Security Considerations
{{security_notes}}`,
  context: `Function: calculatePayrollTaxes
Code: ${complexPayrollFunction}
Language: TypeScript`,
  format: "markdown"
});
```

### Research Synthesis

```typescript
// Synthesize multiple research documents
const synthesis = await prompts.get("research-synthesis", {
  sources: `
    Document 1: Market Analysis Report Q3 2024
    Document 2: Competitor Landscape Review
    Document 3: Customer Feedback Summary
  `,
  research_question: "What are the key market opportunities for our product?",
  output_format: "executive summary"
});
```

## Resource Usage Examples

### Check Available Models

```typescript
const modelsInfo = await resources.read("local://models");
console.log(JSON.parse(modelsInfo.contents[0].text));
// Output: { models: ["llama-3-8b", "codellama-7b"], count: 2 }
```

### Monitor Server Status

```typescript
const status = await resources.read("local://status");
const statusData = JSON.parse(status.contents[0].text);

if (statusData.status === "offline") {
  console.log("LM Studio server is not responding");
} else {
  console.log(`Server online at ${statusData.timestamp}`);
}
```

### View Configuration

```typescript
const config = await resources.read("local://config");
const configData = JSON.parse(config.contents[0].text);

console.log("Available capabilities:", configData.capabilities);
console.log("Supported domains:", configData.supported_domains);
```

## Best Practices

### 1. Privacy-First Workflow

```typescript
// Always check privacy implications first
const privacyCheck = await tools.private_analysis({
  content: userInput,
  analysis_type: "privacy_scan"
});

if (privacyCheck.overallRisk === "high") {
  // Use strict privacy mode
  const result = await tools.secure_rewrite({
    content: userInput,
    style: "anonymized",
    privacy_level: "strict"
  });
} else {
  // Proceed with normal analysis
  const result = await tools.private_analysis({
    content: userInput,
    analysis_type: "sentiment",
    domain: "general"
  });
}
```

### 2. Domain-Specific Processing

```typescript
// Tailor analysis to domain context
const medicalAnalysis = await tools.private_analysis({
  content: patientNotes,
  analysis_type: "entities",
  domain: "medical" // Uses medical terminology and HIPAA considerations
});

const legalAnalysis = await tools.private_analysis({
  content: contractText,
  analysis_type: "classification",
  domain: "legal" // Applies legal terminology and confidentiality
});
```

### 3. Performance Optimization

```typescript
// Use appropriate model parameters for task
const quickDraft = await tools.local_reasoning({
  prompt: "Generate a brief project update",
  model_params: {
    temperature: 0.8,  // Higher creativity
    max_tokens: 500,   // Shorter response
    top_p: 0.95
  }
});

const detailedAnalysis = await tools.local_reasoning({
  prompt: "Provide comprehensive risk assessment",
  model_params: {
    temperature: 0.3,  // More focused
    max_tokens: 2000,  // Longer response
    top_p: 0.8
  }
});
```

### 4. Error Handling

```typescript
try {
  const result = await tools.private_analysis({
    content: userInput,
    analysis_type: "sentiment",
    domain: "financial"
  });

  if (result.isError) {
    console.error("Analysis failed:", result.content[0].text);
    // Fallback to simpler analysis
  }
} catch (error) {
  console.error("Tool execution failed:", error);
  // Handle gracefully
}
```

## Integration Patterns

### 1. Hybrid Cloud-Local Processing

```typescript
// Use local LLM for sensitive data, cloud for general tasks
function processContent(content, isSensitive) {
  if (isSensitive) {
    return tools.private_analysis({
      content,
      analysis_type: "summary",
      domain: "general"
    });
  } else {
    // Use cloud-based analysis for non-sensitive content
    return cloudAnalysisService.analyze(content);
  }
}
```

### 2. Progressive Privacy Protection

```typescript
// Start with privacy scan, then apply appropriate protection level
const privacyAssessment = await tools.private_analysis({
  content: userDocument,
  analysis_type: "privacy_scan"
});

const protectionLevel = privacyAssessment.overallRisk === "high" ? "strict" :
                       privacyAssessment.overallRisk === "medium" ? "moderate" : "minimal";

const processedContent = await tools.secure_rewrite({
  content: userDocument,
  style: "professional",
  privacy_level: protectionLevel
});
```

### 3. Batch Processing

```typescript
// Process multiple documents efficiently
async function processBatch(documents) {
  const results = [];

  for (const doc of documents) {
    const analysis = await tools.private_analysis({
      content: doc.content,
      analysis_type: "key_points",
      domain: doc.domain || "general"
    });

    results.push({
      id: doc.id,
      analysis: analysis,
      timestamp: new Date().toISOString()
    });
  }

  return results;
}
```

These examples demonstrate the flexibility and power of the Local LLM MCP Server for privacy-preserving AI workflows while maintaining the ability to perform sophisticated analysis and content generation tasks.