# Test Suite for Local LLM MCP Server

This directory contains comprehensive tests for the Local LLM MCP Server. The tests are designed to verify real integration with LM Studio and ensure all MCP functionality works correctly.

## Test Files

### 1. `integration-test.ts` - Main Integration Tests
**Requires LM Studio to be running on port 1234**

This is the primary test suite that tests real integration scenarios with LM Studio. It performs actual LLM calls and verifies that:

- MCP server connects to LM Studio successfully
- All tools work with real LLM responses
- Privacy features correctly anonymize sensitive data
- Code analysis detects actual security vulnerabilities
- Template completion generates contextually appropriate content

**Run with:**
```bash
npm test
```

### 2. `basic-mcp-test.ts` - Basic MCP Protocol Tests
**Does not require LM Studio to be running**

Tests the core MCP protocol functionality without making LLM calls:

- Server startup and initialization
- Tool, resource, and prompt discovery
- Schema validation
- Basic resource access
- MCP protocol compliance

**Run with:**
```bash
npm run test:basic
```

### 3. `mcp-client-test.ts` - Comprehensive Test Suite
**Requires LM Studio to be running on port 1234**

Full comprehensive test suite that includes everything from the integration tests plus additional edge cases and error handling scenarios.

**Run with:**
```bash
npm run test:full
```

## Prerequisites

### For Integration Tests (npm test, npm run test:full)

1. **LM Studio Setup:**
   - Install and run LM Studio
   - Load a model (recommended: Llama 3.1 8B or similar)
   - Start the local server on port 1234

2. **Verify LM Studio is working:**
   ```bash
   curl http://localhost:1234/v1/models
   ```

3. **Build the project:**
   ```bash
   npm run build
   ```

### For Basic Tests (npm run test:basic)

Only requires the project to be built:
```bash
npm run build
```

## Test Scenarios

### Real Integration Scenarios Tested

1. **Mathematical Reasoning**: Tests if the local LLM can solve a word problem step-by-step
2. **Sentiment Analysis**: Analyzes mixed sentiment text and verifies structured output
3. **Privacy Protection**: Tests anonymization of sensitive personal and financial data
4. **Security Analysis**: Analyzes vulnerable code and detects SQL injection, hardcoded credentials, etc.
5. **Template Completion**: Fills customer service email templates with contextual information
6. **Prompt Templates**: Tests dynamic prompt generation with custom arguments

### Privacy and Security Validation

The tests specifically verify:
- Personal names, account numbers, and contact information are anonymized
- Credit card numbers and financial data are protected
- Security vulnerabilities in code are properly identified
- Template completion doesn't expose sensitive information from context

## Running Tests

### Quick Integration Test
```bash
# Ensure LM Studio is running, then:
npm test
```

### Development Workflow
```bash
# 1. Basic protocol test (no LM Studio needed)
npm run test:basic

# 2. If basic tests pass, run integration tests
npm test

# 3. For comprehensive testing
npm run test:full
```

### Continuous Integration
For CI/CD pipelines, use the basic test that doesn't require LM Studio:
```bash
npm run test:basic
```

## Test Output

The tests provide detailed output including:

- ✅ Pass/fail status for each test
- ⏱️ Execution time for performance monitoring
- 📊 Summary statistics (total tests, pass rate, etc.)
- 🔍 Preview of actual LLM responses
- 📋 Detailed error messages for debugging

### Example Output
```
🧪 LM Studio Integration Tests
Testing real integration with LM Studio running on localhost:1234

✅ LM Studio is running with 1 model(s) available
   Active model: llama-3.1-8b-instruct

🚀 Starting MCP server...
✅ MCP server started

🔌 Connecting MCP client...
✅ MCP client connected

🧪 Running integration test: LM Studio Connection
✅ LM Studio Connection - PASSED (234ms)
   LM Studio connected with 1 model(s)

🧪 Running integration test: Local Reasoning with LLM
✅ Local Reasoning with LLM - PASSED (3456ms)
   Generated 187 character response with correct answer
```

## Troubleshooting

### Common Issues

**LM Studio Not Found**
```
❌ LM Studio is not accessible at http://localhost:1234
```
- Ensure LM Studio is running
- Check that a model is loaded
- Verify local server is started in LM Studio

**Test Timeouts**
- Some LLM calls may take longer with larger models
- Consider using smaller/faster models for testing
- Check system resources (CPU, memory)

**Failed Privacy Tests**
- Verify the local model is capable of following instructions
- Some smaller models may struggle with complex anonymization tasks
- Consider using a more capable model for privacy-critical scenarios

### Debug Mode

For verbose output, you can modify the test files to include more logging or run with Node.js debug flags.

## Test Development

When adding new tests:

1. **Follow the existing pattern**: Use the `runTest()` method for consistency
2. **Test real scenarios**: Always test with actual LLM responses, not mocks
3. **Validate outputs**: Check both structure and content of responses
4. **Handle errors gracefully**: Tests should provide useful error messages
5. **Document assumptions**: Clearly state what the test expects from the LLM

### Adding New Integration Tests

```typescript
async testNewFeature(): Promise<any> {
  if (!this.client) throw new Error('Client not connected');

  const result = await this.client.callTool({
    name: 'your_tool',
    arguments: {
      // test arguments
    }
  });

  // Validate response structure
  if (!result.content || result.content.length === 0) {
    throw new Error('No response from tool');
  }

  // Validate response content
  const response = result.content[0].text;
  if (!response.includes('expected_content')) {
    throw new Error('Response missing expected content');
  }

  return {
    summary: 'Brief description of what was tested',
    // additional test data
  };
}
```

This test suite ensures the Local LLM MCP Server works correctly in real-world scenarios with actual LM Studio integration.