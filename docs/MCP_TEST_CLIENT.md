# MCP Test Client - Dynamic Testing Framework

A flexible, interactive MCP client that approaches the server like Claude Desktop does, allowing dynamic test case execution without precompilation.

---

## Features

✅ **Dynamic Test Execution** - No precompiled test cases, run tests on demand
✅ **Interactive Mode** - REPL-style interface for manual testing
✅ **Category-Based Organization** - Tests grouped by protocol, resources, tools, prompts, integration
✅ **Verbose Debugging** - Optional detailed logging for troubleshooting
✅ **Automated Test Suites** - Run all tests or specific categories
✅ **Real-Time Results** - Immediate feedback with timing information

---

## Quick Start

### Run All Tests
```bash
npm run test:client:all
```

### Interactive Mode
```bash
npm run test:client
```

### Verbose Output
```bash
npm run test:client:verbose
```

---

## Usage

### 1. Automated Testing

**Run all tests:**
```bash
npm run test:client:all
```

**Output:**
```
🚀 Running all tests

📂 Running protocol tests (1 tests)
▶️  Running: Protocol Handshake
✅ Server initialized successfully

📂 Running resources tests (5 tests)
▶️  Running: List Resources
✅ Found all 4 expected resources (1ms)
...

============================================================
📊 Test Summary
============================================================
Total:  12
Passed: 12 ✅
Failed: 0 ❌
============================================================
```

### 2. Interactive Mode

**Start interactive mode:**
```bash
npm run test:client
```

**Available commands:**
```
> list           - List all tests
> run <id>       - Run a specific test
> category <cat> - Run all tests in category
> all            - Run all tests
> verbose        - Toggle verbose mode
> quit           - Exit
```

**Example session:**
```bash
npm run test:client

> list

📋 Available Tests:

PROTOCOL:
  protocol.initialize - Protocol Handshake

RESOURCES:
  resources.list - List Resources
  resources.read.models - Read local://models
  resources.read.status - Read local://status
  resources.read.config - Read local://config
  resources.read.capabilities - Read local://capabilities

TOOLS:
  tools.list - List Tools
  tools.call.list_models - Call list_models
  tools.call.get_model_info - Call get_model_info
  tools.error.invalid_model - Test Error Handling

PROMPTS:
  prompts.list - List Prompts

INTEGRATION:
  integration.discovery_workflow - Discovery Workflow

> run resources.read.models

▶️  Running: Read local://models
   Test reading the models resource
✅ Successfully read models (3 total) (2ms)

> category tools

📂 Running tools tests (4 tests)

▶️  Running: List Tools
✅ Found all 8 expected tools

▶️  Running: Call list_models
✅ Found 3 models (2ms)

▶️  Running: Call get_model_info
✅ Got info for Openai/gpt Oss 20b (3ms)

▶️  Running: Test Error Handling
✅ Structured error with code: MODEL_NOT_FOUND (1ms)

> quit
```

---

## Test Categories

### Protocol Tests (1 test)
- **protocol.initialize** - Verify server handshake and capability advertisement

### Resource Tests (5 tests)
- **resources.list** - List all available resources
- **resources.read.models** - Read `local://models` resource
- **resources.read.status** - Read `local://status` resource
- **resources.read.config** - Read `local://config` resource
- **resources.read.capabilities** - Read `local://capabilities` resource

### Tool Tests (4 tests)
- **tools.list** - List all available tools
- **tools.call.list_models** - Call `list_models` tool
- **tools.call.get_model_info** - Call `get_model_info` tool
- **tools.error.invalid_model** - Test structured error handling

### Prompt Tests (1 test)
- **prompts.list** - List available prompt templates

### Integration Tests (1 test)
- **integration.discovery_workflow** - Test complete model discovery workflow

---

## Advanced Usage

### Run Specific Test
```bash
npm run build && tsx src/mcp-test-client.ts -- --test=tools.list
```

### Custom Server Path
```typescript
import { MCPTestClient } from './src/mcp-test-client';

const client = new MCPTestClient(true); // verbose = true
await client.connect('node', ['dist/index.js']);
await client.runAll();
await client.disconnect();
```

---

## Test Results Format

Each test returns a `TestResult` object:

```typescript
interface TestResult {
  success: boolean;      // Test passed/failed
  message: string;       // Human-readable result
  data?: any;           // Response data (if applicable)
  error?: any;          // Error details (if failed)
  duration?: number;    // Execution time in ms
}
```

**Example:**
```json
{
  "success": true,
  "message": "Found all 4 expected resources",
  "data": [
    { "uri": "local://models", "name": "Available Local Models", ... },
    { "uri": "local://status", "name": "Server Status", ... },
    ...
  ],
  "duration": 1
}
```

---

## Adding Custom Tests

### 1. Register a New Test

```typescript
this.addTest({
  id: 'custom.my_test',
  name: 'My Custom Test',
  description: 'Test description',
  category: 'tools', // or 'protocol', 'resources', 'prompts', 'integration'
  run: async (client) => {
    const start = Date.now();
    try {
      // Your test logic here
      const result = await client.callTool({
        name: 'my_tool',
        arguments: { param: 'value' }
      });

      return {
        success: true,
        message: 'Test passed',
        data: result,
        duration: Date.now() - start
      };
    } catch (error) {
      return {
        success: false,
        message: `Test failed: ${(error as Error).message}`,
        error,
        duration: Date.now() - start
      };
    }
  }
});
```

### 2. Run Your Test

**Programmatically:**
```typescript
const client = new MCPTestClient();
await client.connect();
await client.runTest('custom.my_test');
await client.disconnect();
```

**Interactively:**
```bash
npm run test:client

> run custom.my_test
```

---

## Debugging

### Enable Verbose Mode

**CLI:**
```bash
npm run test:client:verbose
```

**Interactive:**
```
> verbose
Verbose mode: ON
```

**Programmatic:**
```typescript
const client = new MCPTestClient(true); // Enable verbose
```

### Verbose Output Includes:

- Full response data (first 200 characters)
- Error stack traces
- Detailed timing information
- Request/response payloads

**Example:**
```
▶️  Running: Read local://models
   Test reading the models resource
✅ Successfully read models (3 total) (2ms)
   Data: {
     "models": [
       {
         "id": "openai/gpt-oss-20b",
         "name": "Openai Gpt Oss 20b",
         "type": "llm",
         ...
       }
     ]
   }
```

---

## Comparison with Claude Desktop

| Feature | MCP Test Client | Claude Desktop |
|---------|----------------|----------------|
| **Dynamic Testing** | ✅ Yes | ✅ Yes |
| **Interactive Mode** | ✅ REPL | ✅ UI |
| **Test Organization** | ✅ Categories | ❌ No |
| **Automated Suites** | ✅ Yes | ❌ Manual |
| **Verbose Debugging** | ✅ Yes | ⚠️ Limited |
| **Timing Metrics** | ✅ Yes | ❌ No |
| **Batch Execution** | ✅ Yes | ❌ No |

---

## CI/CD Integration

### GitHub Actions

```yaml
name: MCP Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: '18'
      - run: npm install
      - run: npm run test:client:all
```

### Exit Codes

- `0` - All tests passed
- `1` - One or more tests failed or fatal error

---

## Troubleshooting

### Issue: "Not connected to server"

**Solution:** Ensure the server is built before running tests:
```bash
npm run build
npm run test:client:all
```

### Issue: Tests hang indefinitely

**Solution:** Check if LM Studio is running. Some tests may wait for model availability.

**Workaround:** Use categories to skip model-dependent tests:
```bash
> category protocol
> category resources
```

### Issue: "Connection refused"

**Solution:** Server might not be running. The test client starts its own server instance automatically.

---

## Performance

**Test Suite Benchmarks:**

| Category | Tests | Avg Time |
|----------|-------|----------|
| Protocol | 1 | <1ms |
| Resources | 5 | ~10ms |
| Tools | 4 | ~8ms |
| Prompts | 1 | <1ms |
| Integration | 1 | ~6ms |
| **Total** | **12** | **~25ms** |

*Times exclude LM Studio connection overhead*

---

## Architecture

```
MCPTestClient
├── registerTestCases()     - Register all test definitions
├── connect()                - Connect to MCP server
├── disconnect()             - Clean up connection
├── runTest(id)              - Run single test
├── runCategory(category)    - Run category of tests
├── runAll()                 - Run all tests
├── interactive()            - Start REPL mode
└── printSummary()           - Display results
```

### Test Execution Flow

```
1. Client.connect()
   ↓
2. Server handshake (initialize)
   ↓
3. Run test case(s)
   ↓
4. Collect results
   ↓
5. Print summary
   ↓
6. Client.disconnect()
```

---

## Best Practices

### 1. Always Disconnect
```typescript
try {
  await client.connect();
  await client.runAll();
} finally {
  await client.disconnect(); // Always cleanup
}
```

### 2. Check Test Results
```typescript
const result = await client.runTest('tools.list');
if (!result.success) {
  console.error('Test failed:', result.message);
  console.error('Error:', result.error);
}
```

### 3. Use Categories for Focused Testing
```bash
# Test only resources during resource development
npm run test:client

> category resources
```

### 4. Enable Verbose for Debugging
```bash
# When troubleshooting failed tests
npm run test:client:verbose
```

---

## Future Enhancements

Planned features for future versions:

- [ ] HTTP/HTTPS transport support
- [ ] Test result export (JSON, JUnit XML)
- [ ] Performance profiling
- [ ] Parallel test execution
- [ ] Test coverage metrics
- [ ] Custom assertion library
- [ ] Screenshot/recording support
- [ ] Load testing capabilities

---

## Contributing

To add new tests:

1. Open `src/mcp-test-client.ts`
2. Add test in `registerTestCases()` method
3. Follow existing test pattern
4. Build and run: `npm run build && npm run test:client:all`

---

## License

MIT

---

**Version:** 1.0.0
**Last Updated:** 2025-10-06
**Status:** ✅ Production Ready
