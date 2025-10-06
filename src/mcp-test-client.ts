#!/usr/bin/env node

/**
 * Dynamic MCP Test Client
 *
 * A flexible client that approaches the MCP server like Claude Desktop does,
 * allowing dynamic test case execution without precompilation.
 *
 * Usage:
 *   npm run test:client                    # Interactive mode
 *   npm run test:client -- --all           # Run all tests
 *   npm run test:client -- --test=tools    # Run specific test
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import * as readline from 'readline';

interface TestCase {
  id: string;
  name: string;
  description: string;
  category: 'protocol' | 'resources' | 'tools' | 'prompts' | 'integration';
  run: (client: Client) => Promise<TestResult>;
}

interface TestResult {
  success: boolean;
  message: string;
  data?: any;
  error?: any;
  duration?: number;
}

export class MCPTestClient {
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;
  private testCases: TestCase[] = [];
  private results: Map<string, TestResult> = new Map();
  private verbose: boolean = false;

  constructor(verbose: boolean = false) {
    this.verbose = verbose;
    this.registerTestCases();
  }

  /**
   * Register all test cases
   */
  private registerTestCases(): void {
    // Protocol Tests
    this.addTest({
      id: 'protocol.initialize',
      name: 'Protocol Handshake',
      description: 'Test server initialization and capability advertisement',
      category: 'protocol',
      run: async (client) => {
        const start = Date.now();
        try {
          // Client is already initialized, check server info
          const serverInfo = (client as any)._serverCapabilities;

          if (!serverInfo) {
            return {
              success: false,
              message: 'Server info not available',
              duration: Date.now() - start
            };
          }

          return {
            success: true,
            message: 'Server initialized successfully',
            data: serverInfo,
            duration: Date.now() - start
          };
        } catch (error) {
          return {
            success: false,
            message: `Initialization failed: ${(error as Error).message}`,
            error,
            duration: Date.now() - start
          };
        }
      }
    });

    // Resource Tests
    this.addTest({
      id: 'resources.list',
      name: 'List Resources',
      description: 'Test resources/list endpoint',
      category: 'resources',
      run: async (client) => {
        const start = Date.now();
        try {
          const result = await client.listResources();

          const expectedResources = [
            'local://models',
            'local://status',
            'local://config',
            'local://capabilities'
          ];

          const foundResources = result.resources.map(r => r.uri);
          const allFound = expectedResources.every(uri => foundResources.includes(uri));

          return {
            success: allFound && result.resources.length === 4,
            message: allFound
              ? `Found all ${result.resources.length} expected resources`
              : `Missing resources. Expected: ${expectedResources.join(', ')}. Found: ${foundResources.join(', ')}`,
            data: result.resources,
            duration: Date.now() - start
          };
        } catch (error) {
          return {
            success: false,
            message: `List resources failed: ${(error as Error).message}`,
            error,
            duration: Date.now() - start
          };
        }
      }
    });

    this.addTest({
      id: 'resources.read.models',
      name: 'Read local://models',
      description: 'Test reading the models resource',
      category: 'resources',
      run: async (client) => {
        const start = Date.now();
        try {
          const result = await client.readResource({ uri: 'local://models' });

          if (!result.contents || result.contents.length === 0) {
            return {
              success: false,
              message: 'No content returned',
              duration: Date.now() - start
            };
          }

          const content = result.contents[0];
          const data = JSON.parse(content.text as string);

          const hasRequiredFields =
            data.models !== undefined &&
            data.llmModels !== undefined &&
            data.embeddingModels !== undefined &&
            data.defaultModel !== undefined;

          return {
            success: hasRequiredFields,
            message: hasRequiredFields
              ? `Successfully read models (${data.models.length} total)`
              : 'Missing required fields in response',
            data,
            duration: Date.now() - start
          };
        } catch (error) {
          return {
            success: false,
            message: `Read models failed: ${(error as Error).message}`,
            error,
            duration: Date.now() - start
          };
        }
      }
    });

    this.addTest({
      id: 'resources.read.status',
      name: 'Read local://status',
      description: 'Test reading the status resource',
      category: 'resources',
      run: async (client) => {
        const start = Date.now();
        try {
          const result = await client.readResource({ uri: 'local://status' });
          const content = JSON.parse(result.contents[0].text as string);

          return {
            success: content.status !== undefined,
            message: `Server status: ${content.status}`,
            data: content,
            duration: Date.now() - start
          };
        } catch (error) {
          return {
            success: false,
            message: `Read status failed: ${(error as Error).message}`,
            error,
            duration: Date.now() - start
          };
        }
      }
    });

    this.addTest({
      id: 'resources.read.config',
      name: 'Read local://config',
      description: 'Test reading the config resource',
      category: 'resources',
      run: async (client) => {
        const start = Date.now();
        try {
          const result = await client.readResource({ uri: 'local://config' });
          const content = JSON.parse(result.contents[0].text as string);

          const hasRequiredFields =
            content.server !== undefined &&
            content.version !== undefined &&
            content.capabilities !== undefined;

          return {
            success: hasRequiredFields,
            message: `Server: ${content.server} v${content.version}`,
            data: content,
            duration: Date.now() - start
          };
        } catch (error) {
          return {
            success: false,
            message: `Read config failed: ${(error as Error).message}`,
            error,
            duration: Date.now() - start
          };
        }
      }
    });

    this.addTest({
      id: 'resources.read.capabilities',
      name: 'Read local://capabilities',
      description: 'Test reading the capabilities resource',
      category: 'resources',
      run: async (client) => {
        const start = Date.now();
        try {
          const result = await client.readResource({ uri: 'local://capabilities' });
          const content = JSON.parse(result.contents[0].text as string);

          return {
            success: content.tools !== undefined,
            message: `Capabilities documented for ${Object.keys(content.tools || {}).length} tools`,
            data: content,
            duration: Date.now() - start
          };
        } catch (error) {
          return {
            success: false,
            message: `Read capabilities failed: ${(error as Error).message}`,
            error,
            duration: Date.now() - start
          };
        }
      }
    });

    // Tool Tests
    this.addTest({
      id: 'tools.list',
      name: 'List Tools',
      description: 'Test tools/list endpoint',
      category: 'tools',
      run: async (client) => {
        const start = Date.now();
        try {
          const result = await client.listTools();

          const expectedTools = [
            'list_models',
            'get_model_info',
            'local_reasoning',
            'private_analysis',
            'secure_rewrite',
            'code_analysis',
            'template_completion',
            'set_default_model'
          ];

          const foundTools = result.tools.map(t => t.name);
          const allFound = expectedTools.every(name => foundTools.includes(name));

          return {
            success: allFound && result.tools.length === 8,
            message: allFound
              ? `Found all ${result.tools.length} expected tools`
              : `Missing tools. Expected: ${expectedTools.join(', ')}. Found: ${foundTools.join(', ')}`,
            data: result.tools,
            duration: Date.now() - start
          };
        } catch (error) {
          return {
            success: false,
            message: `List tools failed: ${(error as Error).message}`,
            error,
            duration: Date.now() - start
          };
        }
      }
    });

    this.addTest({
      id: 'tools.call.list_models',
      name: 'Call list_models',
      description: 'Test the list_models tool',
      category: 'tools',
      run: async (client) => {
        const start = Date.now();
        try {
          const result = await client.callTool({
            name: 'list_models',
            arguments: { type: 'all' }
          });

          const firstContent: any = (result.content as any)[0];
          if (firstContent.type !== 'text') {
            return {
              success: false,
              message: 'Unexpected response type',
              duration: Date.now() - start
            };
          }

          const data = JSON.parse(firstContent.text);

          return {
            success: data.models !== undefined && data.defaultModel !== undefined,
            message: `Found ${data.models?.length || 0} models`,
            data,
            duration: Date.now() - start
          };
        } catch (error) {
          return {
            success: false,
            message: `list_models call failed: ${(error as Error).message}`,
            error,
            duration: Date.now() - start
          };
        }
      }
    });

    this.addTest({
      id: 'tools.call.get_model_info',
      name: 'Call get_model_info',
      description: 'Test the get_model_info tool',
      category: 'tools',
      run: async (client) => {
        const start = Date.now();
        try {
          // First get available models
          const modelsResult = await client.callTool({
            name: 'list_models',
            arguments: { type: 'llm' }
          });

          const modelsContent: any = (modelsResult.content as any)[0];
          const modelsData = JSON.parse(modelsContent.text);
          if (!modelsData.models || modelsData.models.length === 0) {
            return {
              success: false,
              message: 'No models available to test',
              duration: Date.now() - start
            };
          }

          const firstModel = modelsData.models[0].id;

          // Now get info about first model
          const result = await client.callTool({
            name: 'get_model_info',
            arguments: { model: firstModel }
          });

          const resultContent: any = (result.content as any)[0];
          const data = JSON.parse(resultContent.text);

          return {
            success: data.model !== undefined && data.model.id === firstModel,
            message: `Got info for ${data.model?.displayName || firstModel}`,
            data,
            duration: Date.now() - start
          };
        } catch (error) {
          return {
            success: false,
            message: `get_model_info call failed: ${(error as Error).message}`,
            error,
            duration: Date.now() - start
          };
        }
      }
    });

    this.addTest({
      id: 'tools.error.invalid_model',
      name: 'Test Error Handling',
      description: 'Test structured error response for invalid model',
      category: 'tools',
      run: async (client) => {
        const start = Date.now();
        try {
          const result = await client.callTool({
            name: 'get_model_info',
            arguments: { model: 'nonexistent-model-12345' }
          });

          // Should get an error in the response
          const errorContent: any = (result.content as any)[0];
          const data = JSON.parse(errorContent.text);

          const hasStructuredError =
            data.error !== undefined &&
            data.error.code !== undefined &&
            data.error.message !== undefined &&
            data.error.details !== undefined &&
            data.error.availableModels !== undefined;

          return {
            success: hasStructuredError,
            message: hasStructuredError
              ? `Structured error with code: ${data.error.code}`
              : 'Error response missing required fields',
            data,
            duration: Date.now() - start
          };
        } catch (error) {
          return {
            success: false,
            message: `Error handling test failed: ${(error as Error).message}`,
            error,
            duration: Date.now() - start
          };
        }
      }
    });

    // Prompt Tests
    this.addTest({
      id: 'prompts.list',
      name: 'List Prompts',
      description: 'Test prompts/list endpoint',
      category: 'prompts',
      run: async (client) => {
        const start = Date.now();
        try {
          const result = await client.listPrompts();

          return {
            success: true,
            message: `Found ${result.prompts?.length || 0} prompts`,
            data: result.prompts,
            duration: Date.now() - start
          };
        } catch (error) {
          return {
            success: false,
            message: `List prompts failed: ${(error as Error).message}`,
            error,
            duration: Date.now() - start
          };
        }
      }
    });

    // Integration Tests
    this.addTest({
      id: 'integration.discovery_workflow',
      name: 'Discovery Workflow',
      description: 'Test complete model discovery workflow',
      category: 'integration',
      run: async (client) => {
        const start = Date.now();
        try {
          // Step 1: Read resource
          const resourceResult = await client.readResource({ uri: 'local://models' });
          const resourceData = JSON.parse(resourceResult.contents[0].text as string);

          // Step 2: List models via tool
          const toolResult = await client.callTool({
            name: 'list_models',
            arguments: {}
          });
          const toolContent: any = (toolResult.content as any)[0];
          const toolData = JSON.parse(toolContent.text);

          // Step 3: Verify consistency
          const resourceModelCount = resourceData.models.length;
          const toolModelCount = toolData.models.length;

          return {
            success: resourceModelCount === toolModelCount,
            message: `Resource and tool both report ${resourceModelCount} models`,
            data: {
              resourceCount: resourceModelCount,
              toolCount: toolModelCount,
              defaultModel: resourceData.defaultModel
            },
            duration: Date.now() - start
          };
        } catch (error) {
          return {
            success: false,
            message: `Discovery workflow failed: ${(error as Error).message}`,
            error,
            duration: Date.now() - start
          };
        }
      }
    });
  }

  /**
   * Add a test case
   */
  private addTest(testCase: TestCase): void {
    this.testCases.push(testCase);
  }

  /**
   * Connect to the MCP server
   */
  async connect(command: string = 'node', args: string[] = ['dist/index.js']): Promise<void> {
    this.log('🔌 Connecting to MCP server...');

    this.transport = new StdioClientTransport({
      command,
      args
    });

    this.client = new Client(
      {
        name: 'mcp-test-client',
        version: '1.0.0'
      },
      {
        capabilities: {}
      }
    );

    await this.client.connect(this.transport);
    this.log('✅ Connected to server\n');
  }

  /**
   * Disconnect from the server
   */
  async disconnect(): Promise<void> {
    if (this.client) {
      await this.client.close();
      this.log('👋 Disconnected from server');
    }
  }

  /**
   * Run a specific test
   */
  async runTest(testId: string): Promise<TestResult> {
    const test = this.testCases.find(t => t.id === testId);
    if (!test) {
      return {
        success: false,
        message: `Test ${testId} not found`
      };
    }

    if (!this.client) {
      return {
        success: false,
        message: 'Not connected to server'
      };
    }

    this.log(`\n▶️  Running: ${test.name}`);
    this.log(`   ${test.description}`);

    const result = await test.run(this.client);
    this.results.set(testId, result);

    const statusIcon = result.success ? '✅' : '❌';
    this.log(`${statusIcon} ${result.message} ${result.duration ? `(${result.duration}ms)` : ''}`);

    if (this.verbose && result.data) {
      this.log(`   Data: ${JSON.stringify(result.data, null, 2).substring(0, 200)}...`);
    }

    if (!result.success && result.error && this.verbose) {
      this.log(`   Error: ${result.error}`);
    }

    return result;
  }

  /**
   * Run all tests in a category
   */
  async runCategory(category: string): Promise<Map<string, TestResult>> {
    const tests = this.testCases.filter(t => t.category === category);
    const results = new Map<string, TestResult>();

    this.log(`\n📂 Running ${category} tests (${tests.length} tests)\n`);

    for (const test of tests) {
      const result = await this.runTest(test.id);
      results.set(test.id, result);
    }

    return results;
  }

  /**
   * Run all tests
   */
  async runAll(): Promise<void> {
    this.log('\n🚀 Running all tests\n');

    const categories = ['protocol', 'resources', 'tools', 'prompts', 'integration'];

    for (const category of categories) {
      await this.runCategory(category);
    }

    this.printSummary();
  }

  /**
   * Print test summary
   */
  printSummary(): void {
    const passed = Array.from(this.results.values()).filter(r => r.success).length;
    const failed = this.results.size - passed;

    this.log('\n' + '='.repeat(60));
    this.log('📊 Test Summary');
    this.log('='.repeat(60));
    this.log(`Total:  ${this.results.size}`);
    this.log(`Passed: ${passed} ✅`);
    this.log(`Failed: ${failed} ❌`);
    this.log('='.repeat(60));

    if (failed > 0) {
      this.log('\n❌ Failed Tests:');
      for (const [id, result] of this.results.entries()) {
        if (!result.success) {
          this.log(`   ${id}: ${result.message}`);
        }
      }
    }
  }

  /**
   * Get all test cases
   */
  getTests(): TestCase[] {
    return this.testCases;
  }

  /**
   * Get test by ID
   */
  getTest(id: string): TestCase | undefined {
    return this.testCases.find(t => t.id === id);
  }

  /**
   * Log message
   */
  private log(message: string): void {
    console.log(message);
  }

  /**
   * Interactive mode
   */
  async interactive(): Promise<void> {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    const prompt = (question: string): Promise<string> => {
      return new Promise(resolve => {
        rl.question(question, resolve);
      });
    };

    console.log('\n🎯 MCP Test Client - Interactive Mode\n');
    console.log('Available commands:');
    console.log('  list           - List all tests');
    console.log('  run <id>       - Run a specific test');
    console.log('  category <cat> - Run all tests in category');
    console.log('  all            - Run all tests');
    console.log('  verbose        - Toggle verbose mode');
    console.log('  quit           - Exit\n');

    let running = true;

    while (running) {
      const input = await prompt('> ');
      const [cmd, ...args] = input.trim().split(' ');

      switch (cmd) {
        case 'list':
          console.log('\n📋 Available Tests:\n');
          const categories = [...new Set(this.testCases.map(t => t.category))];
          for (const cat of categories) {
            console.log(`\n${cat.toUpperCase()}:`);
            this.testCases
              .filter(t => t.category === cat)
              .forEach(t => console.log(`  ${t.id} - ${t.name}`));
          }
          console.log('');
          break;

        case 'run':
          if (args.length === 0) {
            console.log('❌ Usage: run <test-id>');
          } else {
            await this.runTest(args[0]);
          }
          break;

        case 'category':
          if (args.length === 0) {
            console.log('❌ Usage: category <category-name>');
          } else {
            await this.runCategory(args[0]);
          }
          break;

        case 'all':
          await this.runAll();
          break;

        case 'verbose':
          this.verbose = !this.verbose;
          console.log(`Verbose mode: ${this.verbose ? 'ON' : 'OFF'}`);
          break;

        case 'quit':
        case 'exit':
          running = false;
          break;

        default:
          console.log('❌ Unknown command. Type "list" to see available tests.');
      }
    }

    rl.close();
  }
}

// CLI Runner
if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2);
  const client = new MCPTestClient(args.includes('--verbose') || args.includes('-v'));

  (async () => {
    try {
      await client.connect();

      if (args.includes('--all')) {
        await client.runAll();
      } else if (args.includes('--interactive') || args.length === 0) {
        await client.interactive();
      } else {
        const testArg = args.find(a => a.startsWith('--test='));
        if (testArg) {
          const testId = testArg.split('=')[1];
          await client.runTest(testId);
          client.printSummary();
        } else {
          console.log('Usage:');
          console.log('  npm run test:client                    # Interactive mode');
          console.log('  npm run test:client -- --all           # Run all tests');
          console.log('  npm run test:client -- --test=<id>     # Run specific test');
          console.log('  npm run test:client -- --verbose       # Verbose output');
        }
      }

      await client.disconnect();
      process.exit(0);
    } catch (error) {
      console.error('❌ Fatal error:', error);
      process.exit(1);
    }
  })();
}
