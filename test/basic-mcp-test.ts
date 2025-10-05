#!/usr/bin/env node

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { spawn, ChildProcess } from 'child_process';
import { setTimeout } from 'timers/promises';

interface TestResult {
  name: string;
  success: boolean;
  error?: string;
  duration: number;
}

class BasicMCPTester {
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;
  private serverProcess: ChildProcess | null = null;
  private results: TestResult[] = [];

  async startServer(): Promise<void> {
    console.log('🚀 Starting MCP server...');

    this.serverProcess = spawn('node', ['dist/index.js'], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: process.cwd()
    });

    // Monitor stderr for any startup errors
    this.serverProcess.stderr?.on('data', (data) => {
      console.log('Server stderr:', data.toString());
    });

    // Give the server a moment to start
    await setTimeout(2000);

    if (!this.serverProcess.stdout || !this.serverProcess.stdin) {
      throw new Error('Failed to get server stdio streams');
    }

    console.log('✅ MCP server started');
  }

  async connectClient(): Promise<void> {
    console.log('🔌 Connecting MCP client...');

    if (!this.serverProcess?.stdout || !this.serverProcess?.stdin) {
      throw new Error('Server process not available');
    }

    this.transport = new StdioClientTransport({
      reader: this.serverProcess.stdout,
      writer: this.serverProcess.stdin
    });

    this.client = new Client({
      name: 'basic-test-client',
      version: '1.0.0'
    }, {
      capabilities: {
        tools: {}
      }
    });

    await this.client.connect(this.transport);
    console.log('✅ MCP client connected');
  }

  async runTest(name: string, testFn: () => Promise<any>): Promise<void> {
    const start = Date.now();
    console.log(`\n🧪 Running test: ${name}`);

    try {
      await testFn();
      const duration = Date.now() - start;

      this.results.push({
        name,
        success: true,
        duration
      });

      console.log(`✅ ${name} - PASSED (${duration}ms)`);
    } catch (error) {
      const duration = Date.now() - start;
      const errorMessage = error instanceof Error ? error.message : String(error);

      this.results.push({
        name,
        success: false,
        error: errorMessage,
        duration
      });

      console.log(`❌ ${name} - FAILED (${duration}ms)`);
      console.log(`   Error: ${errorMessage}`);
    }
  }

  async testServerInitialization(): Promise<void> {
    if (!this.client) throw new Error('Client not connected');

    // Just test that we can communicate with the server
    const result = await this.client.listTools();

    if (!result.tools || result.tools.length === 0) {
      throw new Error('Server did not return any tools');
    }

    console.log(`   Found ${result.tools.length} tools`);
  }

  async testToolDiscovery(): Promise<void> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.listTools();

    const expectedTools = [
      'local_reasoning',
      'private_analysis',
      'secure_rewrite',
      'code_analysis',
      'template_completion'
    ];

    const toolNames = result.tools.map(tool => tool.name);

    for (const expected of expectedTools) {
      if (!toolNames.includes(expected)) {
        throw new Error(`Missing expected tool: ${expected}`);
      }
    }

    console.log(`   All ${expectedTools.length} expected tools found`);
  }

  async testResourceDiscovery(): Promise<void> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.listResources();

    if (!result.resources || result.resources.length === 0) {
      throw new Error('No resources returned');
    }

    const expectedResources = [
      'local://models',
      'local://status',
      'local://config'
    ];

    const resourceUris = result.resources.map(resource => resource.uri);

    for (const expected of expectedResources) {
      if (!resourceUris.includes(expected)) {
        throw new Error(`Missing expected resource: ${expected}`);
      }
    }

    console.log(`   All ${expectedResources.length} expected resources found`);
  }

  async testPromptDiscovery(): Promise<void> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.listPrompts();

    if (!result.prompts || result.prompts.length === 0) {
      throw new Error('No prompts returned');
    }

    console.log(`   Found ${result.prompts.length} prompt templates`);

    // Check for a few key prompts
    const promptNames = result.prompts.map(prompt => prompt.name);
    const keyPrompts = ['privacy-analysis', 'secure-rewrite', 'code-security-review'];

    for (const key of keyPrompts) {
      if (!promptNames.includes(key)) {
        throw new Error(`Missing key prompt: ${key}`);
      }
    }
  }

  async testBasicResourceAccess(): Promise<void> {
    if (!this.client) throw new Error('Client not connected');

    // Test config resource (should always work)
    const configResult = await this.client.readResource({ uri: 'local://config' });

    if (!configResult.contents || configResult.contents.length === 0) {
      throw new Error('No config content returned');
    }

    const configContent = configResult.contents[0];
    if (configContent.mimeType !== 'application/json') {
      throw new Error('Config content is not JSON');
    }

    const configData = JSON.parse(configContent.text || '{}');

    const requiredFields = ['server', 'version', 'capabilities'];
    for (const field of requiredFields) {
      if (!configData.hasOwnProperty(field)) {
        throw new Error(`Config missing required field: ${field}`);
      }
    }

    console.log(`   Config resource accessible, server: ${configData.server} v${configData.version}`);
  }

  async testStatusResource(): Promise<void> {
    if (!this.client) throw new Error('Client not connected');

    const statusResult = await this.client.readResource({ uri: 'local://status' });

    if (!statusResult.contents || statusResult.contents.length === 0) {
      throw new Error('No status content returned');
    }

    const statusContent = statusResult.contents[0];
    const statusData = JSON.parse(statusContent.text || '{}');

    if (!statusData.hasOwnProperty('status')) {
      throw new Error('Status data missing status field');
    }

    console.log(`   LM Studio status: ${statusData.status}`);

    // Note: This might be "offline" if LM Studio isn't running, which is OK for basic test
  }

  async testToolSchema(): Promise<void> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.listTools();

    // Test that each tool has proper schema
    for (const tool of result.tools) {
      if (!tool.name) {
        throw new Error('Tool missing name');
      }

      if (!tool.description) {
        throw new Error(`Tool ${tool.name} missing description`);
      }

      if (!tool.inputSchema) {
        throw new Error(`Tool ${tool.name} missing input schema`);
      }

      if (tool.inputSchema.type !== 'object') {
        throw new Error(`Tool ${tool.name} schema is not object type`);
      }

      if (!tool.inputSchema.properties) {
        throw new Error(`Tool ${tool.name} schema missing properties`);
      }

      if (!tool.inputSchema.required || !Array.isArray(tool.inputSchema.required)) {
        throw new Error(`Tool ${tool.name} schema missing or invalid required array`);
      }
    }

    console.log(`   All ${result.tools.length} tools have valid schemas`);
  }

  async testPromptSchema(): Promise<void> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.listPrompts();

    // Test that each prompt has proper schema
    for (const prompt of result.prompts) {
      if (!prompt.name) {
        throw new Error('Prompt missing name');
      }

      if (!prompt.description) {
        throw new Error(`Prompt ${prompt.name} missing description`);
      }

      if (prompt.arguments) {
        if (!Array.isArray(prompt.arguments)) {
          throw new Error(`Prompt ${prompt.name} arguments is not an array`);
        }

        for (const arg of prompt.arguments) {
          if (!arg.name) {
            throw new Error(`Prompt ${prompt.name} has argument missing name`);
          }

          if (typeof arg.required !== 'boolean') {
            throw new Error(`Prompt ${prompt.name} argument ${arg.name} missing required field`);
          }
        }
      }
    }

    console.log(`   All ${result.prompts.length} prompts have valid schemas`);
  }

  async cleanup(): Promise<void> {
    console.log('\n🧹 Cleaning up...');

    if (this.client) {
      try {
        await this.client.close();
      } catch (error) {
        console.log('   Error closing client:', error);
      }
    }

    if (this.serverProcess) {
      this.serverProcess.kill('SIGTERM');
      await setTimeout(2000);

      if (!this.serverProcess.killed) {
        this.serverProcess.kill('SIGKILL');
      }
    }

    console.log('✅ Cleanup complete');
  }

  printResults(): void {
    console.log('\n📊 Basic Test Results');
    console.log('=' .repeat(40));

    const passed = this.results.filter(r => r.success).length;
    const failed = this.results.filter(r => !r.success).length;

    console.log(`Total Tests: ${this.results.length}`);
    console.log(`Passed: ${passed} ✅`);
    console.log(`Failed: ${failed} ❌`);

    if (failed > 0) {
      console.log('\n❌ Failed Tests:');
      this.results
        .filter(r => !r.success)
        .forEach(r => console.log(`  - ${r.name}: ${r.error}`));
    }

    const success = failed === 0;
    console.log(`\n${success ? '🎉' : '❌'} MCP Server ${success ? 'READY' : 'NEEDS ATTENTION'}`);

    if (success) {
      console.log('\nThe MCP server is properly configured and ready for use with Claude Desktop!');
      console.log('\nNext steps:');
      console.log('1. Ensure LM Studio is running on port 1234');
      console.log('2. Restart Claude Desktop to load the MCP server');
      console.log('3. Test the tools in Claude Desktop');
    }
  }

  async runBasicTests(): Promise<void> {
    try {
      await this.startServer();
      await this.connectClient();

      // Core discovery tests
      await this.runTest('Server Initialization', () => this.testServerInitialization());
      await this.runTest('Tool Discovery', () => this.testToolDiscovery());
      await this.runTest('Resource Discovery', () => this.testResourceDiscovery());
      await this.runTest('Prompt Discovery', () => this.testPromptDiscovery());

      // Schema validation tests
      await this.runTest('Tool Schema Validation', () => this.testToolSchema());
      await this.runTest('Prompt Schema Validation', () => this.testPromptSchema());

      // Basic resource access tests
      await this.runTest('Config Resource Access', () => this.testBasicResourceAccess());
      await this.runTest('Status Resource Access', () => this.testStatusResource());

    } catch (error) {
      console.error('❌ Test suite failed:', error);
    } finally {
      await this.cleanup();
      this.printResults();
    }
  }
}

// Run the basic tests
console.log('🚀 Starting Basic MCP Server Tests');
console.log('This will test core MCP functionality without requiring LM Studio\n');

const tester = new BasicMCPTester();
tester.runBasicTests().catch(console.error);