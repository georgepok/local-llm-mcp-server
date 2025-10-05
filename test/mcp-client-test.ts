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
  data?: any;
}

class MCPClientTester {
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

    // Give the server a moment to start
    await setTimeout(1000);

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
      name: 'test-client',
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
      const result = await testFn();
      const duration = Date.now() - start;

      this.results.push({
        name,
        success: true,
        duration,
        data: result
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

  async testListTools(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.listTools();

    if (!result.tools || result.tools.length === 0) {
      throw new Error('No tools returned');
    }

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

    return { toolCount: result.tools.length, tools: toolNames };
  }

  async testListResources(): Promise<any> {
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

    return { resourceCount: result.resources.length, resources: resourceUris };
  }

  async testListPrompts(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.listPrompts();

    if (!result.prompts || result.prompts.length === 0) {
      throw new Error('No prompts returned');
    }

    const expectedPrompts = [
      'privacy-analysis',
      'secure-rewrite',
      'code-security-review',
      'meeting-summary'
    ];

    const promptNames = result.prompts.map(prompt => prompt.name);

    for (const expected of expectedPrompts) {
      if (!promptNames.includes(expected)) {
        throw new Error(`Missing expected prompt: ${expected}`);
      }
    }

    return { promptCount: result.prompts.length, prompts: promptNames };
  }

  async testReadStatusResource(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.readResource({ uri: 'local://status' });

    if (!result.contents || result.contents.length === 0) {
      throw new Error('No status content returned');
    }

    const content = result.contents[0];
    if (content.mimeType !== 'application/json') {
      throw new Error('Expected JSON content type');
    }

    const statusData = JSON.parse(content.text || '{}');

    if (!statusData.hasOwnProperty('status')) {
      throw new Error('Status data missing status field');
    }

    return statusData;
  }

  async testReadModelsResource(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.readResource({ uri: 'local://models' });

    if (!result.contents || result.contents.length === 0) {
      throw new Error('No models content returned');
    }

    const content = result.contents[0];
    const modelsData = JSON.parse(content.text || '{}');

    if (!modelsData.hasOwnProperty('models') || !modelsData.hasOwnProperty('count')) {
      throw new Error('Models data missing required fields');
    }

    return modelsData;
  }

  async testReadConfigResource(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.readResource({ uri: 'local://config' });

    if (!result.contents || result.contents.length === 0) {
      throw new Error('No config content returned');
    }

    const content = result.contents[0];
    const configData = JSON.parse(content.text || '{}');

    const expectedFields = ['server', 'version', 'capabilities', 'privacy_levels', 'supported_domains'];

    for (const field of expectedFields) {
      if (!configData.hasOwnProperty(field)) {
        throw new Error(`Config data missing field: ${field}`);
      }
    }

    return configData;
  }

  async testLocalReasoningTool(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.callTool({
      name: 'local_reasoning',
      arguments: {
        prompt: 'What is 2 + 2? Explain your reasoning.',
        system_prompt: 'You are a helpful math tutor.',
        model_params: {
          temperature: 0.1,
          max_tokens: 100
        }
      }
    });

    if (!result.content || result.content.length === 0) {
      throw new Error('No content returned from local_reasoning tool');
    }

    const content = result.content[0];
    if (content.type !== 'text' || !content.text) {
      throw new Error('Expected text content from local_reasoning tool');
    }

    // Check if response contains mathematical reasoning
    const responseText = content.text.toLowerCase();
    if (!responseText.includes('4') && !responseText.includes('four')) {
      throw new Error('Response does not contain expected answer');
    }

    return { responseLength: content.text.length, preview: content.text.substring(0, 100) + '...' };
  }

  async testPrivateAnalysisTool(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.callTool({
      name: 'private_analysis',
      arguments: {
        content: 'I am very excited about this new product launch! It will revolutionize our industry.',
        analysis_type: 'sentiment',
        domain: 'general'
      }
    });

    if (!result.content || result.content.length === 0) {
      throw new Error('No content returned from private_analysis tool');
    }

    const content = result.content[0];
    if (content.type !== 'text' || !content.text) {
      throw new Error('Expected text content from private_analysis tool');
    }

    // Try to parse as JSON (should be structured analysis result)
    let analysisData;
    try {
      analysisData = JSON.parse(content.text);
    } catch {
      throw new Error('Analysis result is not valid JSON');
    }

    // Check for expected sentiment analysis fields
    if (!analysisData.hasOwnProperty('sentiment')) {
      throw new Error('Analysis result missing sentiment field');
    }

    return { analysis: analysisData };
  }

  async testSecureRewriteTool(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.callTool({
      name: 'secure_rewrite',
      arguments: {
        content: 'Hi John Smith, your account number 12345 has been updated. Contact us at support@company.com if you have questions.',
        style: 'professional',
        privacy_level: 'strict'
      }
    });

    if (!result.content || result.content.length === 0) {
      throw new Error('No content returned from secure_rewrite tool');
    }

    const content = result.content[0];
    if (content.type !== 'text' || !content.text) {
      throw new Error('Expected text content from secure_rewrite tool');
    }

    // Check that personal information has been anonymized
    const rewrittenText = content.text.toLowerCase();
    if (rewrittenText.includes('john smith') || rewrittenText.includes('12345') || rewrittenText.includes('support@company.com')) {
      throw new Error('Personal information not properly anonymized');
    }

    return { originalLength: 107, rewrittenLength: content.text.length, preview: content.text };
  }

  async testCodeAnalysisTool(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const testCode = `
function authenticateUser(username, password) {
  const query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'";
  return db.query(query);
}`;

    const result = await this.client.callTool({
      name: 'code_analysis',
      arguments: {
        code: testCode,
        language: 'javascript',
        analysis_focus: 'security'
      }
    });

    if (!result.content || result.content.length === 0) {
      throw new Error('No content returned from code_analysis tool');
    }

    const content = result.content[0];
    if (content.type !== 'text' || !content.text) {
      throw new Error('Expected text content from code_analysis tool');
    }

    // Try to parse as JSON (should be structured analysis result)
    let analysisData;
    try {
      analysisData = JSON.parse(content.text);
    } catch {
      throw new Error('Code analysis result is not valid JSON');
    }

    // Check for expected code analysis fields
    const expectedFields = ['analysis', 'issues', 'recommendations'];
    for (const field of expectedFields) {
      if (!analysisData.hasOwnProperty(field)) {
        throw new Error(`Code analysis result missing field: ${field}`);
      }
    }

    return { analysis: analysisData };
  }

  async testTemplateCompletionTool(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.callTool({
      name: 'template_completion',
      arguments: {
        template: 'Dear [NAME], Thank you for [ACTION]. We appreciate your [QUALITY] and look forward to [FUTURE].',
        context: 'Customer submitted a support ticket about billing and was very patient during the resolution process',
        format: 'professional email'
      }
    });

    if (!result.content || result.content.length === 0) {
      throw new Error('No content returned from template_completion tool');
    }

    const content = result.content[0];
    if (content.type !== 'text' || !content.text) {
      throw new Error('Expected text content from template_completion tool');
    }

    // Check that template placeholders have been filled
    const completedText = content.text;
    if (completedText.includes('[NAME]') || completedText.includes('[ACTION]') ||
        completedText.includes('[QUALITY]') || completedText.includes('[FUTURE]')) {
      throw new Error('Template placeholders not properly filled');
    }

    return { completedTemplate: completedText };
  }

  async testGetPrompt(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.getPrompt({
      name: 'privacy-analysis',
      arguments: {
        content: 'This document contains personal information about employees.',
        regulation: 'GDPR'
      }
    });

    if (!result.description || !result.messages) {
      throw new Error('Invalid prompt response structure');
    }

    if (result.messages.length === 0) {
      throw new Error('No messages in prompt response');
    }

    const message = result.messages[0];
    if (message.role !== 'user' || !message.content.text) {
      throw new Error('Invalid prompt message structure');
    }

    return {
      description: result.description,
      messageLength: message.content.text.length,
      preview: message.content.text.substring(0, 100) + '...'
    };
  }

  async testErrorHandling(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    try {
      // Test with invalid tool name
      await this.client.callTool({
        name: 'invalid_tool',
        arguments: {}
      });
      throw new Error('Should have thrown error for invalid tool');
    } catch (error) {
      // Expected error
    }

    try {
      // Test with missing required arguments
      await this.client.callTool({
        name: 'local_reasoning',
        arguments: {}
      });
      throw new Error('Should have thrown error for missing arguments');
    } catch (error) {
      // Expected error
    }

    try {
      // Test with invalid resource URI
      await this.client.readResource({ uri: 'invalid://resource' });
      throw new Error('Should have thrown error for invalid resource');
    } catch (error) {
      // Expected error
    }

    return { message: 'Error handling tests passed' };
  }

  async cleanup(): Promise<void> {
    console.log('\n🧹 Cleaning up...');

    if (this.client) {
      await this.client.close();
    }

    if (this.serverProcess) {
      this.serverProcess.kill();
      await setTimeout(1000); // Give it time to shut down
    }

    console.log('✅ Cleanup complete');
  }

  printResults(): void {
    console.log('\n📊 Test Results Summary');
    console.log('=' .repeat(50));

    const passed = this.results.filter(r => r.success).length;
    const failed = this.results.filter(r => !r.success).length;
    const totalTime = this.results.reduce((sum, r) => sum + r.duration, 0);

    console.log(`Total Tests: ${this.results.length}`);
    console.log(`Passed: ${passed} ✅`);
    console.log(`Failed: ${failed} ❌`);
    console.log(`Total Time: ${totalTime}ms`);
    console.log(`Success Rate: ${((passed / this.results.length) * 100).toFixed(1)}%`);

    if (failed > 0) {
      console.log('\n❌ Failed Tests:');
      this.results
        .filter(r => !r.success)
        .forEach(r => console.log(`  - ${r.name}: ${r.error}`));
    }

    console.log('\n📋 Detailed Results:');
    this.results.forEach(r => {
      const status = r.success ? '✅' : '❌';
      console.log(`  ${status} ${r.name} (${r.duration}ms)`);

      if (r.success && r.data) {
        console.log(`    Data: ${JSON.stringify(r.data, null, 2).substring(0, 100)}...`);
      }
    });
  }

  async runAllTests(): Promise<void> {
    try {
      await this.startServer();
      await this.connectClient();

      // Core functionality tests
      await this.runTest('List Tools', () => this.testListTools());
      await this.runTest('List Resources', () => this.testListResources());
      await this.runTest('List Prompts', () => this.testListPrompts());

      // Resource tests
      await this.runTest('Read Status Resource', () => this.testReadStatusResource());
      await this.runTest('Read Models Resource', () => this.testReadModelsResource());
      await this.runTest('Read Config Resource', () => this.testReadConfigResource());

      // Tool tests
      await this.runTest('Local Reasoning Tool', () => this.testLocalReasoningTool());
      await this.runTest('Private Analysis Tool', () => this.testPrivateAnalysisTool());
      await this.runTest('Secure Rewrite Tool', () => this.testSecureRewriteTool());
      await this.runTest('Code Analysis Tool', () => this.testCodeAnalysisTool());
      await this.runTest('Template Completion Tool', () => this.testTemplateCompletionTool());

      // Prompt tests
      await this.runTest('Get Prompt', () => this.testGetPrompt());

      // Error handling tests
      await this.runTest('Error Handling', () => this.testErrorHandling());

    } catch (error) {
      console.error('❌ Test suite failed:', error);
    } finally {
      await this.cleanup();
      this.printResults();
    }
  }
}

// Run the tests
const tester = new MCPClientTester();
tester.runAllTests().catch(console.error);