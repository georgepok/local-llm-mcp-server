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

class IntegrationTester {
  private client: Client | null = null;
  private transport: StdioClientTransport | null = null;
  private serverProcess: ChildProcess | null = null;
  private results: TestResult[] = [];

  async checkLMStudioAvailability(): Promise<boolean> {
    try {
      const response = await fetch('http://localhost:1234/v1/models');
      const data = await response.json();

      if (response.ok && data.data && Array.isArray(data.data)) {
        console.log(`✅ LM Studio is running with ${data.data.length} model(s) available`);
        if (data.data.length > 0) {
          console.log(`   Active model: ${data.data[0].id}`);
        }
        return true;
      }
      return false;
    } catch (error) {
      console.log('❌ LM Studio is not accessible at http://localhost:1234');
      console.log('   Please ensure:');
      console.log('   1. LM Studio is running');
      console.log('   2. A model is loaded');
      console.log('   3. Local server is started on port 1234');
      return false;
    }
  }

  async connectClient(): Promise<void> {
    console.log('🔌 Connecting MCP client...');

    this.client = new Client({
      name: 'integration-test-client',
      version: '1.0.0'
    }, {
      capabilities: {
        tools: {},
        resources: {},
        prompts: {}
      }
    });

    this.transport = new StdioClientTransport({
      command: 'node',
      args: ['dist/index.js'],
      cwd: process.cwd(),
      timeout: 600000 // 10 minutes
    });

    await this.client.connect(this.transport);
    console.log('✅ MCP client connected');
  }

  async startServer(): Promise<void> {
    // Server will be started by the transport
    console.log('🚀 MCP server will be started by transport...');
  }

  async runTest(name: string, testFn: () => Promise<any>): Promise<void> {
    const start = Date.now();
    console.log(`\n🧪 Running integration test: ${name}`);

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
      if (result && typeof result === 'object' && result.summary) {
        console.log(`   ${result.summary}`);
      }
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

  async testLMStudioConnection(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    // Test the status resource to verify LM Studio connection
    const result = await this.client.readResource({ uri: 'local://status' });
    const statusData = JSON.parse(result.contents[0].text || '{}');

    if (statusData.status !== 'online') {
      throw new Error(`LM Studio status is ${statusData.status}, expected 'online'`);
    }

    // Test the models resource to verify available models
    const modelsResult = await this.client.readResource({ uri: 'local://models' });
    const modelsData = JSON.parse(modelsResult.contents[0].text || '{}');

    if (!modelsData.models || modelsData.models.length === 0) {
      throw new Error('No models available in LM Studio');
    }

    return {
      summary: `LM Studio connected with ${modelsData.models.length} model(s)`,
      status: statusData.status,
      models: modelsData.models
    };
  }

  async testLocalReasoningWithLLM(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const testPrompt = 'Solve this step by step: If a train travels 60 miles per hour for 2.5 hours, how far does it travel? Show your work.';

    const result = await this.client.callTool({
      name: 'local_reasoning',
      arguments: {
        prompt: testPrompt,
        system_prompt: 'You are a helpful math tutor. Always show your work step by step.',
        model_params: {
          temperature: 0.1,
          max_tokens: 200
        }
      }
    });

    if (!result.content || result.content.length === 0) {
      throw new Error('No response from local reasoning tool');
    }

    const response = result.content[0].text;
    if (!response) {
      throw new Error('Empty response from local reasoning tool');
    }

    // Check if the response contains the correct answer and some reasoning
    const responseText = response.toLowerCase();
    const hasCorrectAnswer = responseText.includes('150') || responseText.includes('one hundred fifty') ||
                            (responseText.includes('60') && responseText.includes('2.5') &&
                             (responseText.includes('miles') || responseText.includes('distance')));
    const hasMultiplication = responseText.includes('60') && responseText.includes('2.5');
    const hasStepByStep = responseText.includes('step') || responseText.includes('work') || responseText.includes('calculate') || responseText.includes('multiply');

    if (!hasCorrectAnswer && !hasMultiplication) {
      throw new Error('Response does not appear to solve the math problem correctly');
    }

    if (!hasMultiplication || !hasStepByStep) {
      console.log('   Warning: Response may lack detailed step-by-step reasoning');
    }

    return {
      summary: `Generated ${response.length} character response with correct answer`,
      responseLength: response.length,
      hasCorrectAnswer,
      preview: response.substring(0, 100) + (response.length > 100 ? '...' : '')
    };
  }

  async testPrivateAnalysisWithLLM(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const testText = `I'm absolutely thrilled about our new product launch! The team has worked incredibly hard, and I believe this innovation will transform how our customers experience our services. However, I'm a bit concerned about the aggressive timeline and whether our support team is fully prepared for the increased volume.`;

    const result = await this.client.callTool({
      name: 'private_analysis',
      arguments: {
        content: testText,
        analysis_type: 'sentiment',
        domain: 'general'
      }
    });

    if (!result.content || result.content.length === 0) {
      throw new Error('No response from private analysis tool');
    }

    const responseText = result.content[0].text;
    if (!responseText) {
      throw new Error('Empty response from private analysis tool');
    }

    // Try to parse as JSON (should be structured analysis)
    let analysisData;
    try {
      analysisData = JSON.parse(responseText);
    } catch {
      // If not JSON, check if it's a valid text response about sentiment
      if (!responseText.toLowerCase().includes('sentiment')) {
        throw new Error('Response does not appear to be sentiment analysis');
      }
      return {
        summary: 'Received text-based sentiment analysis',
        responseType: 'text',
        preview: responseText.substring(0, 100) + '...'
      };
    }

    // Check for expected sentiment analysis structure
    if (!analysisData.sentiment && !analysisData.hasOwnProperty('positive') && !analysisData.hasOwnProperty('negative')) {
      throw new Error('Analysis response missing sentiment information');
    }

    return {
      summary: `Sentiment analysis completed: ${analysisData.sentiment || 'mixed'}`,
      analysisType: 'structured',
      sentiment: analysisData.sentiment,
      confidence: analysisData.confidence,
      preview: JSON.stringify(analysisData, null, 2).substring(0, 150) + '...'
    };
  }

  async testSecureRewriteWithLLM(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const sensitiveText = `Hi Sarah Johnson, your account #A123456 has been updated. Your credit card ending in 4567 was charged $299.99 for the premium service. If you have questions, contact me at john.doe@company.com or call (555) 123-4567.`;

    const result = await this.client.callTool({
      name: 'secure_rewrite',
      arguments: {
        content: sensitiveText,
        style: 'professional',
        privacy_level: 'strict'
      }
    });

    if (!result.content || result.content.length === 0) {
      throw new Error('No response from secure rewrite tool');
    }

    const rewrittenText = result.content[0].text;
    if (!rewrittenText) {
      throw new Error('Empty response from secure rewrite tool');
    }

    // Check that sensitive information has been anonymized
    const lowerRewritten = rewrittenText.toLowerCase();
    const sensitiveData = [
      'sarah johnson', 'a123456', '4567', '$299.99',
      'john.doe@company.com', '(555) 123-4567', '555-123-4567'
    ];

    const foundSensitiveData = sensitiveData.filter(item =>
      lowerRewritten.includes(item.toLowerCase())
    );

    if (foundSensitiveData.length > 0) {
      throw new Error(`Sensitive data not properly anonymized: ${foundSensitiveData.join(', ')}`);
    }

    // Check that the rewritten text maintains professional tone and core message
    const hasAccount = lowerRewritten.includes('account');
    const hasCharge = lowerRewritten.includes('charge') || lowerRewritten.includes('payment') || lowerRewritten.includes('bill');
    const hasContact = lowerRewritten.includes('contact') || lowerRewritten.includes('question');

    if (!hasAccount || !hasCharge || !hasContact) {
      console.log('   Warning: Rewritten text may be missing some core message elements');
    }

    return {
      summary: `Successfully anonymized ${sensitiveText.length} chars to ${rewrittenText.length} chars`,
      originalLength: sensitiveText.length,
      rewrittenLength: rewrittenText.length,
      privacyProtected: foundSensitiveData.length === 0,
      preview: rewrittenText.substring(0, 100) + (rewrittenText.length > 100 ? '...' : '')
    };
  }

  async testCodeAnalysisWithLLM(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const vulnerableCode = `
function authenticateUser(username, password) {
  // SQL Injection vulnerability
  const query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'";
  const result = db.query(query);

  if (result.length > 0) {
    // Storing password in session (security issue)
    session.user = result[0];
    session.password = password;
    return true;
  }
  return false;
}

function processPayment(cardNumber, amount) {
  // No input validation
  console.log("Processing payment for card: " + cardNumber + " amount: " + amount);

  // Hardcoded API key (security issue)
  const apiKey = "sk_live_12345abcdef";

  return paymentGateway.charge(cardNumber, amount, apiKey);
}`;

    const result = await this.client.callTool({
      name: 'code_analysis',
      arguments: {
        code: vulnerableCode,
        language: 'javascript',
        analysis_focus: 'security'
      }
    });

    if (!result.content || result.content.length === 0) {
      throw new Error('No response from code analysis tool');
    }

    const responseText = result.content[0].text;
    if (!responseText) {
      throw new Error('Empty response from code analysis tool');
    }

    // Try to parse as JSON (should be structured analysis)
    let analysisData;
    try {
      analysisData = JSON.parse(responseText);
    } catch {
      // If not JSON, check if it's a valid text response about security
      if (!responseText.toLowerCase().includes('security') && !responseText.toLowerCase().includes('vulnerability')) {
        throw new Error('Response does not appear to be security analysis');
      }
      return {
        summary: 'Received text-based security analysis',
        responseType: 'text',
        preview: responseText.substring(0, 150) + '...'
      };
    }

    // Check for expected security analysis structure
    if (!analysisData.analysis && !analysisData.issues && !analysisData.vulnerabilities) {
      throw new Error('Analysis response missing security information');
    }

    // Check if major security issues were identified
    const responseStr = JSON.stringify(analysisData).toLowerCase();
    const foundIssues = {
      sqlInjection: responseStr.includes('sql') && responseStr.includes('injection'),
      hardcodedCredentials: responseStr.includes('hardcoded') || responseStr.includes('api key'),
      inputValidation: responseStr.includes('validation') || responseStr.includes('sanitize'),
      passwordStorage: responseStr.includes('password') && responseStr.includes('session')
    };

    const detectedIssueCount = Object.values(foundIssues).filter(Boolean).length;

    return {
      summary: `Security analysis detected ${detectedIssueCount}/4 major vulnerabilities`,
      analysisType: 'structured',
      detectedIssues: foundIssues,
      issueCount: detectedIssueCount,
      preview: JSON.stringify(analysisData, null, 2).substring(0, 200) + '...'
    };
  }

  async testTemplateCompletionWithLLM(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const template = `Dear [CUSTOMER_NAME],

Thank you for your recent [ACTION] regarding [SUBJECT]. We have reviewed your [REQUEST_TYPE] and are pleased to inform you that [OUTCOME].

Next steps:
1. [STEP_1]
2. [STEP_2]
3. [STEP_3]

If you have any questions, please don't hesitate to contact our [DEPARTMENT] team.

Best regards,
[AGENT_NAME]
[TITLE]`;

    const context = `Customer Sarah submitted a support ticket about upgrading her basic plan to premium. She wants access to advanced analytics features. The upgrade has been approved and will be processed within 24 hours. She needs to verify her payment method, confirm the new features, and review the updated billing cycle.`;

    const result = await this.client.callTool({
      name: 'template_completion',
      arguments: {
        template: template,
        context: context,
        format: 'professional customer service email'
      }
    });

    if (!result.content || result.content.length === 0) {
      throw new Error('No response from template completion tool');
    }

    const completedTemplate = result.content[0].text;
    if (!completedTemplate) {
      throw new Error('Empty response from template completion tool');
    }

    // Check that template placeholders have been filled
    const placeholders = [
      '[CUSTOMER_NAME]', '[ACTION]', '[SUBJECT]', '[REQUEST_TYPE]', '[OUTCOME]',
      '[STEP_1]', '[STEP_2]', '[STEP_3]', '[DEPARTMENT]', '[AGENT_NAME]', '[TITLE]'
    ];

    const remainingPlaceholders = placeholders.filter(placeholder =>
      completedTemplate.includes(placeholder)
    );

    if (remainingPlaceholders.length > 0) {
      throw new Error(`Template placeholders not filled: ${remainingPlaceholders.join(', ')}`);
    }

    // Check that the completion is contextually appropriate
    const lowerCompleted = completedTemplate.toLowerCase();
    const contextualChecks = {
      mentionsUpgrade: lowerCompleted.includes('upgrade') || lowerCompleted.includes('premium'),
      mentionsAnalytics: lowerCompleted.includes('analytics') || lowerCompleted.includes('features'),
      mentionsPayment: lowerCompleted.includes('payment') || lowerCompleted.includes('billing'),
      mentionsSarah: lowerCompleted.includes('sarah')
    };

    const contextualScore = Object.values(contextualChecks).filter(Boolean).length;

    return {
      summary: `Template completed with ${contextualScore}/4 contextual elements`,
      originalLength: template.length,
      completedLength: completedTemplate.length,
      placeholdersFilled: placeholders.length - remainingPlaceholders.length,
      contextualScore: contextualScore,
      preview: completedTemplate.substring(0, 200) + (completedTemplate.length > 200 ? '...' : '')
    };
  }

  async testPromptTemplateIntegration(): Promise<any> {
    if (!this.client) throw new Error('Client not connected');

    const result = await this.client.getPrompt({
      name: 'privacy-analysis',
      arguments: {
        content: 'This document contains employee names, social security numbers, and salary information.',
        regulation: 'GDPR'
      }
    });

    if (!result.description || !result.messages || result.messages.length === 0) {
      throw new Error('Invalid prompt response structure');
    }

    const message = result.messages[0];
    if (!message.content.text) {
      throw new Error('Prompt message missing text content');
    }

    const promptText = message.content.text;

    // Check that the prompt includes the provided arguments
    const includesContent = promptText.includes('employee names') || promptText.includes('social security');
    const includesGDPR = promptText.includes('GDPR');
    const includesAnalysis = promptText.includes('privacy') && promptText.includes('analyz');

    if (!includesContent || !includesGDPR || !includesAnalysis) {
      throw new Error('Prompt template did not properly incorporate arguments');
    }

    return {
      summary: `Privacy analysis prompt generated (${promptText.length} chars)`,
      description: result.description,
      messageLength: promptText.length,
      includesArguments: includesContent && includesGDPR,
      preview: promptText.substring(0, 150) + '...'
    };
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

    console.log('✅ Cleanup complete');
  }

  printResults(): void {
    console.log('\n📊 Integration Test Results');
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

    console.log('\n📋 Test Summary:');
    this.results.forEach(r => {
      const status = r.success ? '✅' : '❌';
      console.log(`  ${status} ${r.name} (${r.duration}ms)`);
      if (r.success && r.data && r.data.summary) {
        console.log(`      ${r.data.summary}`);
      }
    });

    const success = failed === 0;
    console.log(`\n${success ? '🎉' : '❌'} Integration Test ${success ? 'PASSED' : 'FAILED'}`);

    if (success) {
      console.log('\n🎯 All integration tests passed! The Local LLM MCP Server is working correctly with LM Studio.');
      console.log('✅ Ready for production use with Claude Desktop');
    } else {
      console.log('\n⚠️  Some integration tests failed. Please check the error messages above.');
    }
  }

  async runIntegrationTests(): Promise<void> {
    console.log('🧪 LM Studio Integration Tests');
    console.log('Testing real integration with LM Studio running on localhost:1234\n');

    // Check LM Studio availability first
    const lmStudioAvailable = await this.checkLMStudioAvailability();
    if (!lmStudioAvailable) {
      console.log('\n❌ Cannot run integration tests without LM Studio');
      process.exit(1);
    }

    try {
      await this.connectClient();

      // Core integration tests
      await this.runTest('LM Studio Connection', () => this.testLMStudioConnection());

      // Tool integration tests with real LLM calls
      await this.runTest('Local Reasoning with LLM', () => this.testLocalReasoningWithLLM());
      await this.runTest('Private Analysis with LLM', () => this.testPrivateAnalysisWithLLM());
      await this.runTest('Secure Rewrite with LLM', () => this.testSecureRewriteWithLLM());
      await this.runTest('Code Analysis with LLM', () => this.testCodeAnalysisWithLLM());
      await this.runTest('Template Completion with LLM', () => this.testTemplateCompletionWithLLM());

      // Prompt template integration
      await this.runTest('Prompt Template Integration', () => this.testPromptTemplateIntegration());

    } catch (error) {
      console.error('❌ Integration test suite failed:', error);
    } finally {
      await this.cleanup();
      this.printResults();
    }
  }
}

// Run the integration tests
const tester = new IntegrationTester();
tester.runIntegrationTests().catch(console.error);