#!/usr/bin/env node

/**
 * Validation Tests for Security and Metadata Fixes
 *
 * This test suite validates the specific fixes made in the security audit:
 * 1. HIGH - Privacy leak in secureRewrite (duplicate sensitive data)
 * 2. MEDIUM - Correct previousDefault capture in set_default_model
 * 3. MEDIUM - Metadata fields (publisher, quantization vs created/ownedBy)
 * 4. LOW - Version consistency (2.0.0 everywhere)
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

class FixValidationTests {
  constructor() {
    this.client = null;
    this.transport = null;
    this.results = [];
  }

  async connect() {
    console.log('🔌 Connecting to MCP server...\n');

    this.transport = new StdioClientTransport({
      command: 'node',
      args: ['dist/index.js']
    });

    this.client = new Client(
      {
        name: 'fix-validation-test-client',
        version: '1.0.0'
      },
      {
        capabilities: {}
      }
    );

    await this.client.connect(this.transport);
    console.log('✅ Connected\n');
  }

  async disconnect() {
    if (this.client) {
      await this.client.close();
      console.log('\n👋 Disconnected');
    }
  }

  logTest(name, success, message, details = null) {
    const icon = success ? '✅' : '❌';
    console.log(`${icon} ${name}`);
    console.log(`   ${message}`);
    if (details) {
      console.log(`   Details: ${JSON.stringify(details, null, 2)}`);
    }
    console.log('');

    this.results.push({ name, success, message, details });
  }

  /**
   * TEST 1: Validate metadata fields are correct
   * Checks that created/ownedBy are removed and real LM Studio fields are present
   */
  async testMetadataFields() {
    console.log('📋 TEST 1: Metadata Fields Validation');
    console.log('─'.repeat(60));

    try {
      // Test via get_model_info tool
      const modelsResult = await this.client.callTool({
        name: 'list_models',
        arguments: { type: 'llm', includeMetadata: true }
      });

      const modelsData = JSON.parse(modelsResult.content[0].text);

      if (!modelsData.models || modelsData.models.length === 0) {
        this.logTest(
          'Metadata Fields Check',
          false,
          'No models available for metadata testing'
        );
        return;
      }

      const firstModel = modelsData.models[0];

      // Get detailed info
      const infoResult = await this.client.callTool({
        name: 'get_model_info',
        arguments: { model: firstModel.id }
      });

      const modelInfo = JSON.parse(infoResult.content[0].text);
      const metadata = modelInfo.model.metadata;

      // Check that old fields are NOT present
      const hasOldFields = metadata.created !== undefined || metadata.ownedBy !== undefined;

      // Check that new fields ARE present
      const hasNewFields =
        metadata.publisher !== undefined ||
        metadata.quantization !== undefined ||
        metadata.maxContextLength !== undefined;

      const success = !hasOldFields && hasNewFields;

      this.logTest(
        'Metadata Fields Correctness',
        success,
        success
          ? 'Metadata contains LM Studio fields (publisher, quantization, maxContextLength) and NOT undefined fields (created, ownedBy)'
          : 'Metadata structure incorrect',
        {
          hasOldUndefinedFields: hasOldFields,
          hasNewValidFields: hasNewFields,
          actualMetadata: metadata
        }
      );

      // Also test via local://models resource
      const resourceResult = await this.client.readResource({ uri: 'local://models' });
      const resourceData = JSON.parse(resourceResult.contents[0].text);
      const resourceModel = resourceData.models[0];
      const resourceMetadata = resourceModel.metadata;

      const resourceHasOld = resourceMetadata.created !== undefined || resourceMetadata.ownedBy !== undefined;
      const resourceHasNew =
        resourceMetadata.publisher !== undefined ||
        resourceMetadata.quantization !== undefined ||
        resourceMetadata.maxContextLength !== undefined;

      this.logTest(
        'Resource Metadata Correctness',
        !resourceHasOld && resourceHasNew,
        !resourceHasOld && resourceHasNew
          ? 'local://models resource also has correct metadata fields'
          : 'local://models resource has incorrect metadata',
        {
          hasOldUndefinedFields: resourceHasOld,
          hasNewValidFields: resourceHasNew,
          actualMetadata: resourceMetadata
        }
      );

    } catch (error) {
      this.logTest(
        'Metadata Fields Test',
        false,
        `Test failed with error: ${error.message}`,
        { error: error.stack }
      );
    }
  }

  /**
   * TEST 2: Validate previousDefault is captured correctly
   * Ensures set_default_model returns the OLD default, not the NEW one
   */
  async testPreviousDefaultCapture() {
    console.log('📋 TEST 2: previousDefault Capture Validation');
    console.log('─'.repeat(60));

    try {
      // Get current default
      const modelsResult = await this.client.readResource({ uri: 'local://models' });
      const modelsData = JSON.parse(modelsResult.contents[0].text);
      const originalDefault = modelsData.defaultModel;

      if (!modelsData.llmModels || modelsData.llmModels.length < 2) {
        this.logTest(
          'previousDefault Capture',
          false,
          'Need at least 2 models to test default switching'
        );
        return;
      }

      // Find a different model to switch to
      const newModel = modelsData.llmModels.find(m => m !== originalDefault);

      // Set new default
      const setResult = await this.client.callTool({
        name: 'set_default_model',
        arguments: { model: newModel }
      });

      const setData = JSON.parse(setResult.content[0].text);

      // Validate response
      const previousMatches = setData.previousDefault === originalDefault;
      const defaultMatches = setData.defaultModel === newModel;
      const previousNotSameAsNew = setData.previousDefault !== setData.defaultModel;

      const success = previousMatches && defaultMatches && previousNotSameAsNew;

      this.logTest(
        'previousDefault Correctness',
        success,
        success
          ? `previousDefault correctly shows old model (${setData.previousDefault}), not new model (${setData.defaultModel})`
          : 'previousDefault does not match original default or equals new default',
        {
          originalDefault,
          newModel,
          responseDefaultModel: setData.defaultModel,
          responsePreviousDefault: setData.previousDefault,
          previousMatchesOriginal: previousMatches,
          previousNotSameAsNew: previousNotSameAsNew
        }
      );

      // Restore original default
      await this.client.callTool({
        name: 'set_default_model',
        arguments: { model: originalDefault }
      });

    } catch (error) {
      this.logTest(
        'previousDefault Capture Test',
        false,
        `Test failed with error: ${error.message}`,
        { error: error.stack }
      );
    }
  }

  /**
   * TEST 3: Validate secure_rewrite handles duplicate sensitive data
   * Tests that ALL occurrences of sensitive data are replaced, not just first
   */
  async testSecureRewriteDuplicates() {
    console.log('📋 TEST 3: secure_rewrite Duplicate Handling');
    console.log('─'.repeat(60));

    try {
      // Test content with duplicate sensitive data
      const testContent = 'Sarah Johnson called today. Sarah Johnson wants to discuss her account.';

      const result = await this.client.callTool({
        name: 'secure_rewrite',
        arguments: {
          content: testContent,
          style: 'professional',
          privacy_level: 'strict'
        }
      });

      const rewrittenText = result.content[0].text;

      // Check that original name does NOT appear in output
      const containsSarah = rewrittenText.toLowerCase().includes('sarah');
      const containsJohnson = rewrittenText.toLowerCase().includes('johnson');
      const containsFullName = rewrittenText.toLowerCase().includes('sarah johnson');

      const success = !containsSarah && !containsJohnson && !containsFullName;

      this.logTest(
        'Duplicate Sensitive Data Removal',
        success,
        success
          ? 'All occurrences of "Sarah Johnson" were successfully removed from output'
          : 'Original sensitive data still appears in rewritten text',
        {
          originalContent: testContent,
          rewrittenContent: rewrittenText,
          containsSarah,
          containsJohnson,
          containsFullName
        }
      );

      // Test with email duplicates
      const emailTest = 'Contact john@example.com for info. Email john@example.com directly.';
      const emailResult = await this.client.callTool({
        name: 'secure_rewrite',
        arguments: {
          content: emailTest,
          style: 'formal',
          privacy_level: 'strict'
        }
      });

      const emailRewritten = emailResult.content[0].text;
      const containsEmail = emailRewritten.toLowerCase().includes('john@example.com');

      this.logTest(
        'Duplicate Email Removal',
        !containsEmail,
        !containsEmail
          ? 'All occurrences of email address were successfully removed'
          : 'Email address still appears in rewritten text',
        {
          originalContent: emailTest,
          rewrittenContent: emailRewritten,
          containsEmail
        }
      );

      // Test with phone duplicates
      const phoneTest = 'Call 555-123-4567 today. Our number is 555-123-4567.';
      const phoneResult = await this.client.callTool({
        name: 'secure_rewrite',
        arguments: {
          content: phoneTest,
          style: 'casual',
          privacy_level: 'moderate'
        }
      });

      const phoneRewritten = phoneResult.content[0].text;
      const containsPhone = phoneRewritten.includes('555-123-4567');

      this.logTest(
        'Duplicate Phone Removal',
        !containsPhone,
        !containsPhone
          ? 'All occurrences of phone number were successfully removed'
          : 'Phone number still appears in rewritten text',
        {
          originalContent: phoneTest,
          rewrittenContent: phoneRewritten,
          containsPhone
        }
      );

    } catch (error) {
      this.logTest(
        'secure_rewrite Duplicates Test',
        false,
        `Test failed with error: ${error.message}`,
        { error: error.stack }
      );
    }
  }

  /**
   * TEST 4: Validate version consistency
   * Checks that version 2.0.0 is reported everywhere
   */
  async testVersionConsistency() {
    console.log('📋 TEST 4: Version Consistency Validation');
    console.log('─'.repeat(60));

    try {
      // Check config resource version (was previously hardcoded as 1.0.0, now should be 2.0.0)
      const configResult = await this.client.readResource({ uri: 'local://config' });
      const configData = JSON.parse(configResult.contents[0].text);
      const configVersion = configData.version;

      const expectedVersion = '2.0.0';
      const configMatches = configVersion === expectedVersion;

      this.logTest(
        'Config Resource Version',
        configMatches,
        configMatches
          ? `Config resource correctly reports version ${expectedVersion}`
          : `Config version mismatch: expected ${expectedVersion}, got ${configVersion}`,
        {
          expectedVersion,
          configVersion
        }
      );

      // Verify server name is consistent too
      const serverNameCorrect = configData.server === 'local-llm-mcp-server';

      this.logTest(
        'Server Name Consistency',
        serverNameCorrect,
        serverNameCorrect
          ? 'Server name is correctly reported in config'
          : 'Server name mismatch in config',
        {
          serverName: configData.server
        }
      );

    } catch (error) {
      this.logTest(
        'Version Consistency Test',
        false,
        `Test failed with error: ${error.message}`,
        { error: error.stack }
      );
    }
  }

  /**
   * Run all validation tests
   */
  async runAll() {
    console.log('🚀 Fix Validation Test Suite\n');
    console.log('Testing fixes for security and metadata issues\n');
    console.log('='.repeat(60));
    console.log('');

    await this.testMetadataFields();
    await this.testPreviousDefaultCapture();
    await this.testSecureRewriteDuplicates();
    await this.testVersionConsistency();

    this.printSummary();
  }

  /**
   * Print test summary
   */
  printSummary() {
    const passed = this.results.filter(r => r.success).length;
    const failed = this.results.length - passed;

    console.log('='.repeat(60));
    console.log('📊 Test Summary');
    console.log('='.repeat(60));
    console.log(`Total:  ${this.results.length}`);
    console.log(`Passed: ${passed} ✅`);
    console.log(`Failed: ${failed} ❌`);
    console.log('='.repeat(60));

    if (failed > 0) {
      console.log('\n❌ Failed Tests:');
      for (const result of this.results.filter(r => !r.success)) {
        console.log(`   ${result.name}: ${result.message}`);
      }
      console.log('');
      process.exit(1);
    } else {
      console.log('\n✅ All validation tests passed!');
      console.log('All fixes are working correctly.\n');
      process.exit(0);
    }
  }
}

// Run tests
(async () => {
  const tests = new FixValidationTests();

  try {
    await tests.connect();
    await tests.runAll();
  } catch (error) {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  } finally {
    await tests.disconnect();
  }
})();
