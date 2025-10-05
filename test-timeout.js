#!/usr/bin/env node

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

async function testTimeoutHandling() {
  console.log('🧪 Testing Timeout Handling...\n');

  const client = new Client({
    name: 'timeout-test-client',
    version: '1.0.0'
  }, {
    capabilities: { tools: {} }
  });

  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/index.js'],
    timeout: 600000 // 10 minutes
  });

  await client.connect(transport);

  try {
    console.log('Testing basic reasoning (should work with new timeout)...');

    const result = await client.callTool({
      name: 'local_reasoning',
      arguments: {
        prompt: 'What is 5 + 3? Please explain your reasoning.',
        model_params: {
          temperature: 0.1,
          max_tokens: 100
        }
      }
    });

    console.log(`✅ Basic reasoning completed: ${result.content[0].text.substring(0, 100)}...`);

  } catch (error) {
    if (error.message.includes('timeout')) {
      console.log(`⚠️  Timeout occurred: ${error.message}`);
    } else {
      console.log(`❌ Other error: ${error.message}`);
    }
  }

  try {
    console.log('\nTesting complex code analysis (longer timeout expected)...');

    const complexCode = `
function processPayment(cardNumber, amount, userId) {
  const query = "SELECT * FROM users WHERE id = " + userId;
  const user = db.query(query);

  if (user.balance >= amount) {
    const apiKey = "sk_live_12345abcdef";
    return paymentGateway.charge(cardNumber, amount, apiKey);
  }

  return { error: "Insufficient funds" };
}

class AuthManager {
  constructor() {
    this.secretKey = "hardcoded_secret_123";
  }

  validateToken(token) {
    return token === this.secretKey;
  }
}`;

    const result = await client.callTool({
      name: 'code_analysis',
      arguments: {
        code: complexCode,
        language: 'javascript',
        analysis_focus: 'security'
      }
    });

    console.log(`✅ Code analysis completed successfully`);

  } catch (error) {
    if (error.message.includes('timeout')) {
      console.log(`⚠️  Code analysis timeout: ${error.message}`);
    } else {
      console.log(`❌ Code analysis error: ${error.message}`);
    }
  }

  await client.close();
  console.log('\n✅ Timeout testing completed');
}

testTimeoutHandling().catch(console.error);