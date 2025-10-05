#!/usr/bin/env node

/**
 * Full MCP Protocol Test over HTTPS
 * Tests actual MCP functionality through HTTPS/SSE transport
 */

import https from 'https';
import { EventSource } from 'eventsource';

const PORT = process.env.PORT || 3010;
const HOST = process.env.HOST || 'localhost';
const BASE_URL = `https://${HOST}:${PORT}`;

// Allow self-signed certificates
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

const colors = {
  green: '\x1b[32m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  yellow: '\x1b[33m',
  reset: '\x1b[0m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

let sessionId = null;
let messageId = 0;

async function sendMessage(method, params = {}) {
  return new Promise((resolve, reject) => {
    const message = {
      jsonrpc: '2.0',
      id: ++messageId,
      method,
      params
    };

    const body = JSON.stringify(message);
    const url = `${BASE_URL}/message?sessionId=${sessionId}`;

    const req = https.request(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body)
      },
      rejectUnauthorized: false
    }, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        try {
          const response = JSON.parse(data);
          if (response.error) {
            reject(new Error(`${response.error.message} (code: ${response.error.code})`));
          } else {
            resolve(response.result);
          }
        } catch (error) {
          reject(new Error(`Failed to parse response: ${data}`));
        }
      });
    });

    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

async function runTests() {
  log('\n🔒 Testing MCP Protocol over HTTPS', 'cyan');
  log(`Server: ${BASE_URL}`, 'cyan');
  log('='.repeat(60), 'cyan');

  let passed = 0;
  let failed = 0;
  let eventSource = null;

  try {
    // Test 1: Establish SSE Connection
    log('\n[Test 1] Establishing SSE Connection', 'yellow');

    eventSource = new EventSource(`${BASE_URL}/sse`, {
      https: { rejectUnauthorized: false }
    });

    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('SSE connection timeout')), 5000);

      eventSource.onopen = () => {
        clearTimeout(timeout);
        log('✓ SSE connection established', 'green');
        passed++;
        resolve();
      };

      eventSource.onerror = (error) => {
        clearTimeout(timeout);
        reject(error);
      };

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.endpoint) {
            // Extract session ID from endpoint
            const match = data.endpoint.match(/sessionId=([^&]+)/);
            if (match) {
              sessionId = match[1];
              log(`  Session ID: ${sessionId}`, 'cyan');
            }
          }
        } catch (e) {
          // Ignore parse errors for non-JSON messages
        }
      };
    });

    // Test 2: Initialize MCP Protocol
    log('\n[Test 2] MCP Initialize Handshake', 'yellow');

    const initResult = await sendMessage('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: {
        name: 'test-client',
        version: '1.0.0'
      }
    });

    log('✓ Initialize successful', 'green');
    log(`  Server: ${initResult.serverInfo.name}`, 'cyan');
    log(`  Version: ${initResult.serverInfo.version}`, 'cyan');
    log(`  Protocol: ${initResult.protocolVersion}`, 'cyan');
    passed++;

    // Test 3: List Tools
    log('\n[Test 3] List Available Tools', 'yellow');

    const toolsResult = await sendMessage('tools/list');

    log('✓ Tools list retrieved', 'green');
    log(`  Tools available: ${toolsResult.tools.length}`, 'cyan');
    toolsResult.tools.slice(0, 3).forEach(tool => {
      log(`    - ${tool.name}: ${tool.description.substring(0, 50)}...`, 'cyan');
    });
    passed++;

    // Test 4: List Resources
    log('\n[Test 4] List Available Resources', 'yellow');

    const resourcesResult = await sendMessage('resources/list');

    log('✓ Resources list retrieved', 'green');
    log(`  Resources available: ${resourcesResult.resources.length}`, 'cyan');
    resourcesResult.resources.forEach(resource => {
      log(`    - ${resource.name} (${resource.uri})`, 'cyan');
    });
    passed++;

    // Test 5: List Prompts
    log('\n[Test 5] List Available Prompts', 'yellow');

    const promptsResult = await sendMessage('prompts/list');

    log('✓ Prompts list retrieved', 'green');
    log(`  Prompts available: ${promptsResult.prompts.length}`, 'cyan');
    promptsResult.prompts.slice(0, 3).forEach(prompt => {
      log(`    - ${prompt.name}: ${prompt.description.substring(0, 50)}...`, 'cyan');
    });
    passed++;

    // Test 6: Read a Resource
    log('\n[Test 6] Read Resource (local://models)', 'yellow');

    const readResult = await sendMessage('resources/read', {
      uri: 'local://models'
    });

    log('✓ Resource read successful', 'green');
    log(`  Content type: ${readResult.contents[0].mimeType}`, 'cyan');
    const modelsData = JSON.parse(readResult.contents[0].text);
    log(`  Available models: ${modelsData.availableModels.length}`, 'cyan');
    if (modelsData.defaultModel) {
      log(`  Default model: ${modelsData.defaultModel}`, 'cyan');
    }
    passed++;

  } catch (error) {
    log(`✗ Test failed: ${error.message}`, 'red');
    failed++;
  } finally {
    if (eventSource) {
      eventSource.close();
    }
  }

  // Summary
  log('\n' + '='.repeat(60), 'cyan');
  log(`Test Results: ${passed}/${passed + failed} passed`, passed === passed + failed ? 'green' : 'yellow');

  if (failed > 0) {
    log(`${failed} test(s) failed`, 'red');
    process.exit(1);
  } else {
    log('All tests passed! ✓', 'green');
    log('\n✅ HTTPS MCP Server is fully functional!', 'green');
    log('   Full MCP protocol working over HTTPS/SSE transport', 'green');
    process.exit(0);
  }
}

// Run tests
runTests().catch(error => {
  log(`\n❌ Test suite failed: ${error.message}`, 'red');
  log(`\nMake sure the HTTPS server is running on port ${PORT}:`, 'yellow');
  log(`  ./scripts/start-https.sh --port ${PORT}`, 'cyan');
  process.exit(1);
});
