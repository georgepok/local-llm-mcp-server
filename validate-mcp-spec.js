#!/usr/bin/env node

/**
 * Comprehensive MCP SSE Transport Specification Validator
 *
 * Validates implementation against official MCP SDK requirements:
 * 1. SSE connection with proper headers
 * 2. Endpoint event format
 * 3. Session ID handling
 * 4. Message POST handling
 * 5. Response via SSE
 * 6. JSON-RPC compliance
 */

import https from 'https';

const PORT = 3010;
const BASE_URL = `https://localhost:${PORT}`;

// Allow self-signed certs
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

console.log('╔══════════════════════════════════════════════════════════╗');
console.log('║  MCP SSE Transport Specification Validator              ║');
console.log('╚══════════════════════════════════════════════════════════╝\n');

const results = {
  passed: [],
  failed: []
};

function pass(test) {
  results.passed.push(test);
  console.log(`✅ PASS: ${test}`);
}

function fail(test, error) {
  results.failed.push({ test, error });
  console.log(`❌ FAIL: ${test}`);
  console.log(`   Error: ${error}\n`);
}

// Test 1: SSE Connection Headers
console.log('Test 1: SSE Connection Headers');
console.log('─────────────────────────────────────────────────────────\n');

const req = https.request(`${BASE_URL}/sse`, {
  method: 'GET',
  rejectUnauthorized: false
}, (res) => {

  // Validate headers
  if (res.headers['content-type'] === 'text/event-stream') {
    pass('Content-Type: text/event-stream');
  } else {
    fail('Content-Type: text/event-stream', `Got: ${res.headers['content-type']}`);
  }

  if (res.headers['cache-control'] === 'no-cache, no-transform') {
    pass('Cache-Control: no-cache, no-transform');
  } else {
    fail('Cache-Control: no-cache, no-transform', `Got: ${res.headers['cache-control']}`);
  }

  if (res.headers['connection'] === 'keep-alive') {
    pass('Connection: keep-alive');
  } else {
    fail('Connection: keep-alive', `Got: ${res.headers['connection']}`);
  }

  if (res.statusCode === 200) {
    pass('HTTP Status 200 for SSE connection');
  } else {
    fail('HTTP Status 200 for SSE connection', `Got: ${res.statusCode}`);
  }

  let sessionId = null;
  let receivedEndpointEvent = false;
  let receivedMessageEvents = [];

  console.log('\nTest 2: SSE Event Format');
  console.log('─────────────────────────────────────────────────────────\n');

  res.on('data', (chunk) => {
    const data = chunk.toString();

    // Parse SSE event
    const lines = data.split('\n');
    let event = null;
    let eventData = null;

    lines.forEach(line => {
      if (line.startsWith('event:')) {
        event = line.substring(6).trim();
      } else if (line.startsWith('data:')) {
        eventData = line.substring(5).trim();
      }
    });

    // Test endpoint event
    if (event === 'endpoint' && eventData && !receivedEndpointEvent) {
      receivedEndpointEvent = true;
      pass('Received "event: endpoint"');

      // Validate endpoint format
      if (eventData.startsWith('/message?sessionId=')) {
        pass('Endpoint format: /message?sessionId=...');

        // Extract session ID
        const match = eventData.match(/sessionId=([^&\s]+)/);
        if (match) {
          sessionId = match[1];

          // Validate UUID format
          const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
          if (uuidRegex.test(sessionId)) {
            pass('Session ID is valid UUID format');
          } else {
            fail('Session ID is valid UUID format', `Got: ${sessionId}`);
          }

          console.log(`   Session ID: ${sessionId}\n`);

          // Test 3: Message POST
          setTimeout(() => {
            console.log('Test 3: Message POST Handling');
            console.log('─────────────────────────────────────────────────────────\n');
            testMessagePost(sessionId);
          }, 500);

          // Test 4: Multiple messages
          setTimeout(() => {
            console.log('\nTest 4: Multiple Messages');
            console.log('─────────────────────────────────────────────────────────\n');
            testToolsList(sessionId);
          }, 1500);

          // Test 5: Resource read
          setTimeout(() => {
            console.log('\nTest 5: Resource Handling');
            console.log('─────────────────────────────────────────────────────────\n');
            testResourceRead(sessionId);
          }, 2500);
        } else {
          fail('Extract sessionId from endpoint', 'No sessionId found');
        }
      } else {
        fail('Endpoint format: /message?sessionId=...', `Got: ${eventData}`);
      }
    }

    // Test message events
    if (event === 'message' && eventData) {
      receivedMessageEvents.push(eventData);

      try {
        const message = JSON.parse(eventData);

        // Validate JSON-RPC format
        if (message.jsonrpc === '2.0') {
          pass(`JSON-RPC 2.0 format (id: ${message.id})`);
        } else {
          fail('JSON-RPC 2.0 format', `Got version: ${message.jsonrpc}`);
        }

        if (message.id !== undefined) {
          pass(`Message has ID field (id: ${message.id})`);
        } else {
          fail('Message has ID field', 'Missing id');
        }

        if (message.result || message.error) {
          pass('Message has result or error');

          // Log abbreviated response
          if (message.result) {
            const resultStr = JSON.stringify(message.result);
            const preview = resultStr.length > 100 ?
              resultStr.substring(0, 100) + '...' : resultStr;
            console.log(`   Result: ${preview}`);
          }
        } else {
          fail('Message has result or error', 'Missing both result and error');
        }

      } catch (e) {
        fail('Parse JSON-RPC message', `Invalid JSON: ${e.message}`);
      }
    }
  });

  // Cleanup and summary
  setTimeout(() => {
    console.log('\n\n╔══════════════════════════════════════════════════════════╗');
    console.log('║  Validation Summary                                      ║');
    console.log('╚══════════════════════════════════════════════════════════╝\n');

    console.log(`Total Tests: ${results.passed.length + results.failed.length}`);
    console.log(`✅ Passed: ${results.passed.length}`);
    console.log(`❌ Failed: ${results.failed.length}\n`);

    if (results.failed.length > 0) {
      console.log('Failed Tests:');
      results.failed.forEach(({ test, error }) => {
        console.log(`  ❌ ${test}`);
        console.log(`     ${error}`);
      });
      console.log('');
    }

    const compliance = (results.passed.length / (results.passed.length + results.failed.length) * 100).toFixed(1);
    console.log(`\nMCP Specification Compliance: ${compliance}%`);

    if (compliance === '100.0') {
      console.log('\n🎉 Implementation is fully compliant with MCP SSE Transport Specification!\n');
    } else {
      console.log('\n⚠️  Implementation has specification violations that need to be fixed.\n');
    }

    req.destroy();
    process.exit(results.failed.length > 0 ? 1 : 0);
  }, 4000);
});

req.on('error', (error) => {
  console.error('\n❌ Connection Error:', error.message);
  console.error('\nMake sure the MCP server is running on port 3010:');
  console.error('  npm run start:https\n');
  process.exit(1);
});

req.end();

// Helper function to test message POST
function testMessagePost(sessionId) {
  const initMessage = {
    jsonrpc: '2.0',
    id: 1,
    method: 'initialize',
    params: {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'spec-validator', version: '1.0.0' }
    }
  };

  const body = JSON.stringify(initMessage);
  const postReq = https.request(`${BASE_URL}/message?sessionId=${sessionId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body)
    },
    rejectUnauthorized: false
  }, (res) => {

    // Validate HTTP 202 Accepted
    if (res.statusCode === 202) {
      pass('POST /message returns HTTP 202 Accepted');
    } else {
      fail('POST /message returns HTTP 202 Accepted', `Got: ${res.statusCode}`);
    }

    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
      if (data === 'Accepted') {
        pass('POST /message returns "Accepted" body');
      } else {
        fail('POST /message returns "Accepted" body', `Got: ${data}`);
      }
    });
  });

  postReq.on('error', error => {
    fail('POST /message request', error.message);
  });

  postReq.write(body);
  postReq.end();
}

function testToolsList(sessionId) {
  const toolsMessage = {
    jsonrpc: '2.0',
    id: 2,
    method: 'tools/list',
    params: {}
  };

  const body = JSON.stringify(toolsMessage);
  const postReq = https.request(`${BASE_URL}/message?sessionId=${sessionId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body)
    },
    rejectUnauthorized: false
  }, (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
      if (res.statusCode === 202 && data === 'Accepted') {
        pass('tools/list request accepted');
      }
    });
  });

  postReq.write(body);
  postReq.end();
}

function testResourceRead(sessionId) {
  const readMessage = {
    jsonrpc: '2.0',
    id: 3,
    method: 'resources/list',
    params: {}
  };

  const body = JSON.stringify(readMessage);
  const postReq = https.request(`${BASE_URL}/message?sessionId=${sessionId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body)
    },
    rejectUnauthorized: false
  }, (res) => {
    let data = '';
    res.on('data', chunk => data += chunk);
    res.on('end', () => {
      if (res.statusCode === 202 && data === 'Accepted') {
        pass('resources/list request accepted');
      }
    });
  });

  postReq.write(body);
  postReq.end();
}
