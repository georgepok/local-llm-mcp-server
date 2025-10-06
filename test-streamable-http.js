#!/usr/bin/env node

/**
 * Test Streamable HTTP implementation
 * Tests the new MCP 2025-03-26 protocol
 */

import https from 'https';

const PORT = 3010;
const BASE_URL = `https://localhost:${PORT}`;

// Allow self-signed certs
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

console.log('Testing Streamable HTTP MCP Server');
console.log('═══════════════════════════════════════════════════════════\n');

let sessionId = null;

// Test 1: Initialize
async function testInitialize() {
  return new Promise((resolve, reject) => {
    const initMessage = {
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2025-03-26',
        capabilities: {},
        clientInfo: { name: 'streamable-test', version: '1.0.0' }
      }
    };

    const body = JSON.stringify(initMessage);
    const req = https.request(`${BASE_URL}/mcp`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/event-stream',
        'Mcp-Protocol-Version': '2025-03-26',
        'Content-Length': Buffer.byteLength(body)
      },
      rejectUnauthorized: false
    }, (res) => {
      console.log('Test 1: Initialize');
      console.log(`Status: ${res.statusCode}`);
      console.log(`Headers:`, res.headers);

      // Extract session ID from response header
      sessionId = res.headers['mcp-session-id'];
      if (sessionId) {
        console.log(`✅ Session ID: ${sessionId}`);
      } else {
        console.log(`⚠️  No session ID in response headers`);
      }

      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          // Parse SSE format: "event: message\ndata: {json}\n\n"
          const lines = data.split('\n');
          let jsonData = null;

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              jsonData = line.substring(6);
              break;
            }
          }

          if (!jsonData) {
            console.log(`❌ No data found in SSE response: ${data}\n`);
            reject(new Error('No data in SSE'));
            return;
          }

          const response = JSON.parse(jsonData);
          console.log(`Response:`, JSON.stringify(response, null, 2));

          if (response.result) {
            console.log(`✅ Initialize successful\n`);
            resolve();
          } else {
            console.log(`❌ Initialize failed:`, response.error, '\n');
            reject(new Error('Initialize failed'));
          }
        } catch (e) {
          console.log(`❌ Failed to parse response: ${data}\n`);
          reject(e);
        }
      });
    });

    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// Test 2: List tools with session
async function testToolsList() {
  return new Promise((resolve, reject) => {
    const toolsMessage = {
      jsonrpc: '2.0',
      id: 2,
      method: 'tools/list',
      params: {}
    };

    const body = JSON.stringify(toolsMessage);
    const headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json, text/event-stream',
      'Mcp-Protocol-Version': '2025-03-26',
      'Content-Length': Buffer.byteLength(body)
    };

    // Include session ID if we have one
    if (sessionId) {
      headers['Mcp-Session-Id'] = sessionId;
    }

    const req = https.request(`${BASE_URL}/mcp`, {
      method: 'POST',
      headers,
      rejectUnauthorized: false
    }, (res) => {
      console.log('Test 2: List Tools');
      console.log(`Status: ${res.statusCode}`);

      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          // Parse SSE format
          const lines = data.split('\n');
          let jsonData = null;

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              jsonData = line.substring(6);
              break;
            }
          }

          const response = JSON.parse(jsonData);

          if (response.result && response.result.tools) {
            console.log(`✅ Found ${response.result.tools.length} tools:`);
            response.result.tools.forEach(tool => {
              console.log(`   - ${tool.name}`);
            });
            console.log('');
            resolve();
          } else {
            console.log(`❌ Failed to list tools:`, response.error, '\n');
            reject(new Error('Failed to list tools'));
          }
        } catch (e) {
          console.log(`❌ Failed to parse response\n`);
          reject(e);
        }
      });
    });

    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// Test 3: List resources
async function testResourcesList() {
  return new Promise((resolve, reject) => {
    const resourcesMessage = {
      jsonrpc: '2.0',
      id: 3,
      method: 'resources/list',
      params: {}
    };

    const body = JSON.stringify(resourcesMessage);
    const headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json, text/event-stream',
      'Mcp-Protocol-Version': '2025-03-26',
      'Content-Length': Buffer.byteLength(body)
    };

    if (sessionId) {
      headers['Mcp-Session-Id'] = sessionId;
    }

    const req = https.request(`${BASE_URL}/mcp`, {
      method: 'POST',
      headers,
      rejectUnauthorized: false
    }, (res) => {
      console.log('Test 3: List Resources');
      console.log(`Status: ${res.statusCode}`);

      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          // Parse SSE format
          const lines = data.split('\n');
          let jsonData = null;

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              jsonData = line.substring(6);
              break;
            }
          }

          const response = JSON.parse(jsonData);

          if (response.result && response.result.resources) {
            console.log(`✅ Found ${response.result.resources.length} resources:`);
            response.result.resources.forEach(resource => {
              console.log(`   - ${resource.uri}: ${resource.name}`);
            });
            console.log('');
            resolve();
          } else {
            console.log(`❌ Failed to list resources:`, response.error, '\n');
            reject(new Error('Failed to list resources'));
          }
        } catch (e) {
          console.log(`❌ Failed to parse response\n`);
          reject(e);
        }
      });
    });

    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// Run tests
(async () => {
  try {
    await testInitialize();
    await testToolsList();
    await testResourcesList();

    console.log('═══════════════════════════════════════════════════════════');
    console.log('✅ All Streamable HTTP tests passed!');
    console.log('');
    process.exit(0);
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    process.exit(1);
  }
})();
