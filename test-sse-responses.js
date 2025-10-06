#!/usr/bin/env node

/**
 * Test SSE responses from MCP server
 * Validates that responses come via SSE event stream
 */

import https from 'https';

const PORT = 3010;
const BASE_URL = `https://localhost:${PORT}`;

// Allow self-signed certs
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';

console.log('Testing SSE Response Flow\n');

// Step 1: Connect to SSE
const req = https.request(`${BASE_URL}/sse`, {
  method: 'GET',
  rejectUnauthorized: false
}, (res) => {
  console.log('✓ SSE connection established');
  console.log('Listening for SSE events...\n');

  let sessionId = null;

  res.on('data', (chunk) => {
    const data = chunk.toString();
    console.log('SSE Event received:');
    console.log(data);

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

    if (event === 'endpoint' && eventData) {
      // Extract session ID
      const match = eventData.match(/sessionId=([^&]+)/);
      if (match) {
        sessionId = match[1];
        console.log(`Session ID: ${sessionId}\n`);

        // Send a test message
        setTimeout(() => {
          sendMessage(sessionId, 'initialize');
        }, 1000);

        setTimeout(() => {
          sendMessage(sessionId, 'tools/list');
        }, 2000);
      }
    } else if (event === 'message' && eventData) {
      // This is a response!
      try {
        const response = JSON.parse(eventData);
        console.log('✓ Received MCP response:');
        console.log(JSON.stringify(response, null, 2));
        console.log('');
      } catch (e) {
        console.log('Failed to parse response:', eventData);
      }
    }
  });

  setTimeout(() => {
    console.log('\nTest completed. Closing connection.');
    req.destroy();
    process.exit(0);
  }, 5000);
});

req.on('error', (error) => {
  console.error('Error:', error.message);
  process.exit(1);
});

req.end();

function sendMessage(sessionId, method) {
  const message = {
    jsonrpc: '2.0',
    id: method === 'initialize' ? 1 : 2,
    method,
    params: method === 'initialize' ? {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'test', version: '1.0.0' }
    } : {}
  };

  const body = JSON.stringify(message);
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
      console.log(`POST /message (${method}) response: ${data}`);
    });
  });

  postReq.on('error', error => {
    console.error(`Error sending ${method}:`, error.message);
  });

  postReq.write(body);
  postReq.end();
}
