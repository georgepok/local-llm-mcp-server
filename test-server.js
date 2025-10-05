#!/usr/bin/env node

// Quick test to verify the MCP server can start
import { spawn } from 'child_process';

console.log('Testing MCP server startup...');

const server = spawn('node', ['dist/index.js'], {
  stdio: ['pipe', 'pipe', 'pipe']
});

let hasStarted = false;

// Send initial request to test server
setTimeout(() => {
  if (!hasStarted) {
    console.log('Sending test request...');
    const initRequest = {
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {
          tools: {}
        },
        clientInfo: {
          name: 'test-client',
          version: '1.0.0'
        }
      }
    };

    server.stdin.write(JSON.stringify(initRequest) + '\n');
  }
}, 1000);

server.stdout.on('data', (data) => {
  try {
    const response = JSON.parse(data.toString().trim());
    console.log('✅ Server responded:', response);
    hasStarted = true;
    server.kill();
    console.log('✅ MCP server test successful!');
    process.exit(0);
  } catch (e) {
    console.log('Server output:', data.toString());
  }
});

server.stderr.on('data', (data) => {
  console.error('Server error:', data.toString());
});

server.on('close', (code) => {
  if (!hasStarted) {
    console.log('❌ Server closed with code:', code);
    process.exit(1);
  }
});

// Timeout after 10 seconds
setTimeout(() => {
  if (!hasStarted) {
    console.log('❌ Server test timeout');
    server.kill();
    process.exit(1);
  }
}, 10000);