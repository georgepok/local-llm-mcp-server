#!/usr/bin/env node

/**
 * Simple test script for HTTP transport mode
 */

import { spawn } from 'child_process';
import http from 'http';

console.log('Testing HTTP Transport Mode...\n');

// Start server in HTTP mode
const serverProcess = spawn('node', ['dist/index.js'], {
  env: {
    ...process.env,
    MCP_TRANSPORT: 'http',
    PORT: '3001',
  },
  stdio: ['ignore', 'pipe', 'pipe']
});

serverProcess.stdout.on('data', (data) => {
  console.log('[Server]', data.toString().trim());
});

serverProcess.stderr.on('data', (data) => {
  console.error('[Server Error]', data.toString().trim());
});

// Wait for server to start
setTimeout(async () => {
  console.log('\n--- Testing HTTP Endpoints ---\n');

  try {
    // Test 1: Health check
    console.log('1. Testing /health endpoint...');
    const healthResponse = await fetch('http://localhost:3001/health');
    const healthData = await healthResponse.json();
    console.log('   ✓ Health check:', JSON.stringify(healthData, null, 2));

    // Test 2: Root endpoint
    console.log('\n2. Testing / endpoint...');
    const rootResponse = await fetch('http://localhost:3001/');
    const rootData = await rootResponse.json();
    console.log('   ✓ Server info:', JSON.stringify(rootData, null, 2));

    // Test 3: SSE endpoint (just connect, don't wait for events)
    console.log('\n3. Testing /sse endpoint...');
    const sseResponse = await fetch('http://localhost:3001/sse');
    console.log('   ✓ SSE connection established, status:', sseResponse.status);

    console.log('\n✅ All HTTP endpoints working!\n');

  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
  } finally {
    // Cleanup
    console.log('Stopping server...');
    serverProcess.kill('SIGINT');
    setTimeout(() => process.exit(0), 1000);
  }
}, 2000);

// Handle errors
serverProcess.on('error', (error) => {
  console.error('Failed to start server:', error);
  process.exit(1);
});
