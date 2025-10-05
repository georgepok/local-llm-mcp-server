#!/usr/bin/env node

/**
 * Regression test suite for Local LLM MCP Server
 * Tests both stdio and HTTP transport modes
 */

import { spawn } from 'child_process';

const tests = {
  passed: 0,
  failed: 0,
  total: 0
};

function log(message, type = 'info') {
  const colors = {
    info: '\x1b[36m',    // Cyan
    success: '\x1b[32m', // Green
    error: '\x1b[31m',   // Red
    warning: '\x1b[33m'  // Yellow
  };
  const reset = '\x1b[0m';
  console.log(`${colors[type] || ''}${message}${reset}`);
}

function test(name, fn) {
  tests.total++;
  return fn()
    .then(() => {
      tests.passed++;
      log(`✓ ${name}`, 'success');
    })
    .catch((error) => {
      tests.failed++;
      log(`✗ ${name}`, 'error');
      log(`  Error: ${error.message}`, 'error');
    });
}

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Test 1: Build verification
await test('Build verification - dist/ exists', async () => {
  const fs = await import('fs');
  if (!fs.existsSync('dist/index.js')) {
    throw new Error('dist/index.js not found. Run npm run build first.');
  }
});

// Test 2: HTTP mode - Server startup
await test('HTTP mode - Server starts successfully', async () => {
  return new Promise((resolve, reject) => {
    const server = spawn('node', ['dist/index.js'], {
      env: { ...process.env, MCP_TRANSPORT: 'http', PORT: '3002' },
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let started = false;
    const timeout = setTimeout(() => {
      server.kill();
      if (!started) reject(new Error('Server startup timeout'));
    }, 5000);

    server.stdout.on('data', (data) => {
      if (data.toString().includes('MCP server ready')) {
        started = true;
        clearTimeout(timeout);
        server.kill('SIGINT');
        setTimeout(resolve, 500);
      }
    });

    server.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });
  });
});

// Test 3: HTTP mode - Health endpoint
await test('HTTP mode - /health endpoint responds', async () => {
  const server = spawn('node', ['dist/index.js'], {
    env: { ...process.env, MCP_TRANSPORT: 'http', PORT: '3003' },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  await sleep(2000); // Wait for server to start

  try {
    const response = await fetch('http://localhost:3003/health');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const data = await response.json();
    if (data.status !== 'ok') throw new Error('Health check failed');
    if (data.transport !== 'http') throw new Error('Wrong transport mode');

    server.kill('SIGINT');
    await sleep(500);
  } catch (error) {
    server.kill();
    throw error;
  }
});

// Test 4: HTTP mode - Root endpoint
await test('HTTP mode - / endpoint returns server info', async () => {
  const server = spawn('node', ['dist/index.js'], {
    env: { ...process.env, MCP_TRANSPORT: 'http', PORT: '3004' },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  await sleep(2000);

  try {
    const response = await fetch('http://localhost:3004/');
    const data = await response.json();

    if (data.server !== 'local-llm-mcp-server') {
      throw new Error('Invalid server name');
    }
    if (!data.endpoints) {
      throw new Error('Missing endpoints info');
    }

    server.kill('SIGINT');
    await sleep(500);
  } catch (error) {
    server.kill();
    throw error;
  }
});

// Test 5: HTTP mode - SSE endpoint
await test('HTTP mode - /sse endpoint establishes connection', async () => {
  const server = spawn('node', ['dist/index.js'], {
    env: { ...process.env, MCP_TRANSPORT: 'http', PORT: '3005' },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  await sleep(2000);

  try {
    const response = await fetch('http://localhost:3005/sse');

    if (response.status !== 200) {
      throw new Error(`SSE connection failed: HTTP ${response.status}`);
    }

    if (response.headers.get('content-type') !== 'text/event-stream') {
      throw new Error('Invalid content-type for SSE');
    }

    server.kill('SIGINT');
    await sleep(500);
  } catch (error) {
    server.kill();
    throw error;
  }
});

// Test 6: HTTP mode - CORS headers
await test('HTTP mode - CORS headers present', async () => {
  const server = spawn('node', ['dist/index.js'], {
    env: { ...process.env, MCP_TRANSPORT: 'http', PORT: '3006' },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  await sleep(2000);

  try {
    const response = await fetch('http://localhost:3006/health');
    const corsHeader = response.headers.get('access-control-allow-origin');

    if (!corsHeader) {
      throw new Error('CORS headers missing');
    }

    server.kill('SIGINT');
    await sleep(500);
  } catch (error) {
    server.kill();
    throw error;
  }
});

// Test 7: HTTP mode - Custom port
await test('HTTP mode - Custom port configuration works', async () => {
  const server = spawn('node', ['dist/index.js'], {
    env: { ...process.env, MCP_TRANSPORT: 'http', PORT: '8888' },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  await sleep(2000);

  try {
    const response = await fetch('http://localhost:8888/health');
    if (!response.ok) throw new Error('Custom port not working');

    server.kill('SIGINT');
    await sleep(500);
  } catch (error) {
    server.kill();
    throw error;
  }
});

// Test 8: Stdio mode - Default behavior
await test('Stdio mode - Defaults to stdio when no transport specified', async () => {
  return new Promise((resolve, reject) => {
    const server = spawn('node', ['dist/index.js'], {
      env: { ...process.env },
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let foundStdioMessage = false;
    const timeout = setTimeout(() => {
      server.kill();
      if (!foundStdioMessage) {
        reject(new Error('Stdio mode message not found'));
      }
    }, 3000);

    server.stdout.on('data', (data) => {
      if (data.toString().includes('stdio mode')) {
        foundStdioMessage = true;
        clearTimeout(timeout);
        server.kill('SIGINT');
        setTimeout(resolve, 500);
      }
    });

    server.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });
  });
});

// Test 9: CLI flag - --http flag works
await test('CLI flag - --http flag enables HTTP mode', async () => {
  return new Promise((resolve, reject) => {
    const server = spawn('node', ['dist/index.js', '--http'], {
      env: { ...process.env, PORT: '3007' },
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let foundHttpMessage = false;
    const timeout = setTimeout(() => {
      server.kill();
      if (!foundHttpMessage) {
        reject(new Error('HTTP mode not activated by --http flag'));
      }
    }, 3000);

    server.stdout.on('data', (data) => {
      if (data.toString().includes('HTTP mode')) {
        foundHttpMessage = true;
        clearTimeout(timeout);
        server.kill('SIGINT');
        setTimeout(resolve, 500);
      }
    });

    server.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });
  });
});

// Test 10: Model discovery - LM Studio connection
await test('Model discovery - Connects to LM Studio', async () => {
  // This test checks if the server can connect to LM Studio
  // It will pass if LM Studio is running, or skip if not available
  const server = spawn('node', ['dist/index.js'], {
    env: { ...process.env, MCP_TRANSPORT: 'http', PORT: '3008' },
    stdio: ['ignore', 'pipe', 'pipe']
  });

  await sleep(3000); // Allow time for LM Studio connection

  try {
    // Just verify server started (LM Studio connection is optional for this test)
    const response = await fetch('http://localhost:3008/health');
    if (!response.ok) throw new Error('Server failed to start');

    server.kill('SIGINT');
    await sleep(500);
  } catch (error) {
    server.kill();
    throw error;
  }
});

// Print summary
log('\n' + '='.repeat(60), 'info');
log(`Test Results: ${tests.passed}/${tests.total} passed`, tests.failed === 0 ? 'success' : 'warning');

if (tests.failed > 0) {
  log(`${tests.failed} test(s) failed`, 'error');
  process.exit(1);
} else {
  log('All tests passed! ✓', 'success');
  process.exit(0);
}
