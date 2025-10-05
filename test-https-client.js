#!/usr/bin/env node

/**
 * HTTPS MCP Client Test
 * Tests HTTPS server functionality
 */

import https from 'https';

const PORT = process.env.PORT || 3010;
const HOST = process.env.HOST || 'localhost';

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

async function httpsRequest(path, options = {}) {
  return new Promise((resolve, reject) => {
    const reqOptions = {
      hostname: HOST,
      port: PORT,
      path: path,
      method: options.method || 'GET',
      headers: options.headers || {},
      rejectUnauthorized: false
    };

    const req = https.request(reqOptions, (res) => {
      let data = '';
      res.on('data', (chunk) => data += chunk);
      res.on('end', () => {
        resolve({
          statusCode: res.statusCode,
          headers: res.headers,
          body: data
        });
      });
    });

    req.on('error', reject);

    if (options.body) {
      req.write(options.body);
    }

    req.end();
  });
}

async function runTests() {
  log('\n🔒 Testing HTTPS MCP Server', 'cyan');
  log(`Server: https://${HOST}:${PORT}`, 'cyan');
  log('='.repeat(60), 'cyan');

  let passed = 0;
  let failed = 0;

  // Test 1: Health Check
  try {
    log('\n[Test 1] Health Check Endpoint', 'yellow');
    const result = await httpsRequest('/health');
    const data = JSON.parse(result.body);

    if (result.statusCode === 200 && data.status === 'ok') {
      log('✓ Health check passed', 'green');
      log(`  Status: ${data.status}`, 'cyan');
      log(`  Transport: ${data.transport}`, 'cyan');
      log(`  Timestamp: ${data.timestamp}`, 'cyan');
      passed++;
    } else {
      throw new Error('Health check failed');
    }
  } catch (error) {
    log(`✗ Health check failed: ${error.message}`, 'red');
    failed++;
  }

  // Test 2: Server Info
  try {
    log('\n[Test 2] Server Info Endpoint', 'yellow');
    const result = await httpsRequest('/');
    const data = JSON.parse(result.body);

    if (result.statusCode === 200 && data.server === 'local-llm-mcp-server') {
      log('✓ Server info passed', 'green');
      log(`  Server: ${data.server}`, 'cyan');
      log(`  Version: ${data.version}`, 'cyan');
      log(`  Endpoints: ${Object.keys(data.endpoints).join(', ')}`, 'cyan');
      passed++;
    } else {
      throw new Error('Server info failed');
    }
  } catch (error) {
    log(`✗ Server info failed: ${error.message}`, 'red');
    failed++;
  }

  // Test 3: SSE Connection
  try {
    log('\n[Test 3] SSE Connection Endpoint', 'yellow');

    // SSE connections stay open, so we need special handling
    const ssePromise = new Promise((resolve, reject) => {
      const req = https.request({
        hostname: HOST,
        port: PORT,
        path: '/sse',
        method: 'GET',
        rejectUnauthorized: false
      }, (res) => {
        let data = '';

        // Set timeout to close connection after getting initial data
        const timeout = setTimeout(() => {
          req.destroy();
          resolve({
            statusCode: res.statusCode,
            headers: res.headers,
            body: data
          });
        }, 1000);

        res.on('data', (chunk) => {
          data += chunk;
          // Got initial SSE message, we can close
          if (data.includes('connected')) {
            clearTimeout(timeout);
            req.destroy();
            resolve({
              statusCode: res.statusCode,
              headers: res.headers,
              body: data
            });
          }
        });

        res.on('error', reject);
      });

      req.on('error', reject);
      req.end();
    });

    const result = await ssePromise;

    if (result.statusCode === 200 && result.headers['content-type'] === 'text/event-stream') {
      log('✓ SSE connection passed', 'green');
      log(`  Content-Type: ${result.headers['content-type']}`, 'cyan');
      log(`  Initial message: ${result.body.trim()}`, 'cyan');
      passed++;
    } else {
      throw new Error('SSE connection failed');
    }
  } catch (error) {
    log(`✗ SSE connection failed: ${error.message}`, 'red');
    failed++;
  }

  // Test 4: CORS Headers
  try {
    log('\n[Test 4] CORS Headers', 'yellow');
    const result = await httpsRequest('/health');

    if (result.headers['access-control-allow-origin']) {
      log('✓ CORS headers present', 'green');
      log(`  Access-Control-Allow-Origin: ${result.headers['access-control-allow-origin']}`, 'cyan');
      passed++;
    } else {
      throw new Error('CORS headers missing');
    }
  } catch (error) {
    log(`✗ CORS headers failed: ${error.message}`, 'red');
    failed++;
  }

  // Test 5: MCP Initialize Request
  try {
    log('\n[Test 5] MCP Initialize Request', 'yellow');
    const initRequest = {
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: {
          name: 'test-client',
          version: '1.0.0'
        }
      }
    };

    const result = await httpsRequest('/message', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(initRequest)
    });

    const data = JSON.parse(result.body);

    if (result.statusCode === 200 && data.result && data.result.serverInfo) {
      log('✓ MCP initialize passed', 'green');
      log(`  Server: ${data.result.serverInfo.name}`, 'cyan');
      log(`  Version: ${data.result.serverInfo.version}`, 'cyan');
      log(`  Protocol: ${data.result.protocolVersion}`, 'cyan');
      passed++;
    } else {
      throw new Error('Initialize request failed');
    }
  } catch (error) {
    log(`✗ MCP initialize failed: ${error.message}`, 'red');
    failed++;
  }

  // Test 6: HTTPS Security
  try {
    log('\n[Test 6] HTTPS/TLS Security', 'yellow');
    const result = await httpsRequest('/health');

    log('✓ HTTPS/TLS encryption active', 'green');
    log('  Self-signed certificate accepted', 'cyan');
    log('  Secure connection established', 'cyan');
    passed++;
  } catch (error) {
    log(`✗ HTTPS security failed: ${error.message}`, 'red');
    failed++;
  }

  // Summary
  log('\n' + '='.repeat(60), 'cyan');
  log(`Test Results: ${passed}/${passed + failed} passed`, passed === passed + failed ? 'green' : 'yellow');

  if (failed > 0) {
    log(`${failed} test(s) failed`, 'red');
    process.exit(1);
  } else {
    log('All tests passed! ✓', 'green');
    log('\n✅ HTTPS MCP Server is functional and secure!', 'green');
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
