#!/usr/bin/env tsx
/**
 * Test: Thinking Leak Fix
 *
 * Validates that the MCP server never returns raw reasoning/thinking
 * content to the user, and that thinking-retry produces visible content
 * instead of empty responses.
 *
 * Test cases:
 * 1. Low max_tokens (100) — model will exhaust budget on thinking → retry should fix
 * 2. Medium max_tokens (400) — reported problem case from users
 * 3. Normal max_tokens (1024) — should work on first attempt
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

// Patterns that indicate leaked reasoning/thinking content
const THINKING_PATTERNS = [
  /^Okay,?\s+(the user|so|let me|I need)/i,
  /^Hmm,?\s/i,
  /^Let me (think|analyze|consider|break)/i,
  /^The user (is asking|wants|has|asks)/i,
  /^I (need to|should|must|will) (think|analyze|consider|understand)/i,
  /^First,? (I need|let me|I should)/i,
  /^Wait,?\s/i,
  /^So the user/i,
];

function looksLikeThinking(text: string): boolean {
  const trimmed = text.trim();
  return THINKING_PATTERNS.some(p => p.test(trimmed));
}

interface TestResult {
  name: string;
  passed: boolean;
  response: string;
  retried: boolean;
  detail: string;
}

async function runTest(
  client: Client,
  name: string,
  prompt: string,
  maxTokens: number,
): Promise<TestResult> {
  console.log(`\n--- ${name} (max_tokens=${maxTokens}) ---`);
  console.log(`Prompt: "${prompt.substring(0, 80)}..."`);

  const start = Date.now();

  const result = await client.callTool({
    name: 'local_reasoning',
    arguments: {
      prompt,
      model_params: {
        temperature: 0.5,
        max_tokens: maxTokens,
      },
    },
  });

  const elapsed = Date.now() - start;
  const content = (result.content as any)?.[0]?.text || '';
  const isEmpty = content.trim().length === 0;
  const isThinking = looksLikeThinking(content);

  // Check stderr for retry log (captured separately — we infer from timing)
  // A retry doubles the request, so >2x normal latency hints at retry
  const likelyRetried = elapsed > 15000; // rough heuristic

  let passed = true;
  let detail = '';

  if (isEmpty) {
    passed = false;
    detail = 'FAIL: Empty response returned to user';
  } else if (isThinking) {
    passed = false;
    detail = `FAIL: Thinking leak detected. Starts with: "${content.substring(0, 100)}"`;
  } else {
    detail = `OK: Got ${content.length} chars of visible content in ${elapsed}ms`;
  }

  console.log(detail);
  console.log(`Response preview: "${content.substring(0, 150).replace(/\n/g, ' ')}"`);

  return { name, passed, response: content, retried: likelyRetried, detail };
}

async function main() {
  console.log('=== Thinking Leak Fix — MCP Integration Test ===\n');

  // Connect via stdio
  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/index.js'],
  });

  const client = new Client(
    { name: 'thinking-leak-test', version: '1.0.0' },
    { capabilities: {} },
  );

  await client.connect(transport);
  console.log('Connected to MCP server via stdio\n');

  const results: TestResult[] = [];

  // Test 1: Very low token budget — will definitely exhaust on thinking
  results.push(
    await runTest(
      client,
      'Test 1: Low budget (100 tokens)',
      'What is the capital of France?',
      100,
    ),
  );

  // Test 2: Medium budget — the reported problem case
  results.push(
    await runTest(
      client,
      'Test 2: Medium budget (400 tokens) — complex prompt',
      `Here's partial data from our A/B test:
Group A (control): 1000 users, 52 conversions (5.2%)
Group B (new checkout): 1000 users, 71 conversions (7.1%)
Group C (new checkout + free shipping badge): 500 users, 48 conversions (9.6%)
But wait — Group C was only run on weekend traffic.
Generate 3 hypotheses about what's really happening.`,
      400,
    ),
  );

  // Test 3: Normal budget — should work without retry
  results.push(
    await runTest(
      client,
      'Test 3: Normal budget (1024 tokens)',
      'Explain what a binary search tree is in 2-3 sentences.',
      1024,
    ),
  );

  // Summary
  console.log('\n\n=== RESULTS ===');
  const passed = results.filter(r => r.passed).length;
  const failed = results.length - passed;

  for (const r of results) {
    console.log(`  ${r.passed ? 'PASS' : 'FAIL'} ${r.name}: ${r.detail}`);
  }

  console.log(`\nTotal: ${results.length} | Passed: ${passed} | Failed: ${failed}`);

  await client.close();
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
