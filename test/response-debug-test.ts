#!/usr/bin/env node
/**
 * Response Debug Test
 *
 * Tests the MCP server with varying prompt lengths to identify
 * why longer prompts return empty responses.
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';

interface TestCase {
  name: string;
  prompt: string;
  expectedMinLength: number;
}

const testCases: TestCase[] = [
  {
    name: 'Short prompt',
    prompt: 'What is 2+2?',
    expectedMinLength: 1,
  },
  {
    name: 'Medium prompt',
    prompt: 'Explain the concept of recursion in programming. Provide a simple example.',
    expectedMinLength: 50,
  },
  {
    name: 'Long prompt requiring longer response',
    prompt: `Please provide a detailed explanation of the following topics:
1. What is the Model Context Protocol (MCP)?
2. How does it enable communication between AI assistants and external tools?
3. What are the main components of an MCP server?
4. Describe the request/response flow in MCP.
5. What are the benefits of using MCP for AI integration?

Please be thorough in your explanation, providing examples where appropriate.`,
    expectedMinLength: 200,
  },
  {
    name: 'Complex analysis prompt',
    prompt: `Analyze the following code and provide:
1. A summary of what the code does
2. Potential bugs or issues
3. Performance considerations
4. Suggestions for improvement

\`\`\`javascript
function fibonacci(n) {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

function memoizedFib() {
  const cache = {};
  return function fib(n) {
    if (n in cache) return cache[n];
    if (n <= 1) return n;
    cache[n] = fib(n - 1) + fib(n - 2);
    return cache[n];
  };
}

const fastFib = memoizedFib();
console.log(fastFib(50));
\`\`\`

Be thorough in your analysis.`,
    expectedMinLength: 300,
  },
];

async function runDiagnostic() {
  console.log('='.repeat(70));
  console.log('MCP Response Debug Test');
  console.log('='.repeat(70));
  console.log('');

  // Connect to server
  console.log('[1] Connecting to MCP server via stdio...');

  const transport = new StdioClientTransport({
    command: 'node',
    args: ['dist/index.js'],
  });

  const client = new Client(
    { name: 'response-debug-test', version: '1.0.0' },
    { capabilities: {} }
  );

  await client.connect(transport);
  console.log('[1] Connected successfully\n');

  // Get available models first
  console.log('[2] Fetching available models...');
  const modelsResult = await client.callTool({
    name: 'list_models',
    arguments: { type: 'llm' }
  });

  const modelsContent = (modelsResult.content as any)[0];
  const modelsData = JSON.parse(modelsContent.text);
  console.log(`[2] Found ${modelsData.models?.length || 0} LLM models`);
  console.log(`[2] Default model: ${modelsData.defaultModel || 'none'}\n`);

  if (!modelsData.models || modelsData.models.length === 0) {
    console.log('ERROR: No models available. Make sure LM Studio is running with a loaded model.');
    await client.close();
    process.exit(1);
  }

  // Run test cases
  console.log('[3] Running prompt tests...\n');
  console.log('-'.repeat(70));

  for (let i = 0; i < testCases.length; i++) {
    const tc = testCases[i];
    console.log(`\nTest ${i + 1}/${testCases.length}: ${tc.name}`);
    console.log(`Prompt length: ${tc.prompt.length} chars`);
    console.log(`Prompt preview: "${tc.prompt.substring(0, 80)}..."`);
    console.log('');

    const startTime = Date.now();

    try {
      const result = await client.callTool({
        name: 'local_reasoning',
        arguments: {
          prompt: tc.prompt,
          model_params: {
            max_tokens: 2000,
            temperature: 0.7
          }
        }
      });

      const duration = Date.now() - startTime;

      // Log full response structure for debugging
      console.log('--- Response Debug Info ---');
      console.log(`Duration: ${duration}ms`);
      console.log(`Result type: ${typeof result}`);
      console.log(`Result keys: ${Object.keys(result).join(', ')}`);
      console.log(`Content array length: ${(result.content as any)?.length}`);
      console.log(`isError: ${(result as any).isError}`);

      if (result.content && (result.content as any).length > 0) {
        const content = (result.content as any)[0];
        console.log(`Content[0] type: ${content.type}`);
        console.log(`Content[0] text type: ${typeof content.text}`);
        console.log(`Content[0] text length: ${content.text?.length || 0}`);

        if (content.text) {
          const text = content.text;
          console.log(`\n--- Response Preview (first 500 chars) ---`);
          console.log(text.substring(0, 500));
          if (text.length > 500) {
            console.log(`... (${text.length - 500} more chars)`);
          }
          console.log('--- End Preview ---\n');

          // Determine success
          if (text.length === 0) {
            console.log(`RESULT: EMPTY RESPONSE`);
          } else if (text.length < tc.expectedMinLength) {
            console.log(`RESULT: SHORT RESPONSE (${text.length} chars, expected >= ${tc.expectedMinLength})`);
          } else {
            console.log(`RESULT: SUCCESS (${text.length} chars)`);
          }
        } else {
          console.log('RESULT: TEXT IS NULL/UNDEFINED');
        }
      } else {
        console.log('RESULT: NO CONTENT IN RESPONSE');
        console.log('Full result:', JSON.stringify(result, null, 2));
      }

    } catch (error) {
      const duration = Date.now() - startTime;
      console.log(`Duration before error: ${duration}ms`);
      console.log(`RESULT: ERROR`);
      console.log(`Error type: ${(error as any)?.constructor?.name}`);
      console.log(`Error message: ${(error as Error).message}`);
      if ((error as any).cause) {
        console.log(`Error cause: ${(error as any).cause}`);
      }
    }

    console.log('-'.repeat(70));
  }

  // Cleanup
  console.log('\n[4] Closing connection...');
  await client.close();
  console.log('[4] Done\n');
}

// Run the diagnostic
runDiagnostic().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
