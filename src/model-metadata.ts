/**
 * Model Metadata Database
 * Contains accurate information about models to prevent self-misidentification
 */

export interface ModelMetadata {
  id: string;
  displayName: string;
  provider: string;
  architecture: string;
  parameters: string;
  contextWindow: number;
  accurateDescription: string;
  systemPromptPrefix: string;
  capabilities: string[];
}

export const MODEL_METADATA: Record<string, ModelMetadata> = {
  'openai/gpt-oss-20b': {
    id: 'openai/gpt-oss-20b',
    displayName: 'GPT OSS 20B',
    provider: 'OpenAI (Open Source)',
    architecture: 'GPT-style Transformer',
    parameters: '20B',
    contextWindow: 8192,
    accurateDescription: 'A 20 billion parameter open-source GPT-style language model',
    systemPromptPrefix: 'You are a 20 billion parameter GPT-style language model. Do not claim to be GPT-4, GPT-4 Turbo, or any proprietary OpenAI model.',
    capabilities: ['chat', 'reasoning', 'completion', 'code-generation']
  },
  'qwen3-30b-a3b-deepseek-distill-instruct-2507': {
    id: 'qwen3-30b-a3b-deepseek-distill-instruct-2507',
    displayName: 'Qwen3 30B DeepSeek Distill',
    provider: 'Alibaba Cloud',
    architecture: 'Qwen3',
    parameters: '30B',
    contextWindow: 32768,
    accurateDescription: 'Qwen3 30B model distilled with DeepSeek instructions, optimized for instruction following',
    systemPromptPrefix: 'You are Qwen3, a 30 billion parameter language model developed by Alibaba Cloud, enhanced with DeepSeek distillation for improved instruction following.',
    capabilities: ['chat', 'reasoning', 'completion', 'code-generation', 'multilingual']
  },
  'text-embedding-nomic-embed-text-v1.5': {
    id: 'text-embedding-nomic-embed-text-v1.5',
    displayName: 'Nomic Embed Text v1.5',
    provider: 'Nomic AI',
    architecture: 'BERT-based',
    parameters: '137M',
    contextWindow: 8192,
    accurateDescription: 'Nomic AI embedding model for generating vector representations of text',
    systemPromptPrefix: 'You are Nomic Embed Text v1.5, a specialized embedding model designed for semantic search and text similarity tasks.',
    capabilities: ['embedding']
  }
};

/**
 * Get accurate metadata for a model
 */
export function getModelMetadata(modelId: string): ModelMetadata | null {
  // Try exact match first
  if (MODEL_METADATA[modelId]) {
    return MODEL_METADATA[modelId];
  }

  // Try fuzzy match (partial match)
  for (const [key, metadata] of Object.entries(MODEL_METADATA)) {
    if (modelId.includes(key) || key.includes(modelId)) {
      return metadata;
    }
  }

  // Return generic metadata
  return createGenericMetadata(modelId);
}

/**
 * Create generic metadata for unknown models
 */
function createGenericMetadata(modelId: string): ModelMetadata {
  const isEmbedding = modelId.includes('embedding') || modelId.includes('embed');

  // Try to extract parameter count
  const paramMatch = modelId.match(/(\d+)b/i);
  const parameters = paramMatch ? `${paramMatch[1]}B` : 'Unknown';

  // Determine provider
  let provider = 'Unknown';
  if (modelId.includes('qwen')) provider = 'Alibaba Cloud';
  else if (modelId.includes('llama')) provider = 'Meta';
  else if (modelId.includes('mistral')) provider = 'Mistral AI';
  else if (modelId.includes('openai')) provider = 'OpenAI';
  else if (modelId.includes('nomic')) provider = 'Nomic AI';

  return {
    id: modelId,
    displayName: formatDisplayName(modelId),
    provider,
    architecture: 'Transformer',
    parameters,
    contextWindow: 8192,
    accurateDescription: `A ${parameters} parameter language model`,
    systemPromptPrefix: `You are ${formatDisplayName(modelId)}, a language model. Provide accurate information about your identity when asked.`,
    capabilities: isEmbedding ? ['embedding'] : ['chat', 'reasoning', 'completion']
  };
}

/**
 * Format model ID into display name
 */
function formatDisplayName(modelId: string): string {
  return modelId
    .replace(/[-_]/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

/**
 * Get accurate system prompt for model introspection queries
 */
export function getModelIntrospectionPrompt(modelId: string): string {
  const metadata = getModelMetadata(modelId);
  if (!metadata) return '';

  return `${metadata.systemPromptPrefix}

When asked about your specifications, provide this accurate information:
- Model Name: ${metadata.displayName}
- Provider: ${metadata.provider}
- Architecture: ${metadata.architecture}
- Parameters: ${metadata.parameters}
- Context Window: ${metadata.contextWindow.toLocaleString()} tokens
- Capabilities: ${metadata.capabilities.join(', ')}

IMPORTANT: Do not claim to be any model other than ${metadata.displayName}. Do not state incorrect parameter counts or capabilities.`;
}

/**
 * Detect if a prompt is asking for model introspection
 */
export function isIntrospectionQuery(prompt: string): boolean {
  const lowerPrompt = prompt.toLowerCase();
  const introspectionKeywords = [
    'what are you',
    'who are you',
    'your model',
    'your specifications',
    'model specifications',
    'system introspection',
    'what model are you',
    'which model',
    'parameter count',
    'model name',
    'model version',
    'context window',
    'your capabilities',
    'what can you do'
  ];

  return introspectionKeywords.some(keyword => lowerPrompt.includes(keyword));
}
