import { OpenAI } from 'openai';
import type { LMStudioConfig, ModelParams } from './types.js';
import { getModelMetadata, getModelIntrospectionPrompt, isIntrospectionQuery } from './model-metadata.js';

export class LMStudioClient {
  private client: OpenAI;
  private config: LMStudioConfig;
  private defaultModel: string | null = null;

  constructor(config: Partial<LMStudioConfig> = {}) {
    this.config = {
      baseUrl: config.baseUrl || 'http://localhost:1234/v1',
      model: config.model || '',
      timeout: config.timeout || 600000, // Increased to 10 minutes for slow local models
      retries: config.retries || 3,
      adaptiveTimeout: config.adaptiveTimeout !== false, // Default to true
    };

    this.client = new OpenAI({
      baseURL: this.config.baseUrl,
      apiKey: 'lm-studio',
      timeout: this.config.timeout,
    });
  }

  async initialize(): Promise<void> {
    // Set first available model as default if not already set
    if (!this.defaultModel) {
      const models = await this.getAvailableModels();
      if (models.length > 0) {
        this.defaultModel = models[0];
      }
    }
  }

  setDefaultModel(modelName: string): void {
    this.defaultModel = modelName;
  }

  getDefaultModel(): string | null {
    return this.defaultModel;
  }

  async isAvailable(): Promise<boolean> {
    try {
      await this.client.models.list();
      return true;
    } catch {
      return false;
    }
  }

  async getAvailableModels(): Promise<string[]> {
    try {
      // Use LM Studio REST API to get only loaded models
      const baseUrl = this.config.baseUrl.replace('/v1', '');
      const response = await fetch(`${baseUrl}/api/v0/models`);
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }
      const data = await response.json();
      // Filter only loaded models
      return data.data
        .filter((model: any) => model.state === 'loaded')
        .map((model: any) => model.id);
    } catch (error) {
      console.error('Failed to get available models:', error);
      return [];
    }
  }

  async getAvailableModelsWithMetadata(): Promise<any[]> {
    try {
      // Use LM Studio REST API to get detailed metadata for loaded models
      const baseUrl = this.config.baseUrl.replace('/v1', '');
      const response = await fetch(`${baseUrl}/api/v0/models`);
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }
      const data = await response.json();
      // Filter only loaded models and return full metadata
      return data.data
        .filter((model: any) => model.state === 'loaded')
        .map((model: any) => ({
          id: model.id,
          name: model.id,
          object: model.object,
          type: model.type,
          publisher: model.publisher,
          architecture: model.arch,
          compatibilityType: model.compatibility_type,
          quantization: model.quantization,
          state: model.state,
          maxContextLength: model.max_context_length,
          loadedContextLength: model.loaded_context_length,
          capabilities: model.capabilities || [],
        }));
    } catch (error) {
      console.error('Failed to get models with metadata:', error);
      return [];
    }
  }

  async generateResponse(
    prompt: string,
    systemPrompt?: string,
    params: Partial<ModelParams> = {},
    modelOverride?: string
  ): Promise<string> {
    const modelToUse = modelOverride || this.defaultModel;
    if (!modelToUse) {
      throw new Error('No model specified and no default model set. Use setDefaultModel() or pass a model parameter.');
    }

    const input: Array<{ role: 'system' | 'user'; content: string }> = [];

    // Inject accurate model metadata for introspection queries
    if (isIntrospectionQuery(prompt)) {
      const introspectionPrompt = getModelIntrospectionPrompt(modelToUse);
      if (introspectionPrompt) {
        input.push({ role: 'system', content: introspectionPrompt });
      }
    } else if (systemPrompt) {
      input.push({ role: 'system', content: systemPrompt });
    }

    input.push({ role: 'user', content: prompt });

    // Calculate adaptive timeout based on content complexity
    const adaptiveTimeout = this.config.adaptiveTimeout
      ? this.calculateAdaptiveTimeout(prompt, systemPrompt, params.max_tokens)
      : this.config.timeout;

    // Create a client with adaptive timeout if different from default
    const clientToUse = adaptiveTimeout !== this.config.timeout
      ? new OpenAI({
          baseURL: this.config.baseUrl,
          apiKey: 'lm-studio',
          timeout: adaptiveTimeout,
        })
      : this.client;

    try {
      const response = await clientToUse.responses.create({
        model: modelToUse,
        input,
        temperature: params.temperature,
        max_output_tokens: params.max_tokens,
        top_p: params.top_p,
      });

      return response.output_text || '';
    } catch (error) {
      console.error('LM Studio API error:', error);

      if (error instanceof Error && error.message.includes('timeout')) {
        const suggestions = this.getPerformanceSuggestions(prompt, params.max_tokens);
        throw new Error(`Request timed out after ${adaptiveTimeout / 1000}s. ${suggestions}`);
      }

      throw new Error(`Failed to generate response: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private calculateAdaptiveTimeout(
    prompt: string,
    systemPrompt?: string,
    maxTokens?: number
  ): number {
    // Base timeout
    let timeout = this.config.timeout;

    // Estimate complexity
    const totalPromptLength = prompt.length + (systemPrompt?.length || 0);
    const requestedTokens = maxTokens || 1000;

    // With 10-minute base timeout, no need for additional increases
    // The base timeout is already generous for all request types

    return timeout;
  }

  private getPerformanceSuggestions(prompt: string, maxTokens?: number): string {
    const suggestions = [];

    if ((maxTokens || 1000) > 1500) {
      suggestions.push('Reduce max_tokens to 1000 or less');
    }

    if (prompt.length > 2000) {
      suggestions.push('Simplify or shorten the prompt');
    }

    if (prompt.includes('```') && prompt.includes('analyze')) {
      suggestions.push('Consider using a faster model for code analysis, or break complex code into smaller chunks');
    }

    if (suggestions.length === 0) {
      suggestions.push('Try using a faster/smaller model, increase timeout in config, or restart LM Studio');
    }

    return `Performance suggestions: ${suggestions.join(', ')}.`;
  }

  async generateStreamResponse(
    prompt: string,
    systemPrompt?: string,
    params: Partial<ModelParams> = {},
    modelOverride?: string
  ): Promise<AsyncIterable<string>> {
    const input: Array<{ role: 'system' | 'user'; content: string }> = [];

    if (systemPrompt) {
      input.push({ role: 'system', content: systemPrompt });
    }

    input.push({ role: 'user', content: prompt });

    const modelToUse = modelOverride || this.defaultModel;
    if (!modelToUse) {
      throw new Error('No model specified and no default model set. Use setDefaultModel() or pass a model parameter.');
    }

    try {
      const stream = await this.client.responses.create({
        model: modelToUse,
        input,
        temperature: params.temperature,
        max_output_tokens: params.max_tokens,
        top_p: params.top_p,
        stream: true,
      });

      return this.streamToAsyncIterable(stream);
    } catch (error) {
      console.error('LM Studio streaming error:', error);
      throw new Error(`Failed to generate stream response: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private async* streamToAsyncIterable(stream: any): AsyncIterable<string> {
    for await (const chunk of stream) {
      // Responses API uses delta.output for streaming content
      const content = chunk.delta?.output || chunk.output;
      if (content) {
        yield content;
      }
    }
  }

  updateConfig(newConfig: Partial<LMStudioConfig>): void {
    this.config = { ...this.config, ...newConfig };

    this.client = new OpenAI({
      baseURL: this.config.baseUrl,
      apiKey: 'lm-studio',
      timeout: this.config.timeout,
    });
  }
}