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
      // Try LM Studio REST API first (for loaded models only)
      const baseUrl = this.config.baseUrl.replace('/v1', '');
      const lmStudioResponse = await fetch(`${baseUrl}/api/v0/models`);
      if (lmStudioResponse.ok) {
        const data = await lmStudioResponse.json();
        return data.data
          .filter((model: any) => model.state === 'loaded')
          .map((model: any) => model.id);
      }
    } catch {
      // LM Studio API not available, try OpenAI-compatible endpoint
    }

    try {
      // Fallback to standard OpenAI /v1/models endpoint (llama.cpp, vLLM, etc.)
      const models = await this.client.models.list();
      return models.data.map(model => model.id);
    } catch (error) {
      console.error('Failed to get available models:', error);
      return [];
    }
  }

  async getAvailableModelsWithMetadata(): Promise<any[]> {
    try {
      // Try LM Studio REST API first (for detailed metadata)
      const baseUrl = this.config.baseUrl.replace('/v1', '');
      const lmStudioResponse = await fetch(`${baseUrl}/api/v0/models`);
      if (lmStudioResponse.ok) {
        const data = await lmStudioResponse.json();
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
      }
    } catch {
      // LM Studio API not available, try OpenAI-compatible endpoint
    }

    try {
      // Fallback to standard OpenAI /v1/models endpoint (llama.cpp, vLLM, etc.)
      const models = await this.client.models.list();
      return models.data.map(model => ({
        id: model.id,
        name: model.id,
        object: model.object,
        created: model.created,
        ownedBy: model.owned_by,
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

    // Try LM Studio Responses API first, then fall back to Chat Completions API
    try {
      const result = await this.tryResponsesAPI(clientToUse, modelToUse, input, params);
      if (result !== null) return result;

      // Responses API returned no visible content — fall through to Chat Completions
    } catch {
      // Responses API not available — fall through to Chat Completions
    }

    // Chat Completions API (llama.cpp, vLLM, etc.)
    const chatResult = await this.tryChatCompletionsAPI(clientToUse, modelToUse, input, params, adaptiveTimeout, prompt);

    if (!chatResult || chatResult.trim().length === 0) {
      const maxTok = params.max_tokens || 'default';
      throw new Error(
        `Model produced no visible response (max_tokens=${maxTok}). ` +
        `The model may have spent its entire token budget on internal reasoning. ` +
        `Try increasing max_tokens (e.g. 2048+) or simplifying the prompt.`
      );
    }

    return chatResult;
  }

  /**
   * Attempt the Responses API. Returns the visible content string,
   * or null if only reasoning was produced (thinking-only response).
   *
   * On thinking exhaustion, returns null so the caller falls through to
   * Chat Completions API which can disable thinking via chat_template_kwargs.
   */
  private async tryResponsesAPI(
    client: OpenAI,
    model: string,
    input: Array<{ role: 'system' | 'user'; content: string }>,
    params: Partial<ModelParams>,
  ): Promise<string | null> {
    // Stream so bytes flow continuously and undici's headersTimeout / bodyTimeout
    // (300s default) cannot abort a long generation. The OpenAI SDK's `timeout`
    // option only drives an AbortController; it does not extend undici's per-chunk
    // timers, which fire whenever the connection is idle waiting on a non-streaming
    // response. We accumulate deltas and return a single string to keep the
    // generateResponse contract unchanged.
    const stream = await client.responses.create({
      model,
      input,
      temperature: params.temperature,
      max_output_tokens: params.max_tokens,
      top_p: params.top_p,
      stream: true,
    });

    let visibleText = '';
    let sawReasoning = false;

    for await (const chunk of stream as any) {
      if (chunk.type === 'response.output_text.delta' && typeof chunk.delta === 'string') {
        visibleText += chunk.delta;
      } else if (
        chunk.type === 'response.reasoning_summary_text.delta' ||
        chunk.type === 'response.reasoning_text.delta'
      ) {
        sawReasoning = true;
      }
    }

    if (visibleText) {
      return visibleText;
    }

    if (sawReasoning) {
      console.error(
        `[thinking-retry] Responses API: model produced only reasoning ` +
        `(max_tokens=${params.max_tokens}), will retry via Chat Completions with thinking disabled`
      );
    }

    // No visible content — signal caller to try Chat Completions fallback
    return null;
  }

  /**
   * Attempt the Chat Completions API. Returns visible content string.
   *
   * If the model produces only reasoning (thinking exhaustion), retries once
   * with thinking disabled via chat_template_kwargs. This handles models that
   * expand reasoning to fill any token budget (Parkinson's Law of Reasoning).
   */
  private async tryChatCompletionsAPI(
    client: OpenAI,
    model: string,
    input: Array<{ role: 'system' | 'user'; content: string }>,
    params: Partial<ModelParams>,
    adaptiveTimeout: number,
    prompt: string,
    disableThinking = false,
  ): Promise<string> {
    try {
      const messages: Array<{ role: 'system' | 'user'; content: string }> = input;

      // Build request body — optionally disable thinking on retry.
      // Always stream: a non-streaming completion holds the connection idle
      // until generation finishes, which lets undici's headersTimeout (~300s)
      // abort before our configured timeout even though tokens are still being
      // produced. Streaming keeps the socket active per chunk.
      const requestBody: Record<string, any> = {
        model,
        messages,
        temperature: params.temperature,
        max_tokens: params.max_tokens,
        top_p: params.top_p,
        frequency_penalty: params.frequency_penalty,
        presence_penalty: params.presence_penalty,
        stop: params.stop,
        stream: true,
      };

      if (disableThinking) {
        // vLLM: disable thinking via chat template
        requestBody.chat_template_kwargs = { enable_thinking: false };
      }

      const stream = await (client.chat.completions.create as any)(requestBody);

      let visibleText = '';
      let sawReasoning = false;
      let finishReason: string | null | undefined;

      for await (const chunk of stream as any) {
        const choice = chunk.choices?.[0];
        if (!choice) continue;
        const delta = choice.delta || {};
        if (typeof delta.content === 'string' && delta.content.length > 0) {
          visibleText += delta.content;
        }
        if (delta.reasoning_content) {
          sawReasoning = true;
        }
        if (choice.finish_reason) {
          finishReason = choice.finish_reason;
        }
      }

      // Primary: return visible content if available
      if (visibleText) {
        return visibleText;
      }

      // Model produced only reasoning — retry with thinking disabled.
      // Doubling tokens doesn't help: thinking models expand reasoning to
      // fill any budget. Disabling thinking forces direct response.
      if (sawReasoning && !disableThinking) {
        console.error(
          `[thinking-retry] Chat Completions: model produced only reasoning ` +
          `(max_tokens=${params.max_tokens}), retrying with thinking disabled`
        );
        return this.tryChatCompletionsAPI(
          client, model, input, params,
          adaptiveTimeout, prompt, true,
        );
      }

      // No content produced — caller will raise a descriptive error
      console.error(
        `[empty-response] Chat Completions: no visible content ` +
        `(finish_reason=${finishReason}, thinking_disabled=${disableThinking}, ` +
        `has_reasoning=${sawReasoning})`
      );
      return '';
    } catch (chatError) {
      console.error('API error:', chatError);

      if (chatError instanceof Error && chatError.message.includes('timeout')) {
        const suggestions = this.getPerformanceSuggestions(prompt, params.max_tokens);
        throw new Error(`Request timed out after ${adaptiveTimeout / 1000}s. ${suggestions}`);
      }

      throw new Error(`Failed to generate response: ${chatError instanceof Error ? chatError.message : 'Unknown error'}`);
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

    // Try LM Studio Responses API first, then fall back to Chat Completions API
    try {
      const stream = await this.client.responses.create({
        model: modelToUse,
        input,
        temperature: params.temperature,
        max_output_tokens: params.max_tokens,
        top_p: params.top_p,
        stream: true,
      });

      return this.streamResponsesAPI(stream);
    } catch {
      // Responses API not available, fall back to Chat Completions API
      try {
        const messages: Array<{ role: 'system' | 'user'; content: string }> = input;
        const stream = await this.client.chat.completions.create({
          model: modelToUse,
          messages,
          temperature: params.temperature,
          max_tokens: params.max_tokens,
          top_p: params.top_p,
          stream: true,
        });

        return this.streamChatCompletionsAPI(stream);
      } catch (error) {
        console.error('Streaming error:', error);
        throw new Error(`Failed to generate stream response: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }
  }

  private async* streamResponsesAPI(stream: any): AsyncIterable<string> {
    for await (const chunk of stream) {
      // Only yield visible output text, never reasoning/thinking content
      if (chunk.type === 'response.output_text.delta' && chunk.delta) {
        yield chunk.delta;
      } else if (chunk.type !== 'response.reasoning_summary_text.delta') {
        // Fallback for non-typed chunks: use output but skip reasoning
        const content = chunk.delta?.output || chunk.output;
        if (content && typeof content === 'string') {
          yield content;
        }
      }
    }
  }

  private async* streamChatCompletionsAPI(stream: any): AsyncIterable<string> {
    for await (const chunk of stream) {
      // Chat Completions API uses delta.content for streaming
      const content = chunk.choices[0]?.delta?.content;
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