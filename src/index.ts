#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
  InitializeRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

import { LMStudioClient } from './lm-studio-client.js';
import { PrivacyTools } from './privacy-tools.js';
import { AnalysisTools } from './analysis-tools.js';
import { PromptTemplates } from './prompt-templates.js';
import { HttpTransport } from './http-transport.js';
import type { MCPResponse } from './types.js';
import type { CallToolResult } from '@modelcontextprotocol/sdk/types.js';

class LocalLLMMCPServer {
  private server: Server;
  private lmStudio: LMStudioClient;
  private privacyTools: PrivacyTools;
  private analysisTools: AnalysisTools;
  private promptTemplates: PromptTemplates;

  constructor() {
    this.server = new Server(
      {
        name: 'local-llm-mcp-server',
        version: '1.0.0',
      },
      {
        capabilities: {
          tools: {},
          resources: {},
          prompts: {},
        },
      }
    );

    this.lmStudio = new LMStudioClient();
    this.privacyTools = new PrivacyTools(this.lmStudio);
    this.analysisTools = new AnalysisTools(this.lmStudio);
    this.promptTemplates = new PromptTemplates();

    this.setupHandlers();
  }

  private setupHandlers(): void {
    // Protocol handshake - advertise server capabilities
    this.server.setRequestHandler(InitializeRequestSchema, async (request) => {
      return {
        protocolVersion: '2024-11-05',
        serverInfo: {
          name: 'local-llm-mcp-server',
          version: '2.0.0',
        },
        capabilities: {
          tools: {
            description: 'Provides local LLM inference tools for private reasoning, analysis, and code review',
            listChanged: false,
          },
          resources: {
            description: 'Provides access to model metadata, server status, and configuration',
            subscribe: false,
            listChanged: false,
          },
          prompts: {
            description: 'Provides pre-configured prompt templates for common tasks',
            listChanged: false,
          },
        },
      };
    });

    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'list_models',
          description: 'List all available LLM and embedding models with their capabilities and metadata',
          inputSchema: {
            type: 'object',
            properties: {
              type: {
                type: 'string',
                enum: ['all', 'llm', 'embedding'],
                description: 'Filter models by type (default: all)',
              },
              includeMetadata: {
                type: 'boolean',
                description: 'Include detailed model metadata (default: true)',
              },
            },
          },
        },
        {
          name: 'get_model_info',
          description: 'Get detailed information about a specific model including capabilities, parameters, and performance characteristics',
          inputSchema: {
            type: 'object',
            properties: {
              model: {
                type: 'string',
                description: 'Model identifier (e.g., "openai/gpt-oss-20b")',
              },
            },
            required: ['model'],
          },
        },
        {
          name: 'local_reasoning',
          description: 'Use local LLM for specialized reasoning tasks while keeping data private',
          inputSchema: {
            type: 'object',
            properties: {
              prompt: { type: 'string', description: 'The reasoning prompt or question' },
              system_prompt: { type: 'string', description: 'Optional system prompt for context' },
              model_params: {
                type: 'object',
                description: 'Optional model parameters (temperature, max_tokens, etc.)'
              },
              model: { type: 'string', description: 'Optional specific model to use (use list_models tool to see available models)' },
            },
            required: ['prompt'],
          },
        },
        {
          name: 'private_analysis',
          description: 'Analyze sensitive content locally without cloud exposure',
          inputSchema: {
            type: 'object',
            properties: {
              content: { type: 'string', description: 'Content to analyze' },
              analysis_type: {
                type: 'string',
                enum: ['sentiment', 'entities', 'classification', 'summary', 'key_points', 'privacy_scan', 'security_audit'],
                description: 'Type of analysis to perform'
              },
              domain: {
                type: 'string',
                enum: ['general', 'medical', 'legal', 'financial', 'technical', 'academic'],
                description: 'Domain context for specialized analysis'
              },
            },
            required: ['content', 'analysis_type'],
          },
        },
        {
          name: 'secure_rewrite',
          description: 'Rewrite or transform text locally for privacy',
          inputSchema: {
            type: 'object',
            properties: {
              content: { type: 'string', description: 'Content to rewrite' },
              style: { type: 'string', description: 'Target style or tone' },
              privacy_level: {
                type: 'string',
                enum: ['strict', 'moderate', 'minimal'],
                description: 'Level of privacy protection to apply'
              },
            },
            required: ['content', 'style'],
          },
        },
        {
          name: 'code_analysis',
          description: 'Analyze code locally for security, quality, or documentation',
          inputSchema: {
            type: 'object',
            properties: {
              code: { type: 'string', description: 'Code to analyze' },
              language: { type: 'string', description: 'Programming language' },
              analysis_focus: {
                type: 'string',
                enum: ['security', 'quality', 'documentation', 'optimization', 'bugs'],
                description: 'Focus area for code analysis'
              },
            },
            required: ['code', 'language', 'analysis_focus'],
          },
        },
        {
          name: 'template_completion',
          description: 'Complete templates or forms using local LLM',
          inputSchema: {
            type: 'object',
            properties: {
              template: { type: 'string', description: 'Template with placeholders' },
              context: { type: 'string', description: 'Context information for completion' },
              format: { type: 'string', description: 'Output format requirements' },
            },
            required: ['template', 'context'],
          },
        },
        {
          name: 'set_default_model',
          description: 'Set the default model for the session',
          inputSchema: {
            type: 'object',
            properties: {
              model: { type: 'string', description: 'Model name to set as default (use local://models resource to see available models)' },
            },
            required: ['model'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(ListResourcesRequestSchema, async () => ({
      resources: [
        {
          uri: 'local://models',
          name: 'Available Local Models',
          description: 'Lists LLM models and embedding models separately. Use llmModels for text generation tools. Returns: {llmModels: string[], embeddingModels: string[], defaultModel: string}',
          mimeType: 'application/json',
        },
        {
          uri: 'local://status',
          name: 'Server Status',
          description: 'Current status of the local LLM server (online/offline)',
          mimeType: 'application/json',
        },
        {
          uri: 'local://config',
          name: 'Server Configuration',
          description: 'Server capabilities, supported domains, privacy levels, and available analysis types',
          mimeType: 'application/json',
        },
        {
          uri: 'local://capabilities',
          name: 'Model Capabilities',
          description: 'Detailed information about what each available model can do and how to use them',
          mimeType: 'application/json',
        },
      ],
    }));

    this.server.setRequestHandler(ListPromptsRequestSchema, async () => ({
      prompts: this.promptTemplates.getAllPrompts(),
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        let result: CallToolResult;

        switch (name) {
          case 'list_models':
            result = await this.handleListModels(args);
            break;
          case 'get_model_info':
            result = await this.handleGetModelInfo(args);
            break;
          case 'local_reasoning':
            result = await this.handleLocalReasoning(args);
            break;
          case 'private_analysis':
            result = await this.handlePrivateAnalysis(args);
            break;
          case 'secure_rewrite':
            result = await this.handleSecureRewrite(args);
            break;
          case 'code_analysis':
            result = await this.handleCodeAnalysis(args);
            break;
          case 'template_completion':
            result = await this.handleTemplateCompletion(args);
            break;
          case 'set_default_model':
            result = await this.handleSetDefaultModel(args);
            break;
          default:
            throw new Error(`Unknown tool: ${name}`);
        }

        return result;
      } catch (error) {
        return {
          content: [
            {
              type: 'text',
              text: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
            },
          ],
          isError: true,
        };
      }
    });

    this.server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
      const { uri } = request.params;

      switch (uri) {
        case 'local://models':
          const allModels = await this.lmStudio.getAvailableModelsWithMetadata();
          const defaultModel = this.lmStudio.getDefaultModel();

          // Enrich models with metadata
          const enrichedModels = allModels.map(model => {
            const isEmbedding = model.id.includes('embedding') || model.id.includes('embed');
            const modelInfo = this.parseModelName(model.id);

            return {
              id: model.id,
              name: modelInfo.displayName,
              type: isEmbedding ? 'embedding' : 'llm',
              provider: modelInfo.provider,
              parameters: modelInfo.parameters,
              capabilities: isEmbedding ? ['embedding'] : ['chat', 'reasoning', 'completion'],
              contextWindow: modelInfo.contextWindow,
              isDefault: model.id === defaultModel,
              status: 'ready',
              metadata: {
                created: model.created,
                ownedBy: model.ownedBy,
                architecture: modelInfo.architecture,
              }
            };
          });

          const llmModels = enrichedModels.filter(m => m.type === 'llm');
          const embeddingModels = enrichedModels.filter(m => m.type === 'embedding');

          return {
            contents: [
              {
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                  models: enrichedModels,
                  llmModels: llmModels.map(m => m.id),
                  embeddingModels: embeddingModels.map(m => m.id),
                  defaultModel,
                  count: enrichedModels.length,
                  summary: {
                    totalModels: enrichedModels.length,
                    llmCount: llmModels.length,
                    embeddingCount: embeddingModels.length
                  }
                }, null, 2),
              },
            ],
          };

        case 'local://status':
          const isAvailable = await this.lmStudio.isAvailable();
          return {
            contents: [
              {
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                  status: isAvailable ? 'online' : 'offline',
                  timestamp: new Date().toISOString(),
                }, null, 2),
              },
            ],
          };

        case 'local://config':
          return {
            contents: [
              {
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                  server: 'local-llm-mcp-server',
                  version: '1.0.0',
                  capabilities: ['reasoning', 'analysis', 'rewriting', 'code_analysis', 'templates'],
                  privacy_levels: ['strict', 'moderate', 'minimal'],
                  supported_domains: ['general', 'medical', 'legal', 'financial', 'technical', 'academic'],
                  analysis_types: ['sentiment', 'entities', 'classification', 'summary', 'key_points', 'privacy_scan', 'security_audit'],
                }, null, 2),
              },
            ],
          };

        case 'local://capabilities':
          const capabilityModels = await this.lmStudio.getAvailableModels();
          const capabilityDefault = this.lmStudio.getDefaultModel();
          return {
            contents: [
              {
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                  modelManagement: {
                    availableModels: capabilityModels,
                    defaultModel: capabilityDefault,
                    howToUse: {
                      readModels: 'Read resource: local://models',
                      setDefault: 'Use tool: set_default_model with {"model": "model-name"}',
                      useSpecificModel: 'Add "model" parameter to any tool call: {"prompt": "...", "model": "model-name"}',
                      useDefaultModel: 'Omit "model" parameter to use the default model',
                    },
                  },
                  tools: {
                    local_reasoning: {
                      purpose: 'General-purpose reasoning and question answering',
                      parameters: {
                        prompt: 'required - Your question or task',
                        system_prompt: 'optional - Context or role for the model',
                        model_params: 'optional - {temperature, max_tokens, etc.}',
                        model: 'optional - Specific model to use (overrides default)',
                      },
                      example: '{"prompt": "Explain quantum computing", "model": "' + (capabilityModels[0] || 'model-name') + '"}',
                    },
                    private_analysis: {
                      purpose: 'Analyze content locally (sentiment, entities, classification, etc.)',
                      parameters: {
                        content: 'required - Text to analyze',
                        analysis_type: 'required - One of: sentiment, entities, classification, summary, key_points, privacy_scan, security_audit',
                        domain: 'optional - Context domain: general, medical, legal, financial, technical, academic',
                      },
                      example: '{"content": "Great product!", "analysis_type": "sentiment", "domain": "general"}',
                    },
                    secure_rewrite: {
                      purpose: 'Rewrite text with privacy protection',
                      parameters: {
                        content: 'required - Text to rewrite',
                        style: 'required - Target style (e.g., "formal", "casual", "professional")',
                        privacy_level: 'optional - strict, moderate, or minimal',
                      },
                      example: '{"content": "Hi there!", "style": "formal", "privacy_level": "moderate"}',
                    },
                    code_analysis: {
                      purpose: 'Analyze code for security, quality, bugs, etc.',
                      parameters: {
                        code: 'required - Code to analyze',
                        language: 'required - Programming language (e.g., "python", "javascript")',
                        analysis_focus: 'required - Focus: security, quality, documentation, optimization, bugs',
                      },
                      example: '{"code": "def hello():\\n    print(\\"hi\\")", "language": "python", "analysis_focus": "quality"}',
                    },
                    template_completion: {
                      purpose: 'Fill in templates with placeholders',
                      parameters: {
                        template: 'required - Template with [PLACEHOLDER] markers',
                        context: 'required - Information to use for filling placeholders',
                        format: 'optional - Output format requirements',
                      },
                      example: '{"template": "Dear [NAME],", "context": "Customer named John Smith"}',
                    },
                    set_default_model: {
                      purpose: 'Change the default model for this session',
                      parameters: {
                        model: 'required - Model name from local://models resource',
                      },
                      example: '{"model": "' + (capabilityModels[1] || capabilityModels[0] || 'model-name') + '"}',
                    },
                  },
                  workflow: {
                    step1: 'Read local://models to see available models',
                    step2: 'Optionally set a default model using set_default_model tool',
                    step3: 'Use any tool - it will use default model unless you specify "model" parameter',
                    step4: 'Override default by passing "model" parameter in any tool call',
                  },
                }, null, 2),
              },
            ],
          };

        default:
          throw new Error(`Unknown resource: ${uri}`);
      }
    });

    this.server.setRequestHandler(GetPromptRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;
      return this.promptTemplates.getPrompt(name, args);
    });
  }

  private parseModelName(modelId: string): any {
    // Parse model name to extract metadata
    const parts = modelId.split(/[-\/]/);

    let provider = 'Unknown';
    let displayName = modelId;
    let parameters = 'Unknown';
    let architecture = 'Transformer';
    let contextWindow = 8192;

    // Detect provider
    if (modelId.includes('openai')) {
      provider = 'OpenAI';
    } else if (modelId.includes('qwen')) {
      provider = 'Alibaba Cloud';
    } else if (modelId.includes('llama')) {
      provider = 'Meta';
    } else if (modelId.includes('mistral')) {
      provider = 'Mistral AI';
    } else if (modelId.includes('nomic')) {
      provider = 'Nomic AI';
    }

    // Extract parameter count
    const paramMatch = modelId.match(/(\d+)b/i);
    if (paramMatch) {
      parameters = `${paramMatch[1]}B`;
    }

    // Estimate context window based on model
    if (modelId.includes('qwen3') || modelId.includes('32k')) {
      contextWindow = 32768;
    } else if (modelId.includes('128k')) {
      contextWindow = 131072;
    }

    // Generate display name
    displayName = modelId
      .replace(/[-_]/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');

    return {
      provider,
      displayName,
      parameters,
      architecture,
      contextWindow
    };
  }

  private async handleListModels(args: any): Promise<CallToolResult> {
    const { type = 'all', includeMetadata = true } = args;

    const allModels = await this.lmStudio.getAvailableModelsWithMetadata();
    const defaultModel = this.lmStudio.getDefaultModel();

    // Enrich models
    const enrichedModels = allModels.map(model => {
      const isEmbedding = model.id.includes('embedding') || model.id.includes('embed');
      const modelInfo = this.parseModelName(model.id);

      return {
        id: model.id,
        name: modelInfo.displayName,
        type: isEmbedding ? 'embedding' : 'llm',
        provider: modelInfo.provider,
        parameters: modelInfo.parameters,
        capabilities: isEmbedding ? ['embedding'] : ['chat', 'reasoning', 'completion'],
        contextWindow: modelInfo.contextWindow,
        isDefault: model.id === defaultModel,
        status: 'ready',
        ...(includeMetadata && {
          metadata: {
            created: model.created,
            ownedBy: model.ownedBy,
            architecture: modelInfo.architecture,
          }
        })
      };
    });

    // Filter by type
    let filteredModels = enrichedModels;
    if (type === 'llm') {
      filteredModels = enrichedModels.filter(m => m.type === 'llm');
    } else if (type === 'embedding') {
      filteredModels = enrichedModels.filter(m => m.type === 'embedding');
    }

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({
          models: filteredModels,
          defaultModel,
          summary: {
            total: filteredModels.length,
            llmCount: enrichedModels.filter(m => m.type === 'llm').length,
            embeddingCount: enrichedModels.filter(m => m.type === 'embedding').length,
          }
        }, null, 2)
      }]
    };
  }

  private async handleGetModelInfo(args: any): Promise<CallToolResult> {
    const { model: modelId } = args;

    if (!modelId) {
      return {
        content: [{
          type: 'text',
          text: 'Error: model parameter is required'
        }],
        isError: true
      };
    }

    const allModels = await this.lmStudio.getAvailableModelsWithMetadata();
    const modelData = allModels.find(m => m.id === modelId);

    if (!modelData) {
      const availableIds = allModels.map(m => m.id);
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            error: {
              code: 'MODEL_NOT_FOUND',
              message: `Model '${modelId}' not found`,
              details: {
                requestedModel: modelId,
                suggestion: 'Use list_models tool to see available models'
              },
              availableModels: availableIds
            }
          }, null, 2)
        }],
        isError: true
      };
    }

    const isEmbedding = modelId.includes('embedding') || modelId.includes('embed');
    const modelInfo = this.parseModelName(modelId);
    const defaultModel = this.lmStudio.getDefaultModel();

    const detailedInfo = {
      id: modelId,
      displayName: modelInfo.displayName,
      provider: modelInfo.provider,
      architecture: modelInfo.architecture,
      parameters: modelInfo.parameters,
      type: isEmbedding ? 'embedding' : 'llm',
      capabilities: isEmbedding ? ['embedding'] : ['chat', 'reasoning', 'completion', 'code-generation'],
      contextWindow: modelInfo.contextWindow,
      maxTokens: Math.floor(modelInfo.contextWindow / 2),
      isDefault: modelId === defaultModel,
      status: 'ready',
      metadata: {
        created: modelData.created,
        ownedBy: modelData.ownedBy,
      },
      usage: {
        purpose: isEmbedding ? 'Generate vector embeddings for similarity search' : 'Text generation, reasoning, and completion tasks',
        bestFor: isEmbedding ? ['Semantic search', 'Document similarity', 'Clustering'] : ['Chat', 'Analysis', 'Code generation', 'Reasoning']
      }
    };

    return {
      content: [{
        type: 'text',
        text: JSON.stringify({ model: detailedInfo }, null, 2)
      }]
    };
  }

  private async handleLocalReasoning(args: any): Promise<CallToolResult> {
    const { prompt, system_prompt, model_params = {}, model } = args;

    // Validate model parameter - common mistake is using resource URI instead of model name
    if (model && model.startsWith('local://')) {
      return {
        content: [{
          type: 'text',
          text: `Error: Invalid model "${model}". This looks like a resource URI, not a model name.\n\nTo see available models, read the resource "local://models" first, then use one of the model names from that list (e.g., "openai/gpt-oss-20b").`
        }],
        isError: true
      };
    }

    const response = await this.lmStudio.generateResponse(
      prompt,
      system_prompt,
      model_params,
      model
    );

    return {
      content: [
        {
          type: 'text',
          text: response,
        },
      ],
    };
  }

  private async handlePrivateAnalysis(args: any): Promise<CallToolResult> {
    const { content, analysis_type, domain = 'general' } = args;

    const result = await this.analysisTools.analyzeContent(
      content,
      analysis_type,
      domain
    );

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  }

  private async handleSecureRewrite(args: any): Promise<CallToolResult> {
    const { content, style, privacy_level = 'moderate' } = args;

    const result = await this.privacyTools.secureRewrite(
      content,
      style,
      privacy_level
    );

    return {
      content: [
        {
          type: 'text',
          text: result,
        },
      ],
    };
  }

  private async handleCodeAnalysis(args: any): Promise<CallToolResult> {
    const { code, language, analysis_focus } = args;

    const result = await this.analysisTools.analyzeCode(
      code,
      language,
      analysis_focus
    );

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(result, null, 2),
        },
      ],
    };
  }

  private async handleTemplateCompletion(args: any): Promise<CallToolResult> {
    const { template, context, format } = args;

    const result = await this.analysisTools.completeTemplate(
      template,
      context,
      format
    );

    return {
      content: [
        {
          type: 'text',
          text: result,
        },
      ],
    };
  }

  private async handleSetDefaultModel(args: any): Promise<CallToolResult> {
    const { model } = args;

    // Verify model exists
    const availableModels = await this.lmStudio.getAvailableModels();
    if (!availableModels.includes(model)) {
      // Check if this is a probe request from the client
      const isProbe = model.includes('probe') || model.includes('test');

      return {
        content: [
          {
            type: 'text',
            text: JSON.stringify({
              error: `Model "${model}" not found`,
              availableModels,
              currentDefault: this.lmStudio.getDefaultModel(),
              message: isProbe
                ? 'This appears to be a probe request. Use one of the available models listed above.'
                : `Model "${model}" is not available. Choose from the available models listed above.`,
            }, null, 2),
          },
        ],
        isError: true,
      };
    }

    this.lmStudio.setDefaultModel(model);

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify({
            success: true,
            defaultModel: model,
            previousDefault: this.lmStudio.getDefaultModel(),
            message: `Default model set to "${model}"`,
          }, null, 2),
        },
      ],
    };
  }

  async run(): Promise<void> {
    // Initialize LM Studio client before connecting
    await this.lmStudio.initialize();

    // Determine transport mode from environment or CLI args
    const transportMode = process.env.MCP_TRANSPORT ||
                         (process.argv.includes('--http') ? 'http' :
                          process.argv.includes('--https') ? 'https' : 'stdio');

    // Check if dual mode is enabled (stdio + http/https)
    const enableDual = process.env.MCP_DUAL_MODE === 'true' || process.argv.includes('--dual');

    if (transportMode === 'http' || transportMode === 'https' || enableDual) {
      // HTTP/HTTPS transport for remote access
      const port = parseInt(process.env.PORT || '3000');
      const host = process.env.HOST || '0.0.0.0';
      const useHttps = transportMode === 'https' || process.env.MCP_HTTPS === 'true';

      const protocol = useHttps ? 'HTTPS' : 'HTTP';
      console.error(`[${protocol}] Starting MCP server on ${host}:${port}...`);

      const httpTransport = new HttpTransport(
        {
          port,
          host,
          cors: true,
          https: useHttps ? {
            certPath: process.env.HTTPS_CERT_PATH || '',
            keyPath: process.env.HTTPS_KEY_PATH || ''
          } : undefined
        },
        this.server
      );

      await httpTransport.start();

      console.error(`[${protocol}] MCP server ready for remote connections`);
      console.error(`[${protocol}] Access at: ${useHttps ? 'https' : 'http'}://${host}:${port}`);

      // If dual mode, also start stdio transport
      if (enableDual) {
        console.error('[Dual] Also starting stdio transport for Claude Desktop...');
        const stdioTransport = new StdioServerTransport();
        await this.server.connect(stdioTransport);
        console.error('[Dual] Both transports active: stdio + ' + protocol);
      } else {
        console.error(`[${protocol}] Press Ctrl+C to stop`);
      }

      // Keep the process alive
      process.on('SIGINT', async () => {
        console.error(`\n[${protocol}] Shutting down...`);
        await httpTransport.stop();
        process.exit(0);
      });
    } else {
      // Stdio transport for local access (default)
      // IMPORTANT: In stdio mode, stdout is reserved for JSON-RPC messages
      // Use stderr for any logging/debugging
      console.error('[Stdio] Starting MCP server in stdio mode...');
      const transport = new StdioServerTransport();
      await this.server.connect(transport);
      console.error('[Stdio] MCP server connected and ready');
    }
  }
}

const server = new LocalLLMMCPServer();
server.run().catch(console.error);