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
} from '@modelcontextprotocol/sdk/types.js';

import { LMStudioClient } from './lm-studio-client.js';
import { PrivacyTools } from './privacy-tools.js';
import { AnalysisTools } from './analysis-tools.js';
import { PromptTemplates } from './prompt-templates.js';
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
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
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
              model: { type: 'string', description: 'Optional specific model to use (use local://models resource to see available models)' },
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
          description: 'List all models loaded in LM Studio with current default model. Use this to discover which models you can reference in tool calls.',
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
          const models = await this.lmStudio.getAvailableModels();
          const defaultModel = this.lmStudio.getDefaultModel();
          return {
            contents: [
              {
                uri,
                mimeType: 'application/json',
                text: JSON.stringify({
                  models,
                  count: models.length,
                  defaultModel,
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

  private async handleLocalReasoning(args: any): Promise<CallToolResult> {
    const { prompt, system_prompt, model_params = {}, model } = args;

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

    const transport = new StdioServerTransport();
    await this.server.connect(transport);
  }
}

const server = new LocalLLMMCPServer();
server.run().catch(console.error);