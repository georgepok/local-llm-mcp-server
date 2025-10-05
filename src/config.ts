import { z } from 'zod';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { join } from 'path';
import type { LMStudioConfig, ModelParams } from './types.js';

const ConfigSchema = z.object({
  lmStudio: z.object({
    baseUrl: z.string().url().default('http://localhost:1234/v1'),
    defaultModel: z.string().default('local-model'),
    timeout: z.number().min(1000).max(600000).default(120000), // Increased max to 10 minutes, default 2 minutes
    retries: z.number().min(0).max(10).default(3),
    adaptiveTimeout: z.boolean().default(true),
  }),
  models: z.record(z.object({
    name: z.string(),
    description: z.string(),
    capabilities: z.array(z.string()),
    defaultParams: z.object({
      temperature: z.number().min(0).max(2).default(0.7),
      max_tokens: z.number().min(1).max(8192).default(1000),
      top_p: z.number().min(0).max(1).default(0.9),
      frequency_penalty: z.number().min(-2).max(2).default(0),
      presence_penalty: z.number().min(-2).max(2).default(0),
    }).partial(),
  })),
  privacy: z.object({
    defaultLevel: z.enum(['strict', 'moderate', 'minimal']).default('moderate'),
    enableLogging: z.boolean().default(false),
    logRetentionDays: z.number().min(0).max(365).default(7),
  }),
  performance: z.object({
    cacheEnabled: z.boolean().default(true),
    cacheTTL: z.number().min(0).max(86400).default(3600), // 1 hour
    maxConcurrentRequests: z.number().min(1).max(100).default(5),
    requestTimeout: z.number().min(1000).max(300000).default(60000), // 1 minute
  }),
  features: z.object({
    enableStreamingResponses: z.boolean().default(true),
    enableMultimodalSupport: z.boolean().default(false),
    enableCustomPrompts: z.boolean().default(true),
    enableAnalytics: z.boolean().default(false),
  }),
});

export type ServerConfig = z.infer<typeof ConfigSchema>;

export class ConfigManager {
  private config: ServerConfig;
  private configPath: string;

  constructor(configPath?: string) {
    this.configPath = configPath || join(process.cwd(), 'config.json');
    this.config = this.loadConfig();
  }

  private loadConfig(): ServerConfig {
    if (existsSync(this.configPath)) {
      try {
        const configFile = readFileSync(this.configPath, 'utf-8');
        const parsed = JSON.parse(configFile);
        return ConfigSchema.parse(parsed);
      } catch (error) {
        console.warn(`Failed to load config from ${this.configPath}:`, error);
        return this.getDefaultConfig();
      }
    }
    return this.getDefaultConfig();
  }

  private getDefaultConfig(): ServerConfig {
    return {
      lmStudio: {
        baseUrl: 'http://localhost:1234/v1',
        defaultModel: 'local-model',
        timeout: 600000, // 10 minutes default
        retries: 3,
        adaptiveTimeout: true,
      },
      models: {
        'reasoning': {
          name: 'Reasoning Model',
          description: 'Optimized for complex reasoning tasks',
          capabilities: ['reasoning', 'analysis', 'problem-solving'],
          defaultParams: {
            temperature: 0.7,
            max_tokens: 2000,
            top_p: 0.9,
          },
        },
        'analysis': {
          name: 'Analysis Model',
          description: 'Specialized for content analysis and classification',
          capabilities: ['analysis', 'classification', 'extraction'],
          defaultParams: {
            temperature: 0.3,
            max_tokens: 1500,
            top_p: 0.8,
          },
        },
        'creative': {
          name: 'Creative Model',
          description: 'Optimized for creative writing and content generation',
          capabilities: ['writing', 'rewriting', 'creative'],
          defaultParams: {
            temperature: 0.8,
            max_tokens: 2000,
            top_p: 0.95,
          },
        },
        'privacy': {
          name: 'Privacy Model',
          description: 'Specialized for privacy-preserving tasks',
          capabilities: ['privacy', 'anonymization', 'security'],
          defaultParams: {
            temperature: 0.2,
            max_tokens: 1500,
            top_p: 0.7,
          },
        },
      },
      privacy: {
        defaultLevel: 'moderate',
        enableLogging: false,
        logRetentionDays: 7,
      },
      performance: {
        cacheEnabled: true,
        cacheTTL: 3600,
        maxConcurrentRequests: 5,
        requestTimeout: 60000,
      },
      features: {
        enableStreamingResponses: true,
        enableMultimodalSupport: false,
        enableCustomPrompts: true,
        enableAnalytics: false,
      },
    };
  }

  saveConfig(): void {
    try {
      const configJson = JSON.stringify(this.config, null, 2);
      writeFileSync(this.configPath, configJson, 'utf-8');
    } catch (error) {
      console.error(`Failed to save config to ${this.configPath}:`, error);
    }
  }

  getConfig(): ServerConfig {
    return { ...this.config };
  }

  getLMStudioConfig(): LMStudioConfig {
    return {
      baseUrl: this.config.lmStudio.baseUrl,
      model: this.config.lmStudio.defaultModel,
      timeout: this.config.lmStudio.timeout,
      retries: this.config.lmStudio.retries,
    };
  }

  getModelConfig(modelType: string): {
    config: ServerConfig['models'][string];
    params: ModelParams;
  } | null {
    const modelConfig = this.config.models[modelType];
    if (!modelConfig) {
      return null;
    }

    return {
      config: modelConfig,
      params: {
        temperature: modelConfig.defaultParams.temperature || 0.7,
        max_tokens: modelConfig.defaultParams.max_tokens || 1000,
        top_p: modelConfig.defaultParams.top_p || 0.9,
        frequency_penalty: modelConfig.defaultParams.frequency_penalty || 0,
        presence_penalty: modelConfig.defaultParams.presence_penalty || 0,
      },
    };
  }

  updateLMStudioConfig(updates: Partial<LMStudioConfig>): void {
    this.config.lmStudio = { ...this.config.lmStudio, ...updates };
    this.saveConfig();
  }

  addModel(
    key: string,
    config: {
      name: string;
      description: string;
      capabilities: string[];
      defaultParams?: Partial<ModelParams>;
    }
  ): void {
    this.config.models[key] = {
      name: config.name,
      description: config.description,
      capabilities: config.capabilities,
      defaultParams: config.defaultParams || {},
    };
    this.saveConfig();
  }

  removeModel(key: string): boolean {
    if (this.config.models[key]) {
      delete this.config.models[key];
      this.saveConfig();
      return true;
    }
    return false;
  }

  updatePrivacySettings(updates: Partial<ServerConfig['privacy']>): void {
    this.config.privacy = { ...this.config.privacy, ...updates };
    this.saveConfig();
  }

  updatePerformanceSettings(updates: Partial<ServerConfig['performance']>): void {
    this.config.performance = { ...this.config.performance, ...updates };
    this.saveConfig();
  }

  updateFeatureSettings(updates: Partial<ServerConfig['features']>): void {
    this.config.features = { ...this.config.features, ...updates };
    this.saveConfig();
  }

  getAvailableModels(): Array<{
    key: string;
    name: string;
    description: string;
    capabilities: string[];
  }> {
    return Object.entries(this.config.models).map(([key, model]) => ({
      key,
      name: model.name,
      description: model.description,
      capabilities: model.capabilities,
    }));
  }

  validateConfig(): { isValid: boolean; errors: string[] } {
    try {
      ConfigSchema.parse(this.config);
      return { isValid: true, errors: [] };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return {
          isValid: false,
          errors: error.errors.map(err => `${err.path.join('.')}: ${err.message}`),
        };
      }
      return { isValid: false, errors: ['Unknown validation error'] };
    }
  }

  resetToDefaults(): void {
    this.config = this.getDefaultConfig();
    this.saveConfig();
  }

  exportConfig(): string {
    return JSON.stringify(this.config, null, 2);
  }

  importConfig(configJson: string): { success: boolean; errors: string[] } {
    try {
      const parsed = JSON.parse(configJson);
      const validated = ConfigSchema.parse(parsed);
      this.config = validated;
      this.saveConfig();
      return { success: true, errors: [] };
    } catch (error) {
      if (error instanceof z.ZodError) {
        return {
          success: false,
          errors: error.errors.map(err => `${err.path.join('.')}: ${err.message}`),
        };
      }
      return { success: false, errors: ['Invalid JSON or configuration format'] };
    }
  }

  getModelForCapability(capability: string): string | null {
    for (const [key, model] of Object.entries(this.config.models)) {
      if (model.capabilities.includes(capability)) {
        return key;
      }
    }
    return null;
  }

  isFeatureEnabled(feature: keyof ServerConfig['features']): boolean {
    return this.config.features[feature];
  }

  getPrivacyLevel(): ServerConfig['privacy']['defaultLevel'] {
    return this.config.privacy.defaultLevel;
  }

  getPerformanceSettings(): ServerConfig['performance'] {
    return { ...this.config.performance };
  }
}