import { z } from 'zod';

export const ModelParamsSchema = z.object({
  temperature: z.number().min(0).max(2).optional(),
  max_tokens: z.number().min(1).max(8192).optional(),
  top_p: z.number().min(0).max(1).optional(),
  frequency_penalty: z.number().min(-2).max(2).optional(),
  presence_penalty: z.number().min(-2).max(2).optional(),
  stop: z.array(z.string()).optional(),
});

export const AnalysisTypeSchema = z.enum([
  'sentiment',
  'entities',
  'classification',
  'summary',
  'key_points',
  'privacy_scan',
  'security_audit'
]);

export const DomainTypeSchema = z.enum([
  'general',
  'medical',
  'legal',
  'financial',
  'technical',
  'academic'
]);

export const PrivacyLevelSchema = z.enum([
  'strict',
  'moderate',
  'minimal'
]);

export type ModelParams = z.infer<typeof ModelParamsSchema>;
export type AnalysisType = z.infer<typeof AnalysisTypeSchema>;
export type DomainType = z.infer<typeof DomainTypeSchema>;
export type PrivacyLevel = z.infer<typeof PrivacyLevelSchema>;

export interface LMStudioConfig {
  baseUrl: string;
  model: string;
  timeout: number;
  retries: number;
  adaptiveTimeout?: boolean;
}

export interface MCPResponse {
  content: Array<{
    type: 'text' | 'resource';
    text?: string;
    resource?: string;
  }>;
  isError?: boolean;
}