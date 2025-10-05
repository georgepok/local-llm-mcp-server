import type { Prompt } from '@modelcontextprotocol/sdk/types.js';

export class PromptTemplates {
  private templates: Map<string, Prompt> = new Map();

  constructor() {
    this.initializeDefaultTemplates();
  }

  private initializeDefaultTemplates(): void {
    // Privacy-focused prompts
    this.templates.set('privacy-analysis', {
      name: 'privacy-analysis',
      description: 'Analyze content for privacy and data protection issues',
      arguments: [
        {
          name: 'content',
          description: 'Content to analyze for privacy issues',
          required: true,
        },
        {
          name: 'regulation',
          description: 'Specific regulation to check against (GDPR, CCPA, HIPAA, etc.)',
          required: false,
        },
      ],
    });

    this.templates.set('secure-rewrite', {
      name: 'secure-rewrite',
      description: 'Rewrite content while maintaining privacy and removing sensitive information',
      arguments: [
        {
          name: 'content',
          description: 'Original content to rewrite',
          required: true,
        },
        {
          name: 'target_audience',
          description: 'Target audience for the rewritten content',
          required: false,
        },
        {
          name: 'privacy_level',
          description: 'Level of privacy protection (strict, moderate, minimal)',
          required: false,
        },
      ],
    });

    // Analysis prompts
    this.templates.set('document-analysis', {
      name: 'document-analysis',
      description: 'Comprehensive analysis of documents including sentiment, entities, and key insights',
      arguments: [
        {
          name: 'document',
          description: 'Document content to analyze',
          required: true,
        },
        {
          name: 'focus_areas',
          description: 'Specific areas to focus on (comma-separated)',
          required: false,
        },
        {
          name: 'domain',
          description: 'Domain context (medical, legal, financial, etc.)',
          required: false,
        },
      ],
    });

    this.templates.set('risk-assessment', {
      name: 'risk-assessment',
      description: 'Assess risks and provide mitigation strategies',
      arguments: [
        {
          name: 'scenario',
          description: 'Scenario or situation to assess',
          required: true,
        },
        {
          name: 'risk_types',
          description: 'Types of risks to focus on (security, financial, operational, etc.)',
          required: false,
        },
        {
          name: 'context',
          description: 'Additional context for the assessment',
          required: false,
        },
      ],
    });

    // Code-focused prompts
    this.templates.set('code-security-review', {
      name: 'code-security-review',
      description: 'Security-focused code review and vulnerability assessment',
      arguments: [
        {
          name: 'code',
          description: 'Code to review',
          required: true,
        },
        {
          name: 'language',
          description: 'Programming language',
          required: true,
        },
        {
          name: 'security_focus',
          description: 'Specific security areas to focus on',
          required: false,
        },
      ],
    });

    this.templates.set('code-optimization', {
      name: 'code-optimization',
      description: 'Analyze and suggest optimizations for code performance and maintainability',
      arguments: [
        {
          name: 'code',
          description: 'Code to optimize',
          required: true,
        },
        {
          name: 'language',
          description: 'Programming language',
          required: true,
        },
        {
          name: 'optimization_goals',
          description: 'Optimization goals (performance, readability, maintainability)',
          required: false,
        },
      ],
    });

    // Business and professional prompts
    this.templates.set('meeting-summary', {
      name: 'meeting-summary',
      description: 'Summarize meeting content with action items and key decisions',
      arguments: [
        {
          name: 'meeting_content',
          description: 'Meeting transcript or notes',
          required: true,
        },
        {
          name: 'participants',
          description: 'Meeting participants (optional for context)',
          required: false,
        },
        {
          name: 'focus_areas',
          description: 'Specific areas to focus on in the summary',
          required: false,
        },
      ],
    });

    this.templates.set('email-draft', {
      name: 'email-draft',
      description: 'Draft professional emails based on context and requirements',
      arguments: [
        {
          name: 'purpose',
          description: 'Purpose of the email',
          required: true,
        },
        {
          name: 'recipient',
          description: 'Recipient information or role',
          required: true,
        },
        {
          name: 'tone',
          description: 'Desired tone (formal, casual, friendly, etc.)',
          required: false,
        },
        {
          name: 'key_points',
          description: 'Key points to include',
          required: false,
        },
      ],
    });

    // Research and academic prompts
    this.templates.set('research-synthesis', {
      name: 'research-synthesis',
      description: 'Synthesize multiple research sources into coherent insights',
      arguments: [
        {
          name: 'sources',
          description: 'Research sources or documents to synthesize',
          required: true,
        },
        {
          name: 'research_question',
          description: 'Main research question or focus',
          required: false,
        },
        {
          name: 'output_format',
          description: 'Desired output format (summary, report, bullet points)',
          required: false,
        },
      ],
    });

    this.templates.set('literature-review', {
      name: 'literature-review',
      description: 'Conduct systematic literature review and identify gaps',
      arguments: [
        {
          name: 'topic',
          description: 'Research topic or domain',
          required: true,
        },
        {
          name: 'sources',
          description: 'Literature sources to review',
          required: true,
        },
        {
          name: 'methodology',
          description: 'Review methodology or criteria',
          required: false,
        },
      ],
    });

    // Creative and content prompts
    this.templates.set('content-adaptation', {
      name: 'content-adaptation',
      description: 'Adapt content for different audiences, platforms, or formats',
      arguments: [
        {
          name: 'original_content',
          description: 'Original content to adapt',
          required: true,
        },
        {
          name: 'target_format',
          description: 'Target format (blog post, social media, presentation, etc.)',
          required: true,
        },
        {
          name: 'target_audience',
          description: 'Target audience characteristics',
          required: false,
        },
      ],
    });

    this.templates.set('technical-documentation', {
      name: 'technical-documentation',
      description: 'Create technical documentation from code or specifications',
      arguments: [
        {
          name: 'source_material',
          description: 'Source code, API, or technical specifications',
          required: true,
        },
        {
          name: 'doc_type',
          description: 'Type of documentation (API docs, user guide, technical specs)',
          required: true,
        },
        {
          name: 'audience_level',
          description: 'Technical level of the audience (beginner, intermediate, expert)',
          required: false,
        },
      ],
    });
  }

  getAllPrompts(): Prompt[] {
    return Array.from(this.templates.values());
  }

  getPrompt(name: string, args?: Record<string, unknown>): {
    description: string;
    messages: Array<{ role: 'user' | 'assistant'; content: { type: 'text'; text: string } }>;
  } {
    const template = this.templates.get(name);
    if (!template) {
      throw new Error(`Prompt template '${name}' not found`);
    }

    const promptContent = this.generatePromptContent(name, args || {});

    return {
      description: template.description || 'Custom prompt template',
      messages: [
        {
          role: 'user',
          content: {
            type: 'text',
            text: promptContent,
          },
        },
      ],
    };
  }

  private generatePromptContent(name: string, args: Record<string, unknown>): string {
    switch (name) {
      case 'privacy-analysis':
        return this.generatePrivacyAnalysisPrompt(args);
      case 'secure-rewrite':
        return this.generateSecureRewritePrompt(args);
      case 'document-analysis':
        return this.generateDocumentAnalysisPrompt(args);
      case 'risk-assessment':
        return this.generateRiskAssessmentPrompt(args);
      case 'code-security-review':
        return this.generateCodeSecurityReviewPrompt(args);
      case 'code-optimization':
        return this.generateCodeOptimizationPrompt(args);
      case 'meeting-summary':
        return this.generateMeetingSummaryPrompt(args);
      case 'email-draft':
        return this.generateEmailDraftPrompt(args);
      case 'research-synthesis':
        return this.generateResearchSynthesisPrompt(args);
      case 'literature-review':
        return this.generateLiteratureReviewPrompt(args);
      case 'content-adaptation':
        return this.generateContentAdaptationPrompt(args);
      case 'technical-documentation':
        return this.generateTechnicalDocumentationPrompt(args);
      default:
        throw new Error(`No prompt generator for template '${name}'`);
    }
  }

  private generatePrivacyAnalysisPrompt(args: Record<string, unknown>): string {
    const content = args.content as string;
    const regulation = args.regulation as string || 'general privacy principles';

    return `Analyze the following content for privacy and data protection issues according to ${regulation}:

Content to analyze:
${content}

Please provide:
1. Identified privacy concerns
2. Specific data protection issues
3. Compliance assessment
4. Recommended actions
5. Risk level (low/medium/high)

Focus on identifying:
- Personal Identifiable Information (PII)
- Sensitive personal data
- Data collection and processing activities
- Consent and transparency issues
- Data retention and deletion practices`;
  }

  private generateSecureRewritePrompt(args: Record<string, unknown>): string {
    const content = args.original_content as string;
    const targetAudience = args.target_audience as string || 'general audience';
    const privacyLevel = args.privacy_level as string || 'moderate';

    return `Rewrite the following content while maintaining privacy and removing sensitive information:

Original content:
${content}

Requirements:
- Target audience: ${targetAudience}
- Privacy protection level: ${privacyLevel}
- Maintain the core message and intent
- Remove or generalize personal information
- Ensure compliance with privacy standards

Please provide the rewritten content that preserves meaning while protecting privacy.`;
  }

  private generateDocumentAnalysisPrompt(args: Record<string, unknown>): string {
    const document = args.document as string;
    const focusAreas = args.focus_areas as string || 'general analysis';
    const domain = args.domain as string || 'general';

    return `Perform a comprehensive analysis of the following document:

Document:
${document}

Analysis parameters:
- Domain context: ${domain}
- Focus areas: ${focusAreas}

Please provide:
1. Executive summary
2. Key themes and topics
3. Sentiment analysis
4. Important entities and relationships
5. Action items or recommendations
6. Potential risks or concerns
7. Overall assessment and insights`;
  }

  private generateRiskAssessmentPrompt(args: Record<string, unknown>): string {
    const scenario = args.scenario as string;
    const riskTypes = args.risk_types as string || 'all risk types';
    const context = args.context as string || '';

    return `Conduct a risk assessment for the following scenario:

Scenario:
${scenario}

${context ? `Additional context:\n${context}\n` : ''}

Risk assessment focus: ${riskTypes}

Please provide:
1. Risk identification and categorization
2. Probability and impact assessment
3. Risk severity ratings
4. Mitigation strategies
5. Contingency plans
6. Monitoring and review recommendations
7. Overall risk summary`;
  }

  private generateCodeSecurityReviewPrompt(args: Record<string, unknown>): string {
    const code = args.code as string;
    const language = args.language as string;
    const securityFocus = args.security_focus as string || 'comprehensive security review';

    return `Perform a security-focused code review for the following ${language} code:

Code:
\`\`\`${language}
${code}
\`\`\`

Security focus: ${securityFocus}

Please analyze for:
1. Common vulnerabilities (OWASP Top 10)
2. Input validation and sanitization
3. Authentication and authorization issues
4. Data exposure risks
5. Injection vulnerabilities
6. Cryptographic issues
7. Error handling and logging
8. Dependency security

Provide specific recommendations for each identified issue.`;
  }

  private generateCodeOptimizationPrompt(args: Record<string, unknown>): string {
    const code = args.code as string;
    const language = args.language as string;
    const optimizationGoals = args.optimization_goals as string || 'performance and maintainability';

    return `Analyze and optimize the following ${language} code:

Code:
\`\`\`${language}
${code}
\`\`\`

Optimization goals: ${optimizationGoals}

Please provide:
1. Performance analysis and bottlenecks
2. Code quality assessment
3. Maintainability improvements
4. Best practices recommendations
5. Refactoring suggestions
6. Alternative approaches
7. Optimized code examples
8. Testing recommendations`;
  }

  private generateMeetingSummaryPrompt(args: Record<string, unknown>): string {
    const meetingContent = args.meeting_content as string;
    const participants = args.participants as string || '';
    const focusAreas = args.focus_areas as string || 'comprehensive summary';

    return `Summarize the following meeting content:

Meeting content:
${meetingContent}

${participants ? `Participants: ${participants}\n` : ''}
Focus areas: ${focusAreas}

Please provide:
1. Meeting overview and purpose
2. Key discussion points
3. Decisions made
4. Action items with owners and deadlines
5. Follow-up requirements
6. Next steps
7. Unresolved issues
8. Summary of outcomes`;
  }

  private generateEmailDraftPrompt(args: Record<string, unknown>): string {
    const purpose = args.purpose as string;
    const recipient = args.recipient as string;
    const tone = args.tone as string || 'professional';
    const keyPoints = args.key_points as string || '';

    return `Draft a professional email with the following parameters:

Purpose: ${purpose}
Recipient: ${recipient}
Tone: ${tone}
${keyPoints ? `Key points to include: ${keyPoints}\n` : ''}

Please provide:
1. Appropriate subject line
2. Professional greeting
3. Clear and concise body content
4. Professional closing
5. Call to action (if applicable)

Ensure the email is well-structured, appropriate for the recipient, and achieves its intended purpose.`;
  }

  private generateResearchSynthesisPrompt(args: Record<string, unknown>): string {
    const sources = args.sources as string;
    const researchQuestion = args.research_question as string || '';
    const outputFormat = args.output_format as string || 'comprehensive synthesis';

    return `Synthesize the following research sources into coherent insights:

Research sources:
${sources}

${researchQuestion ? `Research question: ${researchQuestion}\n` : ''}
Output format: ${outputFormat}

Please provide:
1. Synthesis of key findings
2. Common themes and patterns
3. Conflicting viewpoints and contradictions
4. Research gaps identified
5. Implications and significance
6. Future research directions
7. Practical applications
8. Overall conclusions`;
  }

  private generateLiteratureReviewPrompt(args: Record<string, unknown>): string {
    const topic = args.topic as string;
    const sources = args.sources as string;
    const methodology = args.methodology as string || 'systematic review';

    return `Conduct a literature review on the following topic:

Topic: ${topic}
Methodology: ${methodology}

Literature sources:
${sources}

Please provide:
1. Introduction and scope
2. Search methodology and criteria
3. Source categorization and analysis
4. Thematic analysis of findings
5. Theoretical frameworks identified
6. Research gaps and limitations
7. Emerging trends and developments
8. Recommendations for future research
9. Conclusion and synthesis`;
  }

  private generateContentAdaptationPrompt(args: Record<string, unknown>): string {
    const originalContent = args.original_content as string;
    const targetFormat = args.target_format as string;
    const targetAudience = args.target_audience as string || 'general audience';

    return `Adapt the following content for a new format and audience:

Original content:
${originalContent}

Target format: ${targetFormat}
Target audience: ${targetAudience}

Please provide:
1. Adapted content appropriate for the target format
2. Audience-specific language and tone adjustments
3. Format-specific structural changes
4. Key message preservation
5. Engagement optimization for the target platform
6. Call-to-action adaptation (if applicable)

Ensure the adapted content maintains the core message while being optimized for the new format and audience.`;
  }

  private generateTechnicalDocumentationPrompt(args: Record<string, unknown>): string {
    const sourceMaterial = args.source_material as string;
    const docType = args.doc_type as string;
    const audienceLevel = args.audience_level as string || 'intermediate';

    return `Create technical documentation from the following source material:

Source material:
${sourceMaterial}

Documentation type: ${docType}
Audience level: ${audienceLevel}

Please provide:
1. Clear and comprehensive documentation
2. Appropriate structure and organization
3. Code examples and usage patterns (if applicable)
4. Installation or setup instructions (if applicable)
5. Configuration options and parameters
6. Best practices and recommendations
7. Troubleshooting and FAQ section
8. References and additional resources

Ensure the documentation is clear, accurate, and appropriate for the target audience level.`;
  }

  addCustomPrompt(prompt: Prompt): void {
    this.templates.set(prompt.name, prompt);
  }

  removePrompt(name: string): boolean {
    return this.templates.delete(name);
  }

  hasPrompt(name: string): boolean {
    return this.templates.has(name);
  }

  getPromptNames(): string[] {
    return Array.from(this.templates.keys());
  }
}