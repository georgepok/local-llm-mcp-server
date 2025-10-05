import type { LMStudioClient } from './lm-studio-client.js';
import type { AnalysisType, DomainType, ModelParams } from './types.js';

export class AnalysisTools {
  constructor(private lmStudio: LMStudioClient) {}

  async analyzeContent(
    content: string,
    analysisType: AnalysisType,
    domain: DomainType = 'general'
  ): Promise<any> {
    switch (analysisType) {
      case 'sentiment':
        return await this.analyzeSentiment(content, domain);
      case 'entities':
        return await this.extractEntities(content, domain);
      case 'classification':
        return await this.classifyContent(content, domain);
      case 'summary':
        return await this.summarizeContent(content, domain);
      case 'key_points':
        return await this.extractKeyPoints(content, domain);
      case 'privacy_scan':
        return await this.scanForPrivacyIssues(content);
      case 'security_audit':
        return await this.performSecurityAudit(content);
      default:
        throw new Error(`Unknown analysis type: ${analysisType}`);
    }
  }

  async analyzeSentiment(content: string, domain: DomainType): Promise<{
    sentiment: 'positive' | 'negative' | 'neutral';
    confidence: number;
    emotions: string[];
    reasoning: string;
  }> {
    const systemPrompt = this.getDomainSpecificPrompt(domain, 'sentiment analysis');
    const prompt = `Analyze the sentiment of the following text. Consider ${domain} domain context.

Text: ${content}

Provide your analysis in the following format:
SENTIMENT: [positive/negative/neutral]
CONFIDENCE: [0.0-1.0]
EMOTIONS: [list emotions separated by commas]
REASONING: [brief explanation]

If possible, also provide JSON format, but text format is acceptable.`;

    const response = await this.lmStudio.generateResponse(prompt, systemPrompt, {
      temperature: 0.1,
      max_tokens: 400,
    });

    // Try to parse as JSON first
    try {
      const jsonResponse = JSON.parse(response);
      if (jsonResponse.sentiment) {
        return jsonResponse;
      }
    } catch {
      // JSON parsing failed, try to extract from text
    }

    // Fallback: parse structured text response
    const lines = response.split('\n');
    let sentiment: 'positive' | 'negative' | 'neutral' = 'neutral';
    let confidence = 0.5;
    let emotions: string[] = [];
    let reasoning = response; // Use full response as reasoning fallback

    for (const line of lines) {
      const trimmed = line.trim().toLowerCase();

      if (trimmed.includes('sentiment:') || trimmed.includes('sentiment is')) {
        if (trimmed.includes('positive')) sentiment = 'positive';
        else if (trimmed.includes('negative')) sentiment = 'negative';
        else sentiment = 'neutral';
      }

      if (trimmed.includes('confidence:')) {
        const match = trimmed.match(/confidence:\s*([0-9.]+)/);
        if (match) confidence = Math.min(1, Math.max(0, parseFloat(match[1])));
      }

      if (trimmed.includes('emotions:')) {
        const emotionText = line.substring(line.toLowerCase().indexOf('emotions:') + 9);
        emotions = emotionText.split(',').map(e => e.trim()).filter(e => e.length > 0);
      }

      if (trimmed.includes('reasoning:')) {
        reasoning = line.substring(line.toLowerCase().indexOf('reasoning:') + 10).trim();
      }
    }

    // Smart sentiment detection if not explicitly found
    if (sentiment === 'neutral' && confidence === 0.5) {
      const lowerResponse = response.toLowerCase();
      const positiveWords = ['positive', 'good', 'great', 'excellent', 'happy', 'pleased', 'satisfied', 'excited'];
      const negativeWords = ['negative', 'bad', 'terrible', 'awful', 'sad', 'angry', 'disappointed', 'frustrated'];

      const positiveCount = positiveWords.filter(word => lowerResponse.includes(word)).length;
      const negativeCount = negativeWords.filter(word => lowerResponse.includes(word)).length;

      if (positiveCount > negativeCount) {
        sentiment = 'positive';
        confidence = Math.min(0.8, 0.5 + (positiveCount * 0.1));
      } else if (negativeCount > positiveCount) {
        sentiment = 'negative';
        confidence = Math.min(0.8, 0.5 + (negativeCount * 0.1));
      }
    }

    return {
      sentiment,
      confidence,
      emotions,
      reasoning: reasoning || response
    };
  }

  async extractEntities(content: string, domain: DomainType): Promise<{
    people: string[];
    organizations: string[];
    locations: string[];
    dates: string[];
    domainSpecific: Record<string, string[]>;
  }> {
    const systemPrompt = this.getDomainSpecificPrompt(domain, 'entity extraction');
    const domainEntities = this.getDomainSpecificEntities(domain);

    const prompt = `Extract entities from the following text. Consider ${domain} domain context.

Text: ${content}

Extract standard entities and domain-specific entities: ${domainEntities.join(', ')}

Return JSON with:
{
  "people": ["person1", "person2"],
  "organizations": ["org1", "org2"],
  "locations": ["loc1", "loc2"],
  "dates": ["date1", "date2"],
  "domainSpecific": {
    "${domainEntities[0]}": ["item1", "item2"],
    "${domainEntities[1]}": ["item3", "item4"]
  }
}`;

    const response = await this.lmStudio.generateResponse(prompt, systemPrompt, {
      temperature: 0.1,
      max_tokens: 600,
    });

    try {
      const jsonResponse = JSON.parse(response);
      if (jsonResponse.people || jsonResponse.organizations) {
        return jsonResponse;
      }
    } catch {
      // JSON parsing failed, extract from text
    }

    // Fallback: extract entities from text response
    const people: string[] = [];
    const organizations: string[] = [];
    const locations: string[] = [];
    const dates: string[] = [];
    const domainSpecific: Record<string, string[]> = {};

    // Simple entity extraction patterns
    const lines = response.split('\n');
    for (const line of lines) {
      const lowerLine = line.toLowerCase();

      if (lowerLine.includes('people:') || lowerLine.includes('persons:')) {
        const entities = this.extractEntitiesFromLine(line);
        people.push(...entities);
      } else if (lowerLine.includes('organizations:') || lowerLine.includes('companies:')) {
        const entities = this.extractEntitiesFromLine(line);
        organizations.push(...entities);
      } else if (lowerLine.includes('locations:') || lowerLine.includes('places:')) {
        const entities = this.extractEntitiesFromLine(line);
        locations.push(...entities);
      } else if (lowerLine.includes('dates:') || lowerLine.includes('times:')) {
        const entities = this.extractEntitiesFromLine(line);
        dates.push(...entities);
      }
    }

    return {
      people: [...new Set(people)], // Remove duplicates
      organizations: [...new Set(organizations)],
      locations: [...new Set(locations)],
      dates: [...new Set(dates)],
      domainSpecific,
    };
  }

  async classifyContent(content: string, domain: DomainType): Promise<{
    categories: string[];
    confidence: Record<string, number>;
    tags: string[];
    complexity: 'low' | 'medium' | 'high';
  }> {
    const categories = this.getDomainCategories(domain);
    const systemPrompt = this.getDomainSpecificPrompt(domain, 'content classification');

    const prompt = `Classify the following content within the ${domain} domain.

Text: ${content}

Available categories: ${categories.join(', ')}

Return JSON with:
{
  "categories": ["category1", "category2"],
  "confidence": {"category1": 0.8, "category2": 0.6},
  "tags": ["tag1", "tag2", "tag3"],
  "complexity": "low|medium|high"
}`;

    const response = await this.lmStudio.generateResponse(prompt, systemPrompt, {
      temperature: 0.2,
      max_tokens: 500,
    });

    try {
      return JSON.parse(response);
    } catch {
      return {
        categories: ['unknown'],
        confidence: { unknown: 0.5 },
        tags: [],
        complexity: 'medium',
      };
    }
  }

  async summarizeContent(content: string, domain: DomainType): Promise<{
    summary: string;
    keyPoints: string[];
    wordCount: number;
    compressionRatio: number;
  }> {
    const systemPrompt = this.getDomainSpecificPrompt(domain, 'summarization');
    const prompt = `Summarize the following ${domain} content, focusing on the most important information.

Text: ${content}

Provide a concise summary and identify key points.`;

    const summary = await this.lmStudio.generateResponse(prompt, systemPrompt, {
      temperature: 0.3,
      max_tokens: 800,
    });

    // Extract key points
    const keyPointsPrompt = `Extract 3-5 key points from this summary:

${summary}

Return as a simple list, one point per line starting with "- ".`;

    const keyPointsResponse = await this.lmStudio.generateResponse(keyPointsPrompt, '', {
      temperature: 0.2,
      max_tokens: 300,
    });

    const keyPoints = keyPointsResponse
      .split('\n')
      .filter(line => line.trim().startsWith('-'))
      .map(line => line.trim().substring(2));

    const originalWords = content.split(/\s+/).length;
    const summaryWords = summary.split(/\s+/).length;

    return {
      summary,
      keyPoints,
      wordCount: summaryWords,
      compressionRatio: originalWords / summaryWords,
    };
  }

  async extractKeyPoints(content: string, domain: DomainType): Promise<{
    keyPoints: string[];
    importance: Record<string, number>;
    actionItems: string[];
    insights: string[];
  }> {
    const systemPrompt = this.getDomainSpecificPrompt(domain, 'key point extraction');
    const prompt = `Extract key points from the following ${domain} content:

Text: ${content}

Identify:
1. Main key points
2. Action items (if any)
3. Important insights
4. Rate importance of each key point (1-10)

Return JSON with the following structure:
{
  "keyPoints": ["point1", "point2"],
  "importance": {"point1": 8, "point2": 6},
  "actionItems": ["action1"],
  "insights": ["insight1"]
}

If JSON format is not possible, provide a structured text response with clear sections.`;

    const response = await this.lmStudio.generateResponse(prompt, systemPrompt, {
      temperature: 0.2,
      max_tokens: 700,
    });

    // Try to parse as JSON first
    try {
      const jsonResponse = JSON.parse(response);
      if (jsonResponse.keyPoints) {
        return {
          keyPoints: jsonResponse.keyPoints || [],
          importance: jsonResponse.importance || {},
          actionItems: jsonResponse.actionItems || [],
          insights: jsonResponse.insights || [],
        };
      }
    } catch {
      // JSON parsing failed, extract from text
    }

    // Fallback: parse structured text response
    const lines = response.split('\n');
    const keyPoints: string[] = [];
    const importance: Record<string, number> = {};
    const actionItems: string[] = [];
    const insights: string[] = [];

    let currentSection = '';

    for (const line of lines) {
      const trimmed = line.trim();
      const lowerTrimmed = trimmed.toLowerCase();

      // Identify sections
      if (lowerTrimmed.includes('key point') || lowerTrimmed.includes('main point')) {
        currentSection = 'keyPoints';
        continue;
      }
      if (lowerTrimmed.includes('action item') || lowerTrimmed.includes('next step')) {
        currentSection = 'actionItems';
        continue;
      }
      if (lowerTrimmed.includes('insight') || lowerTrimmed.includes('observation')) {
        currentSection = 'insights';
        continue;
      }

      // Skip empty lines and section headers
      if (!trimmed || trimmed.length < 3) continue;

      // Parse content based on current section
      if (trimmed.startsWith('-') || trimmed.startsWith('•') || /^\d+\./.test(trimmed)) {
        const content = trimmed.replace(/^[-•\d.]\s*/, '').trim();

        // Extract importance rating if present (e.g., "point (8/10)" or "point - importance: 7")
        let importanceScore: number | undefined;
        const importanceMatch = content.match(/\((\d+)(?:\/10)?\)|importance:\s*(\d+)/i);
        if (importanceMatch) {
          importanceScore = parseInt(importanceMatch[1] || importanceMatch[2]);
        }

        const cleanContent = content.replace(/\s*\((\d+)(?:\/10)?\)|\s*-?\s*importance:\s*\d+/gi, '').trim();

        switch (currentSection) {
          case 'keyPoints':
            if (cleanContent) {
              keyPoints.push(cleanContent);
              if (importanceScore) {
                importance[cleanContent] = importanceScore;
              }
            }
            break;
          case 'actionItems':
            if (cleanContent) actionItems.push(cleanContent);
            break;
          case 'insights':
            if (cleanContent) insights.push(cleanContent);
            break;
          default:
            // If no section identified yet, treat as key point
            if (cleanContent && keyPoints.length === 0) {
              keyPoints.push(cleanContent);
            }
        }
      }
    }

    // If still no key points found, extract from the full text
    if (keyPoints.length === 0) {
      const sentences = response.split(/[.!?]+/).filter(s => s.trim().length > 10);
      const mainSentences = sentences.slice(0, 3).map(s => s.trim());
      keyPoints.push(...mainSentences);
    }

    return {
      keyPoints: keyPoints.length > 0 ? keyPoints : ['Analysis completed - see raw response for details'],
      importance,
      actionItems,
      insights,
    };
  }

  async scanForPrivacyIssues(content: string): Promise<{
    issues: Array<{
      type: string;
      severity: 'low' | 'medium' | 'high';
      description: string;
      location: string;
    }>;
    overallRisk: 'low' | 'medium' | 'high';
    recommendations: string[];
  }> {
    const systemPrompt = `You are a privacy compliance expert. Scan content for privacy issues including:
    - Personal Identifiable Information (PII)
    - GDPR compliance issues
    - Data protection violations
    - Sensitive personal data exposure`;

    const prompt = `Scan the following content for privacy issues:

${content}

Return JSON with detailed privacy analysis including issue types, severity, and recommendations.`;

    const response = await this.lmStudio.generateResponse(prompt, systemPrompt, {
      temperature: 0.1,
      max_tokens: 800,
    });

    try {
      return JSON.parse(response);
    } catch {
      return {
        issues: [],
        overallRisk: 'medium',
        recommendations: ['Manual review required due to analysis error'],
      };
    }
  }

  async performSecurityAudit(content: string): Promise<{
    vulnerabilities: Array<{
      type: string;
      severity: 'low' | 'medium' | 'high' | 'critical';
      description: string;
      recommendation: string;
    }>;
    securityScore: number;
    categories: string[];
  }> {
    const systemPrompt = `You are a cybersecurity expert. Audit content for security issues including:
    - Code vulnerabilities
    - Security misconfigurations
    - Exposed credentials or secrets
    - Insecure practices`;

    const prompt = `Perform a security audit of the following content:

${content}

Return JSON with vulnerability analysis, security score (0-100), and categorized findings.`;

    const response = await this.lmStudio.generateResponse(prompt, systemPrompt, {
      temperature: 0.1,
      max_tokens: 1000,
    });

    try {
      return JSON.parse(response);
    } catch {
      return {
        vulnerabilities: [],
        securityScore: 50,
        categories: ['analysis_error'],
      };
    }
  }

  async analyzeCode(
    code: string,
    language: string,
    analysisFocus: string
  ): Promise<{
    analysis: string;
    issues: Array<{
      line?: number;
      severity: string;
      description: string;
      suggestion: string;
    }>;
    metrics: Record<string, number>;
    recommendations: string[];
  }> {
    const systemPrompt = `You are an expert ${language} developer. Analyze code focusing on ${analysisFocus}.

    Provide detailed analysis including:
    - Code quality assessment
    - Specific issues with line numbers when possible
    - Improvement suggestions
    - Relevant metrics`;

    const prompt = `Analyze this ${language} code with focus on ${analysisFocus}:

\`\`\`${language}
${code}
\`\`\`

Please provide:
1. ANALYSIS: Overall assessment
2. ISSUES: List specific problems (format: "LINE X: SEVERITY - Description - Suggestion")
3. RECOMMENDATIONS: List of improvement suggestions
4. METRICS: Any relevant metrics

You can use JSON format if possible, but structured text is also acceptable.`;

    const response = await this.lmStudio.generateResponse(prompt, systemPrompt, {
      temperature: 0.2,
      max_tokens: 1200,
    });

    // Try to parse as JSON first
    try {
      const jsonResponse = JSON.parse(response);
      if (jsonResponse.analysis || jsonResponse.issues) {
        return {
          analysis: jsonResponse.analysis || 'Code analysis completed',
          issues: jsonResponse.issues || [],
          metrics: jsonResponse.metrics || {},
          recommendations: jsonResponse.recommendations || []
        };
      }
    } catch {
      // JSON parsing failed, try to extract from text
    }

    // Fallback: parse structured text response
    const lines = response.split('\n');
    let analysis = '';
    const issues: Array<{
      line?: number;
      severity: string;
      description: string;
      suggestion: string;
    }> = [];
    const recommendations: string[] = [];
    const metrics: Record<string, number> = {};

    let currentSection = '';

    for (const line of lines) {
      const trimmed = line.trim();
      const lowerTrimmed = trimmed.toLowerCase();

      // Identify sections
      if (lowerTrimmed.includes('analysis:') || lowerTrimmed.includes('assessment:')) {
        currentSection = 'analysis';
        const content = trimmed.substring(trimmed.toLowerCase().indexOf(':') + 1).trim();
        if (content) analysis = content;
        continue;
      }

      if (lowerTrimmed.includes('issues:') || lowerTrimmed.includes('problems:')) {
        currentSection = 'issues';
        continue;
      }

      if (lowerTrimmed.includes('recommendations:') || lowerTrimmed.includes('suggestions:')) {
        currentSection = 'recommendations';
        continue;
      }

      if (lowerTrimmed.includes('metrics:')) {
        currentSection = 'metrics';
        continue;
      }

      // Parse content based on current section
      if (trimmed.length === 0) continue;

      switch (currentSection) {
        case 'analysis':
          if (analysis && !analysis.endsWith('.')) analysis += ' ';
          analysis += trimmed;
          break;

        case 'issues':
          if (trimmed.startsWith('-') || trimmed.startsWith('•') || /^\d+\./.test(trimmed)) {
            const issueText = trimmed.replace(/^[-•\d.]\s*/, '');

            // Try to extract line number, severity, description, and suggestion
            let lineNumber: number | undefined;
            let severity = 'medium';
            let description = issueText;
            let suggestion = 'Consider reviewing this issue';

            // Look for line number pattern
            const lineMatch = issueText.match(/line\s*(\d+)/i);
            if (lineMatch) {
              lineNumber = parseInt(lineMatch[1]);
            }

            // Look for severity keywords
            if (issueText.toLowerCase().includes('critical') || issueText.toLowerCase().includes('high')) {
              severity = 'high';
            } else if (issueText.toLowerCase().includes('low') || issueText.toLowerCase().includes('minor')) {
              severity = 'low';
            }

            // Split description and suggestion if possible
            const parts = issueText.split(' - ');
            if (parts.length >= 2) {
              description = parts[0];
              suggestion = parts.slice(1).join(' - ');
            }

            issues.push({
              line: lineNumber,
              severity,
              description: description.replace(/line\s*\d+:?\s*/i, '').trim(),
              suggestion
            });
          }
          break;

        case 'recommendations':
          if (trimmed.startsWith('-') || trimmed.startsWith('•') || /^\d+\./.test(trimmed)) {
            const recommendation = trimmed.replace(/^[-•\d.]\s*/, '');
            recommendations.push(recommendation);
          }
          break;

        case 'metrics':
          // Try to extract numeric metrics
          const metricMatch = trimmed.match(/(\w+):\s*(\d+(?:\.\d+)?)/);
          if (metricMatch) {
            metrics[metricMatch[1]] = parseFloat(metricMatch[2]);
          }
          break;
      }
    }

    // If no analysis found, use the whole response
    if (!analysis) {
      analysis = response.substring(0, 200) + (response.length > 200 ? '...' : '');
    }

    // Detect common security issues in the response for better issue extraction
    if (issues.length === 0 && analysisFocus.toLowerCase().includes('security')) {
      const securityKeywords = [
        { pattern: /sql.*injection/i, severity: 'high', desc: 'SQL Injection vulnerability detected' },
        { pattern: /hardcoded.*password|password.*hardcoded/i, severity: 'high', desc: 'Hardcoded credentials found' },
        { pattern: /api.*key.*exposed|exposed.*api.*key/i, severity: 'high', desc: 'Exposed API key detected' },
        { pattern: /input.*validation|validation.*missing/i, severity: 'medium', desc: 'Input validation issues' },
        { pattern: /cross.*site.*scripting|xss/i, severity: 'medium', desc: 'XSS vulnerability potential' }
      ];

      const lowerResponse = response.toLowerCase();
      for (const keyword of securityKeywords) {
        if (keyword.pattern.test(lowerResponse)) {
          issues.push({
            severity: keyword.severity,
            description: keyword.desc,
            suggestion: 'Review and implement security best practices'
          });
        }
      }
    }

    // Add basic metrics if none found
    if (Object.keys(metrics).length === 0) {
      metrics.linesOfCode = code.split('\n').length;
      metrics.functionsFound = (code.match(/function\s+\w+/g) || []).length;
    }

    // Add basic recommendations if none found
    if (recommendations.length === 0) {
      recommendations.push('Review code for best practices');
      if (analysisFocus.includes('security')) {
        recommendations.push('Implement input validation and sanitization');
        recommendations.push('Use parameterized queries to prevent SQL injection');
      }
    }

    return {
      analysis: analysis || 'Code analysis completed',
      issues,
      metrics,
      recommendations
    };
  }

  async completeTemplate(
    template: string,
    context: string,
    format?: string
  ): Promise<string> {
    // Extract all placeholders from the template
    const placeholderRegex = /\[([A-Z_]+)\]/g;
    const placeholders = [];
    let match;
    while ((match = placeholderRegex.exec(template)) !== null) {
      placeholders.push(match[1]);
    }

    const systemPrompt = `You are an expert template completion specialist. Your task is to fill in ALL placeholders in the template based on the provided context.

CRITICAL REQUIREMENTS:
1. Replace EVERY placeholder [PLACEHOLDER_NAME] with appropriate content
2. Use information from the context or reasonable defaults
3. Maintain professional tone and formatting
4. Ensure NO placeholders remain unfilled

Placeholders to fill: ${placeholders.join(', ')}`;

    const formatNote = format ? `\n\nOutput format: ${format}` : '';

    const prompt = `Complete the following template by filling in ALL placeholders. Use the context information and professional defaults where needed.

Template with placeholders to fill:
${template}

Context information:
${context}

Placeholders to replace: ${placeholders.map(p => `[${p}]`).join(', ')}

IMPORTANT:
- Replace [CUSTOMER_NAME] with appropriate customer reference
- Replace [ACTION] with relevant action from context
- Replace [SUBJECT] with main topic
- Replace [REQUEST_TYPE] with type of request
- Replace [OUTCOME] with result/decision
- Replace [STEP_1], [STEP_2], [STEP_3] with specific next steps
- Replace [DEPARTMENT] with appropriate department
- Replace [AGENT_NAME] with professional agent name
- Replace [TITLE] with appropriate job title

Ensure ALL placeholders are replaced with meaningful content.${formatNote}

Completed template:`;

    let response = await this.lmStudio.generateResponse(prompt, systemPrompt, {
      temperature: 0.2, // Lower temperature for more consistent completion
      max_tokens: 1200,
    });

    // Post-process to ensure all placeholders are filled
    response = this.ensureAllPlaceholdersFilled(response, placeholders, context);

    return response;
  }

  private ensureAllPlaceholdersFilled(response: string, originalPlaceholders: string[], context: string): string {
    let completedResponse = response;

    // Check for any remaining placeholders and fill them
    const remainingPlaceholders = completedResponse.match(/\[([A-Z_]+)\]/g) || [];

    for (const placeholder of remainingPlaceholders) {
      const placeholderName = placeholder.replace(/[\[\]]/g, '');
      const replacement = this.getDefaultPlaceholderValue(placeholderName, context);
      completedResponse = completedResponse.replace(new RegExp(`\\${placeholder}`, 'g'), replacement);
    }

    return completedResponse;
  }

  private getDefaultPlaceholderValue(placeholderName: string, context: string): string {
    const lowerContext = context.toLowerCase();

    switch (placeholderName) {
      case 'CUSTOMER_NAME':
        // Try to extract name from context, otherwise use generic
        const nameMatch = context.match(/([A-Z][a-z]+ [A-Z][a-z]+)/);
        return nameMatch ? nameMatch[1] : 'the customer';

      case 'ACTION':
        if (lowerContext.includes('upgrade')) return 'upgrade request';
        if (lowerContext.includes('support')) return 'support inquiry';
        if (lowerContext.includes('billing')) return 'billing inquiry';
        return 'recent inquiry';

      case 'SUBJECT':
        if (lowerContext.includes('premium')) return 'premium service upgrade';
        if (lowerContext.includes('analytics')) return 'analytics features';
        if (lowerContext.includes('billing')) return 'billing services';
        return 'your account services';

      case 'REQUEST_TYPE':
        if (lowerContext.includes('upgrade')) return 'upgrade request';
        if (lowerContext.includes('support')) return 'support ticket';
        return 'service request';

      case 'OUTCOME':
        if (lowerContext.includes('approved')) return 'your request has been approved';
        if (lowerContext.includes('processed')) return 'your request is being processed';
        return 'we have reviewed your request';

      case 'STEP_1':
        if (lowerContext.includes('payment')) return 'Verify your payment method';
        if (lowerContext.includes('upgrade')) return 'Confirm your upgrade preferences';
        return 'Review the provided information';

      case 'STEP_2':
        if (lowerContext.includes('features')) return 'Explore your new features';
        if (lowerContext.includes('billing')) return 'Review your billing cycle';
        return 'Complete any required verification';

      case 'STEP_3':
        if (lowerContext.includes('billing')) return 'Monitor your updated billing';
        return 'Contact us if you have any questions';

      case 'DEPARTMENT':
        if (lowerContext.includes('billing')) return 'billing';
        if (lowerContext.includes('technical')) return 'technical support';
        return 'customer service';

      case 'AGENT_NAME':
        return 'Customer Service Team';

      case 'TITLE':
        return 'Customer Success Specialist';

      case 'AGENT_TITLE':
        return 'Customer Success Specialist';

      default:
        // Generic replacement based on placeholder name
        const formatted = placeholderName.toLowerCase().replace(/_/g, ' ');
        return `[${formatted}]`;
    }
  }

  private getDomainSpecificPrompt(domain: DomainType, task: string): string {
    const basePart = `You are an expert in ${domain} domain ${task}.`;

    switch (domain) {
      case 'medical':
        return basePart + ' Apply medical terminology and consider healthcare contexts, patient privacy, and clinical accuracy.';
      case 'legal':
        return basePart + ' Consider legal terminology, regulatory compliance, and confidentiality requirements.';
      case 'financial':
        return basePart + ' Apply financial terminology, consider regulatory requirements, and maintain confidentiality of financial data.';
      case 'technical':
        return basePart + ' Use technical terminology and consider software development, engineering, and IT contexts.';
      case 'academic':
        return basePart + ' Apply academic standards, scholarly terminology, and research methodology considerations.';
      default:
        return basePart + ' Apply general best practices and maintain professional standards.';
    }
  }

  private getDomainSpecificEntities(domain: DomainType): string[] {
    switch (domain) {
      case 'medical':
        return ['medications', 'medical_conditions', 'procedures', 'medical_devices'];
      case 'legal':
        return ['case_numbers', 'legal_terms', 'court_names', 'statutes'];
      case 'financial':
        return ['financial_instruments', 'currencies', 'market_terms', 'institutions'];
      case 'technical':
        return ['technologies', 'programming_languages', 'software_tools', 'protocols'];
      case 'academic':
        return ['research_fields', 'methodologies', 'citations', 'academic_institutions'];
      default:
        return ['topics', 'concepts', 'terms', 'references'];
    }
  }

  private extractEntitiesFromLine(line: string): string[] {
    const colonIndex = line.indexOf(':');
    if (colonIndex === -1) return [];

    const content = line.substring(colonIndex + 1).trim();
    return content.split(',')
      .map(item => item.trim())
      .filter(item => item.length > 0 && !item.toLowerCase().includes('none') && !item.toLowerCase().includes('n/a'));
  }

  private getDomainCategories(domain: DomainType): string[] {
    switch (domain) {
      case 'medical':
        return ['diagnosis', 'treatment', 'research', 'patient_care', 'clinical_trial', 'prevention'];
      case 'legal':
        return ['contract', 'litigation', 'compliance', 'regulation', 'intellectual_property', 'criminal'];
      case 'financial':
        return ['investment', 'banking', 'insurance', 'trading', 'risk_management', 'accounting'];
      case 'technical':
        return ['development', 'infrastructure', 'security', 'data', 'automation', 'integration'];
      case 'academic':
        return ['research', 'theory', 'methodology', 'analysis', 'literature_review', 'case_study'];
      default:
        return ['informational', 'instructional', 'analytical', 'opinion', 'news', 'reference'];
    }
  }
}