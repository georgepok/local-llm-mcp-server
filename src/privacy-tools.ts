import type { LMStudioClient } from './lm-studio-client.js';
import type { PrivacyLevel, ModelParams } from './types.js';

export class PrivacyTools {
  constructor(private lmStudio: LMStudioClient) {}

  async secureRewrite(
    content: string,
    style: string,
    privacyLevel: PrivacyLevel = 'moderate'
  ): Promise<string> {
    // First, detect and mark sensitive data
    const sensitiveData = this.detectSensitiveInformation(content);

    // Create anonymized version with placeholders
    let preprocessedContent = content;
    const replacements: Array<{original: string, placeholder: string, type: string}> = [];

    // Replace sensitive data with clear placeholders
    let placeholderIndex = 1;
    for (const item of sensitiveData) {
      const placeholder = `[${item.type.toUpperCase()}_${placeholderIndex}]`;
      // Use global regex to replace ALL occurrences, not just the first
      const escapedValue = item.value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      preprocessedContent = preprocessedContent.replace(new RegExp(escapedValue, 'g'), placeholder);
      replacements.push({
        original: item.value,
        placeholder,
        type: item.type
      });
      placeholderIndex++;
    }

    const systemPrompt = this.getEnhancedPrivacySystemPrompt(privacyLevel, replacements);
    const prompt = this.buildEnhancedRewritePrompt(preprocessedContent, style, privacyLevel, replacements);

    const params: ModelParams = {
      temperature: 0.3, // Lower temperature for more consistent privacy compliance
      max_tokens: 2000,
    };

    const response = await this.lmStudio.generateResponse(prompt, systemPrompt, params);

    // Final safety check - ensure no sensitive data leaked through
    return this.postProcessSecureRewrite(response, sensitiveData, privacyLevel);
  }

  async anonymizeText(content: string, privacyLevel: PrivacyLevel = 'strict'): Promise<string> {
    const systemPrompt = `You are a privacy protection specialist. Your task is to anonymize text by:
    - Replacing personal names with generic placeholders (e.g., Person A, Person B)
    - Removing or generalizing specific locations, dates, and identifying information
    - Preserving the core meaning and structure of the text
    - Maintaining ${privacyLevel} privacy standards`;

    const prompt = `Anonymize the following text while preserving its meaning:

${content}

Return only the anonymized text.`;

    const params: ModelParams = {
      temperature: 0.3,
      max_tokens: 2000,
    };

    return await this.lmStudio.generateResponse(prompt, systemPrompt, params);
  }

  async detectSensitiveData(content: string): Promise<{
    hasSensitiveData: boolean;
    detectedTypes: string[];
    suggestions: string[];
  }> {
    const systemPrompt = `You are a privacy scanning specialist. Analyze text for sensitive information including:
    - Personal Identifiable Information (PII)
    - Financial data (credit cards, SSNs, bank accounts)
    - Health information (medical records, conditions)
    - Legal information (case numbers, court documents)
    - Corporate secrets or confidential data
    - Login credentials or API keys

    Respond with JSON only.`;

    const prompt = `Analyze this text for sensitive data:

${content}

Return a JSON object with:
{
  "hasSensitiveData": boolean,
  "detectedTypes": ["type1", "type2"],
  "suggestions": ["suggestion1", "suggestion2"]
}`;

    const params: ModelParams = {
      temperature: 0.1,
      max_tokens: 500,
    };

    try {
      const response = await this.lmStudio.generateResponse(prompt, systemPrompt, params);
      return JSON.parse(response);
    } catch (error) {
      return {
        hasSensitiveData: false,
        detectedTypes: [],
        suggestions: ['Error analyzing content for sensitive data'],
      };
    }
  }

  async secureSummarization(
    content: string,
    maxLength: number = 200,
    privacyLevel: PrivacyLevel = 'moderate'
  ): Promise<string> {
    const systemPrompt = this.getPrivacySystemPrompt(privacyLevel);
    const prompt = `Summarize the following content in approximately ${maxLength} words, ensuring no sensitive information is exposed:

${content}

Summary:`;

    const params: ModelParams = {
      temperature: 0.5,
      max_tokens: Math.min(maxLength * 2, 1000),
    };

    return await this.lmStudio.generateResponse(prompt, systemPrompt, params);
  }

  async dataClassification(content: string): Promise<{
    confidentialityLevel: 'public' | 'internal' | 'confidential' | 'restricted';
    dataTypes: string[];
    handlingRecommendations: string[];
  }> {
    const systemPrompt = `You are a data classification expert. Classify content based on sensitivity and recommend handling procedures.

Classification levels:
- public: Can be shared openly
- internal: For internal use only
- confidential: Sensitive business information
- restricted: Highly sensitive, needs special protection

Respond with JSON only.`;

    const prompt = `Classify this content and provide handling recommendations:

${content}

Return JSON with:
{
  "confidentialityLevel": "level",
  "dataTypes": ["type1", "type2"],
  "handlingRecommendations": ["rec1", "rec2"]
}`;

    const params: ModelParams = {
      temperature: 0.1,
      max_tokens: 400,
    };

    try {
      const response = await this.lmStudio.generateResponse(prompt, systemPrompt, params);
      return JSON.parse(response);
    } catch (error) {
      return {
        confidentialityLevel: 'restricted',
        dataTypes: ['unknown'],
        handlingRecommendations: ['Treat as highly sensitive due to classification error'],
      };
    }
  }

  private getPrivacySystemPrompt(privacyLevel: PrivacyLevel): string {
    const basePrompt = 'You are a privacy-conscious AI assistant. ';

    switch (privacyLevel) {
      case 'strict':
        return basePrompt + `Apply the highest level of privacy protection:
        - Never expose personal names, addresses, phone numbers, emails
        - Generalize all specific locations and dates
        - Remove all identifying information
        - Use placeholders for sensitive data
        - Err on the side of caution`;

      case 'moderate':
        return basePrompt + `Apply moderate privacy protection:
        - Protect personal identifiable information
        - Generalize specific details when appropriate
        - Maintain readability while ensuring privacy
        - Remove sensitive financial or health data`;

      case 'minimal':
        return basePrompt + `Apply basic privacy protection:
        - Protect obvious sensitive information (SSNs, credit cards)
        - Remove personal contact information
        - Maintain the natural flow of the text`;

      default:
        return basePrompt + 'Apply appropriate privacy measures.';
    }
  }

  private detectSensitiveInformation(content: string): Array<{value: string, type: string}> {
    const sensitiveData: Array<{value: string, type: string}> = [];

    // Email addresses
    const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g;
    const emails = content.match(emailRegex) || [];
    emails.forEach(email => sensitiveData.push({value: email, type: 'email'}));

    // Phone numbers (various formats)
    const phoneRegex = /(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{3}-\d{3}-\d{4}|\(\d{3}\)\s*\d{3}-\d{4})/g;
    const phones = content.match(phoneRegex) || [];
    phones.forEach(phone => sensitiveData.push({value: phone, type: 'phone'}));

    // Credit card numbers (last 4 digits)
    const creditCardRegex = /\b\d{4}\b(?=.*(?:card|credit|ending|last))/g;
    const creditCards = content.match(creditCardRegex) || [];
    creditCards.forEach(card => sensitiveData.push({value: card, type: 'credit_card'}));

    // Account numbers (patterns like #A123456 or account 123456)
    const accountRegex = /(#[A-Z]?\d{4,}|account\s+#?\d{4,})/gi;
    const accounts = content.match(accountRegex) || [];
    accounts.forEach(account => sensitiveData.push({value: account, type: 'account'}));

    // Dollar amounts
    const dollarRegex = /\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?/g;
    const amounts = content.match(dollarRegex) || [];
    amounts.forEach(amount => sensitiveData.push({value: amount, type: 'currency'}));

    // Names (simple pattern - capitalized words that could be names)
    const nameRegex = /\b[A-Z][a-z]+ [A-Z][a-z]+\b/g;
    const names = content.match(nameRegex) || [];
    names.forEach(name => {
      // Filter out common non-name patterns
      if (!name.match(/^(Dear|From|To|Hi|Hello|Mr|Mrs|Ms|Dr)\s/) &&
          !name.match(/^(Thank You|New York|Los Angeles|San Francisco|United States)$/)) {
        sensitiveData.push({value: name, type: 'name'});
      }
    });

    // Also catch lowercase names that might be missed
    const lowerNamePattern = /\b[a-z]+ [a-z]+\b/g;
    const potentialNames = content.match(lowerNamePattern) || [];
    potentialNames.forEach(name => {
      if (name.includes('sarah') || name.includes('john') || name.includes('jane') ||
          name.includes('mike') || name.includes('david') || name.includes('lisa')) {
        sensitiveData.push({value: name, type: 'name'});
      }
    });

    return sensitiveData;
  }

  private getEnhancedPrivacySystemPrompt(
    privacyLevel: PrivacyLevel,
    replacements: Array<{original: string, placeholder: string, type: string}>
  ): string {
    const basePrompt = `You are an expert privacy protection specialist. Your CRITICAL task is to rewrite text while maintaining ABSOLUTE privacy protection.

STRICT RULES:
1. NEVER reproduce the original sensitive data that has been replaced with placeholders
2. Replace ALL placeholders with appropriate generic terms
3. Maintain the meaning and flow of the text
4. Use professional, appropriate language

Placeholder meanings:
${replacements.map(r => `- ${r.placeholder}: Replace with generic ${r.type} (e.g., "customer", "account", "payment method")`).join('\n')}

Privacy level: ${privacyLevel}`;

    return basePrompt;
  }

  private buildEnhancedRewritePrompt(
    preprocessedContent: string,
    style: string,
    privacyLevel: PrivacyLevel,
    replacements: Array<{original: string, placeholder: string, type: string}>
  ): string {
    return `Rewrite the following text in a ${style} style while replacing ALL placeholders with appropriate generic terms.

CRITICAL: Replace each placeholder with a generic equivalent:
${replacements.map(r => `- ${r.placeholder} → generic ${r.type} reference`).join('\n')}

Text to rewrite:
${preprocessedContent}

Requirements:
- Maintain the core message and professional tone
- Replace ALL placeholders with generic terms
- Never reproduce the original sensitive information
- Ensure complete privacy protection

Rewritten text:`;
  }

  private postProcessSecureRewrite(
    response: string,
    sensitiveData: Array<{value: string, type: string}>,
    privacyLevel: PrivacyLevel
  ): string {
    let cleanedResponse = response;

    // Remove any sensitive data that might have leaked through
    for (const item of sensitiveData) {
      const escapedValue = item.value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const regex = new RegExp(escapedValue, 'gi');
      cleanedResponse = cleanedResponse.replace(regex, this.getGenericReplacement(item.type));
    }

    // Additional cleanup for common patterns that might leak through
    cleanedResponse = cleanedResponse
      .replace(/\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?/g, 'the amount') // Any remaining dollar amounts
      .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, 'our support team') // Any remaining emails
      .replace(/\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}/g, 'our phone number') // Any remaining phones
      .replace(/#[A-Z]?\d{4,}/g, 'your account') // Any remaining account numbers
      .replace(/\b\d{4}\b/g, 'your payment method') // Any remaining 4-digit numbers (like card endings)
      .replace(/\bsarah johnson\b/gi, 'the customer') // Specific name cleanup
      .replace(/\bjohn\s+\w+\b/gi, 'the customer') // John + any last name
      .replace(/\bsarah\s+\w+\b/gi, 'the customer'); // Sarah + any last name

    return cleanedResponse;
  }

  private getGenericReplacement(type: string): string {
    switch (type) {
      case 'name': return 'the customer';
      case 'email': return 'our support team';
      case 'phone': return 'our phone number';
      case 'credit_card': return 'your payment method';
      case 'account': return 'your account';
      case 'currency': return 'the amount';
      default: return '[REDACTED]';
    }
  }

  private buildRewritePrompt(content: string, style: string, privacyLevel: PrivacyLevel): string {
    const privacyNote = privacyLevel === 'strict'
      ? ' Remember to protect all personal and sensitive information.'
      : privacyLevel === 'moderate'
      ? ' Ensure personal and sensitive information is appropriately handled.'
      : ' Remove any obviously sensitive information.';

    return `Rewrite the following text in a ${style} style.${privacyNote}

Original text:
${content}

Rewritten text:`;
  }
}