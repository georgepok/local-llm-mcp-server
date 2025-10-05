# Usage Examples

Practical examples for common use cases with the Local LLM MCP Server.

## Table of Contents

- [Getting Started](#getting-started)
- [Privacy & Security](#privacy--security)
- [Code Analysis](#code-analysis)
- [Content Processing](#content-processing)
- [Business & Productivity](#business--productivity)
- [Research & Analysis](#research--analysis)
- [Model Management](#model-management)

---

## Getting Started

### 1. Check Available Models

Before using the server, discover what models are loaded:

```
Ask Claude: "What local models are available?"
```

Claude will read the `local://models` resource and show you something like:

```json
{
  "models": [
    "openai/gpt-oss-20b",
    "qwen3-30b-a3b-deepseek-distill-instruct-2507"
  ],
  "count": 2,
  "defaultModel": "openai/gpt-oss-20b"
}
```

### 2. Simple Reasoning Task

```
Ask Claude: "Using the local LLM, explain quantum computing in simple terms"
```

Claude uses the `local_reasoning` tool with the default model.

### 3. Set Preferred Model

```
Ask Claude: "Set qwen3-30b-a3b-deepseek-distill-instruct-2507 as the default model"
```

---

## Privacy & Security

### Analyze Sensitive Documents

**Scenario**: You have a confidential medical report to analyze.

```
Ask Claude: "Analyze this medical report for sentiment locally:

Patient reports feeling better after treatment. Some side effects noted but manageable. Overall positive response to medication regimen."
```

Claude will use `private_analysis` with `domain: medical` to ensure proper context.

### Detect PII in Content

**Scenario**: Check if content contains personally identifiable information.

```
Ask Claude: "Scan this email for privacy issues:

Hi John,

Thanks for your inquiry. Please send the documents to our office at 123 Main St, or email them to processing@company.com. You can also call us at (555) 123-4567.

Best,
Sarah Johnson"
```

Response will identify:
- Names (John, Sarah Johnson)
- Email address
- Phone number
- Physical address

### Rewrite with Privacy Protection

**Scenario**: Share customer feedback without exposing identity.

```
Ask Claude: "Rewrite this customer feedback in a professional style with strict privacy protection:

'Hey, I'm Jane from Acme Corp. Called you guys at 555-1234 yesterday about the billing issue. My account jdoe@email.com was overcharged $500!'"
```

Result:
```
"A customer from an enterprise organization contacted support regarding a billing discrepancy. The account in question was incorrectly charged an amount requiring immediate attention and resolution."
```

---

## Code Analysis

### Security Vulnerability Detection

**Scenario**: Review code for security issues before deployment.

```
Ask Claude: "Analyze this Python code for security vulnerabilities:

```python
def process_user_input(username, password):
    conn = db.connect()
    query = f\"SELECT * FROM users WHERE username='{username}' AND password='{password}'\"
    result = conn.execute(query)
    return result
```
"
```

Response identifies:
- SQL injection vulnerability
- Plaintext password handling
- Lack of input validation

### Code Quality Assessment

**Scenario**: Get suggestions for improving code quality.

```
Ask Claude: "Review this JavaScript function for quality improvements:

```javascript
function calc(a, b, op) {
    if (op == '+') return a+b;
    if (op == '-') return a-b;
    if (op == '*') return a*b;
    if (op == '/') return a/b;
}
```
"
```

Suggestions include:
- Add JSDoc documentation
- Use switch statement
- Add input validation
- Handle division by zero
- Add type checking

### Find Optimization Opportunities

**Scenario**: Optimize performance-critical code.

```
Ask Claude: "Analyze this code for optimization opportunities:

```python
def find_duplicates(numbers):
    duplicates = []
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] == numbers[j] and numbers[i] not in duplicates:
                duplicates.append(numbers[i])
    return duplicates
```
"
```

---

## Content Processing

### Sentiment Analysis with Domain Context

**Scenario**: Analyze financial news sentiment.

```
Ask Claude: "Analyze the sentiment of this financial news locally:

'The company's Q4 earnings exceeded expectations with a 15% increase in revenue. However, rising operational costs and market uncertainties pose challenges for the upcoming quarter.'"
```

Use `domain: financial` for accurate context.

### Extract Entities from Legal Documents

**Scenario**: Extract key entities from a legal contract.

```
Ask Claude: "Extract entities from this contract excerpt:

'This Agreement is entered into on January 15, 2024, between Acme Corporation, a Delaware corporation with offices at 123 Business Ave, and Beta LLC, represented by John Smith, CEO.'"
```

Extracts:
- Organizations: Acme Corporation, Beta LLC
- People: John Smith
- Locations: Delaware, 123 Business Ave
- Dates: January 15, 2024
- Legal terms: Agreement, CEO, corporation

### Classify Content by Topic

**Scenario**: Categorize support tickets.

```
Ask Claude: "Classify this support ticket:

'My account was locked after trying to reset my password. I've tried the forgot password link but haven't received the email. This is urgent as I need access for work.'"
```

Classification:
- Categories: [technical_support, account_access, password_reset]
- Complexity: medium
- Tags: [urgent, authentication, email_delivery]

---

## Business & Productivity

### Complete Email Templates

**Scenario**: Draft a professional response to a customer inquiry.

```
Ask Claude: "Complete this email template with the given context:

Template:
'Dear [CUSTOMER_NAME],

Thank you for your recent [REQUEST_TYPE] regarding [SUBJECT].

We have reviewed your request and [OUTCOME].

Next steps:
1. [STEP_1]
2. [STEP_2]
3. [STEP_3]

Best regards,
[AGENT_NAME]
[TITLE]'

Context:
Customer Alice Johnson requested information about upgrading to our enterprise plan for her team of 50 users. The request was approved, and she needs to complete payment setup."
```

### Summarize Meeting Notes

**Scenario**: Extract key points from a long meeting transcript.

```
Ask Claude: "Extract key points from this meeting:

'Team discussed Q1 goals. Marketing will launch new campaign in February. Development prioritizing mobile app improvements. Budget concerns raised - need to reduce cloud costs by 20%. Action item: Sarah to present cost analysis next week. Decision made to hire two new developers.'"
```

Response:
```json
{
  "keyPoints": [
    "Q1 goals discussion",
    "New marketing campaign launching February",
    "Mobile app improvements prioritized",
    "20% cloud cost reduction needed"
  ],
  "actionItems": [
    "Sarah presents cost analysis next week",
    "Hire two new developers"
  ],
  "insights": [
    "Budget optimization is a key priority",
    "Team expansion planned in development"
  ]
}
```

### Risk Assessment

**Scenario**: Assess risks for a new project.

```
Ask Claude: "Using the risk assessment template, analyze this scenario:

'We're planning to migrate our on-premise database to a cloud provider. Timeline is 3 months. Current database has 10TB of customer data. Team has limited cloud experience.'"
```

---

## Research & Analysis

### Synthesize Research Findings

**Scenario**: Combine insights from multiple research papers.

```
Ask Claude: "Synthesize these research findings:

Source 1: Study shows remote work increases productivity by 13%
Source 2: Remote workers report 22% higher job satisfaction
Source 3: Companies save average of $11,000/year per remote worker
Source 4: 15% of remote workers report feeling isolated"
```

### Academic Literature Review

**Scenario**: Analyze papers for a systematic review.

```
Ask Claude: "Conduct a literature review on these machine learning papers, identifying common themes and gaps"
```

---

## Model Management

### Switch Models for Different Tasks

**Scenario**: Use a coding-specialized model for code tasks.

```
Conversation with Claude:

You: "I have two models loaded: openai/gpt-oss-20b and qwen3-30b-a3b-deepseek-distill-instruct-2507. Set the Qwen model as default since it's better for coding."

Claude: [Uses set_default_model tool]

You: "Now analyze this Python code for bugs"

Claude: [Uses local_reasoning with Qwen model automatically]
```

### Model-Specific Requests

**Scenario**: Compare responses from different models.

```
You: "Use openai/gpt-oss-20b to explain quantum computing"

[Response 1]

You: "Now use qwen3-30b-a3b-deepseek-distill-instruct-2507 to explain the same thing"

[Response 2 - possibly different explanation style]
```

---

## Advanced Scenarios

### Multi-Step Privacy Workflow

```
You: "I have a confidential document. First scan it for privacy issues, then rewrite it with strict privacy protection, then summarize the rewritten version."

Claude will:
1. Use private_analysis with analysis_type: privacy_scan
2. Use secure_rewrite with privacy_level: strict
3. Use private_analysis with analysis_type: summary
```

### Code Security Pipeline

```
You: "Review this authentication code for security issues, then if you find any, provide an optimized secure version"

Claude will:
1. Use code_analysis with analysis_focus: security
2. If issues found, use local_reasoning to generate secure alternative
3. Use code_analysis again to verify the new code
```

### Document Processing Chain

```
You: "Analyze this legal contract for entities, classify it by type, extract key points, and identify any privacy concerns"

Claude will:
1. Use private_analysis with analysis_type: entities, domain: legal
2. Use private_analysis with analysis_type: classification, domain: legal
3. Use private_analysis with analysis_type: key_points, domain: legal
4. Use private_analysis with analysis_type: privacy_scan
```

---

## Tips & Best Practices

### Performance Optimization

```
For faster responses on simple tasks:
{
  "model_params": {
    "temperature": 0.1,
    "max_tokens": 200
  }
}

For creative or complex tasks:
{
  "model_params": {
    "temperature": 0.8,
    "max_tokens": 2000
  }
}
```

### Domain Selection

- Use `domain: medical` for healthcare, patient data, clinical notes
- Use `domain: legal` for contracts, compliance, legal analysis
- Use `domain: financial` for market analysis, financial reports, trading
- Use `domain: technical` for software, engineering, IT documentation
- Use `domain: academic` for research papers, scholarly analysis
- Use `domain: general` for everyday content

### Privacy Levels

- **Strict**: Maximum anonymization, all PII removed (for highly sensitive data)
- **Moderate**: Balance privacy and usability (default, recommended)
- **Minimal**: Basic protection only (for less sensitive content)

---

## Troubleshooting Examples

### Model Not Responding

```
You: "Why isn't the local model responding?"

Check:
1. Read local://status to verify LM Studio is online
2. Read local://models to see if models are loaded
3. Ensure LM Studio local server is running
```

### Choosing the Right Tool

| Task | Tool | Example |
|------|------|---------|
| General questions | `local_reasoning` | "Explain blockchain" |
| Detect sentiment | `private_analysis` (sentiment) | Analyze customer feedback |
| Find security issues | `code_analysis` (security) | Review login function |
| Remove PII | `secure_rewrite` (strict) | Anonymize case study |
| Extract data | `private_analysis` (entities) | Get names from email |
| Fill forms | `template_completion` | Complete email template |
