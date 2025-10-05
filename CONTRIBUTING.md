# Contributing to Local LLM MCP Server

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Project Structure](#project-structure)

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Prioritize the community's best interests

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Trolling or insulting/derogatory comments
- Publishing others' private information
- Other conduct inappropriate for a professional setting

## Getting Started

###Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Types of Contributions

We welcome:

- **Bug fixes**: Improvements to existing functionality
- **New features**: Additional analysis tools, model integrations, etc.
- **Documentation**: Improvements to README, API docs, examples
- **Tests**: Additional test coverage
- **Performance improvements**: Optimization and efficiency
- **Security enhancements**: Privacy and security improvements

### Before You Start

1. **Check existing issues** - Someone might already be working on it
2. **Create an issue** - Discuss major changes before implementing
3. **Read the docs** - Familiarize yourself with the codebase

## Development Setup

### Prerequisites

- Node.js 18+
- LM Studio with at least one model loaded
- Git
- TypeScript knowledge

### Setup Steps

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/local-llm-mcp-server.git
cd local-llm-mcp-server

# Install dependencies
npm install

# Build the project
npm run build

# Run type checking
npm run typecheck
```

### Development Workflow

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes and test
npm run build
npm run typecheck

# Run tests
npm test

# Commit your changes
git add .
git commit -m "feat: add amazing feature"

# Push to your fork
git push origin feature/your-feature-name
```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-ollama-support` - New features
- `fix/model-timeout-issue` - Bug fixes
- `docs/api-examples` - Documentation
- `refactor/analysis-tools` - Code refactoring
- `test/integration-tests` - Test additions

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add support for Ollama backend
fix: resolve model timeout issues
docs: update API documentation
refactor: simplify analysis tools structure
test: add integration tests for privacy tools
chore: update dependencies
```

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## Testing

### Running Tests

```bash
# Build and run all tests
npm test

# Run basic tests
npm run test:basic

# Run full test suite
npm run test:full

# Type checking only
npm run typecheck
```

### Writing Tests

Add tests for new features in the `test/` directory:

```typescript
// test/my-feature-test.ts
import { LMStudioClient } from '../src/lm-studio-client.js';

async function testMyFeature() {
  const client = new LMStudioClient();
  await client.initialize();

  // Your test logic
  const result = await client.myNewFeature();

  if (result.expected !== result.actual) {
    throw new Error('Test failed');
  }

  console.log('✓ My feature test passed');
}

testMyFeature().catch(console.error);
```

### Test Coverage Areas

- **Unit tests**: Individual functions and methods
- **Integration tests**: Tool interactions with LM Studio
- **Error handling**: Edge cases and error conditions
- **Privacy features**: Data protection and anonymization
- **Model management**: Discovery, selection, switching

## Pull Request Process

### Before Submitting

1. ✅ Code builds without errors (`npm run build`)
2. ✅ Type checking passes (`npm run typecheck`)
3. ✅ Tests pass (`npm test`)
4. ✅ Code follows style guidelines
5. ✅ Documentation updated if needed
6. ✅ Commit messages follow conventions

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
How was this tested?

## Checklist
- [ ] Code builds successfully
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Breaking changes documented
```

### Review Process

1. **Automated checks** run on your PR
2. **Maintainers review** your code
3. **Address feedback** by pushing new commits
4. **Approval** from at least one maintainer
5. **Merge** by maintainers

## Coding Standards

### TypeScript Style

```typescript
// Good
async function analyzeContent(
  content: string,
  options: AnalysisOptions
): Promise<AnalysisResult> {
  // Implementation
}

// Bad
async function analyze(c, o) {
  // Implementation
}
```

### Naming Conventions

- **Files**: `kebab-case.ts` (e.g., `analysis-tools.ts`)
- **Classes**: `PascalCase` (e.g., `LMStudioClient`)
- **Functions**: `camelCase` (e.g., `generateResponse`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`)
- **Interfaces**: `PascalCase` (e.g., `ModelParams`)

### Code Organization

```typescript
// 1. Imports
import { Server } from '@modelcontextprotocol/sdk/server/index.js';

// 2. Type definitions
interface MyInterface {
  property: string;
}

// 3. Class definition
export class MyClass {
  // Private fields first
  private client: Client;

  // Constructor
  constructor() {}

  // Public methods
  public async myMethod(): Promise<void> {}

  // Private methods last
  private helperMethod(): void {}
}
```

### Documentation

Add JSDoc comments for public APIs:

```typescript
/**
 * Analyzes content using the specified analysis type.
 *
 * @param content - The text content to analyze
 * @param analysisType - Type of analysis to perform
 * @param domain - Domain context for specialized analysis
 * @returns Analysis results with type-specific data
 *
 * @example
 * ```typescript
 * const result = await analyzeContent(
 *   "Sample text",
 *   "sentiment",
 *   "general"
 * );
 * ```
 */
async analyzeContent(
  content: string,
  analysisType: AnalysisType,
  domain: DomainType = 'general'
): Promise<any> {
  // Implementation
}
```

### Error Handling

```typescript
// Good - Informative errors
throw new Error(
  `Model "${model}" not found. Available models: ${availableModels.join(', ')}`
);

// Bad - Vague errors
throw new Error('Invalid model');
```

### Privacy Considerations

- Never log sensitive user data
- Sanitize error messages
- Document privacy implications
- Follow privacy level settings

## Project Structure

```
src/
├── index.ts              # Main MCP server
├── lm-studio-client.ts   # LM Studio integration
├── analysis-tools.ts     # Content analysis
├── privacy-tools.ts      # Privacy features
├── prompt-templates.ts   # Template management
├── types.ts              # TypeScript definitions
└── config.ts             # Configuration

test/
├── basic-mcp-test.ts     # Basic tests
├── integration-test.ts   # Integration tests
└── mcp-client-test.ts    # MCP client tests

dist/                     # Compiled output
docs/                     # Additional documentation
```

### Adding New Tools

1. **Define types** in `src/types.ts`
2. **Implement logic** in appropriate file
3. **Register tool** in `src/index.ts`
4. **Add tests** in `test/`
5. **Update docs** (README.md, API.md, EXAMPLES.md)

Example:

```typescript
// 1. src/types.ts
export const MyAnalysisSchema = z.object({
  input: z.string(),
  options: z.object({}).optional()
});

// 2. src/analysis-tools.ts
async myNewAnalysis(input: string, options?: object): Promise<Result> {
  // Implementation
}

// 3. src/index.ts
{
  name: 'my_analysis',
  description: 'Performs my custom analysis',
  inputSchema: {
    type: 'object',
    properties: {
      input: { type: 'string', description: 'Input to analyze' }
    },
    required: ['input']
  }
}

// 4. Handle the tool
case 'my_analysis':
  result = await this.handleMyAnalysis(args);
  break;
```

## Release Process

(For maintainers)

1. Update version in `package.json`
2. Update CHANGELOG.md
3. Create release tag
4. Publish to npm (if applicable)
5. Create GitHub release

## Questions?

- **General questions**: [GitHub Discussions](https://github.com/yourusername/local-llm-mcp-server/discussions)
- **Bugs**: [GitHub Issues](https://github.com/yourusername/local-llm-mcp-server/issues)
- **Security issues**: Email maintainers privately

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to Local LLM MCP Server! 🎉
