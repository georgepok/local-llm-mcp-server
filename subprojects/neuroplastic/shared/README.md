# Neuroplastic — Shared Communication Channel

Two-way communication between **Claude Desktop** (research direction) and **Claude Code** (implementation).

## Structure

```
shared/
├── inbox/      ← Claude Desktop writes here (instructions, requirements, reviews, decisions)
├── outbox/     ← Claude Code writes here (reports, results, status updates, questions)
└── README.md
```

## Convention

- **inbox/**: Claude Desktop drops tasks, experiment plans, design decisions, review feedback.
  Claude Code reads from here and acts on it.
- **outbox/**: Claude Code posts results, status updates, findings, and questions.
  Claude Desktop reads from here to track progress and plan next steps.
- Files should be prefixed with date or phase for ordering (e.g., `phase0_status.md`, `2026-03-10_review.md`)
- Neither agent modifies the other's folder — write to your own, read from both.
