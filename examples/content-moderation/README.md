# Content Moderation

Triage user-generated content by severity for moderation queues: Allow (passes checks), Review (needs human review), or Block (immediately removed).

## How to use

This is a skeleton example. To build a real moderator:

1. Collect labelled content examples across severity levels
2. Format as JSON: `[{"text": "...", "tier": "Allow"}, ...]`
3. Train with AXIOM's training pipeline
4. Deploy at <1ms per classification decision

## Run

```bash
cargo run --example content-moderation
```
