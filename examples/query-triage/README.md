# Query Triage

Route support, health, or legal queries by complexity before calling an expensive API. Simple queries go to cached answers or FAQ lookups. Complex queries get the full API treatment.

## How to use

This is a skeleton example. To build real query triage:

1. Collect queries labelled by complexity (Simple, Medium, Complex)
2. Format as JSON: `[{"text": "...", "tier": "Simple"}, ...]`
3. Train with AXIOM's training pipeline
4. Add 90us of latency to your request path for intelligent routing

## Run

```bash
cargo run --example query-triage
```
