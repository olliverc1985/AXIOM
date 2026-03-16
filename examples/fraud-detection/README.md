# Fraud Detection

Classify messages as Clean, Suspicious, or Blocked based on linguistic patterns. The structural encoder detects high-risk signals (urgency markers, excessive capitalisation, unusual punctuation) while the semantic encoder captures meaning-level risk.

## How to use

This is a skeleton example showing the API pattern. To build a real fraud detector:

1. Collect labelled examples of clean, suspicious, and blocked messages
2. Format as JSON: `[{"text": "...", "tier": "Clean"}, ...]`
3. Train: `cargo run --example train-axiom -- --tier-data your_fraud_data.json`
4. Load trained weights in `main.rs` using `AxiomRouter::load("weights.json")`

## Run

```bash
cargo run --example fraud-detection
```
