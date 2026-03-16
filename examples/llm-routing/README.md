# LLM Query Routing

Classify natural language queries into complexity tiers (Surface, Reasoning, Deep) to route them to appropriate LLM models. Surface queries go to fast/cheap models, Deep queries go to capable/expensive models.

## Results

- 94.8% overall accuracy on 1,000 diverse queries
- Surface: 96.6%, Reasoning: 93.9%, Deep: 93.3%
- 90us mean latency, 465us P99
- RouterBench: 31.6% cost reduction on 36,511 queries

## Usage

```bash
# Train from scratch (~4 minutes)
cargo run --release --example train-axiom -- \
  --sts-data data/stsbenchmark.tsv \
  --tier-data data/fusion_training_corpus.json \
  --eval-data data/eval_set1_synthetic.json,data/eval_set2_adversarial.json,data/eval_set3_realworld.json \
  --output axiom_router_weights.json

# Evaluate on the 1,000-query set
cargo run --release --example eval-axiom -- \
  --weights axiom_router_weights.json \
  --eval-data data/eval_set1_synthetic.json,data/eval_set2_adversarial.json,data/eval_set3_realworld.json

# RouterBench evaluation (requires preprocessed data)
cargo run --release --example routerbench-eval -- \
  axiom_router_weights.json data/routerbench/routerbench.jsonl
```

## Files

- `train.rs` — Full training pipeline (STS pretrain + classification fine-tune)
- `eval.rs` — Evaluate on any query set with per-tier accuracy and confusion matrix
- `routerbench.rs` — RouterBench benchmark evaluation (36,511 queries, 11 models)
