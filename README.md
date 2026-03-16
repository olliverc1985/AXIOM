# AXIOM: Sparse Dynamic Routing for LLM Inference

AXIOM is a lightweight query router that classifies natural language queries into complexity tiers (Surface, Reasoning, Deep) for cost-efficient LLM inference. It uses a dual-encoder architecture — a deterministic structural encoder and a classification-trained transformer — fused into a single 256-dimensional representation and classified via linear softmax. Built entirely in Rust with zero external ML framework dependencies. All tensor operations, transformer layers, backpropagation, and Adam optimisation are implemented from scratch.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19016401.svg)](https://doi.org/10.5281/zenodo.19016401)

## Key Results

| Metric | Value |
|--------|-------|
| Overall routing accuracy | 94.8% on 1,000 diverse queries |
| Surface tier accuracy | 96.6% |
| Reasoning tier accuracy | 93.9% |
| Deep tier accuracy | 93.3% |
| Mean routing latency | 90 us |
| P99 latency | 465 us |
| Total parameters | ~1.5M (37K semantic encoder, rest is embeddings) |
| Training time | ~4 minutes on laptop CPU |
| RouterBench cost reduction | 31.6% (Strategy C, 36,511 queries) |
| RouterBench AIQ | 0.0071 |

Evaluation uses three independently constructed query sets: synthetic (334), adversarial (333), and real-world (333).

## Architecture

```
Query
  |
  ├──> Structural Encoder (128-dim)
  |      Hand-crafted syntactic features: position-weighted token embeddings,
  |      type-token ratio, avg token length, punctuation density, normalised count
  |      Deterministic, no learnable parameters
  |
  ├──> Semantic Encoder (128-dim)
  |      2-layer transformer, 4 attention heads, 512-dim FFN
  |      Pretrained on STS Benchmark (sentence similarity)
  |      Fine-tuned end-to-end with classification objective
  |
  └──> Always-Fuse ──> Concatenate (256-dim) ──> Linear ──> 3-class Softmax
                                                              |
                                                    Surface / Reasoning / Deep
```

Both encoders run on every query (no gating or fast-path). The structural features provide syntactic signal while the semantic encoder captures meaning-level complexity.

The system also includes a sparse computation graph with four traversal directions (forward, lateral, feedback, temporal), dynamic coalition formation, and a persistent embedding cache. Every other router surveyed (75+) makes a single-pass decision; AXIOM's graph architecture enables inter-node communication across traversal passes, allowing routing decisions to incorporate feedback from prior inferences.

### Training Pipeline

Two-stage training, both from scratch:

1. **STS pretraining** (~3.5 min): Train the semantic encoder on 8,628 sentence similarity pairs to learn general language representations. MSE loss, 30 epochs.
2. **Classification fine-tuning** (~30s): Fine-tune the encoder end-to-end with tier labels. Differential learning rates (encoder: 3e-4, head: 1e-3). Cross-entropy loss with inverse-frequency class weights. Early stopping with patience 20.

## Quick Start

```bash
# Build
cargo build --release

# Run tests (166 tests)
cargo test

# Train from scratch
cargo run --release --bin train_axiom -- \
  --sts-data data/stsbenchmark.tsv \
  --tier-data data/fusion_training_corpus.json \
  --eval-data data/eval_set1_synthetic.json,data/eval_set2_adversarial.json,data/eval_set3_realworld.json \
  --output axiom_router_weights.json

# Evaluate
cargo run --release --bin eval_axiom -- \
  --weights axiom_router_weights.json \
  --eval-data data/eval_set1_synthetic.json,data/eval_set2_adversarial.json,data/eval_set3_realworld.json

# Run graph architecture benchmark
cargo run --release --bin axiom-bench
```

## Limitations

- **RouterBench performance**: AXIOM achieves 31.6% cost reduction but with a 10.5 percentage point accuracy penalty vs always-using-the-frontier-model. AIQ of 0.0071 is near the random baseline. Complexity-based routing and cost-optimal model selection are fundamentally different objectives — AXIOM routes on linguistic complexity, not on which model happens to get each specific question right. The RouterBench oracle routes 47% of queries to the cheapest model, because many benchmark questions don't need an expensive model regardless of how "complex" they look.
- **Evaluation scale**: The 1,000-query diverse eval set covers synthetic, adversarial, and real-world distributions but is still relatively small compared to production workloads.
- **English only**: The tokeniser and training data are English. Other languages will produce mostly UNK tokens.
- **No online learning**: The router must be retrained offline to incorporate new query types. Cannot adapt from production routing mistakes.
- **Fixed cost model**: Tier routing assumes fixed computational costs per tier. RouterBench provides real cost data showing 31.6% actual savings, but production costs vary by query content and provider.
- **Tier classification vs answer quality**: The 94.8% accuracy measures whether AXIOM assigns the correct complexity tier, not whether the downstream model produces a correct answer.
- **Stability**: 9/10 random seeds hit all accuracy targets. 1/10 dipped below on Reasoning tier (80.9%). Mean overall: 94.0% +/- 2.1%.
- **Vocabulary size**: The 8,192-token word-level vocabulary has limited coverage of rare/technical terms (mapped to UNK).
- **No batching**: Inference processes one query at a time.

## RouterBench Results

Evaluated zero-shot on [RouterBench](https://github.com/withmartian/routerbench) (Hu et al., ICML 2024) — 36,511 queries, 11 models from Mistral-7B ($0.00014/query) to GPT-4 ($0.00794/query).

| Router | Accuracy | Avg $/query | Cost Reduction |
|--------|----------|-------------|----------------|
| Always-Cheap (Mistral-7B) | 50.0% | $0.000139 | 98.3% |
| Random | 61.7% | $0.002156 | 72.9% |
| AXIOM Strategy B (cheapest-in-tier) | 56.4% | $0.003800 | 52.2% |
| AXIOM Strategy A (best-in-tier) | 69.4% | $0.005301 | 33.3% |
| AXIOM Strategy C (confidence cascade) | 70.0% | $0.005433 | 31.6% |
| Always-GPT-4 | 80.5% | $0.007943 | 0.0% |
| Oracle | 84.7% | $0.000378 | 95.2% |

AXIOM's low AIQ (0.0071) reflects the gap between complexity-based routing and task-specific optimal model selection. The router was trained on linguistic complexity labels, not RouterBench task performance data. This is a zero-shot evaluation — no tuning on RouterBench.

To reproduce:
```bash
# Preprocess RouterBench data (requires Python + pandas)
python scripts/preprocess_routerbench.py

# Run evaluation
cargo run --release --bin routerbench_eval -- \
  axiom_router_weights.json data/routerbench/routerbench.jsonl
```

## Use Cases

Where complexity-based routing adds value:

- **API gateway pre-filter** for LLM workloads — route simple queries to fast/cheap models
- **Cost reduction** — 30%+ savings without task-specific tuning
- **Cascade trigger** — try cheap model first, escalate on low confidence
- **SLA enforcement** — route simple queries to fast models, complex ones to capable models
- **Latency-sensitive applications** — 90us routing overhead is negligible vs LLM inference time

## Project Structure

```
src/
├── lib.rs                # Core Tensor type
├── router/mod.rs         # Unified always-fuse router
├── semantic/             # Transformer-based semantic encoder
│   ├── encoder.rs        # High-level encode API
│   ├── transformer.rs    # Forward/backward pass, Adam, weight serialisation
│   ├── vocab.rs          # Word-level vocabulary
│   └── train.rs          # STS training loop
├── input/                # Structural encoder
│   ├── tokeniser.rs      # Whitespace + punctuation tokeniser
│   └── encoder.rs        # Position-weighted features + syntactic features
├── graph/                # Sparse computation graph
│   ├── engine.rs         # SparseGraph, traversal, routing
│   ├── node.rs           # LinearNode, ComputeNode trait, Hebbian learning
│   └── edge.rs           # ConditionalEdge, LateralEdge
├── tiers/                # Hierarchical reasoning tiers
│   ├── tier.rs           # Tier enum, AxiomConfig
│   ├── resolver.rs       # HierarchicalResolver, calibration
│   └── feedback.rs       # FeedbackSignal
├── cache/                # Cosine similarity embedding cache
├── gpu.rs                # Metal GPU compute (feature-gated)
├── tuner.rs              # Auto-tuner rules
└── bin/
    ├── train_axiom.rs    # Training pipeline
    ├── eval_axiom.rs     # Evaluation
    ├── bench.rs          # Graph architecture benchmark
    └── routerbench_eval.rs  # RouterBench evaluation
```

## Technical Details

- **Parameters**: ~1.5M (mostly in the 8192x128 token embedding table)
- **Inference**: ~90us per query (Apple Silicon M-series)
- **Training**: ~4 minutes total (3.5 min STS + 30s classification)
- **Dependencies**: serde, serde_json, uuid, indicatif. No ML frameworks.
- **BLAS**: Uses Apple Accelerate (cblas_sgemm) for matrix multiplication on macOS.

## Paper

- Zenodo: [https://doi.org/10.5281/zenodo.19016401](https://doi.org/10.5281/zenodo.19016401)
- The published paper covers the structural encoder architecture. Updated paper with semantic encoder (Phases 16-19) and RouterBench results is forthcoming.

## License

[MIT](LICENSE)
