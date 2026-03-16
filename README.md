# AXIOM

**Lightweight Rust ML framework for training and deploying small transformer classifiers. Zero external ML dependencies. Microsecond inference. CPU-first with optional GPU acceleration.**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19016401.svg)](https://doi.org/10.5281/zenodo.19016401)

## Why AXIOM

You need to add a small ML classifier to a Rust service. Your options are:

- Pull in PyTorch via tch-rs (200MB+ dependency, C++ bindings, GPU runtime)
- Use Burn or Candle (solid frameworks, but heavyweight for a small classifier)
- Call an external ML API (adds latency, cost, and a network dependency)
- Build it yourself from scratch

AXIOM is the fifth option: a purpose-built library for small, fast text classifiers that compiles to a single binary with no external ML dependencies. Think SQLite vs PostgreSQL, but for ML.

## Features

- Train and run small transformer encoders (configurable layers, heads, dimensions)
- Full backpropagation with Adam/AdamW optimiser
- Multiple training objectives: classification (cross-entropy), contrastive learning, sentence similarity (MSE)
- Hand-crafted structural text features (token patterns, syntactic signals)
- Multi-encoder fusion for combining learned and engineered features
- JSON weight serialisation (no custom binary formats)
- 90us mean inference latency on CPU
- Optional Apple Metal GPU acceleration (`--features gpu`)
- Apple Accelerate BLAS for 18x training speedup on macOS
- Zero `unsafe` code in the ML pipeline

## Quick Start

Add to `Cargo.toml`:

```toml
[dependencies]
axiom = { git = "https://github.com/olliverc1985/AXIOM" }
```

Build and classify:

```rust
use axiom::transformer::TransformerConfig;
use axiom::vocab::Vocab;
use axiom::encoder::SemanticEncoder;
use axiom::features::{StructuralEncoder, Tokeniser};
use axiom::classifier::ClassificationHead;

// Configure a 2-layer transformer
let vocab = Vocab::build_from_corpus(&training_texts, 8192);
let config = TransformerConfig {
    vocab_size: vocab.max_size,
    hidden_dim: 128,
    num_heads: 4,
    num_layers: 2,
    ff_dim: 512,
    max_seq_len: 128,
    pooling: "mean".to_string(),
    activation: "gelu".to_string(),
};

// Build encoder and classifier
let semantic = SemanticEncoder::new_with_config(vocab, config);
let structural = StructuralEncoder::new(128, Tokeniser::default_tokeniser());
let head = ClassificationHead::new(256, 42); // 256 input dim, seed 42

// Classify: structural (128) + semantic (128) → fused (256) → 3 classes
let sf = structural.encode_text_readonly("What is the capital of France?");
let se = semantic.encode("What is the capital of France?");
let mut fused = Vec::with_capacity(256);
fused.extend_from_slice(&sf.data);
fused.extend_from_slice(&se.data);
let (class, confidence) = head.classify(&fused);
```

## Example Use Cases

- **[LLM Query Routing](examples/llm-routing/)** — Classify queries by complexity to route to appropriate model tiers. 94.8% accuracy on 1,000 diverse queries. Full training pipeline and trained weights included.
- **[Fraud Detection](examples/fraud-detection/)** — Classify messages as clean, suspicious, or blocked based on linguistic patterns.
- **[Content Moderation](examples/content-moderation/)** — Triage user content by severity for moderation queues.
- **[Query Triage](examples/query-triage/)** — Route queries by complexity before calling an expensive API.

Run any example:

```bash
cargo run --release --example fraud-detection
cargo run --release --example eval-axiom -- --weights axiom_router_weights.json
```

## Architecture

```
Input Text
  |
  ├──> Structural Encoder (128-dim)
  |      Position-weighted tokens, type-token ratio, avg token length,
  |      punctuation density, normalised count. Deterministic, zero parameters.
  |
  ├──> Semantic Encoder (128-dim)
  |      2-layer transformer, 4 heads, 512-dim FFN.
  |      Pretrained on sentence similarity, fine-tuned for classification.
  |
  └──> Always-Fuse ──> Concatenate (256-dim) ──> Linear ──> N-class Softmax
```

Both encoders run on every query. The structural features provide syntactic signal while the semantic encoder captures meaning-level patterns.

The framework also includes a sparse computation graph with four traversal directions (forward, lateral, feedback, temporal), enabling inter-node communication during classification. Every other router surveyed (75+) makes a single-pass decision.

## LLM Routing Results

The included LLM routing example achieves:

| Metric | Value |
|--------|-------|
| Overall accuracy | 94.8% on 1,000 diverse queries |
| Surface tier | 96.6% |
| Reasoning tier | 93.9% |
| Deep tier | 93.3% |
| Mean latency | 90 us |
| P99 latency | 465 us |
| Total parameters | ~1.5M (37K semantic encoder, rest embeddings) |
| Training time | ~4 minutes on laptop CPU |
| Stability (10 seeds) | 94.0% +/- 2.1% |

### RouterBench Evaluation

Evaluated zero-shot on [RouterBench](https://github.com/withmartian/routerbench) (Hu et al., ICML 2024) — 36,511 queries, 11 models.

| Router | Accuracy | Avg $/query | Cost Reduction |
|--------|----------|-------------|----------------|
| Always-Cheap (Mistral-7B) | 50.0% | $0.000139 | 98.3% |
| AXIOM Strategy C (cascade) | 70.0% | $0.005433 | 31.6% |
| Always-GPT-4 | 80.5% | $0.007943 | 0.0% |
| Oracle | 84.7% | $0.000378 | 95.2% |

AIQ of 0.0071 (near random baseline). Key finding: linguistic complexity classification and cost-optimal model selection are different objectives. AXIOM routes on how complex a query is, not on which model happens to get each specific question right.

## Honest Limitations

- **Not a general ML framework.** AXIOM is for small text classifiers (sub-million parameters). Use Burn or Candle for large models.
- **English only.** The tokeniser and training data are English. Other languages produce mostly UNK tokens.
- **No online learning.** Cannot adapt from production mistakes without retraining.
- **RouterBench AIQ near random.** Complexity-based routing is not cost-optimal model selection. See the paper for analysis.
- **Evaluation scale.** The 1,000-query eval set is independently constructed but still relatively small.
- **Tier accuracy, not answer quality.** The 94.8% measures classification accuracy, not downstream LLM answer correctness.
- **Stability.** 9/10 random seeds hit all targets. 1/10 dipped on one tier.
- **Vocabulary.** 8,192-token word-level vocabulary has limited coverage of rare/technical terms.

## Building

```bash
cargo build --release

# With Metal GPU support (macOS only)
cargo build --release --features gpu

# Run tests (170 tests)
cargo test
```

## API Reference

| Module | Description |
|--------|-------------|
| `axiom::transformer` | `TransformerConfig`, `TransformerEncoder`, forward/backward passes, `AdamState` |
| `axiom::vocab` | `Vocab` — word-level vocabulary with frequency-based construction |
| `axiom::encoder` | `SemanticEncoder` — transformer + pooling for text embeddings |
| `axiom::features` | `StructuralEncoder` — deterministic syntactic features (128-dim) |
| `axiom::classifier` | `ClassificationHead` — N-class linear softmax classifier |
| `axiom::router` | `AxiomRouter` — full LLM routing pipeline (structural + semantic + classifier) |
| `axiom::training` | STS training loops, loss functions, data loading |
| `axiom::optimiser` | `AdamState` — Adam optimiser with configurable learning rate |
| `axiom::weights` | Weight serialisation types for save/load |
| `axiom::metrics` | `accuracy`, `confusion_matrix`, `pearson` correlation |

## Paper

- Zenodo: [https://doi.org/10.5281/zenodo.19016401](https://doi.org/10.5281/zenodo.19016401)
- Paper covers the structural encoder architecture. Updated paper with dual-encoder and RouterBench results forthcoming.

## License

[MIT](LICENSE)
