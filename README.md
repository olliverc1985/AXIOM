# AXIOM

**Adaptive eXecution with Intelligent Operations Memory**

A sparse dynamic routing architecture for cost-efficient LLM inference.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19016401.svg)](https://doi.org/10.5281/zenodo.19016401)

AXIOM routes incoming queries across three model tiers — Surface, Reasoning, and Deep — using a 128-dimensional structural encoder and a hierarchical resolver with dynamic coalition formation and non-local graph communication. No preference data, no GPU, no ML frameworks. Pure Rust.

## Results

| Metric | Value |
|---|---|
| Validation accuracy | 89.5% (105 queries) |
| Simple routing accuracy | 95.0% |
| Adversarial accuracy | 65.0% (40 queries) |
| Cost savings vs all-Opus | 58.1% |
| Routing latency | 1,311 us |
| Parameters | 1,205,376 |
| Training time | 3.4 minutes (Apple Silicon, no GPU) |

## Architecture

AXIOM's sparse computation graph supports four traversal directions — forward, lateral, feedback, and temporal — enabling non-local communication between routing nodes. No existing LLM router provides this.

```
RouteLLM:  Input → [BERT] → score → model selection
FrugalGPT: Input → [Model1] → score → maybe [Model2]
AXIOM:     Input → [Surface1] ←lateral→ [Surface2] → (conditional edge) →
           [Reasoning3] ←coalition→ [Deep6] → (feedback signal upward)
           with temporal_buffer blending throughout
```

Key components:

- **Structural encoder** — 128-dimensional, five feature groups (G1-G5), vocabulary-independent
- **Sparse computation graph** — conditional, lateral, and feedback edges between compute nodes
- **Dynamic coalition formation** — Reasoning and Deep nodes bid on escalated queries, top-4 coalition assembled by weighted random sampling
- **Temporal buffer** — ring buffer blending recent routing decisions into current ones
- **Analytical initialisation** — Surface nodes frozen at simple-input mean direction; Reasoning/Deep nodes orthogonally initialised and trained via Oja's rule

## Quick start

```bash
# Build
cargo build --release

# Run benchmark (trains + evaluates, reproduces paper results)
cargo run --release --bin axiom-bench

# Run tests
cargo test

# Run auto-tuner (adjusts thresholds from bench logs)
cargo run --release --bin axiom-tuner
```

## Project structure

```
src/
  lib.rs                      Tensor type and crate root
  input/encoder.rs            Structural encoder (G1-G5 feature groups, 128-dim output)
  input/tokeniser.rs          Whitespace + punctuation tokeniser
  graph/engine.rs             SparseGraph with conditional routing
  graph/node.rs               LinearNode, ComputeNode trait, Oja's rule, contrastive learning
  graph/edge.rs               ConditionalEdge, LateralEdge
  cache/embedding_cache.rs    Cosine similarity cache (LRU, max 256)
  tiers/resolver.rs           Hierarchical resolver — calibration, coalition, training
  tiers/tier.rs               AxiomConfig (load/save axiom_config.json)
  tiers/feedback.rs           FeedbackSignal, FeedbackReason
  tuner.rs                    Auto-tuner rules
  bin/bench.rs                Benchmark binary — full training and evaluation
  bench/corpus.rs             2658-sentence training corpus
  bench/dashboard.rs          Live HTML dashboard (TCP server on :8080)
axiom-datasets/               Corpus data (simple/moderate/complex sentences)
autoresearch/                  Experiment harness, program spec, results
```

## License

[MIT](LICENSE)
