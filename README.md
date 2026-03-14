# AXIOM

**Adaptive eXecution with Intelligent Operations Memory**

A sparse dynamic routing architecture for cost-efficient LLM inference.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19007918.svg)](https://doi.org/10.5281/zenodo.19007918)

AXIOM routes incoming queries across three model tiers — Surface, Reasoning, and Deep — using a 128-dimensional structural encoder and a hierarchical resolver with dynamic coalition formation and non-local graph communication. No preference data, no GPU, no ML frameworks. Pure Rust.

## Results

| Metric | Value |
|---|---|
| Validation accuracy | 89.5% (105 queries) |
| Simple routing accuracy | 95.0% |
| Adversarial accuracy | 65.0% (40 queries) |
| Cost savings vs all-Opus | 58.1% |
| Routing latency | 1,311 μs |
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

## Quick start

```bash
# Build
cargo build --release

# Run benchmark (trains + evaluates)
cargo run --release --bin axiom-bench

# Run tests
cargo test --workspace
```

## Project structure

```
axiom-core/     Core library — encoder, graph, resolver, cache
axiom-bench/    Benchmark binary — training, validation, adversarial eval
axiom-tuner/    Auto-tuner — adjusts thresholds between runs
axiom-datasets/ Corpus data (simple/moderate/complex sentences)
```

## Paper

The full paper is available as [`axiom_paper.docx`](axiom_paper.docx) and on [Zenodo](https://doi.org/10.5281/zenodo.19007918).

## License

[MIT](LICENSE)
