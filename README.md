# AXIOM

**Adaptive eXecution with Intelligent Operations Memory**

A sparse dynamic routing architecture for cost-efficient LLM inference.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19016401.svg)](https://doi.org/10.5281/zenodo.19016401)

AXIOM routes incoming queries across three model tiers — Surface, Reasoning, and Deep — using a 128-dimensional structural encoder and a hierarchical resolver with dynamic coalition formation and non-local graph communication. No preference data, no GPU, no ML frameworks. Pure Rust.

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

- **Structural encoder** — 128-dimensional, five feature groups (G1–G5), vocabulary-independent
- **Sparse computation graph** — conditional, lateral, and feedback edges between compute nodes
- **Dynamic coalition formation** — Reasoning and Deep nodes bid on escalated queries, top-4 coalition assembled by weighted random sampling
- **Temporal buffer** — ring buffer blending recent routing decisions into current ones
- **Analytical initialisation** — Surface nodes frozen at simple-input mean direction; Reasoning/Deep nodes orthogonally initialised and trained via Oja's rule

## Quick start

```bash
# Build
cargo build --release

# Run (trains on built-in corpus, then evaluates)
cargo run --release

# Run tests
cargo test
```

## Project structure

```
src/
  lib.rs          Public API — re-exports encoder, graph, resolver, types
  encoder.rs      Structural encoder (G1–G5 feature groups, 128-dim output)
  graph.rs        Sparse computation graph with conditional/lateral/feedback edges
  resolver.rs     Hierarchical resolver — routing, coalition, training, cache
  types.rs        Core types — Tier, TraceStep, RouteResult, AxiomConfig, etc.
autoresearch/     Experiment harness, program spec, results
axiom-datasets/   Corpus data (simple/moderate/complex sentences)
```

## License

[MIT](LICENSE)
