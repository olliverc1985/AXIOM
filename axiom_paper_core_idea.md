# AXIOM Core Architectural Idea: Non-Local Communication in Sparse Routing Graphs

## The Problem with Every Other Router

Every existing LLM router makes an **isolated, one-shot decision**:

| System | Decision Model | Communication |
|--------|---------------|---------------|
| RouteLLM | Single forward pass through BERT/MF → scalar score | None — one model, one output |
| FrugalGPT | Sequential cascade: try model A, score it, maybe try B | Vertical only — no lateral |
| AutoMix | Self-verification: ask the model if it's confident | Self-loop only |
| Hybrid LLM | Binary classifier → strong or weak | None — single prediction |
| CSCR | k-NN lookup in embedding space | None — nearest neighbor |
| kNN Router | k=5 nearest neighbors → vote | Neighbor voting only (flat) |

**No existing router has nodes that communicate with each other about the routing decision.**

## AXIOM's Core Idea: Nodes Talk to Distant Nodes

AXIOM's sparse computation graph supports **four distinct communication patterns** that allow information to flow non-locally:

### 1. Forward Traversal (Surface → Reasoning → Deep)
Standard escalation. A Surface node processes input, and if confidence is below threshold, a **conditional edge** fires to activate a Reasoning node. The edge has a condition gate:

```rust
pub enum EdgeCondition {
    Always,
    IfConfidenceAbove(f32),
    IfConfidenceBelow(f32),
    IfTier(Tier),
}
```

The edge only traverses if the condition evaluates to true given the current routing state. This is **not** a fixed pipeline — which edges fire depends on the input.

### 2. Lateral Traversal (Node → Distant Same-Tier Node)
When a Surface node produces low confidence, it doesn't immediately escalate. Instead, **lateral edges** activate other Surface nodes to attempt the same input:

```rust
pub struct LateralEdge {
    pub from: String,        // source node
    pub to: String,          // destination (same tier, different specialisation)
    pub weight: f32,         // influence weight
    pub condition: LateralCondition,  // when to fire
}
```

This models **cortical column behaviour** — neighbouring columns try before escalating to higher cortex. A lateral node might specialise in different vocabulary patterns and recover confidence without escalation.

The RouteResult tracks this:
```rust
pub lateral_count: u32,                    // how many lateral attempts
pub lateral_prevented_escalation: u32,     // how many avoided escalation
```

### 3. Feedback Traversal (Deep → Reasoning → Surface)
When a Deep node resolves an input with high confidence (>0.90), it emits a **FeedbackSignal** upward:

```rust
pub struct FeedbackSignal {
    pub from_node: String,       // "deep_standalone_6"
    pub to_tier: Tier,           // Tier::Reasoning
    pub reason: FeedbackReason,  // LowConfidenceResolved | ContradictionDetected | CacheInvalidation
    pub confidence_delta: f32,   // -0.01 (lower confidence for similar future inputs)
}
```

This is **not backpropagation**. It is directional confidence nudging: "I (Deep) resolved this easily — you (Reasoning) should have caught it. Lower your base confidence so you escalate less next time."

Three feedback reasons:
- **LowConfidenceResolved** — deeper tier handled what shallower tier couldn't
- **ContradictionDetected** — tiers disagreed on the same input
- **CacheInvalidation** — cached result was wrong when reprocessed deeper

### 4. Temporal Traversal (Past → Present)
The temporal buffer stores recent routing results (ring buffer, capacity 16). When a new input arrives, if it's similar to a recent input (cosine similarity > 0.85), the past result **blends** into the current routing:

```
current_output = 0.7 * live_output + 0.3 * temporal_match
```

This means the routing decision for input N is influenced by the routing decision for input N-3, if they're semantically related. The system has **memory across routing decisions**.

### 5. Dynamic Coalition Formation (Cross-Tier Collaboration)
When a query escalates past Surface, AXIOM doesn't just pick one Reasoning or Deep node. It forms a **temporary coalition** of 4 nodes selected stochastically from the Reasoning + Deep pools:

```
Coalition: [
    reasoning_standalone_4(Reasoning, bid=0.943, conf=0.801, FIRED),
    reasoning_standalone_13(Reasoning, bid=0.940, conf=0.800, FIRED),
    deep_standalone_6(Deep, bid=0.943, conf=0.829, FIRED),
    reasoning_standalone_12(Reasoning, bid=0.937, conf=0.799, FIRED)
]
→ resolved_by=deep_standalone_6 cross_tier=true
```

Nodes **bid** based on their confidence. The highest-bidding node's output becomes the final result. But all 4 nodes process the input — their weights all update via Hebbian learning. The coalition is a **competition** where nodes specialise by winning bids on different input types.

Cross-tier coalitions (e.g., a Reasoning node beating Deep nodes) are tracked separately — they represent cases where a cheaper tier's specialised node outperforms a more expensive tier's general node.

## Why This Matters

This non-local communication creates emergent behaviour that no flat classifier can produce:

1. **Specialisation without labels** — Nodes specialise by winning bids on inputs they're good at. No explicit "you handle math, you handle philosophy" — it emerges from Hebbian competition.

2. **Graceful degradation** — If the first Surface node fails, lateral edges try other Surface nodes before escalating. The system explores its own capacity before going expensive.

3. **Self-correction** — Feedback signals allow the system to learn from its own escalation mistakes. If Deep keeps resolving things that Reasoning could have handled, Reasoning's thresholds adjust.

4. **Context-sensitive routing** — The temporal buffer means a burst of complex queries influences routing of the next query, even if it looks simple in isolation.

5. **Interpretability** — The trace shows exactly which nodes fired, which edges activated, which lateral attempts were made, and which coalition won. Every routing decision is fully explainable:

```rust
pub struct TraceStep {
    pub node_id: String,
    pub tier: Tier,
    pub direction: TraversalDirection,  // Forward | Lateral | Feedback | Temporal
    pub confidence_in: f32,
    pub confidence_out: f32,
    pub was_cached: bool,
}
```

## Comparison: Communication Topology

```
RouteLLM:       Input → [BERT] → score → model
FrugalGPT:      Input → [Model₁] → score → maybe [Model₂] → score → maybe [Model₃]
AXIOM:           Input → [Surface₁] ←lateral→ [Surface₂]
                              ↓ (conditional edge, confidence < threshold)
                         [Reasoning₃] ←coalition→ [Deep₆]
                              ↑ (feedback signal)
                         [temporal_buffer] → blend
```

No other router in the literature has this topology. The closest is Tryage's "thalamic router" metaphor, but Tryage is a single neural network that predicts model performance — it doesn't have actual inter-node communication.

## Code Evidence

All four traversal directions are first-class in the type system:

```rust
pub enum TraversalDirection {
    Forward,   // Surface → Reasoning → Deep
    Lateral,   // Node → same-tier neighbour
    Feedback,  // Deep → Reasoning (confidence nudge)
    Temporal,  // Past result → current blend
}
```

Edge conditions are evaluated dynamically per-input:

```rust
pub fn should_traverse(&self, current_confidence: f32, current_tier: Tier) -> bool {
    match &self.condition {
        EdgeCondition::Always => true,
        EdgeCondition::IfConfidenceAbove(t) => current_confidence > *t,
        EdgeCondition::IfConfidenceBelow(t) => current_confidence < *t,
        EdgeCondition::IfTier(t) => current_tier == *t,
    }
}
```

Lateral edges model cortical column recovery:

```rust
/// A lateral connection between two nodes at the same tier.
/// When a node produces low confidence, lateral edges allow neighbouring
/// nodes at the same tier to attempt the input before escalating.
/// Models cortical column behaviour — neighbours try before escalating.
pub struct LateralEdge {
    pub from: String,
    pub to: String,
    pub weight: f32,
    pub condition: LateralCondition,
}
```

Feedback signals propagate upward (not backprop — directional nudging):

```rust
/// When a Deep node resolves an input with high confidence (above 0.90),
/// it emits a FeedbackSignal upward. This is not backprop — it is
/// directional confidence nudging based on resolution outcomes.
pub struct FeedbackSignal {
    pub from_node: String,
    pub to_tier: Tier,
    pub reason: FeedbackReason,
    pub confidence_delta: f32,
}
```
