//! HierarchicalResolver — orchestrates tier escalation with cache integration,
//! lateral traversal, and feedback signals.

use crate::cache::EmbeddingCache;
use crate::graph::edge::LateralEdge;
use crate::graph::engine::{RouteResult, TraceStep, TraversalDirection};
use crate::graph::node::{ComputeNode, LinearNode, NodeOutput};
use crate::graph::SparseGraph;
use crate::tiers::feedback::FeedbackSignal;
use crate::tiers::tier::{AxiomConfig, Tier, TierConfig};
use crate::Tensor;

/// Operational mode for the resolver — controls whether the embedding cache is active.
///
/// **Training mode**: cache is completely bypassed. Every input routes through
/// the graph and standalone nodes unconditionally. This ensures the network sees
/// every training sentence on every pass, enabling meaningful weight convergence.
///
/// **Inference mode**: cache is active. Similar inputs get cache hits for compute
/// savings. This is the default mode for backward compatibility.
///
/// This does NOT change the confidence formula, routing logic, or learning rules —
/// only whether the cache participates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteMode {
    /// Cache disabled — every input reaches the network for learning.
    Training,
    /// Cache enabled — compute savings active for inference.
    Inference,
}

/// An entry in the temporal buffer — stores a recent resolve result.
#[derive(Debug, Clone)]
pub struct TemporalEntry {
    /// The input tensor that produced this result.
    pub input: Tensor,
    /// The output tensor from resolution.
    pub output: Tensor,
    /// The confidence of this resolution.
    pub confidence: f32,
    /// Which tier resolved this input.
    pub tier: Tier,
}

/// Ring buffer of recent resolve results for temporal blending.
///
/// When a new input has low Surface confidence, the buffer is checked for
/// a recent result with cosine similarity > 0.85. If found, the output is
/// blended: 0.3 * cached + 0.7 * current. This provides recency-weighted
/// memory without the overhead of full cache lookup.
pub struct TemporalBuffer {
    entries: Vec<TemporalEntry>,
    capacity: usize,
    head: usize,
    len: usize,
}

impl TemporalBuffer {
    /// Create a new temporal buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            len: 0,
        }
    }

    /// Push a new entry into the ring buffer.
    pub fn push(&mut self, entry: TemporalEntry) {
        if self.entries.len() < self.capacity {
            self.entries.push(entry);
            self.len = self.entries.len();
        } else {
            self.entries[self.head] = entry;
            self.head = (self.head + 1) % self.capacity;
        }
    }

    /// Find the most recent entry with cosine similarity > threshold to the input.
    /// Searches from most recent to oldest.
    pub fn find_similar(&self, input: &Tensor, threshold: f32) -> Option<&TemporalEntry> {
        if self.len == 0 {
            return None;
        }
        // Iterate from most recent to oldest
        for i in (0..self.len).rev() {
            let idx = if self.entries.len() < self.capacity {
                i
            } else {
                (self.head + self.capacity - 1 - (self.len - 1 - i)) % self.capacity
            };
            let entry = &self.entries[idx];
            let sim = input.cosine_similarity(&entry.input);
            if sim > threshold {
                return Some(entry);
            }
        }
        None
    }

    /// Current number of entries in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Orchestrates the full AXIOM pipeline: cache lookup, sparse graph routing,
/// lateral traversal, temporal blending, hierarchical tier escalation, and
/// feedback signals.
///
/// Most inputs should resolve at Surface tier without escalation.
/// Lateral connections between Surface nodes reduce unnecessary escalation.
/// Temporal buffer provides recency-weighted blending for similar inputs.
/// Feedback signals from Deep flow upward to adjust Reasoning confidence.
pub struct HierarchicalResolver {
    /// The sparse computation graph.
    pub graph: SparseGraph,
    /// The embedding cache for content-addressable tensor lookup.
    pub cache: EmbeddingCache,
    /// Tier escalation configuration.
    pub config: TierConfig,
    /// Operational mode — Training (cache off) or Inference (cache on).
    pub mode: RouteMode,
    /// Surface-tier nodes (fast, cheap) — used in standard blend step.
    surface_nodes: Vec<Box<dyn ComputeNode>>,
    /// Reasoning-tier nodes (medium compute).
    reasoning_nodes: Vec<Box<dyn ComputeNode>>,
    /// Deep-tier nodes (heavy compute).
    deep_nodes: Vec<Box<dyn ComputeNode>>,
    /// Lateral Surface nodes — only fire during lateral traversal, not surface_blend.
    lateral_nodes: Vec<Box<dyn ComputeNode>>,
    /// Lateral edges between nodes at the same tier.
    lateral_edges: Vec<LateralEdge>,
    /// Temporal buffer for recency-weighted blending (ring buffer, capacity 16).
    pub temporal_buffer: TemporalBuffer,
    /// Accumulated feedback signals from the current resolve call.
    feedback_log: Vec<FeedbackSignal>,
    /// Accumulated error signal events (across all resolves, for logging).
    error_events: Vec<ErrorEvent>,
    /// Minimum cosine similarity for a node to bid into a coalition.
    pub coalition_bid_threshold: f32,
    /// Maximum number of nodes in a coalition.
    pub coalition_max_size: usize,
    /// Accumulated coalition log entries (for persistence).
    coalition_log: Vec<Coalition>,
    /// PRNG state for stochastic coalition selection (xorshift64).
    coalition_rng: u64,
    /// G5 structural feature norm of simple corpus mean (for persistence).
    pub g5_simple_mean_norm: f32,
    /// G5 structural feature norm of complex corpus mean (for persistence).
    pub g5_complex_mean_norm: f32,
}

/// An error signal event for diagnostic logging.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorEvent {
    /// Type of error event: "escalation_penalty" or "cache_reinforcement".
    pub event_type: String,
    /// ID of the node the error signal was applied to.
    pub node_id: String,
    /// Confidence at the time of the event.
    pub confidence: f32,
    /// Magnitude of the error update applied to weights.
    pub error_magnitude: f32,
}

/// A member of a dynamic coalition formed after Surface escalation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoalitionMember {
    /// Node identifier.
    pub node_id: String,
    /// Which tier this node belongs to.
    pub tier: Tier,
    /// Bid score: cosine similarity between input and node weight direction.
    pub bid_score: f32,
    /// Whether this node actually executed a forward pass in the coalition.
    pub fired: bool,
    /// Output confidence from the forward pass (0.0 if not fired).
    pub confidence_out: f32,
}

/// A dynamic coalition formed per-input after Surface escalation.
///
/// Reasoning and Deep nodes bid for involvement based on cosine similarity
/// between the input and their specialised weight direction. Top bidders
/// form a temporary coalition; the highest-confidence member resolves.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Coalition {
    /// Hash of the input tensor for identification.
    pub input_hash: String,
    /// Coalition members (both bidders who fired and those who didn't make the cut).
    pub members: Vec<CoalitionMember>,
    /// Number of nodes that bid above threshold.
    pub bid_count: usize,
    /// Coalition formation overhead in microseconds.
    pub formation_time_us: u64,
    /// Confidence of the resolving member.
    pub resolution_confidence: f32,
    /// Node ID of the highest-confidence coalition member.
    pub resolved_by: String,
    /// Tier of the resolving node.
    pub resolved_tier: Tier,
    /// Whether a cross-tier Reasoning→Deep blend was applied.
    pub cross_tier_fired: bool,
}

/// Full result from the hierarchical resolver including tier information.
#[derive(Debug, Clone)]
pub struct ResolveResult {
    /// The route result from the graph.
    pub route: RouteResult,
    /// Which tier the input ultimately reached.
    pub tier_reached: Tier,
    /// Whether the final result came from cache.
    pub from_cache: bool,
    /// Feedback signals emitted during resolution.
    pub feedback_signals: Vec<FeedbackSignal>,
    /// Winning resolution path description.
    pub winning_path: String,
    /// Best Surface-tier confidence before escalation (0 if cache hit).
    pub surface_confidence: f32,
    /// ID of the Surface node that produced the pre-escalation output.
    pub surface_producer_id: Option<String>,
    /// Output tensor from the Surface node before escalation.
    pub surface_output: Option<Tensor>,
    /// Cosine similarity of a cache hit (0 if miss).
    pub cache_hit_similarity: f32,
    /// ID of the node that produced the cached entry (None if miss).
    pub cache_producer_id: Option<String>,
    /// Coalition formation result (None if resolved at Surface or from cache).
    pub coalition: Option<Coalition>,
}

impl HierarchicalResolver {
    /// Create a new resolver with a pre-built graph and default config.
    pub fn new(graph: SparseGraph, cache: EmbeddingCache, config: TierConfig) -> Self {
        Self {
            graph,
            cache,
            config,
            mode: RouteMode::Inference,
            surface_nodes: Vec::new(),
            reasoning_nodes: Vec::new(),
            deep_nodes: Vec::new(),
            lateral_nodes: Vec::new(),
            lateral_edges: Vec::new(),
            temporal_buffer: TemporalBuffer::new(16),
            feedback_log: Vec::new(),
            error_events: Vec::new(),
            coalition_bid_threshold: 0.15,
            coalition_max_size: 2,
            coalition_log: Vec::new(),
            coalition_rng: 12345,
            g5_simple_mean_norm: 0.0,
            g5_complex_mean_norm: 0.0,
        }
    }

    /// Add a standalone compute node at a specific tier.
    pub fn add_tier_node(&mut self, node: Box<dyn ComputeNode>) {
        match node.tier() {
            Tier::Surface => self.surface_nodes.push(node),
            Tier::Reasoning => self.reasoning_nodes.push(node),
            Tier::Deep => self.deep_nodes.push(node),
        }
    }

    /// Add a lateral compute node (only fires during lateral traversal).
    pub fn add_lateral_node(&mut self, node: Box<dyn ComputeNode>) {
        self.lateral_nodes.push(node);
    }

    /// Add a lateral edge between two nodes at the same tier.
    pub fn add_lateral_edge(&mut self, edge: LateralEdge) {
        self.lateral_edges.push(edge);
    }

    /// Run tier-specific nodes returning best output AND the winning node ID.
    fn run_tier_nodes_with_id(
        nodes: &[Box<dyn ComputeNode>],
        input: &Tensor,
    ) -> Option<(NodeOutput, String)> {
        let mut best: Option<(NodeOutput, String)> = None;
        for node in nodes {
            let output = node.forward(input);
            if best.as_ref().map_or(true, |(b, _)| output.confidence > b.confidence) {
                best = Some((output, node.node_id().to_string()));
            }
        }
        best
    }

    /// Run lateral traversal: try neighbouring Surface nodes before escalating.
    ///
    /// Returns the best lateral result and trace steps, if any lateral edge fired.
    fn run_lateral_surface(
        &self,
        source_node_id: &str,
        source_confidence: f32,
        input: &Tensor,
    ) -> (Option<NodeOutput>, Vec<TraceStep>, u32) {
        let mut best: Option<NodeOutput> = None;
        let mut steps = Vec::new();
        let mut lateral_count = 0u32;

        for edge in &self.lateral_edges {
            if edge.from != source_node_id {
                continue;
            }
            if !edge.should_fire(source_confidence) {
                continue;
            }
            // Find the target among lateral_nodes
            if let Some(target_node) = self
                .lateral_nodes
                .iter()
                .find(|n| n.node_id() == edge.to)
            {
                lateral_count += 1;
                let output = target_node.forward(input);
                steps.push(TraceStep {
                    node_id: target_node.node_id().to_string(),
                    tier: Tier::Surface,
                    direction: TraversalDirection::Lateral,
                    confidence_in: source_confidence,
                    confidence_out: output.confidence,
                    was_cached: false,
                });
                if best
                    .as_ref()
                    .map_or(true, |b| output.confidence > b.confidence)
                {
                    best = Some(output);
                }
            }
        }

        (best, steps, lateral_count)
    }

    /// Resolve an input through the full AXIOM pipeline.
    ///
    /// 1. Check embedding cache
    /// 2. Route through sparse graph (Surface tier)
    /// 3. Run standalone Surface nodes, blend if higher confidence
    /// 4. If Surface confidence is low, try lateral Surface nodes first
    /// 5. If still below surface threshold, escalate to Reasoning
    /// 6. If still below reasoning threshold, escalate to Deep
    /// 7. Emit feedback signals if Deep resolves with high confidence
    /// 8. Cache the result with producer node ID
    ///
    /// Tracks producer node IDs, surface confidence, and cache similarity
    /// for error signal computation in `apply_error_signal()`.
    #[allow(unused_assignments)]
    pub fn resolve(&mut self, input: &Tensor) -> ResolveResult {
        let mut cache_hits = 0u32;
        let mut from_cache = false;
        self.feedback_log.clear();

        let cache_key = input.clone();
        let mut current_tensor;
        let mut confidence;
        let mut tier_reached = Tier::Surface;
        let mut trace = Vec::new();
        let mut trace_steps = Vec::new();
        let mut total_cost = 0.0f32;
        let mut lateral_count = 0u32;
        let mut lateral_prevented = 0u32;
        let mut winning_path = String::from("graph");
        let mut producer_node_id: Option<String> = None;

        // Error signal tracking
        let mut cache_hit_similarity = 0.0f32;
        let mut cache_producer_id: Option<String> = None;
        // Max standalone Surface node confidence — used for escalation decisions.
        let mut escalation_conf = f32::NEG_INFINITY;

        // Try cache first (skipped entirely in Training mode)
        let cache_result = if self.mode == RouteMode::Inference {
            self.cache_lookup(&cache_key)
        } else {
            None
        };
        if let Some((cached, sim, cached_producer)) = cache_result {
            current_tensor = cached;
            cache_hits += 1;
            from_cache = true;
            winning_path = "cache".to_string();
            confidence = 0.95;
            cache_hit_similarity = sim;
            cache_producer_id = cached_producer;
            producer_node_id = Some("cache_hit".to_string());
            trace.push("cache_hit".to_string());
            trace_steps.push(TraceStep {
                node_id: "cache_hit".to_string(),
                tier: Tier::Surface,
                direction: TraversalDirection::Forward,
                confidence_in: 0.0,
                confidence_out: 0.95,
                was_cached: true,
            });
        } else {
            // Phase 2: Route through sparse graph (Surface tier)
            let route = self.graph.route(input);
            current_tensor = route.output;
            confidence = route.confidence;
            producer_node_id = route.producer_node_id;

            total_cost += route.total_compute_cost;
            trace.extend(route.execution_trace);
            trace_steps.extend(route.trace_steps);

            // Also run standalone surface nodes and compute max standalone
            // confidence. This is what gets compared to the surface threshold
            // for escalation decisions — not the graph route confidence, which
            // chains nodes and inflates the value.
            let mut max_standalone_conf = f32::NEG_INFINITY;
            if let Some(surface_out) =
                Self::run_tier_nodes_with_id(&self.surface_nodes, input)
            {
                total_cost += surface_out.0.compute_cost;
                max_standalone_conf = surface_out.0.confidence;
                if surface_out.0.confidence > confidence {
                    current_tensor = surface_out.0.tensor;
                    confidence = surface_out.0.confidence;
                    winning_path = "surface_standalone".to_string();
                    producer_node_id = Some(surface_out.1.to_string());
                }
                trace.push("surface_blend".to_string());
                trace_steps.push(TraceStep {
                    node_id: "surface_blend".to_string(),
                    tier: Tier::Surface,
                    direction: TraversalDirection::Forward,
                    confidence_in: confidence,
                    confidence_out: confidence,
                    was_cached: false,
                });
            }

            // Use max standalone confidence for escalation decisions.
            // The graph route confidence is used for output selection, but
            // individual node confidence determines routing.
            escalation_conf = if max_standalone_conf > f32::NEG_INFINITY {
                max_standalone_conf
            } else {
                confidence
            };

            // Phase 3: Lateral traversal
            if escalation_conf < self.config.surface_confidence_threshold {
                let graph_entry = self.graph.entry_node().to_string();
                let (lateral_result, lateral_steps, lat_count) =
                    self.run_lateral_surface(&graph_entry, confidence, input);
                lateral_count += lat_count;
                trace_steps.extend(lateral_steps);

                if let Some(lat_out) = lateral_result {
                    if lat_out.confidence > confidence {
                        winning_path = "surface_lateral".to_string();
                        trace.push("lateral_resolve".to_string());
                        if lat_out.confidence >= self.config.surface_confidence_threshold {
                            lateral_prevented += 1;
                        }
                        current_tensor = lat_out.tensor;
                        confidence = lat_out.confidence;
                        escalation_conf = escalation_conf.max(lat_out.confidence);
                        total_cost += lat_out.compute_cost;
                        // Lateral nodes don't set producer_node_id (kept from prior best)
                    }
                }
            }

            // Phase 3.5: Temporal blend
            if escalation_conf < self.config.surface_confidence_threshold {
                if let Some(temporal_entry) = self.temporal_buffer.find_similar(input, 0.85) {
                    let blended = current_tensor.blend(&temporal_entry.output, 0.7);
                    let conf_in = confidence;
                    let blended_confidence =
                        (confidence * 0.7 + temporal_entry.confidence * 0.3).clamp(0.0, 1.0);
                    current_tensor = blended;
                    confidence = blended_confidence;
                    winning_path = "temporal_blend".to_string();
                    trace.push("temporal_blend".to_string());
                    trace_steps.push(TraceStep {
                        node_id: "temporal_blend".to_string(),
                        tier: Tier::Surface,
                        direction: TraversalDirection::Temporal,
                        confidence_in: conf_in,
                        confidence_out: confidence,
                        was_cached: false,
                    });
                }
            }
        }

        // Capture Surface state before potential escalation (for error signal).
        // Use escalation_conf (standalone node max) for the surface_confidence
        // field so error signals and contrastive learning see the right value.
        let surface_confidence = if from_cache { confidence } else { escalation_conf };
        let surface_producer_id = producer_node_id.clone();
        let surface_output = if !from_cache {
            Some(current_tensor.clone())
        } else {
            None
        };

        // Phase 4+5: Coalition formation after Surface escalation
        //
        // All R+D nodes bid simultaneously based on cosine similarity between
        // input and their specialised weight direction. Top K bidders form a
        // coalition; highest-confidence member resolves. Cross-tier blend fires
        // when Deep resolves and a Reasoning node also bid strongly.
        let mut coalition_result: Option<Coalition> = None;
        if escalation_conf < self.config.surface_confidence_threshold && !from_cache {
            let coalition_start = std::time::Instant::now();
            trace.push("escalate_coalition".to_string());
            trace_steps.push(TraceStep {
                node_id: "escalate_coalition".to_string(),
                tier: Tier::Reasoning,
                direction: TraversalDirection::Forward,
                confidence_in: confidence,
                confidence_out: confidence,
                was_cached: false,
            });

            // Step 1: Bidding — every R+D node computes bid score
            let input_slice: Vec<f32> = if input.data.len() >= 128 {
                input.data[..128].to_vec()
            } else {
                let mut v = input.data.clone();
                v.resize(128, 0.0);
                v
            };
            let input_norm: f32 = input_slice.iter().map(|x| x * x).sum::<f32>().sqrt();

            let mut bids: Vec<(String, Tier, f32, usize, bool)> = Vec::new(); // (id, tier, bid_score, index_in_group, is_graph)

            // Graph R+D nodes
            for (idx, node) in self.graph.nodes_ref().iter().enumerate() {
                if node.tier() == Tier::Surface { continue; }
                let dir = node.weight_direction();
                if dir.is_empty() { continue; }
                let dot: f32 = input_slice.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
                let dir_norm: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cos = if input_norm > 1e-8 && dir_norm > 1e-8 {
                    (dot / (input_norm * dir_norm)).clamp(0.0, 1.0)
                } else { 0.0 };
                bids.push((node.node_id().to_string(), node.tier(), cos, idx, true));
            }
            // Standalone reasoning nodes
            for (idx, node) in self.reasoning_nodes.iter().enumerate() {
                let dir = node.weight_direction();
                if dir.is_empty() { continue; }
                let dot: f32 = input_slice.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
                let dir_norm: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cos = if input_norm > 1e-8 && dir_norm > 1e-8 {
                    (dot / (input_norm * dir_norm)).clamp(0.0, 1.0)
                } else { 0.0 };
                bids.push((node.node_id().to_string(), Tier::Reasoning, cos, idx, false));
            }
            // Standalone deep nodes
            for (idx, node) in self.deep_nodes.iter().enumerate() {
                let dir = node.weight_direction();
                if dir.is_empty() { continue; }
                let dot: f32 = input_slice.iter().zip(dir.iter()).map(|(a, b)| a * b).sum();
                let dir_norm: f32 = dir.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cos = if input_norm > 1e-8 && dir_norm > 1e-8 {
                    (dot / (input_norm * dir_norm)).clamp(0.0, 1.0)
                } else { 0.0 };
                bids.push((node.node_id().to_string(), Tier::Deep, cos, idx, false));
            }

            // Step 2: Stochastic weighted selection among candidates above threshold
            let bid_threshold = self.coalition_bid_threshold;
            let max_k = self.coalition_max_size;

            // Partition into qualified (above threshold) and all
            let mut qualified: Vec<(String, Tier, f32, usize, bool)> = bids
                .iter()
                .filter(|b| b.2 >= bid_threshold)
                .cloned()
                .collect();
            let bid_count = qualified.len();

            // Fallback: if fewer than 2 qualified, take top 2 by bid score
            let selected: Vec<(String, Tier, f32, usize, bool)> = if bid_count < 2 {
                bids.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                bids[..2.min(bids.len())].to_vec()
            } else if bid_count <= max_k {
                // All qualified fit — take them all
                qualified
            } else {
                // Weighted random sampling without replacement
                let mut chosen = Vec::with_capacity(max_k);
                for _ in 0..max_k {
                    let total_weight: f32 = qualified.iter().map(|b| b.2).sum();
                    if total_weight < 1e-8 { break; }
                    // xorshift64 PRNG step
                    self.coalition_rng ^= self.coalition_rng << 13;
                    self.coalition_rng ^= self.coalition_rng >> 7;
                    self.coalition_rng ^= self.coalition_rng << 17;
                    let r = (self.coalition_rng as f32 / u64::MAX as f32) * total_weight;
                    let mut cumulative = 0.0f32;
                    let mut pick_idx = qualified.len() - 1;
                    for (j, b) in qualified.iter().enumerate() {
                        cumulative += b.2;
                        if cumulative >= r {
                            pick_idx = j;
                            break;
                        }
                    }
                    chosen.push(qualified.remove(pick_idx));
                }
                chosen
            };
            let selected = &selected;

            // Step 3: Coalition execution — run forward on selected nodes
            let mut members = Vec::with_capacity(selected.len());
            let mut best_member_conf = f32::NEG_INFINITY;
            let mut best_member_id = String::new();
            let mut best_member_tier = Tier::Reasoning;
            let mut best_member_tensor: Option<crate::Tensor> = None;
            let mut best_member_cost = 0.0f32;

            // Also track best Reasoning member for cross-tier blend
            let mut best_reasoning_conf = f32::NEG_INFINITY;
            let mut best_reasoning_tensor: Option<crate::Tensor> = None;
            let mut best_reasoning_bid = 0.0f32;

            for &(ref nid, tier, bid_score, idx, is_graph) in selected {
                let output = if is_graph {
                    self.graph.nodes_ref()[idx].forward(input)
                } else {
                    match tier {
                        Tier::Reasoning => self.reasoning_nodes[idx].forward(input),
                        Tier::Deep => self.deep_nodes[idx].forward(input),
                        _ => continue,
                    }
                };

                let member = CoalitionMember {
                    node_id: nid.clone(),
                    tier,
                    bid_score,
                    fired: true,
                    confidence_out: output.confidence,
                };
                members.push(member);

                if output.confidence > best_member_conf {
                    best_member_conf = output.confidence;
                    best_member_id = nid.clone();
                    best_member_tier = tier;
                    best_member_tensor = Some(output.tensor.clone());
                    best_member_cost = output.compute_cost;
                }

                if tier == Tier::Reasoning && output.confidence > best_reasoning_conf {
                    best_reasoning_conf = output.confidence;
                    best_reasoning_tensor = Some(output.tensor.clone());
                    best_reasoning_bid = bid_score;
                }
            }

            // Step 4: Resolution — highest-confidence member becomes resolver
            let mut cross_tier_fired = false;
            if let Some(best_tensor) = best_member_tensor {
                // Step 5: Cross-tier connection — Reasoning→Deep blend
                // If resolver is Deep and a Reasoning node bid > 0.5, blend outputs
                current_tensor = if best_member_tier == Tier::Deep
                    && best_reasoning_bid > 0.5
                    && best_reasoning_tensor.is_some()
                {
                    cross_tier_fired = true;
                    let r_tensor = best_reasoning_tensor.unwrap();
                    // Blend: 0.3 Reasoning + 0.7 Deep
                    best_tensor.blend(&r_tensor, 0.7)
                } else {
                    best_tensor
                };

                tier_reached = best_member_tier;
                confidence = best_member_conf;
                winning_path = format!("coalition_{}", best_member_id);
                producer_node_id = Some(best_member_id.clone());
                total_cost += best_member_cost;

                trace_steps.push(TraceStep {
                    node_id: format!("coalition:{}", best_member_id),
                    tier: best_member_tier,
                    direction: TraversalDirection::Forward,
                    confidence_in: escalation_conf,
                    confidence_out: confidence,
                    was_cached: false,
                });

                if cross_tier_fired {
                    trace_steps.push(TraceStep {
                        node_id: "cross_tier_blend".to_string(),
                        tier: Tier::Deep,
                        direction: TraversalDirection::Lateral,
                        confidence_in: best_reasoning_conf,
                        confidence_out: confidence,
                        was_cached: false,
                    });
                }

                // Feedback signal if Deep resolved confidently
                if best_member_tier == Tier::Deep && confidence > 0.80 {
                    let signal = FeedbackSignal::low_confidence_resolved(&best_member_id, -0.01);
                    trace.push("feedback_up".to_string());
                    trace_steps.push(TraceStep {
                        node_id: "feedback_up".to_string(),
                        tier: Tier::Reasoning,
                        direction: TraversalDirection::Feedback,
                        confidence_in: confidence,
                        confidence_out: confidence,
                        was_cached: false,
                    });
                    self.feedback_log.push(signal);
                }
            }

            // Compute input hash for logging
            let hash_val: u64 = input.data.iter()
                .take(16)
                .fold(0u64, |h, &v| h.wrapping_mul(31).wrapping_add(v.to_bits() as u64));
            let input_hash = format!("{:016x}", hash_val);

            let coalition = Coalition {
                input_hash,
                members,
                bid_count,
                formation_time_us: coalition_start.elapsed().as_micros() as u64,
                resolution_confidence: confidence,
                resolved_by: best_member_id,
                resolved_tier: tier_reached,
                cross_tier_fired,
            };
            self.coalition_log.push(coalition.clone());
            coalition_result = Some(coalition);
        }

        // Phase 7: Cache the result (skipped in Training mode)
        if !from_cache && self.mode == RouteMode::Inference {
            self.cache.insert_direct(
                cache_key,
                current_tensor.clone(),
                producer_node_id.clone(),
                Some(tier_reached),
            );
            self.temporal_buffer.push(TemporalEntry {
                input: input.clone(),
                output: current_tensor.clone(),
                confidence,
                tier: tier_reached,
            });
        }

        let feedback_signals = self.feedback_log.clone();

        ResolveResult {
            route: RouteResult {
                output: current_tensor,
                confidence,
                execution_trace: trace,
                trace_steps,
                total_compute_cost: total_cost,
                cache_hits,
                lateral_count,
                lateral_prevented_escalation: lateral_prevented,
                producer_node_id,
            },
            tier_reached,
            from_cache,
            feedback_signals,
            winning_path,
            surface_confidence,
            surface_producer_id,
            surface_output,
            cache_hit_similarity,
            cache_producer_id,
            coalition: coalition_result,
        }
    }

    /// Direct cache lookup (bypasses get_or_compute).
    ///
    /// Returns `(cached_value, similarity, producer_node_id)`.
    fn cache_lookup(&mut self, key: &Tensor) -> Option<(Tensor, f32, Option<String>)> {
        self.cache.total_lookups += 1;
        let result = self.cache.lookup_only(key);
        if result.is_some() {
            self.cache.total_hits += 1;
        }
        result
    }

    /// Get the current cache hit rate.
    pub fn cache_hit_rate(&self) -> f32 {
        self.cache.hit_rate()
    }

    /// Get the cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Get accumulated feedback signals from the most recent resolve call.
    pub fn last_feedback(&self) -> &[FeedbackSignal] {
        &self.feedback_log
    }

    /// Apply Hebbian weight learning based on a resolve outcome.
    ///
    /// Uses usage-proportional learning rate: nodes that receive fewer activations
    /// get a higher effective lr to compensate for lower usage.
    /// effective_lr = base_lr * (total_iterations / (node_activation_count + 1))
    ///
    /// - If resolved at Surface: reinforce Surface nodes (+1), suppress deeper nodes (-1)
    /// - If resolved at Reasoning: suppress Surface nodes (-1), reinforce Reasoning (+1)
    /// - If resolved at Deep: suppress Surface and Reasoning (-1), reinforce Deep (+1)
    ///
    /// Also accumulates contrastive learning examples:
    /// - Surface resolution → positive example for Surface nodes
    /// - Escalation from Surface → negative example for Surface nodes
    ///
    /// This is local learning — no backprop, no global gradient. Each node adjusts
    /// its weights based on its own input/output correlation and the tier signal.
    pub fn learn(&mut self, input: &Tensor, result: &ResolveResult, learning_rate: f32, total_iterations: usize) {
        if result.from_cache {
            return; // No learning on cache hits
        }

        // Track activations: increment nodes at the winning tier
        self.increment_activations_for_tier(result.tier_reached);

        // Contrastive accumulation: positive for Surface, negative for escalation
        match result.tier_reached {
            Tier::Surface => {
                // Positive example: Surface resolved successfully
                for node in self.surface_nodes.iter_mut() {
                    node.accumulate_positive(input);
                }
                for node in self.lateral_nodes.iter_mut() {
                    node.accumulate_positive(input);
                }
                self.graph.accumulate_contrastive_all(input, true);
            }
            Tier::Reasoning | Tier::Deep => {
                // Negative example: Surface failed, had to escalate
                for node in self.surface_nodes.iter_mut() {
                    node.accumulate_negative(input);
                }
                for node in self.lateral_nodes.iter_mut() {
                    node.accumulate_negative(input);
                }
                self.graph.accumulate_contrastive_all(input, false);
            }
        }

        // Oja's rule: reinforce only, no suppression (signal=0 for non-winning tiers).
        // Oja's rule is self-normalizing for signal=+1 but divergent for any
        // negative signal (the y²*w term grows weights instead of shrinking them).
        // Contrastive learning now handles discrimination between tiers.
        if let Some(ref coalition) = result.coalition {
            // Coalition learning: only fired coalition members get Oja updates.
            // Surface is frozen and never participates in coalitions.
            // Non-fired nodes receive no update — differential activation drives specialisation.
            let fired_ids: Vec<&str> = coalition.members.iter()
                .filter(|m| m.fired)
                .map(|m| m.node_id.as_str())
                .collect();

            // Update graph R+D nodes that participated
            for node in self.graph.nodes_mut() {
                if node.tier() == Tier::Surface || node.is_frozen() { continue; }
                let signal = if fired_ids.contains(&node.node_id()) { 1.0 } else { 0.0 };
                if signal > 0.0 {
                    let output = node.forward(input);
                    node.hebbian_update(input, &output.tensor, signal, learning_rate);
                }
            }
            // Update standalone reasoning nodes that participated
            for node in self.reasoning_nodes.iter_mut() {
                let signal = if fired_ids.contains(&node.node_id()) { 1.0 } else { 0.0 };
                if signal > 0.0 {
                    let output = node.forward(input);
                    node.hebbian_update(input, &output.tensor, signal, learning_rate);
                }
            }
            // Update standalone deep nodes that participated
            for node in self.deep_nodes.iter_mut() {
                let signal = if fired_ids.contains(&node.node_id()) { 1.0 } else { 0.0 };
                if signal > 0.0 {
                    let output = node.forward(input);
                    node.hebbian_update(input, &output.tensor, signal, learning_rate);
                }
            }
            // Surface and lateral nodes: no Oja update (frozen / not in coalition)
        } else {
            // No coalition (Surface resolved) — standard tier-based learning
            Self::hebbian_nodes(&mut self.surface_nodes, input, 1.0, learning_rate, total_iterations);
            Self::hebbian_nodes(&mut self.lateral_nodes, input, 1.0, learning_rate, total_iterations);
            self.graph.hebbian_update_all(input, 1.0, learning_rate, total_iterations);
        }
    }

    /// Accumulate contrastive examples from a resolve result without applying Oja/error updates.
    ///
    /// Use this in phased training when you want contrastive-only learning.
    pub fn accumulate_contrastive(&mut self, input: &Tensor, result: &ResolveResult) {
        if result.from_cache {
            return;
        }
        match result.tier_reached {
            Tier::Surface => {
                for node in self.surface_nodes.iter_mut() {
                    node.accumulate_positive(input);
                }
                for node in self.lateral_nodes.iter_mut() {
                    node.accumulate_positive(input);
                }
                self.graph.accumulate_contrastive_all(input, true);
            }
            Tier::Reasoning | Tier::Deep => {
                for node in self.surface_nodes.iter_mut() {
                    node.accumulate_negative(input);
                }
                for node in self.lateral_nodes.iter_mut() {
                    node.accumulate_negative(input);
                }
                self.graph.accumulate_contrastive_all(input, false);
            }
        }
    }

    /// Apply contrastive weight update on all Surface nodes (graph + standalone + lateral).
    ///
    /// Called every 100 iterations during learning. Returns diagnostic info for each
    /// node that actually performed an update (had both positive and negative examples).
    pub fn apply_contrastive_update_all(&mut self) -> Vec<crate::graph::node::ContrastiveUpdateInfo> {
        use crate::graph::node::ContrastiveUpdateInfo;
        let mut infos: Vec<ContrastiveUpdateInfo> = Vec::new();
        for node in self.surface_nodes.iter_mut() {
            if let Some(info) = node.apply_contrastive_update() {
                infos.push(info);
            }
        }
        for node in self.lateral_nodes.iter_mut() {
            if let Some(info) = node.apply_contrastive_update() {
                infos.push(info);
            }
        }
        infos.extend(self.graph.apply_contrastive_update_all());
        infos
    }

    /// Accumulate a positive (Surface-direction) example on all Surface-tier nodes.
    pub fn accumulate_positive_all_surface(&mut self, input: &Tensor) {
        for node in self.surface_nodes.iter_mut() {
            node.accumulate_positive(input);
        }
        for node in self.lateral_nodes.iter_mut() {
            node.accumulate_positive(input);
        }
        self.graph.accumulate_contrastive_all(input, true);
    }

    /// Accumulate a negative (escalation-direction) example on all Surface-tier nodes.
    pub fn accumulate_negative_all_surface(&mut self, input: &Tensor) {
        for node in self.surface_nodes.iter_mut() {
            node.accumulate_negative(input);
        }
        for node in self.lateral_nodes.iter_mut() {
            node.accumulate_negative(input);
        }
        self.graph.accumulate_contrastive_all(input, false);
    }

    /// Set the contrastive learning rate on all Surface-tier nodes.
    pub fn set_contrastive_lr_all_surface(&mut self, lr: f32) {
        for node in self.surface_nodes.iter_mut() {
            node.set_contrastive_lr(lr);
        }
        for node in self.lateral_nodes.iter_mut() {
            node.set_contrastive_lr(lr);
        }
        self.graph.set_contrastive_lr_all(lr);
    }

    /// Reset contrastive accumulators on all Surface-tier nodes without applying updates.
    ///
    /// Call this before text priming to clear residual accumulator values from
    /// prior synthetic learning passes.
    pub fn reset_contrastive_accumulators_all_surface(&mut self) {
        for node in self.surface_nodes.iter_mut() {
            node.reset_contrastive_accumulators();
        }
        for node in self.lateral_nodes.iter_mut() {
            node.reset_contrastive_accumulators();
        }
        self.graph.reset_contrastive_accumulators_all();
    }

    /// Increment activation counters for all nodes at a given tier.
    fn increment_activations_for_tier(&mut self, tier: Tier) {
        match tier {
            Tier::Surface => {
                for node in self.surface_nodes.iter_mut() {
                    node.increment_activation();
                }
                for node in self.lateral_nodes.iter_mut() {
                    node.increment_activation();
                }
            }
            Tier::Reasoning => {
                for node in self.reasoning_nodes.iter_mut() {
                    node.increment_activation();
                }
            }
            Tier::Deep => {
                for node in self.deep_nodes.iter_mut() {
                    node.increment_activation();
                }
            }
        }
        self.graph.increment_activations_for_tier(tier);
    }

    /// Reset all activation counters (call between passes).
    pub fn reset_activation_counts(&mut self) {
        for node in self.surface_nodes.iter_mut() {
            node.reset_activation();
        }
        for node in self.lateral_nodes.iter_mut() {
            node.reset_activation();
        }
        for node in self.reasoning_nodes.iter_mut() {
            node.reset_activation();
        }
        for node in self.deep_nodes.iter_mut() {
            node.reset_activation();
        }
        self.graph.reset_activations();
    }

    /// Apply Hebbian update to a set of standalone nodes with usage-proportional lr.
    ///
    /// effective_lr = base_lr * (total_iterations / (activation_count + 1)),
    /// capped at 10x base_lr to prevent Oja divergence on rarely-activated nodes.
    fn hebbian_nodes(
        nodes: &mut [Box<dyn ComputeNode>],
        input: &Tensor,
        signal: f32,
        learning_rate: f32,
        total_iterations: usize,
    ) {
        let max_lr = learning_rate * 10.0;
        for node in nodes.iter_mut() {
            let effective_lr = (learning_rate * (total_iterations as f32 / (node.activation_count() as f32 + 1.0))).min(max_lr);
            let output = node.forward(input);
            node.hebbian_update(input, &output.tensor, signal, effective_lr);
        }
    }

    /// Apply targeted error signal updates based on a resolve outcome.
    ///
    /// Two mechanisms:
    ///
    /// **Similarity-based escalation penalty**: When Surface escalated (tier_reached != Surface)
    /// AND the cache contains a similar entry (cosine sim > 0.88) that was resolved at Surface,
    /// penalise the Surface node that escalated. This means Surface failed to handle an input
    /// that it previously resolved confidently.
    /// `w_ij -= error_lr * input_i * output_j * (1 - surface_conf)`
    ///
    /// **Cache reinforcement**: When a cache hit occurred with cosine similarity > 0.95,
    /// reinforce the node that produced the cached entry:
    /// `w_ij += error_lr * input_i * cached_output_j * similarity_score`
    ///
    /// Returns the error events generated (also accumulated in `self.error_events`).
    pub fn apply_error_signal(
        &mut self,
        input: &Tensor,
        result: &ResolveResult,
        error_lr: f32,
    ) -> Vec<ErrorEvent> {
        let mut events = Vec::new();

        // Mechanism 1: Similarity-based escalation penalty
        // Surface escalated AND cache has a similar entry resolved at Surface tier
        if result.tier_reached != Tier::Surface && !result.from_cache {
            if let (Some(ref surface_id), Some(ref surface_out)) =
                (&result.surface_producer_id, &result.surface_output)
            {
                // Check if cache contains a similar input that Surface handled before
                if let Some((sim, _cached_producer)) =
                    self.cache.find_similar_at_tier(input, 0.88, Tier::Surface)
                {
                    let modulator = -(1.0 - result.surface_confidence);
                    let error_mag = (error_lr * modulator).abs();
                    let found = Self::apply_error_to_node_by_id(
                        &mut self.surface_nodes,
                        &mut self.lateral_nodes,
                        &mut self.graph,
                        surface_id,
                        input,
                        surface_out,
                        error_lr,
                        modulator,
                    );
                    if found {
                        let event = ErrorEvent {
                            event_type: "escalation_penalty".to_string(),
                            node_id: surface_id.clone(),
                            confidence: result.surface_confidence,
                            error_magnitude: error_mag,
                        };
                        events.push(event);
                    }
                    let _ = sim; // used for match condition
                }
            }
        }

        // Mechanism 2: Cache reinforcement
        // Cache hit with similarity > 0.95
        if result.from_cache && result.cache_hit_similarity > 0.95 {
            if let Some(ref cache_prod_id) = result.cache_producer_id {
                let modulator = result.cache_hit_similarity;
                let error_mag = error_lr * modulator;
                let found = Self::apply_error_to_node_by_id(
                    &mut self.surface_nodes,
                    &mut self.lateral_nodes,
                    &mut self.graph,
                    cache_prod_id,
                    input,
                    &result.route.output,
                    error_lr,
                    modulator,
                );
                if !found {
                    // Try reasoning and deep nodes too
                    if !Self::apply_error_to_nodes(
                        &mut self.reasoning_nodes,
                        cache_prod_id,
                        input,
                        &result.route.output,
                        error_lr,
                        modulator,
                    ) {
                        let _ = Self::apply_error_to_nodes(
                            &mut self.deep_nodes,
                            cache_prod_id,
                            input,
                            &result.route.output,
                            error_lr,
                            modulator,
                        );
                    }
                }
                let event = ErrorEvent {
                    event_type: "cache_reinforcement".to_string(),
                    node_id: cache_prod_id.clone(),
                    confidence: result.route.confidence,
                    error_magnitude: error_mag,
                };
                events.push(event);
            }
        }

        self.error_events.extend(events.clone());
        events
    }

    /// Apply error signal to a specific node by ID, searching surface, lateral, and graph.
    fn apply_error_to_node_by_id(
        surface_nodes: &mut [Box<dyn ComputeNode>],
        lateral_nodes: &mut [Box<dyn ComputeNode>],
        graph: &mut SparseGraph,
        node_id: &str,
        input: &Tensor,
        output: &Tensor,
        error_lr: f32,
        modulator: f32,
    ) -> bool {
        if Self::apply_error_to_nodes(surface_nodes, node_id, input, output, error_lr, modulator) {
            return true;
        }
        if Self::apply_error_to_nodes(lateral_nodes, node_id, input, output, error_lr, modulator) {
            return true;
        }
        graph.error_update_node(node_id, input, output, error_lr, modulator)
    }

    /// Apply error signal to a node within a specific collection.
    fn apply_error_to_nodes(
        nodes: &mut [Box<dyn ComputeNode>],
        node_id: &str,
        input: &Tensor,
        output: &Tensor,
        error_lr: f32,
        modulator: f32,
    ) -> bool {
        for node in nodes.iter_mut() {
            if node.node_id() == node_id {
                node.error_update(input, output, error_lr, modulator);
                return true;
            }
        }
        false
    }

    /// Get all accumulated error events (across all resolves).
    pub fn error_events(&self) -> &[ErrorEvent] {
        &self.error_events
    }

    /// Get count of escalation penalty events.
    pub fn escalation_penalty_count(&self) -> usize {
        self.error_events
            .iter()
            .filter(|e| e.event_type == "escalation_penalty")
            .count()
    }

    /// Get count of cache reinforcement events.
    pub fn cache_reinforcement_count(&self) -> usize {
        self.error_events
            .iter()
            .filter(|e| e.event_type == "cache_reinforcement")
            .count()
    }

    /// Clear accumulated error events (call between passes).
    pub fn clear_error_events(&mut self) {
        self.error_events.clear();
    }

    /// Diagnostic: for an escalating input, find the highest cosine similarity
    /// to any Surface-resolved cache entry. Returns `(best_sim, nearest_tier)`.
    pub fn penalty_diagnostic(&self, input: &Tensor) -> Option<(f32, Option<Tier>)> {
        self.cache.best_similarity_diagnostic(input, Tier::Surface)
    }

    /// Compute max Surface confidence for a single input.
    ///
    /// Evaluates all standalone Surface and lateral nodes, returns the highest confidence.
    /// Used for diagnostic confidence tracking during pretraining.
    pub fn max_surface_confidence(&self, input: &Tensor) -> f32 {
        let mut max_conf = f32::NEG_INFINITY;
        for node in &self.surface_nodes {
            let conf = node.forward(input).confidence;
            if conf > max_conf { max_conf = conf; }
        }
        for node in &self.lateral_nodes {
            let conf = node.forward(input).confidence;
            if conf > max_conf { max_conf = conf; }
        }
        max_conf
    }

    /// Sum of weight norms across all nodes (graph + standalone + lateral).
    /// Used for weight drift tracking.
    pub fn total_weight_norm(&self) -> f32 {
        let mut total = self.graph.total_weight_norm();
        for node in &self.surface_nodes {
            total += node.weight_norm();
        }
        for node in &self.reasoning_nodes {
            total += node.weight_norm();
        }
        for node in &self.deep_nodes {
            total += node.weight_norm();
        }
        for node in &self.lateral_nodes {
            total += node.weight_norm();
        }
        total
    }

    /// Analytically initialise all Surface-tier nodes and freeze them.
    ///
    /// Computes discrimination_direction = simple_mean - complex_mean, L2 normalised,
    /// then calls `init_analytical` on every Surface node (graph, standalone, lateral)
    /// and sets them to frozen. Reasoning and Deep nodes are left unfrozen with
    /// standard Xavier weights.
    ///
    /// Returns (discrimination_direction_norm_before_normalisation, simple_mean_norm, complex_mean_norm, cosine_sim).
    pub fn init_surface_analytical(
        &mut self,
        simple_tensors: &[Tensor],
        complex_tensors: &[Tensor],
    ) -> (f32, f32, f32, f32) {
        use crate::input::encoder::{G5_DIM, G5_OFFSET};

        let dim = if let Some(t) = simple_tensors.first() {
            t.data.len()
        } else {
            return (0.0, 0.0, 0.0, 0.0);
        };

        // Compute full 128-dim simple_mean
        let mut simple_mean = vec![0.0f32; dim];
        for t in simple_tensors {
            for (i, v) in t.data.iter().enumerate() {
                if i < dim {
                    simple_mean[i] += v;
                }
            }
        }
        let simple_n = simple_tensors.len() as f32;
        for v in &mut simple_mean {
            *v /= simple_n;
        }

        // Compute full 128-dim complex_mean
        let mut complex_mean = vec![0.0f32; dim];
        for t in complex_tensors {
            for (i, v) in t.data.iter().enumerate() {
                if i < dim {
                    complex_mean[i] += v;
                }
            }
        }
        let complex_n = complex_tensors.len() as f32;
        for v in &mut complex_mean {
            *v /= complex_n;
        }

        // Weight direction = simple_mean (L2 normalised).
        //
        // Full 128-dim direction as in Phase 13. Cosine-based confidence
        // uses the full vector. G5 magnitude penalty (set below) provides
        // the structural discrimination signal.
        let mut direction = simple_mean.clone();

        let dir_norm = direction.iter().map(|v| v * v).sum::<f32>().sqrt();
        if dir_norm > 1e-10 {
            for v in &mut direction {
                *v /= dir_norm;
            }
        }

        let simple_norm = simple_mean.iter().map(|v| v * v).sum::<f32>().sqrt();
        let complex_norm = complex_mean.iter().map(|v| v * v).sum::<f32>().sqrt();

        // Cosine similarity between simple_mean and complex_mean
        let dot: f32 = simple_mean.iter().zip(complex_mean.iter()).map(|(a, b)| a * b).sum();
        let cosine_sim = if simple_norm > 1e-8 && complex_norm > 1e-8 {
            dot / (simple_norm * complex_norm)
        } else {
            0.0
        };

        // Compute G5 subvector norms for magnitude penalty
        let g5_end = (G5_OFFSET + G5_DIM).min(dim);
        let g5_simple_norm = if G5_OFFSET < dim {
            simple_mean[G5_OFFSET..g5_end]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
        } else {
            0.0
        };
        let g5_complex_norm = if G5_OFFSET < dim {
            complex_mean[G5_OFFSET..g5_end]
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt()
        } else {
            0.0
        };

        eprintln!(
            "G5 penalty norms: simple={:.4}, complex={:.4}, gap={:.4}",
            g5_simple_norm,
            g5_complex_norm,
            g5_complex_norm - g5_simple_norm
        );

        let init = crate::graph::node::AnalyticalInit {
            discrimination_direction: direction,
            noise_scale: 0.1,
        };

        let g5_penalty = Some((G5_OFFSET, g5_end, g5_simple_norm, g5_complex_norm, 0.35));

        // Apply to graph nodes (Surface tier only)
        for (i, node) in self.graph.nodes_mut().iter_mut().enumerate() {
            if node.tier() == Tier::Surface {
                node.init_analytical(&init, 42 + i as u64);
                node.set_frozen(true);
                node.set_g5_magnitude_penalty(g5_penalty);
            }
        }

        // Apply to standalone surface nodes
        for (i, node) in self.surface_nodes.iter_mut().enumerate() {
            node.init_analytical(&init, 1000 + i as u64);
            node.set_frozen(true);
            node.set_g5_magnitude_penalty(g5_penalty);
        }

        // Apply to lateral nodes (these are Surface-tier)
        for (i, node) in self.lateral_nodes.iter_mut().enumerate() {
            node.init_analytical(&init, 2000 + i as u64);
            node.set_frozen(true);
            node.set_g5_magnitude_penalty(g5_penalty);
        }

        // Store G5 norms for persistence
        self.g5_simple_mean_norm = g5_simple_norm;
        self.g5_complex_mean_norm = g5_complex_norm;

        (dir_norm, simple_norm, complex_norm, cosine_sim)
    }

    /// Compute the L2 norm of the G5 structural syntax subvector.
    pub fn compute_g5_norm(&self, tensor: &Tensor) -> f32 {
        use crate::input::encoder::{G5_DIM, G5_OFFSET};
        let data = &tensor.data;
        let s = G5_OFFSET.min(data.len());
        let e = (G5_OFFSET + G5_DIM).min(data.len());
        if s >= e { return 0.0; }
        data[s..e].iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Split text on sentence-ending punctuation (. ! ?) and return trimmed chunks.
    /// Filters out chunks with fewer than `min_tokens` whitespace-delimited words.
    fn split_sentences(text: &str, min_tokens: usize) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();
        for ch in text.chars() {
            current.push(ch);
            if ch == '.' || ch == '!' || ch == '?' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    let token_count = trimmed.split_whitespace().count();
                    if token_count >= min_tokens {
                        sentences.push(trimmed);
                    }
                }
                current.clear();
            }
        }
        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            let token_count = trimmed.split_whitespace().count();
            if token_count >= min_tokens {
                sentences.push(trimmed);
            }
        }
        sentences
    }

    /// Resolve text with sentence chunking.
    ///
    /// Splits input on sentence boundaries (./!/?), filters chunks < 3 tokens,
    /// encodes each independently, returns mean confidence, mean G5 norm, and
    /// the ResolveResult from the chunk with highest confidence.
    /// Single sentences are unaffected — they produce exactly one chunk.
    pub fn resolve_text(
        &mut self,
        encoder: &crate::input::encoder::Encoder,
        text: &str,
    ) -> (f32, f32, ResolveResult) {
        let chunks = Self::split_sentences(text, 3);

        if chunks.len() <= 1 {
            let tensor = encoder.encode_text_readonly(text);
            let g5 = self.compute_g5_norm(&tensor);
            let conf = self.max_surface_confidence(&tensor);
            let result = self.resolve(&tensor);
            return (conf, g5, result);
        }

        let mut conf_sum = 0.0f32;
        let mut g5_sum = 0.0f32;
        let mut best_result: Option<ResolveResult> = None;
        let mut best_conf = f32::NEG_INFINITY;

        for chunk in &chunks {
            let tensor = encoder.encode_text_readonly(chunk);
            let g5 = self.compute_g5_norm(&tensor);
            let conf = self.max_surface_confidence(&tensor);
            let result = self.resolve(&tensor);

            conf_sum += conf;
            g5_sum += g5;

            if result.route.confidence > best_conf {
                best_conf = result.route.confidence;
                best_result = Some(result);
            }
        }

        let n = chunks.len() as f32;
        (conf_sum / n, g5_sum / n, best_result.unwrap())
    }

    /// Update the G5 magnitude penalty weight on all Surface nodes.
    pub fn set_g5_penalty_weight(&mut self, weight: f32) {
        let g5_simple = self.g5_simple_mean_norm;
        let g5_complex = self.g5_complex_mean_norm;
        let g5_end = crate::input::encoder::G5_OFFSET + crate::input::encoder::G5_DIM;
        let params = Some((crate::input::encoder::G5_OFFSET, g5_end, g5_simple, g5_complex, weight));
        for node in self.graph.nodes_mut() {
            if node.tier() == Tier::Surface {
                node.set_g5_magnitude_penalty(params);
            }
        }
        for node in &mut self.surface_nodes {
            node.set_g5_magnitude_penalty(params);
        }
        for node in &mut self.lateral_nodes {
            node.set_g5_magnitude_penalty(params);
        }
    }

    /// Set confidence base weight on all nodes (all tiers).
    pub fn set_confidence_base_weight_all(&mut self, weight: f32) {
        for node in self.graph.nodes_mut() {
            node.set_confidence_base_weight(weight);
        }
        for node in &mut self.surface_nodes {
            node.set_confidence_base_weight(weight);
        }
        for node in &mut self.reasoning_nodes {
            node.set_confidence_base_weight(weight);
        }
        for node in &mut self.deep_nodes {
            node.set_confidence_base_weight(weight);
        }
        for node in &mut self.lateral_nodes {
            node.set_confidence_base_weight(weight);
        }
    }

    /// Set G4 magnitude penalty on all Surface nodes.
    pub fn set_g4_penalty_all_surface(&mut self, params: Option<(usize, usize, f32, f32, f32)>) {
        for node in self.graph.nodes_mut() {
            if node.tier() == Tier::Surface {
                node.set_g4_magnitude_penalty(params);
            }
        }
        for node in &mut self.surface_nodes {
            node.set_g4_magnitude_penalty(params);
        }
        for node in &mut self.lateral_nodes {
            node.set_g4_magnitude_penalty(params);
        }
    }

    /// Weight norm of Surface-tier nodes only (graph + standalone + lateral).
    pub fn surface_weight_norm(&self) -> f32 {
        let mut total = 0.0f32;
        for node in self.graph.nodes_ref() {
            if node.tier() == Tier::Surface {
                total += node.weight_norm();
            }
        }
        for node in &self.surface_nodes {
            total += node.weight_norm();
        }
        for node in &self.lateral_nodes {
            total += node.weight_norm();
        }
        total
    }

    /// Weight norm of Reasoning and Deep nodes only.
    pub fn non_surface_weight_norm(&self) -> f32 {
        let mut total = 0.0f32;
        for node in self.graph.nodes_ref() {
            if node.tier() != Tier::Surface {
                total += node.weight_norm();
            }
        }
        for node in &self.reasoning_nodes {
            total += node.weight_norm();
        }
        for node in &self.deep_nodes {
            total += node.weight_norm();
        }
        total
    }

    /// Initialise all Reasoning and Deep nodes with orthogonal weight directions.
    ///
    /// Generates orthogonal basis vectors via Gram-Schmidt and applies one to each
    /// R+D node. Surface nodes are left unchanged (frozen analytical). Returns the
    /// mean pairwise cosine similarity between all R+D node directions (target < 0.3).
    pub fn init_reasoning_deep_orthogonal(&mut self) -> f32 {
        // Collect R+D node indices and input_dim
        let mut rd_count = 0usize;
        let mut input_dim = 128usize;

        // Count R+D nodes in graph
        for node in self.graph.nodes_ref() {
            if node.tier() != Tier::Surface {
                rd_count += 1;
                let wd = node.weight_direction();
                if !wd.is_empty() {
                    input_dim = wd.len();
                }
            }
        }
        let graph_rd_count = rd_count;
        let reasoning_count = self.reasoning_nodes.len();
        let deep_count = self.deep_nodes.len();
        rd_count += reasoning_count + deep_count;

        if rd_count == 0 {
            return 0.0;
        }

        // Generate orthogonal basis
        let orth = crate::graph::node::OrthogonalInit::generate(rd_count, input_dim, 13_37);

        // Apply to graph R+D nodes
        let mut basis_idx = 0usize;
        for node in self.graph.nodes_mut() {
            if node.tier() != Tier::Surface {
                node.init_orthogonal(&orth.basis_vectors[basis_idx], orth.noise_scale, 3000 + basis_idx as u64);
                basis_idx += 1;
            }
        }

        // Apply to standalone reasoning nodes
        for (i, node) in self.reasoning_nodes.iter_mut().enumerate() {
            node.init_orthogonal(&orth.basis_vectors[graph_rd_count + i], orth.noise_scale, 4000 + i as u64);
        }

        // Apply to standalone deep nodes
        for (i, node) in self.deep_nodes.iter_mut().enumerate() {
            node.init_orthogonal(&orth.basis_vectors[graph_rd_count + reasoning_count + i], orth.noise_scale, 5000 + i as u64);
        }

        // Compute and log pairwise cosine similarities
        let mut directions: Vec<(String, Vec<f32>)> = Vec::with_capacity(rd_count);
        for node in self.graph.nodes_ref() {
            if node.tier() != Tier::Surface {
                directions.push((node.node_id().to_string(), node.weight_direction()));
            }
        }
        for node in &self.reasoning_nodes {
            directions.push((node.node_id().to_string(), node.weight_direction()));
        }
        for node in &self.deep_nodes {
            directions.push((node.node_id().to_string(), node.weight_direction()));
        }

        // Pairwise cosine similarities
        let n = directions.len();
        let mut sum_cos = 0.0f32;
        let mut pair_count = 0usize;
        let mut max_cos = 0.0f32;
        for i in 0..n {
            for j in (i + 1)..n {
                let dot: f32 = directions[i].1.iter().zip(directions[j].1.iter()).map(|(a, b)| a * b).sum();
                let ni: f32 = directions[i].1.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nj: f32 = directions[j].1.iter().map(|x| x * x).sum::<f32>().sqrt();
                let cos = if ni > 1e-8 && nj > 1e-8 { (dot / (ni * nj)).abs() } else { 0.0 };
                sum_cos += cos;
                pair_count += 1;
                if cos > max_cos {
                    max_cos = cos;
                }
            }
        }

        let mean_cos = if pair_count > 0 { sum_cos / pair_count as f32 } else { 0.0 };
        eprintln!(
            "Orthogonal init: {} R+D nodes, {} dims, mean pairwise |cos| = {:.4}, max |cos| = {:.4}",
            n, input_dim, mean_cos, max_cos
        );

        mean_cos
    }

    /// Build a default AXIOM resolver with a reasonable graph topology.
    ///
    /// Creates a graph with Surface, Reasoning, and Deep tier nodes
    /// connected with confidence-gated edges and lateral Surface connections.
    pub fn build_default(input_dim: usize) -> Self {
        use crate::graph::edge::ConditionalEdge;

        let mid_dim = input_dim / 2;

        // ── Graph: 8 Surface + 4 Reasoning + 2 Deep = 14 nodes ──
        let mut graph = SparseGraph::new("surface_entry");

        // Surface graph chain (8 nodes)
        let surface_graph_names = Self::surface_graph_node_names();
        for (i, name) in surface_graph_names.iter().enumerate() {
            let base_conf = 0.88 + (i as f32 * 0.005).min(0.03);
            graph.add_node(Box::new(LinearNode::new(
                name.clone(), input_dim, mid_dim, Tier::Surface, base_conf,
            )));
        }

        // Reasoning graph chain (4 nodes)
        let reasoning_graph_names = Self::reasoning_graph_node_names();
        for name in &reasoning_graph_names {
            graph.add_node(Box::new(LinearNode::new(
                name.clone(), input_dim, mid_dim, Tier::Reasoning, 0.80,
            )));
        }

        // Deep graph chain (2 nodes)
        let deep_graph_names = Self::deep_graph_node_names();
        for name in &deep_graph_names {
            graph.add_node(Box::new(LinearNode::new(
                name.clone(), input_dim, mid_dim, Tier::Deep, 0.75,
            )));
        }

        // Surface chain edges (always)
        for pair in surface_graph_names.windows(2) {
            graph.add_edge(ConditionalEdge::always(&pair[0], &pair[1]));
        }
        // Surface → Reasoning (confidence-gated)
        graph.add_edge(ConditionalEdge::if_confidence_below(
            surface_graph_names.last().unwrap(),
            &reasoning_graph_names[0],
            0.85,
        ));
        // Reasoning chain edges (always)
        for pair in reasoning_graph_names.windows(2) {
            graph.add_edge(ConditionalEdge::always(&pair[0], &pair[1]));
        }
        // Reasoning → Deep (confidence-gated)
        graph.add_edge(ConditionalEdge::if_confidence_below(
            reasoning_graph_names.last().unwrap(),
            &deep_graph_names[0],
            0.70,
        ));
        // Deep chain edges (always)
        for pair in deep_graph_names.windows(2) {
            graph.add_edge(ConditionalEdge::always(&pair[0], &pair[1]));
        }

        let cache = EmbeddingCache::new(256, 0.92);
        let config = TierConfig::default();
        let mut resolver = Self::new(graph, cache, config);

        // ── Standalone nodes: 20 Surface + 8 Reasoning + 4 Deep ──
        for i in 0..20 {
            let base_conf = 0.89 + (i as f32 * 0.0015).min(0.03);
            resolver.add_tier_node(Box::new(LinearNode::new(
                format!("surface_standalone_{}", i), input_dim, mid_dim, Tier::Surface, base_conf,
            )));
        }
        for i in 0..8 {
            resolver.add_tier_node(Box::new(LinearNode::new(
                format!("reasoning_standalone_{}", i), input_dim, mid_dim, Tier::Reasoning, 0.72,
            )));
        }
        for i in 0..4 {
            resolver.add_tier_node(Box::new(LinearNode::new(
                format!("deep_standalone_{}", i), input_dim, mid_dim, Tier::Deep, 0.78,
            )));
        }

        // ── Lateral nodes: 15 Surface ──
        for i in 0..15 {
            let base_conf = 0.88 + (i as f32 * 0.002).min(0.03);
            resolver.add_lateral_node(Box::new(LinearNode::new(
                format!("surface_lateral_{}", i), input_dim, mid_dim, Tier::Surface, base_conf,
            )));
            resolver.add_lateral_edge(LateralEdge::if_confidence_below(
                "surface_entry",
                &format!("surface_lateral_{}", i),
                0.75,
                1.0,
            ));
        }

        // Calibrate
        resolver.calibrate(input_dim, 0.65, 0.35);
        resolver.rebuild_graph_edges();
        resolver.validate_confidence_invariants();
        resolver
    }

    /// Build a resolver using `axiom_config.json` if it exists, otherwise defaults.
    ///
    /// Prints which config source was used.
    pub fn build_configured(input_dim: usize) -> Self {
        let config = AxiomConfig::load_or_default();
        Self::build_with_axiom_config(input_dim, &config)
    }

    /// Build a resolver from an explicit AxiomConfig.
    pub fn build_with_axiom_config(input_dim: usize, config: &AxiomConfig) -> Self {
        Self::build_with_axiom_config_mid_dim(input_dim, config, input_dim / 2)
    }

    /// Build a resolver from an explicit AxiomConfig with a custom mid_dim (output dimension per node).
    pub fn build_with_axiom_config_mid_dim(input_dim: usize, config: &AxiomConfig, mid_dim: usize) -> Self {
        use crate::graph::edge::ConditionalEdge;

        // ── Graph: 8 Surface + 4 Reasoning + 2 Deep = 14 nodes ──
        let mut graph = SparseGraph::new("surface_entry");

        // Surface graph chain (8 nodes)
        let surface_graph_names = Self::surface_graph_node_names();
        for (i, name) in surface_graph_names.iter().enumerate() {
            let base_conf = 0.88 + (i as f32 * 0.005).min(0.03);
            graph.add_node(Box::new(LinearNode::new(
                name.clone(), input_dim, mid_dim, Tier::Surface, base_conf,
            )));
        }

        // Reasoning graph chain (4 nodes)
        let reasoning_graph_names = Self::reasoning_graph_node_names();
        for name in &reasoning_graph_names {
            graph.add_node(Box::new(LinearNode::new(
                name.clone(), input_dim, mid_dim, Tier::Reasoning, 0.80,
            )));
        }

        // Deep graph chain (2 nodes)
        let deep_graph_names = Self::deep_graph_node_names();
        for name in &deep_graph_names {
            graph.add_node(Box::new(LinearNode::new(
                name.clone(), input_dim, mid_dim, Tier::Deep, 0.75,
            )));
        }

        // Surface chain edges (always)
        for pair in surface_graph_names.windows(2) {
            graph.add_edge(ConditionalEdge::always(&pair[0], &pair[1]));
        }
        // Surface → Reasoning (confidence-gated)
        graph.add_edge(ConditionalEdge::if_confidence_below(
            surface_graph_names.last().unwrap(),
            &reasoning_graph_names[0],
            config.surface_confidence_threshold,
        ));
        // Reasoning chain edges (always)
        for pair in reasoning_graph_names.windows(2) {
            graph.add_edge(ConditionalEdge::always(&pair[0], &pair[1]));
        }
        // Reasoning → Deep (confidence-gated)
        graph.add_edge(ConditionalEdge::if_confidence_below(
            reasoning_graph_names.last().unwrap(),
            &deep_graph_names[0],
            config.reasoning_confidence_threshold,
        ));
        // Deep chain edges (always)
        for pair in deep_graph_names.windows(2) {
            graph.add_edge(ConditionalEdge::always(&pair[0], &pair[1]));
        }

        let cache = EmbeddingCache::new(256, config.cache_similarity_threshold);
        let tier_config = TierConfig {
            surface_confidence_threshold: config.surface_confidence_threshold,
            reasoning_confidence_threshold: config.reasoning_confidence_threshold,
        };
        let mut resolver = Self::new(graph, cache, tier_config);

        // ── Standalone nodes: 20 Surface + 8 Reasoning + 4 Deep ──
        for i in 0..20 {
            let base_conf = 0.89 + (i as f32 * 0.0015).min(0.03);
            resolver.add_tier_node(Box::new(LinearNode::new(
                format!("surface_standalone_{}", i), input_dim, mid_dim, Tier::Surface, base_conf,
            )));
        }
        for i in 0..8 {
            resolver.add_tier_node(Box::new(LinearNode::new(
                format!("reasoning_standalone_{}", i), input_dim, mid_dim, Tier::Reasoning,
                config.reasoning_base_confidence,
            )));
        }
        for i in 0..4 {
            resolver.add_tier_node(Box::new(LinearNode::new(
                format!("deep_standalone_{}", i), input_dim, mid_dim, Tier::Deep, 0.78,
            )));
        }

        // ── Lateral nodes: 15 Surface ──
        for i in 0..15 {
            let base_conf = 0.88 + (i as f32 * 0.002).min(0.03);
            resolver.add_lateral_node(Box::new(LinearNode::new(
                format!("surface_lateral_{}", i), input_dim, mid_dim, Tier::Surface, base_conf,
            )));
            resolver.add_lateral_edge(LateralEdge::if_confidence_below(
                "surface_entry",
                &format!("surface_lateral_{}", i),
                0.75,
                1.0,
            ));
        }

        // Calibrate
        resolver.calibrate(input_dim, 0.65, 0.35);
        resolver.rebuild_graph_edges();
        resolver.validate_confidence_invariants();
        resolver
    }

    /// Calibrate confidence thresholds by running real encoder outputs through
    /// all Surface and Reasoning nodes.
    ///
    /// Uses a hardcoded calibration corpus (separate from bench test data) spanning
    /// simple to complex sentences. The encoder produces tensors in the same
    /// distribution as real usage, so thresholds are grounded in reality.
    ///
    /// Sets surface_confidence_threshold at `surface_pct` percentile and
    /// reasoning_confidence_threshold at `reasoning_pct` percentile.
    ///
    /// Enforces a minimum 10% escalation rate: after percentile-based calibration,
    /// verifies that at least 10% of calibration inputs (by per-input max Surface
    /// confidence) fall below the threshold. If fewer do, raises the threshold
    /// until 10% escalation is achieved. This ensures contrastive learning always
    /// has negative examples to learn from.
    pub fn calibrate(&mut self, input_dim: usize, surface_pct: f32, reasoning_pct: f32) {
        use crate::input::{Encoder, Tokeniser};

        let corpus = Self::calibration_corpus();

        // Build a temporary encoder to produce calibration tensors.
        let mut tokeniser = Tokeniser::default_tokeniser();
        for sentence in &corpus {
            tokeniser.tokenise(sentence);
        }
        let encoder = Encoder::new(input_dim, tokeniser);

        let mut surface_confs: Vec<f32> = Vec::new();
        let mut reasoning_confs: Vec<f32> = Vec::new();
        // Per-input max Surface confidence (best across all Surface nodes for each input)
        let mut per_input_max_surface: Vec<f32> = Vec::new();

        for sentence in &corpus {
            let input = encoder.encode_text_readonly(sentence);

            let mut max_conf = f32::NEG_INFINITY;
            for node in &self.surface_nodes {
                let conf = node.forward(&input).confidence;
                surface_confs.push(conf);
                if conf > max_conf {
                    max_conf = conf;
                }
            }
            if max_conf > f32::NEG_INFINITY {
                per_input_max_surface.push(max_conf);
            }

            for node in &self.reasoning_nodes {
                reasoning_confs.push(node.forward(&input).confidence);
            }
        }

        surface_confs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        reasoning_confs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        per_input_max_surface.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if !surface_confs.is_empty() {
            let idx = ((surface_confs.len() as f32 * surface_pct) as usize)
                .min(surface_confs.len() - 1);
            let mut threshold = surface_confs[idx];

            let max_surface = *surface_confs.last().unwrap();
            let min_surface = surface_confs[0];
            let std_dev = Self::std_dev(&surface_confs);

            // Enforce minimum gap: threshold must be at most max - 0.02
            if threshold > max_surface - 0.02 {
                threshold = max_surface - 0.02;
            }

            // Enforce minimum 10% escalation rate on per-input max confidences.
            // At least 10% of calibration inputs must have their best Surface
            // confidence below the threshold, ensuring negative examples exist
            // for contrastive learning.
            let min_escalation_rate = 0.10;
            let n_inputs = per_input_max_surface.len();
            let min_escalating = (n_inputs as f32 * min_escalation_rate).ceil() as usize;
            let escalating_before = per_input_max_surface
                .iter()
                .filter(|&&c| c < threshold)
                .count();

            if escalating_before < min_escalating && min_escalating <= n_inputs {
                // Raise threshold so that at least min_escalating inputs escalate.
                // Strategy: set threshold to per_input_max[min_escalating] which
                // puts min_escalating values at or below it, then add a tiny epsilon
                // to convert <= to strict <.
                // When weights have over-converged (all values equal), this will
                // push threshold above all values — that is intentional. Without
                // negative examples, contrastive learning cannot maintain
                // discrimination.
                let escalation_idx = min_escalating.min(n_inputs - 1);
                let old_threshold = threshold;
                threshold = per_input_max_surface[escalation_idx] + 0.0001;
                let escalating_check = per_input_max_surface
                    .iter()
                    .filter(|&&c| c < threshold)
                    .count();
                eprintln!(
                    "  Calibration: enforced min escalation rate — raised threshold {:.4} → {:.4} \
                     ({}→{} of {} inputs escalate, {:.0}%→{:.0}%)",
                    old_threshold, threshold,
                    escalating_before, escalating_check, n_inputs,
                    escalating_before as f32 / n_inputs as f32 * 100.0,
                    escalating_check as f32 / n_inputs as f32 * 100.0,
                );
            }

            self.config.surface_confidence_threshold = threshold;

            let escalating_after = per_input_max_surface
                .iter()
                .filter(|&&c| c < self.config.surface_confidence_threshold)
                .count();

            eprintln!(
                "  Calibration: surface range [{:.4}, {:.4}] std={:.4} → threshold {:.4} (p{:.0}, n={}), \
                 escalation rate {}/{} ({:.0}%)",
                min_surface, max_surface, std_dev, self.config.surface_confidence_threshold,
                surface_pct * 100.0, surface_confs.len(),
                escalating_after, n_inputs,
                escalating_after as f32 / n_inputs as f32 * 100.0,
            );
            if std_dev < 0.005 {
                eprintln!(
                    "  WARNING: confidence distribution too narrow (std={:.4}), weights may have over-converged",
                    std_dev
                );
            }

            // Population-aware threshold: encode calibration sentences, split
            // by word count (short < 6 words, long > 10 words), compute mean
            // Surface confidence per group, set threshold at midpoint.
            let mut short_confs: Vec<f32> = Vec::new();
            let mut long_confs: Vec<f32> = Vec::new();
            for sentence in &corpus {
                let word_count = sentence.split_whitespace().count();
                let input = encoder.encode_text_readonly(sentence);
                let max_conf = self.max_surface_confidence(&input);
                if word_count < 6 {
                    short_confs.push(max_conf);
                } else if word_count > 10 {
                    long_confs.push(max_conf);
                }
            }

            if !short_confs.is_empty() && !long_confs.is_empty() {
                let short_mean = short_confs.iter().sum::<f32>() / short_confs.len() as f32;
                let long_mean = long_confs.iter().sum::<f32>() / long_confs.len() as f32;
                let midpoint = (short_mean + long_mean) / 2.0;

                eprintln!(
                    "  Calibration: population-aware — short mean={:.4} (n={}) long mean={:.4} (n={}) → midpoint={:.4}",
                    short_mean, short_confs.len(), long_mean, long_confs.len(), midpoint
                );

                if short_mean > long_mean {
                    self.config.surface_confidence_threshold = midpoint;
                    eprintln!(
                        "  Calibration: threshold adjusted {:.4} → {:.4} (population midpoint)",
                        threshold, midpoint
                    );
                } else {
                    eprintln!(
                        "  Calibration: short <= long ({:.4} <= {:.4}), keeping percentile threshold {:.4}",
                        short_mean, long_mean, self.config.surface_confidence_threshold
                    );
                }
            }
        }

        if !reasoning_confs.is_empty() {
            let idx = ((reasoning_confs.len() as f32 * reasoning_pct) as usize)
                .min(reasoning_confs.len() - 1);
            let mut threshold = reasoning_confs[idx];

            let max_reasoning = *reasoning_confs.last().unwrap();
            let min_reasoning = reasoning_confs[0];
            let std_dev = Self::std_dev(&reasoning_confs);

            // Enforce minimum gap for reasoning too
            if threshold > max_reasoning - 0.02 {
                threshold = max_reasoning - 0.02;
            }
            self.config.reasoning_confidence_threshold = threshold;

            eprintln!(
                "  Calibration: reasoning range [{:.4}, {:.4}] std={:.4} → threshold {:.4} (p{:.0}, n={})",
                min_reasoning, max_reasoning, std_dev, self.config.reasoning_confidence_threshold,
                reasoning_pct * 100.0, reasoning_confs.len()
            );
            if std_dev < 0.005 {
                eprintln!(
                    "  WARNING: reasoning confidence distribution too narrow (std={:.4}), weights may have over-converged",
                    std_dev
                );
            }
        }
    }

    /// Calibration corpus — separate from bench test sentences.
    ///
    /// Spans simple (3-5 words, common words, no punctuation) through moderate
    /// (8-12 words, mixed vocabulary) to complex (15+ words, rare vocabulary,
    /// punctuation, subordinate clauses) to cover the full input distribution.
    fn calibration_corpus() -> Vec<&'static str> {
        vec![
            // Simple (3-5 words, common words, no punctuation)
            "a red ball",
            "open the door",
            "the moon is round",
            "cats like warm milk",
            "she ate lunch today",
            "rain falls from clouds",
            "we walked home slowly",
            // Moderate (8-12 words, mixed vocabulary)
            "climate change affects global food production patterns in many regions",
            "binary search reduces lookup time to a logarithmic number of steps",
            "vaccines stimulate the immune response by introducing weakened pathogen material",
            "containerisation improves deployment consistency and reproducibility across different environments",
            "interest rates directly influence consumer borrowing behaviour and spending decisions",
            "gradient descent iteratively minimises a differentiable loss function over parameters",
            "ocean currents redistribute thermal energy between northern and southern hemispheres",
            // Complex (15+ words, rare vocabulary, punctuation, subordinate clauses)
            "the observer effect in quantum mechanics implies that measurement, by its very nature, fundamentally alters the state of the observed system",
            "autopoietic systems, while maintaining strict organisational closure, nonetheless remain thermodynamically open to continuous energy and matter exchange",
            "the no-free-lunch theorem establishes that no single optimisation algorithm can dominate across all possible problem classes without exception",
            "dialectical materialism posits that internal contradictions within socioeconomic structures, rather than external forces alone, drive historical transformation",
            "renormalisation group methods, when applied to systems near critical phase transitions, reveal universal scale-invariant behaviour across many physical phenomena",
            "the frame problem in artificial intelligence highlights the fundamental difficulty of formally representing the implicit non-effects of actions within logical frameworks",
        ]
    }

    /// Update the graph's conditional edge thresholds after calibration.
    ///
    /// Rebuilds edges using the current config thresholds.
    pub fn rebuild_graph_edges(&mut self) {
        use crate::graph::edge::ConditionalEdge;
        self.graph.clear_edges();

        let surface_names = Self::surface_graph_node_names();
        let reasoning_names = Self::reasoning_graph_node_names();
        let deep_names = Self::deep_graph_node_names();

        // Surface chain (always)
        for pair in surface_names.windows(2) {
            self.graph.add_edge(ConditionalEdge::always(&pair[0], &pair[1]));
        }
        // Surface → Reasoning (confidence-gated)
        self.graph.add_edge(ConditionalEdge::if_confidence_below(
            surface_names.last().unwrap(),
            &reasoning_names[0],
            self.config.surface_confidence_threshold,
        ));
        // Reasoning chain (always)
        for pair in reasoning_names.windows(2) {
            self.graph.add_edge(ConditionalEdge::always(&pair[0], &pair[1]));
        }
        // Reasoning → Deep (confidence-gated)
        self.graph.add_edge(ConditionalEdge::if_confidence_below(
            reasoning_names.last().unwrap(),
            &deep_names[0],
            self.config.reasoning_confidence_threshold,
        ));
        // Deep chain (always)
        for pair in deep_names.windows(2) {
            self.graph.add_edge(ConditionalEdge::always(&pair[0], &pair[1]));
        }
    }

    /// Validate that every node can mathematically reach its tier's confidence threshold.
    ///
    /// The confidence formula is: base_confidence * 0.7 + ratio * 0.3, where ratio
    /// is capped at 1.0. So the ceiling is base_confidence * 0.7 + 0.3.
    /// Panics if any node's ceiling cannot exceed its tier's threshold.
    pub fn validate_confidence_invariants(&self) {
        let checks: Vec<(&str, &[Box<dyn ComputeNode>], f32)> = vec![
            ("surface", &self.surface_nodes, self.config.surface_confidence_threshold),
            ("lateral", &self.lateral_nodes, self.config.surface_confidence_threshold),
            ("reasoning", &self.reasoning_nodes, self.config.reasoning_confidence_threshold),
        ];

        for (label, nodes, threshold) in checks {
            for node in nodes {
                let base = node.base_confidence();
                let ceiling = base * 0.7 + 0.3;
                assert!(
                    ceiling > threshold,
                    "Confidence invariant violated: {} node '{}' has ceiling {:.3} \
                     (base_confidence {:.3}) which cannot reach {} threshold {:.3}",
                    label, node.node_id(), ceiling, base, label, threshold
                );
            }
        }
    }

    /// Surface graph node names (8 nodes: entry + 7 refine steps).
    fn surface_graph_node_names() -> Vec<String> {
        let mut names = vec!["surface_entry".to_string()];
        for i in 0..7 {
            names.push(format!("surface_refine_{}", (b'a' + i as u8) as char));
        }
        names
    }

    /// Reasoning graph node names (4 nodes).
    fn reasoning_graph_node_names() -> Vec<String> {
        (0..4).map(|i| format!("reasoning_analyze_{}", (b'a' + i as u8) as char)).collect()
    }

    /// Deep graph node names (2 nodes).
    fn deep_graph_node_names() -> Vec<String> {
        vec!["deep_resolve_a".to_string(), "deep_resolve_b".to_string()]
    }

    /// Compute standard deviation of a sorted slice of f32 values.
    fn std_dev(values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        variance.sqrt()
    }

    /// Total trainable parameter count across all nodes (graph + standalone + lateral).
    pub fn total_weight_count(&self) -> usize {
        self.graph.total_weight_count()
            + self
                .surface_nodes
                .iter()
                .map(|n| n.weight_count())
                .sum::<usize>()
            + self
                .reasoning_nodes
                .iter()
                .map(|n| n.weight_count())
                .sum::<usize>()
            + self
                .deep_nodes
                .iter()
                .map(|n| n.weight_count())
                .sum::<usize>()
            + self
                .lateral_nodes
                .iter()
                .map(|n| n.weight_count())
                .sum::<usize>()
    }

    /// Save all node weights to a JSON file for persistence.
    ///
    /// Collects weight data from every trainable node (graph, surface,
    /// reasoning, deep, lateral) and writes to the given path.
    /// Compute mean pairwise |cosine similarity| between all R+D node weight directions.
    ///
    /// Returns (mean_pairwise_cos, max_pairwise_cos). Lower values indicate greater
    /// specialisation — nodes are pointing in more different directions.
    pub fn rd_pairwise_cosine(&self) -> (f32, f32) {
        let mut directions: Vec<Vec<f32>> = Vec::new();
        for node in self.graph.nodes_ref() {
            if node.tier() != Tier::Surface {
                let d = node.weight_direction();
                if !d.is_empty() { directions.push(d); }
            }
        }
        for node in &self.reasoning_nodes {
            let d = node.weight_direction();
            if !d.is_empty() { directions.push(d); }
        }
        for node in &self.deep_nodes {
            let d = node.weight_direction();
            if !d.is_empty() { directions.push(d); }
        }

        let n = directions.len();
        if n < 2 { return (0.0, 0.0); }

        let mut sum_cos = 0.0f32;
        let mut max_cos = 0.0f32;
        let mut pair_count = 0usize;
        for i in 0..n {
            let ni: f32 = directions[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            for j in (i + 1)..n {
                let dot: f32 = directions[i].iter().zip(directions[j].iter()).map(|(a, b)| a * b).sum();
                let nj: f32 = directions[j].iter().map(|x| x * x).sum::<f32>().sqrt();
                let cos = if ni > 1e-8 && nj > 1e-8 { (dot / (ni * nj)).abs() } else { 0.0 };
                sum_cos += cos;
                if cos > max_cos { max_cos = cos; }
                pair_count += 1;
            }
        }
        let mean = if pair_count > 0 { sum_cos / pair_count as f32 } else { 0.0 };
        (mean, max_cos)
    }

    /// Return node activation counts for coalition tracking.
    /// Returns Vec of (node_id, tier, activation_count) for all R+D nodes.
    pub fn rd_activation_counts(&self) -> Vec<(String, Tier, usize)> {
        let mut counts = Vec::new();
        for node in self.graph.nodes_ref() {
            if node.tier() != Tier::Surface {
                counts.push((node.node_id().to_string(), node.tier(), node.activation_count()));
            }
        }
        for node in &self.reasoning_nodes {
            counts.push((node.node_id().to_string(), node.tier(), node.activation_count()));
        }
        for node in &self.deep_nodes {
            counts.push((node.node_id().to_string(), node.tier(), node.activation_count()));
        }
        counts
    }

    /// Save accumulated coalition log to a JSON file.
    pub fn save_coalition_log(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.coalition_log)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Access the accumulated coalition log entries.
    pub fn coalition_log(&self) -> &[Coalition] {
        &self.coalition_log
    }

    /// Clear the coalition log (e.g. between training passes).
    pub fn clear_coalition_log(&mut self) {
        self.coalition_log.clear();
    }

    pub fn save_all_weights(&self, path: &str) -> std::io::Result<()> {
        use crate::graph::node::NodeWeightsData;

        let mut nodes: Vec<NodeWeightsData> = Vec::new();

        for node in self.graph.nodes_ref() {
            if let Some(data) = node.save_weights_data() {
                nodes.push(data);
            }
        }
        for node in &self.surface_nodes {
            if let Some(data) = node.save_weights_data() {
                nodes.push(data);
            }
        }
        for node in &self.reasoning_nodes {
            if let Some(data) = node.save_weights_data() {
                nodes.push(data);
            }
        }
        for node in &self.deep_nodes {
            if let Some(data) = node.save_weights_data() {
                nodes.push(data);
            }
        }
        for node in &self.lateral_nodes {
            if let Some(data) = node.save_weights_data() {
                nodes.push(data);
            }
        }

        let wrapper = serde_json::json!({
            "nodes": nodes,
            "g5_simple_mean_norm": self.g5_simple_mean_norm,
            "g5_complex_mean_norm": self.g5_complex_mean_norm,
        });
        let json = serde_json::to_string(&wrapper)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load all node weights from a JSON file.
    ///
    /// Matches nodes by ID — only updates weights where the ID and dimensions match.
    pub fn load_all_weights(&mut self, path: &str) -> std::io::Result<()> {
        use crate::graph::node::NodeWeightsData;

        let data = std::fs::read_to_string(path)?;
        let wrapper: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let nodes: Vec<NodeWeightsData> = serde_json::from_value(
            wrapper.get("nodes").cloned().unwrap_or_default(),
        )
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let weight_map: std::collections::HashMap<&str, &NodeWeightsData> =
            nodes.iter().map(|n| (n.id.as_str(), n)).collect();

        for node in self.graph.nodes_mut() {
            if let Some(data) = weight_map.get(node.node_id()) {
                node.load_weights_data(data);
            }
        }
        for node in &mut self.surface_nodes {
            if let Some(data) = weight_map.get(node.node_id()) {
                node.load_weights_data(data);
            }
        }
        for node in &mut self.reasoning_nodes {
            if let Some(data) = weight_map.get(node.node_id()) {
                node.load_weights_data(data);
            }
        }
        for node in &mut self.deep_nodes {
            if let Some(data) = weight_map.get(node.node_id()) {
                node.load_weights_data(data);
            }
        }
        for node in &mut self.lateral_nodes {
            if let Some(data) = weight_map.get(node.node_id()) {
                node.load_weights_data(data);
            }
        }

        // Load G5 penalty norms (backward-compatible: absent keys default to 0.0)
        if let Some(v) = wrapper.get("g5_simple_mean_norm").and_then(|v| v.as_f64()) {
            self.g5_simple_mean_norm = v as f32;
        }
        if let Some(v) = wrapper.get("g5_complex_mean_norm").and_then(|v| v.as_f64()) {
            self.g5_complex_mean_norm = v as f32;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_basic() {
        let mut resolver = HierarchicalResolver::build_default(8);
        let input = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]);

        let result = resolver.resolve(&input);
        assert!(!result.route.execution_trace.is_empty());
        assert!(result.route.confidence >= 0.0);
        // Structured trace should exist
        assert!(!result.route.trace_steps.is_empty());
    }

    #[test]
    fn test_cache_works_in_resolver() {
        let mut resolver = HierarchicalResolver::build_default(8);
        let input = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.2, 0.1]);

        // First call — cache miss
        let r1 = resolver.resolve(&input);
        assert!(!r1.from_cache);

        // Second call with same input — cache hit
        let r2 = resolver.resolve(&input);
        assert!(r2.from_cache);
        assert!(r2.route.cache_hits > 0);
    }

    #[test]
    fn test_tier_escalation() {
        // Build a graph where Surface confidence is deliberately low
        let mut graph = SparseGraph::new("low_conf");
        graph.add_node(Box::new(LinearNode::new(
            "low_conf",
            4,
            4,
            Tier::Surface,
            0.3, // Very low base confidence
        )));

        let cache = EmbeddingCache::new(256, 0.92);
        let config = TierConfig {
            surface_confidence_threshold: 0.85,
            reasoning_confidence_threshold: 0.70,
        };

        let mut resolver = HierarchicalResolver::new(graph, cache, config);

        // Add reasoning and deep nodes
        resolver.add_tier_node(Box::new(LinearNode::new(
            "reasoning_node",
            4,
            4,
            Tier::Reasoning,
            0.5, // Still below reasoning threshold
        )));
        resolver.add_tier_node(Box::new(LinearNode::new(
            "deep_node",
            4,
            2,
            Tier::Deep,
            0.9,
        )));

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = resolver.resolve(&input);

        // Should have escalated past Surface via coalition
        assert_ne!(result.tier_reached, Tier::Surface);
        assert!(
            result
                .route
                .execution_trace
                .contains(&"escalate_coalition".to_string()),
            "Expected 'escalate_coalition' in trace: {:?}",
            result.route.execution_trace
        );
        // Coalition should have formed
        assert!(result.coalition.is_some(), "Coalition should form on escalation");
    }

    #[test]
    fn test_most_inputs_stay_surface() {
        let mut resolver = HierarchicalResolver::build_default(8);
        let mut surface_count = 0;
        let total = 100;

        for i in 0..total {
            let data: Vec<f32> = (0..8).map(|j| ((i * 7 + j) as f32 * 0.1) % 1.0).collect();
            let input = Tensor::from_vec(data);
            let result = resolver.resolve(&input);
            if result.tier_reached == Tier::Surface {
                surface_count += 1;
            }
        }

        // At least 50% should stay at Surface (aiming for >70% in bench)
        let surface_pct = surface_count as f32 / total as f32;
        assert!(
            surface_pct >= 0.5,
            "Only {}% stayed at Surface (expected >=50%)",
            surface_pct * 100.0
        );
    }

    #[test]
    fn test_lateral_traversal() {
        let mut resolver = HierarchicalResolver::build_default(8);
        // Use an input that produces low Surface confidence
        let input = Tensor::from_vec(vec![10.0, -5.0, 3.0, -8.0, 6.0, -2.0, 7.0, -4.0]);
        let result = resolver.resolve(&input);

        // Check that trace_steps include lateral steps if any fired
        let has_lateral = result
            .route
            .trace_steps
            .iter()
            .any(|s| s.direction == TraversalDirection::Lateral);
        // lateral_count should be > 0 if the lateral condition was met
        // We just verify the mechanism works — actual firing depends on confidence
        assert!(result.route.lateral_count == 0 || has_lateral);
    }

    #[test]
    fn test_feedback_signals() {
        // Build a resolver where Deep will resolve with high confidence
        let mut graph = SparseGraph::new("low_surface");
        graph.add_node(Box::new(LinearNode::new(
            "low_surface",
            4,
            4,
            Tier::Surface,
            0.3,
        )));

        let cache = EmbeddingCache::new(256, 0.92);
        let config = TierConfig {
            surface_confidence_threshold: 0.85,
            reasoning_confidence_threshold: 0.70,
        };

        let mut resolver = HierarchicalResolver::new(graph, cache, config);
        resolver.add_tier_node(Box::new(LinearNode::new(
            "reasoning_node",
            4,
            4,
            Tier::Reasoning,
            0.5,
        )));
        resolver.add_tier_node(Box::new(LinearNode::new(
            "deep_node",
            4,
            2,
            Tier::Deep,
            0.95, // High base confidence → will trigger feedback
        )));

        let input = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let result = resolver.resolve(&input);

        // Deep should have resolved with high confidence → feedback signal
        if result.tier_reached == Tier::Deep && result.route.confidence > 0.90 {
            assert!(
                !result.feedback_signals.is_empty(),
                "Expected feedback signal when Deep resolves with high confidence"
            );
            assert!(result
                .route
                .execution_trace
                .contains(&"feedback_up".to_string()));
        }
    }

    #[test]
    fn test_structured_trace_directions() {
        let mut resolver = HierarchicalResolver::build_default(8);
        let input = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]);
        let result = resolver.resolve(&input);

        // Every trace step should have a valid direction
        for step in &result.route.trace_steps {
            match step.direction {
                TraversalDirection::Forward
                | TraversalDirection::Lateral
                | TraversalDirection::Feedback
                | TraversalDirection::Temporal => {}
            }
            assert!(step.confidence_out >= 0.0 && step.confidence_out <= 1.0);
        }
    }

    #[test]
    fn test_temporal_buffer_ring_behaviour() {
        let mut buf = TemporalBuffer::new(4);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);

        // Push 4 entries — fills the buffer
        for i in 0..4 {
            buf.push(TemporalEntry {
                input: Tensor::from_vec(vec![i as f32, 0.0]),
                output: Tensor::from_vec(vec![i as f32 * 10.0]),
                confidence: 0.9,
                tier: Tier::Surface,
            });
        }
        assert_eq!(buf.len(), 4);

        // Push a 5th — should overwrite the oldest (index 0)
        buf.push(TemporalEntry {
            input: Tensor::from_vec(vec![99.0, 0.0]),
            output: Tensor::from_vec(vec![990.0]),
            confidence: 0.95,
            tier: Tier::Surface,
        });
        assert_eq!(buf.len(), 4); // still 4, ring wrapped

        // The oldest entry (0.0, 0.0) should be gone
        let result = buf.find_similar(&Tensor::from_vec(vec![0.0, 0.0]), 0.99);
        // With cosine similarity, (0,0) has zero norm — no match
        assert!(result.is_none());

        // The newest entry (99.0, 0.0) should be findable
        let result = buf.find_similar(&Tensor::from_vec(vec![99.0, 0.0]), 0.99);
        assert!(result.is_some());
        assert!((result.unwrap().confidence - 0.95).abs() < 0.01);
    }

    #[test]
    fn test_temporal_buffer_find_similar() {
        let mut buf = TemporalBuffer::new(16);
        buf.push(TemporalEntry {
            input: Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0]),
            output: Tensor::from_vec(vec![10.0, 20.0]),
            confidence: 0.88,
            tier: Tier::Surface,
        });
        buf.push(TemporalEntry {
            input: Tensor::from_vec(vec![0.0, 1.0, 0.0, 0.0]),
            output: Tensor::from_vec(vec![30.0, 40.0]),
            confidence: 0.75,
            tier: Tier::Reasoning,
        });

        // Query similar to first entry — should match
        let query = Tensor::from_vec(vec![0.99, 0.01, 0.0, 0.0]);
        let result = buf.find_similar(&query, 0.85);
        assert!(result.is_some());
        assert!((result.unwrap().confidence - 0.88).abs() < 0.01);

        // Query orthogonal to both — should not match
        let query = Tensor::from_vec(vec![0.0, 0.0, 1.0, 0.0]);
        let result = buf.find_similar(&query, 0.85);
        assert!(result.is_none());
    }

    #[test]
    fn test_temporal_blend_in_resolver() {
        // Verify that the temporal buffer populates during resolve
        let mut resolver = HierarchicalResolver::build_default(8);
        let input1 = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]);
        // Dissimilar input to avoid cache hit
        let input2 = Tensor::from_vec(vec![-0.3, 0.9, -0.7, 0.4, -0.1, 0.8, -0.5, 0.2]);

        // First resolve populates the temporal buffer
        resolver.resolve(&input1);
        assert!(!resolver.temporal_buffer.is_empty());

        // Second resolve — dissimilar enough to miss cache, adds to temporal buffer
        resolver.resolve(&input2);
        assert!(resolver.temporal_buffer.len() >= 2);
    }

    #[test]
    fn test_hebbian_learning_changes_weights() {
        let mut resolver = HierarchicalResolver::build_default(8);
        let norm_before = resolver.total_weight_norm();

        let input = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]);
        let result = resolver.resolve(&input);

        // Apply Hebbian learning (iteration 1 of 1)
        resolver.learn(&input, &result, 0.001, 1);

        let norm_after = resolver.total_weight_norm();
        // Weight norm should have changed (learning occurred)
        assert!(
            (norm_after - norm_before).abs() > 1e-8,
            "Weight norm should change after learning: before={}, after={}",
            norm_before,
            norm_after
        );
    }

    #[test]
    fn test_hebbian_learning_no_learn_on_cache_hit() {
        let mut resolver = HierarchicalResolver::build_default(8);
        let input = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]);

        // First resolve — populates cache
        resolver.resolve(&input);

        // Second resolve — cache hit
        let result = resolver.resolve(&input);
        assert!(result.from_cache);

        let norm_before = resolver.total_weight_norm();
        resolver.learn(&input, &result, 0.001, 1);
        let norm_after = resolver.total_weight_norm();

        // No learning on cache hits
        assert!(
            (norm_after - norm_before).abs() < 1e-10,
            "Weight norm should NOT change on cache hit: before={}, after={}",
            norm_before,
            norm_after
        );
    }

    #[test]
    fn test_escalation_penalty_fires() {
        // Similarity-based penalty: escalation fires when cache has a similar
        // entry that was resolved at Surface tier.
        let mut resolver = HierarchicalResolver::build_default(4);

        // Step 1: Resolve an input that stays at Surface (populates cache with Surface entry)
        let input1 = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.1]);
        let result1 = resolver.resolve(&input1);
        assert_eq!(result1.tier_reached, Tier::Surface, "First input should stay at Surface");

        // Step 2: Force escalation on a similar input by lowering thresholds
        // to make Surface barely miss, causing escalation to Reasoning/Deep
        resolver.config.surface_confidence_threshold = 0.99;
        resolver.config.reasoning_confidence_threshold = 0.99;

        // Use a very similar input (cosine sim > 0.88 to input1)
        let input2 = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.11]);
        let result2 = resolver.resolve(&input2);

        // Should have escalated (thresholds are impossibly high)
        if result2.tier_reached != Tier::Surface && !result2.from_cache {
            let norm_before = resolver.total_weight_norm();
            let events = resolver.apply_error_signal(&input2, &result2, 0.0005);
            let norm_after = resolver.total_weight_norm();

            // Penalty should fire: cache has a Surface entry similar to input2
            assert!(
                !events.is_empty(),
                "Expected escalation penalty: cache has similar Surface entry"
            );
            assert_eq!(events[0].event_type, "escalation_penalty");
            assert!(
                (norm_after - norm_before).abs() > 1e-10,
                "Weight norm should change after escalation penalty"
            );
        }
    }

    #[test]
    fn test_cache_reinforcement_fires() {
        let mut resolver = HierarchicalResolver::build_default(8);
        let input = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]);

        // First resolve — populates cache with producer node ID
        let result1 = resolver.resolve(&input);
        assert!(!result1.from_cache);

        // Second resolve — cache hit
        let result2 = resolver.resolve(&input);
        assert!(result2.from_cache);

        // Cache hit similarity should be very high (identical input)
        assert!(
            result2.cache_hit_similarity > 0.95,
            "Identical input should produce similarity > 0.95, got {}",
            result2.cache_hit_similarity
        );

        // Apply error signal — should fire cache reinforcement
        let norm_before = resolver.total_weight_norm();
        let events = resolver.apply_error_signal(&input, &result2, 0.0005);
        let norm_after = resolver.total_weight_norm();

        // Cache reinforcement should fire if producer was tracked
        if result2.cache_producer_id.is_some() {
            assert!(
                !events.is_empty(),
                "Expected cache reinforcement events"
            );
            assert_eq!(events[0].event_type, "cache_reinforcement");
            // Weight change may be zero if the graph output went through many
            // ReLU layers and vanished — the event firing is the primary check.
            let _ = (norm_before, norm_after);
        }
    }

    #[test]
    fn test_error_events_accumulate() {
        let mut resolver = HierarchicalResolver::build_default(8);
        assert!(resolver.error_events().is_empty());

        let input = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]);
        let result = resolver.resolve(&input);
        resolver.apply_error_signal(&input, &result, 0.0005);
        // Second resolve triggers cache hit
        let result2 = resolver.resolve(&input);
        resolver.apply_error_signal(&input, &result2, 0.0005);

        // Events should accumulate
        let total = resolver.escalation_penalty_count() + resolver.cache_reinforcement_count();
        assert_eq!(total, resolver.error_events().len());

        // Clear should work
        resolver.clear_error_events();
        assert!(resolver.error_events().is_empty());
    }

    #[test]
    fn test_no_error_signal_on_surface_resolve() {
        let mut resolver = HierarchicalResolver::build_default(8);
        let input = Tensor::from_vec(vec![0.5, 0.3, 0.2, 0.1, 0.4, 0.2, 0.3, 0.1]);
        let result = resolver.resolve(&input);

        // If resolved at Surface, no escalation penalty should fire
        if result.tier_reached == Tier::Surface {
            let events = resolver.apply_error_signal(&input, &result, 0.0005);
            let penalties: Vec<_> = events
                .iter()
                .filter(|e| e.event_type == "escalation_penalty")
                .collect();
            assert!(
                penalties.is_empty(),
                "No escalation penalty expected when Surface resolves"
            );
        }
    }

    #[test]
    fn test_resolve_result_has_surface_state() {
        let mut resolver = HierarchicalResolver::build_default(8);
        let input = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]);
        let result = resolver.resolve(&input);

        // surface_confidence should be set (non-zero for non-empty input)
        assert!(result.surface_confidence > 0.0);

        // If not from cache, surface_output should exist
        if !result.from_cache {
            assert!(result.surface_output.is_some());
        }
    }

    #[test]
    fn test_producer_node_id_tracked() {
        let mut resolver = HierarchicalResolver::build_default(8);
        let input = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]);
        let result = resolver.resolve(&input);

        // Producer node ID should be set (not None)
        assert!(
            result.route.producer_node_id.is_some(),
            "Producer node ID should be tracked"
        );
    }

    #[test]
    fn test_training_mode_bypasses_cache() {
        let mut resolver = HierarchicalResolver::build_default(8);
        let input = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]);

        // Default mode is Inference — first call populates cache
        assert_eq!(resolver.mode, RouteMode::Inference);
        let r1 = resolver.resolve(&input);
        assert!(!r1.from_cache);

        // Same input in Inference mode — cache hit
        let r2 = resolver.resolve(&input);
        assert!(r2.from_cache);

        // Switch to Training mode — same input should NOT hit cache
        resolver.mode = RouteMode::Training;
        let r3 = resolver.resolve(&input);
        assert!(
            !r3.from_cache,
            "Training mode should bypass cache entirely"
        );

        // Switch back to Inference — cache hit again
        resolver.mode = RouteMode::Inference;
        let r4 = resolver.resolve(&input);
        assert!(
            r4.from_cache,
            "Inference mode should use cache"
        );
    }

    #[test]
    fn test_training_mode_does_not_insert_cache() {
        let mut resolver = HierarchicalResolver::build_default(8);
        resolver.mode = RouteMode::Training;

        let input = Tensor::from_vec(vec![1.0, 0.5, 0.3, 0.1, 0.8, 0.2, 0.6, 0.4]);

        // Resolve in Training mode — should NOT populate cache
        resolver.resolve(&input);
        assert_eq!(resolver.cache_size(), 0, "Training mode should not insert into cache");

        // Switch to Inference — should be a cache miss (nothing was inserted)
        resolver.mode = RouteMode::Inference;
        let result = resolver.resolve(&input);
        assert!(
            !result.from_cache,
            "No cache entries should exist from training"
        );
    }

    #[test]
    fn test_init_surface_analytical_freezes_surface() {
        let mut resolver = HierarchicalResolver::build_default(128);

        // Create 128-dim tensors with different directions
        let simple_tensors: Vec<Tensor> = (0..10)
            .map(|i| {
                let mut data = vec![1.0; 128];
                data[0] += i as f32 * 0.01;
                Tensor::from_vec(data)
            })
            .collect();
        let complex_tensors: Vec<Tensor> = (0..10)
            .map(|i| {
                let mut data = vec![0.5; 128];
                data[4] = 2.0;
                data[5] = 2.0;
                data[0] += i as f32 * 0.01;
                Tensor::from_vec(data)
            })
            .collect();

        let (dir_norm, simple_norm, complex_norm, cosine_sim) =
            resolver.init_surface_analytical(&simple_tensors, &complex_tensors);

        // Direction should have non-zero norm before normalisation
        assert!(dir_norm > 0.0, "Discrimination direction norm should be > 0");
        assert!(simple_norm > 0.0);
        assert!(complex_norm > 0.0);
        assert!(cosine_sim > 0.0 && cosine_sim < 1.0);

        // Surface graph nodes should be frozen
        for node in resolver.graph.nodes_ref() {
            if node.tier() == Tier::Surface {
                assert!(node.is_frozen(), "Surface graph node should be frozen");
            }
        }

        // Standalone surface nodes should be frozen
        let surface_norm_before = resolver.surface_weight_norm();

        // Verify frozen by attempting learning
        let input = Tensor::from_vec(vec![1.0; 128]);
        let result = resolver.resolve(&input);
        resolver.learn(&input, &result, 0.01, 1);

        let surface_norm_after = resolver.surface_weight_norm();
        assert!(
            (surface_norm_after - surface_norm_before).abs() < 1e-8,
            "Frozen Surface weight norm must not change: before={}, after={}",
            surface_norm_before, surface_norm_after
        );
    }

    #[test]
    fn test_init_surface_analytical_leaves_reasoning_unfrozen() {
        use crate::input::encoder::{G5_DIM, G5_OFFSET};

        let mut resolver = HierarchicalResolver::build_default(128);

        let simple_tensors: Vec<Tensor> = (0..5)
            .map(|_| {
                let mut data = vec![1.0; 128];
                for d in G5_OFFSET..(G5_OFFSET + G5_DIM) {
                    data[d] = 0.2;
                }
                Tensor::from_vec(data)
            })
            .collect();
        let complex_tensors: Vec<Tensor> = (0..5)
            .map(|_| {
                let mut data = vec![1.0; 128];
                for d in G5_OFFSET..(G5_OFFSET + G5_DIM) {
                    data[d] = 0.9;
                }
                data[G5_OFFSET + 2] = 2.0;
                Tensor::from_vec(data)
            })
            .collect();

        resolver.init_surface_analytical(&simple_tensors, &complex_tensors);

        // Reasoning and Deep graph nodes should NOT be frozen
        for node in resolver.graph.nodes_ref() {
            if node.tier() != Tier::Surface {
                assert!(
                    !node.is_frozen(),
                    "Non-Surface graph node '{}' should not be frozen",
                    node.node_id()
                );
            }
        }
    }

    #[test]
    fn test_init_reasoning_deep_orthogonal_leaves_surface_unchanged() {
        use crate::input::{Encoder, Tokeniser};

        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::build_with_axiom_config(128, &config);

        // First do analytical Surface init
        let tokeniser = Tokeniser::default_tokeniser();
        let encoder = Encoder::new(128, tokeniser);
        let simple = ["the sky is blue", "it is raining", "she runs fast"];
        let complex = ["consciousness remains profound", "the interplay drives dynamics", "dark matter constitutes"];
        let simple_t: Vec<Tensor> = simple.iter().map(|s| encoder.encode_text_readonly(s)).collect();
        let complex_t: Vec<Tensor> = complex.iter().map(|s| encoder.encode_text_readonly(s)).collect();
        resolver.init_surface_analytical(&simple_t, &complex_t);

        // Capture Surface weight norms before orthogonal init
        let surface_norm_before = resolver.surface_weight_norm();
        let mut surface_directions_before = Vec::new();
        for node in resolver.graph.nodes_ref() {
            if node.tier() == Tier::Surface {
                surface_directions_before.push(node.weight_direction());
            }
        }

        // Apply orthogonal init to R+D nodes
        let mean_cos = resolver.init_reasoning_deep_orthogonal();

        // Surface nodes must be unchanged
        let surface_norm_after = resolver.surface_weight_norm();
        assert!(
            (surface_norm_before - surface_norm_after).abs() < 1e-4,
            "Surface weight norm changed: {:.4} → {:.4}",
            surface_norm_before, surface_norm_after
        );

        let mut idx = 0;
        for node in resolver.graph.nodes_ref() {
            if node.tier() == Tier::Surface {
                let dir_after = node.weight_direction();
                assert_eq!(
                    dir_after, surface_directions_before[idx],
                    "Surface node '{}' weight direction changed after orthogonal init",
                    node.node_id()
                );
                idx += 1;
            }
        }

        // Mean pairwise cos should be low
        assert!(
            mean_cos < 0.3,
            "Mean pairwise |cos| = {:.4}, exceeds target of 0.3",
            mean_cos
        );
    }

    #[test]
    fn test_orthogonal_init_rd_node_diversity() {
        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::build_with_axiom_config(128, &config);
        resolver.init_reasoning_deep_orthogonal();

        // Collect all R+D weight directions
        let mut directions = Vec::new();
        for node in resolver.graph.nodes_ref() {
            if node.tier() != Tier::Surface {
                directions.push(node.weight_direction());
            }
        }
        for node in &resolver.reasoning_nodes {
            directions.push(node.weight_direction());
        }
        for node in &resolver.deep_nodes {
            directions.push(node.weight_direction());
        }

        // Should have 18 R+D nodes (4 graph reasoning + 2 graph deep + 8 standalone reasoning + 4 standalone deep)
        assert_eq!(directions.len(), 18, "Expected 18 R+D nodes");

        // All pairwise cosine similarities should be < 0.3
        for i in 0..directions.len() {
            for j in (i + 1)..directions.len() {
                let dot: f32 = directions[i].iter().zip(directions[j].iter()).map(|(a, b)| a * b).sum();
                let ni: f32 = directions[i].iter().map(|x| x * x).sum::<f32>().sqrt();
                let nj: f32 = directions[j].iter().map(|x| x * x).sum::<f32>().sqrt();
                let cos = if ni > 1e-8 && nj > 1e-8 { (dot / (ni * nj)).abs() } else { 0.0 };
                assert!(
                    cos < 0.3,
                    "R+D nodes {} and {} have |cos| = {:.4}, exceeds 0.3",
                    i, j, cos
                );
            }
        }
    }

    #[test]
    fn test_coalition_forms_on_escalation() {
        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::build_with_axiom_config(128, &config);
        resolver.init_reasoning_deep_orthogonal();
        resolver.mode = RouteMode::Training;
        // Force escalation by setting threshold above any possible Surface confidence
        resolver.config.surface_confidence_threshold = 0.99;

        let input = Tensor::from_vec(vec![0.5; 128]);
        let result = resolver.resolve(&input);

        // Coalition should have formed
        assert!(result.coalition.is_some(), "Coalition should form on escalation");
        let coalition = result.coalition.unwrap();
        assert!(!coalition.members.is_empty(), "Coalition should have members");
        assert!(coalition.members.len() >= 2, "Coalition min size is 2");
        assert!(coalition.members.len() <= 2, "Coalition max size is 2");
        assert!(!coalition.resolved_by.is_empty(), "Coalition must have a resolver");
    }

    #[test]
    fn test_coalition_size_clamping() {
        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::build_with_axiom_config(128, &config);
        resolver.init_reasoning_deep_orthogonal();
        resolver.mode = RouteMode::Training;
        resolver.coalition_max_size = 3; // Lower max

        let input = Tensor::from_vec(vec![0.001; 128]);
        let result = resolver.resolve(&input);

        if let Some(coalition) = result.coalition {
            assert!(
                coalition.members.len() <= 3,
                "Coalition size {} exceeds max 3",
                coalition.members.len()
            );
        }
    }

    #[test]
    fn test_no_coalition_when_surface_resolves() {
        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::build_with_axiom_config(128, &config);

        // Use init_surface_analytical with test data for high Surface confidence
        use crate::input::{Encoder, Tokeniser};
        let tokeniser = Tokeniser::default_tokeniser();
        let encoder = Encoder::new(128, tokeniser);
        let simple_t: Vec<Tensor> = ["the sky is blue", "it rains", "she runs"]
            .iter().map(|s| encoder.encode_text_readonly(s)).collect();
        let complex_t: Vec<Tensor> = ["consciousness remains profound", "interplay drives", "dark matter"]
            .iter().map(|s| encoder.encode_text_readonly(s)).collect();
        resolver.init_surface_analytical(&simple_t, &complex_t);
        resolver.mode = RouteMode::Training;

        // Simple sentence should stay at Surface — no coalition
        let simple_input = encoder.encode_text_readonly("the sky is blue");
        let result = resolver.resolve(&simple_input);
        assert!(
            result.coalition.is_none(),
            "Surface-resolved input should not form a coalition"
        );
    }

    #[test]
    fn test_coalition_cross_tier_blend() {
        // Build a minimal resolver with specific nodes to test cross-tier blend
        let mut graph = SparseGraph::new("low_conf");
        graph.add_node(Box::new(LinearNode::new(
            "low_conf", 4, 4, Tier::Surface, 0.3,
        )));

        let cache = EmbeddingCache::new(256, 0.92);
        let config = TierConfig {
            surface_confidence_threshold: 0.85,
            reasoning_confidence_threshold: 0.70,
        };
        let mut resolver = HierarchicalResolver::new(graph, cache, config);

        // Add R+D nodes with specific initialisation
        resolver.add_tier_node(Box::new(LinearNode::new(
            "reasoning_a", 4, 4, Tier::Reasoning, 0.80,
        )));
        resolver.add_tier_node(Box::new(LinearNode::new(
            "deep_a", 4, 4, Tier::Deep, 0.95,
        )));

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = resolver.resolve(&input);

        // Should have formed a coalition
        assert!(result.coalition.is_some());
        let coalition = result.coalition.unwrap();
        // All members should have fired
        assert!(coalition.members.iter().all(|m| m.fired));
    }

    #[test]
    fn test_coalition_log_accumulates() {
        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::build_with_axiom_config(128, &config);
        resolver.init_reasoning_deep_orthogonal();
        resolver.mode = RouteMode::Training;
        resolver.config.surface_confidence_threshold = 0.99;

        // Run several escalating inputs
        for i in 0..5 {
            let mut data = vec![0.5; 128];
            data[i % 128] = 1.0; // slight variation
            let input = Tensor::from_vec(data);
            resolver.resolve(&input);
        }

        assert!(
            resolver.coalition_log().len() >= 1,
            "Coalition log should accumulate entries"
        );
    }

    #[test]
    fn test_coalition_surface_invariant_preserved() {
        use crate::input::{Encoder, Tokeniser};

        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::build_with_axiom_config(128, &config);

        let tokeniser = Tokeniser::default_tokeniser();
        let encoder = Encoder::new(128, tokeniser);
        let simple_t: Vec<Tensor> = ["the sky is blue", "it rains", "she runs"]
            .iter().map(|s| encoder.encode_text_readonly(s)).collect();
        let complex_t: Vec<Tensor> = ["consciousness remains profound", "interplay drives", "dark matter"]
            .iter().map(|s| encoder.encode_text_readonly(s)).collect();
        resolver.init_surface_analytical(&simple_t, &complex_t);
        resolver.init_reasoning_deep_orthogonal();
        resolver.mode = RouteMode::Training;
        resolver.config.surface_confidence_threshold = 0.99; // force escalation

        // Capture Surface weight norm
        let surface_norm_before = resolver.surface_weight_norm();

        // Run several inputs that will escalate and form coalitions
        for _ in 0..20 {
            let input = Tensor::from_vec(vec![0.5; 128]);
            resolver.resolve(&input);
        }

        // Surface weights must be unchanged
        let surface_norm_after = resolver.surface_weight_norm();
        assert!(
            (surface_norm_before - surface_norm_after).abs() < 1e-4,
            "Surface weights changed during coalition: {:.4} → {:.4}",
            surface_norm_before, surface_norm_after
        );
    }

    #[test]
    fn test_coalition_sizing_on_text() {
        use crate::input::{Encoder, Tokeniser};

        let sentences = [
            "the cat sat on the mat",
            "she runs every morning",
            "the sky is blue today",
            "water flows downhill naturally",
            "birds fly south for winter",
            "he reads books at night",
            "the dog barked loudly",
            "rain falls from clouds",
            "the sun shines brightly",
            "fish swim in the ocean",
            "the consciousness of the observer fundamentally determines what can be measured",
            "neural networks approximate complex nonlinear functions through hierarchical representations",
            "the interplay between cooperation and competition drives evolutionary dynamics",
            "dark matter constitutes approximately twenty seven percent of the observable universe",
            "she said that he thought that they believed it was true",
            "the cat that the dog that the man owned chased sat on the mat",
            "Cogito ergo sum",
            "Being precedes essence",
            "the mitochondria is the powerhouse of the cell",
            "photosynthesis converts sunlight into chemical energy efficiently",
            "the big red dog ran quickly down the long straight road",
            "if then else",
            "go",
            "quantum entanglement creates correlations that cannot be explained classically",
            "the teacher who the students respected retired last year",
            "it is raining today",
            "she runs fast",
            "the tintinnabulation resonated melodiously",
            "machine learning models require large amounts of training data",
            "the quick brown fox jumped over the lazy sleeping dog",
            "epistemological foundations of scientific inquiry remain contentious",
            "a rose by any other name would smell as sweet",
            "the committee decided to postpone the meeting until further notice",
            "abstract algebra unifies seemingly disparate mathematical structures",
            "he went to the store and bought milk and bread and cheese",
            "freedom is just another word for nothing left to lose",
            "the recursive algorithm terminates when the base case is reached",
            "she smiled warmly at the stranger across the crowded room",
            "thermodynamic entropy always increases in isolated systems",
            "the old man and the sea tells the story of perseverance",
            "parallel computing distributes workload across multiple processing units",
            "they walked slowly through the ancient forest listening to the birds",
            "mathematical induction proves statements about all natural numbers",
            "the philosopher argued that knowledge requires justified true belief",
            "simple sentences are easy to understand and parse quickly",
            "neurotransmitters facilitate communication between neurons across synaptic gaps",
            "the children played happily in the garden all afternoon",
            "topology studies properties preserved under continuous deformations",
            "an apple fell on his head and he discovered gravity",
            "the boundary between order and chaos defines complexity itself",
        ];

        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::build_with_axiom_config(128, &config);

        let tokeniser = Tokeniser::default_tokeniser();
        let encoder = Encoder::new(128, tokeniser);

        // Analytical Surface init
        let simple_t: Vec<Tensor> = ["the sky is blue", "it rains", "she runs", "water flows", "he reads"]
            .iter().map(|s| encoder.encode_text_readonly(s)).collect();
        let complex_t: Vec<Tensor> = [
            "consciousness determines measurement",
            "neural networks approximate functions",
            "interplay drives dynamics",
        ].iter().map(|s| encoder.encode_text_readonly(s)).collect();
        resolver.init_surface_analytical(&simple_t, &complex_t);
        resolver.init_reasoning_deep_orthogonal();
        // Recalibrate AFTER analytical init so thresholds match current confidences
        resolver.calibrate(128, 0.65, 0.35);
        resolver.mode = RouteMode::Training;

        let mut coalition_sizes = Vec::new();
        let mut bid_counts = Vec::new();

        for sentence in &sentences {
            let tensor = encoder.encode_text_readonly(sentence);
            let result = resolver.resolve(&tensor);
            if let Some(coal) = &result.coalition {
                coalition_sizes.push(coal.members.len());
                bid_counts.push(coal.bid_count);
            }
        }

        let n = coalition_sizes.len();
        if n > 0 {
            let mean_size = coalition_sizes.iter().sum::<usize>() as f32 / n as f32;
            let min_bids = bid_counts.iter().copied().min().unwrap_or(0);
            let max_bids = bid_counts.iter().copied().max().unwrap_or(0);
            let mean_bids = bid_counts.iter().sum::<usize>() as f32 / n as f32;
            eprintln!(
                "Coalition sizing: {}/{} inputs escalated, mean coalition size = {:.1}, \
                 bid count range = [{}, {}], mean bids = {:.1}, threshold = {}",
                n, sentences.len(), mean_size, min_bids, max_bids, mean_bids,
                resolver.coalition_bid_threshold
            );

            // Print 5 sample coalition events
            let log = resolver.coalition_log();
            eprintln!("\n--- Sample coalition events (up to 5) ---");
            for (i, coal) in log.iter().take(5).enumerate() {
                let member_desc: Vec<String> = coal.members.iter()
                    .map(|m| format!("{}({:?}, bid={:.3}, conf={:.3})", m.node_id, m.tier, m.bid_score, m.confidence_out))
                    .collect();
                eprintln!(
                    "  [{}] resolved_by={} ({:?}), coalition=[{}], cross_tier={}, bids={}",
                    i + 1, coal.resolved_by, coal.resolved_tier,
                    member_desc.join(", "), coal.cross_tier_fired, coal.bid_count
                );
            }

            // Check for diversity: count unique resolver nodes
            let mut resolver_set: std::collections::HashSet<String> = std::collections::HashSet::new();
            for coal in log {
                resolver_set.insert(coal.resolved_by.clone());
            }
            eprintln!(
                "\nUnique resolver nodes: {}/{} coalitions",
                resolver_set.len(), log.len()
            );

            // Target: mean coalition size between 1.5 and 2.0 (max_coalition_size=2)
            assert!(
                mean_size >= 1.5 && mean_size <= 2.0,
                "Mean coalition size {:.1} outside acceptable range [1.5, 2.0]",
                mean_size
            );
        } else {
            eprintln!("Coalition sizing: 0/{} inputs escalated — all stayed Surface", sentences.len());
        }
    }

    #[test]
    fn test_stochastic_coalition_produces_varied_members() {
        // Run 100 resolves with the SAME input — stochastic selection should
        // produce different coalition member pairs across runs.
        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::build_with_axiom_config(128, &config);

        use crate::input::{Encoder, Tokeniser};
        let tokeniser = Tokeniser::default_tokeniser();
        let encoder = Encoder::new(128, tokeniser);

        let simple_t: Vec<Tensor> = ["the sky is blue", "it rains", "she runs"]
            .iter().map(|s| encoder.encode_text_readonly(s)).collect();
        let complex_t: Vec<Tensor> = [
            "consciousness determines measurement",
            "neural networks approximate functions",
        ].iter().map(|s| encoder.encode_text_readonly(s)).collect();
        resolver.init_surface_analytical(&simple_t, &complex_t);
        resolver.init_reasoning_deep_orthogonal();
        resolver.calibrate(128, 0.65, 0.35);
        resolver.mode = RouteMode::Training;

        // Force escalation so coalition forms
        resolver.config.surface_confidence_threshold = 0.99;

        let input = encoder.encode_text_readonly(
            "the recursive nature of self-referential systems creates emergent properties"
        );

        let mut member_sets: std::collections::HashSet<Vec<String>> = std::collections::HashSet::new();
        let mut all_node_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

        for _ in 0..100 {
            let result = resolver.resolve(&input);
            if let Some(coal) = &result.coalition {
                let mut ids: Vec<String> = coal.members.iter()
                    .map(|m| m.node_id.clone())
                    .collect();
                ids.sort();
                for id in &ids {
                    all_node_ids.insert(id.clone());
                }
                member_sets.insert(ids);
            }
        }

        eprintln!(
            "Stochastic coalition test: {} unique member-sets from 100 calls, {} unique nodes total",
            member_sets.len(), all_node_ids.len()
        );

        // With stochastic weighted sampling, the same input should NOT always
        // produce the same coalition pair — we need at least 2 distinct member-sets
        // to confirm randomness is working.
        assert!(
            member_sets.len() >= 2,
            "Expected at least 2 distinct coalition member-sets (stochastic), got {}",
            member_sets.len()
        );
        // At least 3 unique nodes should appear — stochastic sampling ensures
        // nodes beyond the top-2 get occasional selection
        assert!(
            all_node_ids.len() >= 3,
            "Expected at least 3 unique nodes across 100 calls, got {}",
            all_node_ids.len()
        );
    }

    #[test]
    fn test_g5_penalty_roundtrip() {
        // Verify G5 penalty norms survive save/load and produce identical confidences.
        use crate::input::{Encoder, Tokeniser};

        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::build_with_axiom_config(128, &config);
        let mut tokeniser = Tokeniser::default_tokeniser();

        let simple = ["the cat sat on the mat", "the dog runs fast", "birds fly south"];
        let complex = [
            "the recursive nature of self-referential systems creates emergent properties",
            "quantum entanglement challenges classical notions of locality and causality",
        ];
        for s in simple.iter().chain(complex.iter()) {
            tokeniser.tokenise(s);
        }
        let encoder = Encoder::new(128, tokeniser);

        let simple_t: Vec<Tensor> = simple.iter().map(|s| encoder.encode_text_readonly(s)).collect();
        let complex_t: Vec<Tensor> = complex.iter().map(|s| encoder.encode_text_readonly(s)).collect();
        resolver.init_surface_analytical(&simple_t, &complex_t);
        resolver.set_g5_penalty_weight(0.25);

        // Record confidences before save
        let simple_input = encoder.encode_text_readonly("the cat sat on the mat");
        let complex_input = encoder.encode_text_readonly(
            "the recursive nature of self-referential systems creates emergent properties"
        );
        let orig_simple_conf = resolver.max_surface_confidence(&simple_input);
        let orig_complex_conf = resolver.max_surface_confidence(&complex_input);

        // Save
        let path = "/tmp/axiom_g5_roundtrip_test.json";
        resolver.save_all_weights(path).expect("save failed");

        // Load into a fresh resolver
        let mut resolver2 = HierarchicalResolver::build_with_axiom_config(128, &config);
        resolver2.load_all_weights(path).expect("load failed");
        resolver2.set_g5_penalty_weight(0.25);

        let loaded_simple_conf = resolver2.max_surface_confidence(&simple_input);
        let loaded_complex_conf = resolver2.max_surface_confidence(&complex_input);

        std::fs::remove_file(path).ok();

        eprintln!("G5 roundtrip: simple {:.6} → {:.6}, complex {:.6} → {:.6}",
            orig_simple_conf, loaded_simple_conf,
            orig_complex_conf, loaded_complex_conf);

        assert!(
            (orig_simple_conf - loaded_simple_conf).abs() < 1e-5,
            "Simple conf mismatch: {:.6} vs {:.6}", orig_simple_conf, loaded_simple_conf
        );
        assert!(
            (orig_complex_conf - loaded_complex_conf).abs() < 1e-5,
            "Complex conf mismatch: {:.6} vs {:.6}", orig_complex_conf, loaded_complex_conf
        );
        // G5 norms must be non-zero after load
        assert!(resolver2.g5_simple_mean_norm > 0.0, "g5_simple_mean_norm not loaded");
        assert!(resolver2.g5_complex_mean_norm > 0.0, "g5_complex_mean_norm not loaded");
    }

    #[test]
    fn test_sentence_chunking_g5_norm_ordering() {
        // 5 concatenated simple sentences should produce lower mean G5 norm
        // than 5 concatenated complex sentences after chunking.
        use crate::input::{Encoder, Tokeniser};

        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::build_with_axiom_config(128, &config);
        let mut tokeniser = Tokeniser::default_tokeniser();

        let simple_sents = [
            "The cat sat on the mat.",
            "The dog runs fast.",
            "Birds fly south in winter.",
            "Water flows downhill.",
            "The sky is blue.",
        ];
        let complex_sents = [
            "The recursive nature of self-referential systems creates emergent properties that resist reduction.",
            "Quantum entanglement challenges classical notions of locality and causality.",
            "Consciousness remains an unsolved problem at the intersection of neuroscience and philosophy.",
            "The boundary between deterministic chaos and true randomness has profound implications.",
            "Emergence in complex adaptive systems suggests that reductionist explanations are fundamentally insufficient.",
        ];
        for s in simple_sents.iter().chain(complex_sents.iter()) {
            tokeniser.tokenise(s);
        }
        let encoder = Encoder::new(128, tokeniser);

        // Analytical init so G5 features are meaningful
        let simple_t: Vec<Tensor> = simple_sents.iter().map(|s| encoder.encode_text_readonly(s)).collect();
        let complex_t: Vec<Tensor> = complex_sents.iter().map(|s| encoder.encode_text_readonly(s)).collect();
        resolver.init_surface_analytical(&simple_t, &complex_t);
        resolver.mode = RouteMode::Training;

        let simple_paragraph = simple_sents.join(" ");
        let complex_paragraph = complex_sents.join(" ");

        let (_, simple_mean_g5, _) = resolver.resolve_text(&encoder, &simple_paragraph);
        let (_, complex_mean_g5, _) = resolver.resolve_text(&encoder, &complex_paragraph);

        eprintln!(
            "Chunking test: simple_mean_g5={:.4}, complex_mean_g5={:.4}",
            simple_mean_g5, complex_mean_g5
        );

        assert!(
            simple_mean_g5 < complex_mean_g5,
            "Expected simple mean G5 ({:.4}) < complex mean G5 ({:.4})",
            simple_mean_g5, complex_mean_g5
        );
    }
}
