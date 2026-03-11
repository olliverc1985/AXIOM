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

/// Orchestrates the full AXIOM pipeline: cache lookup, sparse graph routing,
/// lateral traversal, hierarchical tier escalation, and feedback signals.
///
/// Most inputs should resolve at Surface tier without escalation.
/// Lateral connections between Surface nodes reduce unnecessary escalation.
/// Feedback signals from Deep flow upward to adjust Reasoning confidence.
pub struct HierarchicalResolver {
    /// The sparse computation graph.
    pub graph: SparseGraph,
    /// The embedding cache for content-addressable tensor lookup.
    pub cache: EmbeddingCache,
    /// Tier escalation configuration.
    pub config: TierConfig,
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
    /// Accumulated feedback signals from the current resolve call.
    feedback_log: Vec<FeedbackSignal>,
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
    pub winning_path: String,
}

impl HierarchicalResolver {
    /// Create a new resolver with a pre-built graph and default config.
    pub fn new(graph: SparseGraph, cache: EmbeddingCache, config: TierConfig) -> Self {
        Self {
            graph,
            cache,
            config,
            surface_nodes: Vec::new(),
            reasoning_nodes: Vec::new(),
            deep_nodes: Vec::new(),
            lateral_nodes: Vec::new(),
            lateral_edges: Vec::new(),
            feedback_log: Vec::new(),
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

    /// Run a set of tier-specific nodes on the input, returning best output.
    fn run_tier_nodes(
        nodes: &[Box<dyn ComputeNode>],
        input: &Tensor,
    ) -> Option<NodeOutput> {
        let mut best: Option<NodeOutput> = None;
        for node in nodes {
            let output = node.forward(input);
            if best.as_ref().map_or(true, |b| output.confidence > b.confidence) {
                best = Some(output);
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
    /// 8. Cache the result
    #[allow(unused_assignments)]
    pub fn resolve(&mut self, input: &Tensor) -> ResolveResult {
        let mut cache_hits = 0u32;
        let mut from_cache = false;
        self.feedback_log.clear();

        // Phase 1: Check cache
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

        // Try cache first
        if let Some((cached, _sim)) = self.cache_lookup(&cache_key) {
            current_tensor = cached;
            cache_hits += 1;
            from_cache = true;
            winning_path = "cache".to_string();
            confidence = 0.95; // High confidence for cached results
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

            total_cost += route.total_compute_cost;
            trace.extend(route.execution_trace);
            trace_steps.extend(route.trace_steps);

            // Also run standalone surface nodes
            if let Some(surface_out) = Self::run_tier_nodes(&self.surface_nodes, input) {
                total_cost += surface_out.compute_cost;
                // Blend with graph output if surface nodes provide higher confidence
                if surface_out.confidence > confidence {
                    current_tensor = surface_out.tensor;
                    confidence = surface_out.confidence;
                    winning_path = "surface_standalone".to_string();
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

            // Phase 3: Lateral traversal — try neighbouring Surface nodes before escalating
            if confidence < self.config.surface_confidence_threshold {
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
                            // Lateral resolved it — no escalation needed
                            lateral_prevented += 1;
                        }
                        current_tensor = lat_out.tensor;
                        confidence = lat_out.confidence;
                        total_cost += lat_out.compute_cost;
                    }
                }
            }
        }

        // Phase 4: Escalate to Reasoning if needed
        // Standalone nodes receive the ORIGINAL input, not graph output.
        // The graph is one path; standalone nodes are alternative paths.
        if confidence < self.config.surface_confidence_threshold && !from_cache {
            tier_reached = Tier::Reasoning;
            trace.push("escalate_reasoning".to_string());
            trace_steps.push(TraceStep {
                node_id: "escalate_reasoning".to_string(),
                tier: Tier::Reasoning,
                direction: TraversalDirection::Forward,
                confidence_in: confidence,
                confidence_out: confidence,
                was_cached: false,
            });

            if let Some(reasoning_out) =
                Self::run_tier_nodes(&self.reasoning_nodes, input)
            {
                let conf_in = confidence;
                current_tensor = reasoning_out.tensor;
                winning_path = "reasoning_standalone".to_string();
                confidence = reasoning_out.confidence;
                total_cost += reasoning_out.compute_cost;
                trace_steps.push(TraceStep {
                    node_id: "reasoning_standalone".to_string(),
                    tier: Tier::Reasoning,
                    direction: TraversalDirection::Forward,
                    confidence_in: conf_in,
                    confidence_out: confidence,
                    was_cached: false,
                });
            }
        }

        // Phase 5: Escalate to Deep if still not confident
        // Deep standalone also receives the original input.
        if confidence < self.config.reasoning_confidence_threshold && !from_cache {
            tier_reached = Tier::Deep;
            trace.push("escalate_deep".to_string());
            trace_steps.push(TraceStep {
                node_id: "escalate_deep".to_string(),
                tier: Tier::Deep,
                direction: TraversalDirection::Forward,
                confidence_in: confidence,
                confidence_out: confidence,
                was_cached: false,
            });

            if let Some(deep_out) = Self::run_tier_nodes(&self.deep_nodes, input) {
                let conf_in = confidence;
                current_tensor = deep_out.tensor;
                winning_path = "deep_standalone".to_string();
                confidence = deep_out.confidence;
                total_cost += deep_out.compute_cost;
                trace_steps.push(TraceStep {
                    node_id: "deep_standalone".to_string(),
                    tier: Tier::Deep,
                    direction: TraversalDirection::Forward,
                    confidence_in: conf_in,
                    confidence_out: confidence,
                    was_cached: false,
                });

                // Phase 6: Feedback — if Deep resolved with high confidence,
                // emit a signal to Reasoning to lower its threshold
                if confidence > 0.80 {
                    let delta = -0.01; // Nudge Reasoning base_confidence down
                    let signal =
                        FeedbackSignal::low_confidence_resolved("deep_standalone", delta);
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
        }

        // Phase 7: Cache the result (if not from cache)
        if !from_cache {
            self.cache
                .insert_direct(cache_key, current_tensor.clone());
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
            },
            tier_reached,
            from_cache,
            feedback_signals,
            winning_path,
        }
    }

    /// Direct cache lookup (bypasses get_or_compute).
    fn cache_lookup(&mut self, key: &Tensor) -> Option<(Tensor, f32)> {
        self.cache.total_lookups += 1;
        // Manual lookup without the compute closure
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

    /// Build a default AXIOM resolver with a reasonable graph topology.
    ///
    /// Creates a graph with Surface, Reasoning, and Deep tier nodes
    /// connected with confidence-gated edges and lateral Surface connections.
    pub fn build_default(input_dim: usize) -> Self {
        use crate::graph::edge::ConditionalEdge;

        let mid_dim = input_dim;
        let out_dim = input_dim / 2;

        // Build graph
        let mut graph = SparseGraph::new("surface_entry");

        // Surface tier nodes in the graph
        graph.add_node(Box::new(LinearNode::new(
            "surface_entry",
            input_dim,
            mid_dim,
            Tier::Surface,
            0.88,
        )));
        graph.add_node(Box::new(LinearNode::new(
            "surface_refine",
            mid_dim,
            mid_dim,
            Tier::Surface,
            0.90,
        )));

        // Reasoning tier node in the graph
        graph.add_node(Box::new(LinearNode::new(
            "reasoning_analyze",
            mid_dim,
            mid_dim,
            Tier::Reasoning,
            0.80,
        )));

        // Deep tier node in the graph
        graph.add_node(Box::new(LinearNode::new(
            "deep_resolve",
            mid_dim,
            out_dim,
            Tier::Deep,
            0.75,
        )));

        // Edges: surface_entry -> surface_refine (always)
        graph.add_edge(ConditionalEdge::always("surface_entry", "surface_refine"));

        // surface_refine -> reasoning_analyze (if confidence below surface threshold)
        graph.add_edge(ConditionalEdge::if_confidence_below(
            "surface_refine",
            "reasoning_analyze",
            0.85,
        ));

        // reasoning_analyze -> deep_resolve (if confidence below reasoning threshold)
        graph.add_edge(ConditionalEdge::if_confidence_below(
            "reasoning_analyze",
            "deep_resolve",
            0.70,
        ));

        let cache = EmbeddingCache::new(256, 0.92);
        let config = TierConfig::default();

        let mut resolver = Self::new(graph, cache, config);

        // Standalone Surface nodes — compete with graph during surface_blend
        resolver.add_tier_node(Box::new(LinearNode::new(
            "surface_standalone_a",
            input_dim,
            mid_dim,
            Tier::Surface,
            0.91,
        )));
        resolver.add_tier_node(Box::new(LinearNode::new(
            "surface_standalone_b",
            input_dim,
            mid_dim,
            Tier::Surface,
            0.90,
        )));

        // Add standalone tier nodes for blending
        resolver.add_tier_node(Box::new(LinearNode::new(
            "reasoning_standalone",
            input_dim,
            mid_dim,
            Tier::Reasoning,
            0.72,
        )));
        resolver.add_tier_node(Box::new(LinearNode::new(
            "deep_standalone",
            mid_dim,
            out_dim,
            Tier::Deep,
            0.78,
        )));

        // Lateral Surface nodes — only fire during lateral traversal
        resolver.add_lateral_node(Box::new(LinearNode::new(
            "surface_lateral_a",
            input_dim,
            mid_dim,
            Tier::Surface,
            0.90,
        )));
        resolver.add_lateral_node(Box::new(LinearNode::new(
            "surface_lateral_b",
            input_dim,
            mid_dim,
            Tier::Surface,
            0.89,
        )));

        // Lateral edges: surface_entry → surface lateral nodes
        resolver.add_lateral_edge(LateralEdge::if_confidence_below(
            "surface_entry",
            "surface_lateral_a",
            0.75,
            1.0,
        ));
        resolver.add_lateral_edge(LateralEdge::if_confidence_below(
            "surface_entry",
            "surface_lateral_b",
            0.75,
            1.0,
        ));

        // Calibrate: measure actual node confidence distribution, set thresholds
        // at 65th percentile (surface) and 35th percentile (reasoning).
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
        use crate::graph::edge::ConditionalEdge;

        let mid_dim = input_dim;
        let out_dim = input_dim / 2;

        let mut graph = SparseGraph::new("surface_entry");

        graph.add_node(Box::new(LinearNode::new(
            "surface_entry",
            input_dim,
            mid_dim,
            Tier::Surface,
            0.88,
        )));
        graph.add_node(Box::new(LinearNode::new(
            "surface_refine",
            mid_dim,
            mid_dim,
            Tier::Surface,
            0.90,
        )));
        graph.add_node(Box::new(LinearNode::new(
            "reasoning_analyze",
            mid_dim,
            mid_dim,
            Tier::Reasoning,
            0.80,
        )));
        graph.add_node(Box::new(LinearNode::new(
            "deep_resolve",
            mid_dim,
            out_dim,
            Tier::Deep,
            0.75,
        )));

        graph.add_edge(ConditionalEdge::always("surface_entry", "surface_refine"));
        graph.add_edge(ConditionalEdge::if_confidence_below(
            "surface_refine",
            "reasoning_analyze",
            config.surface_confidence_threshold,
        ));
        graph.add_edge(ConditionalEdge::if_confidence_below(
            "reasoning_analyze",
            "deep_resolve",
            config.reasoning_confidence_threshold,
        ));

        let cache = EmbeddingCache::new(256, config.cache_similarity_threshold);
        let tier_config = TierConfig {
            surface_confidence_threshold: config.surface_confidence_threshold,
            reasoning_confidence_threshold: config.reasoning_confidence_threshold,
        };

        let mut resolver = Self::new(graph, cache, tier_config);

        // Surface lateral nodes
        resolver.add_lateral_node(Box::new(LinearNode::new(
            "surface_lateral_a",
            input_dim,
            mid_dim,
            Tier::Surface,
            0.90,
        )));
        resolver.add_lateral_node(Box::new(LinearNode::new(
            "surface_lateral_b",
            input_dim,
            mid_dim,
            Tier::Surface,
            0.89,
        )));

        // Standalone Surface nodes — compete with graph during surface_blend
        resolver.add_tier_node(Box::new(LinearNode::new(
            "surface_standalone_a",
            input_dim,
            mid_dim,
            Tier::Surface,
            0.91,
        )));
        resolver.add_tier_node(Box::new(LinearNode::new(
            "surface_standalone_b",
            input_dim,
            mid_dim,
            Tier::Surface,
            0.90,
        )));

        resolver.add_tier_node(Box::new(LinearNode::new(
            "reasoning_standalone",
            input_dim,
            mid_dim,
            Tier::Reasoning,
            config.reasoning_base_confidence,
        )));
        resolver.add_tier_node(Box::new(LinearNode::new(
            "deep_standalone",
            mid_dim,
            out_dim,
            Tier::Deep,
            0.78,
        )));

        // Lateral edges: surface_entry → surface lateral nodes
        resolver.add_lateral_edge(LateralEdge::if_confidence_below(
            "surface_entry",
            "surface_lateral_a",
            0.75,
            1.0,
        ));
        resolver.add_lateral_edge(LateralEdge::if_confidence_below(
            "surface_entry",
            "surface_lateral_b",
            0.75,
            1.0,
        ));

        // Calibrate: measure actual node confidence distribution, set thresholds.
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

        for sentence in &corpus {
            let input = encoder.encode_text_readonly(sentence);

            for node in &self.surface_nodes {
                surface_confs.push(node.forward(&input).confidence);
            }

            for node in &self.reasoning_nodes {
                reasoning_confs.push(node.forward(&input).confidence);
            }
        }

        surface_confs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        reasoning_confs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if !surface_confs.is_empty() {
            let idx = ((surface_confs.len() as f32 * surface_pct) as usize)
                .min(surface_confs.len() - 1);
            self.config.surface_confidence_threshold = surface_confs[idx];

            let max_surface = *surface_confs.last().unwrap();
            let min_surface = surface_confs[0];
            eprintln!(
                "  Calibration: surface range [{:.4}, {:.4}] → threshold {:.4} (p{:.0}, n={})",
                min_surface, max_surface, self.config.surface_confidence_threshold,
                surface_pct * 100.0, surface_confs.len()
            );
            if self.config.surface_confidence_threshold >= max_surface {
                eprintln!(
                    "  WARNING: surface threshold {:.4} >= max observed {:.4} — no input can reach Surface!",
                    self.config.surface_confidence_threshold, max_surface
                );
            }
        }

        if !reasoning_confs.is_empty() {
            let idx = ((reasoning_confs.len() as f32 * reasoning_pct) as usize)
                .min(reasoning_confs.len() - 1);
            self.config.reasoning_confidence_threshold = reasoning_confs[idx];

            let max_reasoning = *reasoning_confs.last().unwrap();
            let min_reasoning = reasoning_confs[0];
            eprintln!(
                "  Calibration: reasoning range [{:.4}, {:.4}] → threshold {:.4} (p{:.0}, n={})",
                min_reasoning, max_reasoning, self.config.reasoning_confidence_threshold,
                reasoning_pct * 100.0, reasoning_confs.len()
            );
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
        self.graph.add_edge(ConditionalEdge::always("surface_entry", "surface_refine"));
        self.graph.add_edge(ConditionalEdge::if_confidence_below(
            "surface_refine",
            "reasoning_analyze",
            self.config.surface_confidence_threshold,
        ));
        self.graph.add_edge(ConditionalEdge::if_confidence_below(
            "reasoning_analyze",
            "deep_resolve",
            self.config.reasoning_confidence_threshold,
        ));
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

        // Should have escalated past Surface
        assert_ne!(result.tier_reached, Tier::Surface);
        assert!(
            result
                .route
                .execution_trace
                .contains(&"escalate_reasoning".to_string())
        );
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
                | TraversalDirection::Feedback => {}
            }
            assert!(step.confidence_out >= 0.0 && step.confidence_out <= 1.0);
        }
    }
}
