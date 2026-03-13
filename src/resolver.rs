//! Hierarchical resolver orchestrating routing decisions with dynamic
//! coalition formation, embedding cache, and temporal buffer.

use crate::encoder::{cosine_similarity, g5_norm_from_embedding, StructuralEncoder};
use crate::graph::ComputationGraph;
use crate::types::*;
use std::collections::VecDeque;

/// Embedding cache entry.
#[derive(Debug, Clone)]
struct CacheEntry {
    embedding: Vec<f32>,
    result: RouteResult,
}

/// Temporal buffer entry.
#[derive(Debug, Clone)]
struct TemporalEntry {
    embedding: Vec<f32>,
    result: RouteResult,
}

/// The hierarchical resolver.
pub struct HierarchicalResolver {
    pub config: AxiomConfig,
    pub encoder: StructuralEncoder,
    pub graph: ComputationGraph,
    pub mode: RouteMode,
    cache: Vec<CacheEntry>,
    temporal_buffer: VecDeque<TemporalEntry>,
    /// Simple pseudo-random state for coalition sampling.
    rng_state: u64,
}

impl HierarchicalResolver {
    pub fn new(config: AxiomConfig) -> Self {
        let graph = ComputationGraph::new(&config);
        Self {
            encoder: StructuralEncoder::new(),
            graph,
            mode: RouteMode::Training,
            cache: Vec::new(),
            temporal_buffer: VecDeque::new(),
            rng_state: 12345,
            config,
        }
    }

    /// Set operating mode.
    pub fn set_mode(&mut self, mode: RouteMode) {
        self.mode = mode;
        if mode == RouteMode::Training {
            self.cache.clear();
        }
    }

    /// Route an input text through the computation graph.
    pub fn route(&mut self, text: &str) -> RouteResult {
        let embedding = self.encoder.encode(text);
        self.route_embedding(&embedding)
    }

    /// Route a pre-computed embedding.
    pub fn route_embedding(&mut self, embedding: &[f32]) -> RouteResult {
        // Check embedding cache (inference mode only)
        if self.mode == RouteMode::Inference {
            if let Some(cached) = self.check_cache(embedding) {
                return cached;
            }
        }

        let g5_norm = g5_norm_from_embedding(embedding);
        let g5_penalty = self.encoder.g5_penalty(g5_norm);

        let mut result = self.traverse(embedding, g5_penalty);

        // Temporal blending
        if let Some(temporal_match) = self.check_temporal(embedding) {
            result.confidence = self.config.temporal_blend_live * result.confidence
                + self.config.temporal_blend_past * temporal_match.confidence;
        }

        // Update temporal buffer
        self.push_temporal(embedding.to_vec(), result.clone());

        // Update cache (inference mode only)
        if self.mode == RouteMode::Inference {
            self.cache.push(CacheEntry {
                embedding: embedding.to_vec(),
                result: result.clone(),
            });
        }

        result
    }

    /// Multi-sentence routing with chunk escalation.
    pub fn route_chunked(&mut self, text: &str) -> RouteResult {
        let (embedding, avg_g5_norm) = self.encoder.encode_chunked(text);

        // Check embedding cache
        if self.mode == RouteMode::Inference {
            if let Some(cached) = self.check_cache(&embedding) {
                return cached;
            }
        }

        let g5_penalty = self.encoder.g5_penalty(avg_g5_norm);

        // Check per-chunk escalation
        let sentences = crate::encoder::split_sentences(text);
        if sentences.len() > 1 {
            let mut below_threshold = 0;
            for sentence in &sentences {
                let sent_emb = self.encoder.encode(sentence);
                let sent_g5 = g5_norm_from_embedding(&sent_emb);
                let sent_penalty = self.encoder.g5_penalty(sent_g5);

                let surface_nodes = self.graph.node_ids_by_tier(Tier::Surface);
                let mut max_conf = 0.0f32;
                for node_id in &surface_nodes {
                    if let Some(node) = self.graph.get_node(node_id) {
                        max_conf = max_conf.max(node.compute_confidence(&sent_emb, sent_penalty));
                    }
                }
                if max_conf < self.config.surface_escalation_threshold {
                    below_threshold += 1;
                }
            }

            let escalation_ratio = below_threshold as f32 / sentences.len() as f32;
            if escalation_ratio > self.config.chunk_escalation_threshold {
                // Force escalation to at least Reasoning
                let mut result = self.traverse_from_tier(&embedding, g5_penalty, Tier::Reasoning);
                result
                    .trace
                    .push(TraceStep {
                        node_id: "chunk_escalation".into(),
                        tier: Tier::Reasoning,
                        direction: TraversalDirection::Forward,
                        confidence_in: escalation_ratio,
                        confidence_out: result.confidence,
                        was_cached: false,
                    });
                return result;
            }
        }

        let mut result = self.traverse(&embedding, g5_penalty);

        // Temporal blending
        if let Some(temporal_match) = self.check_temporal(&embedding) {
            result.confidence = self.config.temporal_blend_live * result.confidence
                + self.config.temporal_blend_past * temporal_match.confidence;
        }

        self.push_temporal(embedding.to_vec(), result.clone());

        if self.mode == RouteMode::Inference {
            self.cache.push(CacheEntry {
                embedding: embedding.clone(),
                result: result.clone(),
            });
        }

        result
    }

    /// Core graph traversal.
    fn traverse(&mut self, embedding: &[f32], g5_penalty: f32) -> RouteResult {
        let mut trace = Vec::new();
        let mut lateral_count = 0usize;
        let mut lateral_prevented = 0usize;

        // Phase 1: Surface tier evaluation with lateral communication
        let surface_ids = self.graph.node_ids_by_tier(Tier::Surface);
        let mut best_surface_conf = 0.0f32;
        let mut best_surface_id = surface_ids[0].clone();

        for node_id in &surface_ids {
            let node = self.graph.get_node(node_id).unwrap();
            let conf = node.compute_confidence(embedding, g5_penalty);

            trace.push(TraceStep {
                node_id: node_id.clone(),
                tier: Tier::Surface,
                direction: TraversalDirection::Forward,
                confidence_in: 0.0,
                confidence_out: conf,
                was_cached: false,
            });

            if conf > best_surface_conf {
                best_surface_conf = conf;
                best_surface_id = node_id.clone();
            }
        }

        // Lateral traversal: if best surface confidence is low, try lateral edges
        if best_surface_conf < self.config.surface_escalation_threshold {
            for edge in &self.graph.lateral_edges.clone() {
                if edge.condition.evaluate(best_surface_conf) {
                    let (target_id, _source_id) = if edge.node_a == best_surface_id {
                        (&edge.node_b, &edge.node_a)
                    } else if edge.node_b == best_surface_id {
                        (&edge.node_a, &edge.node_b)
                    } else {
                        continue;
                    };

                    if let Some(target_node) = self.graph.get_node(target_id) {
                        if target_node.tier != Tier::Surface {
                            continue;
                        }
                        let lat_conf = target_node.compute_confidence(embedding, g5_penalty);
                        lateral_count += 1;

                        trace.push(TraceStep {
                            node_id: target_id.clone(),
                            tier: Tier::Surface,
                            direction: TraversalDirection::Lateral,
                            confidence_in: best_surface_conf,
                            confidence_out: lat_conf,
                            was_cached: false,
                        });

                        if lat_conf >= self.config.surface_escalation_threshold {
                            lateral_prevented += 1;
                            best_surface_conf = lat_conf;
                            best_surface_id = target_id.clone();
                        }
                    }
                }
            }
        }

        // If Surface confidence is sufficient, route to Surface
        if best_surface_conf >= self.config.surface_escalation_threshold {
            let mut result = RouteResult::new(Tier::Surface, best_surface_conf, best_surface_id);
            result.trace = trace;
            result.lateral_count = lateral_count;
            result.lateral_prevented_escalation = lateral_prevented;
            return result;
        }

        // Phase 2: Escalate to Reasoning+Deep via coalition formation
        self.traverse_from_tier_with_trace(embedding, g5_penalty, Tier::Reasoning, trace, lateral_count, lateral_prevented)
    }

    /// Traverse starting from a specific tier (for forced escalation).
    fn traverse_from_tier(
        &mut self,
        embedding: &[f32],
        g5_penalty: f32,
        min_tier: Tier,
    ) -> RouteResult {
        self.traverse_from_tier_with_trace(embedding, g5_penalty, min_tier, Vec::new(), 0, 0)
    }

    fn traverse_from_tier_with_trace(
        &mut self,
        embedding: &[f32],
        g5_penalty: f32,
        _min_tier: Tier,
        mut trace: Vec<TraceStep>,
        lateral_count: usize,
        lateral_prevented: usize,
    ) -> RouteResult {
        // Dynamic coalition formation across Reasoning and Deep tiers
        let rd_ids: Vec<String> = self
            .graph
            .nodes
            .iter()
            .filter(|n| n.tier != Tier::Surface)
            .map(|n| n.id.clone())
            .collect();

        // Compute bids
        let mut bids: Vec<(String, Tier, f32)> = Vec::new();
        for node_id in &rd_ids {
            if let Some(node) = self.graph.get_node(node_id) {
                let bid = node.bid(embedding);
                if bid >= self.config.bid_threshold {
                    bids.push((node_id.clone(), node.tier, bid));
                }
            }
        }

        // Sort by bid descending
        bids.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Weighted random sampling for coalition (up to max_coalition_size)
        let coalition = self.sample_coalition(&bids);

        if coalition.is_empty() {
            // No nodes bid above threshold — default to Deep
            let mut result = RouteResult::new(Tier::Deep, 0.3, "default_deep".into());
            result.trace = trace;
            result.lateral_count = lateral_count;
            result.lateral_prevented_escalation = lateral_prevented;
            return result;
        }

        // Process coalition: highest bidder resolves
        let (resolved_id, resolved_tier) = (coalition[0].0.clone(), coalition[0].1);
        let mut max_conf = 0.0f32;

        let coalition_ids: Vec<String> = coalition.iter().map(|(id, _, _)| id.clone()).collect();

        for (node_id, tier, bid) in &coalition {
            let conf = if let Some(node) = self.graph.get_node(node_id) {
                node.compute_confidence(embedding, g5_penalty)
            } else {
                continue;
            };

            trace.push(TraceStep {
                node_id: node_id.clone(),
                tier: *tier,
                direction: TraversalDirection::Forward,
                confidence_in: *bid,
                confidence_out: conf,
                was_cached: false,
            });

            if conf > max_conf {
                max_conf = conf;
            }
        }

        // Determine if cross-tier resolution occurred
        let cross_tier = coalition
            .iter()
            .any(|(_, t, _)| *t != resolved_tier);

        // Emit feedback if Deep resolved with high confidence
        if resolved_tier == Tier::Deep && max_conf > 0.90 {
            let signal = FeedbackSignal {
                from_node: resolved_id.clone(),
                to_tier: Tier::Reasoning,
                reason: FeedbackReason::LowConfidenceResolved,
                confidence_delta: -0.02,
            };
            self.graph.apply_feedback(&signal);

            trace.push(TraceStep {
                node_id: resolved_id.clone(),
                tier: Tier::Deep,
                direction: TraversalDirection::Feedback,
                confidence_in: max_conf,
                confidence_out: max_conf,
                was_cached: false,
            });
        }

        // Update activation counts
        for (node_id, _, _) in &coalition {
            if let Some(node) = self.graph.get_node_mut(node_id) {
                node.activation_count += 1;
            }
        }

        let mut result = RouteResult::new(resolved_tier, max_conf, resolved_id);
        result.trace = trace;
        result.lateral_count = lateral_count;
        result.lateral_prevented_escalation = lateral_prevented;
        result.coalition_members = coalition_ids;
        result.cross_tier_resolution = cross_tier;

        result
    }

    /// Weighted random sampling for coalition.
    fn sample_coalition(&mut self, bids: &[(String, Tier, f32)]) -> Vec<(String, Tier, f32)> {
        if bids.is_empty() {
            return Vec::new();
        }

        let max_size = self.config.max_coalition_size.min(bids.len());
        let mut selected = Vec::with_capacity(max_size);
        let mut available: Vec<(String, Tier, f32)> = bids.to_vec();

        // Always include the highest bidder
        selected.push(available.remove(0));

        // Weighted sampling for remaining slots
        while selected.len() < max_size && !available.is_empty() {
            let total_weight: f32 = available.iter().map(|(_, _, b)| *b).sum();
            if total_weight < 1e-8 {
                break;
            }

            let threshold = self.pseudo_random() * total_weight;
            let mut cumulative = 0.0f32;
            let mut pick_idx = 0;
            for (i, (_, _, bid)) in available.iter().enumerate() {
                cumulative += bid;
                if cumulative >= threshold {
                    pick_idx = i;
                    break;
                }
            }

            selected.push(available.remove(pick_idx));
        }

        selected
    }

    /// Simple pseudo-random number generator (xorshift64).
    fn pseudo_random(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f32) / (u64::MAX as f32)
    }

    /// Check embedding cache for similar input.
    fn check_cache(&self, embedding: &[f32]) -> Option<RouteResult> {
        for entry in &self.cache {
            let sim = cosine_similarity(embedding, &entry.embedding);
            if sim >= self.config.cache_similarity_threshold {
                let mut result = entry.result.clone();
                result.was_cached = true;
                return Some(result);
            }
        }
        None
    }

    /// Check temporal buffer for similar recent input.
    fn check_temporal(&self, embedding: &[f32]) -> Option<&RouteResult> {
        for entry in &self.temporal_buffer {
            let sim = cosine_similarity(embedding, &entry.embedding);
            if sim >= self.config.temporal_similarity_threshold {
                return Some(&entry.result);
            }
        }
        None
    }

    /// Push to temporal buffer (ring buffer of fixed capacity).
    fn push_temporal(&mut self, embedding: Vec<f32>, result: RouteResult) {
        if self.temporal_buffer.len() >= self.config.temporal_buffer_capacity {
            self.temporal_buffer.pop_front();
        }
        self.temporal_buffer.push_back(TemporalEntry { embedding, result });
    }

    /// Train on a single example with known complexity label.
    pub fn train_example(&mut self, text: &str, is_simple: bool) {
        assert_eq!(self.mode, RouteMode::Training, "Must be in training mode");

        let embedding = self.encoder.encode(text);
        let g5_norm = g5_norm_from_embedding(&embedding);
        self.encoder.accumulate_g5(g5_norm, is_simple);

        // Route to get activations
        let result = self.route_embedding(&embedding);

        // Oja update on activated Reasoning+Deep nodes
        let learning_rate = 0.01;
        for member_id in &result.coalition_members {
            if let Some(node) = self.graph.get_node_mut(member_id) {
                node.oja_update(&embedding, learning_rate);
            }
        }
    }

    /// Finalize training: calibrate G5 parameters and compute thresholds.
    pub fn finalize_training(&mut self) {
        self.encoder.calibrate_g5();
    }

    /// Export weights to serializable format.
    pub fn export_weights(&self) -> AxiomWeights {
        AxiomWeights {
            g5_simple_mean_norm: self.encoder.g5_simple_mean_norm,
            g5_complex_mean_norm: self.encoder.g5_complex_mean_norm,
            node_weights: self
                .graph
                .nodes
                .iter()
                .map(|n| NodeWeightEntry {
                    node_id: n.id.clone(),
                    tier: n.tier,
                    weights: n.weights.clone(),
                    base_confidence: n.base_confidence,
                })
                .collect(),
        }
    }

    /// Import weights from saved format.
    pub fn import_weights(&mut self, weights: &AxiomWeights) {
        self.encoder.g5_simple_mean_norm = weights.g5_simple_mean_norm;
        self.encoder.g5_complex_mean_norm = weights.g5_complex_mean_norm;

        for entry in &weights.node_weights {
            if let Some(node) = self.graph.get_node_mut(&entry.node_id) {
                node.weights = entry.weights.clone();
                node.base_confidence = entry.base_confidence;
            }
        }
    }

    /// Get routing distribution statistics.
    pub fn routing_distribution(&self, results: &[RouteResult]) -> (f32, f32, f32) {
        let total = results.len() as f32;
        if total == 0.0 {
            return (0.0, 0.0, 0.0);
        }
        let surface = results
            .iter()
            .filter(|r| r.selected_tier == Tier::Surface)
            .count() as f32
            / total;
        let reasoning = results
            .iter()
            .filter(|r| r.selected_tier == Tier::Reasoning)
            .count() as f32
            / total;
        let deep = results
            .iter()
            .filter(|r| r.selected_tier == Tier::Deep)
            .count() as f32
            / total;
        (surface, reasoning, deep)
    }

    /// Compute cost savings vs all-Opus routing.
    pub fn cost_savings(results: &[RouteResult]) -> f64 {
        let cost_model = default_cost_model();
        let opus_cost = cost_model
            .iter()
            .find(|c| c.tier == Tier::Deep)
            .unwrap()
            .query_cost();

        let mut total_axiom_cost = 0.0f64;
        for result in results {
            let tier_cost = cost_model
                .iter()
                .find(|c| c.tier == result.selected_tier)
                .unwrap()
                .query_cost();
            total_axiom_cost += tier_cost;
        }

        let total_opus_cost = opus_cost * results.len() as f64;

        if total_opus_cost > 0.0 {
            (1.0 - total_axiom_cost / total_opus_cost) * 100.0
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_resolver() -> HierarchicalResolver {
        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::new(config);

        // Initialize graph
        let simple_examples: Vec<Vec<f32>> = (0..10)
            .map(|_| resolver.encoder.encode("What time is it?"))
            .collect();
        resolver.graph.analytical_init_surface(&simple_examples);
        resolver.graph.orthogonal_init_reasoning_deep(42);
        resolver.set_mode(RouteMode::Inference);
        resolver
    }

    #[test]
    fn test_route_simple_query() {
        let mut resolver = make_resolver();
        let result = resolver.route("What time is it?");
        assert!(result.trace.len() > 0);
        // Should produce a valid tier
        assert!(
            result.selected_tier == Tier::Surface
                || result.selected_tier == Tier::Reasoning
                || result.selected_tier == Tier::Deep
        );
    }

    #[test]
    fn test_route_produces_trace() {
        let mut resolver = make_resolver();
        let result = resolver.route("Hello world");
        assert!(!result.trace.is_empty(), "Routing should produce a trace");
    }

    #[test]
    fn test_cache_hit() {
        let mut resolver = make_resolver();

        let _result1 = resolver.route("What is the weather?");
        let result2 = resolver.route("What is the weather?");

        assert!(result2.was_cached, "Identical input should hit cache");
    }

    #[test]
    fn test_cache_disabled_in_training() {
        let mut resolver = make_resolver();
        resolver.set_mode(RouteMode::Training);

        let _result1 = resolver.route("What is the weather?");
        let result2 = resolver.route("What is the weather?");

        assert!(!result2.was_cached, "Cache should be disabled in training mode");
    }

    #[test]
    fn test_coalition_formation() {
        let mut resolver = make_resolver();
        let result = resolver.route(
            "Although the phenomenon has been observed across multiple contexts, \
             the underlying mechanisms that drive it remain poorly understood, \
             which complicates efforts to develop interventions.",
        );
        // Complex query should potentially form coalition
        if result.selected_tier != Tier::Surface {
            assert!(
                !result.coalition_members.is_empty(),
                "Non-surface routing should have coalition members"
            );
        }
    }

    #[test]
    fn test_cost_savings() {
        let results = vec![
            RouteResult::new(Tier::Surface, 0.9, "s0".into()),
            RouteResult::new(Tier::Surface, 0.9, "s0".into()),
            RouteResult::new(Tier::Reasoning, 0.6, "r0".into()),
            RouteResult::new(Tier::Deep, 0.5, "d0".into()),
        ];
        let savings = HierarchicalResolver::cost_savings(&results);
        assert!(savings > 0.0, "Mixed routing should save vs all-Opus");
    }

    #[test]
    fn test_training_mode() {
        let config = AxiomConfig::default();
        let mut resolver = HierarchicalResolver::new(config);
        resolver.set_mode(RouteMode::Training);

        // Initialize
        let simple_embs: Vec<Vec<f32>> = (0..5)
            .map(|_| resolver.encoder.encode("Hello"))
            .collect();
        resolver.graph.analytical_init_surface(&simple_embs);
        resolver.graph.orthogonal_init_reasoning_deep(42);

        // Train on some examples
        resolver.train_example("What time is it?", true);
        resolver.train_example(
            "Analyze the epistemological implications of quantum decoherence \
             on the measurement problem in light of competing interpretations.",
            false,
        );

        resolver.finalize_training();
        // Should not panic and G5 params should be updated
    }

    #[test]
    fn test_export_import_weights() {
        let resolver = make_resolver();
        let weights = resolver.export_weights();

        assert_eq!(
            weights.node_weights.len(),
            resolver.graph.nodes.len()
        );

        // Import into a fresh resolver
        let config = AxiomConfig::default();
        let mut resolver2 = HierarchicalResolver::new(config);
        resolver2.import_weights(&weights);

        assert_eq!(
            resolver2.encoder.g5_simple_mean_norm,
            resolver.encoder.g5_simple_mean_norm
        );
    }
}
