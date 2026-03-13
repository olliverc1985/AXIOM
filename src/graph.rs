//! Sparse computation graph with four distinct traversal directions:
//! forward, lateral, feedback, and temporal.

use crate::encoder::{cosine_similarity, normalize};
use crate::types::*;
use serde::{Deserialize, Serialize};

/// A compute node in the routing graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeNode {
    pub id: String,
    pub tier: Tier,
    pub weights: Vec<f32>,
    pub base_confidence: f32,
    pub init_strategy: InitStrategy,
    /// Accumulated activation count for specialisation tracking.
    pub activation_count: u64,
}

impl ComputeNode {
    pub fn new(id: String, tier: Tier, dim: usize, init_strategy: InitStrategy) -> Self {
        Self {
            id,
            tier,
            weights: vec![0.0; dim],
            base_confidence: match tier {
                Tier::Surface => 0.8,
                Tier::Reasoning => 0.6,
                Tier::Deep => 0.5,
            },
            init_strategy,
            activation_count: 0,
        }
    }

    /// Compute confidence for an input embedding.
    /// Surface confidence formula includes G5 penalty.
    pub fn compute_confidence(
        &self,
        input: &[f32],
        g5_penalty: f32,
    ) -> f32 {
        let cos_sim = cosine_similarity(input, &self.weights).max(0.0);
        let mut confidence = self.base_confidence * 0.7 + cos_sim * 0.3;

        if self.tier == Tier::Surface {
            confidence -= g5_penalty * 0.35;
        }

        confidence.clamp(0.0, 1.0)
    }

    /// Compute bid strength (cosine similarity to input).
    pub fn bid(&self, input: &[f32]) -> f32 {
        cosine_similarity(input, &self.weights).max(0.0)
    }

    /// Apply Oja's rule weight update.
    /// w(t+1) = w(t) + η * y * (x - y * w(t))
    /// where y = w·x (the output), x = input.
    pub fn oja_update(&mut self, input: &[f32], learning_rate: f32) {
        if self.init_strategy == InitStrategy::AnalyticalInit {
            // Surface nodes are frozen
            return;
        }

        let y: f32 = self
            .weights
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum();

        for (w, x) in self.weights.iter_mut().zip(input.iter()) {
            *w += learning_rate * y * (x - y * *w);
        }

        // Re-normalize to unit length
        normalize(&mut self.weights);
    }
}

/// Conditional edge (forward traversal: Surface → Reasoning → Deep).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalEdge {
    pub from_node: String,
    pub to_node: String,
    pub condition: EdgeCondition,
}

/// Lateral edge (same-tier communication).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LateralEdge {
    pub node_a: String,
    pub node_b: String,
    pub weight: f32,
    pub condition: LateralCondition,
}

/// The sparse computation graph.
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    pub nodes: Vec<ComputeNode>,
    pub forward_edges: Vec<ConditionalEdge>,
    pub lateral_edges: Vec<LateralEdge>,
}

impl ComputationGraph {
    /// Build the default graph topology.
    pub fn new(config: &AxiomConfig) -> Self {
        let dim = config.mid_dim;
        let mut nodes = Vec::new();

        // Surface nodes (AnalyticalInit, frozen)
        for i in 0..config.surface_node_count {
            nodes.push(ComputeNode::new(
                format!("surface_{}", i),
                Tier::Surface,
                dim,
                InitStrategy::AnalyticalInit,
            ));
        }

        // Reasoning nodes (OrthogonalInit)
        for i in 0..config.reasoning_node_count {
            nodes.push(ComputeNode::new(
                format!("reasoning_standalone_{}", i),
                Tier::Reasoning,
                dim,
                InitStrategy::OrthogonalInit,
            ));
        }

        // Deep nodes (OrthogonalInit)
        for i in 0..config.deep_node_count {
            nodes.push(ComputeNode::new(
                format!("deep_standalone_{}", i),
                Tier::Deep,
                dim,
                InitStrategy::OrthogonalInit,
            ));
        }

        // Forward edges: Surface → Reasoning (conditional on low confidence)
        let mut forward_edges = Vec::new();
        for i in 0..config.surface_node_count {
            for j in 0..config.reasoning_node_count {
                forward_edges.push(ConditionalEdge {
                    from_node: format!("surface_{}", i),
                    to_node: format!("reasoning_standalone_{}", j),
                    condition: EdgeCondition::IfConfidenceBelow(0.5),
                });
            }
        }

        // Forward edges: Reasoning → Deep (conditional on low confidence)
        for i in 0..config.reasoning_node_count {
            for j in 0..config.deep_node_count {
                forward_edges.push(ConditionalEdge {
                    from_node: format!("reasoning_standalone_{}", i),
                    to_node: format!("deep_standalone_{}", j),
                    condition: EdgeCondition::IfConfidenceBelow(0.4),
                });
            }
        }

        // Lateral edges: Surface ↔ Surface
        let mut lateral_edges = Vec::new();
        for i in 0..config.surface_node_count {
            for j in (i + 1)..config.surface_node_count {
                lateral_edges.push(LateralEdge {
                    node_a: format!("surface_{}", i),
                    node_b: format!("surface_{}", j),
                    weight: 0.5,
                    condition: LateralCondition::IfConfidenceBelow(0.5),
                });
            }
        }

        // Lateral edges: Reasoning ↔ Reasoning
        for i in 0..config.reasoning_node_count {
            for j in (i + 1)..config.reasoning_node_count {
                lateral_edges.push(LateralEdge {
                    node_a: format!("reasoning_standalone_{}", i),
                    node_b: format!("reasoning_standalone_{}", j),
                    weight: 0.3,
                    condition: LateralCondition::IfConfidenceBelow(0.4),
                });
            }
        }

        Self {
            nodes,
            forward_edges,
            lateral_edges,
        }
    }

    /// Find a node by ID.
    pub fn get_node(&self, id: &str) -> Option<&ComputeNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Find a mutable node by ID.
    pub fn get_node_mut(&mut self, id: &str) -> Option<&mut ComputeNode> {
        self.nodes.iter_mut().find(|n| n.id == id)
    }

    /// Get all nodes of a given tier.
    pub fn nodes_by_tier(&self, tier: Tier) -> Vec<&ComputeNode> {
        self.nodes.iter().filter(|n| n.tier == tier).collect()
    }

    /// Get mutable references to all nodes of a given tier.
    pub fn node_ids_by_tier(&self, tier: Tier) -> Vec<String> {
        self.nodes
            .iter()
            .filter(|n| n.tier == tier)
            .map(|n| n.id.clone())
            .collect()
    }

    /// Initialize Surface nodes analytically from simple examples.
    pub fn analytical_init_surface(&mut self, simple_embeddings: &[Vec<f32>]) {
        if simple_embeddings.is_empty() {
            return;
        }

        let dim = simple_embeddings[0].len();
        let n = simple_embeddings.len() as f32;

        // Compute mean direction of simple examples
        let mut mean_direction = vec![0.0f32; dim];
        for emb in simple_embeddings {
            for (i, v) in emb.iter().enumerate() {
                mean_direction[i] += v / n;
            }
        }
        normalize(&mut mean_direction);

        // Set all Surface nodes to mean direction
        for node in &mut self.nodes {
            if node.tier == Tier::Surface {
                node.weights = mean_direction.clone();
            }
        }
    }

    /// Initialize Reasoning and Deep nodes with near-orthogonal directions.
    pub fn orthogonal_init_reasoning_deep(&mut self, seed: u64) {
        let dim = if let Some(node) = self.nodes.first() {
            node.weights.len()
        } else {
            return;
        };

        let rd_nodes: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.tier != Tier::Surface)
            .map(|(i, _)| i)
            .collect();

        // Generate near-orthogonal directions using a simple deterministic scheme
        let mut rng_state = seed;
        for (idx, &node_idx) in rd_nodes.iter().enumerate() {
            let mut direction = vec![0.0f32; dim];
            for d in 0..dim {
                // Simple pseudo-random: xorshift
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                let val = ((rng_state as f32) / (u64::MAX as f32)) * 2.0 - 1.0;
                direction[d] = val;
            }

            // Apply Gram-Schmidt against previous directions for near-orthogonality
            for prev_idx in &rd_nodes[..idx] {
                let prev = self.nodes[*prev_idx].weights.clone();
                let dot: f32 = direction.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                let prev_norm_sq: f32 = prev.iter().map(|x| x * x).sum();
                if prev_norm_sq > 1e-8 {
                    for (d, p) in direction.iter_mut().zip(prev.iter()) {
                        *d -= dot / prev_norm_sq * p;
                    }
                }
            }

            normalize(&mut direction);
            self.nodes[node_idx].weights = direction;
        }
    }

    /// Apply a feedback signal: nudge base confidence of target tier nodes.
    pub fn apply_feedback(&mut self, signal: &FeedbackSignal) {
        for node in &mut self.nodes {
            if node.tier == signal.to_tier {
                node.base_confidence += signal.confidence_delta;
                node.base_confidence = node.base_confidence.clamp(0.1, 0.95);
            }
        }
    }

    /// Count total parameters (node weights).
    pub fn total_parameters(&self) -> usize {
        self.nodes.iter().map(|n| n.weights.len()).sum()
    }

    /// Compute mean pairwise cosine similarity of Reasoning+Deep nodes.
    pub fn rd_mean_pairwise_cosine(&self) -> f32 {
        let rd_nodes: Vec<&ComputeNode> = self
            .nodes
            .iter()
            .filter(|n| n.tier != Tier::Surface)
            .collect();

        if rd_nodes.len() < 2 {
            return 0.0;
        }

        let mut total = 0.0f32;
        let mut count = 0u32;
        for i in 0..rd_nodes.len() {
            for j in (i + 1)..rd_nodes.len() {
                total += cosine_similarity(&rd_nodes[i].weights, &rd_nodes[j].weights).abs();
                count += 1;
            }
        }

        if count > 0 {
            total / count as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_construction() {
        let config = AxiomConfig::default();
        let graph = ComputationGraph::new(&config);
        assert_eq!(
            graph.nodes.len(),
            config.surface_node_count + config.reasoning_node_count + config.deep_node_count
        );
    }

    #[test]
    fn test_surface_nodes_frozen() {
        let config = AxiomConfig::default();
        let mut graph = ComputationGraph::new(&config);

        // Initialize
        let simple_embs = vec![vec![1.0f32; 128]; 5];
        graph.analytical_init_surface(&simple_embs);

        let surface_node = graph.get_node("surface_0").unwrap();
        let weights_before = surface_node.weights.clone();

        // Try to update
        let input = vec![0.5f32; 128];
        let node = graph.get_node_mut("surface_0").unwrap();
        node.oja_update(&input, 0.01);

        let node = graph.get_node("surface_0").unwrap();
        assert_eq!(node.weights, weights_before, "Surface weights should be frozen");
    }

    #[test]
    fn test_orthogonal_init() {
        let config = AxiomConfig::default();
        let mut graph = ComputationGraph::new(&config);
        graph.orthogonal_init_reasoning_deep(42);

        let mean_cosine = graph.rd_mean_pairwise_cosine();
        assert!(
            mean_cosine < 0.1,
            "Mean pairwise cosine should be near zero, got {}",
            mean_cosine
        );
    }

    #[test]
    fn test_node_confidence() {
        let mut node = ComputeNode::new("test".into(), Tier::Surface, 128, InitStrategy::AnalyticalInit);
        node.weights = vec![1.0; 128];
        crate::encoder::normalize(&mut node.weights);

        let mut input = vec![1.0f32; 128];
        crate::encoder::normalize(&mut input);

        let conf = node.compute_confidence(&input, 0.0);
        // base_confidence * 0.7 + cos_sim * 0.3 = 0.8 * 0.7 + 1.0 * 0.3 = 0.86
        assert!((conf - 0.86).abs() < 0.01);
    }

    #[test]
    fn test_g5_penalty_reduces_surface_confidence() {
        let mut node = ComputeNode::new("test".into(), Tier::Surface, 128, InitStrategy::AnalyticalInit);
        node.weights = vec![1.0; 128];
        crate::encoder::normalize(&mut node.weights);

        let mut input = vec![1.0f32; 128];
        crate::encoder::normalize(&mut input);

        let conf_no_penalty = node.compute_confidence(&input, 0.0);
        let conf_with_penalty = node.compute_confidence(&input, 0.5);
        assert!(conf_with_penalty < conf_no_penalty);
    }

    #[test]
    fn test_feedback_signal() {
        let config = AxiomConfig::default();
        let mut graph = ComputationGraph::new(&config);

        let original = graph.nodes_by_tier(Tier::Reasoning)[0].base_confidence;

        graph.apply_feedback(&FeedbackSignal {
            from_node: "deep_0".into(),
            to_tier: Tier::Reasoning,
            reason: FeedbackReason::LowConfidenceResolved,
            confidence_delta: -0.05,
        });

        let updated = graph.nodes_by_tier(Tier::Reasoning)[0].base_confidence;
        assert!((updated - (original - 0.05)).abs() < 1e-6);
    }
}
