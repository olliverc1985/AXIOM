//! Sparse graph engine — routes inputs through conditional computation paths.

use crate::graph::edge::ConditionalEdge;
use crate::graph::node::ComputeNode;
use crate::tiers::Tier;
use crate::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Direction of travel through the computation graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TraversalDirection {
    /// Standard forward flow (Surface → Reasoning → Deep).
    Forward,
    /// Lateral flow between nodes at the same tier.
    Lateral,
    /// Upward feedback from deeper tiers to shallower ones.
    Feedback,
}

impl std::fmt::Display for TraversalDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraversalDirection::Forward => write!(f, "Forward"),
            TraversalDirection::Lateral => write!(f, "Lateral"),
            TraversalDirection::Feedback => write!(f, "Feedback"),
        }
    }
}

/// A single step in the structured execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceStep {
    /// Which node was executed.
    pub node_id: String,
    /// The tier of the node.
    pub tier: Tier,
    /// Direction of traversal at this step.
    pub direction: TraversalDirection,
    /// Confidence entering this node.
    pub confidence_in: f32,
    /// Confidence exiting this node.
    pub confidence_out: f32,
    /// Whether this step's result came from cache.
    pub was_cached: bool,
}

/// Result of routing an input through the sparse graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteResult {
    /// Final output tensor.
    pub output: Tensor,
    /// Final confidence score.
    pub confidence: f32,
    /// Flat execution trace (node IDs) for backward compatibility.
    pub execution_trace: Vec<String>,
    /// Structured execution trace with direction and confidence per step.
    pub trace_steps: Vec<TraceStep>,
    /// Total accumulated compute cost.
    pub total_compute_cost: f32,
    /// Number of cache hits during routing.
    pub cache_hits: u32,
    /// Number of lateral traversals attempted.
    pub lateral_count: u32,
    /// Number of lateral traversals that prevented escalation.
    pub lateral_prevented_escalation: u32,
}

/// The sparse computation graph.
///
/// Holds compute nodes and conditional edges. Routes inputs by evaluating
/// edge conditions at each step, skipping nodes where conditions aren't met.
pub struct SparseGraph {
    nodes: Vec<Box<dyn ComputeNode>>,
    edges: Vec<ConditionalEdge>,
    /// Entry node ID — routing starts here.
    entry_node: String,
}

impl SparseGraph {
    /// Create a new sparse graph with an entry node.
    pub fn new(entry_node: impl Into<String>) -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            entry_node: entry_node.into(),
        }
    }

    /// Add a compute node to the graph.
    pub fn add_node(&mut self, node: Box<dyn ComputeNode>) {
        self.nodes.push(node);
    }

    /// Add a conditional edge.
    pub fn add_edge(&mut self, edge: ConditionalEdge) {
        self.edges.push(edge);
    }

    /// Remove all edges (used when rebuilding edges after calibration).
    pub fn clear_edges(&mut self) {
        self.edges.clear();
    }

    /// Get a node by ID.
    fn find_node(&self, id: &str) -> Option<&dyn ComputeNode> {
        self.nodes.iter().find(|n| n.node_id() == id).map(|n| &**n)
    }

    /// Get outgoing edges from a given node.
    fn outgoing_edges(&self, node_id: &str) -> Vec<&ConditionalEdge> {
        self.edges.iter().filter(|e| e.from == node_id).collect()
    }

    /// Route an input through the graph, starting from the entry node.
    ///
    /// Evaluates edges at each step, only traversing edges whose conditions are met.
    /// Returns the final output plus both flat and structured execution traces.
    #[allow(unused_assignments)]
    pub fn route(&self, input: &Tensor) -> RouteResult {
        let mut current_tensor = input.clone();
        let mut confidence = 0.5_f32; // start neutral
        let mut trace = Vec::new();
        let mut trace_steps = Vec::new();
        let mut total_cost = 0.0_f32;
        let mut current_tier = Tier::Surface;
        let mut visited: HashMap<String, bool> = HashMap::new();

        // Start at entry node
        let mut current_node_id = self.entry_node.clone();

        for _step in 0..16 {
            // Max 16 steps to prevent infinite loops
            if visited.contains_key(&current_node_id) {
                break; // Prevent cycles
            }

            let node = match self.find_node(&current_node_id) {
                Some(n) => n,
                None => break,
            };

            visited.insert(current_node_id.clone(), true);
            trace.push(current_node_id.clone());

            let confidence_in = confidence;

            // Forward pass through this node
            let output = node.forward(&current_tensor);
            current_tensor = output.tensor;
            confidence = output.confidence;
            total_cost += output.compute_cost;
            current_tier = node.tier();

            trace_steps.push(TraceStep {
                node_id: current_node_id.clone(),
                tier: node.tier(),
                direction: TraversalDirection::Forward,
                confidence_in,
                confidence_out: confidence,
                was_cached: false,
            });

            // Find next node: evaluate outgoing edges
            let outgoing = self.outgoing_edges(&current_node_id);
            let mut next_node: Option<String> = None;

            for edge in outgoing {
                if edge.should_traverse(confidence, current_tier)
                    && !visited.contains_key(&edge.to)
                {
                    next_node = Some(edge.to.clone());
                    break; // Take the first valid edge
                }
            }

            match next_node {
                Some(id) => current_node_id = id,
                None => break, // No valid outgoing edges — we're done
            }
        }

        RouteResult {
            output: current_tensor,
            confidence,
            execution_trace: trace,
            trace_steps,
            total_compute_cost: total_cost,
            cache_hits: 0, // Set by the resolver when cache is involved
            lateral_count: 0,
            lateral_prevented_escalation: 0,
        }
    }

    /// Get the entry node ID.
    pub fn entry_node(&self) -> &str {
        &self.entry_node
    }

    /// Get all node IDs.
    pub fn node_ids(&self) -> Vec<String> {
        self.nodes.iter().map(|n| n.node_id().to_string()).collect()
    }

    /// Total trainable parameter count across all nodes in the graph.
    pub fn total_weight_count(&self) -> usize {
        self.nodes.iter().map(|n| n.weight_count()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::node::LinearNode;

    fn build_simple_graph() -> SparseGraph {
        let mut graph = SparseGraph::new("node_a");

        graph.add_node(Box::new(LinearNode::new("node_a", 4, 4, Tier::Surface, 0.9)));
        graph.add_node(Box::new(LinearNode::new("node_b", 4, 4, Tier::Surface, 0.8)));
        graph.add_node(Box::new(LinearNode::new(
            "node_c",
            4,
            2,
            Tier::Reasoning,
            0.7,
        )));

        // a -> b (always), b -> c (only if confidence below 0.85)
        graph.add_edge(ConditionalEdge::always("node_a", "node_b"));
        graph.add_edge(ConditionalEdge::if_confidence_below(
            "node_b", "node_c", 0.85,
        ));

        graph
    }

    #[test]
    fn test_route_through_graph() {
        let graph = build_simple_graph();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = graph.route(&input);

        // Should have executed at least the entry node
        assert!(!result.execution_trace.is_empty());
        assert_eq!(result.execution_trace[0], "node_a");
        assert!(result.total_compute_cost > 0.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        // Structured trace should match flat trace
        assert_eq!(result.trace_steps.len(), result.execution_trace.len());
        assert_eq!(result.trace_steps[0].direction, TraversalDirection::Forward);
    }

    #[test]
    fn test_route_skips_nodes() {
        let mut graph = SparseGraph::new("start");
        // A high-confidence node
        graph.add_node(Box::new(LinearNode::new(
            "start",
            4,
            4,
            Tier::Surface,
            0.95,
        )));
        // This should be skipped because confidence will be high
        graph.add_node(Box::new(LinearNode::new(
            "escalate",
            4,
            2,
            Tier::Reasoning,
            0.7,
        )));

        // Only escalate if confidence is below 0.5 (unlikely with 0.95 base)
        graph.add_edge(ConditionalEdge::if_confidence_below(
            "start", "escalate", 0.5,
        ));

        let input = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let result = graph.route(&input);

        // Should only hit "start" — escalate is skipped
        assert_eq!(result.execution_trace, vec!["start"]);
    }

    #[test]
    fn test_route_prevents_cycles() {
        let mut graph = SparseGraph::new("a");
        graph.add_node(Box::new(LinearNode::new("a", 4, 4, Tier::Surface, 0.8)));
        graph.add_node(Box::new(LinearNode::new("b", 4, 4, Tier::Surface, 0.8)));

        // Circular edges
        graph.add_edge(ConditionalEdge::always("a", "b"));
        graph.add_edge(ConditionalEdge::always("b", "a"));

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = graph.route(&input);

        // Should visit each node at most once
        assert!(result.execution_trace.len() <= 2);
    }

    #[test]
    fn test_varied_traces() {
        let graph = build_simple_graph();

        let input1 = Tensor::from_vec(vec![0.1, 0.1, 0.1, 0.1]);
        let input2 = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0]);

        let r1 = graph.route(&input1);
        let r2 = graph.route(&input2);

        // Both should start at node_a
        assert_eq!(r1.execution_trace[0], "node_a");
        assert_eq!(r2.execution_trace[0], "node_a");

        // They should produce different outputs
        assert_ne!(r1.output.data, r2.output.data);
    }

    #[test]
    fn test_trace_steps_have_confidence() {
        let graph = build_simple_graph();
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let result = graph.route(&input);

        for step in &result.trace_steps {
            assert!(step.confidence_out >= 0.0 && step.confidence_out <= 1.0);
        }
        // First step starts with neutral confidence
        assert!((result.trace_steps[0].confidence_in - 0.5).abs() < 1e-6);
    }
}
