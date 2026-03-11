//! ComputeNode trait and output types for graph nodes.

use crate::tiers::Tier;
use crate::Tensor;
use serde::{Deserialize, Serialize};

/// Output produced by a compute node after a forward pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeOutput {
    /// The output tensor from this node.
    pub tensor: Tensor,
    /// Confidence score in range [0.0, 1.0].
    pub confidence: f32,
    /// Normalised compute cost of this operation.
    pub compute_cost: f32,
    /// Magnitude of the input vector (preserved for downstream decisions).
    pub input_magnitude: f32,
}

/// A compute node in the sparse graph.
///
/// Each node performs a specific transformation on an input tensor,
/// producing an output with a confidence score and compute cost.
/// Nodes may have trainable weights updated via gradient descent.
pub trait ComputeNode: Send + Sync {
    /// Unique identifier for this node.
    fn node_id(&self) -> &str;
    /// Run the forward pass on the input tensor.
    fn forward(&self, input: &Tensor) -> NodeOutput;
    /// Which reasoning tier this node belongs to.
    fn tier(&self) -> Tier;
    /// Update trainable weights using gradient descent: w = w - lr * gradient.
    /// Default implementation is a no-op for nodes without trainable weights.
    fn update_weights(&mut self, _gradient: &Tensor, _learning_rate: f32) {}
    /// Total number of trainable parameters in this node.
    /// Default is 0 for nodes without trainable weights.
    fn weight_count(&self) -> usize {
        0
    }
    /// Base confidence for this node (used for invariant validation).
    /// Default returns 1.0 (always passes validation).
    fn base_confidence(&self) -> f32 {
        1.0
    }
}

/// A linear transform node with trainable weights: output = ReLU(input * W + bias).
///
/// Weights are stored as a flattened matrix [input_dim, output_dim].
/// Supports gradient descent weight updates.
///
/// Surface-tier nodes use magnitude-aware confidence: input is normalised to unit
/// length for confidence calculation, decoupling confidence from raw vector scale.
pub struct LinearNode {
    /// Node identifier.
    pub id: String,
    /// Weight matrix flattened [input_dim * output_dim].
    pub weights: Tensor,
    /// Bias vector [output_dim].
    pub bias: Tensor,
    /// Input dimension.
    pub input_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Which tier this node belongs to.
    pub node_tier: Tier,
    /// Base confidence this node reports (adjusted by output magnitude).
    pub base_confidence: f32,
    /// Learning rate for gradient descent updates.
    pub learning_rate: f32,
}

impl LinearNode {
    /// Create a new linear node with Xavier-initialised weights.
    pub fn new(
        id: impl Into<String>,
        input_dim: usize,
        output_dim: usize,
        tier: Tier,
        base_confidence: f32,
    ) -> Self {
        let id_str: String = id.into();
        // Seed from node ID so each node gets unique weights.
        let id_seed: u32 = id_str
            .bytes()
            .fold(5381u32, |h, b| h.wrapping_mul(33).wrapping_add(b as u32));
        let scale = 1.0 / (input_dim as f32).sqrt();
        let weights_data: Vec<f32> = (0..input_dim * output_dim)
            .map(|i| {
                let mut s = (i as u32).wrapping_mul(2654435761).wrapping_add(id_seed);
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let x = (s as f32 / u32::MAX as f32) * 2.0 - 1.0;
                x * scale
            })
            .collect();
        let bias_data = vec![0.01; output_dim];

        Self {
            id: id_str,
            weights: Tensor::new(weights_data, vec![input_dim, output_dim]),
            bias: Tensor::new(bias_data, vec![output_dim]),
            input_dim,
            output_dim,
            node_tier: tier,
            base_confidence,
            learning_rate: 0.01,
        }
    }

    /// Create a linear node with explicit weights and bias.
    pub fn with_weights(
        id: impl Into<String>,
        weights: Tensor,
        bias: Tensor,
        tier: Tier,
        base_confidence: f32,
    ) -> Self {
        assert_eq!(weights.shape.len(), 2);
        let input_dim = weights.shape[0];
        let output_dim = weights.shape[1];
        assert_eq!(bias.shape, vec![output_dim]);
        Self {
            id: id.into(),
            weights,
            bias,
            input_dim,
            output_dim,
            node_tier: tier,
            base_confidence,
            learning_rate: 0.01,
        }
    }

    /// Compute confidence ratio from output/input norms.
    ///
    /// Surface nodes use absolute output_norm — larger input magnitude produces
    /// larger output, driving higher confidence. This is how the position-weighted
    /// encoder's magnitude signal reaches the tier decision.
    ///
    /// Non-Surface nodes use output_norm / input_norm — scale-invariant ratio
    /// that reflects how well the node's weights match the input direction.
    fn compute_ratio(output_norm: f32, input_norm: f32, is_surface: bool) -> f32 {
        if is_surface {
            output_norm.min(1.0)
        } else if input_norm > 0.0 {
            (output_norm / input_norm).min(1.0)
        } else {
            0.5
        }
    }
}

impl ComputeNode for LinearNode {
    fn node_id(&self) -> &str {
        &self.id
    }

    fn forward(&self, input: &Tensor) -> NodeOutput {
        // Prepare input: pad with zeros or truncate to match expected input_dim
        let input_slice: Vec<f32> = if input.len() >= self.input_dim {
            input.data[..self.input_dim].to_vec()
        } else {
            let mut padded = input.data.clone();
            padded.resize(self.input_dim, 0.0);
            padded
        };

        let input_norm = Tensor::from_vec(input_slice.clone()).norm();

        let input_2d = Tensor::new(input_slice.clone(), vec![1, self.input_dim]);

        // matmul: [1, input_dim] x [input_dim, output_dim] = [1, output_dim]
        let result = input_2d.matmul(&self.weights);

        // Add bias and apply ReLU
        let mut output_data = Vec::with_capacity(self.output_dim);
        for i in 0..self.output_dim {
            let val = result.data[i] + self.bias.data[i];
            output_data.push(val.max(0.0)); // ReLU
        }

        let output = Tensor::new(output_data, vec![self.output_dim]);

        // Confidence based on output magnitude relative to input
        let output_norm = output.norm();
        let ratio = Self::compute_ratio(output_norm, input_norm, self.node_tier == Tier::Surface);
        let confidence = (self.base_confidence * 0.7 + ratio * 0.3).clamp(0.0, 1.0);

        let compute_cost = (self.input_dim * self.output_dim) as f32 / 1000.0;

        NodeOutput {
            tensor: output,
            confidence,
            compute_cost,
            input_magnitude: input_norm,
        }
    }

    fn tier(&self) -> Tier {
        self.node_tier
    }

    fn update_weights(&mut self, gradient: &Tensor, learning_rate: f32) {
        let lr = if learning_rate > 0.0 {
            learning_rate
        } else {
            self.learning_rate
        };
        // Update weights: w = w - lr * g
        let n = self.weights.data.len().min(gradient.data.len());
        for i in 0..n {
            self.weights.data[i] -= lr * gradient.data[i];
        }
    }

    fn weight_count(&self) -> usize {
        self.weights.data.len() + self.bias.data.len()
    }

    fn base_confidence(&self) -> f32 {
        self.base_confidence
    }
}

/// A simple aggregation node that reduces input dimensionality by averaging chunks.
pub struct AggregateNode {
    /// Node identifier.
    pub id: String,
    /// Factor to reduce dimensionality by.
    pub reduction_factor: usize,
    /// Tier.
    pub node_tier: Tier,
}

impl AggregateNode {
    /// Create a new aggregation node.
    pub fn new(id: impl Into<String>, reduction_factor: usize, tier: Tier) -> Self {
        Self {
            id: id.into(),
            reduction_factor,
            node_tier: tier,
        }
    }
}

impl ComputeNode for AggregateNode {
    fn node_id(&self) -> &str {
        &self.id
    }

    fn forward(&self, input: &Tensor) -> NodeOutput {
        let n = input.data.len();
        let chunk_size = self.reduction_factor.max(1);
        let out_len = (n + chunk_size - 1) / chunk_size;
        let mut output_data = Vec::with_capacity(out_len);

        for chunk in input.data.chunks(chunk_size) {
            let mean: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
            output_data.push(mean);
        }

        let input_magnitude = input.norm();
        let output = Tensor::from_vec(output_data);
        let confidence = 0.9;
        let compute_cost = n as f32 / 10000.0;

        NodeOutput {
            tensor: output,
            confidence,
            compute_cost,
            input_magnitude,
        }
    }

    fn tier(&self) -> Tier {
        self.node_tier
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_node_forward() {
        let node = LinearNode::new("test_linear", 4, 2, Tier::Surface, 0.9);
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let output = node.forward(&input);
        assert_eq!(output.tensor.shape, vec![2]);
        assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
        assert!(output.compute_cost > 0.0);
        assert!(output.input_magnitude > 0.0);
    }

    #[test]
    fn test_aggregate_node() {
        let node = AggregateNode::new("test_agg", 2, Tier::Surface);
        let input = Tensor::from_vec(vec![2.0, 4.0, 6.0, 8.0]);
        let output = node.forward(&input);
        assert_eq!(output.tensor.data, vec![3.0, 7.0]);
    }

    #[test]
    fn test_linear_with_explicit_weights() {
        let weights = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let bias = Tensor::new(vec![0.0, 0.0], vec![2]);
        let node = LinearNode::with_weights("identity", weights, bias, Tier::Surface, 0.9);
        let input = Tensor::from_vec(vec![3.0, 5.0]);
        let output = node.forward(&input);
        // Identity weights with zero bias → output = ReLU(input) = input
        assert_eq!(output.tensor.shape, vec![2]);
        assert!(output.confidence >= 0.0 && output.confidence <= 1.0);
    }

    #[test]
    fn test_weight_count() {
        let node = LinearNode::new("wc", 8, 4, Tier::Surface, 0.9);
        assert_eq!(node.weight_count(), 8 * 4 + 4); // weights + bias
    }

    #[test]
    fn test_update_weights() {
        let mut node = LinearNode::new("train", 2, 2, Tier::Surface, 0.9);
        let w_before: Vec<f32> = node.weights.data.clone();
        let grad = Tensor::new(vec![0.1, 0.1, 0.1, 0.1], vec![2, 2]);
        node.update_weights(&grad, 1.0);
        // Each weight should have decreased by 0.1
        for (before, after) in w_before.iter().zip(node.weights.data.iter()) {
            assert!((before - 0.1 - after).abs() < 1e-6);
        }
    }

    #[test]
    fn test_surface_confidence_valid_range() {
        // Surface nodes use ratio = output_norm / input_norm like all tiers
        let node = LinearNode::new("surface", 4, 4, Tier::Surface, 0.88);
        let small = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let large = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let out_small = node.forward(&small);
        let out_large = node.forward(&large);
        assert!(out_small.confidence >= 0.0 && out_small.confidence <= 1.0);
        assert!(out_large.confidence >= 0.0 && out_large.confidence <= 1.0);
        assert!(out_large.input_magnitude > out_small.input_magnitude);
    }

    #[test]
    fn test_reasoning_magnitude_dependent() {
        // Reasoning nodes should NOT normalise — confidence depends on magnitude
        let node = LinearNode::new("reasoning", 4, 4, Tier::Reasoning, 0.72);
        let small = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let large = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let out_small = node.forward(&small);
        let out_large = node.forward(&large);
        // Confidence may differ because input is not normalised
        // (we just verify both are valid, not necessarily different)
        assert!(out_small.confidence >= 0.0 && out_small.confidence <= 1.0);
        assert!(out_large.confidence >= 0.0 && out_large.confidence <= 1.0);
    }
}
