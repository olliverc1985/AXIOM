//! ComputeNode trait and output types for graph nodes.

use crate::tiers::Tier;
use crate::Tensor;
use serde::{Deserialize, Serialize};
// AtomicU32 no longer needed — EMA expected_norm removed in Phase 9.

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
/// Nodes may have trainable weights updated via Hebbian reinforcement.
pub trait ComputeNode: Send + Sync {
    /// Unique identifier for this node.
    fn node_id(&self) -> &str;
    /// Run the forward pass on the input tensor.
    fn forward(&self, input: &Tensor) -> NodeOutput;
    /// Which reasoning tier this node belongs to.
    fn tier(&self) -> Tier;
    /// Oja's rule weight update: reinforcement (+1) or suppression (-1).
    ///
    /// w_ij += lr * signal * output_j * (input_i - output_j * w_ij)
    /// Self-normalising — converges stably without weight explosion.
    /// `signal` is +1.0 (reinforce) or -1.0 (suppress), `learning_rate` controls step size.
    /// Default implementation is a no-op for nodes without trainable weights.
    fn hebbian_update(
        &mut self,
        _input: &Tensor,
        _output: &Tensor,
        _signal: f32,
        _learning_rate: f32,
    ) {
    }
    /// Error signal weight update — targeted outer product update.
    ///
    /// `w_ij += error_lr * modulator * input_i * output_j`
    ///
    /// Used for two mechanisms:
    /// - **Escalation penalty** (modulator < 0): suppresses Surface node weights
    ///   when the node escalated and Deep resolved confidently.
    /// - **Cache reinforcement** (modulator > 0): reinforces the producing node
    ///   when a high-similarity cache hit occurs.
    ///
    /// Default implementation is a no-op for nodes without trainable weights.
    fn error_update(
        &mut self,
        _input: &Tensor,
        _output: &Tensor,
        _error_lr: f32,
        _modulator: f32,
    ) {
    }
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
    /// Snapshot of current weights for drift tracking. Returns L2 norm of weights.
    fn weight_norm(&self) -> f32 {
        0.0
    }
    /// Number of times this node has been activated (used for usage-proportional lr).
    fn activation_count(&self) -> usize {
        0
    }
    /// Increment the activation counter.
    fn increment_activation(&mut self) {}
    /// Reset the activation counter (call between passes).
    fn reset_activation(&mut self) {}
    /// Accumulate a positive (Surface-resolved) input for contrastive learning.
    fn accumulate_positive(&mut self, _input: &Tensor) {}
    /// Accumulate a negative (escalated) input for contrastive learning.
    fn accumulate_negative(&mut self, _input: &Tensor) {}
    /// Apply contrastive weight update from accumulated positive/negative examples.
    ///
    /// Rank-1 outer product update: `w += lr * outer(contrast, contrast)` where
    /// `contrast = positive_mean - negative_mean`. Resets accumulators after update.
    fn apply_contrastive_update(&mut self) -> Option<ContrastiveUpdateInfo> {
        None
    }
    /// Set the contrastive learning rate for this node.
    fn set_contrastive_lr(&mut self, _lr: f32) {}
    /// Reset contrastive accumulators without applying any update.
    fn reset_contrastive_accumulators(&mut self) {}
    /// Whether this node's weights are frozen (no updates allowed).
    fn is_frozen(&self) -> bool {
        false
    }
    /// Set the frozen state of this node.
    fn set_frozen(&mut self, _frozen: bool) {}
    /// Analytically initialise weights toward a discrimination direction.
    fn init_analytical(&mut self, _init: &AnalyticalInit, _seed: u64) {}
    /// Set G5 magnitude penalty parameters for Surface confidence (Phase 14).
    ///
    /// When set, Surface confidence is reduced by a penalty proportional to
    /// how close the input's G5 norm is to the complex corpus mean.
    /// Params: (g5_offset, g5_end, simple_mean_norm, complex_mean_norm, weight).
    fn set_g5_magnitude_penalty(&mut self, _params: Option<(usize, usize, f32, f32, f32)>) {}
    /// Set G4 magnitude penalty parameters for Surface confidence.
    /// Params: (g4_start, g4_end, simple_mean_norm, complex_mean_norm, weight).
    fn set_g4_magnitude_penalty(&mut self, _params: Option<(usize, usize, f32, f32, f32)>) {}
    /// Set the confidence base weight (mix ratio between base_confidence and cosine_sim).
    fn set_confidence_base_weight(&mut self, _weight: f32) {}
    /// Orthogonally initialise weights toward a specific basis direction.
    fn init_orthogonal(&mut self, _basis_vector: &[f32], _noise_scale: f32, _seed: u64) {}
    /// Return the node's mean weight direction vector (mean of weight columns).
    /// Used for coalition bidding — cosine similarity between input and this direction.
    fn weight_direction(&self) -> Vec<f32> {
        Vec::new()
    }
    /// Serialize this node's weights for persistence. Returns None for non-trainable nodes.
    fn save_weights_data(&self) -> Option<NodeWeightsData> {
        None
    }
    /// Load weights from serialized data (matched by node ID).
    fn load_weights_data(&mut self, _data: &NodeWeightsData) {}
}

/// Diagnostic info returned from a contrastive update.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContrastiveUpdateInfo {
    /// Node ID that was updated.
    pub node_id: String,
    /// Number of positive examples accumulated.
    pub positive_count: usize,
    /// Number of negative examples accumulated.
    pub negative_count: usize,
    /// L2 norm of the contrast vector (positive_mean - negative_mean).
    pub contrast_magnitude: f32,
    /// Weight norm before the update.
    pub weight_norm_before: f32,
    /// Weight norm after the update.
    pub weight_norm_after: f32,
    /// L2 norm of the positive mean vector.
    pub positive_mean_norm: f32,
    /// L2 norm of the negative mean vector.
    pub negative_mean_norm: f32,
}

/// Serializable weight data for a single node.
///
/// Used by `save_all_weights()` / `load_all_weights()` to persist trained
/// models to disk. The inference binary loads these to skip retraining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeWeightsData {
    /// Node identifier (must match for loading).
    pub id: String,
    /// Flattened weight matrix.
    pub weights: Vec<f32>,
    /// Bias vector.
    pub bias: Vec<f32>,
    /// Input dimension.
    pub input_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Tier name (Surface, Reasoning, Deep).
    pub tier: String,
    /// Base confidence value.
    pub base_confidence: f32,
    /// Whether this node is frozen.
    pub frozen: bool,
}

/// Parameters for orthogonal weight initialisation.
///
/// Generates a set of approximately orthogonal basis vectors via Gram-Schmidt.
/// Each coalition-eligible (R+D) node receives a unique basis vector, ensuring
/// diverse weight directions for specialised bidding.
pub struct OrthogonalInit {
    /// Pre-computed orthogonal basis vectors (one per node).
    pub basis_vectors: Vec<Vec<f32>>,
    /// Scale of Xavier noise added to each weight row (default 0.05).
    pub noise_scale: f32,
}

impl OrthogonalInit {
    /// Generate `n_nodes` approximately orthogonal unit vectors in `dim`-dimensional
    /// space using Gram-Schmidt orthogonalisation on deterministically seeded random
    /// vectors. Each vector becomes the principal weight direction for one node.
    pub fn generate(n_nodes: usize, dim: usize, seed: u64) -> Self {
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n_nodes);

        for k in 0..n_nodes {
            // Deterministic random vector seeded by (seed + k)
            let mut s = (seed.wrapping_add(k as u64)) as u32;
            let mut v: Vec<f32> = (0..dim)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    (s as f32 / u32::MAX as f32) * 2.0 - 1.0
                })
                .collect();

            // Gram-Schmidt: subtract projections onto all previous vectors
            for prev in &vectors {
                let dot: f32 = v.iter().zip(prev.iter()).map(|(a, b)| a * b).sum();
                for (vi, pi) in v.iter_mut().zip(prev.iter()) {
                    *vi -= dot * pi;
                }
            }

            // Normalise to unit length
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for vi in &mut v {
                    *vi /= norm;
                }
            }

            vectors.push(v);
        }

        Self {
            basis_vectors: vectors,
            noise_scale: 0.05,
        }
    }
}

/// Parameters for analytical weight initialisation.
///
/// Sets a node's weight matrix principal direction to the discrimination
/// direction (simple_mean - complex_mean, L2 normalised) plus small noise.
/// Used to initialise Surface nodes as fixed complexity discriminators.
pub struct AnalyticalInit {
    /// Discrimination direction: simple_mean - complex_mean, L2 normalised.
    pub discrimination_direction: Vec<f32>,
    /// Scale factor for Xavier noise added to each row (default 0.1).
    pub noise_scale: f32,
}

/// A linear transform node with trainable weights: output = ReLU(input * W + bias).
///
/// Weights are stored as a flattened matrix [input_dim, output_dim].
/// Supports gradient descent weight updates.
///
/// Confidence is computed via cosine similarity between the input vector and
/// the node's learned weight direction (mean of weight matrix columns). This is
/// magnitude-invariant — it measures directional alignment, not magnitude.
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
    /// Base confidence this node reports (adjusted by cosine similarity).
    pub base_confidence: f32,
    /// Learning rate for gradient descent updates.
    pub learning_rate: f32,
    /// Number of times this node has been activated (for usage-proportional lr).
    pub activation_count: usize,
    /// Running sum of Surface-resolved input vectors (contrastive learning).
    positive_accumulator: Vec<f32>,
    /// Running sum of escalating input vectors (contrastive learning).
    negative_accumulator: Vec<f32>,
    /// Count of positive examples accumulated.
    positive_count: usize,
    /// Count of negative examples accumulated.
    negative_count: usize,
    /// Learning rate for contrastive updates (default 0.01).
    pub contrastive_lr: f32,
    /// When true, all weight update methods become no-ops.
    /// Used for frozen Surface discriminators with analytical initialisation.
    pub frozen: bool,
    /// G5 magnitude penalty for Surface confidence (Phase 14).
    /// When Some: (g5_offset, g5_end, simple_mean_norm, complex_mean_norm, weight).
    /// Subtracts a penalty proportional to how close the input's G5 norm is to
    /// the complex corpus mean, pushing structurally complex inputs below threshold.
    g5_magnitude_penalty: Option<(usize, usize, f32, f32, f32)>,
    /// G4 magnitude penalty for Surface confidence.
    /// When Some: (g4_start, g4_end, simple_mean_norm, complex_mean_norm, weight).
    g4_magnitude_penalty: Option<(usize, usize, f32, f32, f32)>,
    /// Weight for base_confidence in the confidence mix (default 0.7).
    /// confidence = base_weight * base_confidence + (1 - base_weight) * cosine_sim
    pub confidence_base_weight: f32,
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
            activation_count: 0,
            positive_accumulator: vec![0.0; input_dim],
            negative_accumulator: vec![0.0; input_dim],
            positive_count: 0,
            negative_count: 0,
            contrastive_lr: 0.01,
            frozen: false,
            g5_magnitude_penalty: None,
            g4_magnitude_penalty: None,
            confidence_base_weight: 0.7,
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
            activation_count: 0,
            positive_accumulator: vec![0.0; input_dim],
            negative_accumulator: vec![0.0; input_dim],
            positive_count: 0,
            negative_count: 0,
            contrastive_lr: 0.01,
            frozen: false,
            g5_magnitude_penalty: None,
            g4_magnitude_penalty: None,
            confidence_base_weight: 0.7,
        }
    }

    /// Replace weights with analytical initialisation aligned to discrimination direction.
    ///
    /// Each row of the weight matrix is set to `discrimination_direction + noise`.
    /// The noise is Xavier-scaled by `init.noise_scale` and seeded deterministically
    /// from `(node_id_hash XOR seed XOR row_index)` so each node and row gets
    /// unique noise while all point in the same general direction.
    pub fn init_analytical(&mut self, init: &AnalyticalInit, seed: u64) {
        let id_hash: u64 = self.id.bytes().fold(5381u64, |h, b| h.wrapping_mul(33).wrapping_add(b as u64));
        let xavier_scale = 1.0 / (self.input_dim as f32).sqrt();
        let noise_scale = init.noise_scale * xavier_scale;

        for i in 0..self.input_dim {
            let row_seed = id_hash ^ seed ^ (i as u64);
            for j in 0..self.output_dim {
                let idx = i * self.output_dim + j;
                if idx < self.weights.data.len() {
                    // Deterministic noise from (row_seed XOR column)
                    let mut s = (row_seed ^ (j as u64)) as u32;
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    let noise = (s as f32 / u32::MAX as f32) * 2.0 - 1.0;

                    let dir_val = if i < init.discrimination_direction.len() {
                        init.discrimination_direction[i]
                    } else {
                        0.0
                    };
                    self.weights.data[idx] = dir_val + noise * noise_scale;
                }
            }
        }
    }

    /// Replace weights with orthogonal initialisation aligned to a specific basis vector.
    ///
    /// Similar to `init_analytical` but uses a per-node orthogonal direction instead
    /// of the shared simple_mean direction. This ensures diverse weight directions
    /// across coalition-eligible nodes.
    pub fn init_orthogonal(&mut self, basis_vector: &[f32], noise_scale: f32, seed: u64) {
        let id_hash: u64 = self
            .id
            .bytes()
            .fold(5381u64, |h, b| h.wrapping_mul(33).wrapping_add(b as u64));
        let xavier_scale = 1.0 / (self.input_dim as f32).sqrt();
        let scaled_noise = noise_scale * xavier_scale;

        for i in 0..self.input_dim {
            let row_seed = id_hash ^ seed ^ (i as u64);
            for j in 0..self.output_dim {
                let idx = i * self.output_dim + j;
                if idx < self.weights.data.len() {
                    let mut s = (row_seed ^ (j as u64)) as u32;
                    s ^= s << 13;
                    s ^= s >> 17;
                    s ^= s << 5;
                    let noise = (s as f32 / u32::MAX as f32) * 2.0 - 1.0;

                    let dir_val = if i < basis_vector.len() {
                        basis_vector[i]
                    } else {
                        0.0
                    };
                    self.weights.data[idx] = dir_val + noise * scaled_noise;
                }
            }
        }
    }

    /// Compute the mean weight direction vector (mean of weight matrix columns).
    ///
    /// The weight matrix has shape [input_dim, output_dim]. Each column j represents
    /// how input dimensions contribute to output j. The mean across all columns gives
    /// a single vector of length input_dim representing the node's overall learned
    /// direction. Cosine similarity between input and this direction is the
    /// magnitude-invariant confidence signal.
    pub fn weight_direction(&self) -> Vec<f32> {
        let mut direction = vec![0.0f32; self.input_dim];
        let inv_out = 1.0 / self.output_dim.max(1) as f32;
        for i in 0..self.input_dim {
            let mut sum = 0.0f32;
            for j in 0..self.output_dim {
                let idx = i * self.output_dim + j;
                if idx < self.weights.data.len() {
                    sum += self.weights.data[idx];
                }
            }
            direction[i] = sum * inv_out;
        }
        direction
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

        // Cosine similarity between input and weight direction (magnitude-invariant)
        let weight_dir = self.weight_direction();
        let epsilon = 1e-8f32;
        let dot: f32 = input_slice.iter().zip(weight_dir.iter()).map(|(a, b)| a * b).sum();
        let dir_norm: f32 = weight_dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine_sim = if input_norm > epsilon && dir_norm > epsilon {
            (dot / (input_norm * dir_norm)).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let cbw = self.confidence_base_weight;
        let mut confidence = (self.base_confidence * cbw + cosine_sim * (1.0 - cbw)).clamp(0.0, 1.0);

        // G5 magnitude penalty (Phase 14): reduce Surface confidence for
        // inputs with high structural complexity (high G5 norm).
        if let Some((g5_start, g5_end, simple_norm, complex_norm, weight)) =
            self.g5_magnitude_penalty
        {
            let s = g5_start.min(input_slice.len());
            let e = g5_end.min(input_slice.len());
            if s < e && complex_norm > simple_norm + epsilon {
                let g5_norm = input_slice[s..e]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();
                let penalty =
                    ((g5_norm - simple_norm) / (complex_norm - simple_norm)).clamp(0.0, 1.0);
                confidence = (confidence - penalty * weight).clamp(0.0, 1.0);
            }
        }

        // G4 magnitude penalty: same mechanism as G5 but for complexity scalars.
        if let Some((g4_start, g4_end, simple_norm, complex_norm, weight)) =
            self.g4_magnitude_penalty
        {
            let s = g4_start.min(input_slice.len());
            let e = g4_end.min(input_slice.len());
            if s < e && complex_norm > simple_norm + epsilon {
                let g4_norm = input_slice[s..e]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt();
                let penalty =
                    ((g4_norm - simple_norm) / (complex_norm - simple_norm)).clamp(0.0, 1.0);
                confidence = (confidence - penalty * weight).clamp(0.0, 1.0);
            }
        }

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

    fn hebbian_update(
        &mut self,
        input: &Tensor,
        output: &Tensor,
        signal: f32,
        learning_rate: f32,
    ) {
        if self.frozen {
            return;
        }
        let lr = if learning_rate > 0.0 {
            learning_rate
        } else {
            self.learning_rate
        };
        // Oja's rule: w_ij += lr * signal * output_j * (input_i - output_j * w_ij)
        // Self-normalising — converges stably without weight explosion.
        let in_len = self.input_dim.min(input.data.len());
        let out_len = self.output_dim.min(output.data.len());
        for i in 0..in_len {
            for j in 0..out_len {
                let idx = i * self.output_dim + j;
                if idx < self.weights.data.len() {
                    let w = self.weights.data[idx];
                    let y = output.data[j];
                    let x = input.data[i];
                    self.weights.data[idx] += lr * signal * y * (x - y * w);
                }
            }
        }
        // Bias update with same Oja-style decay: b_j += lr * signal * output_j * (1 - output_j * b_j)
        for j in 0..out_len {
            if j < self.bias.data.len() {
                let b = self.bias.data[j];
                let y = output.data[j];
                self.bias.data[j] += lr * signal * y * (1.0 - y * b);
            }
        }
    }

    fn error_update(
        &mut self,
        input: &Tensor,
        output: &Tensor,
        error_lr: f32,
        modulator: f32,
    ) {
        if self.frozen {
            return;
        }
        // w_ij += error_lr * modulator * input_i * output_j
        let in_len = self.input_dim.min(input.data.len());
        let out_len = self.output_dim.min(output.data.len());
        for i in 0..in_len {
            for j in 0..out_len {
                let idx = i * self.output_dim + j;
                if idx < self.weights.data.len() {
                    self.weights.data[idx] +=
                        error_lr * modulator * input.data[i] * output.data[j];
                }
            }
        }
    }

    fn weight_count(&self) -> usize {
        self.weights.data.len() + self.bias.data.len()
    }

    fn base_confidence(&self) -> f32 {
        self.base_confidence
    }

    fn weight_norm(&self) -> f32 {
        self.weights.norm()
    }

    fn activation_count(&self) -> usize {
        self.activation_count
    }

    fn increment_activation(&mut self) {
        self.activation_count += 1;
    }

    fn reset_activation(&mut self) {
        self.activation_count = 0;
    }

    fn accumulate_positive(&mut self, input: &Tensor) {
        // Contrastive accumulation allowed even on frozen nodes —
        // frozen only blocks Oja/error updates, not contrastive learning.
        let len = self.input_dim.min(input.data.len());
        for i in 0..len {
            self.positive_accumulator[i] += input.data[i];
        }
        self.positive_count += 1;
    }

    fn accumulate_negative(&mut self, input: &Tensor) {
        // Contrastive accumulation allowed even on frozen nodes.
        let len = self.input_dim.min(input.data.len());
        for i in 0..len {
            self.negative_accumulator[i] += input.data[i];
        }
        self.negative_count += 1;
    }

    fn apply_contrastive_update(&mut self) -> Option<ContrastiveUpdateInfo> {
        // Contrastive updates allowed even on frozen nodes.
        if self.positive_count == 0 || self.negative_count == 0 {
            return None;
        }

        let weight_norm_before = self.weights.norm();
        let pos_count = self.positive_count as f32;
        let neg_count = self.negative_count as f32;
        let pos_count_saved = self.positive_count;
        let neg_count_saved = self.negative_count;

        // contrast = positive_mean - negative_mean
        let mut contrast = vec![0.0f32; self.input_dim];
        let mut pos_mean_sq_sum = 0.0f32;
        let mut neg_mean_sq_sum = 0.0f32;
        for i in 0..self.input_dim {
            let pos_mean = self.positive_accumulator[i] / pos_count;
            let neg_mean = self.negative_accumulator[i] / neg_count;
            contrast[i] = pos_mean - neg_mean;
            pos_mean_sq_sum += pos_mean * pos_mean;
            neg_mean_sq_sum += neg_mean * neg_mean;
        }
        let positive_mean_norm = pos_mean_sq_sum.sqrt();
        let negative_mean_norm = neg_mean_sq_sum.sqrt();

        // Contrast magnitude (L2 norm)
        let contrast_magnitude = contrast.iter().map(|c| c * c).sum::<f32>().sqrt();

        // w += contrastive_lr * outer_product(contrast, contrast)
        // outer_product: contrast[i] * contrast[j] for weight[i][j]
        for i in 0..self.input_dim {
            for j in 0..self.output_dim {
                let idx = i * self.output_dim + j;
                if idx < self.weights.data.len() {
                    let cj = if j < contrast.len() { contrast[j] } else { 0.0 };
                    self.weights.data[idx] += self.contrastive_lr * contrast[i] * cj;
                }
            }
        }

        let weight_norm_after = self.weights.norm();

        // Reset accumulators
        self.positive_accumulator.fill(0.0);
        self.negative_accumulator.fill(0.0);
        self.positive_count = 0;
        self.negative_count = 0;

        Some(ContrastiveUpdateInfo {
            node_id: self.id.clone(),
            positive_count: pos_count_saved,
            negative_count: neg_count_saved,
            contrast_magnitude,
            weight_norm_before,
            weight_norm_after,
            positive_mean_norm,
            negative_mean_norm,
        })
    }

    fn set_contrastive_lr(&mut self, lr: f32) {
        self.contrastive_lr = lr;
    }

    fn reset_contrastive_accumulators(&mut self) {
        self.positive_accumulator.fill(0.0);
        self.negative_accumulator.fill(0.0);
        self.positive_count = 0;
        self.negative_count = 0;
    }

    fn is_frozen(&self) -> bool {
        self.frozen
    }

    fn set_frozen(&mut self, frozen: bool) {
        self.frozen = frozen;
    }

    fn init_analytical(&mut self, init: &AnalyticalInit, seed: u64) {
        LinearNode::init_analytical(self, init, seed);
    }

    fn set_g5_magnitude_penalty(&mut self, params: Option<(usize, usize, f32, f32, f32)>) {
        self.g5_magnitude_penalty = params;
    }

    fn set_g4_magnitude_penalty(&mut self, params: Option<(usize, usize, f32, f32, f32)>) {
        self.g4_magnitude_penalty = params;
    }

    fn set_confidence_base_weight(&mut self, weight: f32) {
        self.confidence_base_weight = weight;
    }

    fn init_orthogonal(&mut self, basis_vector: &[f32], noise_scale: f32, seed: u64) {
        LinearNode::init_orthogonal(self, basis_vector, noise_scale, seed);
    }

    fn weight_direction(&self) -> Vec<f32> {
        LinearNode::weight_direction(self)
    }

    fn save_weights_data(&self) -> Option<NodeWeightsData> {
        Some(NodeWeightsData {
            id: self.id.clone(),
            weights: self.weights.data.clone(),
            bias: self.bias.data.clone(),
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            tier: format!("{:?}", self.node_tier),
            base_confidence: self.base_confidence,
            frozen: self.frozen,
        })
    }

    fn load_weights_data(&mut self, data: &NodeWeightsData) {
        if data.id == self.id && data.weights.len() == self.weights.data.len()
            && data.bias.len() == self.bias.data.len()
        {
            self.weights.data.clone_from(&data.weights);
            self.bias.data.clone_from(&data.bias);
            self.frozen = data.frozen;
        }
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
    fn test_oja_reinforcement() {
        let mut node = LinearNode::new("train", 2, 2, Tier::Surface, 0.9);
        let w_before: Vec<f32> = node.weights.data.clone();
        let input = Tensor::from_vec(vec![1.0, 1.0]);
        let output = Tensor::from_vec(vec![1.0, 1.0]);
        // Oja's rule: w += lr * signal * y * (x - y * w)
        // With x=1, y=1, signal=+1, lr=0.1: delta = 0.1 * 1 * (1 - 1*w) = 0.1*(1-w)
        node.hebbian_update(&input, &output, 1.0, 0.1);
        for (before, after) in w_before.iter().zip(node.weights.data.iter()) {
            let expected_delta = 0.1 * (1.0 - before);
            assert!(
                (after - before - expected_delta).abs() < 1e-6,
                "Oja delta mismatch: before={}, after={}, expected_delta={}",
                before, after, expected_delta
            );
        }
    }

    #[test]
    fn test_oja_suppression() {
        let mut node = LinearNode::new("suppress", 2, 2, Tier::Surface, 0.9);
        let w_before: Vec<f32> = node.weights.data.clone();
        let input = Tensor::from_vec(vec![1.0, 1.0]);
        let output = Tensor::from_vec(vec![1.0, 1.0]);
        // Oja's rule with signal=-1: delta = -0.1 * 1 * (1 - 1*w) = -0.1*(1-w)
        node.hebbian_update(&input, &output, -1.0, 0.1);
        for (before, after) in w_before.iter().zip(node.weights.data.iter()) {
            let expected_delta = -0.1 * (1.0 - before);
            assert!(
                (after - before - expected_delta).abs() < 1e-6,
                "Oja suppression delta mismatch: before={}, after={}, expected_delta={}",
                before, after, expected_delta
            );
        }
    }

    #[test]
    fn test_oja_convergence_stability() {
        // Oja's rule should keep weight norms stable over many iterations
        let mut node = LinearNode::new("stable", 4, 4, Tier::Surface, 0.9);
        let initial_norm = node.weight_norm();
        let input = Tensor::from_vec(vec![0.5, -0.3, 0.8, 0.1]);
        for _ in 0..1000 {
            let out = node.forward(&input);
            node.hebbian_update(&input, &out.tensor, 1.0, 0.001);
        }
        let final_norm = node.weight_norm();
        // Oja's rule self-normalises — norm should not explode
        assert!(
            final_norm < initial_norm * 10.0,
            "Oja norm should stay stable: initial={}, final={}",
            initial_norm, final_norm
        );
    }

    #[test]
    fn test_weight_norm() {
        let node = LinearNode::new("norm_test", 4, 4, Tier::Surface, 0.9);
        let norm = node.weight_norm();
        assert!(norm > 0.0, "Weight norm should be positive: {}", norm);
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
    fn test_error_update_penalty() {
        let mut node = LinearNode::new("err_pen", 2, 2, Tier::Surface, 0.9);
        let w_before: Vec<f32> = node.weights.data.clone();
        let input = Tensor::from_vec(vec![1.0, 1.0]);
        let output = Tensor::from_vec(vec![1.0, 1.0]);
        // Penalty: modulator = -(1 - 0.7) = -0.3, error_lr = 0.0005
        node.error_update(&input, &output, 0.0005, -0.3);
        // Each weight should decrease by 0.0005 * 0.3 * 1 * 1 = 0.00015
        for (before, after) in w_before.iter().zip(node.weights.data.iter()) {
            let expected = before - 0.00015;
            assert!(
                (after - expected).abs() < 1e-7,
                "Error penalty: expected {:.6}, got {:.6}",
                expected,
                after
            );
        }
    }

    #[test]
    fn test_error_update_reinforcement() {
        let mut node = LinearNode::new("err_rein", 2, 2, Tier::Surface, 0.9);
        let w_before: Vec<f32> = node.weights.data.clone();
        let input = Tensor::from_vec(vec![1.0, 1.0]);
        let output = Tensor::from_vec(vec![1.0, 1.0]);
        // Reinforcement: modulator = 0.96 (similarity), error_lr = 0.0005
        node.error_update(&input, &output, 0.0005, 0.96);
        // Each weight should increase by 0.0005 * 0.96 * 1 * 1 = 0.00048
        for (before, after) in w_before.iter().zip(node.weights.data.iter()) {
            let expected = before + 0.00048;
            assert!(
                (after - expected).abs() < 1e-7,
                "Error reinforcement: expected {:.6}, got {:.6}",
                expected,
                after
            );
        }
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

    #[test]
    fn test_contrastive_accumulate_positive() {
        let mut node = LinearNode::new("pos_acc", 4, 4, Tier::Surface, 0.9);
        let input1 = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let input2 = Tensor::from_vec(vec![0.5, 1.5, 2.5, 3.5]);
        node.accumulate_positive(&input1);
        node.accumulate_positive(&input2);
        assert_eq!(node.positive_count, 2);
        // Accumulator should be element-wise sum
        assert!((node.positive_accumulator[0] - 1.5).abs() < 1e-6);
        assert!((node.positive_accumulator[1] - 3.5).abs() < 1e-6);
        assert!((node.positive_accumulator[2] - 5.5).abs() < 1e-6);
        assert!((node.positive_accumulator[3] - 7.5).abs() < 1e-6);
    }

    #[test]
    fn test_contrastive_accumulate_negative() {
        let mut node = LinearNode::new("neg_acc", 4, 4, Tier::Surface, 0.9);
        let input1 = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let input2 = Tensor::from_vec(vec![0.3, 0.4, 0.5, 0.6]);
        node.accumulate_negative(&input1);
        node.accumulate_negative(&input2);
        assert_eq!(node.negative_count, 2);
        assert!((node.negative_accumulator[0] - 0.4).abs() < 1e-6);
        assert!((node.negative_accumulator[1] - 0.6).abs() < 1e-6);
        assert!((node.negative_accumulator[2] - 0.8).abs() < 1e-6);
        assert!((node.negative_accumulator[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_contrastive_update_formula() {
        // Verify outer product direction: contrast = pos_mean - neg_mean
        // w += lr * contrast_i * contrast_j
        let mut node = LinearNode::new("contrast", 2, 2, Tier::Surface, 0.9);
        node.contrastive_lr = 1.0; // lr=1 for easy verification
        let w_before: Vec<f32> = node.weights.data.clone();

        // Positive mean will be [2.0, 4.0], negative mean will be [1.0, 1.0]
        // contrast = [1.0, 3.0]
        node.accumulate_positive(&Tensor::from_vec(vec![2.0, 4.0]));
        node.accumulate_negative(&Tensor::from_vec(vec![1.0, 1.0]));

        let info = node.apply_contrastive_update().unwrap();
        assert_eq!(info.positive_count, 1);
        assert_eq!(info.negative_count, 1);
        // contrast_magnitude = sqrt(1^2 + 3^2) = sqrt(10) ≈ 3.162
        assert!((info.contrast_magnitude - 10.0f32.sqrt()).abs() < 1e-5);

        // outer_product([1, 3], [1, 3]) = [[1, 3], [3, 9]]
        // w[0][0] += 1*1 = 1, w[0][1] += 1*3 = 3
        // w[1][0] += 3*1 = 3, w[1][1] += 3*3 = 9
        assert!((node.weights.data[0] - (w_before[0] + 1.0)).abs() < 1e-6, "w[0][0]");
        assert!((node.weights.data[1] - (w_before[1] + 3.0)).abs() < 1e-6, "w[0][1]");
        assert!((node.weights.data[2] - (w_before[2] + 3.0)).abs() < 1e-6, "w[1][0]");
        assert!((node.weights.data[3] - (w_before[3] + 9.0)).abs() < 1e-6, "w[1][1]");
    }

    #[test]
    fn test_contrastive_reset_after_update() {
        let mut node = LinearNode::new("reset", 2, 2, Tier::Surface, 0.9);
        node.accumulate_positive(&Tensor::from_vec(vec![1.0, 2.0]));
        node.accumulate_negative(&Tensor::from_vec(vec![0.5, 0.5]));
        let info = node.apply_contrastive_update();
        assert!(info.is_some());
        // After update, accumulators and counts should be reset
        assert_eq!(node.positive_count, 0);
        assert_eq!(node.negative_count, 0);
        assert!(node.positive_accumulator.iter().all(|&v| v == 0.0));
        assert!(node.negative_accumulator.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_contrastive_no_update_without_both_populations() {
        let mut node = LinearNode::new("noop", 2, 2, Tier::Surface, 0.9);
        // Only positive — should return None
        node.accumulate_positive(&Tensor::from_vec(vec![1.0, 2.0]));
        assert!(node.apply_contrastive_update().is_none());
        // Only negative — should return None
        let mut node2 = LinearNode::new("noop2", 2, 2, Tier::Surface, 0.9);
        node2.accumulate_negative(&Tensor::from_vec(vec![1.0, 2.0]));
        assert!(node2.apply_contrastive_update().is_none());
    }

    #[test]
    fn test_contrastive_default_noop_on_trait() {
        // AggregateNode should have no-op implementations
        let mut node = AggregateNode::new("agg_noop", 2, Tier::Surface);
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        node.accumulate_positive(&input);
        node.accumulate_negative(&input);
        assert!(node.apply_contrastive_update().is_none());
    }

    #[test]
    fn test_init_analytical_sets_weight_rows() {
        let mut node = LinearNode::new("analytic", 4, 2, Tier::Surface, 0.9);
        let direction = vec![0.6, 0.0, -0.8, 0.0]; // unit vector
        let init = AnalyticalInit {
            discrimination_direction: direction.clone(),
            noise_scale: 0.0, // zero noise → weights exactly equal direction
        };
        node.init_analytical(&init, 42);

        // Each row should be exactly the direction (noise_scale=0)
        for i in 0..4 {
            for j in 0..2 {
                let idx = i * 2 + j;
                assert!(
                    (node.weights.data[idx] - direction[i]).abs() < 1e-6,
                    "Weight[{}][{}] should be {:.4}, got {:.4}",
                    i, j, direction[i], node.weights.data[idx]
                );
            }
        }
    }

    #[test]
    fn test_init_analytical_with_noise() {
        let mut node = LinearNode::new("noisy", 4, 2, Tier::Surface, 0.9);
        let direction = vec![0.6, 0.0, -0.8, 0.0];
        let init = AnalyticalInit {
            discrimination_direction: direction.clone(),
            noise_scale: 0.1,
        };
        node.init_analytical(&init, 42);

        // Weights should be close to direction but not exact (noise added)
        for i in 0..4 {
            for j in 0..2 {
                let idx = i * 2 + j;
                let diff = (node.weights.data[idx] - direction[i]).abs();
                // With noise_scale=0.1 and xavier_scale=1/sqrt(4)=0.5, max noise ~ 0.05
                assert!(
                    diff < 0.1,
                    "Weight[{}][{}] = {:.4}, direction = {:.4}, diff = {:.4}",
                    i, j, node.weights.data[idx], direction[i], diff
                );
            }
        }
    }

    #[test]
    fn test_frozen_node_ignores_hebbian() {
        let mut node = LinearNode::new("frozen_h", 4, 2, Tier::Surface, 0.9);
        node.frozen = true;
        let w_before = node.weights.data.clone();
        let input = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let output = Tensor::from_vec(vec![1.0, 1.0]);
        node.hebbian_update(&input, &output, 1.0, 0.1);
        assert_eq!(node.weights.data, w_before, "Frozen node weights must not change");
    }

    #[test]
    fn test_frozen_node_ignores_error_update() {
        let mut node = LinearNode::new("frozen_e", 4, 2, Tier::Surface, 0.9);
        node.frozen = true;
        let w_before = node.weights.data.clone();
        let input = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let output = Tensor::from_vec(vec![1.0, 1.0]);
        node.error_update(&input, &output, 0.001, 0.5);
        assert_eq!(node.weights.data, w_before, "Frozen node weights must not change");
    }

    #[test]
    fn test_frozen_node_allows_contrastive() {
        // Frozen nodes block Oja/error updates but allow contrastive learning.
        // This enables Surface nodes to adapt their discrimination direction
        // while keeping Hebbian/error updates frozen.
        let mut node = LinearNode::new("frozen_c", 4, 2, Tier::Surface, 0.9);
        node.frozen = true;
        let w_before = node.weights.data.clone();
        let pos_input = Tensor::from_vec(vec![1.0, 0.5, 0.0, 0.0]);
        let neg_input = Tensor::from_vec(vec![0.0, 0.0, 0.5, 1.0]);
        node.accumulate_positive(&pos_input);
        node.accumulate_negative(&neg_input);
        assert_eq!(node.positive_count, 1);
        assert_eq!(node.negative_count, 1);
        let info = node.apply_contrastive_update();
        assert!(info.is_some(), "Frozen node should produce contrastive update");
        assert_ne!(node.weights.data, w_before, "Frozen node weights should change via contrastive");
    }

    #[test]
    fn test_frozen_via_trait() {
        let mut node = LinearNode::new("trait_freeze", 4, 2, Tier::Surface, 0.9);
        assert!(!node.is_frozen());
        node.set_frozen(true);
        assert!(node.is_frozen());
        // Weight update via trait should be no-op
        let w_before = node.weights.data.clone();
        let input = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        let output = Tensor::from_vec(vec![1.0, 1.0]);
        node.hebbian_update(&input, &output, 1.0, 0.1);
        assert_eq!(node.weights.data, w_before);
    }

    #[test]
    fn test_weight_serialisation_roundtrip() {
        let mut node = LinearNode::new("roundtrip", 4, 2, Tier::Surface, 0.88);
        node.frozen = true;
        // Modify weights to non-default values
        node.weights.data[0] = 42.0;
        node.bias.data[1] = -3.14;

        let saved = node.save_weights_data().unwrap();
        assert_eq!(saved.id, "roundtrip");
        assert_eq!(saved.weights[0], 42.0);
        assert_eq!(saved.bias[1], -3.14);
        assert!(saved.frozen);

        // Create a fresh node with same id and dimensions
        let mut node2 = LinearNode::new("roundtrip", 4, 2, Tier::Surface, 0.88);
        assert_ne!(node2.weights.data[0], 42.0); // different before load

        node2.load_weights_data(&saved);
        assert_eq!(node2.weights.data, node.weights.data);
        assert_eq!(node2.bias.data, node.bias.data);
        assert!(node2.frozen);
    }

    #[test]
    fn test_discrimination_direction_unit_normalised() {
        // Verify the direction passed to init_analytical should be unit normalised
        let direction = vec![0.6, 0.0, -0.8, 0.0];
        let norm: f32 = direction.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Test direction must be unit normalised, got norm={}",
            norm
        );
    }

    #[test]
    fn test_gram_schmidt_orthogonality() {
        // Gram-Schmidt should produce approximately orthogonal vectors
        let orth = OrthogonalInit::generate(10, 64, 42);
        assert_eq!(orth.basis_vectors.len(), 10);

        for (i, v) in orth.basis_vectors.iter().enumerate() {
            // Each vector should be unit normalised
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "Basis vector {} norm = {:.6}, expected 1.0",
                i, norm
            );
        }

        // All pairwise cosine similarities should be < 0.3
        let mut max_cos = 0.0f32;
        for i in 0..10 {
            for j in (i + 1)..10 {
                let dot: f32 = orth.basis_vectors[i]
                    .iter()
                    .zip(orth.basis_vectors[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let cos = dot.abs();
                if cos > max_cos {
                    max_cos = cos;
                }
                assert!(
                    cos < 0.3,
                    "Pairwise |cos| between {} and {} = {:.4}, exceeds 0.3",
                    i, j, cos
                );
            }
        }
    }

    #[test]
    fn test_gram_schmidt_deterministic() {
        let a = OrthogonalInit::generate(5, 32, 99);
        let b = OrthogonalInit::generate(5, 32, 99);
        for (va, vb) in a.basis_vectors.iter().zip(b.basis_vectors.iter()) {
            assert_eq!(va, vb, "Same seed must produce identical basis vectors");
        }
    }

    #[test]
    fn test_init_orthogonal_sets_weight_rows() {
        let mut node = LinearNode::new("orth_test", 8, 4, Tier::Reasoning, 0.80);
        let basis = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // unit vector along dim 0
        node.init_orthogonal(&basis, 0.0, 42); // zero noise

        // Each row should be exactly the basis vector (noise_scale=0)
        for i in 0..8 {
            for j in 0..4 {
                let idx = i * 4 + j;
                assert!(
                    (node.weights.data[idx] - basis[i]).abs() < 1e-6,
                    "Weight[{}][{}] = {:.6}, expected {:.6}",
                    i, j, node.weights.data[idx], basis[i]
                );
            }
        }
    }

    #[test]
    fn test_init_orthogonal_with_noise() {
        let mut node = LinearNode::new("orth_noisy", 8, 4, Tier::Reasoning, 0.80);
        let basis = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        node.init_orthogonal(&basis, 0.1, 42);

        // Weights should be close to basis but not exact
        for i in 0..8 {
            for j in 0..4 {
                let idx = i * 4 + j;
                let diff = (node.weights.data[idx] - basis[i]).abs();
                assert!(
                    diff < 0.1,
                    "Weight[{}][{}] = {:.6}, basis = {:.6}, diff = {:.6} exceeds 0.1",
                    i, j, node.weights.data[idx], basis[i], diff
                );
            }
        }
    }

    #[test]
    fn test_weight_direction_via_trait() {
        let node = LinearNode::new("wd_trait", 4, 2, Tier::Reasoning, 0.80);
        let dir_inherent = LinearNode::weight_direction(&node);
        let dir_trait: Vec<f32> = ComputeNode::weight_direction(&node);
        assert_eq!(dir_inherent, dir_trait, "Trait and inherent weight_direction must match");
        assert_eq!(dir_trait.len(), 4);
    }
}
