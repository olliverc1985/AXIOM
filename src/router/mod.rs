//! Unified AXIOM router — always-fuse architecture with end-to-end trained encoder.
//!
//! Routes queries to Surface/Reasoning/Deep tiers using a structural encoder
//! (128-dim hand-crafted features) and a semantic encoder (128-dim transformer),
//! concatenated into 256-dim and classified via a linear head.

use crate::input::{Encoder, Tokeniser};
use crate::semantic::encoder::SemanticEncoder;
use crate::tiers::Tier;
use serde::{Deserialize, Serialize};

/// Classification head: linear 256 → 3 with softmax.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationHead {
    pub weights: Vec<f32>, // [input_dim * 3]
    pub biases: Vec<f32>,  // [3]
    pub input_dim: usize,
}

impl ClassificationHead {
    pub fn new(input_dim: usize, seed: u64) -> Self {
        let scale = (2.0 / (input_dim + 3) as f32).sqrt();
        let mut weights = vec![0.0f32; input_dim * 3];
        let mut s = seed;
        for w in weights.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u1 = (s >> 33) as f32 / (1u64 << 31) as f32;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u2 = (s >> 33) as f32 / (1u64 << 31) as f32;
            let z = (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            *w = z * scale;
        }
        Self {
            weights,
            biases: vec![0.0; 3],
            input_dim,
        }
    }

    /// Forward pass: input → logits.
    pub fn logits(&self, input: &[f32]) -> [f32; 3] {
        let h = self.input_dim;
        let mut out = [self.biases[0], self.biases[1], self.biases[2]];
        for i in 0..h {
            let x = input[i];
            out[0] += x * self.weights[i * 3];
            out[1] += x * self.weights[i * 3 + 1];
            out[2] += x * self.weights[i * 3 + 2];
        }
        out
    }

    /// Softmax of 3-class logits.
    pub fn softmax(logits: &[f32; 3]) -> [f32; 3] {
        let max = logits[0].max(logits[1]).max(logits[2]);
        let e = [
            (logits[0] - max).exp(),
            (logits[1] - max).exp(),
            (logits[2] - max).exp(),
        ];
        let sum = e[0] + e[1] + e[2];
        [e[0] / sum, e[1] / sum, e[2] / sum]
    }

    /// Classify input → (predicted tier index, confidence scores).
    pub fn classify(&self, input: &[f32]) -> (usize, [f32; 3]) {
        let logits = self.logits(input);
        let probs = Self::softmax(&logits);
        let pred = if probs[1] > probs[0] && probs[1] > probs[2] {
            1
        } else if probs[2] > probs[0] {
            2
        } else {
            0
        };
        (pred, probs)
    }
}

/// Result of routing a query.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Predicted tier.
    pub tier: Tier,
    /// Confidence scores: [Surface, Reasoning, Deep].
    pub confidence: [f32; 3],
    /// Structural encoder features (128-dim).
    pub structural_features: Vec<f32>,
    /// Semantic encoder embedding (128-dim).
    pub semantic_embedding: Vec<f32>,
    /// Routing latency in microseconds.
    pub latency_us: u64,
}

/// Saved weights for the full AXIOM router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxiomRouterWeights {
    pub encoder: crate::semantic::encoder::SemanticEncoderData,
    pub head: ClassificationHead,
}

/// Unified AXIOM router.
pub struct AxiomRouter {
    pub structural_encoder: Encoder,
    pub semantic_encoder: SemanticEncoder,
    pub head: ClassificationHead,
}

impl AxiomRouter {
    /// Create a new router with a trained semantic encoder and classification head.
    pub fn new(
        structural_encoder: Encoder,
        semantic_encoder: SemanticEncoder,
        head: ClassificationHead,
    ) -> Self {
        Self {
            structural_encoder,
            semantic_encoder,
            head,
        }
    }

    /// Load router from saved weights file.
    pub fn load(weights_path: &str) -> Self {
        let json = std::fs::read_to_string(weights_path).expect("Failed to read router weights");
        let data: AxiomRouterWeights =
            serde_json::from_str(&json).expect("Failed to parse router weights");

        let semantic_encoder = SemanticEncoder::from_data(&data.encoder);
        let tokeniser = Tokeniser::default_tokeniser();
        let structural_encoder = Encoder::new(128, tokeniser);

        Self {
            structural_encoder,
            semantic_encoder,
            head: data.head,
        }
    }

    /// Save router weights to a JSON file.
    pub fn save(&self, path: &str) {
        let data = AxiomRouterWeights {
            encoder: crate::semantic::encoder::SemanticEncoderData {
                vocab: self.semantic_encoder.vocab.clone(),
                weights: self.semantic_encoder.transformer.save_weights(),
            },
            head: self.head.clone(),
        };
        let json = serde_json::to_string(&data).expect("Failed to serialise weights");
        std::fs::write(path, json).expect("Failed to write weights");
    }

    /// Route a query to a tier.
    pub fn route(&self, query: &str) -> RoutingDecision {
        let start = std::time::Instant::now();

        // Structural encoder: 128-dim hand-crafted features
        let structural = self.structural_encoder.encode_text_readonly(query);

        // Semantic encoder: 128-dim transformer embedding
        let semantic = self.semantic_encoder.encode(query);

        // Concatenate → 256-dim
        let mut fused = Vec::with_capacity(256);
        fused.extend_from_slice(&structural.data);
        fused.extend_from_slice(&semantic.data);

        // Classify
        let (pred_idx, confidence) = self.head.classify(&fused);

        let tier = match pred_idx {
            0 => Tier::Surface,
            1 => Tier::Reasoning,
            _ => Tier::Deep,
        };

        let latency_us = start.elapsed().as_micros() as u64;

        RoutingDecision {
            tier,
            confidence,
            structural_features: structural.data,
            semantic_embedding: semantic.data,
            latency_us,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::transformer::TransformerConfig;
    use crate::semantic::vocab::Vocab;

    fn make_test_router() -> AxiomRouter {
        let vocab =
            Vocab::build_from_corpus(&["hello world".to_string(), "test query".to_string()], 100);
        let config = TransformerConfig {
            vocab_size: vocab.max_size,
            hidden_dim: 128,
            num_heads: 4,
            num_layers: 2,
            ff_dim: 512,
            max_seq_len: 128,
            pooling: "mean".to_string(),
            activation: "gelu".to_string(),
        };
        let semantic = SemanticEncoder::new_with_config(vocab, config);
        let tokeniser = Tokeniser::default_tokeniser();
        let structural = Encoder::new(128, tokeniser);
        let head = ClassificationHead::new(256, 42);
        AxiomRouter::new(structural, semantic, head)
    }

    #[test]
    fn test_route_returns_valid_tier() {
        let router = make_test_router();
        let decision = router.route("what is the capital of France");
        assert!(matches!(
            decision.tier,
            Tier::Surface | Tier::Reasoning | Tier::Deep
        ));
    }

    #[test]
    fn test_confidence_sums_to_one() {
        let router = make_test_router();
        let decision = router.route("hello world");
        let sum: f32 = decision.confidence.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "confidence sum = {}", sum);
    }

    #[test]
    fn test_route_deterministic() {
        let router = make_test_router();
        let d1 = router.route("test query");
        let d2 = router.route("test query");
        assert_eq!(d1.tier, d2.tier);
        assert_eq!(d1.confidence, d2.confidence);
    }

    #[test]
    fn test_structural_and_semantic_dims() {
        let router = make_test_router();
        let decision = router.route("hello");
        assert_eq!(decision.structural_features.len(), 128);
        assert_eq!(decision.semantic_embedding.len(), 128);
    }

    #[test]
    fn test_classification_head_softmax() {
        let probs = ClassificationHead::softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_weight_save_load_roundtrip() {
        let router = make_test_router();
        let tmp = "/tmp/axiom_test_weights.json";
        router.save(tmp);
        let loaded = AxiomRouter::load(tmp);
        let d1 = router.route("test roundtrip");
        let d2 = loaded.route("test roundtrip");
        assert_eq!(d1.tier, d2.tier);
        for i in 0..3 {
            assert!((d1.confidence[i] - d2.confidence[i]).abs() < 1e-5);
        }
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_empty_query() {
        let router = make_test_router();
        let decision = router.route("");
        assert!(matches!(
            decision.tier,
            Tier::Surface | Tier::Reasoning | Tier::Deep
        ));
    }

    #[test]
    fn test_latency_under_3ms() {
        let router = make_test_router();
        let decision = router.route("what is the meaning of life");
        assert!(
            decision.latency_us < 3000,
            "latency={}us",
            decision.latency_us
        );
    }
}
