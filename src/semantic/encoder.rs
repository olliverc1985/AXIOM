use super::transformer::{SemanticWeightsData, TransformerConfig, TransformerEncoder};
use super::vocab::Vocab;
use crate::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEncoderData {
    pub vocab: Vocab,
    pub weights: SemanticWeightsData,
}

pub struct SemanticEncoder {
    pub transformer: TransformerEncoder,
    pub vocab: Vocab,
}

impl SemanticEncoder {
    pub fn new(vocab: Vocab) -> Self {
        let config = TransformerConfig {
            vocab_size: vocab.max_size,
            ..TransformerConfig::default()
        };
        Self {
            transformer: TransformerEncoder::new(config),
            vocab,
        }
    }

    pub fn new_with_config(vocab: Vocab, config: TransformerConfig) -> Self {
        Self {
            transformer: TransformerEncoder::new(config),
            vocab,
        }
    }

    pub fn from_data(data: &SemanticEncoderData) -> Self {
        Self {
            transformer: TransformerEncoder::from_weights(&data.weights),
            vocab: data.vocab.clone(),
        }
    }

    /// Encode text to a 128-dim embedding.
    pub fn encode(&self, text: &str) -> Tensor {
        let ids = self.vocab.tokenize(text);
        if ids.is_empty() {
            return Tensor::from_vec(vec![0.0; self.transformer.config.hidden_dim]);
        }
        let (emb, _) = self.transformer.forward(&ids);
        Tensor::from_vec(emb)
    }

    /// Encode with forward cache (for training).
    pub fn encode_with_cache(&self, text: &str) -> (Vec<f32>, super::transformer::ForwardCache) {
        let ids = self.vocab.tokenize(text);
        if ids.is_empty() {
            let h = self.transformer.config.hidden_dim;
            let dummy_cache = super::transformer::ForwardCache {
                token_ids: vec![0],
                seq_len: 1,
                embedded: vec![0.0; h],
                layer_caches: Vec::new(),
                pre_final_ln: vec![0.0; h],
                final_ln_xhat: vec![0.0; h],
                final_ln_inv_std: vec![1.0],
                final_output: vec![0.0; h],
                pooling_max_indices: Vec::new(),
            };
            return (vec![0.0; h], dummy_cache);
        }
        self.transformer.forward(&ids)
    }

    /// GPU-accelerated encode with forward cache (for training).
    #[cfg(feature = "gpu")]
    pub fn encode_with_cache_gpu(
        &self,
        text: &str,
        gpu: &crate::gpu::GpuContext,
    ) -> (Vec<f32>, super::transformer::ForwardCache) {
        let ids = self.vocab.tokenize(text);
        if ids.is_empty() {
            let h = self.transformer.config.hidden_dim;
            let dummy_cache = super::transformer::ForwardCache {
                token_ids: vec![0],
                seq_len: 1,
                embedded: vec![0.0; h],
                layer_caches: Vec::new(),
                pre_final_ln: vec![0.0; h],
                final_ln_xhat: vec![0.0; h],
                final_ln_inv_std: vec![1.0],
                final_output: vec![0.0; h],
                pooling_max_indices: Vec::new(),
            };
            return (vec![0.0; h], dummy_cache);
        }
        self.transformer.forward_gpu(&ids, gpu)
    }

    pub fn param_count(&self) -> usize {
        self.transformer.param_count()
    }

    pub fn save(&self, path: &str) {
        let data = SemanticEncoderData {
            vocab: self.vocab.clone(),
            weights: self.transformer.save_weights(),
        };
        let json = serde_json::to_string(&data).unwrap();
        std::fs::write(path, json).unwrap();
    }

    pub fn load(path: &str) -> Self {
        let json = std::fs::read_to_string(path).unwrap();
        let data: SemanticEncoderData = serde_json::from_str(&json).unwrap();
        Self::from_data(&data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_output_dim() {
        let sentences = vec!["the cat sat".to_string(), "dogs run fast".to_string()];
        let vocab = Vocab::build_from_corpus(&sentences, 100);
        let encoder = SemanticEncoder::new(vocab);
        let emb = encoder.encode("the cat sat on the mat");
        assert_eq!(emb.shape, vec![128]);
        assert_eq!(emb.data.len(), 128);
    }

    #[test]
    fn test_encode_empty_text() {
        let vocab = Vocab::new(100);
        let encoder = SemanticEncoder::new(vocab);
        let emb = encoder.encode("");
        assert_eq!(emb.data.len(), 128);
    }

    #[test]
    fn test_same_text_same_embedding() {
        let sentences = vec!["hello world".to_string()];
        let vocab = Vocab::build_from_corpus(&sentences, 100);
        let encoder = SemanticEncoder::new(vocab);
        let e1 = encoder.encode("hello world");
        let e2 = encoder.encode("hello world");
        assert_eq!(e1.data, e2.data);
    }
}
