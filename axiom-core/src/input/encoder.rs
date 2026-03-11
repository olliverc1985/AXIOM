//! Structural encoder — converts token sequences to fixed-dimension tensors
//! with position-weighted content and explicit syntactic complexity features.
//!
//! Replaces the Phase 3 bag-of-words encoder. Preserves word-order information
//! through position weighting. Extracts four syntactic features — token count,
//! type-token ratio, average token length, and punctuation density — appended
//! as explicit dimensions that are NOT normalised.
//!
//! **No normalisation on the sum vector.** Magnitude encodes sentence complexity:
//! short simple sentences have lower magnitude, long complex sentences have higher.
//! Only per-token normalisation (unit vectors) is applied before summation.
//! The final output preserves magnitude differences across sentences.

use crate::input::tokeniser::Tokeniser;
use crate::Tensor;
use std::collections::HashSet;

/// Maximum sentence length for normalising the token-count feature.
const MAX_SENTENCE_TOKENS: f32 = 30.0;

/// Maximum token length for normalising the average token length feature.
const MAX_TOKEN_LENGTH: f32 = 15.0;

/// Number of syntactic features appended after the content dimensions.
const SYNTACTIC_FEATURES: usize = 4;

/// Structural encoder producing fixed-dimension tensors from text.
///
/// The output vector has two parts:
/// - **Content** (first `output_dim - 4` dimensions): position-weighted token sum
///   with no normalisation. Magnitude grows with sentence length/complexity.
///   Each token vector is unit-normalised before weighted summation.
/// - **Features** (last 4 dimensions): token count, type-token ratio,
///   average token length, punctuation density. NOT normalised — carry absolute signal.
pub struct Encoder {
    /// Output dimension (must be > 4 to leave room for syntactic features).
    pub output_dim: usize,
    /// Reference to tokeniser for end-to-end encoding.
    pub tokeniser: Tokeniser,
}

impl Encoder {
    /// Create a new structural encoder with the given output dimension and tokeniser.
    ///
    /// # Panics
    /// Panics if `output_dim` is not greater than `SYNTACTIC_FEATURES` (4).
    pub fn new(output_dim: usize, tokeniser: Tokeniser) -> Self {
        assert!(
            output_dim > SYNTACTIC_FEATURES,
            "output_dim {} must be > {} to leave room for syntactic features",
            output_dim,
            SYNTACTIC_FEATURES
        );
        Self {
            output_dim,
            tokeniser,
        }
    }

    /// Number of content dimensions (output_dim minus syntactic feature slots).
    fn content_dim(&self) -> usize {
        self.output_dim - SYNTACTIC_FEATURES
    }

    /// Generate a deterministic pseudo-random unit vector for a given token ID.
    ///
    /// Uses the same PRNG as Phase 3 for weight continuity.
    fn token_vector(&self, token_id: usize) -> Vec<f32> {
        let content_dim = self.content_dim();
        let mut vec = vec![0.0f32; content_dim];
        let mut seed = ((token_id + 1) as u32).wrapping_mul(2654435761).wrapping_add(1);
        for v in vec.iter_mut() {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            *v = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
        }
        // Normalise to unit length.
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in vec.iter_mut() {
                *v /= norm;
            }
        }
        vec
    }

    /// Encode token IDs and raw text into a structural tensor.
    ///
    /// 1. Position-weighted sum of token vectors (weight = 1/(1+pos))
    ///    — no normalisation on the sum; magnitude preserved
    /// 2. Extract and append 4 syntactic features from raw text
    fn encode_with_features(&self, token_ids: &[usize], text: &str) -> Tensor {
        let content_dim = self.content_dim();
        let mut sum = vec![0.0f32; content_dim];

        // Position-weighted token sum: early tokens contribute more signal.
        // Each token vector is unit-normalised (in token_vector), but the
        // sum is NOT normalised — magnitude grows with sentence length/complexity.
        for (pos, &id) in token_ids.iter().enumerate() {
            let weight = 1.0 / (1.0 + pos as f32);
            let tv = self.token_vector(id);
            for (s, v) in sum.iter_mut().zip(tv.iter()) {
                *s += v * weight;
            }
        }

        // Extract syntactic features from raw text.
        let token_strings = Tokeniser::split(text);
        let token_count = token_strings.len();

        // Feature 1: Token count normalised to [0, 1] against max sentence length.
        let token_count_norm = (token_count as f32 / MAX_SENTENCE_TOKENS).min(1.0);

        // Feature 2: Type-token ratio (unique tokens / total tokens).
        // High ratio = lexically diverse.
        let unique: HashSet<&str> = token_strings.iter().map(|s| s.as_str()).collect();
        let ttr = if token_count > 0 {
            unique.len() as f32 / token_count as f32
        } else {
            0.0
        };

        // Feature 3: Average token length normalised to [0, 1] against max token length.
        let avg_token_len = if token_count > 0 {
            let total_len: usize = token_strings.iter().map(|t| t.len()).sum();
            (total_len as f32 / token_count as f32 / MAX_TOKEN_LENGTH).min(1.0)
        } else {
            0.0
        };

        // Feature 4: Punctuation density (punctuation chars / total tokens).
        let punct_count = text.chars().filter(|c| c.is_ascii_punctuation()).count();
        let punct_density = if token_count > 0 {
            (punct_count as f32 / token_count as f32).min(1.0)
        } else {
            0.0
        };

        // Append syntactic features — NOT normalised, carry absolute signal.
        sum.push(token_count_norm);
        sum.push(ttr);
        sum.push(avg_token_len);
        sum.push(punct_density);

        Tensor::from_vec(sum)
    }

    /// End-to-end: tokenise text and encode to tensor.
    ///
    /// Modifies the tokeniser vocabulary (adds new tokens).
    pub fn encode_text(&mut self, text: &str) -> Tensor {
        let ids = self.tokeniser.tokenise(text);
        self.encode_with_features(&ids, text)
    }

    /// End-to-end encoding without modifying vocabulary.
    pub fn encode_text_readonly(&self, text: &str) -> Tensor {
        let ids = self.tokeniser.tokenise_readonly(text);
        self.encode_with_features(&ids, text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_produces_correct_dim() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(16, tok);
        let t = enc.encode_text("hello world");
        assert_eq!(t.shape, vec![16]);
    }

    #[test]
    fn test_position_weighting_order_matters() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(64, tok);
        let t1 = enc.encode_text("cat sat mat");
        let t2 = enc.encode_text("mat sat cat");
        // Different word order → different content vectors (position weighting)
        let content_dim = 60;
        assert_ne!(t1.data[..content_dim], t2.data[..content_dim]);
    }

    #[test]
    fn test_syntactic_feature_extraction() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(64, tok);
        let t = enc.encode_text("hello world");
        let features = &t.data[60..];

        // Feature 0: Token count normalised: 2 / 30
        assert!(
            (features[0] - 2.0 / 30.0).abs() < 0.01,
            "token_count_norm: expected {:.4}, got {:.4}",
            2.0 / 30.0,
            features[0]
        );

        // Feature 1: TTR: 2 unique / 2 total = 1.0
        assert!(
            (features[1] - 1.0).abs() < 0.01,
            "TTR: expected 1.0, got {:.4}",
            features[1]
        );

        // Feature 2: Avg token length: (5+5)/2 = 5.0 → 5.0/15.0
        assert!(
            (features[2] - 5.0 / 15.0).abs() < 0.01,
            "avg_token_len: expected {:.4}, got {:.4}",
            5.0 / 15.0,
            features[2]
        );

        // Feature 3: Punctuation: 0
        assert!(
            features[3].abs() < 0.01,
            "punct_density: expected 0.0, got {:.4}",
            features[3]
        );
    }

    #[test]
    fn test_content_magnitude_increases_with_length() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(64, tok);
        let short = enc.encode_text("hello world");
        let long = enc.encode_text("the quick brown fox jumps over the lazy dog and runs through the fields");
        // No normalisation on sum — more tokens contribute more magnitude
        let short_norm: f32 = short.data[..60].iter().map(|x| x * x).sum::<f32>().sqrt();
        let long_norm: f32 = long.data[..60].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            long_norm > short_norm,
            "Long content norm {} should exceed short content norm {}",
            long_norm,
            short_norm
        );
    }

    #[test]
    fn test_long_higher_magnitude_than_short() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(32, tok);
        let short = enc.encode_text("hello");
        let long = enc.encode_text("the quick brown fox jumps over the lazy dog");
        // Magnitude grows with token count — longer sentences have higher total magnitude
        assert!(
            long.norm() > short.norm(),
            "Long text norm {} should exceed short text norm {}",
            long.norm(),
            short.norm()
        );
    }

    #[test]
    fn test_complexity_features_increase_with_length() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(64, tok);
        let simple = enc.encode_text("the cat sat");
        let complex = enc.encode_text(
            "the recursive nature of self-referential systems creates emergent properties",
        );
        // Token count feature should be higher for complex sentence
        assert!(
            complex.data[60] > simple.data[60],
            "Token count: complex={} should exceed simple={}",
            complex.data[60],
            simple.data[60]
        );
    }

    #[test]
    fn test_punctuation_density_feature() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(64, tok);
        let no_punct = enc.encode_text("hello world");
        let with_punct = enc.encode_text("hello, world! how are you?");
        // Punctuation density (last feature) should be higher
        assert!(
            with_punct.data[63] > no_punct.data[63],
            "Punctuation density: with_punct={} should exceed no_punct={}",
            with_punct.data[63],
            no_punct.data[63]
        );
    }

    #[test]
    fn test_different_texts_different_vectors() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(16, tok);
        let t1 = enc.encode_text("hello world");
        let t2 = enc.encode_text("goodbye universe");
        assert_ne!(t1.data, t2.data);
    }

    #[test]
    fn test_similar_texts_higher_similarity() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(64, tok);
        let t1 = enc.encode_text("the cat sat on the mat");
        let t2 = enc.encode_text("the cat sat on the rug");
        let t3 = enc.encode_text("quantum mechanics describes wave particle duality");
        let sim_close = t1.cosine_similarity(&t2);
        let sim_far = t1.cosine_similarity(&t3);
        assert!(
            sim_close > sim_far,
            "Similar texts should have higher cosine sim: close={} far={}",
            sim_close,
            sim_far
        );
    }

    #[test]
    fn test_empty_text() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(8, tok);
        let t = enc.encode_text("");
        assert_eq!(t.shape, vec![8]);
        // All syntactic features should be 0.0 for empty input
        for i in 4..8 {
            assert_eq!(t.data[i], 0.0, "Feature at dim {} should be 0.0", i);
        }
    }
}
