//! Semantic encoder — transformer with pooling for text embedding.
//!
//! Wraps a [`TransformerEncoder`](crate::transformer::TransformerEncoder) with
//! vocabulary lookup and mean pooling to produce fixed-size embeddings from
//! variable-length text.
//!
//! # Example
//!
//! ```
//! use axiom::vocab::Vocab;
//! use axiom::transformer::TransformerConfig;
//! use axiom::encoder::SemanticEncoder;
//!
//! let vocab = Vocab::build_from_corpus(
//!     &["hello world".to_string()], 100,
//! );
//! let config = TransformerConfig {
//!     vocab_size: vocab.max_size,
//!     hidden_dim: 128,
//!     num_heads: 4,
//!     num_layers: 2,
//!     ff_dim: 512,
//!     max_seq_len: 128,
//!     pooling: "mean".to_string(),
//!     activation: "gelu".to_string(),
//! };
//! let encoder = SemanticEncoder::new_with_config(vocab, config);
//! let embedding = encoder.encode("hello world");
//! assert_eq!(embedding.data.len(), 128);
//! ```

pub use crate::semantic::encoder::{SemanticEncoder, SemanticEncoderData};
