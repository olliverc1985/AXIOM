//! Structural text features — hand-crafted syntactic encoder.
//!
//! Deterministic feature extraction with no learnable parameters. Produces
//! a 128-dimensional feature vector from raw text including position-weighted
//! token embeddings, type-token ratio, average token length, punctuation
//! density, and normalised token count.
//!
//! # Example
//!
//! ```
//! use axiom::features::{StructuralEncoder, Tokeniser};
//!
//! let tokeniser = Tokeniser::default_tokeniser();
//! let encoder = StructuralEncoder::new(128, tokeniser);
//! let features = encoder.encode_text_readonly("What is the capital of France?");
//! assert_eq!(features.data.len(), 128);
//! ```

pub use crate::input::encoder::Encoder;
pub use crate::input::tokeniser::Tokeniser;

/// Alias for [`Encoder`] to distinguish from [`SemanticEncoder`](crate::encoder::SemanticEncoder).
pub type StructuralEncoder = Encoder;
