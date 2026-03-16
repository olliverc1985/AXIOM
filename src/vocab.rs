//! Vocabulary and tokenisation.
//!
//! Word-level vocabulary with configurable size. Includes PAD and UNK tokens.
//! Built from a text corpus with frequency-based selection.
//!
//! # Example
//!
//! ```
//! use axiom::vocab::Vocab;
//!
//! let corpus = vec!["hello world".to_string(), "hello rust".to_string()];
//! let vocab = Vocab::build_from_corpus(&corpus, 100);
//! let ids = vocab.tokenize("hello world");
//! assert!(!ids.is_empty());
//! ```

pub use crate::semantic::vocab::{Vocab, PAD_TOKEN, UNK_TOKEN};
