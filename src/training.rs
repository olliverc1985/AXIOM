//! Training loops and loss functions.
//!
//! Provides STS (Semantic Textual Similarity) training for the semantic encoder
//! using cosine similarity loss. Supports MSE and contrastive loss functions,
//! word dropout augmentation, and GPU-accelerated training on macOS.
//!
//! For classification fine-tuning with cross-entropy loss, see the
//! [LLM routing example](https://github.com/olliverc1985/AXIOM/tree/main/examples/llm-routing).

pub use crate::semantic::train::{
    evaluate, load_sts_data, pearson, train, train_with_augmentation, word_dropout, StsExample,
};

#[cfg(feature = "gpu")]
pub use crate::semantic::train::{train_gpu, train_with_augmentation_gpu};
