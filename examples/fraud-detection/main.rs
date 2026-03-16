//! Fraud Detection — classify messages as Clean, Suspicious, or Blocked.
//!
//! This skeleton demonstrates how to use AXIOM's transformer classifier for
//! fraud detection. The same dual-encoder architecture (structural features +
//! semantic transformer) classifies messages by risk level.
//!
//! To make this production-ready, you would:
//! 1. Collect labelled examples of clean, suspicious, and blocked messages
//! 2. Save them as JSON in the same format as data/fusion_training_corpus.json
//! 3. Train with: cargo run --example train-axiom -- --tier-data your_data.json
//! 4. Load the trained weights here instead of the placeholder

use axiom::classifier::ClassificationHead;
use axiom::encoder::SemanticEncoder;
use axiom::features::{StructuralEncoder, Tokeniser};
use axiom::transformer::TransformerConfig;
use axiom::vocab::Vocab;

fn main() {
    println!("AXIOM Fraud Detection Example");
    println!("─────────────────────────────");
    println!();

    // Step 1: Build encoder from scratch (in production, load trained weights)
    let corpus = vec![
        "please send payment to this account".to_string(),
        "your order has been shipped".to_string(),
        "URGENT: verify your identity now or lose access".to_string(),
        "meeting scheduled for Tuesday at 3pm".to_string(),
    ];
    let vocab = Vocab::build_from_corpus(&corpus, 1000);

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
    let structural = StructuralEncoder::new(128, tokeniser);

    // Step 2: Build classifier (3 classes: Clean, Suspicious, Blocked)
    let head = ClassificationHead::new(256, 42);

    // Step 3: Classify messages
    let messages = [
        "Hey, can we reschedule the meeting to Thursday?",
        "URGENT: Your account will be SUSPENDED unless you click here NOW",
        "Invoice #4521 attached for your review",
        "Congratulations!!! You won $1,000,000!!! Send your bank details to claim",
    ];

    let class_names = ["Clean", "Suspicious", "Blocked"];

    for msg in &messages {
        let structural_features = structural.encode_text_readonly(msg);
        let semantic_embedding = semantic.encode(msg);

        let mut fused = Vec::with_capacity(256);
        fused.extend_from_slice(&structural_features.data);
        fused.extend_from_slice(&semantic_embedding.data);

        let (class, confidence) = head.classify(&fused);

        println!(
            "  {:>10} ({:.0}%) | {}",
            class_names[class],
            confidence[class] * 100.0,
            &msg[..msg.len().min(60)],
        );
    }

    println!();
    println!("Note: predictions are random — this is an untrained skeleton.");
    println!("Train on labelled fraud data to get meaningful results.");
}
