//! Content Moderation — triage user content by severity.
//!
//! Classifies user-generated content into severity levels for moderation queues:
//! - Allow: content passes automated checks
//! - Review: flagged for human review
//! - Block: immediately blocked
//!
//! Uses AXIOM's structural encoder to detect high-risk linguistic patterns
//! (ALL CAPS, excessive punctuation, URL density) combined with the semantic
//! encoder for meaning-level risk assessment.

use axiom::classifier::ClassificationHead;
use axiom::encoder::SemanticEncoder;
use axiom::features::{StructuralEncoder, Tokeniser};
use axiom::transformer::TransformerConfig;
use axiom::vocab::Vocab;

fn main() {
    println!("AXIOM Content Moderation Example");
    println!("────────────────────────────────");
    println!();

    // Build untrained encoder (load trained weights in production)
    let vocab = Vocab::build_from_corpus(&["sample content for moderation".to_string()], 1000);
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
    let structural = StructuralEncoder::new(128, Tokeniser::default_tokeniser());
    let head = ClassificationHead::new(256, 42);

    let posts = [
        "Great product, would recommend to friends!",
        "This is absolutely TERRIBLE and you should be ASHAMED",
        "Check out my profile for special offers!!!",
        "Thanks for the help, that solved my problem",
    ];

    let levels = ["Allow", "Review", "Block"];

    for post in &posts {
        let sf = structural.encode_text_readonly(post);
        let se = semantic.encode(post);
        let mut fused = Vec::with_capacity(256);
        fused.extend_from_slice(&sf.data);
        fused.extend_from_slice(&se.data);
        let (class, conf) = head.classify(&fused);

        println!(
            "  {:>6} ({:.0}%) | {}",
            levels[class],
            conf[class] * 100.0,
            post
        );
    }

    println!();
    println!("Note: untrained skeleton. Train on moderation-labelled data for real use.");
}
