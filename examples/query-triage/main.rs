//! Query Triage — route queries by complexity before calling an expensive API.
//!
//! Classifies incoming queries into complexity tiers so you can route them
//! to appropriate backends:
//! - Simple: answer from cache or FAQ database
//! - Medium: standard API call
//! - Complex: premium API with extended context
//!
//! This is the core AXIOM use case, generalised beyond LLM routing to any
//! tiered service architecture.

use axiom::classifier::ClassificationHead;
use axiom::encoder::SemanticEncoder;
use axiom::features::{StructuralEncoder, Tokeniser};
use axiom::transformer::TransformerConfig;
use axiom::vocab::Vocab;

fn main() {
    println!("AXIOM Query Triage Example");
    println!("──────────────────────────");
    println!();

    // Build untrained encoder (load trained weights in production)
    let vocab = Vocab::build_from_corpus(&["sample query for triage".to_string()], 1000);
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

    let queries = [
        "What are your opening hours?",
        "I need to dispute a charge on my account from last month",
        "Can you explain the tax implications of converting my traditional IRA to a Roth IRA given that I also have a 401k rollover from a previous employer and I'm currently in the 32% tax bracket?",
        "How do I reset my password?",
    ];

    let tiers = ["Simple", "Medium", "Complex"];

    for query in &queries {
        let sf = structural.encode_text_readonly(query);
        let se = semantic.encode(query);
        let mut fused = Vec::with_capacity(256);
        fused.extend_from_slice(&sf.data);
        fused.extend_from_slice(&se.data);
        let (class, conf) = head.classify(&fused);

        println!(
            "  {:>7} ({:.0}%) | {}",
            tiers[class],
            conf[class] * 100.0,
            &query[..query.len().min(70)],
        );
    }

    println!();
    println!("Note: untrained skeleton. Train on your query data for real triage.");
}
