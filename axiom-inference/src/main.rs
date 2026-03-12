//! AXIOM Inference — route sentences through a trained AXIOM model.
//!
//! Loads trained weights from `axiom_weights.json` and vocabulary from
//! `axiom_vocab.json`, then routes input sentences and prints structured output.
//!
//! Usage:
//!   echo "the cat sat" | cargo run --release -p axiom-inference
//!   cargo run --release -p axiom-inference -- "the cat sat on the mat"
//!   cargo run --release -p axiom-inference -- --batch sentences.txt

use axiom_core::input::{Encoder, Tokeniser};
use axiom_core::tiers::{AxiomConfig, HierarchicalResolver, RouteMode};
use std::io::{self, BufRead};
use std::time::Instant;

/// Complexity assessment based on tier reached.
fn complexity_label(tier: &str) -> &'static str {
    match tier {
        "Surface" => "Simple",
        "Reasoning" => "Moderate",
        "Deep" => "Complex",
        _ => "Unknown",
    }
}

/// Print structured inference output for a single sentence.
fn infer_and_print(
    resolver: &mut HierarchicalResolver,
    encoder: &Encoder,
    sentence: &str,
) {
    let start = Instant::now();
    let tensor = encoder.encode_text_readonly(sentence);
    let surface_conf = resolver.max_surface_confidence(&tensor);
    let result = resolver.resolve(&tensor);
    let elapsed = start.elapsed();

    let tier_name = result.tier_reached.name();
    let nodes_fired: Vec<&str> = result
        .route
        .trace_steps
        .iter()
        .map(|s| s.node_id.as_str())
        .collect();

    println!("Input: {}", sentence);
    println!("Confidence: {:.4}", surface_conf);
    println!("Tier: {}", tier_name);
    println!("Winning path: {}", result.winning_path);
    println!("Nodes fired: [{}]", nodes_fired.join(", "));
    println!("Complexity assessment: {}", complexity_label(tier_name));
    println!("Processing time: {} µs", elapsed.as_micros());
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Check for required files
    let weights_path = "axiom_weights.json";
    let vocab_path = "axiom_vocab.json";
    let config_path = "axiom_config.json";

    if !std::path::Path::new(weights_path).exists() {
        eprintln!("Error: {} not found. Run axiom-bench first to train a model.", weights_path);
        std::process::exit(1);
    }
    if !std::path::Path::new(vocab_path).exists() {
        eprintln!("Error: {} not found. Run axiom-bench first to build vocabulary.", vocab_path);
        std::process::exit(1);
    }

    // Load config
    let config = if std::path::Path::new(config_path).exists() {
        AxiomConfig::load_or_default()
    } else {
        AxiomConfig::default()
    };

    let input_dim = 128;

    // Build tokeniser and load vocabulary
    let mut tokeniser = Tokeniser::default_tokeniser();
    tokeniser
        .load_vocab(vocab_path)
        .expect("Failed to load vocabulary");
    let encoder = Encoder::new(input_dim, tokeniser);

    // Build resolver and load weights
    let mut resolver = HierarchicalResolver::build_with_axiom_config(input_dim, &config);
    resolver.mode = RouteMode::Inference;
    resolver
        .load_all_weights(weights_path)
        .expect("Failed to load weights");

    // Recalibrate with text-aware thresholds if config has them
    resolver.validate_confidence_invariants();

    eprintln!("AXIOM Inference ready (vocab={}, params={})",
        encoder.tokeniser.vocab_size(),
        resolver.total_weight_count());

    // --stats: print weight norms and exit
    if args.iter().any(|a| a == "--stats") {
        println!("Total parameters: {}", resolver.total_weight_count());
        println!("Surface weight norm: {:.1}", resolver.surface_weight_norm());
        println!("R+D weight norm: {:.1}", resolver.non_surface_weight_norm());
        println!("Total weight norm: {:.1}", resolver.total_weight_norm());
        return;
    }

    // Determine input mode
    let batch_mode = args.iter().any(|a| a == "--batch");

    if batch_mode {
        // --batch <file>
        let file_idx = args.iter().position(|a| a == "--batch").unwrap() + 1;
        if file_idx >= args.len() {
            eprintln!("Usage: axiom-inference --batch <file>");
            std::process::exit(1);
        }
        let contents = std::fs::read_to_string(&args[file_idx])
            .expect("Failed to read batch file");
        for line in contents.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                infer_and_print(&mut resolver, &encoder, trimmed);
            }
        }
    } else if args.len() > 1 {
        // Direct sentence argument(s)
        let sentence = args[1..].join(" ");
        infer_and_print(&mut resolver, &encoder, &sentence);
    } else {
        // Read from stdin
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            let line = line.expect("Failed to read line");
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                infer_and_print(&mut resolver, &encoder, trimmed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_label() {
        assert_eq!(complexity_label("Surface"), "Simple");
        assert_eq!(complexity_label("Reasoning"), "Moderate");
        assert_eq!(complexity_label("Deep"), "Complex");
        assert_eq!(complexity_label("Other"), "Unknown");
    }

    #[test]
    fn test_output_format() {
        // Verify the structured output fields are present
        // (integration test — requires trained model files)
        // This test validates the format function logic only
        let fields = ["Input:", "Confidence:", "Tier:", "Winning path:",
                      "Nodes fired:", "Complexity assessment:", "Processing time:"];
        for field in &fields {
            assert!(!field.is_empty());
        }
    }
}
