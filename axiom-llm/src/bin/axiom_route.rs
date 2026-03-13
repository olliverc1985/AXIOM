//! AXIOM Route — CLI binary for routing queries through AXIOM to Anthropic LLMs.
//!
//! Usage:
//!   ANTHROPIC_API_KEY=sk-... cargo run --release --bin axiom_route -- "your query here"

use axiom_llm::router::AxiomRouter;
use axiom_llm::ModelConfig;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: axiom_route \"your query here\"");
        eprintln!("       ANTHROPIC_API_KEY must be set");
        std::process::exit(1);
    }

    let query = args[1..].join(" ");

    let weights_path = "axiom_weights.json";
    let vocab_path = "axiom_vocab.json";
    let config_path = "axiom_config.json";

    // Check required files
    for (path, name) in [
        (weights_path, "weights"),
        (vocab_path, "vocabulary"),
    ] {
        if !std::path::Path::new(path).exists() {
            eprintln!("Error: {} not found ({}). Run axiom-bench first.", path, name);
            std::process::exit(1);
        }
    }

    let model_config = ModelConfig::default();

    let mut router = match AxiomRouter::from_trained(
        weights_path,
        vocab_path,
        config_path,
        model_config,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error initialising router: {}", e);
            std::process::exit(1);
        }
    };

    eprintln!(
        "AXIOM Router ready (params={})",
        router.resolver.total_weight_count()
    );

    match router.route_and_call(&query) {
        Ok(resp) => {
            println!("Query: {}", query);
            println!("AXIOM routing: {} µs", resp.axiom_routing_us);
            println!("Tier: {}", resp.tier);
            println!("Model: {}", resp.model);
            println!("Confidence: {:.4}", resp.confidence);
            if !resp.coalition_members.is_empty() {
                println!("Coalition: [{}]", resp.coalition_members.join(", "));
            }
            println!("Response: {}", resp.response);
            println!(
                "Tokens: {}/{}",
                router.cost_log.last().map(|r| r.input_tokens).unwrap_or(0),
                router.cost_log.last().map(|r| r.output_tokens).unwrap_or(0)
            );
            println!(
                "Cost: ${:.6} (saved ${:.6} vs premium)",
                resp.actual_cost_usd, resp.saved_vs_premium_usd
            );
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}
