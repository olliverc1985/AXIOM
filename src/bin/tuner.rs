//! AXIOM auto-tuner CLI — reads bench log, produces axiom_config.json.
//!
//! Usage: cargo run -p axiom-tuner [-- path/to/bench_log.json]

use axiom::tiers::AxiomConfig;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let log_path = args.get(1).map(|s| s.as_str()).unwrap_or("axiom_bench_log.json");

    println!("AXIOM Auto-Tuner");
    println!("  Reading: {}", log_path);

    let stats = match axiom::tuner::compute_stats(log_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  Error: {}", e);
            std::process::exit(1);
        }
    };

    println!();
    println!("  Bench Stats:");
    println!("    Total entries:  {}", stats.total);
    println!("    Surface:        {:.1}%", stats.surface_pct);
    println!("    Reasoning:      {:.1}%", stats.reasoning_pct);
    println!("    Deep:           {:.1}%", stats.deep_pct);
    println!("    Cache hit rate: {:.1}%", stats.cache_hit_pct);
    println!("    Avg confidence: {:.4}", stats.avg_confidence);
    println!("    Avg cost:       {:.4}", stats.avg_cost);

    let current = AxiomConfig::load_or_default();
    println!();
    println!("  Current config:");
    println!("    surface_threshold:      {:.2}", current.surface_confidence_threshold);
    println!("    reasoning_threshold:    {:.2}", current.reasoning_confidence_threshold);
    println!("    reasoning_base_conf:    {:.2}", current.reasoning_base_confidence);
    println!("    cache_sim_threshold:    {:.2}", current.cache_similarity_threshold);

    let recommended = axiom::tuner::tune(&stats, &current);

    println!();
    println!("  Recommended config:");
    println!("    surface_threshold:      {:.2}", recommended.surface_confidence_threshold);
    println!("    reasoning_threshold:    {:.2}", recommended.reasoning_confidence_threshold);
    println!("    reasoning_base_conf:    {:.2}", recommended.reasoning_base_confidence);
    println!("    cache_sim_threshold:    {:.2}", recommended.cache_similarity_threshold);
    println!();
    println!("  Rationale: {}", recommended.rationale);

    match recommended.save() {
        Ok(()) => println!("\n  Written to: axiom_config.json"),
        Err(e) => eprintln!("\n  Failed to write config: {}", e),
    }
}
