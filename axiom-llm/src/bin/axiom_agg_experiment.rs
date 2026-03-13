//! AXIOM Aggregation Strategy Experiment
//!
//! Tests 4 different chunk aggregation strategies against multi-paragraph
//! scenarios to find which strategy best handles multi-paragraph inputs.
//!
//! The current `resolve_text` picks the chunk with the HIGHEST confidence
//! (most Surface-like), which means multi-paragraph inputs always route
//! Surface because there is always one simple sentence. This experiment
//! compares alternatives.
//!
//! Usage:
//!   cargo run --release --bin axiom_agg_experiment

use axiom_core::cache::EmbeddingCache;
use axiom_core::input::{Encoder, Tokeniser};
use axiom_core::tiers::{AxiomConfig, HierarchicalResolver, RouteMode, Tier};
use serde::Deserialize;

// ── Scenario data ──

#[derive(Debug, Deserialize)]
struct ScenarioEntry {
    id: String,
    label: String,
    #[allow(dead_code)]
    notes: String,
    text: String,
}

// ── Per-chunk routing result ──

#[derive(Debug, Clone)]
struct ChunkResult {
    text: String,
    surface_confidence: f32,
    tier_reached: Tier,
    from_cache: bool,
}

// ── Result of a single strategy on a single scenario ──

#[derive(Debug, Clone)]
struct StrategyScenarioResult {
    scenario_id: String,
    ground_truth: String,
    routed_tier: String,
    agg_confidence: f32,
    correct: bool,
}

// ── Strategy names ──

const STRATEGY_NAMES: [&str; 5] = [
    "A: Mean",
    "B: Median",
    "C: Threshold Count (40%)",
    "D: Weighted Blend",
    "E: Full Text (no chunking)",
];

// ── Correctness check (same as axiom_report.rs) ──

fn is_correct(ground_truth: &str, axiom_tier: &str) -> bool {
    match ground_truth {
        "simple" => axiom_tier == "Surface",
        "moderate" => axiom_tier == "Reasoning",
        "complex" => axiom_tier == "Reasoning" || axiom_tier == "Deep",
        _ => false,
    }
}

// ── Sentence splitting (same logic as resolve_text) ──

fn split_sentences(text: &str) -> Vec<String> {
    let min_tokens = 3;
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                let token_count = trimmed.split_whitespace().count();
                if token_count >= min_tokens {
                    sentences.push(trimmed);
                }
            }
            current.clear();
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        let token_count = trimmed.split_whitespace().count();
        if token_count >= min_tokens {
            sentences.push(trimmed);
        }
    }
    if sentences.is_empty() {
        // Fallback: use the whole text as one chunk
        sentences.push(text.to_string());
    }
    sentences
}

// ── Aggregation strategies ──

/// Determine routed tier given an aggregated confidence and per-chunk results.
/// If agg_conf >= threshold → Surface
/// Otherwise use the chunk with the LOWEST confidence to pick Reasoning vs Deep.
fn determine_tier(
    agg_conf: f32,
    surface_threshold: f32,
    chunk_results: &[ChunkResult],
) -> String {
    if agg_conf >= surface_threshold {
        return "Surface".to_string();
    }
    // Find chunk with lowest confidence → its tier_reached drives the decision
    let lowest = chunk_results
        .iter()
        .min_by(|a, b| a.surface_confidence.partial_cmp(&b.surface_confidence).unwrap())
        .unwrap();
    lowest.tier_reached.name().to_string()
}

/// Strategy A: Mean confidence
fn strategy_mean(chunk_results: &[ChunkResult], surface_threshold: f32) -> (f32, String) {
    let mean = chunk_results.iter().map(|c| c.surface_confidence).sum::<f32>()
        / chunk_results.len() as f32;
    let tier = determine_tier(mean, surface_threshold, chunk_results);
    (mean, tier)
}

/// Strategy B: Median confidence
fn strategy_median(chunk_results: &[ChunkResult], surface_threshold: f32) -> (f32, String) {
    let mut confs: Vec<f32> = chunk_results.iter().map(|c| c.surface_confidence).collect();
    confs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if confs.len() % 2 == 1 {
        confs[confs.len() / 2]
    } else {
        (confs[confs.len() / 2 - 1] + confs[confs.len() / 2]) / 2.0
    };
    let tier = determine_tier(median, surface_threshold, chunk_results);
    (median, tier)
}

/// Strategy C: Threshold count — escalate if >40% of chunks are below surface threshold
fn strategy_threshold_count(
    chunk_results: &[ChunkResult],
    surface_threshold: f32,
) -> (f32, String) {
    let below_count = chunk_results
        .iter()
        .filter(|c| c.surface_confidence < surface_threshold)
        .count();
    let below_pct = below_count as f32 / chunk_results.len() as f32;
    // Use below_pct as the "confidence" metric (inverted: higher below_pct = less surface-like)
    // For routing: if >40% are below threshold → escalate
    let agg_conf = if below_pct > 0.40 {
        // Force below threshold: use mean of below-threshold chunks as the confidence
        let mean = chunk_results.iter().map(|c| c.surface_confidence).sum::<f32>()
            / chunk_results.len() as f32;
        mean.min(surface_threshold - 0.001)
    } else {
        // Stay at surface: use mean as confidence
        chunk_results.iter().map(|c| c.surface_confidence).sum::<f32>()
            / chunk_results.len() as f32
    };
    let tier = determine_tier(agg_conf, surface_threshold, chunk_results);
    (agg_conf, tier)
}

/// Strategy D: Weighted blend — 0.3 * min + 0.7 * mean
fn strategy_weighted_blend(
    chunk_results: &[ChunkResult],
    surface_threshold: f32,
) -> (f32, String) {
    let mean = chunk_results.iter().map(|c| c.surface_confidence).sum::<f32>()
        / chunk_results.len() as f32;
    let min = chunk_results
        .iter()
        .map(|c| c.surface_confidence)
        .fold(f32::INFINITY, f32::min);
    let blend = 0.3 * min + 0.7 * mean;
    let tier = determine_tier(blend, surface_threshold, chunk_results);
    (blend, tier)
}

// ── Main ──

fn main() {
    let weights_path = "axiom_weights.json";
    let vocab_path = "axiom_vocab.json";
    let config_path = "axiom_config.json";

    // Check required files
    for (path, desc) in [
        (weights_path, "trained weights"),
        (vocab_path, "vocabulary"),
    ] {
        if !std::path::Path::new(path).exists() {
            eprintln!(
                "Error: {} not found ({}). Run axiom-bench first.",
                path, desc
            );
            std::process::exit(1);
        }
    }

    // Load config
    let config = if std::path::Path::new(config_path).exists() {
        AxiomConfig::load_or_default()
    } else {
        AxiomConfig::default()
    };

    let input_dim = 128;

    // Build tokeniser and encoder
    let mut tokeniser = Tokeniser::default_tokeniser();
    tokeniser
        .load_vocab(vocab_path)
        .expect("Failed to load vocabulary");
    let encoder = Encoder::new(input_dim, tokeniser);

    // Build resolver and load weights (mid_dim=128 matches bench training)
    let mut resolver = HierarchicalResolver::build_with_axiom_config_mid_dim(input_dim, &config, 128);
    resolver.mode = RouteMode::Inference;
    resolver
        .load_all_weights(weights_path)
        .expect("Failed to load weights");

    // Restore G5 magnitude penalty
    if resolver.g5_simple_mean_norm > 0.0 || resolver.g5_complex_mean_norm > 0.0 {
        resolver.set_g5_penalty_weight(0.25);
    }
    resolver.validate_confidence_invariants();

    let surface_threshold = resolver.config.surface_confidence_threshold;

    eprintln!(
        "AXIOM Aggregation Experiment ready (vocab={}, params={}, surface_threshold={:.4})",
        encoder.tokeniser.vocab_size(),
        resolver.total_weight_count(),
        surface_threshold,
    );

    // Load scenarios
    let scenarios_path = "axiom-datasets/scenarios.json";
    let scenarios: Vec<ScenarioEntry> = {
        let data = std::fs::read_to_string(scenarios_path).unwrap_or_else(|e| {
            eprintln!("Error: failed to read {}: {}", scenarios_path, e);
            std::process::exit(1);
        });
        serde_json::from_str(&data).unwrap_or_else(|e| {
            eprintln!("Error: failed to parse {}: {}", scenarios_path, e);
            std::process::exit(1);
        })
    };

    eprintln!("Loaded {} scenarios", scenarios.len());

    // ── Phase 1: Route each scenario's chunks individually ──

    // For each scenario, store per-chunk results
    let mut scenario_chunks: Vec<(String, String, Vec<ChunkResult>)> = Vec::new();

    for sc in &scenarios {
        let sentences = split_sentences(&sc.text);
        let mut chunk_results = Vec::new();

        for sentence in &sentences {
            // Reset cache BETWEEN chunks so each gets an independent score
            resolver.cache = EmbeddingCache::new(256, 0.75);

            let encoded = encoder.encode_text_readonly(sentence);
            let result = resolver.resolve(&encoded);

            chunk_results.push(ChunkResult {
                text: sentence.clone(),
                surface_confidence: result.surface_confidence,
                tier_reached: result.tier_reached,
                from_cache: result.from_cache,
            });
        }

        eprintln!(
            "  {} [{}]: {} chunks, confidences: [{}]",
            sc.id,
            sc.label,
            chunk_results.len(),
            chunk_results
                .iter()
                .map(|c| format!("{:.3}", c.surface_confidence))
                .collect::<Vec<_>>()
                .join(", ")
        );

        scenario_chunks.push((sc.id.clone(), sc.label.clone(), chunk_results));
    }

    // ── Phase 1b: Route full text (no chunking) ──

    let mut full_text_results: Vec<(String, String, f32, String)> = Vec::new();
    for sc in &scenarios {
        resolver.cache = EmbeddingCache::new(256, 0.75);
        let encoded = encoder.encode_text_readonly(&sc.text);
        let result = resolver.resolve(&encoded);
        let tier_name = result.tier_reached.name().to_string();
        eprintln!(
            "  {} [{}] full-text: conf={:.4} tier={}",
            sc.id, sc.label, result.surface_confidence, tier_name
        );
        full_text_results.push((sc.id.clone(), sc.label.clone(), result.surface_confidence, tier_name));
    }

    // ── Phase 2: Apply each strategy and collect results ──

    // strategies[strategy_idx][scenario_idx]
    let mut strategy_results: Vec<Vec<StrategyScenarioResult>> = Vec::new();

    for strategy_idx in 0..5 {
        let mut results = Vec::new();

        if strategy_idx == 4 {
            // Strategy E: full text routing
            for (id, label, conf, tier) in &full_text_results {
                let correct = is_correct(label, tier);
                results.push(StrategyScenarioResult {
                    scenario_id: id.clone(),
                    ground_truth: label.clone(),
                    routed_tier: tier.clone(),
                    agg_confidence: *conf,
                    correct,
                });
            }
        } else {
            for (id, label, chunks) in &scenario_chunks {
                let (agg_conf, routed_tier) = match strategy_idx {
                    0 => strategy_mean(chunks, surface_threshold),
                    1 => strategy_median(chunks, surface_threshold),
                    2 => strategy_threshold_count(chunks, surface_threshold),
                    3 => strategy_weighted_blend(chunks, surface_threshold),
                    _ => unreachable!(),
                };

                let correct = is_correct(label, &routed_tier);

                results.push(StrategyScenarioResult {
                    scenario_id: id.clone(),
                    ground_truth: label.clone(),
                    routed_tier,
                    agg_confidence: agg_conf,
                    correct,
                });
            }
        }

        strategy_results.push(results);
    }

    // ── Phase 3: Print per-chunk detail ──

    println!("================================================================");
    println!("  AXIOM Aggregation Strategy Experiment");
    println!("  Surface threshold: {:.4}", surface_threshold);
    println!("================================================================");
    println!();

    println!("── Per-Chunk Detail ──");
    println!();
    for (id, label, chunks) in &scenario_chunks {
        println!(
            "  {} [{}] — {} chunks",
            id,
            label,
            chunks.len()
        );
        for (i, chunk) in chunks.iter().enumerate() {
            let text_trunc = if chunk.text.len() > 70 {
                format!("{}...", &chunk.text[..67])
            } else {
                chunk.text.clone()
            };
            println!(
                "    chunk {:>2}: conf={:.4}  tier={}  cache={}  \"{}\"",
                i + 1,
                chunk.surface_confidence,
                chunk.tier_reached.name(),
                if chunk.from_cache { "HIT" } else { "---" },
                text_trunc,
            );
        }
        println!();
    }

    // ── Phase 4: Print per-strategy results tables ──

    for (strategy_idx, results) in strategy_results.iter().enumerate() {
        println!("── Strategy {} ──", STRATEGY_NAMES[strategy_idx]);
        println!();
        println!(
            "  {:<14} {:<10} {:<12} {:<10} {:<8}",
            "Scenario", "Truth", "Routed", "AggConf", "Correct"
        );
        println!(
            "  {:<14} {:<10} {:<12} {:<10} {:<8}",
            "-".repeat(14),
            "-".repeat(10),
            "-".repeat(12),
            "-".repeat(10),
            "-".repeat(8)
        );

        for r in results {
            let correct_mark = if r.correct { "Yes" } else { "**No**" };
            println!(
                "  {:<14} {:<10} {:<12} {:<10.4} {:<8}",
                r.scenario_id, r.ground_truth, r.routed_tier, r.agg_confidence, correct_mark,
            );
        }

        let correct_count = results.iter().filter(|r| r.correct).count();
        let accuracy = correct_count as f64 / results.len() as f64 * 100.0;
        println!();
        println!(
            "  Accuracy: {}/{} ({:.1}%)",
            correct_count,
            results.len(),
            accuracy
        );
        println!();
    }

    // ── Phase 5: Summary comparison table ──

    println!("================================================================");
    println!("  Summary Comparison");
    println!("================================================================");
    println!();

    // Header row
    let scenario_ids: Vec<&str> = scenario_chunks
        .iter()
        .map(|(id, _, _)| id.as_str())
        .collect();

    print!("  {:<28} {:<10}", "Strategy", "Accuracy");
    for sid in &scenario_ids {
        print!(" {:<14}", sid);
    }
    println!();

    print!("  {:<28} {:<10}", "-".repeat(28), "-".repeat(10));
    for _ in &scenario_ids {
        print!(" {:<14}", "-".repeat(14));
    }
    println!();

    for (strategy_idx, results) in strategy_results.iter().enumerate() {
        let correct_count = results.iter().filter(|r| r.correct).count();
        let accuracy = format!(
            "{}/{} ({:.0}%)",
            correct_count,
            results.len(),
            correct_count as f64 / results.len() as f64 * 100.0
        );

        print!("  {:<28} {:<10}", STRATEGY_NAMES[strategy_idx], accuracy);
        for r in results {
            let cell = format!(
                "{} {}",
                r.routed_tier,
                if r.correct { "OK" } else { "MISS" }
            );
            print!(" {:<14}", cell);
        }
        println!();
    }

    // Ground truth row
    print!("  {:<28} {:<10}", "Ground Truth", "");
    for (_, label, _) in &scenario_chunks {
        print!(" {:<14}", label);
    }
    println!();
    println!();

    // ── Phase 6: Winner ──

    let best_idx = strategy_results
        .iter()
        .enumerate()
        .max_by_key(|(_, results)| results.iter().filter(|r| r.correct).count())
        .map(|(idx, _)| idx)
        .unwrap();

    let best_correct = strategy_results[best_idx]
        .iter()
        .filter(|r| r.correct)
        .count();
    let best_accuracy = best_correct as f64 / strategy_results[best_idx].len() as f64 * 100.0;

    println!(
        "Best strategy: {} — {}/{} correct ({:.1}%)",
        STRATEGY_NAMES[best_idx],
        best_correct,
        strategy_results[best_idx].len(),
        best_accuracy,
    );
    println!();

    // Also note if there are ties
    let tied: Vec<&str> = strategy_results
        .iter()
        .enumerate()
        .filter(|(_, results)| results.iter().filter(|r| r.correct).count() == best_correct)
        .map(|(idx, _)| STRATEGY_NAMES[idx])
        .collect();

    if tied.len() > 1 {
        println!("  (Tied with: {})", tied.join(", "));
        println!();
    }
}
