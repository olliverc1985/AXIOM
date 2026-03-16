//! AXIOM Evaluation — evaluate the trained router on eval sets.
//!
//! Usage:
//!   cargo run --release --bin eval_axiom -- \
//!     --weights axiom_router_weights.json \
//!     --eval-data data/eval_set1_synthetic.json,data/eval_set2_adversarial.json,data/eval_set3_realworld.json

use axiom::router::AxiomRouter;
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct EvalQuery {
    text: String,
    tier: String,
    #[serde(default)]
    alt_tier: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    domain: String,
    #[serde(default)]
    #[allow(dead_code)]
    category: String,
}

fn tier_idx(name: &str) -> usize {
    match name {
        "Surface" => 0,
        "Reasoning" => 1,
        "Deep" => 2,
        _ => 0,
    }
}

fn is_correct(pred_idx: usize, gt_name: &str, alt_tier: &Option<String>) -> bool {
    if pred_idx == tier_idx(gt_name) {
        return true;
    }
    if let Some(alt) = alt_tier {
        if pred_idx == tier_idx(alt) {
            return true;
        }
    }
    false
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut weights_path = "axiom_router_weights.json".to_string();
    let mut eval_paths: Vec<String> = vec![
        "data/eval_set1_synthetic.json".into(),
        "data/eval_set2_adversarial.json".into(),
        "data/eval_set3_realworld.json".into(),
    ];

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" | "-w" => {
                i += 1;
                weights_path = args[i].clone();
            }
            "--eval-data" => {
                i += 1;
                eval_paths = args[i].split(',').map(|s| s.to_string()).collect();
            }
            "--help" | "-h" => {
                println!("Usage: eval_axiom [OPTIONS]");
                println!(
                    "  --weights PATH     Router weights file (default: axiom_router_weights.json)"
                );
                println!("  --eval-data PATHS  Comma-separated eval JSON files");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown flag: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  AXIOM Router Evaluation                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load router
    println!("Loading weights from {}...", weights_path);
    let router = AxiomRouter::load(&weights_path);
    println!("  Loaded.");

    // Load eval data
    let mut all_queries: Vec<EvalQuery> = Vec::new();
    for path in &eval_paths {
        let json =
            std::fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {}", path));
        let queries: Vec<EvalQuery> =
            serde_json::from_str(&json).unwrap_or_else(|_| panic!("Failed to parse {}", path));
        println!("  Loaded {} queries from {}", queries.len(), path);
        all_queries.extend(queries);
    }
    println!("  Total: {} queries", all_queries.len());
    println!();

    // Evaluate
    let start = Instant::now();
    let mut confusion = [[0usize; 3]; 3];
    let mut correct = 0;
    let mut s_correct = 0;
    let mut s_total = 0;
    let mut r_correct = 0;
    let mut r_total = 0;
    let mut d_correct = 0;
    let mut d_total = 0;
    let mut latencies: Vec<u64> = Vec::with_capacity(all_queries.len());
    let mut errors: Vec<(String, String, String)> = Vec::new(); // (text, expected, predicted)

    for q in &all_queries {
        let decision = router.route(&q.text);
        let pred_idx = match decision.tier {
            axiom::tiers::Tier::Surface => 0,
            axiom::tiers::Tier::Reasoning => 1,
            axiom::tiers::Tier::Deep => 2,
        };
        let gt = tier_idx(&q.tier);
        confusion[gt][pred_idx] += 1;
        latencies.push(decision.latency_us);

        let ok = is_correct(pred_idx, &q.tier, &q.alt_tier);
        if ok {
            correct += 1;
        } else {
            let pred_name = match pred_idx {
                0 => "Surface",
                1 => "Reasoning",
                _ => "Deep",
            };
            errors.push((q.text.clone(), q.tier.clone(), pred_name.to_string()));
        }
        match q.tier.as_str() {
            "Surface" => {
                s_total += 1;
                if ok {
                    s_correct += 1;
                }
            }
            "Reasoning" => {
                r_total += 1;
                if ok {
                    r_correct += 1;
                }
            }
            "Deep" => {
                d_total += 1;
                if ok {
                    d_correct += 1;
                }
            }
            _ => {}
        }
    }

    let elapsed = start.elapsed();

    // Results
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Results                                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  {:>12}  {:>6} / {:>6}  ({:>5.1}%)",
        "Surface",
        s_correct,
        s_total,
        s_correct as f32 / s_total.max(1) as f32 * 100.0
    );
    println!(
        "  {:>12}  {:>6} / {:>6}  ({:>5.1}%)",
        "Reasoning",
        r_correct,
        r_total,
        r_correct as f32 / r_total.max(1) as f32 * 100.0
    );
    println!(
        "  {:>12}  {:>6} / {:>6}  ({:>5.1}%)",
        "Deep",
        d_correct,
        d_total,
        d_correct as f32 / d_total.max(1) as f32 * 100.0
    );
    println!(
        "  {:>12}  {:>6} / {:>6}  ({:>5.1}%)",
        "Overall",
        correct,
        all_queries.len(),
        correct as f32 / all_queries.len() as f32 * 100.0
    );

    // Confusion matrix
    println!();
    println!("  Confusion Matrix:");
    println!("  {:>12} {:>8} {:>8} {:>8}", "", "→Surf", "→Reas", "→Deep");
    for (i, name) in ["Surface", "Reasoning", "Deep"].iter().enumerate() {
        println!(
            "  {:>12} {:>8} {:>8} {:>8}",
            name, confusion[i][0], confusion[i][1], confusion[i][2]
        );
    }

    // Latency stats
    latencies.sort();
    let avg_lat = latencies.iter().sum::<u64>() as f32 / latencies.len() as f32;
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() as f32 * 0.95) as usize];
    let p99 = latencies[(latencies.len() as f32 * 0.99) as usize];
    println!();
    println!("  Latency:");
    println!("    Mean: {:.0} us ({:.2} ms)", avg_lat, avg_lat / 1000.0);
    println!("    P50:  {} us", p50);
    println!("    P95:  {} us", p95);
    println!("    P99:  {} us", p99);
    println!("    Total eval time: {:.2}s", elapsed.as_secs_f32());

    // Error analysis (first 20)
    if !errors.is_empty() {
        println!();
        println!("  Errors ({} total, showing first 20):", errors.len());
        for (text, expected, predicted) in errors.iter().take(20) {
            let short = if text.len() > 60 { &text[..60] } else { text };
            println!("    {} → {} (expected {})", short, predicted, expected);
        }
    }

    // Target check
    println!();
    let sa = s_correct as f32 / s_total.max(1) as f32 * 100.0;
    let ra = r_correct as f32 / r_total.max(1) as f32 * 100.0;
    let da = d_correct as f32 / d_total.max(1) as f32 * 100.0;
    let oa = correct as f32 / all_queries.len() as f32 * 100.0;
    if sa >= 80.0 && ra >= 85.0 && da >= 80.0 && oa >= 85.0 {
        println!("  ALL TARGETS MET (S>=80%, R>=85%, D>=80%, O>=85%)");
    } else {
        println!("  Targets:");
        println!(
            "    Surface >=80%:   {} ({:.1}%)",
            if sa >= 80.0 { "PASS" } else { "FAIL" },
            sa
        );
        println!(
            "    Reasoning >=85%: {} ({:.1}%)",
            if ra >= 85.0 { "PASS" } else { "FAIL" },
            ra
        );
        println!(
            "    Deep >=80%:      {} ({:.1}%)",
            if da >= 80.0 { "PASS" } else { "FAIL" },
            da
        );
        println!(
            "    Overall >=85%:   {} ({:.1}%)",
            if oa >= 85.0 { "PASS" } else { "FAIL" },
            oa
        );
    }
}
