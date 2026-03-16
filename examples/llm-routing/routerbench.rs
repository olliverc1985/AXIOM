//! AXIOM RouterBench Evaluation — evaluate AXIOM against the RouterBench benchmark.
//!
//! Reads preprocessed JSONL, routes each prompt through AxiomRouter,
//! maps tiers to models, computes accuracy/cost/AIQ metrics.

use axiom::router::AxiomRouter;
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Deserialize)]
struct Record {
    prompt: String,
    #[allow(dead_code)]
    eval_name: String,
    dataset: String,
    scores: HashMap<String, f64>,
    costs: HashMap<String, f64>,
    #[allow(dead_code)]
    oracle: String,
}

// Tier → model mapping
// Costs (avg $/query): mistral-7b $0.00014, wizardlm-13b $0.00021, mixtral $0.00041,
// codellama $0.00053, yi-34b $0.00056, claude-instant $0.00061, llama-70b $0.00062,
// gpt-3.5 $0.00071, claude-v1 $0.00587, claude-v2 $0.00615, gpt-4 $0.00794

const SURFACE_MODELS: &[&str] = &["mistral-7b", "wizardlm-13b"];
const REASONING_MODELS: &[&str] = &[
    "mixtral-8x7b",
    "codellama-34b",
    "yi-34b",
    "claude-instant",
    "llama-70b",
    "gpt-3.5",
];
const _DEEP_MODELS: &[&str] = &["claude-v1", "claude-v2", "gpt-4"];

// Strategy A: best-in-tier (highest avg accuracy)
const BEST_SURFACE: &str = "wizardlm-13b"; // 0.539
const BEST_REASONING: &str = "yi-34b"; // 0.715
const BEST_DEEP: &str = "gpt-4"; // 0.805

// Strategy B: cheapest-in-tier
const CHEAP_SURFACE: &str = "mistral-7b"; // $0.00014
const CHEAP_REASONING: &str = "mixtral-8x7b"; // $0.00041
const CHEAP_DEEP: &str = "claude-v1"; // $0.00587

// All models for baselines
const ALL_MODELS: &[&str] = &[
    "mistral-7b",
    "wizardlm-13b",
    "mixtral-8x7b",
    "codellama-34b",
    "yi-34b",
    "claude-instant",
    "llama-70b",
    "gpt-3.5",
    "claude-v1",
    "claude-v2",
    "gpt-4",
];

struct Metrics {
    total_score: f64,
    total_cost: f64,
    count: usize,
}

impl Metrics {
    fn new() -> Self {
        Self {
            total_score: 0.0,
            total_cost: 0.0,
            count: 0,
        }
    }
    fn add(&mut self, score: f64, cost: f64) {
        self.total_score += score;
        self.total_cost += cost;
        self.count += 1;
    }
    fn accuracy(&self) -> f64 {
        if self.count > 0 {
            self.total_score / self.count as f64
        } else {
            0.0
        }
    }
    fn avg_cost(&self) -> f64 {
        if self.count > 0 {
            self.total_cost / self.count as f64
        } else {
            0.0
        }
    }
}

fn _tier_for_model(model: &str) -> &'static str {
    if SURFACE_MODELS.contains(&model) {
        "Surface"
    } else if REASONING_MODELS.contains(&model) {
        "Reasoning"
    } else {
        "Deep"
    }
}

fn main() {
    let weights_path = std::env::args()
        .nth(1)
        .unwrap_or("axiom_router_weights.json".to_string());
    let data_path = std::env::args()
        .nth(2)
        .unwrap_or("data/routerbench/routerbench.jsonl".to_string());

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  AXIOM × RouterBench Evaluation                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load router
    println!("Loading AXIOM router from {}...", weights_path);
    let router = AxiomRouter::load(&weights_path);
    println!("  Loaded.");

    // Load data
    println!("Loading RouterBench data from {}...", data_path);
    let content = std::fs::read_to_string(&data_path).expect("Failed to read data");
    let records: Vec<Record> = content
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    println!("  {} records loaded.", records.len());
    println!();

    // ─── Route all prompts ───
    println!("Routing {} prompts through AXIOM...", records.len());
    let route_start = Instant::now();

    struct RoutedRecord {
        tier: usize, // 0=Surface, 1=Reasoning, 2=Deep
        confidence: [f32; 3],
    }

    let mut routed: Vec<RoutedRecord> = Vec::with_capacity(records.len());
    for rec in &records {
        let decision = router.route(&rec.prompt);
        let tier = match decision.tier {
            axiom::tiers::Tier::Surface => 0,
            axiom::tiers::Tier::Reasoning => 1,
            axiom::tiers::Tier::Deep => 2,
        };
        routed.push(RoutedRecord {
            tier,
            confidence: decision.confidence,
        });
    }

    let route_elapsed = route_start.elapsed();
    let avg_latency_us = route_elapsed.as_micros() as f64 / records.len() as f64;
    println!(
        "  Done in {:.2}s ({:.0} us/query avg)",
        route_elapsed.as_secs_f64(),
        avg_latency_us
    );

    // Tier distribution
    let tier_counts = [
        routed.iter().filter(|r| r.tier == 0).count(),
        routed.iter().filter(|r| r.tier == 1).count(),
        routed.iter().filter(|r| r.tier == 2).count(),
    ];
    println!(
        "  Tier distribution: Surface={} ({:.1}%), Reasoning={} ({:.1}%), Deep={} ({:.1}%)",
        tier_counts[0],
        tier_counts[0] as f64 / records.len() as f64 * 100.0,
        tier_counts[1],
        tier_counts[1] as f64 / records.len() as f64 * 100.0,
        tier_counts[2],
        tier_counts[2] as f64 / records.len() as f64 * 100.0
    );
    println!();

    // ─── Compute metrics for all strategies ───
    let mut axiom_a = Metrics::new(); // best-in-tier
    let mut axiom_b = Metrics::new(); // cheapest-in-tier
    let mut axiom_c = Metrics::new(); // cascade
    let mut always_cheap = Metrics::new(); // always mistral-7b
    let mut always_best = Metrics::new(); // always gpt-4
    let mut oracle = Metrics::new();
    let mut random_m = Metrics::new();

    // Per-dataset metrics
    let mut ds_axiom_a: HashMap<String, Metrics> = HashMap::new();
    let mut ds_axiom_b: HashMap<String, Metrics> = HashMap::new();
    let mut ds_always_cheap: HashMap<String, Metrics> = HashMap::new();
    let mut ds_always_best: HashMap<String, Metrics> = HashMap::new();
    let mut ds_oracle: HashMap<String, Metrics> = HashMap::new();

    // Per-tier accuracy analysis
    let mut tier_score_a = [Metrics::new(), Metrics::new(), Metrics::new()];

    // Under/over routing analysis
    let mut under_route = 0usize; // Surface but only expensive models correct
    let mut over_route = 0usize; // Deep but cheap models handle it

    let mut seed: u64 = 42;

    for (i, rec) in records.iter().enumerate() {
        let rt = &routed[i];

        // Strategy A: best-in-tier
        let model_a = match rt.tier {
            0 => BEST_SURFACE,
            1 => BEST_REASONING,
            _ => BEST_DEEP,
        };
        let score_a = *rec.scores.get(model_a).unwrap_or(&0.0);
        let cost_a = *rec.costs.get(model_a).unwrap_or(&0.0);
        axiom_a.add(score_a, cost_a);
        tier_score_a[rt.tier].add(score_a, cost_a);

        // Strategy B: cheapest-in-tier
        let model_b = match rt.tier {
            0 => CHEAP_SURFACE,
            1 => CHEAP_REASONING,
            _ => CHEAP_DEEP,
        };
        let score_b = *rec.scores.get(model_b).unwrap_or(&0.0);
        let cost_b = *rec.costs.get(model_b).unwrap_or(&0.0);
        axiom_b.add(score_b, cost_b);

        // Strategy C: cascade (Surface first, escalate if score < threshold)
        // Use confidence: if Surface confidence > 0.8, stay Surface; else try Reasoning; else Deep
        let model_c = if rt.confidence[0] > 0.8 {
            BEST_SURFACE
        } else if rt.confidence[0] + rt.confidence[1] > 0.8 {
            BEST_REASONING
        } else {
            BEST_DEEP
        };
        let score_c = *rec.scores.get(model_c).unwrap_or(&0.0);
        let cost_c = *rec.costs.get(model_c).unwrap_or(&0.0);
        axiom_c.add(score_c, cost_c);

        // Baselines
        let score_cheap = *rec.scores.get("mistral-7b").unwrap_or(&0.0);
        let cost_cheap = *rec.costs.get("mistral-7b").unwrap_or(&0.0);
        always_cheap.add(score_cheap, cost_cheap);

        let score_best = *rec.scores.get("gpt-4").unwrap_or(&0.0);
        let cost_best = *rec.costs.get("gpt-4").unwrap_or(&0.0);
        always_best.add(score_best, cost_best);

        // Oracle: cheapest model that gets the query right (score > 0)
        let mut oracle_score = 0.0;
        let mut oracle_cost = f64::MAX;
        for &m in ALL_MODELS {
            let s = *rec.scores.get(m).unwrap_or(&0.0);
            let c = *rec.costs.get(m).unwrap_or(&0.0);
            if s > 0.0 && c < oracle_cost {
                oracle_score = s;
                oracle_cost = c;
            }
        }
        if oracle_cost == f64::MAX {
            // No model correct — use cheapest
            oracle_cost = cost_cheap;
        }
        oracle.add(oracle_score, oracle_cost);

        // Random: pick a random model
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let rand_idx = (seed >> 33) as usize % ALL_MODELS.len();
        let rand_model = ALL_MODELS[rand_idx];
        let score_rand = *rec.scores.get(rand_model).unwrap_or(&0.0);
        let cost_rand = *rec.costs.get(rand_model).unwrap_or(&0.0);
        random_m.add(score_rand, cost_rand);

        // Per-dataset
        let ds = rec.dataset.clone();
        ds_axiom_a
            .entry(ds.clone())
            .or_insert_with(Metrics::new)
            .add(score_a, cost_a);
        ds_axiom_b
            .entry(ds.clone())
            .or_insert_with(Metrics::new)
            .add(score_b, cost_b);
        ds_always_cheap
            .entry(ds.clone())
            .or_insert_with(Metrics::new)
            .add(score_cheap, cost_cheap);
        ds_always_best
            .entry(ds.clone())
            .or_insert_with(Metrics::new)
            .add(score_best, cost_best);
        ds_oracle
            .entry(ds.clone())
            .or_insert_with(Metrics::new)
            .add(oracle_score, oracle_cost);

        // Under/over routing
        let surface_ok = SURFACE_MODELS
            .iter()
            .any(|m| *rec.scores.get(*m).unwrap_or(&0.0) > 0.0);
        let _deep_needed = !surface_ok
            && !REASONING_MODELS
                .iter()
                .any(|m| *rec.scores.get(*m).unwrap_or(&0.0) > 0.0);
        if rt.tier == 0 && !surface_ok {
            under_route += 1;
        }
        if rt.tier == 2 && surface_ok {
            over_route += 1;
        }
    }

    // ─── Print results ───
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║  Overall Results ({} queries)                     ║",
        records.len()
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let gpt4_cost = always_best.avg_cost();

    println!(
        "  {:>16} {:>8} {:>12} {:>8}",
        "Router", "Acc", "Avg $/q", "CostRed"
    );
    println!(
        "  {:>16} {:>8} {:>12} {:>8}",
        "──────", "───", "───────", "───────"
    );

    let print_row = |name: &str, m: &Metrics| {
        let cr = if gpt4_cost > 0.0 {
            (1.0 - m.avg_cost() / gpt4_cost) * 100.0
        } else {
            0.0
        };
        println!(
            "  {:>16} {:>7.1}% {:>11.6} {:>7.1}%",
            name,
            m.accuracy() * 100.0,
            m.avg_cost(),
            cr
        );
    };

    print_row("Always-Cheap", &always_cheap);
    print_row("Random", &random_m);
    print_row("AXIOM (A:best)", &axiom_a);
    print_row("AXIOM (B:cheap)", &axiom_b);
    print_row("AXIOM (C:casc)", &axiom_c);
    print_row("Always-GPT4", &always_best);
    print_row("Oracle", &oracle);

    // ─── Per-tier accuracy ───
    println!();
    println!("  Per-tier accuracy (Strategy A):");
    let tier_names = ["Surface", "Reasoning", "Deep"];
    for (i, name) in tier_names.iter().enumerate() {
        if tier_score_a[i].count > 0 {
            println!(
                "    {:>10}: {:.1}% ({} queries, avg cost ${:.6})",
                name,
                tier_score_a[i].accuracy() * 100.0,
                tier_score_a[i].count,
                tier_score_a[i].avg_cost()
            );
        }
    }

    // ─── Routing efficiency ───
    println!();
    println!("  Routing efficiency:");
    println!(
        "    Under-routed (Surface but no Surface model correct): {} ({:.1}%)",
        under_route,
        under_route as f64 / records.len() as f64 * 100.0
    );
    println!(
        "    Over-routed  (Deep but Surface models handle it):   {} ({:.1}%)",
        over_route,
        over_route as f64 / records.len() as f64 * 100.0
    );

    // ─── Per-dataset breakdown ───
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Per-Dataset Breakdown                                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "  {:>20} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "Dataset", "N", "Cheap%", "AXM-A%", "GPT-4%", "Oracl%"
    );
    println!(
        "  {:>20} {:>6} {:>8} {:>8} {:>8} {:>8}",
        "───────", "─", "──────", "──────", "──────", "──────"
    );

    let mut datasets: Vec<String> = ds_axiom_a.keys().cloned().collect();
    datasets.sort_by(|a, b| ds_axiom_a[b].count.cmp(&ds_axiom_a[a].count));

    for ds in &datasets {
        let n = ds_axiom_a[ds].count;
        let cheap_acc = ds_always_cheap.get(ds).map(|m| m.accuracy()).unwrap_or(0.0) * 100.0;
        let axiom_acc = ds_axiom_a[ds].accuracy() * 100.0;
        let gpt4_acc = ds_always_best.get(ds).map(|m| m.accuracy()).unwrap_or(0.0) * 100.0;
        let oracle_acc = ds_oracle.get(ds).map(|m| m.accuracy()).unwrap_or(0.0) * 100.0;
        println!(
            "  {:>20} {:>6} {:>7.1} {:>7.1} {:>7.1} {:>7.1}",
            ds, n, cheap_acc, axiom_acc, gpt4_acc, oracle_acc
        );
    }

    // ─── AIQ approximation ───
    // AIQ = area between router's cost-quality curve and random baseline
    // We compute a simplified version: (router_acc - random_acc) * (1 - router_cost / max_cost)
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  AIQ Approximation                                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Compute points on cost-quality curve for AXIOM by varying confidence thresholds
    // At each threshold, queries above threshold go to Surface, rest to Reasoning/Deep
    let mut curve_points: Vec<(f64, f64)> = vec![
        (always_cheap.avg_cost(), always_cheap.accuracy()),
        (axiom_b.avg_cost(), axiom_b.accuracy()),
        (axiom_a.avg_cost(), axiom_a.accuracy()),
        (axiom_c.avg_cost(), axiom_c.accuracy()),
        (always_best.avg_cost(), always_best.accuracy()),
    ];

    // Sort by cost
    curve_points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    println!("  Cost-Quality curve points:");
    for (cost, quality) in &curve_points {
        println!("    ${:.6}  →  {:.1}%", cost, quality * 100.0);
    }

    // Simple AIQ: normalized area under cost-quality curve
    let c_min = always_cheap.avg_cost();
    let c_max = always_best.avg_cost();
    let range = c_max - c_min;
    if range > 0.0 {
        // Trapezoidal integration of quality over normalized cost
        let mut auc = 0.0;
        for i in 1..curve_points.len() {
            let dc = (curve_points[i].0 - curve_points[i - 1].0) / range;
            let avg_q = (curve_points[i].1 + curve_points[i - 1].1) / 2.0;
            auc += dc * avg_q;
        }
        let random_auc = random_m.accuracy(); // random is flat
        let aiq = auc - random_auc;
        println!("  AUC (normalized): {:.4}", auc);
        println!("  Random baseline:  {:.4}", random_auc);
        println!("  AIQ (AUC - random): {:.4}", aiq);
    }

    // ─── Summary ───
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Summary                                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    let cr_a = (1.0 - axiom_a.avg_cost() / gpt4_cost) * 100.0;
    let cr_b = (1.0 - axiom_b.avg_cost() / gpt4_cost) * 100.0;
    println!(
        "  AXIOM routes {} queries at {:.0} us/query avg",
        records.len(),
        avg_latency_us
    );
    println!(
        "  Strategy A (best-in-tier): {:.1}% accuracy, {:.1}% cost reduction vs GPT-4",
        axiom_a.accuracy() * 100.0,
        cr_a
    );
    println!(
        "  Strategy B (cheapest-in-tier): {:.1}% accuracy, {:.1}% cost reduction vs GPT-4",
        axiom_b.accuracy() * 100.0,
        cr_b
    );
    println!(
        "  GPT-4 alone: {:.1}% accuracy at ${:.6}/query",
        always_best.accuracy() * 100.0,
        always_best.avg_cost()
    );
    println!(
        "  Cheapest alone (Mistral-7B): {:.1}% accuracy at ${:.6}/query",
        always_cheap.accuracy() * 100.0,
        always_cheap.avg_cost()
    );
    println!();
    println!("Done.");
}
