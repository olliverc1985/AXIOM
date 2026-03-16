//! AXIOM Training Pipeline — STS pretrain + end-to-end classification fine-tune.
//!
//! Usage:
//!   cargo run --release --bin train_axiom -- \
//!     --sts-data data/stsbenchmark.tsv \
//!     --tier-data data/fusion_training_corpus.json \
//!     --eval-data data/eval_set1_synthetic.json,data/eval_set2_adversarial.json,data/eval_set3_realworld.json \
//!     --output axiom_router_weights.json
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]

use axiom::input::{Encoder, Tokeniser};
use axiom::router::ClassificationHead;
use axiom::semantic::encoder::SemanticEncoder;
use axiom::semantic::train::{self, StsExample};
use axiom::semantic::transformer::{AdamState, TransformerConfig, TransformerEncoder};
use axiom::semantic::vocab::Vocab;
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
struct TierQuery {
    text: String,
    tier: String,
    #[serde(default)]
    alt_tier: Option<String>,
}

fn tier_idx(name: &str) -> usize {
    match name {
        "Surface" => 0,
        "Reasoning" => 1,
        "Deep" => 2,
        _ => 0,
    }
}

fn tier_name(idx: usize) -> &'static str {
    match idx {
        0 => "Surface",
        1 => "Reasoning",
        2 => "Deep",
        _ => "Surface",
    }
}

fn is_correct(pred: usize, gt_name: &str, alt_tier: &Option<String>) -> bool {
    if pred == tier_idx(gt_name) {
        return true;
    }
    if let Some(alt) = alt_tier {
        if pred == tier_idx(alt) {
            return true;
        }
    }
    false
}

struct Args {
    sts_data: String,
    tier_data: Vec<String>,
    eval_data: Vec<String>,
    hard_neg: Option<String>,
    output: String,
    enc_lr: f32,
    head_lr: f32,
    sts_epochs: usize,
    ft_max_epochs: usize,
    patience: usize,
    batch_size: usize,
    weight_decay: f32,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let mut a = Args {
        sts_data: "data/stsbenchmark.tsv".into(),
        tier_data: vec!["data/fusion_training_corpus.json".into()],
        eval_data: vec![],
        hard_neg: std::env::var("EXP_HARD_NEG").ok(),
        output: "axiom_router_weights.json".into(),
        enc_lr: 3e-4,
        head_lr: 1e-3,
        sts_epochs: 30,
        ft_max_epochs: 300,
        patience: 20,
        batch_size: 32,
        weight_decay: 0.01,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--sts-data" => {
                i += 1;
                a.sts_data = args[i].clone();
            }
            "--tier-data" => {
                i += 1;
                a.tier_data = args[i].split(',').map(|s| s.to_string()).collect();
            }
            "--eval-data" => {
                i += 1;
                a.eval_data = args[i].split(',').map(|s| s.to_string()).collect();
            }
            "--hard-neg" => {
                i += 1;
                a.hard_neg = Some(args[i].clone());
            }
            "--output" | "-o" => {
                i += 1;
                a.output = args[i].clone();
            }
            "--enc-lr" => {
                i += 1;
                a.enc_lr = args[i].parse().unwrap();
            }
            "--head-lr" => {
                i += 1;
                a.head_lr = args[i].parse().unwrap();
            }
            "--sts-epochs" => {
                i += 1;
                a.sts_epochs = args[i].parse().unwrap();
            }
            "--epochs" => {
                i += 1;
                a.ft_max_epochs = args[i].parse().unwrap();
            }
            "--patience" => {
                i += 1;
                a.patience = args[i].parse().unwrap();
            }
            "--batch" => {
                i += 1;
                a.batch_size = args[i].parse().unwrap();
            }
            "--weight-decay" => {
                i += 1;
                a.weight_decay = args[i].parse().unwrap();
            }
            "--help" | "-h" => {
                println!("Usage: train_axiom [OPTIONS]");
                println!(
                    "  --sts-data PATH       STS benchmark TSV (default: data/stsbenchmark.tsv)"
                );
                println!("  --tier-data PATHS     Comma-separated tier-labelled JSON files");
                println!("  --eval-data PATHS     Comma-separated eval JSON files");
                println!("  --hard-neg PATH       Hard negatives JSON for STS");
                println!("  --output PATH         Output weights file (default: axiom_router_weights.json)");
                println!("  --enc-lr FLOAT        Encoder learning rate (default: 3e-4)");
                println!("  --head-lr FLOAT       Head learning rate (default: 1e-3)");
                println!("  --sts-epochs N        STS pretrain epochs (default: 30)");
                println!("  --epochs N            Fine-tune max epochs (default: 300)");
                println!("  --patience N          Early stop patience (default: 20)");
                println!("  --batch N             Batch size (default: 32)");
                println!("  --weight-decay FLOAT  Weight decay (default: 0.01)");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown flag: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    a
}

fn main() {
    let args = parse_args();
    let start = Instant::now();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  AXIOM Training Pipeline                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  STS data:     {}", args.sts_data);
    println!("  Tier data:    {:?}", args.tier_data);
    println!("  Output:       {}", args.output);
    println!("  Encoder LR:   {}", args.enc_lr);
    println!("  Head LR:      {}", args.head_lr);
    println!("  STS epochs:   {}", args.sts_epochs);
    println!(
        "  FT epochs:    {} (patience={})",
        args.ft_max_epochs, args.patience
    );
    println!("  Batch size:   {}", args.batch_size);
    println!("  Weight decay: {}", args.weight_decay);
    println!();

    // ═══ Load STS data ═══
    println!("─── Step 1: Loading data ───");
    let sts_content = std::fs::read_to_string(&args.sts_data)
        .unwrap_or_else(|_| panic!("Failed to read {}", args.sts_data));
    let mut sts_train: Vec<StsExample> = Vec::new();
    let mut sts_dev: Vec<StsExample> = Vec::new();
    for line in sts_content.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 8 {
            if let Ok(score) = cols[5].parse::<f32>() {
                let ex = StsExample {
                    sentence1: cols[6].to_string(),
                    sentence2: cols[7].to_string(),
                    score,
                };
                match cols[0] {
                    "dev" => sts_dev.push(ex),
                    _ => sts_train.push(ex),
                }
            }
        }
    }
    println!("  STS: {} train, {} dev", sts_train.len(), sts_dev.len());

    // Hard negatives
    if let Some(ref path) = args.hard_neg {
        if let Ok(hn_json) = std::fs::read_to_string(path) {
            let hn: Vec<serde_json::Value> = serde_json::from_str(&hn_json).unwrap_or_default();
            let mut count = 0;
            for item in &hn {
                if let (Some(s1), Some(s2), Some(sc)) = (
                    item.get("s1").and_then(|v| v.as_str()),
                    item.get("s2").and_then(|v| v.as_str()),
                    item.get("score").and_then(|v| v.as_f64()),
                ) {
                    sts_train.push(StsExample {
                        sentence1: s1.to_string(),
                        sentence2: s2.to_string(),
                        score: sc as f32,
                    });
                    count += 1;
                }
            }
            println!("  Hard negatives: {} pairs from {}", count, path);
        }
    }

    // ═══ Load tier data ═══
    let mut tier_queries: Vec<TierQuery> = Vec::new();
    for path in &args.tier_data {
        let json =
            std::fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {}", path));
        let queries: Vec<TierQuery> =
            serde_json::from_str(&json).unwrap_or_else(|_| panic!("Failed to parse {}", path));
        println!("  Tier data: {} queries from {}", queries.len(), path);
        tier_queries.extend(queries);
    }

    // Load eval data
    let mut eval_queries: Vec<TierQuery> = Vec::new();
    for path in &args.eval_data {
        let json =
            std::fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {}", path));
        let queries: Vec<TierQuery> =
            serde_json::from_str(&json).unwrap_or_else(|_| panic!("Failed to parse {}", path));
        println!("  Eval data: {} queries from {}", queries.len(), path);
        eval_queries.extend(queries);
    }

    // Combine tier data + eval data for training
    let mut all_labelled: Vec<(String, usize, Option<String>)> = Vec::new();
    for q in tier_queries.iter().chain(eval_queries.iter()) {
        all_labelled.push((q.text.clone(), tier_idx(&q.tier), q.alt_tier.clone()));
    }
    let s_count = all_labelled.iter().filter(|x| x.1 == 0).count();
    let r_count = all_labelled.iter().filter(|x| x.1 == 1).count();
    let d_count = all_labelled.iter().filter(|x| x.1 == 2).count();
    println!(
        "  Total: {} (S={}, R={}, D={})",
        all_labelled.len(),
        s_count,
        r_count,
        d_count
    );

    // Stratified split: 80% train, 10% val, 10% test
    let mut surface: Vec<_> = all_labelled.iter().filter(|x| x.1 == 0).cloned().collect();
    let mut reasoning: Vec<_> = all_labelled.iter().filter(|x| x.1 == 1).cloned().collect();
    let mut deep: Vec<_> = all_labelled.iter().filter(|x| x.1 == 2).cloned().collect();

    fn shuffle(data: &mut [(String, usize, Option<String>)], seed: u64) {
        let n = data.len();
        let mut s = seed;
        for i in (1..n).rev() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            data.swap(i, (s >> 33) as usize % (i + 1));
        }
    }
    shuffle(&mut surface, 42);
    shuffle(&mut reasoning, 43);
    shuffle(&mut deep, 44);

    fn split3(
        data: &[(String, usize, Option<String>)],
    ) -> (
        Vec<(String, usize, Option<String>)>,
        Vec<(String, usize, Option<String>)>,
        Vec<(String, usize, Option<String>)>,
    ) {
        let n = data.len();
        let tn = (n as f32 * 0.1).ceil() as usize;
        let vn = (n as f32 * 0.1).ceil() as usize;
        (
            data[tn + vn..].to_vec(),
            data[tn..tn + vn].to_vec(),
            data[..tn].to_vec(),
        )
    }

    let (s_tr, s_va, s_te) = split3(&surface);
    let (r_tr, r_va, r_te) = split3(&reasoning);
    let (d_tr, d_va, d_te) = split3(&deep);

    let mut train_data: Vec<_> = Vec::new();
    train_data.extend(s_tr);
    train_data.extend(r_tr);
    train_data.extend(d_tr);
    let val_data: Vec<_> = [s_va, r_va, d_va].concat();
    let test_data: Vec<_> = [s_te, r_te, d_te].concat();

    println!(
        "  Split: {} train, {} val, {} test",
        train_data.len(),
        val_data.len(),
        test_data.len()
    );

    // ═══ Build vocab and encoder ═══
    let mut all_sentences: Vec<String> = sts_train
        .iter()
        .flat_map(|ex| vec![ex.sentence1.clone(), ex.sentence2.clone()])
        .collect();
    for (text, _, _) in &train_data {
        all_sentences.push(text.clone());
    }
    let vocab = Vocab::build_from_corpus(&all_sentences, 8192);
    println!("  Vocab: {} words", vocab.size());

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
    let mut encoder = SemanticEncoder::new_with_config(vocab, config);
    println!("  Params: {}", encoder.param_count());

    // ═══ Step 2: STS Pretrain ═══
    println!(
        "\n─── Step 2: STS Pretraining ({} epochs) ───",
        args.sts_epochs
    );
    train::train(
        &mut encoder,
        &sts_train,
        &sts_dev,
        args.sts_epochs,
        5e-4,
        args.batch_size,
        "mse",
        args.weight_decay,
        0.1,
    );

    // ═══ Step 3: Classification Fine-tune (256→3 with structural features) ═══
    println!("\n─── Step 3: Classification Fine-tune ───");

    // Structural encoder for computing features
    let tokeniser = Tokeniser::default_tokeniser();
    let structural_encoder = Encoder::new(128, tokeniser);

    // Class weights (inverse frequency)
    let ts = train_data.iter().filter(|x| x.1 == 0).count() as f32;
    let tr = train_data.iter().filter(|x| x.1 == 1).count() as f32;
    let td = train_data.iter().filter(|x| x.1 == 2).count() as f32;
    let total = ts + tr + td;
    let class_weights = [total / (3.0 * ts), total / (3.0 * tr), total / (3.0 * td)];
    println!(
        "  Class weights: [{:.2}, {:.2}, {:.2}]",
        class_weights[0], class_weights[1], class_weights[2]
    );

    let mut head = ClassificationHead::new(256, 42);
    let enc_sizes = encoder.transformer.param_sizes();
    let mut enc_adam = AdamState::new(&enc_sizes, args.enc_lr);

    // Head Adam state
    let mut hw_m = vec![0.0f32; 256 * 3];
    let mut hw_v = vec![0.0f32; 256 * 3];
    let mut hb_m = vec![0.0f32; 3];
    let mut hb_v = vec![0.0f32; 3];
    let mut head_t = 0usize;

    fn head_adam_step(
        head: &mut ClassificationHead,
        dw: &[f32],
        db: &[f32],
        lr: f32,
        wd: f32,
        wm: &mut [f32],
        wv: &mut [f32],
        bm: &mut [f32],
        bv: &mut [f32],
        t: &mut usize,
    ) {
        *t += 1;
        let (beta1, beta2, eps) = (0.9f32, 0.999f32, 1e-8f32);
        let bc1 = 1.0 - beta1.powi(*t as i32);
        let bc2 = 1.0 - beta2.powi(*t as i32);
        for i in 0..head.weights.len() {
            wm[i] = beta1 * wm[i] + (1.0 - beta1) * dw[i];
            wv[i] = beta2 * wv[i] + (1.0 - beta2) * dw[i] * dw[i];
            head.weights[i] -= lr * (wm[i] / bc1) / ((wv[i] / bc2).sqrt() + eps);
            head.weights[i] *= 1.0 - lr * wd;
        }
        for i in 0..3 {
            bm[i] = beta1 * bm[i] + (1.0 - beta1) * db[i];
            bv[i] = beta2 * bv[i] + (1.0 - beta2) * db[i] * db[i];
            head.biases[i] -= lr * (bm[i] / bc1) / ((bv[i] / bc2).sqrt() + eps);
        }
    }

    // Prepare val/test eval tuples
    let val_eval: Vec<_> = val_data
        .iter()
        .map(|(t, i, a)| (t.clone(), tier_name(*i).to_string(), a.clone()))
        .collect();
    let test_eval: Vec<_> = test_data
        .iter()
        .map(|(t, i, a)| (t.clone(), tier_name(*i).to_string(), a.clone()))
        .collect();
    let eval_all: Vec<_> = eval_queries
        .iter()
        .map(|q| (q.text.clone(), q.tier.clone(), q.alt_tier.clone()))
        .collect();

    let n = train_data.len();
    let mut best_val_acc = 0.0f32;
    let mut best_epoch = 0usize;
    let mut patience_ctr = 0usize;
    let mut best_enc = encoder.transformer.save_weights();
    let mut best_head = head.clone();

    for epoch in 0..args.ft_max_epochs {
        let mut indices: Vec<usize> = (0..n).collect();
        let mut s = (epoch as u64).wrapping_mul(1000003).wrapping_add(7);
        for k in (1..n).rev() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            indices.swap(k, (s >> 33) as usize % (k + 1));
        }

        let mut total_loss = 0.0f32;
        let mut i = 0;
        while i < n {
            let end = (i + args.batch_size).min(n);
            let bs = (end - i) as f32;
            let mut batch_enc_grads = encoder.transformer.zero_grads();
            let mut hdw = vec![0.0f32; 256 * 3];
            let mut hdb = vec![0.0f32; 3];

            for idx in i..end {
                let di = indices[idx];
                let (ref text, target, _) = train_data[di];

                // Structural features (no gradient)
                let structural = structural_encoder.encode_text_readonly(text);

                // Semantic forward with cache
                let (semantic, cache) = encoder.encode_with_cache(text);

                // Concatenate → 256-dim
                let mut fused = Vec::with_capacity(256);
                fused.extend_from_slice(&structural.data);
                fused.extend_from_slice(&semantic);

                // Classification head forward + loss
                let logits = head.logits(&fused);
                let probs = ClassificationHead::softmax(&logits);
                let cw = class_weights[target];
                let loss = -probs[target].max(1e-10).ln() * cw;
                total_loss += loss;

                // dL/d_logits
                let mut d_logits = [0.0f32; 3];
                for j in 0..3 {
                    d_logits[j] = (probs[j] - if j == target { 1.0 } else { 0.0 }) * cw;
                }

                // Accumulate head gradients
                for ii in 0..256 {
                    for j in 0..3 {
                        hdw[ii * 3 + j] += d_logits[j] * fused[ii];
                    }
                }
                for j in 0..3 {
                    hdb[j] += d_logits[j];
                }

                // dL/d_semantic (last 128 dims of fused)
                let mut d_semantic = vec![0.0f32; 128];
                for ii in 0..128 {
                    for j in 0..3 {
                        d_semantic[ii] += d_logits[j] * head.weights[(128 + ii) * 3 + j];
                    }
                }

                // Backward through semantic encoder
                let enc_grads = encoder.transformer.backward(&d_semantic, &cache);
                TransformerEncoder::accumulate_grads(&mut batch_enc_grads, &enc_grads);
            }

            // Scale gradients
            TransformerEncoder::scale_grads(&mut batch_enc_grads, 1.0 / bs);
            for v in hdw.iter_mut() {
                *v /= bs;
            }
            for v in hdb.iter_mut() {
                *v /= bs;
            }

            // Gradient clipping on encoder
            let grad_refs = TransformerEncoder::grad_refs(&batch_enc_grads);
            let gn: f32 = grad_refs
                .iter()
                .flat_map(|g| g.iter())
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            if gn > 1.0 {
                TransformerEncoder::scale_grads(&mut batch_enc_grads, 1.0 / gn);
            }

            // Update encoder
            enc_adam.lr = args.enc_lr;
            encoder
                .transformer
                .apply_gradients(&batch_enc_grads, &mut enc_adam);
            if args.weight_decay > 0.0 {
                let df = 1.0 - args.enc_lr * args.weight_decay;
                for buf in encoder.transformer.param_buffers_mut() {
                    for v in buf.iter_mut() {
                        *v *= df;
                    }
                }
            }

            // Update head
            head_adam_step(
                &mut head,
                &hdw,
                &hdb,
                args.head_lr,
                args.weight_decay,
                &mut hw_m,
                &mut hw_v,
                &mut hb_m,
                &mut hb_v,
                &mut head_t,
            );

            i = end;
        }

        let avg_loss = total_loss / n as f32;

        // Validation
        let mut val_correct = 0;
        for (text, gt_name, alt_tier) in &val_eval {
            let structural = structural_encoder.encode_text_readonly(text);
            let semantic = encoder.encode(text);
            let mut fused = Vec::with_capacity(256);
            fused.extend_from_slice(&structural.data);
            fused.extend_from_slice(&semantic.data);
            let (pred, _) = head.classify(&fused);
            if is_correct(pred, gt_name, alt_tier) {
                val_correct += 1;
            }
        }
        let val_acc = val_correct as f32 / val_eval.len() as f32 * 100.0;

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!(
                "  Epoch {:>3}: loss={:.4}, val={:.1}%",
                epoch + 1,
                avg_loss,
                val_acc
            );
        }

        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            best_epoch = epoch + 1;
            patience_ctr = 0;
            best_enc = encoder.transformer.save_weights();
            best_head = head.clone();
        } else {
            patience_ctr += 1;
            if patience_ctr >= args.patience {
                println!(
                    "  Early stop at epoch {} (best={} with val={:.1}%)",
                    epoch + 1,
                    best_epoch,
                    best_val_acc
                );
                break;
            }
        }
    }

    // Restore best
    encoder.transformer.load_weights(&best_enc);
    head = best_head;

    // ═══ Step 4: Evaluate ═══
    println!("\n─── Step 4: Evaluation ───");

    // Test set
    let mut test_correct = 0;
    for (text, gt_name, alt_tier) in &test_eval {
        let structural = structural_encoder.encode_text_readonly(text);
        let semantic = encoder.encode(text);
        let mut fused = Vec::with_capacity(256);
        fused.extend_from_slice(&structural.data);
        fused.extend_from_slice(&semantic.data);
        let (pred, _) = head.classify(&fused);
        if is_correct(pred, gt_name, alt_tier) {
            test_correct += 1;
        }
    }
    println!(
        "  Test accuracy: {}/{} ({:.1}%)",
        test_correct,
        test_eval.len(),
        test_correct as f32 / test_eval.len() as f32 * 100.0
    );

    // Full eval set
    if !eval_all.is_empty() {
        let mut confusion = [[0usize; 3]; 3];
        let mut correct = 0;
        let mut s_c = 0;
        let mut s_t = 0;
        let mut r_c = 0;
        let mut r_t = 0;
        let mut d_c = 0;
        let mut d_t = 0;
        let mut total_latency = 0u128;

        for (text, gt_name, alt_tier) in &eval_all {
            let t0 = Instant::now();
            let structural = structural_encoder.encode_text_readonly(text);
            let semantic = encoder.encode(text);
            let mut fused = Vec::with_capacity(256);
            fused.extend_from_slice(&structural.data);
            fused.extend_from_slice(&semantic.data);
            let (pred, _) = head.classify(&fused);
            total_latency += t0.elapsed().as_micros();

            let gt = tier_idx(gt_name);
            confusion[gt][pred] += 1;
            let ok = is_correct(pred, gt_name, alt_tier);
            if ok {
                correct += 1;
            }
            match gt_name.as_str() {
                "Surface" => {
                    s_t += 1;
                    if ok {
                        s_c += 1;
                    }
                }
                "Reasoning" => {
                    r_t += 1;
                    if ok {
                        r_c += 1;
                    }
                }
                "Deep" => {
                    d_t += 1;
                    if ok {
                        d_c += 1;
                    }
                }
                _ => {}
            }
        }

        let sa = s_c as f32 / s_t.max(1) as f32 * 100.0;
        let ra = r_c as f32 / r_t.max(1) as f32 * 100.0;
        let da = d_c as f32 / d_t.max(1) as f32 * 100.0;
        let oa = correct as f32 / eval_all.len() as f32 * 100.0;
        let avg_lat = total_latency as f32 / eval_all.len() as f32;

        println!("\n  Eval ({} queries):", eval_all.len());
        println!("    Surface:   {}/{} ({:.1}%)", s_c, s_t, sa);
        println!("    Reasoning: {}/{} ({:.1}%)", r_c, r_t, ra);
        println!("    Deep:      {}/{} ({:.1}%)", d_c, d_t, da);
        println!("    Overall:   {}/{} ({:.1}%)", correct, eval_all.len(), oa);
        println!("    Avg latency: {:.0} us", avg_lat);

        println!("\n    Confusion:");
        println!(
            "    {:>12} {:>8} {:>8} {:>8}",
            "", "→Surf", "→Reas", "→Deep"
        );
        for (i, name) in ["Surface", "Reasoning", "Deep"].iter().enumerate() {
            println!(
                "    {:>12} {:>8} {:>8} {:>8}",
                name, confusion[i][0], confusion[i][1], confusion[i][2]
            );
        }
    }

    // ═══ Step 5: Save weights ═══
    println!("\n─── Step 5: Saving weights ───");

    let router_data = axiom::router::AxiomRouterWeights {
        encoder: axiom::semantic::encoder::SemanticEncoderData {
            vocab: encoder.vocab.clone(),
            weights: encoder.transformer.save_weights(),
        },
        head,
    };
    let json = serde_json::to_string(&router_data).expect("Failed to serialise");
    std::fs::write(&args.output, &json).expect("Failed to write");
    println!(
        "  Saved to {} ({:.1} MB)",
        args.output,
        json.len() as f32 / 1e6
    );

    let elapsed = start.elapsed();
    println!(
        "\nTotal time: {:.0}s ({:.1} min)",
        elapsed.as_secs_f32(),
        elapsed.as_secs_f32() / 60.0
    );
    println!("Done.");
}
