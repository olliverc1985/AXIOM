use super::encoder::SemanticEncoder;
use super::transformer::{AdamState, TransformerEncoder};

#[derive(Debug, Clone)]
pub struct StsExample {
    pub sentence1: String,
    pub sentence2: String,
    pub score: f32,
}

/// Load STS Benchmark TSV file.
pub fn load_sts_data(path: &str) -> Vec<StsExample> {
    let content = std::fs::read_to_string(path).expect("Failed to read STS file");
    let mut examples = Vec::new();
    for line in content.lines().skip(1) {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 8 {
            if let Ok(score) = cols[5].parse::<f32>() {
                examples.push(StsExample {
                    sentence1: cols[6].to_string(),
                    sentence2: cols[7].to_string(),
                    score,
                });
            }
        }
    }
    examples
}

/// Cosine similarity between two slices.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Gradient of cosine similarity w.r.t. vector a.
fn d_cosine_da(a: &[f32], b: &[f32]) -> Vec<f32> {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-8 || nb < 1e-8 {
        return vec![0.0; a.len()];
    }
    let cos = dot / (na * nb);
    let inv_na_nb = 1.0 / (na * nb);
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| bi * inv_na_nb - cos * ai / (na * na))
        .collect()
}

/// Contrastive loss: pulls similar pairs together, pushes dissimilar apart.
/// Uses cosine distance. margin controls the boundary.
fn contrastive_loss_and_grad(
    emb_a: &[f32],
    emb_b: &[f32],
    target: f32,
    margin: f32,
) -> (f32, Vec<f32>, Vec<f32>) {
    let cos = cosine_sim(emb_a, emb_b);
    let dist = 1.0 - cos;

    if target > 0.5 {
        // Similar pair: loss = dist^2
        let loss = dist * dist;
        let d_cos = -2.0 * dist; // d(dist^2)/d(cos) = -2*(1-cos)
        let d_a: Vec<f32> = d_cosine_da(emb_a, emb_b)
            .iter()
            .map(|&v| v * d_cos)
            .collect();
        let d_b: Vec<f32> = d_cosine_da(emb_b, emb_a)
            .iter()
            .map(|&v| v * d_cos)
            .collect();
        (loss, d_a, d_b)
    } else {
        // Dissimilar pair: loss = max(0, margin - dist)^2
        let diff = (margin - dist).max(0.0);
        let loss = diff * diff;
        if diff <= 0.0 {
            return (loss, vec![0.0; emb_a.len()], vec![0.0; emb_b.len()]);
        }
        let d_cos = 2.0 * diff; // d(max(0,m-d)^2)/d(cos) = 2*(m-d)*1
        let d_a: Vec<f32> = d_cosine_da(emb_a, emb_b)
            .iter()
            .map(|&v| v * d_cos)
            .collect();
        let d_b: Vec<f32> = d_cosine_da(emb_b, emb_a)
            .iter()
            .map(|&v| v * d_cos)
            .collect();
        (loss, d_a, d_b)
    }
}

/// Word dropout: drop each word with probability drop_rate, using deterministic PRNG.
/// At least 1 word must remain.
pub fn word_dropout(text: &str, drop_rate: f32, seed: u64) -> String {
    if drop_rate <= 0.0 {
        return text.to_string();
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= 1 {
        return text.to_string();
    }
    let mut kept = Vec::new();
    let mut state = seed;
    for (i, word) in words.iter().enumerate() {
        // Simple xorshift PRNG
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(i as u64);
        state ^= state >> 17;
        let r = (state as u32) as f32 / u32::MAX as f32;
        if r >= drop_rate {
            kept.push(*word);
        }
    }
    // At least 1 word must remain
    if kept.is_empty() {
        // Keep the first word deterministically
        kept.push(words[0]);
    }
    kept.join(" ")
}

/// Train the semantic encoder on STS data.
pub fn train(
    encoder: &mut SemanticEncoder,
    train_data: &[StsExample],
    dev_data: &[StsExample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    loss_fn: &str,
    weight_decay: f32,
    warmup_frac: f32,
) {
    let sizes = encoder.transformer.param_sizes();
    let mut adam = AdamState::new(&sizes, lr);

    let n = train_data.len();
    let total_steps = n.div_ceil(batch_size) * epochs;
    let warmup_steps = (warmup_frac * total_steps as f32) as usize;
    let mut global_step = 0usize;

    println!("Training semantic encoder: {} examples, {} epochs, lr={}, batch={}, loss={}, wd={}, warmup={:.2}",
        n, epochs, lr, batch_size, loss_fn, weight_decay, warmup_frac);

    for epoch in 0..epochs {
        let mut total_loss = 0.0f32;
        let mut count = 0usize;

        // Simple sequential training (deterministic order, shuffle would need PRNG)
        let mut i = 0;
        while i < n {
            let end = (i + batch_size).min(n);
            let mut batch_grads = encoder.transformer.zero_grads();
            let mut batch_loss = 0.0f32;

            // Compute effective learning rate with warmup
            let effective_lr = if warmup_steps > 0 && global_step < warmup_steps {
                lr * (global_step as f32 + 1.0) / warmup_steps as f32
            } else {
                lr
            };
            adam.lr = effective_lr;

            for j in i..end {
                let ex = &train_data[j];

                // Forward both sentences
                let (emb_a, cache_a) = encoder.encode_with_cache(&ex.sentence1);
                let (emb_b, cache_b) = encoder.encode_with_cache(&ex.sentence2);

                let (loss, d_emb_a, d_emb_b) = match loss_fn {
                    "contrastive" => {
                        // Threshold target at 0.6 (scores > 3.0 are "similar")
                        let binary_target = if ex.score > 3.0 { 1.0 } else { 0.0 };
                        contrastive_loss_and_grad(&emb_a, &emb_b, binary_target, 0.5)
                    }
                    _ => {
                        // MSE loss (default)
                        let target = ex.score / 5.0;
                        let pred = cosine_sim(&emb_a, &emb_b);
                        let diff = pred - target;
                        let loss = diff * diff;
                        let d_loss = 2.0 * diff;
                        let d_a: Vec<f32> = d_cosine_da(&emb_a, &emb_b)
                            .iter()
                            .map(|&v| v * d_loss)
                            .collect();
                        let d_b: Vec<f32> = d_cosine_da(&emb_b, &emb_a)
                            .iter()
                            .map(|&v| v * d_loss)
                            .collect();
                        (loss, d_a, d_b)
                    }
                };

                batch_loss += loss;

                // Backward through transformer for both sentences
                let grads_a = encoder.transformer.backward(&d_emb_a, &cache_a);
                let grads_b = encoder.transformer.backward(&d_emb_b, &cache_b);

                TransformerEncoder::accumulate_grads(&mut batch_grads, &grads_a);
                TransformerEncoder::accumulate_grads(&mut batch_grads, &grads_b);
            }

            let bs = (end - i) as f32;
            TransformerEncoder::scale_grads(&mut batch_grads, 1.0 / bs);

            // Gradient clipping (max norm = 1.0)
            let grad_refs = TransformerEncoder::grad_refs(&batch_grads);
            let grad_norm: f32 = grad_refs
                .iter()
                .flat_map(|g| g.iter())
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            if grad_norm > 1.0 {
                TransformerEncoder::scale_grads(&mut batch_grads, 1.0 / grad_norm);
            }

            encoder.transformer.apply_gradients(&batch_grads, &mut adam);

            // AdamW weight decay: param[i] *= 1.0 - lr * weight_decay
            if weight_decay > 0.0 {
                let decay_factor = 1.0 - effective_lr * weight_decay;
                let bufs = encoder.transformer.param_buffers_mut();
                for buf in bufs {
                    for v in buf.iter_mut() {
                        *v *= decay_factor;
                    }
                }
            }

            total_loss += batch_loss;
            count += end - i;
            i = end;
            global_step += 1;
        }

        let avg_loss = total_loss / count as f32;

        // Dev evaluation
        if !dev_data.is_empty() {
            let dev_corr = evaluate(encoder, dev_data);
            println!(
                "  Epoch {}/{}: loss={:.6}, dev_pearson={:.4}",
                epoch + 1,
                epochs,
                avg_loss,
                dev_corr
            );
        } else {
            println!("  Epoch {}/{}: loss={:.6}", epoch + 1, epochs, avg_loss);
        }
    }
}

/// Train with word dropout augmentation. Wraps `train` but applies word dropout before each epoch.
pub fn train_with_augmentation(
    encoder: &mut SemanticEncoder,
    train_data: &[StsExample],
    dev_data: &[StsExample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    loss_fn: &str,
    weight_decay: f32,
    warmup_frac: f32,
    drop_rate: f32,
) {
    if drop_rate <= 0.0 {
        train(
            encoder,
            train_data,
            dev_data,
            epochs,
            lr,
            batch_size,
            loss_fn,
            weight_decay,
            warmup_frac,
        );
        return;
    }

    println!("Training with word dropout rate={}", drop_rate);

    // Train one epoch at a time with augmented data
    for epoch in 0..epochs {
        let seed_base = (epoch as u64).wrapping_mul(1000003);
        let augmented: Vec<StsExample> = train_data
            .iter()
            .enumerate()
            .map(|(i, ex)| {
                let s1 = word_dropout(
                    &ex.sentence1,
                    drop_rate,
                    seed_base.wrapping_add(i as u64 * 2),
                );
                let s2 = word_dropout(
                    &ex.sentence2,
                    drop_rate,
                    seed_base.wrapping_add(i as u64 * 2 + 1),
                );
                StsExample {
                    sentence1: s1,
                    sentence2: s2,
                    score: ex.score,
                }
            })
            .collect();

        // Train 1 epoch on augmented data, passing remaining warmup state
        // We call train with epochs=1 to process one epoch at a time
        train(
            encoder,
            &augmented,
            dev_data,
            1,
            lr,
            batch_size,
            loss_fn,
            weight_decay,
            warmup_frac,
        );
    }
}

/// GPU-accelerated training. Same logic as `train` but uses Metal compute.
#[cfg(feature = "gpu")]
pub fn train_gpu(
    encoder: &mut SemanticEncoder,
    train_data: &[StsExample],
    dev_data: &[StsExample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    loss_fn: &str,
    weight_decay: f32,
    warmup_frac: f32,
    gpu: &crate::gpu::GpuContext,
) {
    let sizes = encoder.transformer.param_sizes();
    let mut adam = AdamState::new(&sizes, lr);

    let n = train_data.len();
    let total_steps = ((n + batch_size - 1) / batch_size) * epochs;
    let warmup_steps = (warmup_frac * total_steps as f32) as usize;
    let mut global_step = 0usize;

    println!("Training semantic encoder (GPU): {} examples, {} epochs, lr={}, batch={}, loss={}, wd={}, warmup={:.2}",
        n, epochs, lr, batch_size, loss_fn, weight_decay, warmup_frac);

    for epoch in 0..epochs {
        let mut total_loss = 0.0f32;
        let mut count = 0usize;

        let mut i = 0;
        while i < n {
            let end = (i + batch_size).min(n);
            let mut batch_grads = encoder.transformer.zero_grads();
            let mut batch_loss = 0.0f32;

            let effective_lr = if warmup_steps > 0 && global_step < warmup_steps {
                lr * (global_step as f32 + 1.0) / warmup_steps as f32
            } else {
                lr
            };
            adam.lr = effective_lr;

            for j in i..end {
                let ex = &train_data[j];

                // GPU-accelerated forward
                let (emb_a, cache_a) = encoder.encode_with_cache_gpu(&ex.sentence1, gpu);
                let (emb_b, cache_b) = encoder.encode_with_cache_gpu(&ex.sentence2, gpu);

                let (loss, d_emb_a, d_emb_b) = match loss_fn {
                    "contrastive" => {
                        let binary_target = if ex.score > 3.0 { 1.0 } else { 0.0 };
                        contrastive_loss_and_grad(&emb_a, &emb_b, binary_target, 0.5)
                    }
                    _ => {
                        let target = ex.score / 5.0;
                        let pred = cosine_sim(&emb_a, &emb_b);
                        let diff = pred - target;
                        let loss = diff * diff;
                        let d_loss = 2.0 * diff;
                        let d_a: Vec<f32> = d_cosine_da(&emb_a, &emb_b)
                            .iter()
                            .map(|&v| v * d_loss)
                            .collect();
                        let d_b: Vec<f32> = d_cosine_da(&emb_b, &emb_a)
                            .iter()
                            .map(|&v| v * d_loss)
                            .collect();
                        (loss, d_a, d_b)
                    }
                };

                batch_loss += loss;

                // GPU-accelerated backward
                let grads_a = encoder.transformer.backward_gpu(&d_emb_a, &cache_a, gpu);
                let grads_b = encoder.transformer.backward_gpu(&d_emb_b, &cache_b, gpu);

                TransformerEncoder::accumulate_grads(&mut batch_grads, &grads_a);
                TransformerEncoder::accumulate_grads(&mut batch_grads, &grads_b);
            }

            let bs = (end - i) as f32;
            TransformerEncoder::scale_grads(&mut batch_grads, 1.0 / bs);

            let grad_refs = TransformerEncoder::grad_refs(&batch_grads);
            let grad_norm: f32 = grad_refs
                .iter()
                .flat_map(|g| g.iter())
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            if grad_norm > 1.0 {
                TransformerEncoder::scale_grads(&mut batch_grads, 1.0 / grad_norm);
            }

            encoder.transformer.apply_gradients(&batch_grads, &mut adam);

            if weight_decay > 0.0 {
                let decay_factor = 1.0 - effective_lr * weight_decay;
                let bufs = encoder.transformer.param_buffers_mut();
                for buf in bufs {
                    for v in buf.iter_mut() {
                        *v *= decay_factor;
                    }
                }
            }

            total_loss += batch_loss;
            count += end - i;
            i = end;
            global_step += 1;
        }

        let avg_loss = total_loss / count as f32;

        if !dev_data.is_empty() {
            let dev_corr = evaluate(encoder, dev_data);
            println!(
                "  Epoch {}/{}: loss={:.6}, dev_pearson={:.4}",
                epoch + 1,
                epochs,
                avg_loss,
                dev_corr
            );
        } else {
            println!("  Epoch {}/{}: loss={:.6}", epoch + 1, epochs, avg_loss);
        }
    }
}

/// GPU-accelerated training with word dropout.
#[cfg(feature = "gpu")]
pub fn train_with_augmentation_gpu(
    encoder: &mut SemanticEncoder,
    train_data: &[StsExample],
    dev_data: &[StsExample],
    epochs: usize,
    lr: f32,
    batch_size: usize,
    loss_fn: &str,
    weight_decay: f32,
    warmup_frac: f32,
    drop_rate: f32,
    gpu: &crate::gpu::GpuContext,
) {
    if drop_rate <= 0.0 {
        train_gpu(
            encoder,
            train_data,
            dev_data,
            epochs,
            lr,
            batch_size,
            loss_fn,
            weight_decay,
            warmup_frac,
            gpu,
        );
        return;
    }

    println!("Training with word dropout rate={} (GPU)", drop_rate);

    for epoch in 0..epochs {
        let seed_base = (epoch as u64).wrapping_mul(1000003);
        let augmented: Vec<StsExample> = train_data
            .iter()
            .enumerate()
            .map(|(i, ex)| {
                let s1 = word_dropout(
                    &ex.sentence1,
                    drop_rate,
                    seed_base.wrapping_add(i as u64 * 2),
                );
                let s2 = word_dropout(
                    &ex.sentence2,
                    drop_rate,
                    seed_base.wrapping_add(i as u64 * 2 + 1),
                );
                StsExample {
                    sentence1: s1,
                    sentence2: s2,
                    score: ex.score,
                }
            })
            .collect();

        train_gpu(
            encoder,
            &augmented,
            dev_data,
            1,
            lr,
            batch_size,
            loss_fn,
            weight_decay,
            warmup_frac,
            gpu,
        );
    }
}

/// Evaluate Pearson correlation on STS data.
pub fn evaluate(encoder: &SemanticEncoder, data: &[StsExample]) -> f32 {
    let mut preds = Vec::with_capacity(data.len());
    let mut golds = Vec::with_capacity(data.len());
    for ex in data {
        let emb_a = encoder.encode(&ex.sentence1);
        let emb_b = encoder.encode(&ex.sentence2);
        preds.push(emb_a.cosine_similarity(&emb_b));
        golds.push(ex.score / 5.0);
    }
    pearson(&preds, &golds)
}

pub fn pearson(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mx: f32 = x.iter().sum::<f32>() / n;
    let my: f32 = y.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut vx = 0.0f32;
    let mut vy = 0.0f32;
    for (a, b) in x.iter().zip(y.iter()) {
        let dx = a - mx;
        let dy = b - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx < 1e-12 || vy < 1e-12 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}
