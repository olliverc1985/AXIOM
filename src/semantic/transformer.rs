use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════
// BLAS — Apple Accelerate (system framework, zero external deps)
// ═══════════════════════════════════════════════════════════════════

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

// ═══════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════

fn default_pooling() -> String {
    "mean".to_string()
}
fn default_activation() -> String {
    "gelu".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ff_dim: usize,
    pub max_seq_len: usize,
    #[serde(default = "default_pooling")]
    pub pooling: String,
    #[serde(default = "default_activation")]
    pub activation: String,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8192,
            hidden_dim: 128,
            num_heads: 4,
            num_layers: 2,
            ff_dim: 512,
            max_seq_len: 128,
            pooling: "mean".to_string(),
            activation: "gelu".to_string(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Weight structures
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerLayer {
    pub ln1_gamma: Vec<f32>,
    pub ln1_beta: Vec<f32>,
    pub w_q: Vec<f32>,
    pub b_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub b_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub b_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub b_o: Vec<f32>,
    pub ln2_gamma: Vec<f32>,
    pub ln2_beta: Vec<f32>,
    pub ff_w1: Vec<f32>,
    pub ff_b1: Vec<f32>,
    pub ff_w2: Vec<f32>,
    pub ff_b2: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerEncoder {
    pub config: TransformerConfig,
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub layers: Vec<TransformerLayer>,
    pub final_ln_gamma: Vec<f32>,
    pub final_ln_beta: Vec<f32>,
}

// ═══════════════════════════════════════════════════════════════════
// Deterministic PRNG — matches codebase pattern
// ═══════════════════════════════════════════════════════════════════

fn xorshift(mut s: u32) -> u32 {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    s
}

fn rand_f32(seed: u32) -> f32 {
    (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn xavier_vec(size: usize, fan_in: usize, seed: u64) -> Vec<f32> {
    let scale = 1.0 / (fan_in as f32).sqrt();
    (0..size)
        .map(|i| {
            let s = xorshift(
                (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed as u32),
            );
            rand_f32(s) * scale
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// Forward helpers
// ═══════════════════════════════════════════════════════════════════

const LN_EPS: f32 = 1e-5;

/// Layer norm forward. Returns (output, x_hat, inv_std).
fn ln_fwd(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    s: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0; s * d];
    let mut x_hat = vec![0.0; s * d];
    let mut inv_std = vec![0.0; s];
    for t in 0..s {
        let off = t * d;
        let mean: f32 = x[off..off + d].iter().sum::<f32>() / d as f32;
        let var: f32 = x[off..off + d]
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f32>()
            / d as f32;
        let is = 1.0 / (var + LN_EPS).sqrt();
        inv_std[t] = is;
        for i in 0..d {
            x_hat[off + i] = (x[off + i] - mean) * is;
            out[off + i] = gamma[i] * x_hat[off + i] + beta[i];
        }
    }
    (out, x_hat, inv_std)
}

/// Layer norm backward. Returns (d_input, d_gamma, d_beta).
fn ln_bwd(
    dout: &[f32],
    x_hat: &[f32],
    inv_std: &[f32],
    gamma: &[f32],
    s: usize,
    d: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0; s * d];
    let mut dgamma = vec![0.0; d];
    let mut dbeta = vec![0.0; d];
    let inv_d = 1.0 / d as f32;
    for t in 0..s {
        let off = t * d;
        let is = inv_std[t];
        // d_xhat
        let mut dxhat = vec![0.0; d];
        for i in 0..d {
            dxhat[i] = dout[off + i] * gamma[i];
            dgamma[i] += dout[off + i] * x_hat[off + i];
            dbeta[i] += dout[off + i];
        }
        let mean_dxhat: f32 = dxhat.iter().sum::<f32>() * inv_d;
        let mean_dxhat_xhat: f32 = dxhat
            .iter()
            .zip(&x_hat[off..off + d])
            .map(|(a, b)| a * b)
            .sum::<f32>()
            * inv_d;
        for i in 0..d {
            dx[off + i] = is * (dxhat[i] - mean_dxhat - x_hat[off + i] * mean_dxhat_xhat);
        }
    }
    (dx, dgamma, dbeta)
}

/// Linear forward: [S, in_d] @ [in_d, out_d] + [out_d] → [S, out_d]
/// Uses Apple Accelerate BLAS on macOS for optimized SIMD + cache tiling.
#[cfg(target_os = "macos")]
fn linear_fwd(x: &[f32], w: &[f32], b: &[f32], s: usize, id: usize, od: usize) -> Vec<f32> {
    let mut out = vec![0.0; s * od];
    // Init with bias
    for t in 0..s {
        out[t * od..(t + 1) * od].copy_from_slice(b);
    }
    // C = alpha * A @ B + beta * C
    // A: [S, id], B: [id, od], C: [S, od]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            s as i32,  // M
            od as i32, // N
            id as i32, // K
            1.0,       // alpha
            x.as_ptr(),
            id as i32, // lda
            w.as_ptr(),
            od as i32, // ldb
            1.0,       // beta (add to bias-initialized output)
            out.as_mut_ptr(),
            od as i32, // ldc
        );
    }
    out
}

#[cfg(not(target_os = "macos"))]
fn linear_fwd(x: &[f32], w: &[f32], b: &[f32], s: usize, id: usize, od: usize) -> Vec<f32> {
    let mut out = vec![0.0; s * od];
    for t in 0..s {
        let out_off = t * od;
        let x_off = t * id;
        out[out_off..out_off + od].copy_from_slice(b);
        for i in 0..id {
            let xi = x[x_off + i];
            let w_off = i * od;
            for j in 0..od {
                out[out_off + j] += xi * w[w_off + j];
            }
        }
    }
    out
}

/// Linear backward. Returns (d_input, d_weight, d_bias).
/// Uses Apple Accelerate BLAS on macOS.
#[cfg(target_os = "macos")]
fn linear_bwd(
    dout: &[f32],
    x: &[f32],
    w: &[f32],
    s: usize,
    id: usize,
    od: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0; s * id];
    let mut dw = vec![0.0; id * od];
    let mut db = vec![0.0; od];

    // db = sum(dout, axis=0)
    for t in 0..s {
        for j in 0..od {
            db[j] += dout[t * od + j];
        }
    }

    // dx = dout @ W^T: [S, od] @ [od, id] → [S, id]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS, // W^T
            s as i32,    // M
            id as i32,   // N
            od as i32,   // K
            1.0,
            dout.as_ptr(),
            od as i32, // lda
            w.as_ptr(),
            od as i32, // ldb (W is [id, od], transposed access)
            0.0,
            dx.as_mut_ptr(),
            id as i32, // ldc
        );
    }

    // dw = X^T @ dout: [id, S] @ [S, od] → [id, od]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS, // X^T
            CBLAS_NO_TRANS,
            id as i32, // M
            od as i32, // N
            s as i32,  // K
            1.0,
            x.as_ptr(),
            id as i32, // lda (X is [S, id], transposed access)
            dout.as_ptr(),
            od as i32, // ldb
            0.0,
            dw.as_mut_ptr(),
            od as i32, // ldc
        );
    }

    (dx, dw, db)
}

#[cfg(not(target_os = "macos"))]
fn linear_bwd(
    dout: &[f32],
    x: &[f32],
    w: &[f32],
    s: usize,
    id: usize,
    od: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0; s * id];
    let mut dw = vec![0.0; id * od];
    let mut db = vec![0.0; od];
    for t in 0..s {
        for j in 0..od {
            let d = dout[t * od + j];
            db[j] += d;
            for i in 0..id {
                dx[t * id + i] += d * w[i * od + j];
                dw[i * od + j] += x[t * id + i] * d;
            }
        }
    }
    (dx, dw, db)
}

/// GELU forward (sigmoid approximation): x * sigmoid(1.702 * x)
fn gelu_fwd(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let s = 1.0 / (1.0 + (-1.702 * v).exp());
            v * s
        })
        .collect()
}

/// GELU backward.
fn gelu_bwd(dout: &[f32], x: &[f32]) -> Vec<f32> {
    dout.iter()
        .zip(x.iter())
        .map(|(&d, &v)| {
            let s = 1.0 / (1.0 + (-1.702 * v).exp());
            d * (s + v * 1.702 * s * (1.0 - s))
        })
        .collect()
}

/// ReLU forward.
fn relu_fwd(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// ReLU backward.
fn relu_bwd(dout: &[f32], x: &[f32]) -> Vec<f32> {
    dout.iter()
        .zip(x.iter())
        .map(|(&d, &v)| if v > 0.0 { d } else { 0.0 })
        .collect()
}

/// SiLU (Swish) forward: x * sigmoid(x).
fn silu_fwd(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let s = 1.0 / (1.0 + (-v).exp());
            v * s
        })
        .collect()
}

/// SiLU (Swish) backward.
fn silu_bwd(dout: &[f32], x: &[f32]) -> Vec<f32> {
    dout.iter()
        .zip(x.iter())
        .map(|(&d, &v)| {
            let s = 1.0 / (1.0 + (-v).exp());
            d * (s + v * s * (1.0 - s))
        })
        .collect()
}

/// Dispatch activation forward based on name.
fn activation_fwd(x: &[f32], name: &str) -> Vec<f32> {
    match name {
        "relu" => relu_fwd(x),
        "silu" => silu_fwd(x),
        _ => gelu_fwd(x),
    }
}

/// Dispatch activation backward based on name.
fn activation_bwd(dout: &[f32], x: &[f32], name: &str) -> Vec<f32> {
    match name {
        "relu" => relu_bwd(dout, x),
        "silu" => silu_bwd(dout, x),
        _ => gelu_bwd(dout, x),
    }
}

/// Row-wise softmax forward.
fn softmax_fwd(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        let off = r * cols;
        let max_val = x[off..off + cols]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for c in 0..cols {
            let e = (x[off + c] - max_val).exp();
            out[off + c] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for c in 0..cols {
            out[off + c] *= inv;
        }
    }
    out
}

/// Row-wise softmax backward.
fn softmax_bwd(dout: &[f32], y: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut dx = vec![0.0; rows * cols];
    for r in 0..rows {
        let off = r * cols;
        let dot: f32 = (0..cols).map(|c| dout[off + c] * y[off + c]).sum();
        for c in 0..cols {
            dx[off + c] = y[off + c] * (dout[off + c] - dot);
        }
    }
    dx
}

// ═══════════════════════════════════════════════════════════════════
// Forward cache
// ═══════════════════════════════════════════════════════════════════

pub struct LayerCache {
    pub input: Vec<f32>,
    pub ln1_xhat: Vec<f32>,
    pub ln1_inv_std: Vec<f32>,
    pub ln1_out: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_weights: Vec<f32>,
    pub concat: Vec<f32>,
    pub post_attn: Vec<f32>,
    pub ln2_xhat: Vec<f32>,
    pub ln2_inv_std: Vec<f32>,
    pub ln2_out: Vec<f32>,
    pub ff_pre_gelu: Vec<f32>,
    pub ff_act: Vec<f32>,
}

pub struct ForwardCache {
    pub token_ids: Vec<usize>,
    pub seq_len: usize,
    pub embedded: Vec<f32>,
    pub layer_caches: Vec<LayerCache>,
    pub pre_final_ln: Vec<f32>,
    pub final_ln_xhat: Vec<f32>,
    pub final_ln_inv_std: Vec<f32>,
    pub final_output: Vec<f32>,
    /// For "max" pooling: which token provided the max for each hidden dim.
    pub pooling_max_indices: Vec<usize>,
}

// ═══════════════════════════════════════════════════════════════════
// Gradient structures
// ═══════════════════════════════════════════════════════════════════

pub struct LayerGradients {
    pub d_ln1_gamma: Vec<f32>,
    pub d_ln1_beta: Vec<f32>,
    pub d_w_q: Vec<f32>,
    pub d_b_q: Vec<f32>,
    pub d_w_k: Vec<f32>,
    pub d_b_k: Vec<f32>,
    pub d_w_v: Vec<f32>,
    pub d_b_v: Vec<f32>,
    pub d_w_o: Vec<f32>,
    pub d_b_o: Vec<f32>,
    pub d_ln2_gamma: Vec<f32>,
    pub d_ln2_beta: Vec<f32>,
    pub d_ff_w1: Vec<f32>,
    pub d_ff_b1: Vec<f32>,
    pub d_ff_w2: Vec<f32>,
    pub d_ff_b2: Vec<f32>,
}

pub struct TransformerGradients {
    pub d_token_emb: Vec<f32>,
    pub d_pos_emb: Vec<f32>,
    pub layers: Vec<LayerGradients>,
    pub d_final_ln_gamma: Vec<f32>,
    pub d_final_ln_beta: Vec<f32>,
}

// ═══════════════════════════════════════════════════════════════════
// Adam optimizer state
// ═══════════════════════════════════════════════════════════════════

pub struct AdamState {
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
    pub t: usize,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl AdamState {
    pub fn new(param_sizes: &[usize], lr: f32) -> Self {
        Self {
            m: param_sizes.iter().map(|&s| vec![0.0; s]).collect(),
            v: param_sizes.iter().map(|&s| vec![0.0; s]).collect(),
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    pub fn step(&mut self, params: &mut [&mut Vec<f32>], grads: &[&Vec<f32>]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for (idx, (p, g)) in params.iter_mut().zip(grads.iter()).enumerate() {
            for i in 0..p.len() {
                self.m[idx][i] = self.beta1 * self.m[idx][i] + (1.0 - self.beta1) * g[i];
                self.v[idx][i] = self.beta2 * self.v[idx][i] + (1.0 - self.beta2) * g[i] * g[i];
                let mh = self.m[idx][i] / bc1;
                let vh = self.v[idx][i] / bc2;
                p[i] -= self.lr * mh / (vh.sqrt() + self.eps);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// TransformerEncoder implementation
// ═══════════════════════════════════════════════════════════════════

impl TransformerEncoder {
    pub fn new(config: TransformerConfig) -> Self {
        let h = config.hidden_dim;
        let ff = config.ff_dim;
        let v = config.vocab_size;
        let ms = config.max_seq_len;

        let mut seed: u64 = 42;
        let mut next_seed = || -> u64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };

        let token_emb = xavier_vec(v * h, h, next_seed());
        let pos_emb = xavier_vec(ms * h, h, next_seed());

        let layers: Vec<TransformerLayer> = (0..config.num_layers)
            .map(|_| TransformerLayer {
                ln1_gamma: vec![1.0; h],
                ln1_beta: vec![0.0; h],
                w_q: xavier_vec(h * h, h, next_seed()),
                b_q: vec![0.0; h],
                w_k: xavier_vec(h * h, h, next_seed()),
                b_k: vec![0.0; h],
                w_v: xavier_vec(h * h, h, next_seed()),
                b_v: vec![0.0; h],
                w_o: xavier_vec(h * h, h, next_seed()),
                b_o: vec![0.0; h],
                ln2_gamma: vec![1.0; h],
                ln2_beta: vec![0.0; h],
                ff_w1: xavier_vec(h * ff, h, next_seed()),
                ff_b1: vec![0.0; ff],
                ff_w2: xavier_vec(ff * h, ff, next_seed()),
                ff_b2: vec![0.0; h],
            })
            .collect();

        Self {
            config,
            token_emb,
            pos_emb,
            layers,
            final_ln_gamma: vec![1.0; h],
            final_ln_beta: vec![0.0; h],
        }
    }

    pub fn param_count(&self) -> usize {
        let h = self.config.hidden_dim;
        let ff = self.config.ff_dim;
        let emb = self.token_emb.len() + self.pos_emb.len();
        let per_layer = h * 2  // ln1
            + h * h * 4 + h * 4  // qkvo weights + biases
            + h * 2  // ln2
            + h * ff + ff + ff * h + h; // ffn
        emb + per_layer * self.config.num_layers + h * 2
    }

    /// Forward pass. Returns (embedding [H], cache for backward).
    pub fn forward(&self, token_ids: &[usize]) -> (Vec<f32>, ForwardCache) {
        let h = self.config.hidden_dim;
        let s = token_ids.len().min(self.config.max_seq_len);
        let nh = self.config.num_heads;
        let hd = h / nh;

        // Embedding lookup + position
        let mut x = vec![0.0; s * h];
        for t in 0..s {
            let tid = token_ids[t].min(self.config.vocab_size - 1);
            for i in 0..h {
                x[t * h + i] = self.token_emb[tid * h + i] + self.pos_emb[t * h + i];
            }
        }
        let embedded = x.clone();

        let mut layer_caches = Vec::with_capacity(self.config.num_layers);

        for layer in &self.layers {
            let input = x.clone();

            // Pre-LN attention
            let (ln1_out, ln1_xhat, ln1_inv_std) =
                ln_fwd(&x, &layer.ln1_gamma, &layer.ln1_beta, s, h);
            let q = linear_fwd(&ln1_out, &layer.w_q, &layer.b_q, s, h, h);
            let k = linear_fwd(&ln1_out, &layer.w_k, &layer.b_k, s, h, h);
            let v = linear_fwd(&ln1_out, &layer.w_v, &layer.b_v, s, h, h);

            // Multi-head attention
            let mut attn_weights = vec![0.0; nh * s * s];
            let mut concat = vec![0.0; s * h];
            let scale = 1.0 / (hd as f32).sqrt();

            for head in 0..nh {
                let ho = head * hd;
                // Compute scores
                let mut scores = vec![0.0; s * s];
                for t in 0..s {
                    for u in 0..s {
                        let mut dot = 0.0f32;
                        for i in 0..hd {
                            dot += q[t * h + ho + i] * k[u * h + ho + i];
                        }
                        scores[t * s + u] = dot * scale;
                    }
                }
                let attn = softmax_fwd(&scores, s, s);

                // Context
                for t in 0..s {
                    for i in 0..hd {
                        let mut sum = 0.0f32;
                        for u in 0..s {
                            sum += attn[t * s + u] * v[u * h + ho + i];
                        }
                        concat[t * h + ho + i] = sum;
                    }
                }

                attn_weights[head * s * s..(head + 1) * s * s].copy_from_slice(&attn);
            }

            // Output projection + residual
            let attn_out = linear_fwd(&concat, &layer.w_o, &layer.b_o, s, h, h);
            let mut post_attn = vec![0.0; s * h];
            for i in 0..s * h {
                post_attn[i] = input[i] + attn_out[i];
            }

            // Pre-LN FFN
            let (ln2_out, ln2_xhat, ln2_inv_std) =
                ln_fwd(&post_attn, &layer.ln2_gamma, &layer.ln2_beta, s, h);
            let ff_pre_gelu = linear_fwd(
                &ln2_out,
                &layer.ff_w1,
                &layer.ff_b1,
                s,
                h,
                self.config.ff_dim,
            );
            let ff_act = activation_fwd(&ff_pre_gelu, &self.config.activation);
            let ff_out = linear_fwd(
                &ff_act,
                &layer.ff_w2,
                &layer.ff_b2,
                s,
                self.config.ff_dim,
                h,
            );

            // Residual
            x = vec![0.0; s * h];
            for i in 0..s * h {
                x[i] = post_attn[i] + ff_out[i];
            }

            layer_caches.push(LayerCache {
                input,
                ln1_xhat,
                ln1_inv_std,
                ln1_out,
                q,
                k,
                v,
                attn_weights,
                concat,
                post_attn,
                ln2_xhat,
                ln2_inv_std,
                ln2_out,
                ff_pre_gelu,
                ff_act,
            });
        }

        let pre_final_ln = x.clone();
        let (final_out, final_xhat, final_inv_std) =
            ln_fwd(&x, &self.final_ln_gamma, &self.final_ln_beta, s, h);

        // Pooling
        let mut pooling_max_indices: Vec<usize> = Vec::new();
        let embedding = match self.config.pooling.as_str() {
            "cls" => {
                // CLS pooling: take the first token's output
                final_out[0..h].to_vec()
            }
            "max" => {
                // Max pooling: element-wise max across all tokens
                let mut emb = vec![f32::NEG_INFINITY; h];
                let mut indices = vec![0usize; h];
                for t in 0..s {
                    for i in 0..h {
                        if final_out[t * h + i] > emb[i] {
                            emb[i] = final_out[t * h + i];
                            indices[i] = t;
                        }
                    }
                }
                pooling_max_indices = indices;
                emb
            }
            "mean_max" => {
                // Mean-max pooling: average of mean and max pooling
                let inv_s = 1.0 / s as f32;
                let mut mean_emb = vec![0.0; h];
                let mut max_emb = vec![f32::NEG_INFINITY; h];
                let mut indices = vec![0usize; h];
                for t in 0..s {
                    for i in 0..h {
                        let v = final_out[t * h + i];
                        mean_emb[i] += v * inv_s;
                        if v > max_emb[i] {
                            max_emb[i] = v;
                            indices[i] = t;
                        }
                    }
                }
                pooling_max_indices = indices;
                let mut emb = vec![0.0; h];
                for i in 0..h {
                    emb[i] = (mean_emb[i] + max_emb[i]) / 2.0;
                }
                emb
            }
            _ => {
                // Mean pooling (default)
                let mut emb = vec![0.0; h];
                let inv_s = 1.0 / s as f32;
                for t in 0..s {
                    for i in 0..h {
                        emb[i] += final_out[t * h + i] * inv_s;
                    }
                }
                emb
            }
        };

        let cache = ForwardCache {
            token_ids: token_ids[..s].to_vec(),
            seq_len: s,
            embedded,
            layer_caches,
            pre_final_ln,
            final_ln_xhat: final_xhat,
            final_ln_inv_std: final_inv_std,
            final_output: final_out,
            pooling_max_indices,
        };

        (embedding, cache)
    }

    /// Backward pass. Takes gradient of loss w.r.t. the embedding output.
    pub fn backward(&self, d_emb: &[f32], cache: &ForwardCache) -> TransformerGradients {
        let h = self.config.hidden_dim;
        let s = cache.seq_len;
        let nh = self.config.num_heads;
        let hd = h / nh;
        let ff = self.config.ff_dim;

        // Backward through pooling
        let mut dx = vec![0.0; s * h];
        match self.config.pooling.as_str() {
            "cls" => {
                // CLS: gradient goes only to position 0
                for i in 0..h {
                    dx[i] = d_emb[i];
                }
            }
            "max" => {
                // Max: gradient goes to the token that had the max for each dim
                for i in 0..h {
                    let t = cache.pooling_max_indices[i];
                    dx[t * h + i] = d_emb[i];
                }
            }
            "mean_max" => {
                // Mean-max: gradient is 0.5 * (mean_backward + max_backward)
                let inv_s = 1.0 / s as f32;
                for i in 0..h {
                    let d = d_emb[i] * 0.5;
                    // Mean backward part
                    for t in 0..s {
                        dx[t * h + i] += d * inv_s;
                    }
                    // Max backward part
                    let t_max = cache.pooling_max_indices[i];
                    dx[t_max * h + i] += d;
                }
            }
            _ => {
                // Mean pooling backward (default)
                let inv_s = 1.0 / s as f32;
                for t in 0..s {
                    for i in 0..h {
                        dx[t * h + i] = d_emb[i] * inv_s;
                    }
                }
            }
        }

        // Backward through final layer norm
        let (dx_fln, d_fln_gamma, d_fln_beta) = ln_bwd(
            &dx,
            &cache.final_ln_xhat,
            &cache.final_ln_inv_std,
            &self.final_ln_gamma,
            s,
            h,
        );
        dx = dx_fln;

        let mut layer_grads = Vec::with_capacity(self.config.num_layers);

        // Backward through layers in reverse
        for (l, layer) in self.layers.iter().enumerate().rev() {
            let lc = &cache.layer_caches[l];

            // Residual from FFN: x = post_attn + ff_out
            let d_post_attn = dx.clone();
            let d_ff_out = dx;

            // Backward through FFN output linear
            let (d_ff_act, d_ff_w2, d_ff_b2) =
                linear_bwd(&d_ff_out, &lc.ff_act, &layer.ff_w2, s, ff, h);
            let d_ff_pre = activation_bwd(&d_ff_act, &lc.ff_pre_gelu, &self.config.activation);
            let (d_ln2_out, d_ff_w1, d_ff_b1) =
                linear_bwd(&d_ff_pre, &lc.ln2_out, &layer.ff_w1, s, h, ff);

            // Backward through LN2
            let (d_post_attn2, d_ln2_gamma, d_ln2_beta) = ln_bwd(
                &d_ln2_out,
                &lc.ln2_xhat,
                &lc.ln2_inv_std,
                &layer.ln2_gamma,
                s,
                h,
            );

            // Combine residual gradients
            let mut d_x = vec![0.0; s * h];
            for i in 0..s * h {
                d_x[i] = d_post_attn[i] + d_post_attn2[i];
            }

            // Residual from attention: post_attn = input + attn_out
            let d_input_res = d_x.clone();
            let d_attn_out = d_x;

            // Backward through output projection
            let (d_concat, d_w_o, d_b_o) = linear_bwd(&d_attn_out, &lc.concat, &layer.w_o, s, h, h);

            // Backward through multi-head attention
            let mut d_q = vec![0.0; s * h];
            let mut d_k = vec![0.0; s * h];
            let mut d_v = vec![0.0; s * h];
            let scale = 1.0 / (hd as f32).sqrt();

            for head in 0..nh {
                let ho = head * hd;
                let aw_off = head * s * s;

                // d_context → d_attn, d_V
                let mut d_attn = vec![0.0; s * s];
                for t in 0..s {
                    for i in 0..hd {
                        let dc = d_concat[t * h + ho + i];
                        for u in 0..s {
                            d_attn[t * s + u] += dc * lc.v[u * h + ho + i];
                            d_v[u * h + ho + i] += lc.attn_weights[aw_off + t * s + u] * dc;
                        }
                    }
                }

                // Softmax backward
                let d_scores = softmax_bwd(&d_attn, &lc.attn_weights[aw_off..aw_off + s * s], s, s);

                // d_Q, d_K from scores = Q @ K^T * scale
                for t in 0..s {
                    for u in 0..s {
                        let ds = d_scores[t * s + u] * scale;
                        for i in 0..hd {
                            d_q[t * h + ho + i] += ds * lc.k[u * h + ho + i];
                            d_k[u * h + ho + i] += ds * lc.q[t * h + ho + i];
                        }
                    }
                }
            }

            // Backward through Q, K, V projections
            let (d_ln1_q, d_w_q, d_b_q) = linear_bwd(&d_q, &lc.ln1_out, &layer.w_q, s, h, h);
            let (d_ln1_k, d_w_k, d_b_k) = linear_bwd(&d_k, &lc.ln1_out, &layer.w_k, s, h, h);
            let (d_ln1_v, d_w_v, d_b_v) = linear_bwd(&d_v, &lc.ln1_out, &layer.w_v, s, h, h);

            let mut d_ln1_out = vec![0.0; s * h];
            for i in 0..s * h {
                d_ln1_out[i] = d_ln1_q[i] + d_ln1_k[i] + d_ln1_v[i];
            }

            // Backward through LN1
            let (d_input_ln1, d_ln1_gamma, d_ln1_beta) = ln_bwd(
                &d_ln1_out,
                &lc.ln1_xhat,
                &lc.ln1_inv_std,
                &layer.ln1_gamma,
                s,
                h,
            );

            // Combine residual
            dx = vec![0.0; s * h];
            for i in 0..s * h {
                dx[i] = d_input_res[i] + d_input_ln1[i];
            }

            layer_grads.push(LayerGradients {
                d_ln1_gamma,
                d_ln1_beta,
                d_w_q,
                d_b_q,
                d_w_k,
                d_b_k,
                d_w_v,
                d_b_v,
                d_w_o,
                d_b_o,
                d_ln2_gamma,
                d_ln2_beta,
                d_ff_w1,
                d_ff_b1,
                d_ff_w2,
                d_ff_b2,
            });
        }

        // Reverse layer grads to match forward order
        layer_grads.reverse();

        // Embedding gradients
        let mut d_token_emb = vec![0.0; self.token_emb.len()];
        let mut d_pos_emb = vec![0.0; self.pos_emb.len()];
        for t in 0..s {
            let tid = cache.token_ids[t].min(self.config.vocab_size - 1);
            for i in 0..h {
                d_token_emb[tid * h + i] += dx[t * h + i];
                d_pos_emb[t * h + i] += dx[t * h + i];
            }
        }

        TransformerGradients {
            d_token_emb,
            d_pos_emb,
            layers: layer_grads,
            d_final_ln_gamma: d_fln_gamma,
            d_final_ln_beta: d_fln_beta,
        }
    }

    /// Collect mutable references to all parameter buffers.
    pub fn param_buffers_mut(&mut self) -> Vec<&mut Vec<f32>> {
        let mut bufs: Vec<&mut Vec<f32>> = Vec::new();
        bufs.push(&mut self.token_emb);
        bufs.push(&mut self.pos_emb);
        for layer in &mut self.layers {
            bufs.push(&mut layer.ln1_gamma);
            bufs.push(&mut layer.ln1_beta);
            bufs.push(&mut layer.w_q);
            bufs.push(&mut layer.b_q);
            bufs.push(&mut layer.w_k);
            bufs.push(&mut layer.b_k);
            bufs.push(&mut layer.w_v);
            bufs.push(&mut layer.b_v);
            bufs.push(&mut layer.w_o);
            bufs.push(&mut layer.b_o);
            bufs.push(&mut layer.ln2_gamma);
            bufs.push(&mut layer.ln2_beta);
            bufs.push(&mut layer.ff_w1);
            bufs.push(&mut layer.ff_b1);
            bufs.push(&mut layer.ff_w2);
            bufs.push(&mut layer.ff_b2);
        }
        bufs.push(&mut self.final_ln_gamma);
        bufs.push(&mut self.final_ln_beta);
        bufs
    }

    /// Collect references to gradient buffers in same order as param_buffers_mut.
    pub fn grad_refs(grads: &TransformerGradients) -> Vec<&Vec<f32>> {
        let mut refs: Vec<&Vec<f32>> = Vec::new();
        refs.push(&grads.d_token_emb);
        refs.push(&grads.d_pos_emb);
        for lg in &grads.layers {
            refs.push(&lg.d_ln1_gamma);
            refs.push(&lg.d_ln1_beta);
            refs.push(&lg.d_w_q);
            refs.push(&lg.d_b_q);
            refs.push(&lg.d_w_k);
            refs.push(&lg.d_b_k);
            refs.push(&lg.d_w_v);
            refs.push(&lg.d_b_v);
            refs.push(&lg.d_w_o);
            refs.push(&lg.d_b_o);
            refs.push(&lg.d_ln2_gamma);
            refs.push(&lg.d_ln2_beta);
            refs.push(&lg.d_ff_w1);
            refs.push(&lg.d_ff_b1);
            refs.push(&lg.d_ff_w2);
            refs.push(&lg.d_ff_b2);
        }
        refs.push(&grads.d_final_ln_gamma);
        refs.push(&grads.d_final_ln_beta);
        refs
    }

    /// Get sizes of all parameter buffers (for Adam state init).
    pub fn param_sizes(&self) -> Vec<usize> {
        let mut sizes = Vec::new();
        sizes.push(self.token_emb.len());
        sizes.push(self.pos_emb.len());
        for layer in &self.layers {
            sizes.push(layer.ln1_gamma.len());
            sizes.push(layer.ln1_beta.len());
            sizes.push(layer.w_q.len());
            sizes.push(layer.b_q.len());
            sizes.push(layer.w_k.len());
            sizes.push(layer.b_k.len());
            sizes.push(layer.w_v.len());
            sizes.push(layer.b_v.len());
            sizes.push(layer.w_o.len());
            sizes.push(layer.b_o.len());
            sizes.push(layer.ln2_gamma.len());
            sizes.push(layer.ln2_beta.len());
            sizes.push(layer.ff_w1.len());
            sizes.push(layer.ff_b1.len());
            sizes.push(layer.ff_w2.len());
            sizes.push(layer.ff_b2.len());
        }
        sizes.push(self.final_ln_gamma.len());
        sizes.push(self.final_ln_beta.len());
        sizes
    }

    /// Apply gradients using Adam optimizer.
    pub fn apply_gradients(&mut self, grads: &TransformerGradients, adam: &mut AdamState) {
        let grad_refs = Self::grad_refs(grads);
        let mut param_bufs = self.param_buffers_mut();
        adam.step(
            &mut param_bufs.iter_mut().map(|b| &mut **b).collect::<Vec<_>>(),
            &grad_refs,
        );
    }

    /// Accumulate gradients (for averaging over a batch).
    pub fn accumulate_grads(target: &mut TransformerGradients, source: &TransformerGradients) {
        fn acc(t: &mut Vec<f32>, s: &Vec<f32>) {
            for (a, b) in t.iter_mut().zip(s.iter()) {
                *a += *b;
            }
        }
        acc(&mut target.d_token_emb, &source.d_token_emb);
        acc(&mut target.d_pos_emb, &source.d_pos_emb);
        acc(&mut target.d_final_ln_gamma, &source.d_final_ln_gamma);
        acc(&mut target.d_final_ln_beta, &source.d_final_ln_beta);
        for (tl, sl) in target.layers.iter_mut().zip(source.layers.iter()) {
            acc(&mut tl.d_ln1_gamma, &sl.d_ln1_gamma);
            acc(&mut tl.d_ln1_beta, &sl.d_ln1_beta);
            acc(&mut tl.d_w_q, &sl.d_w_q);
            acc(&mut tl.d_b_q, &sl.d_b_q);
            acc(&mut tl.d_w_k, &sl.d_w_k);
            acc(&mut tl.d_b_k, &sl.d_b_k);
            acc(&mut tl.d_w_v, &sl.d_w_v);
            acc(&mut tl.d_b_v, &sl.d_b_v);
            acc(&mut tl.d_w_o, &sl.d_w_o);
            acc(&mut tl.d_b_o, &sl.d_b_o);
            acc(&mut tl.d_ln2_gamma, &sl.d_ln2_gamma);
            acc(&mut tl.d_ln2_beta, &sl.d_ln2_beta);
            acc(&mut tl.d_ff_w1, &sl.d_ff_w1);
            acc(&mut tl.d_ff_b1, &sl.d_ff_b1);
            acc(&mut tl.d_ff_w2, &sl.d_ff_w2);
            acc(&mut tl.d_ff_b2, &sl.d_ff_b2);
        }
    }

    /// Scale all gradients by a factor (e.g. 1/batch_size).
    pub fn scale_grads(grads: &mut TransformerGradients, factor: f32) {
        fn sc(v: &mut Vec<f32>, f: f32) {
            for x in v.iter_mut() {
                *x *= f;
            }
        }
        sc(&mut grads.d_token_emb, factor);
        sc(&mut grads.d_pos_emb, factor);
        sc(&mut grads.d_final_ln_gamma, factor);
        sc(&mut grads.d_final_ln_beta, factor);
        for lg in &mut grads.layers {
            sc(&mut lg.d_ln1_gamma, factor);
            sc(&mut lg.d_ln1_beta, factor);
            sc(&mut lg.d_w_q, factor);
            sc(&mut lg.d_b_q, factor);
            sc(&mut lg.d_w_k, factor);
            sc(&mut lg.d_b_k, factor);
            sc(&mut lg.d_w_v, factor);
            sc(&mut lg.d_b_v, factor);
            sc(&mut lg.d_w_o, factor);
            sc(&mut lg.d_b_o, factor);
            sc(&mut lg.d_ln2_gamma, factor);
            sc(&mut lg.d_ln2_beta, factor);
            sc(&mut lg.d_ff_w1, factor);
            sc(&mut lg.d_ff_b1, factor);
            sc(&mut lg.d_ff_w2, factor);
            sc(&mut lg.d_ff_b2, factor);
        }
    }

    pub fn zero_grads(&self) -> TransformerGradients {
        let h = self.config.hidden_dim;
        let ff = self.config.ff_dim;
        TransformerGradients {
            d_token_emb: vec![0.0; self.token_emb.len()],
            d_pos_emb: vec![0.0; self.pos_emb.len()],
            layers: (0..self.config.num_layers)
                .map(|_| LayerGradients {
                    d_ln1_gamma: vec![0.0; h],
                    d_ln1_beta: vec![0.0; h],
                    d_w_q: vec![0.0; h * h],
                    d_b_q: vec![0.0; h],
                    d_w_k: vec![0.0; h * h],
                    d_b_k: vec![0.0; h],
                    d_w_v: vec![0.0; h * h],
                    d_b_v: vec![0.0; h],
                    d_w_o: vec![0.0; h * h],
                    d_b_o: vec![0.0; h],
                    d_ln2_gamma: vec![0.0; h],
                    d_ln2_beta: vec![0.0; h],
                    d_ff_w1: vec![0.0; h * ff],
                    d_ff_b1: vec![0.0; ff],
                    d_ff_w2: vec![0.0; ff * h],
                    d_ff_b2: vec![0.0; h],
                })
                .collect(),
            d_final_ln_gamma: vec![0.0; h],
            d_final_ln_beta: vec![0.0; h],
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Serialization
// ═══════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticWeightsData {
    pub config: TransformerConfig,
    pub token_emb: Vec<f32>,
    pub pos_emb: Vec<f32>,
    pub layers: Vec<TransformerLayer>,
    pub final_ln_gamma: Vec<f32>,
    pub final_ln_beta: Vec<f32>,
}

impl TransformerEncoder {
    pub fn save_weights(&self) -> SemanticWeightsData {
        SemanticWeightsData {
            config: self.config.clone(),
            token_emb: self.token_emb.clone(),
            pos_emb: self.pos_emb.clone(),
            layers: self.layers.clone(),
            final_ln_gamma: self.final_ln_gamma.clone(),
            final_ln_beta: self.final_ln_beta.clone(),
        }
    }

    pub fn load_weights(&mut self, data: &SemanticWeightsData) {
        self.token_emb = data.token_emb.clone();
        self.pos_emb = data.pos_emb.clone();
        self.layers = data.layers.clone();
        self.final_ln_gamma = data.final_ln_gamma.clone();
        self.final_ln_beta = data.final_ln_beta.clone();
    }

    pub fn from_weights(data: &SemanticWeightsData) -> Self {
        Self {
            config: data.config.clone(),
            token_emb: data.token_emb.clone(),
            pos_emb: data.pos_emb.clone(),
            layers: data.layers.clone(),
            final_ln_gamma: data.final_ln_gamma.clone(),
            final_ln_beta: data.final_ln_beta.clone(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// GPU-accelerated forward/backward (Metal compute on unified memory)
// ═══════════════════════════════════════════════════════════════════

#[cfg(feature = "gpu")]
impl TransformerEncoder {
    /// GPU-accelerated forward pass. Linear projections run on Metal GPU,
    /// attention inner loop stays on CPU (small per-head matrices).
    pub fn forward_gpu(
        &self,
        token_ids: &[usize],
        gpu: &crate::gpu::GpuContext,
    ) -> (Vec<f32>, ForwardCache) {
        let h = self.config.hidden_dim;
        let s = token_ids.len().min(self.config.max_seq_len);
        let nh = self.config.num_heads;
        let hd = h / nh;

        // Embedding lookup + position (CPU — trivial)
        let mut x = vec![0.0; s * h];
        for t in 0..s {
            let tid = token_ids[t].min(self.config.vocab_size - 1);
            for i in 0..h {
                x[t * h + i] = self.token_emb[tid * h + i] + self.pos_emb[t * h + i];
            }
        }
        let embedded = x.clone();

        let mut layer_caches = Vec::with_capacity(self.config.num_layers);

        for layer in &self.layers {
            let input = x.clone();

            // Pre-LN attention
            let (ln1_out, ln1_xhat, ln1_inv_std) =
                ln_fwd(&x, &layer.ln1_gamma, &layer.ln1_beta, s, h);

            // GPU: Q/K/V projections
            let q = gpu.linear_fwd(&ln1_out, &layer.w_q, &layer.b_q, s, h, h);
            let k = gpu.linear_fwd(&ln1_out, &layer.w_k, &layer.b_k, s, h, h);
            let v = gpu.linear_fwd(&ln1_out, &layer.w_v, &layer.b_v, s, h, h);

            // Multi-head attention (CPU — small per-head matrices)
            let mut attn_weights = vec![0.0; nh * s * s];
            let mut concat = vec![0.0; s * h];
            let scale = 1.0 / (hd as f32).sqrt();

            for head in 0..nh {
                let ho = head * hd;
                let mut scores = vec![0.0; s * s];
                for t in 0..s {
                    for u in 0..s {
                        let mut dot = 0.0f32;
                        for i in 0..hd {
                            dot += q[t * h + ho + i] * k[u * h + ho + i];
                        }
                        scores[t * s + u] = dot * scale;
                    }
                }
                let attn = softmax_fwd(&scores, s, s);

                for t in 0..s {
                    for i in 0..hd {
                        let mut sum = 0.0f32;
                        for u in 0..s {
                            sum += attn[t * s + u] * v[u * h + ho + i];
                        }
                        concat[t * h + ho + i] = sum;
                    }
                }

                attn_weights[head * s * s..(head + 1) * s * s].copy_from_slice(&attn);
            }

            // GPU: Output projection
            let attn_out = gpu.linear_fwd(&concat, &layer.w_o, &layer.b_o, s, h, h);
            let post_attn = gpu.vec_add(&input, &attn_out);

            // Pre-LN FFN
            let (ln2_out, ln2_xhat, ln2_inv_std) =
                ln_fwd(&post_attn, &layer.ln2_gamma, &layer.ln2_beta, s, h);

            // GPU: FFN
            let ff_pre_gelu = gpu.linear_fwd(
                &ln2_out,
                &layer.ff_w1,
                &layer.ff_b1,
                s,
                h,
                self.config.ff_dim,
            );
            let ff_act = gpu.activation_fwd(&ff_pre_gelu, &self.config.activation);
            let ff_out = gpu.linear_fwd(
                &ff_act,
                &layer.ff_w2,
                &layer.ff_b2,
                s,
                self.config.ff_dim,
                h,
            );

            // Residual
            x = gpu.vec_add(&post_attn, &ff_out);

            layer_caches.push(LayerCache {
                input,
                ln1_xhat,
                ln1_inv_std,
                ln1_out,
                q,
                k,
                v,
                attn_weights,
                concat,
                post_attn,
                ln2_xhat,
                ln2_inv_std,
                ln2_out,
                ff_pre_gelu,
                ff_act,
            });
        }

        let pre_final_ln = x.clone();
        let (final_out, final_xhat, final_inv_std) =
            ln_fwd(&x, &self.final_ln_gamma, &self.final_ln_beta, s, h);

        // Pooling (CPU — simple reduction)
        let mut pooling_max_indices: Vec<usize> = Vec::new();
        let embedding = match self.config.pooling.as_str() {
            "cls" => final_out[0..h].to_vec(),
            "max" => {
                let mut emb = vec![f32::NEG_INFINITY; h];
                let mut indices = vec![0usize; h];
                for t in 0..s {
                    for i in 0..h {
                        if final_out[t * h + i] > emb[i] {
                            emb[i] = final_out[t * h + i];
                            indices[i] = t;
                        }
                    }
                }
                pooling_max_indices = indices;
                emb
            }
            "mean_max" => {
                let inv_s = 1.0 / s as f32;
                let mut mean_emb = vec![0.0; h];
                let mut max_emb = vec![f32::NEG_INFINITY; h];
                let mut indices = vec![0usize; h];
                for t in 0..s {
                    for i in 0..h {
                        let val = final_out[t * h + i];
                        mean_emb[i] += val * inv_s;
                        if val > max_emb[i] {
                            max_emb[i] = val;
                            indices[i] = t;
                        }
                    }
                }
                pooling_max_indices = indices;
                let mut emb = vec![0.0; h];
                for i in 0..h {
                    emb[i] = (mean_emb[i] + max_emb[i]) / 2.0;
                }
                emb
            }
            _ => {
                let mut emb = vec![0.0; h];
                let inv_s = 1.0 / s as f32;
                for t in 0..s {
                    for i in 0..h {
                        emb[i] += final_out[t * h + i] * inv_s;
                    }
                }
                emb
            }
        };

        let cache = ForwardCache {
            token_ids: token_ids[..s].to_vec(),
            seq_len: s,
            embedded,
            layer_caches,
            pre_final_ln,
            final_ln_xhat: final_xhat,
            final_ln_inv_std: final_inv_std,
            final_output: final_out,
            pooling_max_indices,
        };

        (embedding, cache)
    }

    /// GPU-accelerated backward pass.
    pub fn backward_gpu(
        &self,
        d_emb: &[f32],
        cache: &ForwardCache,
        gpu: &crate::gpu::GpuContext,
    ) -> TransformerGradients {
        let h = self.config.hidden_dim;
        let s = cache.seq_len;
        let nh = self.config.num_heads;
        let hd = h / nh;
        let ff = self.config.ff_dim;

        // Backward through pooling (CPU — trivial)
        let mut dx = vec![0.0; s * h];
        match self.config.pooling.as_str() {
            "cls" => {
                for i in 0..h {
                    dx[i] = d_emb[i];
                }
            }
            "max" => {
                for i in 0..h {
                    let t = cache.pooling_max_indices[i];
                    dx[t * h + i] = d_emb[i];
                }
            }
            "mean_max" => {
                let inv_s = 1.0 / s as f32;
                for i in 0..h {
                    let d = d_emb[i] * 0.5;
                    for t in 0..s {
                        dx[t * h + i] += d * inv_s;
                    }
                    let t_max = cache.pooling_max_indices[i];
                    dx[t_max * h + i] += d;
                }
            }
            _ => {
                let inv_s = 1.0 / s as f32;
                for t in 0..s {
                    for i in 0..h {
                        dx[t * h + i] = d_emb[i] * inv_s;
                    }
                }
            }
        }

        // Backward through final layer norm (CPU — reduction, small)
        let (dx_fln, d_fln_gamma, d_fln_beta) = ln_bwd(
            &dx,
            &cache.final_ln_xhat,
            &cache.final_ln_inv_std,
            &self.final_ln_gamma,
            s,
            h,
        );
        dx = dx_fln;

        let mut layer_grads = Vec::with_capacity(self.config.num_layers);

        for (l, layer) in self.layers.iter().enumerate().rev() {
            let lc = &cache.layer_caches[l];

            let d_post_attn = dx.clone();
            let d_ff_out = dx;

            // GPU: Backward through FFN
            let (d_ff_act, d_ff_w2, d_ff_b2) =
                gpu.linear_bwd(&d_ff_out, &lc.ff_act, &layer.ff_w2, s, ff, h);
            let d_ff_pre = gpu.activation_bwd(&d_ff_act, &lc.ff_pre_gelu, &self.config.activation);
            let (d_ln2_out, d_ff_w1, d_ff_b1) =
                gpu.linear_bwd(&d_ff_pre, &lc.ln2_out, &layer.ff_w1, s, h, ff);

            // LN2 backward (CPU)
            let (d_post_attn2, d_ln2_gamma, d_ln2_beta) = ln_bwd(
                &d_ln2_out,
                &lc.ln2_xhat,
                &lc.ln2_inv_std,
                &layer.ln2_gamma,
                s,
                h,
            );

            // Combine residual gradients
            let d_x = gpu.vec_add(&d_post_attn, &d_post_attn2);

            let d_input_res = d_x.clone();
            let d_attn_out = d_x;

            // GPU: Backward through output projection
            let (d_concat, d_w_o, d_b_o) =
                gpu.linear_bwd(&d_attn_out, &lc.concat, &layer.w_o, s, h, h);

            // Backward through multi-head attention (CPU — small per-head)
            let mut d_q = vec![0.0; s * h];
            let mut d_k = vec![0.0; s * h];
            let mut d_v = vec![0.0; s * h];
            let scale = 1.0 / (hd as f32).sqrt();

            for head in 0..nh {
                let ho = head * hd;
                let aw_off = head * s * s;

                let mut d_attn = vec![0.0; s * s];
                for t in 0..s {
                    for i in 0..hd {
                        let dc = d_concat[t * h + ho + i];
                        for u in 0..s {
                            d_attn[t * s + u] += dc * lc.v[u * h + ho + i];
                            d_v[u * h + ho + i] += lc.attn_weights[aw_off + t * s + u] * dc;
                        }
                    }
                }

                let d_scores = softmax_bwd(&d_attn, &lc.attn_weights[aw_off..aw_off + s * s], s, s);

                for t in 0..s {
                    for u in 0..s {
                        let ds = d_scores[t * s + u] * scale;
                        for i in 0..hd {
                            d_q[t * h + ho + i] += ds * lc.k[u * h + ho + i];
                            d_k[u * h + ho + i] += ds * lc.q[t * h + ho + i];
                        }
                    }
                }
            }

            // GPU: Backward through Q, K, V projections
            let (d_ln1_q, d_w_q, d_b_q) = gpu.linear_bwd(&d_q, &lc.ln1_out, &layer.w_q, s, h, h);
            let (d_ln1_k, d_w_k, d_b_k) = gpu.linear_bwd(&d_k, &lc.ln1_out, &layer.w_k, s, h, h);
            let (d_ln1_v, d_w_v, d_b_v) = gpu.linear_bwd(&d_v, &lc.ln1_out, &layer.w_v, s, h, h);

            let mut d_ln1_out = vec![0.0; s * h];
            for i in 0..s * h {
                d_ln1_out[i] = d_ln1_q[i] + d_ln1_k[i] + d_ln1_v[i];
            }

            // LN1 backward (CPU)
            let (d_input_ln1, d_ln1_gamma, d_ln1_beta) = ln_bwd(
                &d_ln1_out,
                &lc.ln1_xhat,
                &lc.ln1_inv_std,
                &layer.ln1_gamma,
                s,
                h,
            );

            dx = vec![0.0; s * h];
            for i in 0..s * h {
                dx[i] = d_input_res[i] + d_input_ln1[i];
            }

            layer_grads.push(LayerGradients {
                d_ln1_gamma,
                d_ln1_beta,
                d_w_q,
                d_b_q,
                d_w_k,
                d_b_k,
                d_w_v,
                d_b_v,
                d_w_o,
                d_b_o,
                d_ln2_gamma,
                d_ln2_beta,
                d_ff_w1,
                d_ff_b1,
                d_ff_w2,
                d_ff_b2,
            });
        }

        layer_grads.reverse();

        // Embedding gradients (CPU — sparse scatter)
        let mut d_token_emb = vec![0.0; self.token_emb.len()];
        let mut d_pos_emb = vec![0.0; self.pos_emb.len()];
        for t in 0..s {
            let tid = cache.token_ids[t].min(self.config.vocab_size - 1);
            for i in 0..h {
                d_token_emb[tid * h + i] += dx[t * h + i];
                d_pos_emb[t * h + i] += dx[t * h + i];
            }
        }

        TransformerGradients {
            d_token_emb,
            d_pos_emb,
            layers: layer_grads,
            d_final_ln_gamma: d_fln_gamma,
            d_final_ln_beta: d_fln_beta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_output_dim() {
        let config = TransformerConfig {
            vocab_size: 100,
            hidden_dim: 128,
            num_heads: 4,
            num_layers: 2,
            ff_dim: 512,
            max_seq_len: 64,
            ..TransformerConfig::default()
        };
        let encoder = TransformerEncoder::new(config);
        let (emb, _cache) = encoder.forward(&[1, 2, 3, 4, 5]);
        assert_eq!(emb.len(), 128);
    }

    #[test]
    fn test_forward_deterministic() {
        let config = TransformerConfig {
            vocab_size: 50,
            ..TransformerConfig::default()
        };
        let encoder = TransformerEncoder::new(config);
        let (e1, _) = encoder.forward(&[1, 2, 3]);
        let (e2, _) = encoder.forward(&[1, 2, 3]);
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_different_inputs_different_outputs() {
        let config = TransformerConfig {
            vocab_size: 50,
            ..TransformerConfig::default()
        };
        let encoder = TransformerEncoder::new(config);
        let (e1, _) = encoder.forward(&[1, 2, 3]);
        let (e2, _) = encoder.forward(&[4, 5, 6]);
        assert_ne!(e1, e2);
    }

    #[test]
    fn test_backward_produces_gradients() {
        let config = TransformerConfig {
            vocab_size: 50,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 1,
            ff_dim: 32,
            max_seq_len: 16,
            ..TransformerConfig::default()
        };
        let encoder = TransformerEncoder::new(config);
        let (emb, cache) = encoder.forward(&[1, 2, 3]);
        let d_emb = vec![1.0; emb.len()];
        let grads = encoder.backward(&d_emb, &cache);
        // Token embedding gradients should be non-zero for tokens 1, 2, 3
        let has_nonzero = grads.d_token_emb.iter().any(|&v| v != 0.0);
        assert!(has_nonzero, "Should have non-zero embedding gradients");
    }

    #[test]
    fn test_param_count() {
        let encoder = TransformerEncoder::new(TransformerConfig::default());
        let count = encoder.param_count();
        // token_emb: 8192*128 = 1048576
        // pos_emb: 128*128 = 16384
        // per layer: 128*2 + 128*128*4 + 128*4 + 128*2 + 128*512 + 512 + 512*128 + 128 = 198272
        // 2 layers: 396544
        // final_ln: 256
        assert!(count > 1_000_000, "Expected >1M params, got {}", count);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = TransformerConfig {
            vocab_size: 50,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 1,
            ff_dim: 32,
            max_seq_len: 16,
            ..TransformerConfig::default()
        };
        let encoder = TransformerEncoder::new(config);
        let saved = encoder.save_weights();
        let json = serde_json::to_string(&saved).unwrap();
        let loaded: SemanticWeightsData = serde_json::from_str(&json).unwrap();
        let restored = TransformerEncoder::from_weights(&loaded);
        let (e1, _) = encoder.forward(&[1, 2, 3]);
        let (e2, _) = restored.forward(&[1, 2, 3]);
        assert_eq!(e1, e2);
    }
}
