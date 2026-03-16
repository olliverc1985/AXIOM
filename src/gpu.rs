//! GPU compute module — Metal compute shaders on Apple Silicon unified memory.
//!
//! All kernels are written from scratch in Metal Shading Language (MSL).
//! Uses unified memory (StorageModeShared) for zero-copy CPU<->GPU data sharing.
//! Gated behind the `gpu` Cargo feature.

#[cfg(feature = "gpu")]
mod metal_compute {
    use metal::*;
    use std::ffi::c_void;
    use std::mem;

    const TILE: u64 = 16;

    // ═══════════════════════════════════════════════════════════════════
    // Metal Shading Language kernels — all written from scratch
    // ═══════════════════════════════════════════════════════════════════
    const MSL_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE = 16;

// ═══ Tiled General Matrix Multiply: C = A @ B ═══
// A: [M, K], B: [K, N], C: [M, N]
// Uses threadgroup shared memory for cache-friendly tiling.
kernel void gemm_nn(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup float sA[TILE][TILE];
    threadgroup float sB[TILE][TILE];

    uint row = gid.y;
    uint col = gid.x;
    float acc = 0.0f;

    uint tiles = (K + TILE - 1) / TILE;
    for (uint t = 0; t < tiles; t++) {
        uint aCol = t * TILE + lid.x;
        uint bRow = t * TILE + lid.y;

        sA[lid.y][lid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[lid.y][lid.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {
            acc += sA[lid.y][i] * sB[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ═══ Fused Linear Forward: Y = X @ W + bias ═══
// X: [S, in_d], W: [in_d, out_d], bias: [out_d], Y: [S, out_d]
kernel void linear_fwd(
    device const float* X [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* Y [[buffer(3)]],
    constant uint& S [[buffer(4)]],
    constant uint& in_d [[buffer(5)]],
    constant uint& out_d [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    threadgroup float sX[TILE][TILE];
    threadgroup float sW[TILE][TILE];

    uint row = gid.y;  // sequence position
    uint col = gid.x;  // output dimension

    float acc = 0.0f;

    uint tiles = (in_d + TILE - 1) / TILE;
    for (uint t = 0; t < tiles; t++) {
        uint xCol = t * TILE + lid.x;
        uint wRow = t * TILE + lid.y;

        sX[lid.y][lid.x] = (row < S && xCol < in_d) ? X[row * in_d + xCol] : 0.0f;
        sW[lid.y][lid.x] = (wRow < in_d && col < out_d) ? W[wRow * out_d + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE; i++) {
            acc += sX[lid.y][i] * sW[i][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < S && col < out_d) {
        Y[row * out_d + col] = acc + bias[col];
    }
}

// ═══ GELU forward: y = x * sigmoid(1.702 * x) ═══
kernel void gelu_fwd(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float v = X[gid];
    float s = 1.0f / (1.0f + exp(-1.702f * v));
    Y[gid] = v * s;
}

// ═══ GELU backward ═══
kernel void gelu_bwd(
    device const float* dout [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* dX [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float v = X[gid];
    float s = 1.0f / (1.0f + exp(-1.702f * v));
    dX[gid] = dout[gid] * (s + v * 1.702f * s * (1.0f - s));
}

// ═══ ReLU forward ═══
kernel void relu_fwd(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    Y[gid] = max(X[gid], 0.0f);
}

// ═══ ReLU backward ═══
kernel void relu_bwd(
    device const float* dout [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* dX [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    dX[gid] = (X[gid] > 0.0f) ? dout[gid] : 0.0f;
}

// ═══ SiLU forward: y = x * sigmoid(x) ═══
kernel void silu_fwd(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float v = X[gid];
    float s = 1.0f / (1.0f + exp(-v));
    Y[gid] = v * s;
}

// ═══ SiLU backward ═══
kernel void silu_bwd(
    device const float* dout [[buffer(0)]],
    device const float* X [[buffer(1)]],
    device float* dX [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    float v = X[gid];
    float s = 1.0f / (1.0f + exp(-v));
    dX[gid] = dout[gid] * (s + v * s * (1.0f - s));
}

// ═══ Row sum: db[j] = sum_i M[i,j] ═══
// M: [rows, cols], db: [cols]
kernel void row_sum(
    device const float* M [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= cols) return;
    float acc = 0.0f;
    for (uint r = 0; r < rows; r++) {
        acc += M[r * cols + gid];
    }
    out[gid] = acc;
}

// ═══ Element-wise add: C = A + B ═══
kernel void vec_add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;
    C[gid] = A[gid] + B[gid];
}

// ═══ Row-wise softmax forward ═══
// X: [rows, cols], Y: [rows, cols]
kernel void softmax_fwd(
    device const float* X [[buffer(0)]],
    device float* Y [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    uint off = gid * cols;

    // Find max
    float mx = X[off];
    for (uint c = 1; c < cols; c++) {
        mx = max(mx, X[off + c]);
    }

    // Exp and sum
    float s = 0.0f;
    for (uint c = 0; c < cols; c++) {
        float e = exp(X[off + c] - mx);
        Y[off + c] = e;
        s += e;
    }

    // Normalize
    float inv = 1.0f / s;
    for (uint c = 0; c < cols; c++) {
        Y[off + c] *= inv;
    }
}

// ═══ Row-wise softmax backward ═══
kernel void softmax_bwd(
    device const float* dout [[buffer(0)]],
    device const float* Y [[buffer(1)]],
    device float* dX [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= rows) return;
    uint off = gid * cols;

    float dot = 0.0f;
    for (uint c = 0; c < cols; c++) {
        dot += dout[off + c] * Y[off + c];
    }
    for (uint c = 0; c < cols; c++) {
        dX[off + c] = Y[off + c] * (dout[off + c] - dot);
    }
}

// ═══ Layer norm forward ═══
// X: [S, D], gamma: [D], beta: [D]
// out: [S, D], x_hat: [S, D], inv_std: [S]
kernel void layernorm_fwd(
    device const float* X [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    device float* x_hat [[buffer(4)]],
    device float* inv_std [[buffer(5)]],
    constant uint& S [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= S) return;
    uint off = gid * D;

    // Mean
    float mean = 0.0f;
    for (uint i = 0; i < D; i++) {
        mean += X[off + i];
    }
    mean /= float(D);

    // Variance
    float var = 0.0f;
    for (uint i = 0; i < D; i++) {
        float d = X[off + i] - mean;
        var += d * d;
    }
    var /= float(D);

    float is = 1.0f / sqrt(var + 1e-5f);
    inv_std[gid] = is;

    for (uint i = 0; i < D; i++) {
        float xh = (X[off + i] - mean) * is;
        x_hat[off + i] = xh;
        out[off + i] = gamma[i] * xh + beta[i];
    }
}

// ═══ Adam optimizer step ═══
// Updates params in-place, updates m and v states
kernel void adam_step(
    device float* params [[buffer(0)]],
    device const float* grads [[buffer(1)]],
    device float* m_state [[buffer(2)]],
    device float* v_state [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& bc1 [[buffer(7)]],
    constant float& bc2 [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    constant float& weight_decay [[buffer(10)]],
    constant uint& count [[buffer(11)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= count) return;

    float g = grads[gid];
    // AdamW weight decay
    if (weight_decay > 0.0f) {
        params[gid] -= lr * weight_decay * params[gid];
    }

    float m = beta1 * m_state[gid] + (1.0f - beta1) * g;
    float v = beta2 * v_state[gid] + (1.0f - beta2) * g * g;
    m_state[gid] = m;
    v_state[gid] = v;

    float mh = m / bc1;
    float vh = v / bc2;
    params[gid] -= lr * mh / (sqrt(vh) + eps);
}
"#;

    // Pre-allocated buffer capacity: 256K floats = 1MB per buffer.
    // Covers all operations for models up to hidden=256, ff=1024, seq=128.
    const BUF_CAP: usize = 256 * 1024;
    const NUM_BUFS: usize = 6;

    /// GPU compute context — manages Metal device, command queue, compiled pipelines,
    /// and pre-allocated shared-memory buffers for zero-allocation dispatch.
    pub struct GpuContext {
        device: Device,
        queue: CommandQueue,
        gemm_pipeline: ComputePipelineState,
        linear_fwd_pipeline: ComputePipelineState,
        gelu_fwd_pipeline: ComputePipelineState,
        gelu_bwd_pipeline: ComputePipelineState,
        relu_fwd_pipeline: ComputePipelineState,
        relu_bwd_pipeline: ComputePipelineState,
        silu_fwd_pipeline: ComputePipelineState,
        silu_bwd_pipeline: ComputePipelineState,
        row_sum_pipeline: ComputePipelineState,
        vec_add_pipeline: ComputePipelineState,
        softmax_fwd_pipeline: ComputePipelineState,
        softmax_bwd_pipeline: ComputePipelineState,
        layernorm_fwd_pipeline: ComputePipelineState,
        adam_step_pipeline: ComputePipelineState,
        /// Pre-allocated shared-memory scratch buffers (unified memory = zero copy).
        /// Reused across all operations to avoid per-call buffer allocation.
        bufs: Vec<Buffer>,
    }

    impl GpuContext {
        /// Create a new GPU context. Returns None if no Metal device is available.
        pub fn new() -> Option<Self> {
            let device = Device::system_default()?;
            let name = device.name().to_string();
            println!("  GPU: {}", name);

            let options = CompileOptions::new();
            let library = device
                .new_library_with_source(MSL_SOURCE, &options)
                .unwrap_or_else(|e| panic!("Failed to compile Metal shaders: {}", e));

            let make_pipeline = |kernel_name: &str| -> ComputePipelineState {
                let func = library
                    .get_function(kernel_name, None)
                    .unwrap_or_else(|e| panic!("Missing kernel '{}': {}", kernel_name, e));
                device
                    .new_compute_pipeline_state_with_function(&func)
                    .unwrap_or_else(|e| panic!("Pipeline '{}': {}", kernel_name, e))
            };

            // Pre-allocate scratch buffers on unified memory
            let buf_bytes = (BUF_CAP * mem::size_of::<f32>()) as u64;
            let bufs: Vec<Buffer> = (0..NUM_BUFS)
                .map(|_| device.new_buffer(buf_bytes, MTLResourceOptions::StorageModeShared))
                .collect();
            println!(
                "  GPU buffers: {}x {:.0}KB = {:.1}MB unified memory",
                NUM_BUFS,
                buf_bytes as f64 / 1024.0,
                (NUM_BUFS as f64 * buf_bytes as f64) / (1024.0 * 1024.0)
            );

            let ctx = Self {
                queue: device.new_command_queue(),
                gemm_pipeline: make_pipeline("gemm_nn"),
                linear_fwd_pipeline: make_pipeline("linear_fwd"),
                gelu_fwd_pipeline: make_pipeline("gelu_fwd"),
                gelu_bwd_pipeline: make_pipeline("gelu_bwd"),
                relu_fwd_pipeline: make_pipeline("relu_fwd"),
                relu_bwd_pipeline: make_pipeline("relu_bwd"),
                silu_fwd_pipeline: make_pipeline("silu_fwd"),
                silu_bwd_pipeline: make_pipeline("silu_bwd"),
                row_sum_pipeline: make_pipeline("row_sum"),
                vec_add_pipeline: make_pipeline("vec_add"),
                softmax_fwd_pipeline: make_pipeline("softmax_fwd"),
                softmax_bwd_pipeline: make_pipeline("softmax_bwd"),
                layernorm_fwd_pipeline: make_pipeline("layernorm_fwd"),
                adam_step_pipeline: make_pipeline("adam_step"),
                bufs,
                device,
            };

            Some(ctx)
        }

        // ═══ Buffer helpers (pre-allocated, unified memory) ═══

        /// Write data into a pre-allocated scratch buffer (memcpy, no allocation).
        fn write_buf(&self, idx: usize, data: &[f32]) {
            assert!(
                data.len() <= BUF_CAP,
                "Data ({}) exceeds buffer capacity ({})",
                data.len(),
                BUF_CAP
            );
            let ptr = self.bufs[idx].contents() as *mut f32;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
        }

        /// Read data from a pre-allocated scratch buffer.
        fn read_buf(&self, idx: usize, count: usize) -> Vec<f32> {
            let ptr = self.bufs[idx].contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec()
        }

        /// For data that doesn't fit in scratch buffers (rare), allocate on demand.
        fn make_buffer(&self, data: &[f32]) -> Buffer {
            self.device.new_buffer_with_data(
                data.as_ptr() as *const c_void,
                (data.len() * mem::size_of::<f32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        fn set_u32(enc: &ComputeCommandEncoderRef, index: u64, val: u32) {
            enc.set_bytes(
                index,
                mem::size_of::<u32>() as u64,
                &val as *const u32 as *const c_void,
            );
        }

        fn set_f32(enc: &ComputeCommandEncoderRef, index: u64, val: f32) {
            enc.set_bytes(
                index,
                mem::size_of::<f32>() as u64,
                &val as *const f32 as *const c_void,
            );
        }

        // ═══ Core operations (all use pre-allocated scratch buffers) ═══

        /// General matrix multiply: C = A @ B
        /// A: [M, K], B: [K, N] -> C: [M, N]
        /// Uses scratch bufs [0]=A, [1]=B, [2]=C
        pub fn gemm(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
            self.write_buf(0, a);
            self.write_buf(1, b);

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.gemm_pipeline);
            enc.set_buffer(0, Some(&self.bufs[0]), 0);
            enc.set_buffer(1, Some(&self.bufs[1]), 0);
            enc.set_buffer(2, Some(&self.bufs[2]), 0);
            Self::set_u32(enc, 3, m as u32);
            Self::set_u32(enc, 4, k as u32);
            Self::set_u32(enc, 5, n as u32);

            let groups_x = (n as u64 + TILE - 1) / TILE;
            let groups_y = (m as u64 + TILE - 1) / TILE;
            enc.dispatch_thread_groups(
                MTLSize::new(groups_x, groups_y, 1),
                MTLSize::new(TILE, TILE, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            self.read_buf(2, m * n)
        }

        /// Fused linear forward: Y = X @ W + bias
        /// Uses scratch bufs [0]=X, [1]=W, [2]=bias, [3]=Y
        pub fn linear_fwd(
            &self,
            x: &[f32],
            w: &[f32],
            b: &[f32],
            s: usize,
            id: usize,
            od: usize,
        ) -> Vec<f32> {
            self.write_buf(0, x);
            self.write_buf(1, w);
            self.write_buf(2, b);

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.linear_fwd_pipeline);
            enc.set_buffer(0, Some(&self.bufs[0]), 0);
            enc.set_buffer(1, Some(&self.bufs[1]), 0);
            enc.set_buffer(2, Some(&self.bufs[2]), 0);
            enc.set_buffer(3, Some(&self.bufs[3]), 0);
            Self::set_u32(enc, 4, s as u32);
            Self::set_u32(enc, 5, id as u32);
            Self::set_u32(enc, 6, od as u32);

            let groups_x = (od as u64 + TILE - 1) / TILE;
            let groups_y = (s as u64 + TILE - 1) / TILE;
            enc.dispatch_thread_groups(
                MTLSize::new(groups_x, groups_y, 1),
                MTLSize::new(TILE, TILE, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            self.read_buf(3, s * od)
        }

        /// Linear backward: returns (dx, dw, db)
        pub fn linear_bwd(
            &self,
            dout: &[f32],
            x: &[f32],
            w: &[f32],
            s: usize,
            id: usize,
            od: usize,
        ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            // dx = dout @ W^T
            let w_t = transpose(w, id, od);
            let dx = self.gemm(dout, &w_t, s, od, id);

            // dw = X^T @ dout
            let x_t = transpose(x, s, id);
            let dw = self.gemm(&x_t, dout, id, s, od);

            // db = sum(dout, axis=0)
            let db = self.row_sum(dout, s, od);

            (dx, dw, db)
        }

        /// Element-wise activation forward. Uses bufs [0]=X, [1]=Y
        pub fn activation_fwd(&self, x: &[f32], name: &str) -> Vec<f32> {
            let pipeline = match name {
                "relu" => &self.relu_fwd_pipeline,
                "silu" => &self.silu_fwd_pipeline,
                _ => &self.gelu_fwd_pipeline,
            };

            let count = x.len();
            self.write_buf(0, x);

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(&self.bufs[0]), 0);
            enc.set_buffer(1, Some(&self.bufs[1]), 0);
            Self::set_u32(enc, 2, count as u32);

            enc.dispatch_threads(MTLSize::new(count as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            self.read_buf(1, count)
        }

        /// Element-wise activation backward. Uses bufs [0]=dout, [1]=X, [2]=dX
        pub fn activation_bwd(&self, dout: &[f32], x: &[f32], name: &str) -> Vec<f32> {
            let pipeline = match name {
                "relu" => &self.relu_bwd_pipeline,
                "silu" => &self.silu_bwd_pipeline,
                _ => &self.gelu_bwd_pipeline,
            };

            let count = x.len();
            self.write_buf(0, dout);
            self.write_buf(1, x);

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            enc.set_buffer(0, Some(&self.bufs[0]), 0);
            enc.set_buffer(1, Some(&self.bufs[1]), 0);
            enc.set_buffer(2, Some(&self.bufs[2]), 0);
            Self::set_u32(enc, 3, count as u32);

            enc.dispatch_threads(MTLSize::new(count as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            self.read_buf(2, count)
        }

        /// Row sum: out[j] = sum_i mat[i, j]. Uses bufs [0]=mat, [1]=out
        pub fn row_sum(&self, mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            self.write_buf(0, mat);

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.row_sum_pipeline);
            enc.set_buffer(0, Some(&self.bufs[0]), 0);
            enc.set_buffer(1, Some(&self.bufs[1]), 0);
            Self::set_u32(enc, 2, rows as u32);
            Self::set_u32(enc, 3, cols as u32);

            enc.dispatch_threads(MTLSize::new(cols as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            self.read_buf(1, cols)
        }

        /// Element-wise add: c = a + b. Uses bufs [0]=a, [1]=b, [2]=c
        pub fn vec_add(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
            let count = a.len();
            self.write_buf(0, a);
            self.write_buf(1, b);

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.vec_add_pipeline);
            enc.set_buffer(0, Some(&self.bufs[0]), 0);
            enc.set_buffer(1, Some(&self.bufs[1]), 0);
            enc.set_buffer(2, Some(&self.bufs[2]), 0);
            Self::set_u32(enc, 3, count as u32);

            enc.dispatch_threads(MTLSize::new(count as u64, 1, 1), MTLSize::new(256, 1, 1));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            self.read_buf(2, count)
        }

        /// Row-wise softmax forward. Uses bufs [0]=X, [1]=Y
        pub fn softmax_fwd(&self, x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            self.write_buf(0, x);

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.softmax_fwd_pipeline);
            enc.set_buffer(0, Some(&self.bufs[0]), 0);
            enc.set_buffer(1, Some(&self.bufs[1]), 0);
            Self::set_u32(enc, 2, rows as u32);
            Self::set_u32(enc, 3, cols as u32);

            enc.dispatch_threads(
                MTLSize::new(rows as u64, 1, 1),
                MTLSize::new(256.min(rows as u64), 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            self.read_buf(1, rows * cols)
        }

        /// Row-wise softmax backward. Uses bufs [0]=dout, [1]=Y, [2]=dX
        pub fn softmax_bwd(&self, dout: &[f32], y: &[f32], rows: usize, cols: usize) -> Vec<f32> {
            self.write_buf(0, dout);
            self.write_buf(1, y);

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.softmax_bwd_pipeline);
            enc.set_buffer(0, Some(&self.bufs[0]), 0);
            enc.set_buffer(1, Some(&self.bufs[1]), 0);
            enc.set_buffer(2, Some(&self.bufs[2]), 0);
            Self::set_u32(enc, 3, rows as u32);
            Self::set_u32(enc, 4, cols as u32);

            enc.dispatch_threads(
                MTLSize::new(rows as u64, 1, 1),
                MTLSize::new(256.min(rows as u64), 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            self.read_buf(2, rows * cols)
        }

        /// Layer norm forward. Returns (output, x_hat, inv_std).
        /// Uses bufs [0]=X, [1]=gamma, [2]=beta, [3]=out, [4]=xhat, [5]=inv_std
        pub fn layernorm_fwd(
            &self,
            x: &[f32],
            gamma: &[f32],
            beta: &[f32],
            s: usize,
            d: usize,
        ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            self.write_buf(0, x);
            self.write_buf(1, gamma);
            self.write_buf(2, beta);

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.layernorm_fwd_pipeline);
            enc.set_buffer(0, Some(&self.bufs[0]), 0);
            enc.set_buffer(1, Some(&self.bufs[1]), 0);
            enc.set_buffer(2, Some(&self.bufs[2]), 0);
            enc.set_buffer(3, Some(&self.bufs[3]), 0);
            enc.set_buffer(4, Some(&self.bufs[4]), 0);
            enc.set_buffer(5, Some(&self.bufs[5]), 0);
            Self::set_u32(enc, 6, s as u32);
            Self::set_u32(enc, 7, d as u32);

            enc.dispatch_threads(
                MTLSize::new(s as u64, 1, 1),
                MTLSize::new(256.min(s as u64), 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            (
                self.read_buf(3, s * d),
                self.read_buf(4, s * d),
                self.read_buf(5, s),
            )
        }
    }

    /// CPU transpose: [rows, cols] -> [cols, rows]
    fn transpose(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = m[r * cols + c];
            }
        }
        out
    }
}

#[cfg(feature = "gpu")]
pub use metal_compute::GpuContext;

// ═══ Stub when GPU feature is disabled ═══
#[cfg(not(feature = "gpu"))]
pub struct GpuContext;

#[cfg(not(feature = "gpu"))]
impl GpuContext {
    pub fn new() -> Option<Self> {
        None
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::metal_compute::GpuContext;

    #[test]
    fn test_gpu_gemm_correctness() {
        let gpu = GpuContext::new().expect("Need Metal device for GPU test");

        // [2, 3] @ [3, 2] = [2, 2]
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = gpu.gemm(&a, &b, 2, 3, 2);

        // row0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // row1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        assert_eq!(c.len(), 4);
        assert!((c[0] - 58.0).abs() < 0.01, "c[0]={}", c[0]);
        assert!((c[1] - 64.0).abs() < 0.01, "c[1]={}", c[1]);
        assert!((c[2] - 139.0).abs() < 0.01, "c[2]={}", c[2]);
        assert!((c[3] - 154.0).abs() < 0.01, "c[3]={}", c[3]);
    }

    #[test]
    fn test_gpu_linear_fwd_correctness() {
        let gpu = GpuContext::new().expect("Need Metal device for GPU test");

        // Compare GPU vs CPU linear_fwd
        let s = 4;
        let id = 8;
        let od = 6;
        let x: Vec<f32> = (0..s * id).map(|i| (i as f32) * 0.1).collect();
        let w: Vec<f32> = (0..id * od).map(|i| ((i as f32) * 0.01 - 0.24)).collect();
        let b: Vec<f32> = (0..od).map(|i| i as f32 * 0.05).collect();

        let gpu_result = gpu.linear_fwd(&x, &w, &b, s, id, od);

        // CPU reference
        let mut cpu_result = vec![0.0f32; s * od];
        for t in 0..s {
            for j in 0..od {
                cpu_result[t * od + j] = b[j];
                for i in 0..id {
                    cpu_result[t * od + j] += x[t * id + i] * w[i * od + j];
                }
            }
        }

        for i in 0..gpu_result.len() {
            assert!(
                (gpu_result[i] - cpu_result[i]).abs() < 0.01,
                "Mismatch at {}: gpu={}, cpu={}",
                i,
                gpu_result[i],
                cpu_result[i]
            );
        }
    }

    #[test]
    fn test_gpu_linear_bwd_correctness() {
        let gpu = GpuContext::new().expect("Need Metal device for GPU test");

        let s = 3;
        let id = 4;
        let od = 5;
        let dout: Vec<f32> = (0..s * od).map(|i| (i as f32) * 0.1 - 0.7).collect();
        let x: Vec<f32> = (0..s * id).map(|i| (i as f32) * 0.05).collect();
        let w: Vec<f32> = (0..id * od).map(|i| (i as f32) * 0.02 - 0.2).collect();

        let (gpu_dx, gpu_dw, gpu_db) = gpu.linear_bwd(&dout, &x, &w, s, id, od);

        // CPU reference
        let mut cpu_dx = vec![0.0f32; s * id];
        let mut cpu_dw = vec![0.0f32; id * od];
        let mut cpu_db = vec![0.0f32; od];
        for t in 0..s {
            for j in 0..od {
                let d = dout[t * od + j];
                cpu_db[j] += d;
                for i in 0..id {
                    cpu_dx[t * id + i] += d * w[i * od + j];
                    cpu_dw[i * od + j] += x[t * id + i] * d;
                }
            }
        }

        for i in 0..gpu_dx.len() {
            assert!(
                (gpu_dx[i] - cpu_dx[i]).abs() < 0.01,
                "dx[{}]: gpu={} cpu={}",
                i,
                gpu_dx[i],
                cpu_dx[i]
            );
        }
        for i in 0..gpu_dw.len() {
            assert!(
                (gpu_dw[i] - cpu_dw[i]).abs() < 0.01,
                "dw[{}]: gpu={} cpu={}",
                i,
                gpu_dw[i],
                cpu_dw[i]
            );
        }
        for i in 0..gpu_db.len() {
            assert!(
                (gpu_db[i] - cpu_db[i]).abs() < 0.01,
                "db[{}]: gpu={} cpu={}",
                i,
                gpu_db[i],
                cpu_db[i]
            );
        }
    }

    #[test]
    fn test_gpu_activation_fwd_bwd() {
        let gpu = GpuContext::new().expect("Need Metal device for GPU test");

        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let gelu = gpu.activation_fwd(&x, "gelu");
        assert_eq!(gelu.len(), 6);
        // GELU(0) ≈ 0
        assert!(
            gelu[2].abs() < 0.01,
            "GELU(0) should be ~0, got {}",
            gelu[2]
        );
        // GELU(2) ≈ 1.93
        assert!(gelu[5] > 1.5, "GELU(2) should be >1.5, got {}", gelu[5]);

        let dout = vec![1.0; 6];
        let grad = gpu.activation_bwd(&dout, &x, "gelu");
        assert_eq!(grad.len(), 6);
    }

    #[test]
    fn test_gpu_transformer_forward_matches_cpu() {
        use crate::semantic::transformer::{TransformerConfig, TransformerEncoder};

        let gpu = GpuContext::new().expect("Need Metal device for GPU test");

        let config = TransformerConfig {
            vocab_size: 50,
            hidden_dim: 16,
            num_heads: 2,
            num_layers: 1,
            ff_dim: 32,
            max_seq_len: 16,
            pooling: "mean".to_string(),
            activation: "gelu".to_string(),
        };
        let encoder = TransformerEncoder::new(config);
        let tokens = vec![1, 2, 3, 4, 5];

        let (cpu_emb, _) = encoder.forward(&tokens);
        let (gpu_emb, _) = encoder.forward_gpu(&tokens, &gpu);

        assert_eq!(cpu_emb.len(), gpu_emb.len());
        let max_diff: f32 = cpu_emb
            .iter()
            .zip(gpu_emb.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.05,
            "CPU vs GPU forward max diff = {} (should be < 0.05)",
            max_diff
        );
    }
}
