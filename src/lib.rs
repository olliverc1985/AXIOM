//! AXIOM Core — Adaptive eXecution with Intelligent Operations Memory
//!
//! Sparse dynamic routing architecture for cost-efficient LLM inference.
//! Dual-encoder (structural + semantic) always-fuse design with 3-tier classification.
//! Built from scratch in pure Rust — no external ML framework dependencies.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::field_reassign_with_default)]

pub mod cache;
pub mod gpu;
pub mod graph;
pub mod input;
pub mod router;
pub mod semantic;
pub mod tiers;
pub mod tuner;

use serde::{Deserialize, Serialize};

/// Multi-dimensional tensor with flat storage and shape metadata.
///
/// All operations are implemented from scratch — no external ML dependencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    /// Flat storage of f32 values in row-major order.
    pub data: Vec<f32>,
    /// Shape dimensions (e.g. [3, 4] for a 3x4 matrix).
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor with the given data and shape.
    ///
    /// # Panics
    /// Panics if data length doesn't match the product of shape dimensions.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected
        );
        Self { data, shape }
    }

    /// Create a 1-D tensor (vector) from a slice.
    pub fn from_vec(data: Vec<f32>) -> Self {
        let len = data.len();
        Self {
            data,
            shape: vec![len],
        }
    }

    /// Create a zero tensor with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Dot product of two 1-D tensors.
    ///
    /// # Panics
    /// Panics if tensors are not 1-D or have different lengths.
    pub fn dot(&self, other: &Tensor) -> f32 {
        assert_eq!(self.shape.len(), 1, "dot requires 1-D tensors");
        assert_eq!(other.shape.len(), 1, "dot requires 1-D tensors");
        assert_eq!(self.data.len(), other.data.len(), "dot: length mismatch");
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Cosine similarity between two 1-D tensors.
    ///
    /// Returns 0.0 if either tensor has zero magnitude.
    pub fn cosine_similarity(&self, other: &Tensor) -> f32 {
        let dot = self.dot(other);
        let mag_a: f32 = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = other.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }
        dot / (mag_a * mag_b)
    }

    /// Element-wise addition. Tensors must have the same shape.
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "add: shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Element-wise multiplication (Hadamard product). Tensors must have the same shape.
    pub fn multiply(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "multiply: shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Scalar multiplication.
    pub fn scale(&self, s: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|x| x * s).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Matrix multiplication for 2-D tensors.
    ///
    /// self is [M, K], other is [K, N], result is [M, N].
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "matmul requires 2-D tensors");
        assert_eq!(other.shape.len(), 2, "matmul requires 2-D tensors");
        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(k, other.shape[0], "matmul: inner dimensions must match");
        let n = other.shape[1];

        let mut data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                data[i * n + j] = sum;
            }
        }
        Tensor {
            data,
            shape: vec![m, n],
        }
    }

    /// L2 norm of the tensor.
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Blend two tensors: result = self * alpha + other * (1 - alpha).
    ///
    /// If tensors have different lengths, blends up to the shorter length
    /// and pads the rest from the longer tensor.
    pub fn blend(&self, other: &Tensor, alpha: f32) -> Tensor {
        let len = self.data.len().max(other.data.len());
        let mut data = Vec::with_capacity(len);
        for i in 0..len {
            let a = if i < self.data.len() {
                self.data[i]
            } else {
                0.0
            };
            let b = if i < other.data.len() {
                other.data[i]
            } else {
                0.0
            };
            data.push(a * alpha + b * (1.0 - alpha));
        }
        Tensor {
            data,
            shape: vec![len],
        }
    }

    /// Simple hash for logging — sum of absolute values truncated.
    pub fn content_hash(&self) -> u64 {
        let sum: f64 = self.data.iter().map(|x| (*x as f64).abs()).sum();
        (sum * 1_000_000.0) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0]);
        assert!((a.dot(&b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
        assert!((a.cosine_similarity(&a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = Tensor::from_vec(vec![1.0, 0.0]);
        let b = Tensor::from_vec(vec![0.0, 1.0]);
        assert!(a.cosine_similarity(&b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero() {
        let a = Tensor::from_vec(vec![0.0, 0.0]);
        let b = Tensor::from_vec(vec![1.0, 2.0]);
        assert_eq!(a.cosine_similarity(&b), 0.0);
    }

    #[test]
    fn test_elementwise_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0]);
        let b = Tensor::from_vec(vec![3.0, 4.0]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![4.0, 6.0]);
    }

    #[test]
    fn test_elementwise_multiply() {
        let a = Tensor::from_vec(vec![2.0, 3.0]);
        let b = Tensor::from_vec(vec![4.0, 5.0]);
        let c = a.multiply(&b);
        assert_eq!(c.data, vec![8.0, 15.0]);
    }

    #[test]
    fn test_matmul() {
        // [2,3] x [3,2] = [2,2]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        // row0: 1*7+2*9+3*11 = 7+18+33 = 58,  1*8+2*10+3*12 = 8+20+36 = 64
        // row1: 4*7+5*9+6*11 = 28+45+66 = 139, 4*8+5*10+6*12 = 32+50+72 = 154
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_scale() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0]);
        let b = a.scale(2.0);
        assert_eq!(b.data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.data.len(), 6);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }
}
