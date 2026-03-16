//! Evaluation metrics.
//!
//! Utilities for measuring classifier performance: accuracy, confusion matrices,
//! and Pearson correlation for regression tasks.

pub use crate::semantic::train::pearson;

/// Compute classification accuracy from predicted and ground-truth class indices.
pub fn accuracy(predictions: &[usize], ground_truth: &[usize]) -> f32 {
    assert_eq!(predictions.len(), ground_truth.len());
    if predictions.is_empty() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .zip(ground_truth)
        .filter(|(p, g)| p == g)
        .count();
    correct as f32 / predictions.len() as f32
}

/// Compute an N-class confusion matrix.
///
/// Returns `matrix[true_class][predicted_class]` counts.
pub fn confusion_matrix(
    predictions: &[usize],
    ground_truth: &[usize],
    num_classes: usize,
) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0usize; num_classes]; num_classes];
    for (&pred, &gt) in predictions.iter().zip(ground_truth) {
        if pred < num_classes && gt < num_classes {
            matrix[gt][pred] += 1;
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy() {
        assert!((accuracy(&[0, 1, 2, 0], &[0, 1, 2, 1]) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_accuracy_empty() {
        assert_eq!(accuracy(&[], &[]), 0.0);
    }

    #[test]
    fn test_confusion_matrix() {
        let preds = vec![0, 1, 2, 0, 1];
        let truth = vec![0, 1, 1, 0, 2];
        let cm = confusion_matrix(&preds, &truth, 3);
        assert_eq!(cm[0][0], 2); // true=0, pred=0
        assert_eq!(cm[1][1], 1); // true=1, pred=1
        assert_eq!(cm[1][2], 1); // true=1, pred=2
        assert_eq!(cm[2][1], 1); // true=2, pred=1
    }
}
