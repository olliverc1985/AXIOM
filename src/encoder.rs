//! Structural encoder producing 128-dimensional vocabulary-independent representations.
//!
//! The encoder divides its output into five feature groups:
//! - G1 (26 dims): Character n-gram profiles, amplified 3.0×
//! - G2 (36 dims): Syntactic proxy features, amplified 3.0×
//! - G3 (39 dims): Position-weighted token signal
//! - G4 (15 dims): Scalar complexity measures, amplified 2.0×
//! - G5 (12 dims): Structural syntax features, amplified 3.0×

use crate::types::*;
use std::collections::{HashMap, HashSet};

/// Amplification factors per feature group.
const G1_AMP: f32 = 3.0;
const G2_AMP: f32 = 3.0;
const G3_AMP: f32 = 1.0; // G3 has no amplification mentioned in paper
const G4_AMP: f32 = 2.0;
const G5_AMP: f32 = 3.0;

/// Structural encoder for AXIOM.
#[derive(Debug, Clone)]
pub struct StructuralEncoder {
    /// Persisted G5 calibration parameters.
    pub g5_simple_mean_norm: f32,
    pub g5_complex_mean_norm: f32,

    /// Running accumulators for G5 calibration during training.
    g5_simple_norms: Vec<f32>,
    g5_complex_norms: Vec<f32>,
}

impl StructuralEncoder {
    pub fn new() -> Self {
        Self {
            g5_simple_mean_norm: 2.4596,
            g5_complex_mean_norm: 3.3316,
            g5_simple_norms: Vec::new(),
            g5_complex_norms: Vec::new(),
        }
    }

    /// Produce a 128-dimensional encoding of the input text.
    pub fn encode(&self, text: &str) -> Vec<f32> {
        let tokens = tokenize(text);
        let mut embedding = Vec::with_capacity(EMBEDDING_DIM);

        let g1 = self.compute_g1(text);
        let g2 = self.compute_g2(text, &tokens);
        let g3 = self.compute_g3(text, &tokens);
        let g4 = self.compute_g4(text, &tokens);
        let g5 = self.compute_g5(text, &tokens);

        debug_assert_eq!(g1.len(), G1_DIM);
        debug_assert_eq!(g2.len(), G2_DIM);
        debug_assert_eq!(g3.len(), G3_DIM);
        debug_assert_eq!(g4.len(), G4_DIM);
        debug_assert_eq!(g5.len(), G5_DIM);

        embedding.extend(g1);
        embedding.extend(g2);
        embedding.extend(g3);
        embedding.extend(g4);
        embedding.extend(g5);

        debug_assert_eq!(embedding.len(), EMBEDDING_DIM);
        embedding
    }

    /// Encode with sentence chunking for multi-sentence inputs.
    /// Splits on sentence boundaries, encodes each independently,
    /// and averages the G5 norms.
    pub fn encode_chunked(&self, text: &str) -> (Vec<f32>, f32) {
        let sentences = split_sentences(text);
        if sentences.len() <= 1 {
            let enc = self.encode(text);
            let g5_norm = g5_norm_from_embedding(&enc);
            return (enc, g5_norm);
        }

        let encodings: Vec<Vec<f32>> = sentences.iter().map(|s| self.encode(s)).collect();
        let n = encodings.len() as f32;

        // Average G5 norms across chunks
        let avg_g5_norm: f32 =
            encodings.iter().map(|e| g5_norm_from_embedding(e)).sum::<f32>() / n;

        // Average the full embeddings
        let mut avg = vec![0.0f32; EMBEDDING_DIM];
        for enc in &encodings {
            for (i, v) in enc.iter().enumerate() {
                avg[i] += v / n;
            }
        }

        (avg, avg_g5_norm)
    }

    /// G5 norm from a full embedding vector.
    pub fn g5_norm(embedding: &[f32]) -> f32 {
        g5_norm_from_embedding(embedding)
    }

    /// Compute G5 penalty for Surface confidence.
    pub fn g5_penalty(&self, g5_norm: f32) -> f32 {
        let range = self.g5_complex_mean_norm - self.g5_simple_mean_norm;
        if range.abs() < 1e-8 {
            return 0.0;
        }
        ((g5_norm - self.g5_simple_mean_norm) / range).clamp(0.0, 1.0)
    }

    /// Accumulate G5 norm for calibration during training.
    pub fn accumulate_g5(&mut self, g5_norm: f32, is_simple: bool) {
        if is_simple {
            self.g5_simple_norms.push(g5_norm);
        } else {
            self.g5_complex_norms.push(g5_norm);
        }
    }

    /// Finalize G5 calibration from accumulated norms.
    pub fn calibrate_g5(&mut self) {
        if !self.g5_simple_norms.is_empty() {
            self.g5_simple_mean_norm =
                self.g5_simple_norms.iter().sum::<f32>() / self.g5_simple_norms.len() as f32;
        }
        if !self.g5_complex_norms.is_empty() {
            self.g5_complex_mean_norm =
                self.g5_complex_norms.iter().sum::<f32>() / self.g5_complex_norms.len() as f32;
        }
    }

    // ── G1: Character n-gram profiles (26 dimensions, amplified 3.0×) ───

    fn compute_g1(&self, text: &str) -> Vec<f32> {
        let mut features = vec![0.0f32; G1_DIM];
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        let total = chars.len().max(1) as f32;

        // Unigram letter frequencies (26 features for a-z)
        // Normalized by total character count
        for &c in &chars {
            if c.is_ascii_lowercase() {
                let idx = (c as u8 - b'a') as usize;
                features[idx] += 1.0 / total;
            }
        }

        // Apply amplification
        for v in &mut features {
            *v *= G1_AMP;
        }

        features
    }

    // ── G2: Syntactic proxy features (36 dimensions, amplified 3.0×) ────

    fn compute_g2(&self, text: &str, tokens: &[String]) -> Vec<f32> {
        let mut features = vec![0.0f32; G2_DIM];
        let n_tokens = tokens.len().max(1) as f32;

        // [0..8] Nested clause depth indicators
        // Count subordinating conjunctions and relative pronouns as clause openers
        let clause_openers = [
            "that", "which", "who", "whom", "whose", "where", "when", "while", "although",
            "because", "since", "unless", "if", "whether", "after", "before", "until", "whereas",
        ];
        let mut depth = 0usize;
        for token in tokens {
            let lower = token.to_lowercase();
            if clause_openers.contains(&lower.as_str()) {
                depth += 1;
            }
        }
        // Encode depth as one-hot in first 8 slots
        let depth_idx = depth.min(7);
        features[depth_idx] = 1.0;

        // [8..16] Pronoun density by category (8 categories)
        let pronoun_groups: [&[&str]; 8] = [
            &["i", "me", "my", "mine", "myself"],
            &["you", "your", "yours", "yourself"],
            &["he", "him", "his", "himself"],
            &["she", "her", "hers", "herself"],
            &["it", "its", "itself"],
            &["we", "us", "our", "ours", "ourselves"],
            &["they", "them", "their", "theirs", "themselves"],
            &["this", "that", "these", "those"],
        ];
        for (i, group) in pronoun_groups.iter().enumerate() {
            let count = tokens
                .iter()
                .filter(|t| group.contains(&t.to_lowercase().as_str()))
                .count();
            features[8 + i] = count as f32 / n_tokens;
        }

        // [16..24] Hapax ratio features
        // Words that appear exactly once / total unique words
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        for t in tokens {
            *word_counts.entry(t.to_lowercase()).or_insert(0) += 1;
        }
        let unique = word_counts.len().max(1) as f32;
        let hapax = word_counts.values().filter(|&&c| c == 1).count() as f32;
        let hapax_ratio = hapax / unique;
        // Spread across 8 bins based on ratio magnitude
        let bin = ((hapax_ratio * 8.0) as usize).min(7);
        features[16 + bin] = hapax_ratio;

        // [24..30] Sentence structure indicators
        // Average clause length, max clause length, clause count ratio
        let sentences = split_sentences(text);
        let n_sentences = sentences.len().max(1) as f32;
        let words_per_sentence: Vec<f32> =
            sentences.iter().map(|s| tokenize(s).len() as f32).collect();
        let avg_words = words_per_sentence.iter().sum::<f32>() / n_sentences;
        let max_words = words_per_sentence
            .iter()
            .cloned()
            .fold(0.0f32, f32::max);
        features[24] = (avg_words / 30.0).min(1.0); // normalized avg sentence length
        features[25] = (max_words / 60.0).min(1.0); // normalized max sentence length
        features[26] = (n_sentences / 10.0).min(1.0); // normalized sentence count
        features[27] = (depth as f32 / n_sentences).min(1.0); // clauses per sentence
        features[28] = if n_sentences > 1.0 {
            let mean = avg_words;
            let var = words_per_sentence
                .iter()
                .map(|w| (w - mean).powi(2))
                .sum::<f32>()
                / n_sentences;
            (var.sqrt() / 20.0).min(1.0)
        } else {
            0.0
        };
        features[29] = (clause_openers
            .iter()
            .filter(|&&o| text.to_lowercase().contains(o))
            .count() as f32
            / 18.0)
            .min(1.0);

        // [30..36] Additional syntactic markers
        // Passive voice indicators, coordination, parallelism
        let passive_markers = ["is", "are", "was", "were", "been", "being"];
        let coordinating = ["and", "but", "or", "nor", "for", "yet", "so"];
        features[30] = tokens
            .iter()
            .filter(|t| passive_markers.contains(&t.to_lowercase().as_str()))
            .count() as f32
            / n_tokens;
        features[31] = tokens
            .iter()
            .filter(|t| coordinating.contains(&t.to_lowercase().as_str()))
            .count() as f32
            / n_tokens;
        // Comma density as coordination signal
        features[32] = text.chars().filter(|&c| c == ',').count() as f32 / n_tokens;
        // Semicolon density
        features[33] = text.chars().filter(|&c| c == ';').count() as f32 / n_tokens;
        // Parenthetical count
        features[34] = text.chars().filter(|&c| c == '(').count() as f32 / n_tokens;
        // Question mark presence
        features[35] = if text.contains('?') { 1.0 } else { 0.0 };

        // Apply amplification
        for v in &mut features {
            *v *= G2_AMP;
        }

        features
    }

    // ── G3: Position-weighted token signal (39 dimensions) ──────────────

    fn compute_g3(&self, _text: &str, tokens: &[String]) -> Vec<f32> {
        let mut features = vec![0.0f32; G3_DIM];
        let n = tokens.len();
        if n == 0 {
            return features;
        }

        // Position-weighted character diversity in sliding windows.
        // Divide the token sequence into windows and compute character diversity
        // weighted by position (early tokens weight more for intent detection).
        let n_windows = G3_DIM.min(n);
        let window_size = (n as f32 / n_windows as f32).ceil() as usize;

        for (wi, chunk_start) in (0..n).step_by(window_size.max(1)).enumerate() {
            if wi >= G3_DIM {
                break;
            }
            let chunk_end = (chunk_start + window_size).min(n);
            let window_tokens = &tokens[chunk_start..chunk_end];

            // Character diversity in this window
            let mut chars_seen = HashSet::new();
            let mut total_chars = 0usize;
            for t in window_tokens {
                for c in t.to_lowercase().chars() {
                    if c.is_alphanumeric() {
                        chars_seen.insert(c);
                        total_chars += 1;
                    }
                }
            }

            let diversity = if total_chars > 0 {
                chars_seen.len() as f32 / total_chars as f32
            } else {
                0.0
            };

            // Position weight: earlier windows get higher weight
            let pos_weight = 1.0 - (wi as f32 / G3_DIM as f32) * 0.5;
            features[wi] = diversity * pos_weight;
        }

        // Apply amplification (1.0× for G3)
        for v in &mut features {
            *v *= G3_AMP;
        }

        features
    }

    // ── G4: Scalar complexity measures (15 dimensions, amplified 2.0×) ──

    fn compute_g4(&self, text: &str, tokens: &[String]) -> Vec<f32> {
        let mut features = vec![0.0f32; G4_DIM];
        let n = tokens.len().max(1) as f32;
        let chars: Vec<char> = text.chars().collect();
        let n_chars = chars.len().max(1) as f32;

        // [0] Type-token ratio
        let unique: HashSet<String> = tokens.iter().map(|t| t.to_lowercase()).collect();
        features[0] = unique.len() as f32 / n;

        // [1] Punctuation density
        let punct_count = chars.iter().filter(|c| c.is_ascii_punctuation()).count();
        features[1] = punct_count as f32 / n_chars;

        // [2] Average word length
        let avg_len: f32 = if tokens.is_empty() {
            0.0
        } else {
            tokens.iter().map(|t| t.len() as f32).sum::<f32>() / n
        };
        features[2] = (avg_len / 10.0).min(1.0);

        // [3] Max word length (normalized)
        let max_len = tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        features[3] = (max_len as f32 / 20.0).min(1.0);

        // [4] Digit density
        features[4] = chars.iter().filter(|c| c.is_ascii_digit()).count() as f32 / n_chars;

        // [5] Uppercase density
        features[5] = chars.iter().filter(|c| c.is_uppercase()).count() as f32 / n_chars;

        // [6] Token count (log-scaled)
        features[6] = (n.ln() / 6.0).min(1.0).max(0.0); // ln(400) ≈ 6

        // [7] Character count (log-scaled)
        features[7] = (n_chars.ln() / 8.0).min(1.0).max(0.0);

        // [8] Whitespace ratio
        features[8] = chars.iter().filter(|c| c.is_whitespace()).count() as f32 / n_chars;

        // [9] Special character ratio (non-alphanumeric, non-whitespace, non-basic-punct)
        features[9] = chars
            .iter()
            .filter(|c| !c.is_alphanumeric() && !c.is_whitespace() && !c.is_ascii_punctuation())
            .count() as f32
            / n_chars;

        // [10] Vocabulary richness (Yule's K approximation)
        let mut freq_of_freq: HashMap<usize, usize> = HashMap::new();
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        for t in tokens {
            *word_freq.entry(t.to_lowercase()).or_insert(0) += 1;
        }
        for &count in word_freq.values() {
            *freq_of_freq.entry(count).or_insert(0) += 1;
        }
        let m1: f32 = n;
        let m2: f32 = freq_of_freq
            .iter()
            .map(|(&i, &vi)| (i * i) as f32 * vi as f32)
            .sum();
        features[10] = if m1 > 1.0 {
            ((m2 - m1) / (m1 * m1)).min(1.0).max(0.0)
        } else {
            0.0
        };

        // [11] Sentence length variance (normalized)
        let sentences = split_sentences(text);
        if sentences.len() > 1 {
            let lens: Vec<f32> = sentences.iter().map(|s| tokenize(s).len() as f32).collect();
            let mean = lens.iter().sum::<f32>() / lens.len() as f32;
            let var = lens.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / lens.len() as f32;
            features[11] = (var.sqrt() / 15.0).min(1.0);
        }

        // [12] Rare bigram density
        // Count character bigrams, fraction that appear only once
        let mut bigram_counts: HashMap<(char, char), usize> = HashMap::new();
        let lower_chars: Vec<char> = text.to_lowercase().chars().collect();
        for w in lower_chars.windows(2) {
            if w[0].is_alphabetic() && w[1].is_alphabetic() {
                *bigram_counts.entry((w[0], w[1])).or_insert(0) += 1;
            }
        }
        let total_bigrams = bigram_counts.len().max(1) as f32;
        let rare_bigrams = bigram_counts.values().filter(|&&c| c == 1).count() as f32;
        features[12] = rare_bigrams / total_bigrams;

        // [13] Contraction density
        let contractions = tokens
            .iter()
            .filter(|t| t.contains('\'') || t.contains('\u{2019}'))
            .count();
        features[13] = contractions as f32 / n;

        // [14] Formality score (formal words / total)
        let formal_markers = [
            "therefore",
            "furthermore",
            "moreover",
            "consequently",
            "nevertheless",
            "notwithstanding",
            "hence",
            "thus",
            "accordingly",
            "whereby",
            "herein",
            "thereof",
            "wherein",
        ];
        let formal_count = tokens
            .iter()
            .filter(|t| formal_markers.contains(&t.to_lowercase().as_str()))
            .count();
        features[14] = (formal_count as f32 / n).min(1.0);

        // Apply amplification
        for v in &mut features {
            *v *= G4_AMP;
        }

        features
    }

    // ── G5: Structural syntax features (12 dimensions, amplified 3.0×) ──
    // These drive the magnitude penalty applied to Surface confidence.

    fn compute_g5(&self, text: &str, tokens: &[String]) -> Vec<f32> {
        let mut features = vec![0.0f32; G5_DIM];
        let n = tokens.len().max(1) as f32;

        // [0..3] Dependency depth proxy
        // Approximate syntactic depth using bracket/parenthesis nesting
        // and subordinating conjunction chains.
        let mut max_depth = 0u32;
        let mut current_depth = 0u32;
        let mut total_depth = 0u32;
        let mut depth_samples = 0u32;
        let subordinators = [
            "that", "which", "who", "because", "although", "while", "if", "when", "where",
            "whether", "unless", "since", "after", "before", "until",
        ];
        for token in tokens {
            let lower = token.to_lowercase();
            if lower == "(" || lower == "[" || lower == "{" {
                current_depth += 1;
            } else if lower == ")" || lower == "]" || lower == "}" {
                current_depth = current_depth.saturating_sub(1);
            } else if subordinators.contains(&lower.as_str()) {
                current_depth += 1;
            }
            max_depth = max_depth.max(current_depth);
            total_depth += current_depth;
            depth_samples += 1;
        }
        let avg_depth = if depth_samples > 0 {
            total_depth as f32 / depth_samples as f32
        } else {
            0.0
        };
        features[0] = (max_depth as f32 / 6.0).min(1.0);
        features[1] = (avg_depth / 3.0).min(1.0);
        features[2] = (current_depth as f32 / 4.0).min(1.0); // unresolved depth

        // [3..6] Constituent length variance
        // Split into clauses (by commas, semicolons, conjunctions) and measure length variance
        let clause_separators = [",", ";", "and", "but", "or", "while", "whereas"];
        let mut clause_lengths: Vec<usize> = Vec::new();
        let mut current_clause_len = 0usize;
        for token in tokens {
            if clause_separators.contains(&token.to_lowercase().as_str()) {
                if current_clause_len > 0 {
                    clause_lengths.push(current_clause_len);
                }
                current_clause_len = 0;
            } else {
                current_clause_len += 1;
            }
        }
        if current_clause_len > 0 {
            clause_lengths.push(current_clause_len);
        }

        let n_clauses = clause_lengths.len().max(1) as f32;
        let mean_clause_len = clause_lengths.iter().sum::<usize>() as f32 / n_clauses;
        let clause_len_var = if clause_lengths.len() > 1 {
            clause_lengths
                .iter()
                .map(|&l| (l as f32 - mean_clause_len).powi(2))
                .sum::<f32>()
                / n_clauses
        } else {
            0.0
        };
        features[3] = (mean_clause_len / 15.0).min(1.0);
        features[4] = (clause_len_var.sqrt() / 10.0).min(1.0);
        features[5] = (n_clauses / 8.0).min(1.0);

        // [6..9] Function word position entropy
        // Function words include determiners, prepositions, auxiliaries.
        // Measure entropy of their relative positions in the sentence.
        let function_words: HashSet<&str> = [
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "by", "from", "is",
            "are", "was", "were", "has", "have", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "be", "been", "being", "not",
            "no", "nor", "as", "than", "so", "if", "then",
        ]
        .iter()
        .copied()
        .collect();

        let mut func_positions: Vec<f32> = Vec::new();
        for (i, token) in tokens.iter().enumerate() {
            if function_words.contains(token.to_lowercase().as_str()) {
                func_positions.push(i as f32 / n);
            }
        }

        let func_density = func_positions.len() as f32 / n;
        let func_entropy = position_entropy(&func_positions, 4);
        let func_spread = if func_positions.len() > 1 {
            let mean_pos = func_positions.iter().sum::<f32>() / func_positions.len() as f32;
            func_positions
                .iter()
                .map(|p| (p - mean_pos).powi(2))
                .sum::<f32>()
                / func_positions.len() as f32
        } else {
            0.0
        };

        features[6] = func_density.min(1.0);
        features[7] = (func_entropy / 2.0).min(1.0); // max entropy of 4 bins = 2.0
        features[8] = func_spread.min(1.0);

        // [9..12] Additional structural signals
        // Relative clause density, prepositional phrase chains, sentence-level coordination
        let relative_pronouns = ["which", "who", "whom", "whose", "that", "where", "when"];
        let prepositions = [
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "through", "between",
            "among", "about", "above", "below", "under", "over",
        ];

        features[9] = tokens
            .iter()
            .filter(|t| relative_pronouns.contains(&t.to_lowercase().as_str()))
            .count() as f32
            / n;

        // Prepositional phrase chain detection: consecutive prepositions within 3 tokens
        let mut prep_chains = 0u32;
        let mut last_prep_pos: Option<usize> = None;
        for (i, token) in tokens.iter().enumerate() {
            if prepositions.contains(&token.to_lowercase().as_str()) {
                if let Some(last) = last_prep_pos {
                    if i - last <= 3 {
                        prep_chains += 1;
                    }
                }
                last_prep_pos = Some(i);
            }
        }
        features[10] = (prep_chains as f32 / n).min(1.0);

        // Sentence-initial variety (how many different first words across sentences)
        let sentences = split_sentences(text);
        let first_words: HashSet<String> = sentences
            .iter()
            .filter_map(|s| {
                let toks = tokenize(s);
                toks.first().map(|t| t.to_lowercase())
            })
            .collect();
        features[11] = if sentences.len() > 1 {
            first_words.len() as f32 / sentences.len() as f32
        } else {
            0.0
        };

        // Apply amplification
        for v in &mut features {
            *v *= G5_AMP;
        }

        features
    }
}

// ── Helper functions ────────────────────────────────────────────────────

/// Simple whitespace tokenizer.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

/// Split text into sentences on punctuation boundaries.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for c in text.chars() {
        current.push(c);
        if c == '.' || c == '!' || c == '?' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() && tokenize(&trimmed).len() >= 2 {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Remaining text without terminal punctuation
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    if sentences.is_empty() {
        sentences.push(text.to_string());
    }

    sentences
}

/// Compute entropy of positions bucketed into `n_bins` bins.
fn position_entropy(positions: &[f32], n_bins: usize) -> f32 {
    if positions.is_empty() || n_bins == 0 {
        return 0.0;
    }

    let mut bins = vec![0u32; n_bins];
    for &p in positions {
        let bin = ((p * n_bins as f32) as usize).min(n_bins - 1);
        bins[bin] += 1;
    }

    let total = positions.len() as f32;
    let mut entropy = 0.0f32;
    for &count in &bins {
        if count > 0 {
            let p = count as f32 / total;
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Extract G5 norm from a full 128-dimensional embedding.
pub fn g5_norm_from_embedding(embedding: &[f32]) -> f32 {
    let g5_start = G1_DIM + G2_DIM + G3_DIM + G4_DIM;
    let g5_end = g5_start + G5_DIM;
    if embedding.len() < g5_end {
        return 0.0;
    }
    let g5_slice = &embedding[g5_start..g5_end];
    let norm_sq: f32 = g5_slice.iter().map(|v| v * v).sum();
    norm_sq.sqrt()
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have same length");
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = norm_a * norm_b + 1e-8;
    (dot / denom).clamp(-1.0, 1.0)
}

/// L2 norm of a vector.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalize a vector to unit length.
pub fn normalize(v: &mut [f32]) {
    let norm = l2_norm(v);
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_dimension() {
        let encoder = StructuralEncoder::new();
        let embedding = encoder.encode("Hello, how are you?");
        assert_eq!(embedding.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_feature_group_dimensions_sum() {
        assert_eq!(G1_DIM + G2_DIM + G3_DIM + G4_DIM + G5_DIM, EMBEDDING_DIM);
    }

    #[test]
    fn test_simple_vs_complex_g5_separation() {
        let encoder = StructuralEncoder::new();
        let simple = "What time is it?";
        let complex = "Although the phenomenon has been observed across multiple contexts, \
                       the underlying mechanisms that drive it remain poorly understood, \
                       which complicates efforts to develop interventions that would \
                       effectively address the root causes while accounting for the \
                       heterogeneous responses observed in different populations.";

        let simple_emb = encoder.encode(simple);
        let complex_emb = encoder.encode(complex);

        let simple_g5 = g5_norm_from_embedding(&simple_emb);
        let complex_g5 = g5_norm_from_embedding(&complex_emb);

        assert!(
            complex_g5 > simple_g5,
            "Complex G5 norm ({}) should exceed simple G5 norm ({})",
            complex_g5,
            simple_g5
        );
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_sentence_splitting() {
        let text = "First sentence. Second sentence! Is this third?";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_g5_penalty() {
        let encoder = StructuralEncoder::new();
        // Below simple mean → penalty 0
        assert_eq!(encoder.g5_penalty(2.0), 0.0);
        // Above complex mean → penalty 1
        assert_eq!(encoder.g5_penalty(4.0), 1.0);
        // Between means → proportional
        let mid = (encoder.g5_simple_mean_norm + encoder.g5_complex_mean_norm) / 2.0;
        let penalty = encoder.g5_penalty(mid);
        assert!((penalty - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_chunked_encoding_single_sentence() {
        let encoder = StructuralEncoder::new();
        let text = "Hello world";
        let (chunked, _) = encoder.encode_chunked(text);
        let direct = encoder.encode(text);
        // Single sentence: chunked should equal direct
        for (a, b) in chunked.iter().zip(direct.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, world! How are you?");
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_empty_input() {
        let encoder = StructuralEncoder::new();
        let embedding = encoder.encode("");
        assert_eq!(embedding.len(), EMBEDDING_DIM);
        // Should not panic on empty input
    }
}
