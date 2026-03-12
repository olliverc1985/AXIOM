//! Structural Encoder V3 — converts text to fixed 128-dimension tensors
//! using four concatenated feature groups: character n-gram profile,
//! syntactic proxy features, position-weighted token signal, and
//! complexity scalars.
//!
//! **No normalisation on the final vector.** Magnitude encodes sentence
//! complexity: simple sentences differ from complex in n-gram distributions,
//! syntactic structure, and vocabulary richness. This is the primary signal
//! for discriminative routing.
//!
//! Phase 7 doubles from 64-dim (V2) to 128-dim for ~500k parameter capacity.

use crate::input::tokeniser::Tokeniser;
use crate::Tensor;
use std::collections::HashSet;

/// Fixed output dimension: 40 + 32 + 48 + 8 = 128.
pub const OUTPUT_DIM: usize = 128;

/// Character n-gram profile dimensions (Group 1).
const NGRAM_DIMS: usize = 40;

/// Syntactic proxy feature dimensions (Group 2).
const SYNTACTIC_DIMS: usize = 32;

/// Position-weighted token signal dimensions (Group 3).
const TOKEN_DIMS: usize = 48;

/// Complexity scalar dimensions (Group 4).
const SCALAR_DIMS: usize = 8;

/// Maximum sentence length (words) for normalisation.
const MAX_SENTENCE_LEN: f32 = 40.0;

/// Maximum token length (chars) for normalisation.
const MAX_TOKEN_LEN: f32 = 15.0;

/// Maximum dependency depth proxy for normalisation.
const MAX_DEP_DEPTH: f32 = 10.0;

/// Function words for density calculation.
const FUNCTION_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of", "and", "or",
    "but",
];

/// Question words for binary presence feature.
const QUESTION_WORDS: &[&str] = &["who", "what", "where", "when", "why", "how", "which"];

/// Negation words for binary presence feature.
const NEGATION_WORDS: &[&str] = &["not", "never", "no", "neither", "nor"];

/// Subordinating conjunctions for binary presence and depth proxy.
const SUBORDINATING: &[&str] = &[
    "because", "although", "while", "since", "if", "when", "unless",
];

/// Structural Encoder V3 producing fixed 128-dimension tensors from text.
///
/// Four feature groups concatenated:
/// - **Group 1** (dims 0–39): Character n-gram profile — bigram/trigram hash buckets,
///   normalised by total n-gram count. 40 buckets for finer-grained discrimination.
/// - **Group 2** (dims 40–71): Syntactic proxy features — word length octiles, variance,
///   function word density, punctuation, capitalisation, binary markers, structural features.
/// - **Group 3** (dims 72–119): Position-weighted token signal — token IDs folded into
///   48 buckets by `id % 48`, weighted by `1/(1+position)`. NOT normalised.
/// - **Group 4** (dims 120–127): Complexity scalars — token count, TTR, mean length,
///   dependency depth, vocabulary richness, length variation, rare word ratio, clause count.
///
/// No normalisation of final vector — magnitude carries complexity information.
pub struct Encoder {
    /// Output dimension (always 128).
    pub output_dim: usize,
    /// Reference to tokeniser for end-to-end encoding.
    pub tokeniser: Tokeniser,
}

impl Encoder {
    /// Create a new V3 structural encoder.
    ///
    /// The `_output_dim` parameter is accepted for API compatibility but the
    /// encoder always produces 128-dimensional vectors (40+32+48+8).
    pub fn new(_output_dim: usize, tokeniser: Tokeniser) -> Self {
        Self {
            output_dim: OUTPUT_DIM,
            tokeniser,
        }
    }

    /// Compute character n-gram profile (40 dimensions).
    ///
    /// Extracts all character bigrams and trigrams from every word in the
    /// sentence. Each n-gram is hashed to one of 40 buckets using a
    /// multiply-add hash. Counts are normalised by total n-gram count.
    ///
    /// Complex sentences with rare vocabulary produce fundamentally different
    /// n-gram distributions than simple sentences with common words.
    fn ngram_profile(words: &[String]) -> [f32; NGRAM_DIMS] {
        let mut buckets = [0u32; NGRAM_DIMS];
        let mut total_ngrams = 0u32;

        for word in words {
            let chars: Vec<char> = word.chars().collect();
            // Character bigrams
            for window in chars.windows(2) {
                let bucket = Self::ngram_hash(window) % NGRAM_DIMS;
                buckets[bucket] += 1;
                total_ngrams += 1;
            }
            // Character trigrams
            for window in chars.windows(3) {
                let bucket = Self::ngram_hash(window) % NGRAM_DIMS;
                buckets[bucket] += 1;
                total_ngrams += 1;
            }
        }

        let mut profile = [0.0f32; NGRAM_DIMS];
        if total_ngrams > 0 {
            for (i, &count) in buckets.iter().enumerate() {
                profile[i] = count as f32 / total_ngrams as f32;
            }
        }
        profile
    }

    /// Multiply-add hash for a character n-gram (DJB2 variant).
    fn ngram_hash(chars: &[char]) -> usize {
        let mut hash = 5381u32;
        for &ch in chars {
            for byte in (ch as u32).to_le_bytes() {
                if byte == 0 {
                    continue;
                }
                hash = hash.wrapping_mul(33).wrapping_add(byte as u32);
            }
        }
        hash as usize
    }

    /// Compute syntactic proxy features (32 dimensions).
    ///
    /// All computed without a parser:
    /// - Dims 0–7: Average word length in each of 8 sentence octiles, normalised by 15.
    /// - Dim 8: Word length variance, normalised.
    /// - Dim 9: Unique word ratio (type-token ratio).
    /// - Dim 10: Function word density.
    /// - Dim 11: Max word length normalised 0–1 against 15.
    /// - Dim 12: Sentence length normalised 0–1 against 40.
    /// - Dim 13: Punctuation count normalised by word count.
    /// - Dim 14: Capitalisation count normalised by word count.
    /// - Dim 15: Number token count normalised by word count.
    /// - Dim 16: Question word presence (binary).
    /// - Dim 17: Negation presence (binary).
    /// - Dim 18: Subordinating conjunction presence (binary).
    /// - Dim 19: Comma density (commas / word count).
    /// - Dim 20: Semicolon/colon presence (binary).
    /// - Dim 21: Parenthetical/bracket presence (binary).
    /// - Dim 22: Hyphenated word ratio.
    /// - Dim 23: Short word ratio (<=3 chars).
    /// - Dim 24: Long word ratio (>=8 chars).
    /// - Dim 25: Word length range (max - min) normalised.
    /// - Dim 26: Sentence-initial capital (binary).
    /// - Dim 27: All-caps word ratio.
    /// - Dim 28: Vowel ratio in words.
    /// - Dim 29: Consonant cluster density proxy.
    /// - Dim 30: Mean syllable count proxy (vowel groups per word).
    /// - Dim 31: Zero padding.
    fn syntactic_features(words: &[String], raw_text: &str) -> [f32; SYNTACTIC_DIMS] {
        let mut features = [0.0f32; SYNTACTIC_DIMS];
        let n = words.len();
        if n == 0 {
            return features;
        }

        let word_lengths: Vec<f32> = words.iter().map(|w| w.len() as f32).collect();

        // Dims 0–7: Average word length in each of 8 sentence octiles
        let octile_size = (n + 7) / 8; // ceiling division
        for q in 0..8 {
            let start = q * octile_size;
            let end = ((q + 1) * octile_size).min(n);
            if start < end {
                let sum: f32 = word_lengths[start..end].iter().sum();
                features[q] = sum / (end - start) as f32 / MAX_TOKEN_LEN;
            }
        }

        // Dim 8: Word length variance (normalised by MAX_TOKEN_LEN^2)
        let mean_len: f32 = word_lengths.iter().sum::<f32>() / n as f32;
        let variance: f32 =
            word_lengths.iter().map(|l| (l - mean_len).powi(2)).sum::<f32>() / n as f32;
        features[8] = variance / (MAX_TOKEN_LEN * MAX_TOKEN_LEN);

        // Dim 9: Unique word ratio
        let unique: HashSet<&str> = words.iter().map(|s| s.as_str()).collect();
        features[9] = unique.len() as f32 / n as f32;

        // Dim 10: Function word density
        let fn_count = words
            .iter()
            .filter(|w| FUNCTION_WORDS.contains(&w.as_str()))
            .count();
        features[10] = fn_count as f32 / n as f32;

        // Dim 11: Max word length normalised 0–1 against 15
        let max_len = word_lengths.iter().cloned().fold(0.0f32, f32::max);
        let min_len = word_lengths.iter().cloned().fold(f32::MAX, f32::min);
        features[11] = (max_len / MAX_TOKEN_LEN).min(1.0);

        // Dim 12: Sentence length normalised 0–1 against 40
        features[12] = (n as f32 / MAX_SENTENCE_LEN).min(1.0);

        // Dim 13: Punctuation count normalised by word count
        let punct_count = raw_text.chars().filter(|c| c.is_ascii_punctuation()).count();
        features[13] = (punct_count as f32 / n as f32).min(1.0);

        // Dim 14: Capitalisation count normalised by word count (from raw text)
        let cap_count = raw_text.chars().filter(|c| c.is_uppercase()).count();
        features[14] = (cap_count as f32 / n as f32).min(1.0);

        // Dim 15: Number token count normalised by word count
        let num_count = words
            .iter()
            .filter(|w| w.chars().any(|c| c.is_ascii_digit()))
            .count();
        features[15] = (num_count as f32 / n as f32).min(1.0);

        // Dim 16: Question word presence (binary)
        features[16] = if words
            .iter()
            .any(|w| QUESTION_WORDS.contains(&w.as_str()))
        {
            1.0
        } else {
            0.0
        };

        // Dim 17: Negation presence (binary)
        features[17] = if words.iter().any(|w| NEGATION_WORDS.contains(&w.as_str())) {
            1.0
        } else {
            0.0
        };

        // Dim 18: Subordinating conjunction presence (binary)
        features[18] = if words.iter().any(|w| SUBORDINATING.contains(&w.as_str())) {
            1.0
        } else {
            0.0
        };

        // Dim 19: Comma density (commas / word count)
        let comma_count = raw_text.chars().filter(|&c| c == ',').count();
        features[19] = (comma_count as f32 / n as f32).min(1.0);

        // Dim 20: Semicolon/colon presence (binary)
        features[20] = if raw_text.chars().any(|c| c == ';' || c == ':') {
            1.0
        } else {
            0.0
        };

        // Dim 21: Parenthetical/bracket presence (binary)
        features[21] = if raw_text.chars().any(|c| c == '(' || c == ')' || c == '[' || c == ']') {
            1.0
        } else {
            0.0
        };

        // Dim 22: Hyphenated word ratio
        let hyphenated = words.iter().filter(|w| w.contains('-')).count();
        features[22] = (hyphenated as f32 / n as f32).min(1.0);

        // Dim 23: Short word ratio (<=3 chars)
        let short_words = words.iter().filter(|w| w.len() <= 3).count();
        features[23] = short_words as f32 / n as f32;

        // Dim 24: Long word ratio (>=8 chars)
        let long_words = words.iter().filter(|w| w.len() >= 8).count();
        features[24] = long_words as f32 / n as f32;

        // Dim 25: Word length range (max - min) normalised
        features[25] = ((max_len - min_len) / MAX_TOKEN_LEN).min(1.0);

        // Dim 26: Sentence-initial capital (binary)
        features[26] = if raw_text.chars().next().map_or(false, |c| c.is_uppercase()) {
            1.0
        } else {
            0.0
        };

        // Dim 27: All-caps word ratio (words where all alpha chars are uppercase)
        let allcaps = words
            .iter()
            .filter(|w| {
                let alpha: Vec<char> = w.chars().filter(|c| c.is_alphabetic()).collect();
                !alpha.is_empty() && alpha.iter().all(|c| c.is_uppercase())
            })
            .count();
        features[27] = allcaps as f32 / n as f32;

        // Dim 28: Vowel ratio in words
        let total_chars: usize = words.iter().map(|w| w.len()).sum();
        if total_chars > 0 {
            let vowels: usize = words
                .iter()
                .flat_map(|w| w.chars())
                .filter(|c| "aeiouAEIOU".contains(*c))
                .count();
            features[28] = vowels as f32 / total_chars as f32;
        }

        // Dim 29: Consonant cluster density proxy (consecutive consonants / total chars)
        if total_chars > 0 {
            let mut clusters = 0usize;
            let mut in_cluster = false;
            for ch in words.iter().flat_map(|w| w.chars()) {
                if ch.is_alphabetic() && !"aeiouAEIOU".contains(ch) {
                    if !in_cluster {
                        in_cluster = true;
                    }
                    clusters += 1;
                } else {
                    in_cluster = false;
                }
            }
            features[29] = (clusters as f32 / total_chars as f32).min(1.0);
        }

        // Dim 30: Mean syllable count proxy (vowel groups per word)
        if n > 0 {
            let mut total_syllables = 0usize;
            for word in words {
                let mut syl = 0usize;
                let mut prev_vowel = false;
                for ch in word.chars() {
                    let is_vowel = "aeiouAEIOU".contains(ch);
                    if is_vowel && !prev_vowel {
                        syl += 1;
                    }
                    prev_vowel = is_vowel;
                }
                total_syllables += syl.max(1); // every word has at least 1 syllable
            }
            features[30] = (total_syllables as f32 / n as f32 / 5.0).min(1.0); // normalise by 5 syllables
        }

        // Dim 31: Zero padding
        features[31] = 0.0;

        features
    }

    /// Compute position-weighted token signal (48 dimensions).
    ///
    /// Each token at position `i` contributes weight `1/(1+i)` to bucket
    /// `token_id % 48`. NOT normalised — magnitude grows with sentence length.
    fn token_signal(token_ids: &[usize]) -> [f32; TOKEN_DIMS] {
        let mut signal = [0.0f32; TOKEN_DIMS];
        for (pos, &id) in token_ids.iter().enumerate() {
            let weight = 1.0 / (1.0 + pos as f32);
            let bucket = id % TOKEN_DIMS;
            signal[bucket] += weight;
        }
        signal
    }

    /// Compute complexity scalars (8 dimensions).
    ///
    /// - Dim 0: Token count normalised 0–1 against max 40.
    /// - Dim 1: Type-token ratio (unique words / total words).
    /// - Dim 2: Mean token length normalised 0–1 against max 15.
    /// - Dim 3: Dependency depth proxy — (subordinating conjunction count +
    ///   comma count) normalised 0–1 against max 10.
    /// - Dim 4: Vocabulary richness — log(unique) / log(total).
    /// - Dim 5: Length variation ratio — max word length / mean word length.
    /// - Dim 6: Rare word ratio — words with >=8 chars / total words.
    /// - Dim 7: Clause count proxy — (commas + subordinating + semicolons) normalised.
    fn complexity_scalars(words: &[String], raw_text: &str) -> [f32; SCALAR_DIMS] {
        let n = words.len();
        if n == 0 {
            return [0.0; SCALAR_DIMS];
        }

        // Dim 0: Token count normalised
        let token_count_norm = (n as f32 / MAX_SENTENCE_LEN).min(1.0);

        // Dim 1: Type-token ratio
        let unique: HashSet<&str> = words.iter().map(|s| s.as_str()).collect();
        let ttr = unique.len() as f32 / n as f32;

        // Dim 2: Mean token length normalised
        let total_char_len: usize = words.iter().map(|w| w.len()).sum();
        let mean_token_len = (total_char_len as f32 / n as f32 / MAX_TOKEN_LEN).min(1.0);

        // Dim 3: Dependency depth proxy
        let subord_count = words
            .iter()
            .filter(|w| SUBORDINATING.contains(&w.as_str()))
            .count();
        let comma_count = raw_text.chars().filter(|&c| c == ',').count();
        let dep_depth = ((subord_count + comma_count) as f32 / MAX_DEP_DEPTH).min(1.0);

        // Dim 4: Vocabulary richness — log(unique) / log(total)
        let vocab_richness = if n > 1 {
            ((unique.len() as f32).ln() / (n as f32).ln()).min(1.0)
        } else {
            1.0
        };

        // Dim 5: Length variation ratio — max word length / mean word length, normalised
        let mean_wl = total_char_len as f32 / n as f32;
        let max_wl = words.iter().map(|w| w.len()).max().unwrap_or(1) as f32;
        let length_variation = if mean_wl > 0.0 {
            (max_wl / mean_wl / 5.0).min(1.0) // normalise by 5
        } else {
            0.0
        };

        // Dim 6: Rare word ratio — words with >=8 chars / total
        let rare_count = words.iter().filter(|w| w.len() >= 8).count();
        let rare_ratio = rare_count as f32 / n as f32;

        // Dim 7: Clause count proxy — (commas + subordinating + semicolons) normalised
        let semicolons = raw_text.chars().filter(|&c| c == ';').count();
        let clause_proxy = ((comma_count + subord_count + semicolons) as f32 / MAX_DEP_DEPTH).min(1.0);

        [token_count_norm, ttr, mean_token_len, dep_depth,
         vocab_richness, length_variation, rare_ratio, clause_proxy]
    }

    /// Amplification factors for each feature group.
    ///
    /// G1 (n-gram) and G2 (syntactic) are the strongest discriminating groups
    /// between simple and complex sentences — amplified by 3.0 to widen the
    /// directional separation in cosine space. G4 (complexity scalars) amplified
    /// by 2.0. G3 (token signal) unchanged.
    const G1_AMP: f32 = 3.0;
    const G2_AMP: f32 = 3.0;
    const G3_AMP: f32 = 1.0;
    const G4_AMP: f32 = 2.0;

    /// Encode token IDs and raw text into a 128-dim structural tensor.
    ///
    /// Concatenates all four feature groups with group-level amplification
    /// to widen directional separation between simple and complex sentences.
    fn encode_with_features(&self, token_ids: &[usize], text: &str) -> Tensor {
        let words = Tokeniser::split(text);

        // Group 1: Character n-gram profile (dims 0–39) — amplified
        let ngrams = Self::ngram_profile(&words);

        // Group 2: Syntactic proxy features (dims 40–71) — amplified
        let syntactic = Self::syntactic_features(&words, text);

        // Group 3: Position-weighted token signal (dims 72–119) — unchanged
        let tokens = Self::token_signal(token_ids);

        // Group 4: Complexity scalars (dims 120–127) — amplified
        let scalars = Self::complexity_scalars(&words, text);

        // Concatenate with amplification: 40 + 32 + 48 + 8 = 128
        let mut data = Vec::with_capacity(OUTPUT_DIM);
        data.extend(ngrams.iter().map(|v| v * Self::G1_AMP));
        data.extend(syntactic.iter().map(|v| v * Self::G2_AMP));
        data.extend(tokens.iter().map(|v| v * Self::G3_AMP));
        data.extend(scalars.iter().map(|v| v * Self::G4_AMP));

        debug_assert_eq!(data.len(), OUTPUT_DIM);
        Tensor::from_vec(data)
    }

    /// End-to-end: tokenise text and encode to 128-dim tensor.
    ///
    /// Modifies the tokeniser vocabulary (adds new tokens).
    pub fn encode_text(&mut self, text: &str) -> Tensor {
        let ids = self.tokeniser.tokenise(text);
        self.encode_with_features(&ids, text)
    }

    /// End-to-end encoding without modifying vocabulary.
    pub fn encode_text_readonly(&self, text: &str) -> Tensor {
        let ids = self.tokeniser.tokenise_readonly(text);
        self.encode_with_features(&ids, text)
    }

    /// Print a detailed feature vector breakdown for diagnostic purposes.
    ///
    /// Shows all 4 feature groups with labels for a given sentence.
    pub fn print_feature_breakdown(&self, text: &str) {
        let ids = self.tokeniser.tokenise_readonly(text);
        let words = Tokeniser::split(text);
        let ngrams = Self::ngram_profile(&words);
        let syntactic = Self::syntactic_features(&words, text);
        let tokens = Self::token_signal(&ids);
        let scalars = Self::complexity_scalars(&words, text);

        let tensor = self.encode_with_features(&ids, text);

        println!("  \"{}\"", text);
        println!("    Tokens: {} words, {} token IDs", words.len(), ids.len());
        println!("    Norm: {:.4}", tensor.norm());
        println!("    Group 1 — N-gram profile ({}d):", NGRAM_DIMS);
        print!("      ");
        for (i, v) in ngrams.iter().enumerate() {
            print!("{:.3}", v);
            if i < NGRAM_DIMS - 1 {
                print!(", ");
            }
        }
        println!();
        println!("    Group 2 — Syntactic features ({}d):", SYNTACTIC_DIMS);
        let labels = [
            "O1_len", "O2_len", "O3_len", "O4_len", "O5_len", "O6_len", "O7_len", "O8_len",
            "len_var", "uniq_r", "func_wd", "max_len", "sent_ln", "punct", "caps", "nums",
            "quest", "negat", "subord", "comma_d", "semi_cl", "parenth", "hyphen",
            "short_r", "long_r", "len_rng", "init_cp", "allcaps", "vowel_r", "cons_cl",
            "syl_cnt", "pad",
        ];
        for (i, (v, label)) in syntactic.iter().zip(labels.iter()).enumerate() {
            print!("      {:>7}: {:.4}", label, v);
            if (i + 1) % 4 == 0 {
                println!();
            }
        }
        println!("    Group 3 — Token signal ({}d):", TOKEN_DIMS);
        print!("      ");
        for (i, v) in tokens.iter().enumerate() {
            print!("{:.3}", v);
            if i < TOKEN_DIMS - 1 {
                print!(", ");
            }
        }
        println!();
        println!("    Group 4 — Complexity scalars ({}d):", SCALAR_DIMS);
        let scalar_labels = ["tok_cnt", "TTR", "mean_ln", "dep_dep",
                             "voc_rch", "len_var", "rare_r", "clause"];
        for (v, label) in scalars.iter().zip(scalar_labels.iter()) {
            print!("      {:>7}: {:.4}", label, v);
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_produces_128_dims() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let t = enc.encode_text("hello world");
        assert_eq!(t.shape, vec![128]);
        assert_eq!(t.data.len(), OUTPUT_DIM);
    }

    #[test]
    fn test_api_compat_ignores_output_dim() {
        // Encoder always produces 128 dims regardless of what's passed
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(16, tok);
        let t = enc.encode_text("hello world");
        assert_eq!(t.shape, vec![128]);
    }

    #[test]
    fn test_ngram_differs_simple_vs_complex() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let simple = enc.encode_text("the dog runs fast");
        let complex = enc.encode_text(
            "the recursive nature of self-referential systems creates emergent properties that resist reduction",
        );
        // N-gram profile is dims 0..NGRAM_DIMS — distributions should differ
        let simple_ngrams = &simple.data[0..NGRAM_DIMS];
        let complex_ngrams = &complex.data[0..NGRAM_DIMS];
        // At least some buckets should differ by more than 0.01
        let diffs: Vec<f32> = simple_ngrams
            .iter()
            .zip(complex_ngrams.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();
        let max_diff = diffs.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max_diff > 0.01,
            "N-gram profiles should differ between simple and complex (max_diff={:.4})",
            max_diff
        );
    }

    #[test]
    fn test_syntactic_features_differ() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let simple = enc.encode_text("the cat sat");
        let complex = enc.encode_text(
            "the recursive nature of self-referential systems creates emergent properties",
        );
        // Sentence length feature (dim 40 + 12) should be higher for complex
        let simple_sent_len = simple.data[40 + 12];
        let complex_sent_len = complex.data[40 + 12];
        assert!(
            complex_sent_len > simple_sent_len,
            "Sentence length: complex={:.4} should exceed simple={:.4}",
            complex_sent_len,
            simple_sent_len
        );
    }

    #[test]
    fn test_token_signal_order_matters() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let t1 = enc.encode_text("cat sat mat");
        let t2 = enc.encode_text("mat sat cat");
        // Token signal (dims 72..120) should differ due to position weighting
        assert_ne!(
            t1.data[72..120],
            t2.data[72..120],
            "Token signal should differ for different word orders"
        );
    }

    #[test]
    fn test_complexity_scalars_increase() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let simple = enc.encode_text("the cat sat");
        let complex = enc.encode_text(
            "the recursive nature of self-referential systems creates emergent properties",
        );
        // Token count scalar (dim 120) should be higher for complex
        assert!(
            complex.data[120] > simple.data[120],
            "Token count: complex={:.4} should exceed simple={:.4}",
            complex.data[120],
            simple.data[120]
        );
    }

    #[test]
    fn test_empty_text() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let t = enc.encode_text("");
        assert_eq!(t.shape, vec![128]);
        // All values should be 0.0 for empty input
        for (i, &v) in t.data.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "Dim {} should be 0.0 for empty input, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_magnitude_increases_with_length() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let short = enc.encode_text("hello world");
        let long =
            enc.encode_text("the quick brown fox jumps over the lazy dog and runs through fields");
        // Token signal is NOT normalised — more tokens contribute more magnitude
        // Overall norm should be higher for longer sentence
        assert!(
            long.norm() > short.norm(),
            "Long norm {} should exceed short norm {}",
            long.norm(),
            short.norm()
        );
    }

    #[test]
    fn test_different_texts_different_vectors() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let t1 = enc.encode_text("hello world");
        let t2 = enc.encode_text("goodbye universe");
        assert_ne!(t1.data, t2.data);
    }

    #[test]
    fn test_similar_texts_higher_similarity() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let t1 = enc.encode_text("the cat sat on the mat");
        let t2 = enc.encode_text("the cat sat on the rug");
        let t3 = enc.encode_text("quantum mechanics describes wave particle duality");
        let sim_close = t1.cosine_similarity(&t2);
        let sim_far = t1.cosine_similarity(&t3);
        assert!(
            sim_close > sim_far,
            "Similar texts should have higher cosine sim: close={:.4} far={:.4}",
            sim_close,
            sim_far
        );
    }

    #[test]
    fn test_subordinating_conjunction_detection() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let no_subord = enc.encode_text("the cat sat on the mat");
        let with_subord = enc.encode_text("the cat sat because it was tired");
        // Subordinating conjunction feature is dim 40 + 18
        assert!(
            with_subord.data[58] > no_subord.data[58],
            "Subordinating: with={:.1} should exceed without={:.1}",
            with_subord.data[58],
            no_subord.data[58]
        );
    }

    #[test]
    fn test_punctuation_feature() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let no_punct = enc.encode_text("hello world");
        let with_punct = enc.encode_text("hello, world! how are you?");
        // Punctuation feature is dim 40 + 13
        assert!(
            with_punct.data[53] > no_punct.data[53],
            "Punctuation: with={:.4} should exceed without={:.4}",
            with_punct.data[53],
            no_punct.data[53]
        );
    }

    #[test]
    fn test_function_word_density() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let low_fn = enc.encode_text("recursive systems create emergent properties");
        let high_fn = enc.encode_text("the cat is on the mat in the room");
        // Function word density is dim 40 + 10
        assert!(
            high_fn.data[50] > low_fn.data[50],
            "Function word density: high={:.4} should exceed low={:.4}",
            high_fn.data[50],
            low_fn.data[50]
        );
    }

    #[test]
    fn test_dependency_depth_proxy() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let simple = enc.encode_text("the cat sat");
        let complex =
            enc.encode_text("because the cat, which was tired, sat down, it slept");
        // Dependency depth proxy is dim 120 + 3 = 123
        assert!(
            complex.data[123] > simple.data[123],
            "Dep depth: complex={:.4} should exceed simple={:.4}",
            complex.data[123],
            simple.data[123]
        );
    }

    #[test]
    fn test_negation_detection() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let positive = enc.encode_text("the cat sat quietly");
        let negative = enc.encode_text("the cat never sat quietly");
        // Negation feature is dim 40 + 17
        assert!(
            negative.data[57] > positive.data[57],
            "Negation: negative={:.1} should exceed positive={:.1}",
            negative.data[57],
            positive.data[57]
        );
    }

    #[test]
    fn test_amplification_factors_applied() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let text = "the recursive nature of self-referential systems creates emergent properties";
        let tensor = enc.encode_text(text);

        // Compute raw features for comparison
        let words = Tokeniser::split(text);
        let ids = enc.tokeniser.tokenise_readonly(text);
        let raw_ngrams = Encoder::ngram_profile(&words);
        let raw_syntactic = Encoder::syntactic_features(&words, text);
        let raw_tokens = Encoder::token_signal(&ids);
        let raw_scalars = Encoder::complexity_scalars(&words, text);

        // G1 (dims 0..40) should be amplified by 3.0
        for i in 0..NGRAM_DIMS {
            let expected = raw_ngrams[i] * Encoder::G1_AMP;
            assert!(
                (tensor.data[i] - expected).abs() < 1e-6,
                "G1 dim {}: expected {:.6}, got {:.6}",
                i, expected, tensor.data[i]
            );
        }

        // G2 (dims 40..72) should be amplified by 3.0
        for i in 0..SYNTACTIC_DIMS {
            let expected = raw_syntactic[i] * Encoder::G2_AMP;
            assert!(
                (tensor.data[NGRAM_DIMS + i] - expected).abs() < 1e-6,
                "G2 dim {}: expected {:.6}, got {:.6}",
                i, expected, tensor.data[NGRAM_DIMS + i]
            );
        }

        // G3 (dims 72..120) should be amplified by 1.0 (unchanged)
        for i in 0..TOKEN_DIMS {
            let expected = raw_tokens[i] * Encoder::G3_AMP;
            assert!(
                (tensor.data[NGRAM_DIMS + SYNTACTIC_DIMS + i] - expected).abs() < 1e-6,
                "G3 dim {}: expected {:.6}, got {:.6}",
                i, expected, tensor.data[NGRAM_DIMS + SYNTACTIC_DIMS + i]
            );
        }

        // G4 (dims 120..128) should be amplified by 2.0
        for i in 0..SCALAR_DIMS {
            let expected = raw_scalars[i] * Encoder::G4_AMP;
            assert!(
                (tensor.data[NGRAM_DIMS + SYNTACTIC_DIMS + TOKEN_DIMS + i] - expected).abs() < 1e-6,
                "G4 dim {}: expected {:.6}, got {:.6}",
                i, expected, tensor.data[NGRAM_DIMS + SYNTACTIC_DIMS + TOKEN_DIMS + i]
            );
        }
    }

    #[test]
    fn test_amplification_widens_cosine_separation() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let simple = enc.encode_text("the dog runs fast");
        let complex = enc.encode_text(
            "the recursive nature of self-referential systems creates emergent properties that resist reduction",
        );

        let cosine_sim = simple.cosine_similarity(&complex);
        // With amplification, cosine similarity between simple and complex should be
        // meaningfully below 1.0 — verifying directional separation exists
        assert!(
            cosine_sim < 0.99,
            "Simple/complex cosine similarity should be < 0.99 with amplification, got {:.4}",
            cosine_sim
        );
    }
}
