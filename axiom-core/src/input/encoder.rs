//! Structural Encoder V4 — converts text to fixed 128-dimension tensors
//! using four concatenated feature groups: character n-gram profile,
//! syntactic proxy features, position-weighted token signal, and
//! complexity scalars.
//!
//! **No normalisation on the final vector.** Magnitude encodes sentence
//! complexity: simple sentences differ from complex in n-gram distributions,
//! syntactic structure, and vocabulary richness. This is the primary signal
//! for discriminative routing.
//!
//! Phase 12 redistributes dimensions to 24+40+48+16 for richer syntactic
//! features and complexity scalars while keeping total at 128.

use crate::input::tokeniser::Tokeniser;
use crate::Tensor;
use std::collections::HashSet;

/// Fixed output dimension: 24 + 40 + 48 + 16 = 128.
pub const OUTPUT_DIM: usize = 128;

/// Character n-gram profile dimensions (Group 1).
const NGRAM_DIMS: usize = 24;

/// Syntactic proxy feature dimensions (Group 2).
const SYNTACTIC_DIMS: usize = 40;

/// Position-weighted token signal dimensions (Group 3).
const TOKEN_DIMS: usize = 48;

/// Complexity scalar dimensions (Group 4).
const SCALAR_DIMS: usize = 16;

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

/// Subordinating conjunctions for depth proxy and presence feature.
const SUBORDINATING: &[&str] = &[
    "because", "although", "while", "since", "if", "when", "unless", "after", "before", "until",
];

/// Relative clause markers for nested clause depth proxy.
const RELATIVE_CLAUSE_MARKERS: &[&str] = &["that", "which", "who", "whom"];

/// Pronouns for density feature.
const PRONOUNS: &[&str] = &[
    "he", "she", "it", "they", "we", "i", "you", "him", "her", "them", "us", "me",
];

/// Structural Encoder V4 producing fixed 128-dimension tensors from text.
///
/// Four feature groups concatenated:
/// - **Group 1** (dims 0–23): Character n-gram profile — bigram/trigram hash buckets,
///   normalised by total n-gram count. 24 buckets.
/// - **Group 2** (dims 24–63): Syntactic proxy features — word length octiles, variance,
///   function word density, punctuation, capitalisation, binary markers, structural features,
///   nested clause depth proxies, pronoun density, clause boundary density, verb density,
///   syllable proxy, hapax ratio, sentence complexity score.
/// - **Group 3** (dims 64–111): Position-weighted token signal — token IDs folded into
///   48 buckets by `id % 48`, weighted by `1/(1+position)`. NOT normalised.
/// - **Group 4** (dims 112–127): Complexity scalars — token count, TTR, mean length,
///   dependency depth, mean clause length, lexical density, bigram diversity, sentence rhythm,
///   redundant repeat of first 4, vocabulary richness, length variation, rare word ratio,
///   clause count.
///
/// No normalisation of final vector — magnitude carries complexity information.
pub struct Encoder {
    /// Output dimension (always 128).
    pub output_dim: usize,
    /// Reference to tokeniser for end-to-end encoding.
    pub tokeniser: Tokeniser,
}

impl Encoder {
    /// Create a new V4 structural encoder.
    ///
    /// The `_output_dim` parameter is accepted for API compatibility but the
    /// encoder always produces 128-dimensional vectors (24+40+48+16).
    pub fn new(_output_dim: usize, tokeniser: Tokeniser) -> Self {
        Self {
            output_dim: OUTPUT_DIM,
            tokeniser,
        }
    }

    /// Compute character n-gram profile (24 dimensions).
    ///
    /// Extracts all character bigrams and trigrams from every word in the
    /// sentence. Each n-gram is hashed to one of 24 buckets using a
    /// multiply-add hash. Counts are normalised by total n-gram count.
    fn ngram_profile(words: &[String]) -> [f32; NGRAM_DIMS] {
        let mut buckets = [0u32; NGRAM_DIMS];
        let mut total_ngrams = 0u32;

        for word in words {
            let chars: Vec<char> = word.chars().collect();
            for window in chars.windows(2) {
                let bucket = Self::ngram_hash(window) % NGRAM_DIMS;
                buckets[bucket] += 1;
                total_ngrams += 1;
            }
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

    /// Compute syntactic proxy features (40 dimensions).
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
    /// - Dim 31: Relative clause marker count (that/which/who/whom) normalised 0–1 against 5.
    /// - Dim 32: Subordinating conjunction count normalised 0–1 against 5.
    /// - Dim 33: Pronoun density (pronoun count / total tokens).
    /// - Dim 34: Clause boundary density ((comma+semicolon+colon) / total tokens).
    /// - Dim 35: Verb density proxy (tokens ending in s/ed/ing / total tokens).
    /// - Dim 36: Average syllable proxy (vowel groups per token, normalised 0–1 against 5).
    /// - Dim 37: Hapax ratio (tokens appearing exactly once / total tokens).
    /// - Dim 38: Sentence complexity score (weighted composite, normalised 0–1 against 20).
    /// - Dim 39: Zero padding.
    fn syntactic_features(words: &[String], raw_text: &str) -> [f32; SYNTACTIC_DIMS] {
        let mut features = [0.0f32; SYNTACTIC_DIMS];
        let n = words.len();
        if n == 0 {
            return features;
        }

        let word_lengths: Vec<f32> = words.iter().map(|w| w.len() as f32).collect();

        // Dims 0–7: Average word length in each of 8 sentence octiles
        let octile_size = (n + 7) / 8;
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
            let total_syllables = Self::count_syllables(words);
            features[30] = (total_syllables as f32 / n as f32 / 5.0).min(1.0);
        }

        // --- Phase 12 new features (dims 31–38) ---

        // Dim 31: Relative clause marker count normalised 0–1 against max 5
        let rel_clause_count = words
            .iter()
            .filter(|w| RELATIVE_CLAUSE_MARKERS.contains(&w.as_str()))
            .count();
        features[31] = (rel_clause_count as f32 / 5.0).min(1.0);

        // Dim 32: Subordinating conjunction count normalised 0–1 against max 5
        let subord_count = words
            .iter()
            .filter(|w| SUBORDINATING.contains(&w.as_str()))
            .count();
        features[32] = (subord_count as f32 / 5.0).min(1.0);

        // Dim 33: Pronoun density (pronoun count / total tokens)
        let pronoun_count = words
            .iter()
            .filter(|w| PRONOUNS.contains(&w.as_str()))
            .count();
        features[33] = (pronoun_count as f32 / n as f32).min(1.0);

        // Dim 34: Clause boundary density ((comma + semicolon + colon) / total tokens)
        let semicolons = raw_text.chars().filter(|&c| c == ';').count();
        let colons = raw_text.chars().filter(|&c| c == ':').count();
        features[34] = ((comma_count + semicolons + colons) as f32 / n as f32).min(1.0);

        // Dim 35: Verb density proxy (tokens ending in s/ed/ing / total tokens)
        let verb_endings = words
            .iter()
            .filter(|w| {
                w.len() > 2
                    && (w.ends_with("ing")
                        || w.ends_with("ed")
                        || w.ends_with('s'))
            })
            .count();
        features[35] = (verb_endings as f32 / n as f32).min(1.0);

        // Dim 36: Average syllable proxy (vowel groups per token, normalised 0–1 against 5)
        if n > 0 {
            let total_syllables = Self::count_syllables(words);
            features[36] = (total_syllables as f32 / n as f32 / 5.0).min(1.0);
        }

        // Dim 37: Hapax ratio (tokens appearing exactly once / total tokens)
        let mut word_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for w in words {
            *word_counts.entry(w.as_str()).or_insert(0) += 1;
        }
        let hapax_count = word_counts.values().filter(|&&c| c == 1).count();
        features[37] = (hapax_count as f32 / n as f32).min(1.0);

        // Dim 38: Sentence complexity score — weighted composite normalised 0–1 against 20
        let rare_words = words.iter().filter(|w| w.len() > 8).count();
        let complexity_raw = rel_clause_count as f32 * 2.0
            + subord_count as f32 * 2.0
            + rare_words as f32
            + punct_count as f32;
        features[38] = (complexity_raw / 20.0).min(1.0);

        // Dim 39: Zero padding
        features[39] = 0.0;

        features
    }

    /// Count total syllables (vowel groups) across all words.
    fn count_syllables(words: &[String]) -> usize {
        let mut total = 0usize;
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
            total += syl.max(1);
        }
        total
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

    /// Compute complexity scalars (16 dimensions).
    ///
    /// - Dim 0: Token count normalised 0–1 against max 40.
    /// - Dim 1: Type-token ratio (unique words / total words).
    /// - Dim 2: Mean token length normalised 0–1 against max 15.
    /// - Dim 3: Dependency depth proxy — (subordinating conjunction count +
    ///   comma count) normalised 0–1 against max 10.
    /// - Dim 4: Mean clause length proxy — total tokens / (comma count + 1),
    ///   normalised 0–1 against max 30.
    /// - Dim 5: Lexical density — content words (not in function word list) / total.
    /// - Dim 6: Bigram diversity — unique adjacent word pairs / total pairs.
    /// - Dim 7: Sentence rhythm — std dev of token lengths normalised 0–1 against max 5.
    /// - Dims 8–11: Redundant repeat of dims 0–3.
    /// - Dim 12: Vocabulary richness — log(unique) / log(total).
    /// - Dim 13: Length variation ratio — max word length / mean word length.
    /// - Dim 14: Rare word ratio — words with >=8 chars / total words.
    /// - Dim 15: Clause count proxy — (commas + subordinating + semicolons) normalised.
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

        // Dim 4: Mean clause length proxy — total tokens / (comma_count + 1)
        let mean_clause_len = (n as f32 / (comma_count as f32 + 1.0) / 30.0).min(1.0);

        // Dim 5: Lexical density — content words / total
        let content_count = words
            .iter()
            .filter(|w| !FUNCTION_WORDS.contains(&w.as_str()))
            .count();
        let lexical_density = content_count as f32 / n as f32;

        // Dim 6: Bigram diversity — unique adjacent word pairs / total pairs
        let bigram_diversity = if n > 1 {
            let mut bigrams = HashSet::new();
            for pair in words.windows(2) {
                bigrams.insert((&pair[0], &pair[1]));
            }
            bigrams.len() as f32 / (n - 1) as f32
        } else {
            1.0
        };

        // Dim 7: Sentence rhythm — std dev of token lengths, normalised 0–1 against max 5
        let mean_wl = total_char_len as f32 / n as f32;
        let len_variance: f32 = words
            .iter()
            .map(|w| (w.len() as f32 - mean_wl).powi(2))
            .sum::<f32>()
            / n as f32;
        let sentence_rhythm = (len_variance.sqrt() / 5.0).min(1.0);

        // Dim 12: Vocabulary richness — log(unique) / log(total)
        let vocab_richness = if n > 1 {
            ((unique.len() as f32).ln() / (n as f32).ln()).min(1.0)
        } else {
            1.0
        };

        // Dim 13: Length variation ratio — max word length / mean word length, normalised
        let max_wl = words.iter().map(|w| w.len()).max().unwrap_or(1) as f32;
        let length_variation = if mean_wl > 0.0 {
            (max_wl / mean_wl / 5.0).min(1.0)
        } else {
            0.0
        };

        // Dim 14: Rare word ratio — words with >=8 chars / total
        let rare_count = words.iter().filter(|w| w.len() >= 8).count();
        let rare_ratio = rare_count as f32 / n as f32;

        // Dim 15: Clause count proxy — (commas + subordinating + semicolons) normalised
        let semicolons = raw_text.chars().filter(|&c| c == ';').count();
        let clause_proxy =
            ((comma_count + subord_count + semicolons) as f32 / MAX_DEP_DEPTH).min(1.0);

        [
            token_count_norm,  // 0
            ttr,               // 1
            mean_token_len,    // 2
            dep_depth,         // 3
            mean_clause_len,   // 4  (new)
            lexical_density,   // 5  (new)
            bigram_diversity,  // 6  (new)
            sentence_rhythm,   // 7  (new)
            token_count_norm,  // 8  (repeat of 0)
            ttr,               // 9  (repeat of 1)
            mean_token_len,    // 10 (repeat of 2)
            dep_depth,         // 11 (repeat of 3)
            vocab_richness,    // 12
            length_variation,  // 13
            rare_ratio,        // 14
            clause_proxy,      // 15
        ]
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

        // Group 1: Character n-gram profile (dims 0–23) — amplified
        let ngrams = Self::ngram_profile(&words);

        // Group 2: Syntactic proxy features (dims 24–63) — amplified
        let syntactic = Self::syntactic_features(&words, text);

        // Group 3: Position-weighted token signal (dims 64–111) — unchanged
        let tokens = Self::token_signal(token_ids);

        // Group 4: Complexity scalars (dims 112–127) — amplified
        let scalars = Self::complexity_scalars(&words, text);

        // Concatenate with amplification: 24 + 40 + 48 + 16 = 128
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
            "syl_cnt", "rel_cls", "sub_cnt", "pron_dn", "cls_bnd", "verb_dn",
            "syl_prx", "hapax_r", "cmplx_s", "pad",
        ];
        for (i, (v, label)) in syntactic.iter().zip(labels.iter()).enumerate() {
            print!("      {:>7}: {:.4}", label, v);
            if (i + 1) % 4 == 0 {
                println!();
            }
        }
        if SYNTACTIC_DIMS % 4 != 0 {
            println!();
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
        let scalar_labels = [
            "tok_cnt", "TTR", "mean_ln", "dep_dep",
            "cls_len", "lex_den", "bi_div", "rhythm",
            "tok_r", "TTR_r", "mln_r", "dep_r",
            "voc_rch", "len_var", "rare_r", "clause",
        ];
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
        let simple_ngrams = &simple.data[0..NGRAM_DIMS];
        let complex_ngrams = &complex.data[0..NGRAM_DIMS];
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
        // Sentence length feature: G2 base (24) + dim 12 = 36
        let simple_sent_len = simple.data[24 + 12];
        let complex_sent_len = complex.data[24 + 12];
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
        // Token signal: G3 base (64) to G3 end (112)
        assert_ne!(
            t1.data[64..112],
            t2.data[64..112],
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
        // Token count scalar: G4 base (112) + dim 0 = 112
        assert!(
            complex.data[112] > simple.data[112],
            "Token count: complex={:.4} should exceed simple={:.4}",
            complex.data[112],
            simple.data[112]
        );
    }

    #[test]
    fn test_empty_text() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let t = enc.encode_text("");
        assert_eq!(t.shape, vec![128]);
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
        // Subordinating conjunction presence: G2 base (24) + dim 18 = 42
        assert!(
            with_subord.data[42] > no_subord.data[42],
            "Subordinating: with={:.1} should exceed without={:.1}",
            with_subord.data[42],
            no_subord.data[42]
        );
    }

    #[test]
    fn test_punctuation_feature() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let no_punct = enc.encode_text("hello world");
        let with_punct = enc.encode_text("hello, world! how are you?");
        // Punctuation feature: G2 base (24) + dim 13 = 37
        assert!(
            with_punct.data[37] > no_punct.data[37],
            "Punctuation: with={:.4} should exceed without={:.4}",
            with_punct.data[37],
            no_punct.data[37]
        );
    }

    #[test]
    fn test_function_word_density() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let low_fn = enc.encode_text("recursive systems create emergent properties");
        let high_fn = enc.encode_text("the cat is on the mat in the room");
        // Function word density: G2 base (24) + dim 10 = 34
        assert!(
            high_fn.data[34] > low_fn.data[34],
            "Function word density: high={:.4} should exceed low={:.4}",
            high_fn.data[34],
            low_fn.data[34]
        );
    }

    #[test]
    fn test_dependency_depth_proxy() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let simple = enc.encode_text("the cat sat");
        let complex =
            enc.encode_text("because the cat, which was tired, sat down, it slept");
        // Dependency depth proxy: G4 base (112) + dim 3 = 115
        assert!(
            complex.data[115] > simple.data[115],
            "Dep depth: complex={:.4} should exceed simple={:.4}",
            complex.data[115],
            simple.data[115]
        );
    }

    #[test]
    fn test_negation_detection() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let positive = enc.encode_text("the cat sat quietly");
        let negative = enc.encode_text("the cat never sat quietly");
        // Negation feature: G2 base (24) + dim 17 = 41
        assert!(
            negative.data[41] > positive.data[41],
            "Negation: negative={:.1} should exceed positive={:.1}",
            negative.data[41],
            positive.data[41]
        );
    }

    #[test]
    fn test_amplification_factors_applied() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let text = "the recursive nature of self-referential systems creates emergent properties";
        let tensor = enc.encode_text(text);

        let words = Tokeniser::split(text);
        let ids = enc.tokeniser.tokenise_readonly(text);
        let raw_ngrams = Encoder::ngram_profile(&words);
        let raw_syntactic = Encoder::syntactic_features(&words, text);
        let raw_tokens = Encoder::token_signal(&ids);
        let raw_scalars = Encoder::complexity_scalars(&words, text);

        // G1 (dims 0..24) should be amplified by 3.0
        for i in 0..NGRAM_DIMS {
            let expected = raw_ngrams[i] * Encoder::G1_AMP;
            assert!(
                (tensor.data[i] - expected).abs() < 1e-6,
                "G1 dim {}: expected {:.6}, got {:.6}",
                i, expected, tensor.data[i]
            );
        }

        // G2 (dims 24..64) should be amplified by 3.0
        for i in 0..SYNTACTIC_DIMS {
            let expected = raw_syntactic[i] * Encoder::G2_AMP;
            assert!(
                (tensor.data[NGRAM_DIMS + i] - expected).abs() < 1e-6,
                "G2 dim {}: expected {:.6}, got {:.6}",
                i, expected, tensor.data[NGRAM_DIMS + i]
            );
        }

        // G3 (dims 64..112) should be amplified by 1.0 (unchanged)
        for i in 0..TOKEN_DIMS {
            let expected = raw_tokens[i] * Encoder::G3_AMP;
            assert!(
                (tensor.data[NGRAM_DIMS + SYNTACTIC_DIMS + i] - expected).abs() < 1e-6,
                "G3 dim {}: expected {:.6}, got {:.6}",
                i, expected, tensor.data[NGRAM_DIMS + SYNTACTIC_DIMS + i]
            );
        }

        // G4 (dims 112..128) should be amplified by 2.0
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
        assert!(
            cosine_sim < 0.99,
            "Simple/complex cosine similarity should be < 0.99 with amplification, got {:.4}",
            cosine_sim
        );
    }

    // --- Phase 12 new feature tests ---

    #[test]
    fn test_nested_clause_depth_relative() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let no_rel = enc.encode_text("the cat sat on the mat");
        let with_rel = enc.encode_text("the cat that the dog chased sat on the mat");
        // Relative clause marker count: G2 base (24) + dim 31 = 55
        assert!(
            with_rel.data[55] > no_rel.data[55],
            "Relative clause markers: with={:.4} should exceed without={:.4}",
            with_rel.data[55],
            no_rel.data[55]
        );
    }

    #[test]
    fn test_nested_clause_depth_subordinating() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let no_sub = enc.encode_text("the cat sat on the mat");
        let with_sub = enc.encode_text("because the cat sat after the dog left until it rained");
        // Subordinating conjunction count: G2 base (24) + dim 32 = 56
        assert!(
            with_sub.data[56] > no_sub.data[56],
            "Subordinating count: with={:.4} should exceed without={:.4}",
            with_sub.data[56],
            no_sub.data[56]
        );
    }

    #[test]
    fn test_pronoun_density() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let no_pron = enc.encode_text("the cat sat on the mat");
        let with_pron = enc.encode_text("he told her that they would help us");
        // Pronoun density: G2 base (24) + dim 33 = 57
        assert!(
            with_pron.data[57] > no_pron.data[57],
            "Pronoun density: with={:.4} should exceed without={:.4}",
            with_pron.data[57],
            no_pron.data[57]
        );
    }

    #[test]
    fn test_clause_boundary_density() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let no_bound = enc.encode_text("the cat sat on the mat");
        let with_bound = enc.encode_text("the cat, which was tired; sat down: sleeping");
        // Clause boundary density: G2 base (24) + dim 34 = 58
        assert!(
            with_bound.data[58] > no_bound.data[58],
            "Clause boundary density: with={:.4} should exceed without={:.4}",
            with_bound.data[58],
            no_bound.data[58]
        );
    }

    #[test]
    fn test_hapax_ratio() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let repeated = enc.encode_text("the the the the the the");
        let unique_words = enc.encode_text("quantum mechanics describes fundamental particle interactions");
        // Hapax ratio: G2 base (24) + dim 37 = 61
        assert!(
            unique_words.data[61] > repeated.data[61],
            "Hapax ratio: unique={:.4} should exceed repeated={:.4}",
            unique_words.data[61],
            repeated.data[61]
        );
    }

    #[test]
    fn test_sentence_complexity_score() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let simple = enc.encode_text("the cat sat");
        let complex = enc.encode_text(
            "because the philosophical implications, which scholars that studied metaphysics debated, remained unresolved"
        );
        // Sentence complexity score: G2 base (24) + dim 38 = 62
        assert!(
            complex.data[62] > simple.data[62],
            "Complexity score: complex={:.4} should exceed simple={:.4}",
            complex.data[62],
            simple.data[62]
        );
    }

    #[test]
    fn test_verb_density_proxy() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let no_verbs = enc.encode_text("the big red ball on the table");
        let with_verbs = enc.encode_text("running jumping climbing creates exhausting challenges");
        // Verb density: G2 base (24) + dim 35 = 59
        assert!(
            with_verbs.data[59] > no_verbs.data[59],
            "Verb density: with={:.4} should exceed without={:.4}",
            with_verbs.data[59],
            no_verbs.data[59]
        );
    }

    #[test]
    fn test_lexical_density_scalar() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        // High function word content
        let function_heavy = enc.encode_text("the cat is on the mat in the room");
        // Low function word content (high lexical density)
        let content_heavy = enc.encode_text("quantum mechanics describes fundamental particle interactions");
        // Lexical density: G4 base (112) + dim 5 = 117
        assert!(
            content_heavy.data[117] > function_heavy.data[117],
            "Lexical density: content_heavy={:.4} should exceed function_heavy={:.4}",
            content_heavy.data[117],
            function_heavy.data[117]
        );
    }

    #[test]
    fn test_bigram_diversity_scalar() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let repeated = enc.encode_text("the cat the cat the cat the cat");
        let diverse = enc.encode_text("quantum mechanics describes fundamental particle interactions");
        // Bigram diversity: G4 base (112) + dim 6 = 118
        assert!(
            diverse.data[118] > repeated.data[118],
            "Bigram diversity: diverse={:.4} should exceed repeated={:.4}",
            diverse.data[118],
            repeated.data[118]
        );
    }

    #[test]
    fn test_weight_serialisation_roundtrip_scalars() {
        // Verify G4 redundant repeat: dims 8-11 should equal dims 0-3
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let t = enc.encode_text("the cat sat because it was tired and confused");
        // G4 base = 112. Dim 0 == Dim 8, Dim 1 == Dim 9, etc.
        for i in 0..4 {
            assert!(
                (t.data[112 + i] - t.data[112 + 8 + i]).abs() < 1e-6,
                "G4 redundant repeat: dim {} ({:.4}) should equal dim {} ({:.4})",
                i, t.data[112 + i], 8 + i, t.data[112 + 8 + i]
            );
        }
    }

    #[test]
    fn test_diagnostic_sentences() {
        let tok = Tokeniser::new(1024);
        let mut enc = Encoder::new(128, tok);

        let sentences = [
            ("The cat sat.", true),
            ("I like dogs.", true),
            ("She runs fast.", true),
            ("The recursive nature of self-referential systems creates emergent properties that resist reduction to simpler components.", false),
            ("Although the philosophical implications were debated, the committee which reviewed the findings concluded that further investigation was warranted.", false),
            ("Because the implementation required careful consideration of edge cases, the developers who maintained the codebase decided to refactor the architecture.", false),
        ];

        let mut simple_vecs = Vec::new();
        let mut complex_vecs = Vec::new();

        println!("\n=== Phase 12 Stage 1 Diagnostic (6 sentences) ===");
        for (text, is_simple) in &sentences {
            let t = enc.encode_text(text);
            println!("  [{}] norm={:.4}  \"{}\"",
                if *is_simple { "S" } else { "C" }, t.norm(), text);
            if *is_simple {
                simple_vecs.push(t);
            } else {
                complex_vecs.push(t);
            }
        }

        // Compute means
        let dim = OUTPUT_DIM;
        let mut simple_mean = vec![0.0f32; dim];
        for v in &simple_vecs {
            for (i, &val) in v.data.iter().enumerate() {
                simple_mean[i] += val;
            }
        }
        for v in simple_mean.iter_mut() {
            *v /= simple_vecs.len() as f32;
        }

        let mut complex_mean = vec![0.0f32; dim];
        for v in &complex_vecs {
            for (i, &val) in v.data.iter().enumerate() {
                complex_mean[i] += val;
            }
        }
        for v in complex_mean.iter_mut() {
            *v /= complex_vecs.len() as f32;
        }

        let simple_t = Tensor::from_vec(simple_mean);
        let complex_t = Tensor::from_vec(complex_mean);
        let cos = simple_t.cosine_similarity(&complex_t);
        let simple_norm = simple_t.norm();
        let complex_norm = complex_t.norm();

        println!("  Simple mean norm:  {:.4}", simple_norm);
        println!("  Complex mean norm: {:.4}", complex_norm);
        println!("  cos(simple_mean, complex_mean) = {:.4}", cos);
        println!("  Norm gap (complex - simple):     {:.4}", complex_norm - simple_norm);

        // Complex sentences should have higher norms due to more features firing
        assert!(
            complex_norm > simple_norm,
            "Complex norm ({:.4}) should exceed simple norm ({:.4})",
            complex_norm, simple_norm
        );
    }

    #[test]
    fn test_calibration_corpus_separation() {
        let tok = Tokeniser::new(1024);
        let mut enc = Encoder::new(128, tok);

        let simple = vec![
            "a red ball",
            "open the door",
            "the moon is round",
            "cats like warm milk",
            "she ate lunch today",
            "rain falls from clouds",
            "we walked home slowly",
        ];
        let complex = vec![
            "the observer effect in quantum mechanics implies that measurement, by its very nature, fundamentally alters the state of the observed system",
            "autopoietic systems, while maintaining strict organisational closure, nonetheless remain thermodynamically open to continuous energy and matter exchange",
            "the no-free-lunch theorem establishes that no single optimisation algorithm can dominate across all possible problem classes without exception",
            "dialectical materialism posits that internal contradictions within socioeconomic structures, rather than external forces alone, drive historical transformation",
            "renormalisation group methods, when applied to systems near critical phase transitions, reveal universal scale-invariant behaviour across many physical phenomena",
            "the frame problem in artificial intelligence highlights the fundamental difficulty of formally representing the implicit non-effects of actions within logical frameworks",
        ];

        let dim = OUTPUT_DIM;

        let simple_vecs: Vec<Tensor> = simple.iter().map(|s| enc.encode_text(s)).collect();
        let complex_vecs: Vec<Tensor> = complex.iter().map(|s| enc.encode_text(s)).collect();

        let mut simple_mean = vec![0.0f32; dim];
        for v in &simple_vecs {
            for (i, &val) in v.data.iter().enumerate() { simple_mean[i] += val; }
        }
        for v in simple_mean.iter_mut() { *v /= simple_vecs.len() as f32; }

        let mut complex_mean = vec![0.0f32; dim];
        for v in &complex_vecs {
            for (i, &val) in v.data.iter().enumerate() { complex_mean[i] += val; }
        }
        for v in complex_mean.iter_mut() { *v /= complex_vecs.len() as f32; }

        let simple_t = Tensor::from_vec(simple_mean);
        let complex_t = Tensor::from_vec(complex_mean);
        let cos = simple_t.cosine_similarity(&complex_t);

        println!("\n=== Calibration Corpus Diagnostics ===");
        println!("Simple (7 sentences):");
        for (s, v) in simple.iter().zip(simple_vecs.iter()) {
            println!("  norm={:.4}  \"{}\"", v.norm(), s);
        }
        println!("Complex (6 sentences):");
        for (s, v) in complex.iter().zip(complex_vecs.iter()) {
            println!("  norm={:.4}  \"{}\"", v.norm(), s);
        }
        println!("Simple mean norm:  {:.4}", simple_t.norm());
        println!("Complex mean norm: {:.4}", complex_t.norm());
        println!("Norm gap:          {:.4}", complex_t.norm() - simple_t.norm());
        println!("cos(simple_mean, complex_mean) = {:.4}", cos);

        // Mean pairwise cos
        let mut sum_cos = 0.0f32;
        let mut count = 0;
        for sv in &simple_vecs {
            for cv in &complex_vecs {
                sum_cos += sv.cosine_similarity(cv);
                count += 1;
            }
        }
        println!("Mean pairwise cos(simple_i, complex_j) = {:.4}", sum_cos / count as f32);

        assert!(
            complex_t.norm() > simple_t.norm(),
            "Complex mean norm should exceed simple mean norm"
        );
    }
}
