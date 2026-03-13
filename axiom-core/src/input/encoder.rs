//! Structural Encoder V5 — converts text to fixed 128-dimension tensors
//! using five concatenated feature groups: character n-gram profile,
//! syntactic proxy features, position-weighted token signal, complexity
//! scalars, and structural syntax features.
//!
//! **No normalisation on the final vector.** Magnitude encodes sentence
//! complexity: simple sentences differ from complex in n-gram distributions,
//! syntactic structure, and vocabulary richness. This is the primary signal
//! for discriminative routing.
//!
//! Phase 14 redistributes dimensions to 26+36+39+15+12=128 and adds
//! vocabulary-independent structural syntax features (G5) targeting the
//! Phase 13 adversarial failure modes.

use crate::input::tokeniser::Tokeniser;
use crate::Tensor;
use std::collections::HashSet;

/// Fixed output dimension: 26 + 36 + 39 + 15 + 12 = 128.
pub const OUTPUT_DIM: usize = 128;

/// Character n-gram profile dimensions (Group 1).
const NGRAM_DIMS: usize = 26;

/// Syntactic proxy feature dimensions (Group 2).
const SYNTACTIC_DIMS: usize = 36;

/// Position-weighted token signal dimensions (Group 3).
const TOKEN_DIMS: usize = 39;

/// Complexity scalar dimensions (Group 4).
const SCALAR_DIMS: usize = 15;

/// Structural syntax feature dimensions (Group 5).
const STRUCTURAL_DIMS: usize = 12;

/// Start offset for G5 structural syntax features in the encoder output.
pub const G5_OFFSET: usize = 26 + 36 + 39 + 15; // = 116

/// Number of G5 structural syntax dimensions.
pub const G5_DIM: usize = 12;

/// Maximum sentence length (words) for normalisation.
const MAX_SENTENCE_LEN: f32 = 40.0;

/// Maximum token length (chars) for normalisation.
const MAX_TOKEN_LEN: f32 = 15.0;

/// Maximum dependency depth proxy for normalisation.
const MAX_DEP_DEPTH: f32 = 10.0;

/// Function words for density calculation (Group 2).
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

/// Extended subordinating conjunctions and relative clause markers for G5
/// dependency depth computation (vocabulary-independent structural markers).
const G5_SUBORDINATORS: &[&str] = &[
    "that", "which", "who", "whom", "because", "although", "while", "since", "if", "when",
    "unless", "after", "before", "until", "whether", "though", "whereas", "whereby",
];

/// Subject pronouns for G5 second-main-verb detection.
const G5_SUBJECT_PRONOUNS: &[&str] = &["he", "she", "it", "they", "we", "i", "you"];

/// Extended function word set for G5 position entropy computation.
/// Includes determiners, auxiliaries, prepositions, conjunctions, and
/// common adverbs — all structure words that distribute differently in
/// simple vs complex sentences.
const G5_FUNCTION_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "or", "and", "but",
    "nor", "so", "yet", "not", "no", "it", "its", "this", "that", "these", "those", "there",
    "here", "very", "just", "also", "only", "even", "both", "either", "each", "all", "any",
    "some", "few", "more", "most",
];

/// Prepositions for G5 prepositional phrase depth tracking.
const G5_PREPOSITIONS: &[&str] = &[
    "of", "in", "by", "with", "for", "through", "among", "between", "within",
];

/// Structural Encoder V5 producing fixed 128-dimension tensors from text.
///
/// Five feature groups concatenated:
/// - **Group 1** (dims 0–25): Character n-gram profile — bigram/trigram hash buckets,
///   normalised by total n-gram count. 26 buckets.
/// - **Group 2** (dims 26–61): Syntactic proxy features — word length octiles, variance,
///   function word density, punctuation, capitalisation, binary markers, structural features,
///   nested clause depth proxies, pronoun density, clause boundary density, mean word length,
///   rare word density, character diversity. 36 dims.
/// - **Group 3** (dims 62–100): Position-weighted token signal — token IDs folded into
///   39 buckets by `id % 39`, weighted by `1/(1+position)`. NOT normalised.
/// - **Group 4** (dims 101–115): Complexity scalars — token count, TTR, mean length,
///   dependency depth, mean clause length, lexical density, bigram diversity, sentence rhythm,
///   vocabulary richness, length variation, rare word ratio, clause count. 15 dims.
/// - **Group 5** (dims 116–127): Structural syntax features — dependency depth proxy (4d),
///   constituent length variance (2d), function word position entropy (5d), 1 pad. 12 dims.
///
/// No normalisation of final vector — magnitude carries complexity information.
pub struct Encoder {
    /// Output dimension (always 128).
    pub output_dim: usize,
    /// Reference to tokeniser for end-to-end encoding.
    pub tokeniser: Tokeniser,
    /// When true, G5 structural features are divided by sqrt(token_count)
    /// to normalize for sentence length. Prevents long simple sentences
    /// from having inflated G5 norms.
    pub g5_length_normalize: bool,
}

impl Encoder {
    /// Create a new V5 structural encoder.
    ///
    /// The `_output_dim` parameter is accepted for API compatibility but the
    /// encoder always produces 128-dimensional vectors (26+36+39+15+12).
    pub fn new(_output_dim: usize, tokeniser: Tokeniser) -> Self {
        Self {
            output_dim: OUTPUT_DIM,
            tokeniser,
            g5_length_normalize: false,
        }
    }

    /// Compute character n-gram profile (26 dimensions).
    ///
    /// Extracts all character bigrams and trigrams from every word in the
    /// sentence. Each n-gram is hashed to one of 26 buckets using a
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

    /// Compute syntactic proxy features (36 dimensions).
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
    /// - Dim 15: Question word presence (binary).
    /// - Dim 16: Negation presence (binary).
    /// - Dim 17: Subordinating conjunction presence (binary).
    /// - Dim 18: Comma density (commas / word count).
    /// - Dim 19: Semicolon/colon presence (binary).
    /// - Dim 20: Parenthetical/bracket presence (binary).
    /// - Dim 21: Hyphenated word ratio.
    /// - Dim 22: Short word ratio (<=3 chars).
    /// - Dim 23: Long word ratio (>=8 chars).
    /// - Dim 24: Word length range (max - min) normalised.
    /// - Dim 25: Sentence-initial capital (binary).
    /// - Dim 26: Vowel ratio in words.
    /// - Dim 27: Consonant cluster density proxy.
    /// - Dim 28: Mean syllable count proxy (vowel groups per word).
    /// - Dim 29: Relative clause marker count normalised 0–1 against 5.
    /// - Dim 30: Subordinating conjunction count normalised 0–1 against 5.
    /// - Dim 31: Pronoun density (pronoun count / total tokens).
    /// - Dim 32: Clause boundary density ((comma+semicolon+colon) / total tokens).
    /// - Dim 33: Mean word length (chars) — mean(len(word)) / 12.0, capped at 1.0.
    /// - Dim 34: Rare word density — count(words > 8 chars) / total_words.
    /// - Dim 35: Character diversity — unique_chars / total_chars.
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

        // Dim 15: Question word presence (binary)
        features[15] = if words
            .iter()
            .any(|w| QUESTION_WORDS.contains(&w.as_str()))
        {
            1.0
        } else {
            0.0
        };

        // Dim 16: Negation presence (binary)
        features[16] = if words.iter().any(|w| NEGATION_WORDS.contains(&w.as_str())) {
            1.0
        } else {
            0.0
        };

        // Dim 17: Subordinating conjunction presence (binary)
        features[17] = if words.iter().any(|w| SUBORDINATING.contains(&w.as_str())) {
            1.0
        } else {
            0.0
        };

        // Dim 18: Comma density (commas / word count)
        let comma_count = raw_text.chars().filter(|&c| c == ',').count();
        features[18] = (comma_count as f32 / n as f32).min(1.0);

        // Dim 19: Semicolon/colon presence (binary)
        features[19] = if raw_text.chars().any(|c| c == ';' || c == ':') {
            1.0
        } else {
            0.0
        };

        // Dim 20: Parenthetical/bracket presence (binary)
        features[20] = if raw_text.chars().any(|c| c == '(' || c == ')' || c == '[' || c == ']') {
            1.0
        } else {
            0.0
        };

        // Dim 21: Hyphenated word ratio
        let hyphenated = words.iter().filter(|w| w.contains('-')).count();
        features[21] = (hyphenated as f32 / n as f32).min(1.0);

        // Dim 22: Short word ratio (<=3 chars)
        let short_words = words.iter().filter(|w| w.len() <= 3).count();
        features[22] = short_words as f32 / n as f32;

        // Dim 23: Long word ratio (>=8 chars)
        let long_words = words.iter().filter(|w| w.len() >= 8).count();
        features[23] = long_words as f32 / n as f32;

        // Dim 24: Word length range (max - min) normalised
        features[24] = ((max_len - min_len) / MAX_TOKEN_LEN).min(1.0);

        // Dim 25: Sentence-initial capital (binary)
        features[25] = if raw_text.chars().next().map_or(false, |c| c.is_uppercase()) {
            1.0
        } else {
            0.0
        };

        // Dim 26: Vowel ratio in words
        let total_chars: usize = words.iter().map(|w| w.len()).sum();
        if total_chars > 0 {
            let vowels: usize = words
                .iter()
                .flat_map(|w| w.chars())
                .filter(|c| "aeiouAEIOU".contains(*c))
                .count();
            features[26] = vowels as f32 / total_chars as f32;
        }

        // Dim 27: Consonant cluster density proxy (consecutive consonants / total chars)
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
            features[27] = (clusters as f32 / total_chars as f32).min(1.0);
        }

        // Dim 28: Mean syllable count proxy (vowel groups per word)
        if n > 0 {
            let total_syllables = Self::count_syllables(words);
            features[28] = (total_syllables as f32 / n as f32 / 5.0).min(1.0);
        }

        // --- Phase 12 features (dims 29–35) ---

        // Dim 29: Relative clause marker count normalised 0–1 against max 5
        let rel_clause_count = words
            .iter()
            .filter(|w| RELATIVE_CLAUSE_MARKERS.contains(&w.as_str()))
            .count();
        features[29] = (rel_clause_count as f32 / 5.0).min(1.0);

        // Dim 30: Subordinating conjunction count normalised 0–1 against max 5
        let subord_count = words
            .iter()
            .filter(|w| SUBORDINATING.contains(&w.as_str()))
            .count();
        features[30] = (subord_count as f32 / 5.0).min(1.0);

        // Dim 31: Pronoun density (pronoun count / total tokens)
        let pronoun_count = words
            .iter()
            .filter(|w| PRONOUNS.contains(&w.as_str()))
            .count();
        features[31] = (pronoun_count as f32 / n as f32).min(1.0);

        // Dim 32: Clause boundary density ((comma + semicolon + colon) / total tokens)
        let semicolons = raw_text.chars().filter(|&c| c == ';').count();
        let colons = raw_text.chars().filter(|&c| c == ':').count();
        features[32] = ((comma_count + semicolons + colons) as f32 / n as f32).min(1.0);

        // Dim 33: Mean word length (chars) — technical/rare words are longer.
        // Normalised by 12.0, capped at 1.0.
        if n > 0 {
            let mean_word_len_chars: f32 =
                words.iter().map(|w| w.len() as f32).sum::<f32>() / n as f32;
            features[33] = (mean_word_len_chars / 12.0).min(1.0);
        }

        // Dim 34: Rare word density — count(words > 8 chars) / total_words.
        // Words longer than 8 characters are typically technical/rare.
        if n > 0 {
            let rare_word_count = words.iter().filter(|w| w.len() > 8).count();
            features[34] = (rare_word_count as f32 / n as f32).min(1.0);
        }

        // Dim 35: Character diversity — unique_chars / total_chars.
        // Complex text uses a more diverse character set.
        if total_chars > 0 {
            let all_chars: Vec<char> = words.iter().flat_map(|w| w.chars()).collect();
            let unique_chars: HashSet<char> = all_chars.iter().copied().collect();
            features[35] = (unique_chars.len() as f32 / total_chars as f32).min(1.0);
        }

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

    /// Compute position-weighted token signal (43 dimensions).
    ///
    /// Each token at position `i` contributes weight `1/(1+i)` to bucket
    /// `token_id % 43`. NOT normalised — magnitude grows with sentence length.
    fn token_signal(token_ids: &[usize]) -> [f32; TOKEN_DIMS] {
        let mut signal = [0.0f32; TOKEN_DIMS];
        for (pos, &id) in token_ids.iter().enumerate() {
            let weight = 1.0 / (1.0 + pos as f32);
            let bucket = id % TOKEN_DIMS;
            signal[bucket] += weight;
        }
        signal
    }

    /// Compute complexity scalars (15 dimensions).
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
    /// - Dim 8: Vocabulary richness — log(unique) / log(total).
    /// - Dim 9: Length variation ratio — max word length / mean word length.
    /// - Dim 10: Rare word ratio — words with >=8 chars / total words.
    /// - Dim 11: Clause count proxy — (commas + subordinating + semicolons) normalised.
    /// - Dims 12–14: Zero padding.
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

        // Dim 8: Vocabulary richness — log(unique) / log(total)
        let vocab_richness = if n > 1 {
            ((unique.len() as f32).ln() / (n as f32).ln()).min(1.0)
        } else {
            1.0
        };

        // Dim 9: Length variation ratio — max word length / mean word length, normalised
        let max_wl = words.iter().map(|w| w.len()).max().unwrap_or(1) as f32;
        let length_variation = if mean_wl > 0.0 {
            (max_wl / mean_wl / 5.0).min(1.0)
        } else {
            0.0
        };

        // Dim 10: Rare word ratio — words with >=8 chars / total
        let rare_count = words.iter().filter(|w| w.len() >= 8).count();
        let rare_ratio = rare_count as f32 / n as f32;

        // Dim 11: Clause count proxy — (commas + subordinating + semicolons) normalised
        let semicolons = raw_text.chars().filter(|&c| c == ';').count();
        let clause_proxy =
            ((comma_count + subord_count + semicolons) as f32 / MAX_DEP_DEPTH).min(1.0);

        [
            token_count_norm, // 0
            ttr,              // 1
            mean_token_len,   // 2
            dep_depth,        // 3
            mean_clause_len,  // 4
            lexical_density,  // 5
            bigram_diversity, // 6
            sentence_rhythm,  // 7
            vocab_richness,   // 8
            length_variation, // 9
            rare_ratio,       // 10
            clause_proxy,     // 11
            0.0,              // 12 (pad)
            0.0,              // 13 (pad)
            0.0,              // 14 (pad)
        ]
    }

    /// Compute dependency depth proxy (4 dimensions).
    ///
    /// Scans tokens left to right maintaining a stack depth counter.
    /// Push when a subordinating conjunction or relative clause marker is found.
    /// Pop when a clause boundary (comma, semicolon, period) or second main verb
    /// (token following subject pronoun) is detected.
    ///
    /// A separate prepositional depth counter pushes on prepositions from
    /// G5_PREPOSITIONS and pops on content words longer than 6 characters
    /// (not in G5_FUNCTION_WORDS), as a proxy for the end of a prepositional phrase.
    ///
    /// Returns: `[max_depth/8, mean_depth/4, std_dev/2, max_prep_depth/6]`, each clamped to [0, 1].
    fn compute_dependency_depth(words: &[String]) -> [f32; 4] {
        if words.is_empty() {
            return [0.0; 4];
        }

        let mut stack_depth: usize = 0;
        let mut depths: Vec<f32> = Vec::with_capacity(words.len());
        let mut max_depth: usize = 0;
        let mut prev_was_pronoun = false;

        let mut prep_depth: usize = 0;
        let mut max_prep_depth: usize = 0;

        for word in words {
            let lower = word.to_lowercase();
            let lower_str = lower.as_str();

            if G5_SUBORDINATORS.contains(&lower_str) {
                stack_depth += 1;
            } else if lower_str == "," || lower_str == ";" || lower_str == "." {
                stack_depth = stack_depth.saturating_sub(1);
            } else if prev_was_pronoun
                && !G5_SUBJECT_PRONOUNS.contains(&lower_str)
                && stack_depth > 0
            {
                // Second main verb detected (token following subject pronoun)
                stack_depth = stack_depth.saturating_sub(1);
            }

            // Prepositional phrase depth tracking
            if G5_PREPOSITIONS.contains(&lower_str) {
                prep_depth += 1;
            } else if word.len() > 6 && !G5_FUNCTION_WORDS.contains(&lower_str) {
                // Content word longer than 6 chars: proxy for end of PP
                prep_depth = prep_depth.saturating_sub(1);
            }

            max_depth = max_depth.max(stack_depth);
            max_prep_depth = max_prep_depth.max(prep_depth);
            depths.push(stack_depth as f32);

            prev_was_pronoun = G5_SUBJECT_PRONOUNS.contains(&lower_str);
        }

        let n = depths.len() as f32;
        let mean = depths.iter().sum::<f32>() / n;
        let variance = depths.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        [
            (max_depth as f32 / 8.0).min(1.0),         // max depth normalised against 8
            (mean / 4.0).min(1.0),                      // mean depth normalised against 4
            (std_dev / 2.0).min(1.0),                   // std dev normalised against 2
            (max_prep_depth as f32 / 6.0).min(1.0),     // max prepositional depth normalised against 6
        ]
    }

    /// Compute constituent length variance (2 dimensions).
    ///
    /// Splits sentence on clause boundaries (comma, semicolon, colon) and on
    /// coordinating/subordinating conjunctions, then measures the variance in
    /// segment lengths. Simple sentences have uniform constituent lengths;
    /// complex sentences have highly variable constituent sizes.
    ///
    /// Coordinating conjunctions: and, or, but, nor, so, yet.
    /// Subordinating conjunctions: that, which, who, whom, because, although,
    /// while, since, if, when, unless, after, before, until, whether, though,
    /// whereas, whereby.
    ///
    /// Returns: `[std_dev/15, max_min_ratio/10]`, each clamped to [0, 1].
    fn compute_constituent_variance(words: &[String]) -> [f32; 2] {
        const COORDINATING: &[&str] = &["and", "or", "but", "nor", "so", "yet"];
        const SUBORDINATING_CONJ: &[&str] = &[
            "that", "which", "who", "whom", "because", "although", "while", "since", "if",
            "when", "unless", "after", "before", "until", "whether", "though", "whereas",
            "whereby",
        ];

        if words.is_empty() {
            return [0.0; 2];
        }

        let mut segments: Vec<usize> = Vec::new();
        let mut current_len: usize = 0;

        for word in words {
            let w = word.as_str();
            let lower = word.to_lowercase();
            let lower_str = lower.as_str();
            if w == "," || w == ";" || w == ":"
                || COORDINATING.contains(&lower_str)
                || SUBORDINATING_CONJ.contains(&lower_str)
            {
                if current_len > 0 {
                    segments.push(current_len);
                }
                current_len = 0;
            } else {
                current_len += 1;
            }
        }
        if current_len > 0 {
            segments.push(current_len);
        }

        if segments.len() < 2 {
            return [0.0; 2];
        }

        let mean = segments.iter().sum::<usize>() as f32 / segments.len() as f32;
        let variance = segments
            .iter()
            .map(|&s| (s as f32 - mean).powi(2))
            .sum::<f32>()
            / segments.len() as f32;
        let std_dev = variance.sqrt();

        let max_seg = *segments.iter().max().unwrap() as f32;
        let min_seg = *segments.iter().min().unwrap() as f32;
        let ratio = if min_seg > 0.0 {
            max_seg / min_seg
        } else {
            max_seg
        };

        [
            (std_dev / 15.0).min(1.0), // std dev of segment lengths, norm against 15
            (ratio / 10.0).min(1.0),   // max/min ratio, norm against 10
        ]
    }

    /// Compute function word position entropy (5 dimensions).
    ///
    /// Computes the positions of function words as a fraction of sentence length
    /// and analyses their distribution. Simple sentences front-load content words
    /// and distribute function words evenly. Complex sentences with embedded
    /// clauses cluster function words at embedding points.
    ///
    /// Returns: `[mean_position, std_dev/0.4, fw_density_first, fw_density_mid, fw_density_last]`,
    /// each clamped to [0, 1].
    ///
    /// - `fw_density_first`: function word count in first third / total function words
    /// - `fw_density_mid`: function word count in middle third / total function words
    /// - `fw_density_last`: function word count in last third / total function words
    fn compute_function_word_entropy(words: &[String]) -> [f32; 5] {
        let n = words.len();
        if n == 0 {
            return [0.0; 5];
        }

        let positions: Vec<f32> = words
            .iter()
            .enumerate()
            .filter(|(_, w)| {
                let lower = w.to_lowercase();
                G5_FUNCTION_WORDS.contains(&lower.as_str())
            })
            .map(|(i, _)| i as f32 / n as f32)
            .collect();

        let fw_count = positions.len();
        if fw_count == 0 {
            return [0.0; 5];
        }

        let mean = positions.iter().sum::<f32>() / fw_count as f32;
        let variance = positions
            .iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f32>()
            / fw_count as f32;
        let std_dev = variance.sqrt();

        // Density window dimensions
        let first_count = positions.iter().filter(|&&p| p < 1.0 / 3.0).count() as f32;
        let mid_count = positions
            .iter()
            .filter(|&&p| p >= 1.0 / 3.0 && p < 2.0 / 3.0)
            .count() as f32;
        let last_count = positions.iter().filter(|&&p| p >= 2.0 / 3.0).count() as f32;
        let fw_total = fw_count as f32;

        let fw_density_first = (first_count / fw_total).min(1.0);
        let fw_density_mid = (mid_count / fw_total).min(1.0);
        let fw_density_last = (last_count / fw_total).min(1.0);

        [
            mean.min(1.0),           // mean function word position
            (std_dev / 0.4).min(1.0), // std dev, norm against 0.4
            fw_density_first,        // fraction of fw in first third
            fw_density_mid,          // fraction of fw in middle third
            fw_density_last,         // fraction of fw in last third
        ]
    }

    /// Compute structural syntax features (12 dimensions).
    ///
    /// Combines three vocabulary-independent structural measurements:
    /// - Dependency depth proxy (4 dims): syntactic embedding depth + prepositional depth
    /// - Constituent length variance (2 dims): clause size variation
    /// - Function word position entropy (5 dims): structure word distribution across thirds
    /// - 1 pad dimension
    fn structural_syntax_features(words: &[String]) -> [f32; STRUCTURAL_DIMS] {
        let dep = Self::compute_dependency_depth(words);
        let con = Self::compute_constituent_variance(words);
        let fwe = Self::compute_function_word_entropy(words);

        [
            dep[0], dep[1], dep[2], dep[3], // dependency depth: max, mean, std_dev, prep_max
            con[0], con[1],                  // constituent variance: std_dev, ratio
            fwe[0], fwe[1], fwe[2], fwe[3], fwe[4], // fw entropy: mean, std, first, mid, last
            0.0,                             // pad
        ]
    }

    /// Amplification factors for each feature group.
    ///
    /// G1 (n-gram) and G2 (syntactic) are the strongest discriminating groups
    /// between simple and complex sentences — amplified by 3.0 to widen the
    /// directional separation in cosine space. G4 (complexity scalars) amplified
    /// by 2.0. G5 (structural syntax) amplified by 3.0 to match structural
    /// groups. G3 (token signal) unchanged.
    const G1_AMP: f32 = 3.0;
    const G2_AMP: f32 = 3.0;
    const G3_AMP: f32 = 1.0;
    const G4_AMP: f32 = 2.0;
    const G5_AMP: f32 = 3.0;

    /// Encode token IDs and raw text into a 128-dim structural tensor.
    ///
    /// Concatenates all five feature groups with group-level amplification
    /// to widen directional separation between simple and complex sentences.
    fn encode_with_features(&self, token_ids: &[usize], text: &str) -> Tensor {
        let words = Tokeniser::split(text);

        // Group 1: Character n-gram profile (dims 0–25) — amplified
        let ngrams = Self::ngram_profile(&words);

        // Group 2: Syntactic proxy features (dims 26–61) — amplified
        let syntactic = Self::syntactic_features(&words, text);

        // Group 3: Position-weighted token signal (dims 62–100) — unchanged
        let tokens = Self::token_signal(token_ids);

        // Group 4: Complexity scalars (dims 101–115) — amplified
        let scalars = Self::complexity_scalars(&words, text);

        // Group 5: Structural syntax features (dims 116–127) — amplified
        let structural = Self::structural_syntax_features(&words);

        // Concatenate with amplification: 26 + 36 + 39 + 15 + 12 = 128
        let mut data = Vec::with_capacity(OUTPUT_DIM);
        data.extend(ngrams.iter().map(|v| v * Self::G1_AMP));
        data.extend(syntactic.iter().map(|v| v * Self::G2_AMP));
        data.extend(tokens.iter().map(|v| v * Self::G3_AMP));
        data.extend(scalars.iter().map(|v| v * Self::G4_AMP));

        // G5: optionally normalize by sqrt(token_count) to decouple length from structure
        let g5_scale = if self.g5_length_normalize {
            let token_count = token_ids.len().max(1) as f32;
            Self::G5_AMP / token_count.sqrt()
        } else {
            Self::G5_AMP
        };
        data.extend(structural.iter().map(|v| v * g5_scale));

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

    /// Print G5 structural syntax feature values for diagnostic purposes.
    pub fn print_g5_features(&self, text: &str) {
        let words = Tokeniser::split(text);
        let dep = Self::compute_dependency_depth(&words);
        let con = Self::compute_constituent_variance(&words);
        let fwe = Self::compute_function_word_entropy(&words);
        let tensor = self.encode_text_readonly(text);

        println!("  \"{}\"", text);
        println!("    Norm: {:.4}", tensor.norm());
        println!(
            "    G5 dep_depth: max={:.4} mean={:.4} std={:.4} prep_max={:.4}",
            dep[0], dep[1], dep[2], dep[3]
        );
        println!(
            "    G5 const_var: std={:.4} ratio={:.4}",
            con[0], con[1]
        );
        println!(
            "    G5 fw_entropy: mean={:.4} std={:.4} first={:.4} mid={:.4} last={:.4}",
            fwe[0], fwe[1], fwe[2], fwe[3], fwe[4]
        );
    }

    /// Print a detailed feature vector breakdown for diagnostic purposes.
    ///
    /// Shows all 5 feature groups with labels for a given sentence.
    pub fn print_feature_breakdown(&self, text: &str) {
        let ids = self.tokeniser.tokenise_readonly(text);
        let words = Tokeniser::split(text);
        let ngrams = Self::ngram_profile(&words);
        let syntactic = Self::syntactic_features(&words, text);
        let tokens = Self::token_signal(&ids);
        let scalars = Self::complexity_scalars(&words, text);
        let structural = Self::structural_syntax_features(&words);

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
            "len_var", "uniq_r", "func_wd", "max_len", "sent_ln", "punct", "caps",
            "quest", "negat", "subord", "comma_d", "semi_cl", "parenth", "hyphen",
            "short_r", "long_r", "len_rng", "init_cp",
            "vowel_r", "cons_cl", "syl_cnt",
            "rel_cls", "sub_cnt", "pron_dn", "cls_bnd",
            "syl_prx", "hapax_r", "cmplx_s",
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
            "voc_rch", "len_var", "rare_r", "clause",
            "pad_0", "pad_1", "pad_2",
        ];
        for (v, label) in scalars.iter().zip(scalar_labels.iter()) {
            print!("      {:>7}: {:.4}", label, v);
        }
        println!();
        println!("    Group 5 — Structural syntax ({}d):", STRUCTURAL_DIMS);
        let struct_labels = [
            "dep_max", "dep_mea", "dep_std", "dep_pre",
            "con_std", "con_rat",
            "fw_mean", "fw_std", "fw_fst", "fw_mid", "fw_lst",
            "pad",
        ];
        for (v, label) in structural.iter().zip(struct_labels.iter()) {
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
        // Sentence length feature: G2 base (26) + dim 12 = 38
        let simple_sent_len = simple.data[26 + 12];
        let complex_sent_len = complex.data[26 + 12];
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
        // Token signal: G3 base (62) to G3 end (101)
        let g3_start = NGRAM_DIMS + SYNTACTIC_DIMS;
        let g3_end = g3_start + TOKEN_DIMS;
        assert_ne!(
            t1.data[g3_start..g3_end],
            t2.data[g3_start..g3_end],
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
        // Token count scalar: G4 base (105) + dim 0
        let g4_base = NGRAM_DIMS + SYNTACTIC_DIMS + TOKEN_DIMS;
        assert!(
            complex.data[g4_base] > simple.data[g4_base],
            "Token count: complex={:.4} should exceed simple={:.4}",
            complex.data[g4_base],
            simple.data[g4_base]
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
        // Subordinating conjunction presence: G2 base (26) + dim 17 = 43
        assert!(
            with_subord.data[26 + 17] > no_subord.data[26 + 17],
            "Subordinating: with={:.1} should exceed without={:.1}",
            with_subord.data[26 + 17],
            no_subord.data[26 + 17]
        );
    }

    #[test]
    fn test_punctuation_feature() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let no_punct = enc.encode_text("hello world");
        let with_punct = enc.encode_text("hello, world! how are you?");
        // Punctuation feature: G2 base (26) + dim 13 = 39
        assert!(
            with_punct.data[26 + 13] > no_punct.data[26 + 13],
            "Punctuation: with={:.4} should exceed without={:.4}",
            with_punct.data[26 + 13],
            no_punct.data[26 + 13]
        );
    }

    #[test]
    fn test_function_word_density() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let low_fn = enc.encode_text("recursive systems create emergent properties");
        let high_fn = enc.encode_text("the cat is on the mat in the room");
        // Function word density: G2 base (26) + dim 10 = 36
        assert!(
            high_fn.data[26 + 10] > low_fn.data[26 + 10],
            "Function word density: high={:.4} should exceed low={:.4}",
            high_fn.data[26 + 10],
            low_fn.data[26 + 10]
        );
    }

    #[test]
    fn test_dependency_depth_proxy() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let simple = enc.encode_text("the cat sat");
        let complex =
            enc.encode_text("because the cat, which was tired, sat down, it slept");
        // Dependency depth proxy: G4 base (105) + dim 3
        let g4_base = NGRAM_DIMS + SYNTACTIC_DIMS + TOKEN_DIMS;
        assert!(
            complex.data[g4_base + 3] > simple.data[g4_base + 3],
            "Dep depth: complex={:.4} should exceed simple={:.4}",
            complex.data[g4_base + 3],
            simple.data[g4_base + 3]
        );
    }

    #[test]
    fn test_negation_detection() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let positive = enc.encode_text("the cat sat quietly");
        let negative = enc.encode_text("the cat never sat quietly");
        // Negation feature: G2 base (26) + dim 16 = 42
        assert!(
            negative.data[26 + 16] > positive.data[26 + 16],
            "Negation: negative={:.1} should exceed positive={:.1}",
            negative.data[26 + 16],
            positive.data[26 + 16]
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
        let raw_structural = Encoder::structural_syntax_features(&words);

        // G1 (dims 0..26) should be amplified by 3.0
        for i in 0..NGRAM_DIMS {
            let expected = raw_ngrams[i] * Encoder::G1_AMP;
            assert!(
                (tensor.data[i] - expected).abs() < 1e-6,
                "G1 dim {}: expected {:.6}, got {:.6}",
                i, expected, tensor.data[i]
            );
        }

        // G2 (dims 26..62) should be amplified by 3.0
        let g2_base = NGRAM_DIMS;
        for i in 0..SYNTACTIC_DIMS {
            let expected = raw_syntactic[i] * Encoder::G2_AMP;
            assert!(
                (tensor.data[g2_base + i] - expected).abs() < 1e-6,
                "G2 dim {}: expected {:.6}, got {:.6}",
                i, expected, tensor.data[g2_base + i]
            );
        }

        // G3 (dims 62..101) should be amplified by 1.0 (unchanged)
        let g3_base = NGRAM_DIMS + SYNTACTIC_DIMS;
        for i in 0..TOKEN_DIMS {
            let expected = raw_tokens[i] * Encoder::G3_AMP;
            assert!(
                (tensor.data[g3_base + i] - expected).abs() < 1e-6,
                "G3 dim {}: expected {:.6}, got {:.6}",
                i, expected, tensor.data[g3_base + i]
            );
        }

        // G4 (dims 101..116) should be amplified by 2.0
        let g4_base = NGRAM_DIMS + SYNTACTIC_DIMS + TOKEN_DIMS;
        for i in 0..SCALAR_DIMS {
            let expected = raw_scalars[i] * Encoder::G4_AMP;
            assert!(
                (tensor.data[g4_base + i] - expected).abs() < 1e-6,
                "G4 dim {}: expected {:.6}, got {:.6}",
                i, expected, tensor.data[g4_base + i]
            );
        }

        // G5 (dims 116..128) should be amplified by 3.0
        let g5_base = NGRAM_DIMS + SYNTACTIC_DIMS + TOKEN_DIMS + SCALAR_DIMS;
        for i in 0..STRUCTURAL_DIMS {
            let expected = raw_structural[i] * Encoder::G5_AMP;
            assert!(
                (tensor.data[g5_base + i] - expected).abs() < 1e-6,
                "G5 dim {}: expected {:.6}, got {:.6}",
                i, expected, tensor.data[g5_base + i]
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

    // --- Phase 12 feature tests ---

    #[test]
    fn test_nested_clause_depth_relative() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let no_rel = enc.encode_text("the cat sat on the mat");
        let with_rel = enc.encode_text("the cat that the dog chased sat on the mat");
        // Relative clause marker count: G2 base (26) + dim 29 = 55
        assert!(
            with_rel.data[26 + 29] > no_rel.data[26 + 29],
            "Relative clause markers: with={:.4} should exceed without={:.4}",
            with_rel.data[26 + 29],
            no_rel.data[26 + 29]
        );
    }

    #[test]
    fn test_nested_clause_depth_subordinating() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let no_sub = enc.encode_text("the cat sat on the mat");
        let with_sub = enc.encode_text("because the cat sat after the dog left until it rained");
        // Subordinating conjunction count: G2 base (26) + dim 30 = 56
        assert!(
            with_sub.data[26 + 30] > no_sub.data[26 + 30],
            "Subordinating count: with={:.4} should exceed without={:.4}",
            with_sub.data[26 + 30],
            no_sub.data[26 + 30]
        );
    }

    #[test]
    fn test_pronoun_density() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let no_pron = enc.encode_text("the cat sat on the mat");
        let with_pron = enc.encode_text("he told her that they would help us");
        // Pronoun density: G2 base (26) + dim 31 = 57
        assert!(
            with_pron.data[26 + 31] > no_pron.data[26 + 31],
            "Pronoun density: with={:.4} should exceed without={:.4}",
            with_pron.data[26 + 31],
            no_pron.data[26 + 31]
        );
    }

    #[test]
    fn test_clause_boundary_density() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let no_bound = enc.encode_text("the cat sat on the mat");
        let with_bound = enc.encode_text("the cat, which was tired; sat down: sleeping");
        // Clause boundary density: G2 base (26) + dim 32 = 58
        assert!(
            with_bound.data[26 + 32] > no_bound.data[26 + 32],
            "Clause boundary density: with={:.4} should exceed without={:.4}",
            with_bound.data[26 + 32],
            no_bound.data[26 + 32]
        );
    }

    #[test]
    fn test_rare_word_density() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let simple = enc.encode_text("the cat sat on the mat");
        let technical = enc.encode_text("quantum mechanical descriptions fundamental interactions");
        // Rare word density (words > 8 chars): G2 base (26) + dim 34 = 60
        assert!(
            technical.data[26 + 34] > simple.data[26 + 34],
            "Rare word density: technical={:.4} should exceed simple={:.4}",
            technical.data[26 + 34],
            simple.data[26 + 34]
        );
    }

    #[test]
    fn test_character_diversity() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let simple = enc.encode_text("the the the the the");
        let diverse = enc.encode_text("complex philosophical metaphysics quantum");
        // Character diversity: G2 base (26) + dim 35 = 61
        assert!(
            diverse.data[26 + 35] > simple.data[26 + 35],
            "Char diversity: diverse={:.4} should exceed simple={:.4}",
            diverse.data[26 + 35],
            simple.data[26 + 35]
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
        // Lexical density: G4 base (105) + dim 5
        let g4_base = NGRAM_DIMS + SYNTACTIC_DIMS + TOKEN_DIMS;
        assert!(
            content_heavy.data[g4_base + 5] > function_heavy.data[g4_base + 5],
            "Lexical density: content_heavy={:.4} should exceed function_heavy={:.4}",
            content_heavy.data[g4_base + 5],
            function_heavy.data[g4_base + 5]
        );
    }

    #[test]
    fn test_bigram_diversity_scalar() {
        let tok = Tokeniser::new(200);
        let mut enc = Encoder::new(128, tok);
        let repeated = enc.encode_text("the cat the cat the cat the cat");
        let diverse = enc.encode_text("quantum mechanics describes fundamental particle interactions");
        // Bigram diversity: G4 base (105) + dim 6
        let g4_base = NGRAM_DIMS + SYNTACTIC_DIMS + TOKEN_DIMS;
        assert!(
            diverse.data[g4_base + 6] > repeated.data[g4_base + 6],
            "Bigram diversity: diverse={:.4} should exceed repeated={:.4}",
            diverse.data[g4_base + 6],
            repeated.data[g4_base + 6]
        );
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

        println!("\n=== Phase 14 Stage 1 Diagnostic (6 sentences) ===");
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

    // --- Phase 14 G5 structural syntax tests ---

    #[test]
    fn test_g5_dimensions() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let t = enc.encode_text("the cat sat on the mat");
        assert_eq!(t.shape, vec![128]);
        // G5 occupies dims 116–127
        let g5_base = NGRAM_DIMS + SYNTACTIC_DIMS + TOKEN_DIMS + SCALAR_DIMS;
        assert_eq!(g5_base, 116);
        assert_eq!(g5_base + STRUCTURAL_DIMS, 128);
    }

    #[test]
    fn test_group_dimensions_sum_to_128() {
        assert_eq!(
            NGRAM_DIMS + SYNTACTIC_DIMS + TOKEN_DIMS + SCALAR_DIMS + STRUCTURAL_DIMS,
            128,
            "Group dimensions must sum to 128: {}+{}+{}+{}+{}",
            NGRAM_DIMS, SYNTACTIC_DIMS, TOKEN_DIMS, SCALAR_DIMS, STRUCTURAL_DIMS
        );
    }

    #[test]
    fn test_dependency_depth_simple_vs_complex() {
        // Simple sentence: no subordinators, no nesting
        let simple = vec!["the", "cat", "sat"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let dep_simple: [f32; 4] = Encoder::compute_dependency_depth(&simple);

        // Complex sentence with multiple embedding levels
        let complex = vec![
            "the", "cat", "that", "the", "dog", "that", "the", "man",
            "owned", "chased", "sat", "on", "the", "mat",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
        let dep_complex: [f32; 4] = Encoder::compute_dependency_depth(&complex);

        assert!(
            dep_complex[0] > dep_simple[0],
            "Max depth: complex={:.4} should exceed simple={:.4}",
            dep_complex[0], dep_simple[0]
        );
        assert!(
            dep_complex[1] > dep_simple[1],
            "Mean depth: complex={:.4} should exceed simple={:.4}",
            dep_complex[1], dep_simple[1]
        );
    }

    #[test]
    fn test_constituent_variance_uniform_vs_variable() {
        // Uniform clause lengths: "the cat sat , the dog ran"
        let uniform = vec!["the", "cat", "sat", ",", "the", "dog", "ran"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let con_uniform = Encoder::compute_constituent_variance(&uniform);

        // Variable clause lengths: "the cat that the dog chased , sat"
        let variable = vec![
            "the", "cat", "that", "the", "dog", "that", "the", "man",
            "owned", "chased", ",", "sat",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
        let con_variable = Encoder::compute_constituent_variance(&variable);

        assert!(
            con_variable[0] > con_uniform[0],
            "Constituent std_dev: variable={:.4} should exceed uniform={:.4}",
            con_variable[0], con_uniform[0]
        );
        assert!(
            con_variable[1] > con_uniform[1],
            "Constituent ratio: variable={:.4} should exceed uniform={:.4}",
            con_variable[1], con_uniform[1]
        );
    }

    #[test]
    fn test_function_word_entropy_front_vs_distributed() {
        // Front-loaded function words: "the a an is cat dog run fast"
        let front = vec!["the", "a", "an", "is", "cat", "dog", "run", "fast"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let fwe_front: [f32; 5] = Encoder::compute_function_word_entropy(&front);

        // Distributed function words: "cat the dog is run an fast a"
        let distributed = vec!["cat", "the", "dog", "is", "run", "an", "fast", "a"]
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let fwe_distributed: [f32; 5] = Encoder::compute_function_word_entropy(&distributed);

        // Front-loaded should have lower mean position
        assert!(
            fwe_front[0] < fwe_distributed[0],
            "Front-loaded mean position ({:.4}) should be less than distributed ({:.4})",
            fwe_front[0], fwe_distributed[0]
        );
    }

    #[test]
    fn test_g5_produces_12_dimensions() {
        let words: Vec<String> = vec!["the", "cat", "that", "sat"]
            .into_iter()
            .map(String::from)
            .collect();
        let g5 = Encoder::structural_syntax_features(&words);
        assert_eq!(g5.len(), 12);
    }

    #[test]
    fn test_g5_total_encoder_output_128() {
        let tok = Tokeniser::new(100);
        let mut enc = Encoder::new(128, tok);
        let t = enc.encode_text("test sentence with some words");
        assert_eq!(t.data.len(), 128);
    }

    #[test]
    fn test_g5_adversarial_discrimination() {
        let tok = Tokeniser::new(1024);
        let mut enc = Encoder::new(128, tok);

        // All 8 Phase 13 adversarial failures
        let complex_misclassified = [
            "the cat that the dog that the man owned chased sat on the mat",
            "she said that he thought that they believed it was true",
            "consciousness remains one of the most profound unsolved problems in all of science",
            "dark matter constitutes approximately twenty seven percent of the total mass energy content of the observable universe",
        ];

        let simple_misclassified = [
            "the tintinnabulation resonated melodiously",
            "photosynthesis converts sunlight efficiently",
            "if then else",
            "go",
        ];

        println!("\n=== Phase 14 Stage 1 — G5 Adversarial Feature Values ===");
        println!("\n  COMPLEX sentences (should escalate, stayed Surface in Phase 13):");
        println!("  {:>4}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}",
            "#", "dep_max", "dep_mea", "dep_std", "dep_pre", "con_std", "con_rat", "fw_mea", "fw_std", "fw_fst", "fw_lst", "norm");

        let mut complex_dep_sum = 0.0f32;
        let mut complex_con_sum = 0.0f32;
        for (i, text) in complex_misclassified.iter().enumerate() {
            let words = Tokeniser::split(text);
            let dep = Encoder::compute_dependency_depth(&words);
            let con = Encoder::compute_constituent_variance(&words);
            let fwe = Encoder::compute_function_word_entropy(&words);
            let tensor = enc.encode_text(text);
            println!(
                "  C{:>2}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}",
                i + 1, dep[0], dep[1], dep[2], dep[3], con[0], con[1], fwe[0], fwe[1], fwe[2], fwe[4], tensor.norm()
            );
            println!("       \"{}\"", text);
            complex_dep_sum += dep[0];
            complex_con_sum += con[0];
        }

        println!("\n  SIMPLE sentences (should stay Surface, escalated in Phase 13):");
        println!("  {:>4}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}",
            "#", "dep_max", "dep_mea", "dep_std", "dep_pre", "con_std", "con_rat", "fw_mea", "fw_std", "fw_fst", "fw_lst", "norm");

        let mut simple_dep_sum = 0.0f32;
        let mut simple_con_sum = 0.0f32;
        for (i, text) in simple_misclassified.iter().enumerate() {
            let words = Tokeniser::split(text);
            let dep = Encoder::compute_dependency_depth(&words);
            let con = Encoder::compute_constituent_variance(&words);
            let fwe = Encoder::compute_function_word_entropy(&words);
            let tensor = enc.encode_text(text);
            println!(
                "  S{:>2}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}  {:>7.4}",
                i + 1, dep[0], dep[1], dep[2], dep[3], con[0], con[1], fwe[0], fwe[1], fwe[2], fwe[4], tensor.norm()
            );
            println!("       \"{}\"", text);
            simple_dep_sum += dep[0];
            simple_con_sum += con[0];
        }

        let complex_avg_dep = complex_dep_sum / complex_misclassified.len() as f32;
        let simple_avg_dep = simple_dep_sum / simple_misclassified.len() as f32;
        let complex_avg_con = complex_con_sum / complex_misclassified.len() as f32;
        let simple_avg_con = simple_con_sum / simple_misclassified.len() as f32;

        println!("\n  Summary:");
        println!("    Complex avg dep_max: {:.4}  Simple avg dep_max: {:.4}", complex_avg_dep, simple_avg_dep);
        println!("    Complex avg con_std: {:.4}  Simple avg con_std: {:.4}", complex_avg_con, simple_avg_con);
        println!("    G5 DISCRIMINATES: dep_max complex > simple = {}", complex_avg_dep > simple_avg_dep);

        // Complex sentences should have higher average max dependency depth
        assert!(
            complex_avg_dep > simple_avg_dep,
            "Complex avg dep_max ({:.4}) should exceed simple avg dep_max ({:.4})",
            complex_avg_dep, simple_avg_dep
        );
    }
}
