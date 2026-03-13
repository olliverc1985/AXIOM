"""
AXIOM Encoder — Autoresearch Target
====================================
This is the ONLY file the agent modifies.

Faithful port of axiom-core/src/input/encoder.rs (Structural Encoder V5).
128-dimensional output = G1(26) + G2(36) + G3(39) + G4(15) + G5(12).

Current baseline: ~100% simple accuracy, ~22% complex accuracy.
Goal: maximize complex accuracy without sacrificing simple accuracy.

The encoder ceiling: G5 structural features cannot detect semantic complexity
without syntactic markers. Short philosophical queries ("Cogito ergo sum")
and domain-complex queries with simple syntax get misrouted to Surface.

Run: python3 train.py
"""

import math
from prepare import evaluate

# ─── Constants ───

FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
    "to", "of", "and", "or", "but",
}

QUESTION_WORDS = {"who", "what", "where", "when", "why", "how", "which"}

NEGATION_WORDS = {"not", "never", "no", "neither", "nor"}

SUBORDINATING = {
    "because", "although", "while", "since", "if", "when", "unless",
    "after", "before", "until",
}

RELATIVE_CLAUSE_MARKERS = {"that", "which", "who", "whom"}

PRONOUNS = {
    "he", "she", "it", "they", "we", "i", "you", "him", "her", "them",
    "us", "me",
}

G5_SUBORDINATORS = {
    "that", "which", "who", "whom", "because", "although", "while",
    "since", "if", "when", "unless", "after", "before", "until",
    "whether", "though", "whereas", "whereby",
}

G5_SUBJECT_PRONOUNS = {"he", "she", "it", "they", "we", "i", "you"}

G5_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "of", "in", "on",
    "at", "to", "for", "with", "by", "from", "as", "or", "and", "but",
    "nor", "so", "yet", "not", "no", "it", "its", "this", "that",
    "these", "those", "there", "here", "very", "just", "also", "only",
    "even", "both", "either", "each", "all", "any", "some", "few",
    "more", "most",
}

G5_PREPOSITIONS = {
    "of", "in", "by", "with", "for", "through", "among", "between", "within",
}

COORDINATING = {"and", "or", "but", "nor", "so", "yet"}

MAX_SENTENCE_LEN = 40.0
MAX_TOKEN_LEN = 15.0
MAX_DEP_DEPTH = 10.0

# ─── Amplification factors ───
G1_AMP = 3.0
G2_AMP = 3.0
G3_AMP = 1.0
G4_AMP = 4.0
G5_AMP = 3.0

# ─── Dimensions ───
NGRAM_DIMS = 26
SYNTACTIC_DIMS = 36
TOKEN_DIMS = 39
SCALAR_DIMS = 15
STRUCTURAL_DIMS = 12
OUTPUT_DIM = 128  # 26 + 36 + 39 + 15 + 12


# ─── Tokenizer ───

def tokenize(text):
    """Split text into lowercase words on non-alphanumeric boundaries (keep apostrophes)."""
    lower = text.lower()
    tokens = []
    current = []
    for ch in lower:
        if ch.isalnum() or ch == "'":
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))
    return tokens


# ─── Group 1: Character N-gram Profile (26 dims) ───

def ngram_hash(chars):
    """DJB2 hash variant for character n-grams."""
    h = 5381
    for ch in chars:
        for byte in ord(ch).to_bytes(4, 'little'):
            if byte == 0:
                continue
            h = ((h * 33) + byte) & 0xFFFFFFFF
    return h


def g1_ngram_profile(words):
    """Character bigram + trigram hash buckets, normalized by total count."""
    buckets = [0] * NGRAM_DIMS
    total = 0

    for word in words:
        chars = list(word)
        for i in range(len(chars) - 1):
            bucket = ngram_hash(chars[i:i+2]) % NGRAM_DIMS
            buckets[bucket] += 1
            total += 1
        for i in range(len(chars) - 2):
            bucket = ngram_hash(chars[i:i+3]) % NGRAM_DIMS
            buckets[bucket] += 1
            total += 1

    if total > 0:
        return [b / total for b in buckets]
    return [0.0] * NGRAM_DIMS


# ─── Group 2: Syntactic Proxy Features (36 dims) ───

def count_syllables(words):
    """Count total syllable-like vowel groups across all words."""
    total = 0
    for word in words:
        syl = 0
        prev_vowel = False
        for ch in word:
            is_vowel = ch in "aeiouAEIOU"
            if is_vowel and not prev_vowel:
                syl += 1
            prev_vowel = is_vowel
        total += max(syl, 1)
    return total


def g2_syntactic_features(words, raw_text):
    """36 syntactic proxy features computed without a parser."""
    features = [0.0] * SYNTACTIC_DIMS
    n = len(words)
    if n == 0:
        return features

    word_lengths = [len(w) for w in words]

    # Dims 0-7: Average word length in each of 8 sentence octiles
    octile_size = (n + 7) // 8
    for q in range(8):
        start = q * octile_size
        end = min((q + 1) * octile_size, n)
        if start < end:
            features[q] = sum(word_lengths[start:end]) / (end - start) / MAX_TOKEN_LEN

    # Dim 8: Word length variance
    mean_len = sum(word_lengths) / n
    variance = sum((l - mean_len) ** 2 for l in word_lengths) / n
    features[8] = variance / (MAX_TOKEN_LEN * MAX_TOKEN_LEN)

    # Dim 9: Unique word ratio
    unique = set(words)
    features[9] = len(unique) / n

    # Dim 10: Function word density
    fn_count = sum(1 for w in words if w in FUNCTION_WORDS)
    features[10] = fn_count / n

    # Dim 11: Max word length normalized
    max_len = max(word_lengths)
    min_len = min(word_lengths)
    features[11] = min(max_len / MAX_TOKEN_LEN, 1.0)

    # Dim 12: Sentence length normalized
    features[12] = min(n / MAX_SENTENCE_LEN, 1.0)

    # Dim 13: Punctuation count normalized
    punct_count = sum(1 for c in raw_text if c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    features[13] = min(punct_count / n, 1.0)

    # Dim 14: Capitalization count normalized
    cap_count = sum(1 for c in raw_text if c.isupper())
    features[14] = min(cap_count / n, 1.0)

    # Dim 15: Question word presence
    features[15] = 1.0 if any(w in QUESTION_WORDS for w in words) else 0.0

    # Dim 16: Negation presence
    features[16] = 1.0 if any(w in NEGATION_WORDS for w in words) else 0.0

    # Dim 17: Subordinating conjunction presence
    features[17] = 1.0 if any(w in SUBORDINATING for w in words) else 0.0

    # Dim 18: Comma density
    comma_count = raw_text.count(',')
    features[18] = min(comma_count / n, 1.0)

    # Dim 19: Semicolon/colon presence
    features[19] = 1.0 if (';' in raw_text or ':' in raw_text) else 0.0

    # Dim 20: Parenthetical presence
    features[20] = 1.0 if any(c in raw_text for c in '()[]') else 0.0

    # Dim 21: Hyphenated word ratio
    hyphenated = sum(1 for w in words if '-' in w)
    features[21] = min(hyphenated / n, 1.0)

    # Dim 22: Short word ratio (<=3 chars)
    short_words = sum(1 for w in words if len(w) <= 3)
    features[22] = short_words / n

    # Dim 23: Long word ratio (>=8 chars)
    long_words = sum(1 for w in words if len(w) >= 8)
    features[23] = long_words / n

    # Dim 24: Word length range normalized
    features[24] = min((max_len - min_len) / MAX_TOKEN_LEN, 1.0)

    # Dim 25: Sentence-initial capital
    features[25] = 1.0 if (raw_text and raw_text[0].isupper()) else 0.0

    # Dim 26: Vowel ratio
    total_chars = sum(word_lengths)
    if total_chars > 0:
        vowels = sum(1 for w in words for c in w if c in "aeiouAEIOU")
        features[26] = vowels / total_chars

    # Dim 27: Consonant cluster density
    if total_chars > 0:
        clusters = 0
        for w in words:
            for ch in w:
                if ch.isalpha() and ch not in "aeiouAEIOU":
                    clusters += 1
        features[27] = min(clusters / total_chars, 1.0)

    # Dim 28: Mean syllable count proxy
    if n > 0:
        total_syl = count_syllables(words)
        features[28] = min(total_syl / n / 5.0, 1.0)

    # Dim 29: Relative clause marker count
    rel_count = sum(1 for w in words if w in RELATIVE_CLAUSE_MARKERS)
    features[29] = min(rel_count / 5.0, 1.0)

    # Dim 30: Subordinating conjunction count
    sub_count = sum(1 for w in words if w in SUBORDINATING)
    features[30] = min(sub_count / 5.0, 1.0)

    # Dim 31: Pronoun density
    pron_count = sum(1 for w in words if w in PRONOUNS)
    features[31] = min(pron_count / n, 1.0)

    # Dim 32: Clause boundary density
    semicolons = raw_text.count(';')
    colons = raw_text.count(':')
    features[32] = min((comma_count + semicolons + colons) / n, 1.0)

    # Dim 33: Mean word length (chars)
    if n > 0:
        features[33] = min(sum(word_lengths) / n / 12.0, 1.0)

    # Dim 34: Rare word density (words > 8 chars)
    if n > 0:
        rare = sum(1 for w in words if len(w) > 8)
        features[34] = min(rare / n, 1.0)

    # Dim 35: Character diversity
    if total_chars > 0:
        all_chars = set(c for w in words for c in w)
        features[35] = min(len(all_chars) / total_chars, 1.0)

    return features


# ─── Group 3: Position-weighted Token Signal (39 dims) ───

def g3_token_signal(words):
    """Position-weighted token hash signal. Vocabulary-independent via DJB2."""
    signal = [0.0] * TOKEN_DIMS
    for pos, word in enumerate(words):
        weight = 1.0 / (1.0 + pos)
        # Hash word to get stable bucket assignment
        h = 5381
        for c in word:
            h = ((h * 33) + ord(c)) & 0xFFFFFFFF
        bucket = h % TOKEN_DIMS
        signal[bucket] += weight
    return signal


# ─── Group 4: Complexity Scalars (15 dims) ───

def g4_complexity_scalars(words, raw_text):
    """15 complexity scalar features."""
    n = len(words)
    if n == 0:
        return [0.0] * SCALAR_DIMS

    word_lengths = [len(w) for w in words]
    total_char_len = sum(word_lengths)

    # Dim 0: Token count normalized
    token_count_norm = min(n / MAX_SENTENCE_LEN, 1.0)

    # Dim 1: Type-token ratio
    unique = set(words)
    ttr = len(unique) / n

    # Dim 2: Mean token length normalized
    mean_token_len = min(total_char_len / n / MAX_TOKEN_LEN, 1.0)

    # Dim 3: Dependency depth proxy
    sub_count = sum(1 for w in words if w in SUBORDINATING)
    comma_count = raw_text.count(',')
    dep_depth = min((sub_count + comma_count) / MAX_DEP_DEPTH, 1.0)

    # Dim 4: Mean clause length proxy
    mean_clause_len = min(n / (comma_count + 1.0) / 30.0, 1.0)

    # Dim 5: Lexical density
    content_count = sum(1 for w in words if w not in FUNCTION_WORDS)
    lexical_density = content_count / n

    # Dim 6: Bigram diversity
    if n > 1:
        bigrams = set()
        for i in range(n - 1):
            bigrams.add((words[i], words[i+1]))
        bigram_diversity = len(bigrams) / (n - 1)
    else:
        bigram_diversity = 1.0

    # Dim 7: Sentence rhythm (std dev of token lengths)
    mean_wl = total_char_len / n
    len_variance = sum((len(w) - mean_wl) ** 2 for w in words) / n
    sentence_rhythm = min(math.sqrt(len_variance) / 5.0, 1.0)

    # Dim 8: Vocabulary richness
    if n > 1:
        vocab_richness = min(math.log(len(unique)) / math.log(n), 1.0)
    else:
        vocab_richness = 1.0

    # Dim 9: Length variation ratio
    max_wl = max(word_lengths)
    if mean_wl > 0:
        length_variation = min(max_wl / mean_wl / 5.0, 1.0)
    else:
        length_variation = 0.0

    # Dim 10: Rare word ratio (>=8 chars)
    rare_count = sum(1 for w in words if len(w) >= 8)
    rare_ratio = rare_count / n

    # Dim 11: Clause count proxy
    semicolons = raw_text.count(';')
    clause_proxy = min((comma_count + sub_count + semicolons) / MAX_DEP_DEPTH, 1.0)

    # Dim 12: Academic/Latin-Greek suffix ratio
    # Words ending in suffixes common in academic/complex text
    academic_suffixes = (
        "tion", "sion", "ment", "ness", "ical", "ious", "eous",
        "ology", "istic", "ence", "ance", "ive", "ism", "ist",
        "able", "ible", "ular", "uous", "ity", "phy", "thy",
    )
    acad_count = sum(1 for w in words if any(w.endswith(s) for s in academic_suffixes))
    academic_ratio = acad_count / n

    # Dim 13: Polysyllabic word ratio (4+ syllables = complex vocabulary)
    poly_count = 0
    for w in words:
        syl = 0
        prev_v = False
        for ch in w:
            is_v = ch in "aeiou"
            if is_v and not prev_v:
                syl += 1
            prev_v = is_v
        if syl >= 4:
            poly_count += 1
    polysyllabic_ratio = poly_count / n

    # Dim 14: Character entropy (higher = more diverse character usage = complex)
    char_freq = {}
    total_c = 0
    for w in words:
        for ch in w:
            char_freq[ch] = char_freq.get(ch, 0) + 1
            total_c += 1
    char_entropy = 0.0
    if total_c > 0:
        for count in char_freq.values():
            p = count / total_c
            if p > 0:
                char_entropy -= p * math.log(p)
        # Normalize: max entropy for 26 chars = ln(26) ≈ 3.26
        char_entropy = min(char_entropy / 3.26, 1.0)

    return [
        token_count_norm,   # 0
        ttr,                # 1
        mean_token_len,     # 2
        dep_depth,          # 3
        mean_clause_len,    # 4
        lexical_density,    # 5
        bigram_diversity,   # 6
        sentence_rhythm,    # 7
        vocab_richness,     # 8
        length_variation,   # 9
        rare_ratio,         # 10
        clause_proxy,       # 11
        academic_ratio,     # 12
        polysyllabic_ratio, # 13
        char_entropy,       # 14
    ]


# ─── Group 5: Structural Syntax Features (12 dims) ───

def compute_dependency_depth(words):
    """Stack-based dependency depth proxy (4 dims)."""
    if not words:
        return [0.0, 0.0, 0.0, 0.0]

    stack_depth = 0
    depths = []
    max_depth = 0
    prev_was_pronoun = False
    prep_depth = 0
    max_prep_depth = 0

    for word in words:
        lower = word.lower()

        if lower in G5_SUBORDINATORS:
            stack_depth += 1
        elif lower in (",", ";", "."):
            stack_depth = max(0, stack_depth - 1)
        elif prev_was_pronoun and lower not in G5_SUBJECT_PRONOUNS and stack_depth > 0:
            stack_depth = max(0, stack_depth - 1)

        if lower in G5_PREPOSITIONS:
            prep_depth += 1
        elif len(word) > 6 and lower not in G5_FUNCTION_WORDS:
            prep_depth = max(0, prep_depth - 1)

        max_depth = max(max_depth, stack_depth)
        max_prep_depth = max(max_prep_depth, prep_depth)
        depths.append(float(stack_depth))

        prev_was_pronoun = lower in G5_SUBJECT_PRONOUNS

    n = len(depths)
    mean = sum(depths) / n
    variance = sum((d - mean) ** 2 for d in depths) / n
    std_dev = math.sqrt(variance)

    return [
        min(max_depth / 8.0, 1.0),
        min(mean / 4.0, 1.0),
        min(std_dev / 2.0, 1.0),
        min(max_prep_depth / 6.0, 1.0),
    ]


def compute_constituent_variance(words):
    """Constituent length variance (2 dims)."""
    if not words:
        return [0.0, 0.0]

    segments = []
    current_len = 0

    for word in words:
        lower = word.lower()
        if word in (",", ";", ":") or lower in COORDINATING or lower in G5_SUBORDINATORS:
            if current_len > 0:
                segments.append(current_len)
            current_len = 0
        else:
            current_len += 1

    if current_len > 0:
        segments.append(current_len)

    if len(segments) < 2:
        return [0.0, 0.0]

    mean = sum(segments) / len(segments)
    variance = sum((s - mean) ** 2 for s in segments) / len(segments)
    std_dev = math.sqrt(variance)

    max_seg = max(segments)
    min_seg = min(segments)
    ratio = max_seg / min_seg if min_seg > 0 else float(max_seg)

    return [
        min(std_dev / 15.0, 1.0),
        min(ratio / 10.0, 1.0),
    ]


def compute_function_word_entropy(words):
    """Function word position entropy (5 dims)."""
    n = len(words)
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    positions = []
    for i, w in enumerate(words):
        if w.lower() in G5_FUNCTION_WORDS:
            positions.append(i / n)

    fw_count = len(positions)
    if fw_count == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    mean = sum(positions) / fw_count
    variance = sum((p - mean) ** 2 for p in positions) / fw_count
    std_dev = math.sqrt(variance)

    first_count = sum(1 for p in positions if p < 1.0 / 3.0)
    mid_count = sum(1 for p in positions if 1.0 / 3.0 <= p < 2.0 / 3.0)
    last_count = sum(1 for p in positions if p >= 2.0 / 3.0)
    fw_total = float(fw_count)

    return [
        min(mean, 1.0),
        min(std_dev / 0.4, 1.0),
        min(first_count / fw_total, 1.0),
        min(mid_count / fw_total, 1.0),
        min(last_count / fw_total, 1.0),
    ]


def g5_structural_syntax(words):
    """12 structural syntax features (4 dep + 2 constituent + 5 fw_entropy + 1 pad)."""
    dep = compute_dependency_depth(words)
    con = compute_constituent_variance(words)
    fwe = compute_function_word_entropy(words)
    return dep + con + fwe + [0.0]  # 4 + 2 + 5 + 1 = 12


# ─── Main Encoder ───

def encode(text):
    """
    Encode text to 128-dimensional feature vector.

    Groups:
        G1 (dims 0-25):   Character n-gram profile, amp 3.0
        G2 (dims 26-61):  Syntactic proxy features, amp 3.0
        G3 (dims 62-100): Position-weighted token signal, amp 1.0
        G4 (dims 101-115): Complexity scalars, amp 2.0
        G5 (dims 116-127): Structural syntax features, amp 3.0

    No L2 normalization — magnitude encodes sentence complexity.
    """
    words = tokenize(text)

    g1 = g1_ngram_profile(words)
    g2 = g2_syntactic_features(words, text)
    g3 = g3_token_signal(words)
    g4 = g4_complexity_scalars(words, text)
    g5 = g5_structural_syntax(words)

    # Concatenate with amplification
    features = []
    features.extend(v * G1_AMP for v in g1)
    features.extend(v * G2_AMP for v in g2)
    features.extend(v * G3_AMP for v in g3)
    features.extend(v * G4_AMP for v in g4)
    features.extend(v * G5_AMP for v in g5)

    assert len(features) == OUTPUT_DIM, f"Expected {OUTPUT_DIM}, got {len(features)}"
    return features


# Mark G5 range for the evaluation harness
encode.g5_start = 116
encode.g5_end = 128


# ─── Confidence Function ───

def confidence(encoding, surface_weight, g5_stats):
    """
    Compute Surface-tier confidence for routing decision.

    Formula:
        base_confidence = min(||input||, 1.0)
        cosine = cosine_sim(input, surface_weight)
        confidence = base_conf * 0.7 + cosine * 0.3 - g5_penalty * 0.35

    If confidence >= threshold → route to Surface (simple)
    If confidence < threshold → escalate (complex)
    """
    from prepare import norm, cosine_sim

    cos = cosine_sim(encoding, surface_weight)
    conf = cos

    # Weighted penalty from ALL G4 dims (indices 101-115, amp = G4_AMP)
    # Each weight controls how much that scalar penalizes surface confidence
    g4_penalty_weights = [
        0.0,    # 0: token_count_norm
        -0.05,  # 1: TTR — high unique ratio = complex vocabulary
        0.0,    # 2: mean_token_len
        0.1,    # 3: dep_depth — subordination proxy
        0.0,    # 4: mean_clause_len
        0.0,    # 5: lexical_density
        0.0,    # 6: bigram_diversity
        0.0,    # 7: sentence_rhythm
        0.0,    # 8: vocab_richness
        0.0,    # 9: length_variation
        0.1,    # 10: rare_ratio
        0.1,    # 11: clause_proxy
        0.2,    # 12: academic_ratio
        0.3,    # 13: polysyllabic_ratio
        0.1,    # 14: char_entropy
    ]
    penalty = 0.0
    for i, w in enumerate(g4_penalty_weights):
        if w != 0.0:
            penalty += (encoding[101 + i] / G4_AMP) * w
    # G2 long word ratio: dim 23, index 26+23=49, amp G2_AMP
    long_word_ratio = encoding[49] / G2_AMP
    penalty += long_word_ratio * 0.2
    # G2 initial capital: dim 25, index 26+25=51, amp G2_AMP
    init_cap = encoding[51] / G2_AMP
    penalty += init_cap * 0.15
    conf = max(0.0, conf - max(0.0, penalty))

    return max(0.0, min(1.0, conf))


# ─── Main ───

if __name__ == '__main__':
    result = evaluate(encode, confidence)

    # Output in autoresearch format
    print(f"val_score:          {result['val_score']:.6f}")
    print(f"simple_accuracy:    {result['simple_accuracy']:.4f}")
    print(f"complex_accuracy:   {result['complex_accuracy']:.4f}")
    print(f"simple_correct:     {result['simple_correct']}/{result['simple_total']}")
    print(f"complex_correct:    {result['complex_correct']}/{result['complex_total']}")
    print(f"confidence_gap:     {result['confidence_gap']:.4f}")
    print(f"surface_threshold:  {result['surface_threshold']:.4f}")
    print(f"simple_mean_conf:   {result['simple_mean_conf']:.4f}")
    print(f"complex_mean_conf:  {result['complex_mean_conf']:.4f}")
    print(f"g5_simple_mean:     {result['g5_simple_mean']:.4f}")
    print(f"g5_complex_mean:    {result['g5_complex_mean']:.4f}")
    print(f"moderate_surface:   {result['moderate_surface_pct']:.1%}")
    print(f"training_seconds:   {result['elapsed_seconds']:.1f}")

    # Print misrouted sentences for debugging
    print()
    print("─── Misrouted sentences ───")
    for sentence, label, conf, routed_surface, correct in result['details']:
        if correct is not None and not correct:
            route = "Surface" if routed_surface else "Escalated"
            print(f"  [{label:>7}] conf={conf:.4f} route={route:>9}  {sentence[:80]}")
