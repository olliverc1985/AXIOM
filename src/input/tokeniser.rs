//! Simple whitespace + punctuation tokeniser with vocabulary building.
//!
//! No external NLP dependencies. Lowercases input, splits on whitespace and
//! punctuation, builds a vocabulary from observed tokens (max 1024).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A simple rule-based tokeniser.
///
/// Lowercases input, splits on whitespace and punctuation,
/// assigns integer IDs to tokens. Vocabulary capped at `max_vocab`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tokeniser {
    /// Token string → integer ID mapping.
    pub vocab: HashMap<String, usize>,
    /// Reverse mapping: ID → token string.
    pub id_to_token: Vec<String>,
    /// Maximum vocabulary size.
    pub max_vocab: usize,
}

impl Tokeniser {
    /// Create a new tokeniser with the given max vocabulary size.
    pub fn new(max_vocab: usize) -> Self {
        Self {
            vocab: HashMap::new(),
            id_to_token: Vec::new(),
            max_vocab,
        }
    }

    /// Create a tokeniser with default max vocabulary (1024).
    pub fn default_tokeniser() -> Self {
        Self::new(1024)
    }

    /// Tokenise a string into a sequence of token IDs.
    ///
    /// Lowercases, splits on non-alphanumeric characters, and assigns IDs.
    /// New tokens are added to the vocabulary until `max_vocab` is reached;
    /// unknown tokens beyond that get ID 0 (mapped to zero vector by encoder).
    pub fn tokenise(&mut self, text: &str) -> Vec<usize> {
        let tokens = Self::split(text);
        tokens
            .into_iter()
            .map(|t| self.get_or_insert(&t))
            .collect()
    }

    /// Tokenise without modifying vocabulary (read-only).
    pub fn tokenise_readonly(&self, text: &str) -> Vec<usize> {
        let tokens = Self::split(text);
        tokens
            .into_iter()
            .map(|t| *self.vocab.get(&t).unwrap_or(&0))
            .collect()
    }

    /// Split text into lowercase token strings.
    ///
    /// Public so the encoder can extract syntactic features from raw token strings.
    pub fn split(text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        let mut tokens = Vec::new();
        let mut current = String::new();

        for ch in lower.chars() {
            if ch.is_alphanumeric() || ch == '\'' {
                current.push(ch);
            } else {
                if !current.is_empty() {
                    tokens.push(std::mem::take(&mut current));
                }
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }

    /// Get or create a token ID.
    fn get_or_insert(&mut self, token: &str) -> usize {
        if let Some(&id) = self.vocab.get(token) {
            return id;
        }
        if self.id_to_token.len() >= self.max_vocab {
            return 0; // Unknown token
        }
        let id = self.id_to_token.len();
        self.id_to_token.push(token.to_string());
        self.vocab.insert(token.to_string(), id);
        id
    }

    /// Current vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Save vocabulary to JSON file.
    pub fn save_vocab(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.vocab)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load vocabulary from JSON file.
    pub fn load_vocab(&mut self, path: &str) -> std::io::Result<()> {
        let data = std::fs::read_to_string(path)?;
        let vocab: HashMap<String, usize> = serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        // Rebuild id_to_token from vocab
        let max_id = vocab.values().copied().max().unwrap_or(0);
        let mut id_to_token = vec![String::new(); max_id + 1];
        for (token, &id) in &vocab {
            id_to_token[id] = token.clone();
        }
        self.vocab = vocab;
        self.id_to_token = id_to_token;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenise() {
        let mut tok = Tokeniser::new(100);
        let ids = tok.tokenise("Hello World");
        assert_eq!(ids.len(), 2);
        // "hello" and "world" should get different IDs
        assert_ne!(ids[0], ids[1]);
    }

    #[test]
    fn test_lowercase() {
        let mut tok = Tokeniser::new(100);
        let ids1 = tok.tokenise("Hello");
        let ids2 = tok.tokenise("hello");
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn test_punctuation_split() {
        let mut tok = Tokeniser::new(100);
        let ids = tok.tokenise("hello, world! how's it going?");
        // "hello", "world", "how's", "it", "going"
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn test_vocab_limit() {
        let mut tok = Tokeniser::new(3);
        tok.tokenise("alpha beta gamma");
        assert_eq!(tok.vocab_size(), 3);
        let ids = tok.tokenise("delta"); // Should map to 0 (unknown)
        assert_eq!(ids[0], 0);
    }

    #[test]
    fn test_consistent_ids() {
        let mut tok = Tokeniser::new(100);
        tok.tokenise("the cat sat");
        let ids = tok.tokenise("the cat sat on the mat");
        // "the" should have same ID both times
        assert_eq!(ids[0], ids[4]); // both "the"
    }
}
