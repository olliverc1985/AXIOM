use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub const PAD_TOKEN: usize = 0;
pub const UNK_TOKEN: usize = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    pub word_to_id: HashMap<String, usize>,
    pub id_to_word: Vec<String>,
    pub max_size: usize,
}

impl Vocab {
    pub fn new(max_size: usize) -> Self {
        let id_to_word = vec!["<PAD>".to_string(), "<UNK>".to_string()];
        let mut word_to_id = HashMap::new();
        word_to_id.insert("<PAD>".to_string(), PAD_TOKEN);
        word_to_id.insert("<UNK>".to_string(), UNK_TOKEN);
        Self {
            word_to_id,
            id_to_word,
            max_size,
        }
    }

    pub fn build_from_corpus(sentences: &[String], max_size: usize) -> Self {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for sentence in sentences {
            for word in Self::split(sentence) {
                *counts.entry(word).or_insert(0) += 1;
            }
        }
        let mut sorted: Vec<(String, usize)> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));

        let mut vocab = Self::new(max_size);
        for (word, _) in sorted.into_iter().take(max_size - 2) {
            let id = vocab.id_to_word.len();
            if id >= max_size {
                break;
            }
            vocab.word_to_id.insert(word.clone(), id);
            vocab.id_to_word.push(word);
        }
        vocab
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        Self::split(text)
            .into_iter()
            .map(|w| *self.word_to_id.get(&w).unwrap_or(&UNK_TOKEN))
            .collect()
    }

    pub fn size(&self) -> usize {
        self.id_to_word.len()
    }

    /// Split text into lowercase tokens — matches existing tokeniser pattern.
    pub fn split(text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        let mut tokens = Vec::new();
        let mut current = String::new();
        for ch in lower.chars() {
            if ch.is_alphanumeric() || ch == '\'' {
                current.push(ch);
            } else if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_special_tokens() {
        let vocab = Vocab::new(100);
        assert_eq!(vocab.size(), 2);
        assert_eq!(vocab.word_to_id["<PAD>"], PAD_TOKEN);
        assert_eq!(vocab.word_to_id["<UNK>"], UNK_TOKEN);
    }

    #[test]
    fn test_vocab_build_and_tokenize() {
        let sentences = vec![
            "the cat sat on the mat".to_string(),
            "the dog ran in the park".to_string(),
        ];
        let vocab = Vocab::build_from_corpus(&sentences, 100);
        assert!(vocab.size() > 2);
        let ids = vocab.tokenize("the cat");
        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0], UNK_TOKEN);
    }

    #[test]
    fn test_unknown_token() {
        let vocab = Vocab::new(100);
        let ids = vocab.tokenize("hello world");
        assert!(ids.iter().all(|&id| id == UNK_TOKEN));
    }
}
