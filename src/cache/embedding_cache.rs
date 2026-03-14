//! Content-addressable embedding cache with cosine similarity lookup and LRU eviction.

use crate::tiers::Tier;
use crate::Tensor;

/// A single cache entry storing key/value tensors and access metadata.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The key tensor used for similarity lookup.
    pub key: Tensor,
    /// The cached value tensor.
    pub value: Tensor,
    /// Number of times this entry has been hit.
    pub hit_count: u64,
    /// Monotonic access counter (higher = more recently accessed).
    pub last_accessed: u64,
    /// ID of the compute node that produced this entry's value.
    pub producer_node_id: Option<String>,
    /// Which tier resolved this entry.
    pub resolved_tier: Option<Tier>,
}

/// Content-addressable embedding cache.
///
/// Stores (key, value) tensor pairs and retrieves them by cosine similarity.
/// Uses LRU eviction when the cache exceeds its maximum size.
pub struct EmbeddingCache {
    entries: Vec<CacheEntry>,
    /// Cosine similarity threshold for a cache hit (default 0.92).
    pub similarity_threshold: f32,
    /// Maximum number of entries before LRU eviction.
    pub max_entries: usize,
    /// Monotonic counter for tracking access recency.
    access_counter: u64,
    /// Total hits across the lifetime of this cache.
    pub total_hits: u64,
    /// Total lookups across the lifetime of this cache.
    pub total_lookups: u64,
}

impl EmbeddingCache {
    /// Create a new cache with the given max size and similarity threshold.
    pub fn new(max_entries: usize, similarity_threshold: f32) -> Self {
        Self {
            entries: Vec::new(),
            similarity_threshold,
            max_entries,
            access_counter: 0,
            total_hits: 0,
            total_lookups: 0,
        }
    }

    /// Create a cache with default settings (256 entries, 0.92 threshold).
    pub fn default_cache() -> Self {
        Self::new(256, 0.92)
    }

    /// Look up the most similar cached entry to the input key.
    ///
    /// Returns `Some((value, similarity, producer_node_id))` if a match above threshold is found.
    fn lookup(&mut self, key: &Tensor) -> Option<(Tensor, f32, Option<String>)> {
        let mut best_sim = -1.0_f32;
        let mut best_idx: Option<usize> = None;

        for (i, entry) in self.entries.iter().enumerate() {
            if entry.key.data.len() != key.data.len() {
                continue;
            }
            let sim = key.cosine_similarity(&entry.key);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        if best_sim >= self.similarity_threshold {
            if let Some(idx) = best_idx {
                self.access_counter += 1;
                self.entries[idx].hit_count += 1;
                self.entries[idx].last_accessed = self.access_counter;
                return Some((
                    self.entries[idx].value.clone(),
                    best_sim,
                    self.entries[idx].producer_node_id.clone(),
                ));
            }
        }

        None
    }

    /// Insert a new entry, evicting the LRU entry if at capacity.
    fn insert(&mut self, key: Tensor, value: Tensor, producer_node_id: Option<String>, resolved_tier: Option<Tier>) {
        if self.entries.len() >= self.max_entries {
            // Evict least recently used
            let lru_idx = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.entries.swap_remove(lru_idx);
        }

        self.access_counter += 1;
        self.entries.push(CacheEntry {
            key,
            value,
            hit_count: 0,
            last_accessed: self.access_counter,
            producer_node_id,
            resolved_tier,
        });
    }

    /// Get a cached result or compute and cache it.
    ///
    /// Returns `(result_tensor, was_cache_hit)`.
    pub fn get_or_compute(
        &mut self,
        input: &Tensor,
        compute_fn: impl Fn(&Tensor) -> Tensor,
    ) -> (Tensor, bool) {
        self.total_lookups += 1;

        if let Some((cached_value, _similarity, _producer)) = self.lookup(input) {
            self.total_hits += 1;
            return (cached_value, true);
        }

        // Cache miss — compute, store, and return
        let result = compute_fn(input);
        self.insert(input.clone(), result.clone(), None, None);
        (result, false)
    }

    /// Current number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache hit rate as a fraction [0.0, 1.0].
    pub fn hit_rate(&self) -> f32 {
        if self.total_lookups == 0 {
            return 0.0;
        }
        self.total_hits as f32 / self.total_lookups as f32
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_hits = 0;
        self.total_lookups = 0;
        self.access_counter = 0;
    }

    /// Lookup only — returns cached value if found, without computing.
    ///
    /// Does not update total_lookups/total_hits counters (caller manages those).
    /// Returns `Some((value, similarity, producer_node_id))`.
    pub fn lookup_only(&mut self, key: &Tensor) -> Option<(Tensor, f32, Option<String>)> {
        self.lookup(key)
    }

    /// Insert a key/value pair directly into the cache with optional producer node ID and tier.
    pub fn insert_direct(
        &mut self,
        key: Tensor,
        value: Tensor,
        producer_node_id: Option<String>,
        resolved_tier: Option<Tier>,
    ) {
        self.insert(key, value, producer_node_id, resolved_tier);
    }

    /// Find the best-matching cache entry with cosine similarity above `threshold`
    /// that was resolved at the given tier. Does not update access counters.
    ///
    /// Returns `(similarity, producer_node_id)` if found.
    pub fn find_similar_at_tier(
        &self,
        key: &Tensor,
        threshold: f32,
        tier: Tier,
    ) -> Option<(f32, Option<String>)> {
        let mut best_sim = -1.0_f32;
        let mut best_idx: Option<usize> = None;

        for (i, entry) in self.entries.iter().enumerate() {
            if entry.key.data.len() != key.data.len() {
                continue;
            }
            if entry.resolved_tier != Some(tier) {
                continue;
            }
            let sim = key.cosine_similarity(&entry.key);
            if sim > best_sim {
                best_sim = sim;
                best_idx = Some(i);
            }
        }

        if best_sim >= threshold {
            best_idx.map(|idx| (best_sim, self.entries[idx].producer_node_id.clone()))
        } else {
            None
        }
    }

    /// Find the highest cosine similarity to any cached entry at the given tier,
    /// regardless of threshold. Returns `(best_similarity, resolved_tier_of_nearest)`
    /// across ALL entries (not just the target tier) for diagnostic purposes.
    ///
    /// Returns `None` if the cache is empty.
    pub fn best_similarity_diagnostic(
        &self,
        key: &Tensor,
        target_tier: Tier,
    ) -> Option<(f32, Option<Tier>)> {
        if self.entries.is_empty() {
            return None;
        }

        let mut best_sim_at_tier = -1.0_f32;
        let mut best_sim_any = -1.0_f32;
        let mut best_any_tier: Option<Tier> = None;

        for entry in &self.entries {
            if entry.key.data.len() != key.data.len() {
                continue;
            }
            let sim = key.cosine_similarity(&entry.key);
            if sim > best_sim_any {
                best_sim_any = sim;
                best_any_tier = entry.resolved_tier;
            }
            if entry.resolved_tier == Some(target_tier) && sim > best_sim_at_tier {
                best_sim_at_tier = sim;
            }
        }

        // Return best sim against Surface entries, with nearest-overall tier for context
        if best_sim_at_tier > -1.0 {
            Some((best_sim_at_tier, best_any_tier))
        } else {
            // No Surface entries in cache at all — return overall best
            Some((best_sim_any, best_any_tier))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_miss_then_hit() {
        let mut cache = EmbeddingCache::new(10, 0.92);

        let key = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let compute = |_: &Tensor| Tensor::from_vec(vec![42.0]);

        // First call: cache miss
        let (result, hit) = cache.get_or_compute(&key, compute);
        assert!(!hit);
        assert_eq!(result.data, vec![42.0]);

        // Second call with identical key: cache hit
        let (result, hit) = cache.get_or_compute(&key, |_| Tensor::from_vec(vec![999.0]));
        assert!(hit);
        assert_eq!(result.data, vec![42.0]); // Returns cached value, not 999
    }

    #[test]
    fn test_cache_similar_key_hit() {
        let mut cache = EmbeddingCache::new(10, 0.92);

        let key1 = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let key2 = Tensor::from_vec(vec![0.99, 0.01, 0.0, 0.0]); // Very similar

        let (_, hit) = cache.get_or_compute(&key1, |_| Tensor::from_vec(vec![42.0]));
        assert!(!hit);

        let (result, hit) = cache.get_or_compute(&key2, |_| Tensor::from_vec(vec![999.0]));
        assert!(hit);
        assert_eq!(result.data, vec![42.0]);
    }

    #[test]
    fn test_cache_dissimilar_key_miss() {
        let mut cache = EmbeddingCache::new(10, 0.92);

        let key1 = Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let key2 = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0]); // Orthogonal

        let (_, hit) = cache.get_or_compute(&key1, |_| Tensor::from_vec(vec![42.0]));
        assert!(!hit);

        let (_, hit) = cache.get_or_compute(&key2, |_| Tensor::from_vec(vec![99.0]));
        assert!(!hit); // Should be a miss — keys are orthogonal
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = EmbeddingCache::new(3, 0.99); // Very strict threshold

        // Fill cache with 3 entries (all orthogonal so no hits)
        for i in 0..3 {
            let mut data = vec![0.0; 4];
            data[i] = 1.0;
            let key = Tensor::from_vec(data);
            cache.get_or_compute(&key, |_| Tensor::from_vec(vec![i as f32]));
        }
        assert_eq!(cache.len(), 3);

        // Adding a 4th should evict the LRU
        let key4 = Tensor::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        cache.get_or_compute(&key4, |_| Tensor::from_vec(vec![99.0]));
        assert_eq!(cache.len(), 3); // Still 3 after eviction
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = EmbeddingCache::new(10, 0.92);

        let key = Tensor::from_vec(vec![1.0, 0.0, 0.0]);
        let compute = |_: &Tensor| Tensor::from_vec(vec![1.0]);

        cache.get_or_compute(&key, compute); // miss
        cache.get_or_compute(&key, compute); // hit
        cache.get_or_compute(&key, compute); // hit

        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 0.01);
    }
}
