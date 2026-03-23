package router

import (
	"strings"
	"sync"
	"time"
	"unicode"
)

// SemanticCache caches LLM responses and returns them for semantically similar
// queries. Inspired by GPTCache (github.com/zilliztech/GPTCache).
//
// Uses Jaccard similarity on normalized word sets for fast, dependency-free
// similarity matching. No external embedding model required.
//
// For higher-quality matching, an optional embedding sidecar can be configured
// (see EmbeddingSidecar).
type SemanticCache struct {
	mu              sync.RWMutex
	entries         map[string]*cacheEntry // normalized key → entry
	order           []string               // LRU order (most recent last)
	maxSize         int
	ttl             time.Duration
	similarityThresh float64 // 0.0–1.0, higher = stricter matching
	hits            int64
	misses          int64
}

type cacheEntry struct {
	key       string // normalized query
	words     map[string]bool // word set for similarity comparison
	response  []byte
	model     string
	createdAt time.Time
}

// CacheConfig holds semantic cache settings.
type CacheConfig struct {
	MaxSize          int     `yaml:"max_size"`           // max cached entries (default: 1000)
	TTLSec           int     `yaml:"ttl_sec"`            // entry TTL in seconds (default: 300)
	SimilarityThresh float64 `yaml:"similarity_thresh"`  // Jaccard threshold 0-1 (default: 0.85)
	Enabled          bool    `yaml:"enabled"`            // enable/disable cache
}

// NewSemanticCache creates a cache with the given config.
func NewSemanticCache(cfg CacheConfig) *SemanticCache {
	maxSize := cfg.MaxSize
	if maxSize == 0 {
		maxSize = 1000
	}
	ttl := time.Duration(cfg.TTLSec) * time.Second
	if ttl == 0 {
		ttl = 5 * time.Minute
	}
	thresh := cfg.SimilarityThresh
	if thresh == 0 {
		thresh = 0.85
	}

	return &SemanticCache{
		entries:          make(map[string]*cacheEntry, maxSize),
		order:            make([]string, 0, maxSize),
		maxSize:          maxSize,
		ttl:              ttl,
		similarityThresh: thresh,
	}
}

// Get looks up a semantically similar cached response.
// Returns (response, model, hit). If hit is false, no similar entry was found.
func (c *SemanticCache) Get(messages []Message) ([]byte, string, bool) {
	queryWords := extractWords(messagesToText(messages))
	if len(queryWords) == 0 {
		return nil, "", false
	}

	c.mu.RLock()
	defer c.mu.RUnlock()

	now := time.Now()
	var bestEntry *cacheEntry
	bestSim := 0.0

	for _, entry := range c.entries {
		// Skip expired entries.
		if now.Sub(entry.createdAt) > c.ttl {
			continue
		}

		sim := jaccardSimilarity(queryWords, entry.words)
		if sim > bestSim && sim >= c.similarityThresh {
			bestSim = sim
			bestEntry = entry
		}
	}

	if bestEntry != nil {
		c.mu.RUnlock()
		c.mu.Lock()
		c.hits++
		c.mu.Unlock()
		c.mu.RLock()
		return bestEntry.response, bestEntry.model, true
	}

	c.mu.RUnlock()
	c.mu.Lock()
	c.misses++
	c.mu.Unlock()
	c.mu.RLock()
	return nil, "", false
}

// Put stores a response in the cache.
func (c *SemanticCache) Put(messages []Message, response []byte, model string) {
	text := messagesToText(messages)
	words := extractWords(text)
	if len(words) == 0 {
		return
	}

	key := normalizeText(text)

	c.mu.Lock()
	defer c.mu.Unlock()

	// Evict oldest if at capacity.
	for len(c.entries) >= c.maxSize && len(c.order) > 0 {
		oldest := c.order[0]
		c.order = c.order[1:]
		delete(c.entries, oldest)
	}

	c.entries[key] = &cacheEntry{
		key:       key,
		words:     words,
		response:  response,
		model:     model,
		createdAt: time.Now(),
	}
	c.order = append(c.order, key)
}

// Stats returns cache hit/miss counts.
func (c *SemanticCache) Stats() (hits, misses int64, size int) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.hits, c.misses, len(c.entries)
}

// jaccardSimilarity computes |A ∩ B| / |A ∪ B| between two word sets.
func jaccardSimilarity(a, b map[string]bool) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}

	intersection := 0
	for word := range a {
		if b[word] {
			intersection++
		}
	}

	union := len(a) + len(b) - intersection
	if union == 0 {
		return 0.0
	}
	return float64(intersection) / float64(union)
}

// extractWords tokenizes and normalizes text into a word set.
func extractWords(text string) map[string]bool {
	words := make(map[string]bool)
	for _, word := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		// Skip very short words and common stop words.
		if len(word) <= 2 || stopWords[word] {
			continue
		}
		words[word] = true
	}
	return words
}

// normalizeText produces a canonical form for exact-match keying.
func normalizeText(text string) string {
	words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	return strings.Join(words, " ")
}

// messagesToText extracts the text content from chat messages.
func messagesToText(messages []Message) string {
	var parts []string
	for _, m := range messages {
		if m.Role == "user" || m.Role == "system" {
			parts = append(parts, m.Content)
		}
	}
	return strings.Join(parts, " ")
}

// stopWords are common English words excluded from similarity comparison.
var stopWords = map[string]bool{
	"the": true, "and": true, "for": true, "are": true, "but": true,
	"not": true, "you": true, "all": true, "can": true, "her": true,
	"was": true, "one": true, "our": true, "out": true, "has": true,
	"have": true, "this": true, "that": true, "with": true, "from": true,
	"they": true, "been": true, "said": true, "each": true, "which": true,
	"their": true, "will": true, "other": true, "about": true, "many": true,
	"then": true, "them": true, "these": true, "some": true, "would": true,
	"make": true, "like": true, "into": true, "could": true, "what": true,
	"does": true, "just": true, "than": true, "when": true, "there": true,
	"also": true, "how": true, "please": true, "help": true,
}
