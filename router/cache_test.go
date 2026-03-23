package router

import (
	"testing"
)

func TestSemanticCacheExactMatch(t *testing.T) {
	c := NewSemanticCache(CacheConfig{MaxSize: 100, TTLSec: 60, SimilarityThresh: 0.8})

	msgs := []Message{{Role: "user", Content: "What is the capital of France?"}}
	c.Put(msgs, []byte(`{"answer":"Paris"}`), "gpt-4o-mini")

	resp, model, hit := c.Get(msgs)
	if !hit {
		t.Fatal("expected cache hit for exact same query")
	}
	if string(resp) != `{"answer":"Paris"}` {
		t.Errorf("response: got %q", resp)
	}
	if model != "gpt-4o-mini" {
		t.Errorf("model: got %q", model)
	}
}

func TestSemanticCacheSimilarMatch(t *testing.T) {
	c := NewSemanticCache(CacheConfig{MaxSize: 100, TTLSec: 60, SimilarityThresh: 0.5})

	// Store response for one phrasing.
	c.Put(
		[]Message{{Role: "user", Content: "What is the capital city of France?"}},
		[]byte(`{"answer":"Paris"}`),
		"gpt-4o-mini",
	)

	// Query with a different but similar phrasing.
	_, _, hit := c.Get([]Message{{Role: "user", Content: "What is the capital of France?"}})
	if !hit {
		t.Error("expected cache hit for similar query")
	}
}

func TestSemanticCacheDissimilarMiss(t *testing.T) {
	c := NewSemanticCache(CacheConfig{MaxSize: 100, TTLSec: 60, SimilarityThresh: 0.8})

	c.Put(
		[]Message{{Role: "user", Content: "What is the capital of France?"}},
		[]byte(`{"answer":"Paris"}`),
		"gpt-4o-mini",
	)

	// Completely different query should miss.
	_, _, hit := c.Get([]Message{{Role: "user", Content: "How do I write a Go program?"}})
	if hit {
		t.Error("expected cache miss for dissimilar query")
	}
}

func TestSemanticCacheEviction(t *testing.T) {
	c := NewSemanticCache(CacheConfig{MaxSize: 2, TTLSec: 60, SimilarityThresh: 0.8})

	c.Put([]Message{{Role: "user", Content: "query one alpha"}}, []byte("r1"), "m1")
	c.Put([]Message{{Role: "user", Content: "query two beta"}}, []byte("r2"), "m2")
	c.Put([]Message{{Role: "user", Content: "query three gamma"}}, []byte("r3"), "m3")

	// First entry should have been evicted.
	_, _, hit := c.Get([]Message{{Role: "user", Content: "query one alpha"}})
	if hit {
		t.Error("expected eviction of oldest entry")
	}

	// Third entry should still be there.
	_, _, hit = c.Get([]Message{{Role: "user", Content: "query three gamma"}})
	if !hit {
		t.Error("expected newest entry to survive")
	}
}

func TestSemanticCacheStats(t *testing.T) {
	c := NewSemanticCache(CacheConfig{MaxSize: 100, TTLSec: 60, SimilarityThresh: 0.5})

	c.Put([]Message{{Role: "user", Content: "test query here"}}, []byte("r"), "m")
	c.Get([]Message{{Role: "user", Content: "test query here"}})       // hit
	c.Get([]Message{{Role: "user", Content: "completely different"}})   // miss

	hits, misses, size := c.Stats()
	if hits != 1 {
		t.Errorf("hits: got %d, want 1", hits)
	}
	if misses != 1 {
		t.Errorf("misses: got %d, want 1", misses)
	}
	if size != 1 {
		t.Errorf("size: got %d, want 1", size)
	}
}

func TestJaccardSimilarity(t *testing.T) {
	a := map[string]bool{"hello": true, "world": true, "foo": true}
	b := map[string]bool{"hello": true, "world": true, "bar": true}

	sim := jaccardSimilarity(a, b)
	// intersection=2, union=4, sim=0.5
	if sim < 0.49 || sim > 0.51 {
		t.Errorf("Jaccard(%v, %v) = %f, want ~0.5", a, b, sim)
	}
}

func TestJaccardSimilarityIdentical(t *testing.T) {
	a := map[string]bool{"hello": true, "world": true}
	sim := jaccardSimilarity(a, a)
	if sim != 1.0 {
		t.Errorf("identical sets: got %f, want 1.0", sim)
	}
}

func TestJaccardSimilarityDisjoint(t *testing.T) {
	a := map[string]bool{"hello": true}
	b := map[string]bool{"world": true}
	sim := jaccardSimilarity(a, b)
	if sim != 0.0 {
		t.Errorf("disjoint sets: got %f, want 0.0", sim)
	}
}

func TestExtractWords(t *testing.T) {
	words := extractWords("What is the capital of France?")
	if !words["capital"] {
		t.Error("expected 'capital' in word set")
	}
	if !words["france"] {
		t.Error("expected 'france' in word set")
	}
	// "the" and "of" should be filtered (stop words or too short).
	if words["the"] {
		t.Error("'the' should be filtered as stop word")
	}
	if words["of"] {
		t.Error("'of' should be filtered (too short)")
	}
}
