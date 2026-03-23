package router

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"sync"
	"time"
)

// Deduplicator coalesces identical in-flight requests so that only one
// upstream call is made. Inspired by ParrotServe's request batching
// (OSDI'24: "Parrot: Efficient Serving of LLM-based Applications with
// Semantic Variable") and kv.run's task merging.
//
// When request A arrives, we hash it. If an identical request B is already
// in flight, B waits for A's result instead of making a second upstream call.
type Deduplicator struct {
	mu       sync.Mutex
	inflight map[string]*pendingResult
	window   time.Duration // max time to wait for a duplicate to resolve
	hits     int64
	misses   int64
}

type pendingResult struct {
	done   chan struct{}
	result []byte
	err    error
	count  int // how many callers are waiting
}

// NewDeduplicator creates a deduplicator with the given coalescing window.
func NewDeduplicator(window time.Duration) *Deduplicator {
	return &Deduplicator{
		inflight: make(map[string]*pendingResult),
		window:   window,
	}
}

// Do executes fn only if no identical request (by key) is already in flight.
// If a duplicate is in flight, it waits for that result instead.
// Returns (result bytes, was-deduped, error).
func (d *Deduplicator) Do(key string, fn func() ([]byte, error)) ([]byte, bool, error) {
	d.mu.Lock()

	if pending, ok := d.inflight[key]; ok {
		// Another request with the same key is in flight — wait for it.
		pending.count++
		d.hits++
		d.mu.Unlock()

		// Wait for the original to finish.
		select {
		case <-pending.done:
			return pending.result, true, pending.err
		case <-time.After(d.window):
			// Timeout — fall through and make our own request.
			result, err := fn()
			return result, false, err
		}
	}

	// We're the first with this key — register and execute.
	pending := &pendingResult{
		done:  make(chan struct{}),
		count: 1,
	}
	d.inflight[key] = pending
	d.misses++
	d.mu.Unlock()

	// Execute the actual request.
	result, err := fn()

	// Store result and signal all waiters.
	pending.result = result
	pending.err = err
	close(pending.done)

	// Clean up.
	d.mu.Lock()
	delete(d.inflight, key)
	d.mu.Unlock()

	return result, false, err
}

// Stats returns dedup hit/miss counts.
func (d *Deduplicator) Stats() (hits, misses int64) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.hits, d.misses
}

// HashRequest produces a deterministic hash of a ChatRequest for dedup keying.
// Only hashes the semantic content (messages, tools, response_format) — not
// ephemeral fields like stream or temperature.
func HashRequest(req *ChatRequest) string {
	// Build a canonical representation.
	canonical := struct {
		Messages       []Message       `json:"m"`
		Tools          []Tool          `json:"t,omitempty"`
		ResponseFormat *ResponseFormat  `json:"rf,omitempty"`
	}{
		Messages:       req.Messages,
		Tools:          req.Tools,
		ResponseFormat: req.ResponseFormat,
	}

	data, _ := json.Marshal(canonical)
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:16]) // 128-bit is plenty for dedup
}
