package router

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestDedupSingleRequest(t *testing.T) {
	d := NewDeduplicator(5 * time.Second)

	result, deduped, err := d.Do("key1", func() ([]byte, error) {
		return []byte("hello"), nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if deduped {
		t.Error("first request should not be deduped")
	}
	if string(result) != "hello" {
		t.Errorf("result: got %q, want 'hello'", result)
	}
}

func TestDedupCoalescesIdentical(t *testing.T) {
	d := NewDeduplicator(5 * time.Second)

	var callCount atomic.Int32
	var wg sync.WaitGroup

	fn := func() ([]byte, error) {
		callCount.Add(1)
		time.Sleep(100 * time.Millisecond) // simulate upstream latency
		return []byte("response"), nil
	}

	// Launch 5 concurrent requests with the same key.
	results := make([][]byte, 5)
	deduped := make([]bool, 5)
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			r, d, _ := d.Do("same-key", fn)
			results[idx] = r
			deduped[idx] = d
		}(i)
		// Small stagger so they all see the in-flight request.
		time.Sleep(10 * time.Millisecond)
	}

	wg.Wait()

	// Should have only called fn once.
	if c := callCount.Load(); c != 1 {
		t.Errorf("fn called %d times, want 1", c)
	}

	// All should get the same result.
	for i, r := range results {
		if string(r) != "response" {
			t.Errorf("result[%d]: got %q, want 'response'", i, r)
		}
	}

	// At least 4 should be deduped.
	dedupCount := 0
	for _, d := range deduped {
		if d {
			dedupCount++
		}
	}
	if dedupCount < 4 {
		t.Errorf("deduped count: got %d, want >= 4", dedupCount)
	}
}

func TestDedupDifferentKeysNotCoalesced(t *testing.T) {
	d := NewDeduplicator(5 * time.Second)

	var callCount atomic.Int32
	var wg sync.WaitGroup

	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			d.Do(fmt.Sprintf("key-%d", idx), func() ([]byte, error) {
				callCount.Add(1)
				time.Sleep(50 * time.Millisecond)
				return []byte("ok"), nil
			})
		}(i)
	}

	wg.Wait()

	if c := callCount.Load(); c != 3 {
		t.Errorf("fn called %d times, want 3 (different keys)", c)
	}
}

func TestDedupStats(t *testing.T) {
	d := NewDeduplicator(5 * time.Second)

	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			d.Do("same", func() ([]byte, error) {
				time.Sleep(50 * time.Millisecond)
				return []byte("ok"), nil
			})
		}()
		time.Sleep(5 * time.Millisecond)
	}

	wg.Wait()

	hits, misses := d.Stats()
	if misses != 1 {
		t.Errorf("misses: got %d, want 1", misses)
	}
	if hits != 2 {
		t.Errorf("hits: got %d, want 2", hits)
	}
}

func TestHashRequestDeterministic(t *testing.T) {
	req := &ChatRequest{
		Messages: []Message{{Role: "user", Content: "hello"}},
	}

	h1 := HashRequest(req)
	h2 := HashRequest(req)
	if h1 != h2 {
		t.Errorf("hash not deterministic: %q != %q", h1, h2)
	}
}

func TestHashRequestDifferentContent(t *testing.T) {
	req1 := &ChatRequest{Messages: []Message{{Role: "user", Content: "hello"}}}
	req2 := &ChatRequest{Messages: []Message{{Role: "user", Content: "world"}}}

	if HashRequest(req1) == HashRequest(req2) {
		t.Error("different requests should have different hashes")
	}
}
