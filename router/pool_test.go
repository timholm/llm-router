package router

import (
	"testing"
	"time"
)

func testModels() []ModelConfig {
	return []ModelConfig{
		{Name: "cheap", Provider: "openai", Model: "gpt-4o-mini", Tier: 1, CostPer1KIn: 0.00015, CostPer1KOut: 0.0006},
		{Name: "mid", Provider: "openai", Model: "gpt-4o", Tier: 2, CostPer1KIn: 0.005, CostPer1KOut: 0.015},
		{Name: "expensive", Provider: "openai", Model: "o1", Tier: 3, CostPer1KIn: 0.015, CostPer1KOut: 0.06},
	}
}

func TestPoolSelectModelBasic(t *testing.T) {
	p := NewPool(testModels())

	req := &ChatRequest{Messages: []Message{{Role: "user", Content: "hi"}}}
	model := p.SelectModel(1, req)
	if model == nil {
		t.Fatal("SelectModel returned nil")
	}
	if model.Name != "cheap" {
		t.Errorf("tier 1: got %q, want 'cheap'", model.Name)
	}
}

func TestPoolSelectModelTier3(t *testing.T) {
	p := NewPool(testModels())

	model := p.SelectModel(3, nil)
	if model == nil {
		t.Fatal("SelectModel returned nil")
	}
	if model.Name != "expensive" {
		t.Errorf("tier 3: got %q, want 'expensive'", model.Name)
	}
}

func TestPoolSelectModelSkipsDown(t *testing.T) {
	p := NewPool(testModels())

	// Mark cheap as down.
	p.MarkDown("cheap")

	req := &ChatRequest{Messages: []Message{{Role: "user", Content: "hi"}}}
	model := p.SelectModel(1, req)
	if model == nil {
		t.Fatal("SelectModel returned nil")
	}
	// Should skip cheap (down) and pick mid (tier 2 >= tier 1).
	if model.Name == "cheap" {
		t.Error("should not select a down backend")
	}
}

func TestPoolSelectModelPrefersLowInFlight(t *testing.T) {
	// Two models at the same tier.
	models := []ModelConfig{
		{Name: "a", Tier: 1, CostPer1KIn: 0.001},
		{Name: "b", Tier: 1, CostPer1KIn: 0.001},
	}
	p := NewPool(models)

	// Add load to "a".
	p.TrackRequest("a")
	p.TrackRequest("a")
	p.TrackRequest("a")

	model := p.SelectModel(1, nil)
	if model == nil {
		t.Fatal("SelectModel returned nil")
	}
	if model.Name != "b" {
		t.Errorf("should prefer 'b' (0 inflight) over 'a' (3 inflight), got %q", model.Name)
	}
}

func TestPoolTrackAndComplete(t *testing.T) {
	p := NewPool(testModels())

	p.TrackRequest("cheap")
	p.TrackRequest("cheap")

	snap := p.Snapshot()
	for _, s := range snap {
		if s.Name == "cheap" {
			if s.InFlight != 2 {
				t.Errorf("inflight: got %d, want 2", s.InFlight)
			}
			if s.TotalReqs != 2 {
				t.Errorf("total_reqs: got %d, want 2", s.TotalReqs)
			}
		}
	}

	p.CompleteRequest("cheap", 100*time.Millisecond, nil)

	snap = p.Snapshot()
	for _, s := range snap {
		if s.Name == "cheap" {
			if s.InFlight != 1 {
				t.Errorf("after complete, inflight: got %d, want 1", s.InFlight)
			}
		}
	}
}

func TestPoolMarkDegraded(t *testing.T) {
	p := NewPool(testModels())

	p.MarkDegraded("mid")

	snap := p.Snapshot()
	for _, s := range snap {
		if s.Name == "mid" {
			if s.Status != "degraded" {
				t.Errorf("status: got %q, want 'degraded'", s.Status)
			}
		}
	}

	// Degraded backends should still be selectable.
	model := p.SelectModel(2, nil)
	if model == nil {
		t.Fatal("SelectModel returned nil for degraded backend")
	}
}

func TestPoolSnapshot(t *testing.T) {
	p := NewPool(testModels())

	snap := p.Snapshot()
	if len(snap) != 3 {
		t.Errorf("snapshot: got %d backends, want 3", len(snap))
	}

	// All should be healthy initially.
	for _, s := range snap {
		if s.Status != "healthy" {
			t.Errorf("%s: status=%q, want 'healthy'", s.Name, s.Status)
		}
	}
}

func TestPoolExplicitModel(t *testing.T) {
	p := NewPool(testModels())

	req := &ChatRequest{
		Model:    "expensive",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}
	model := p.SelectModel(1, req)
	if model == nil {
		t.Fatal("SelectModel returned nil")
	}
	if model.Name != "expensive" {
		t.Errorf("explicit model: got %q, want 'expensive'", model.Name)
	}
}
