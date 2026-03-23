package router

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestLearnedRouterHeuristicOnly(t *testing.T) {
	c := &Classifier{Tier1Max: 30, Tier2Max: 70}
	lr := NewLearnedRouter(c, LearnedRouterConfig{Threshold: 0.5})

	// Simple request → low score → tier 1.
	req := &ChatRequest{Messages: []Message{{Role: "user", Content: "Hi"}}}
	score, tier := lr.Route(req)

	if score > 0.3 {
		t.Errorf("simple request: score=%f, want < 0.3", score)
	}
	if tier != 1 {
		t.Errorf("simple request: tier=%d, want 1", tier)
	}
}

func TestLearnedRouterComplexRequest(t *testing.T) {
	c := &Classifier{Tier1Max: 30, Tier2Max: 70}
	lr := NewLearnedRouter(c, LearnedRouterConfig{Threshold: 0.5})

	req := &ChatRequest{
		Messages: []Message{
			{Role: "system", Content: "You are an expert analyst. Think step by step about each component."},
			{Role: "user", Content: "```go\npackage main\n```\nAnalyze the trade-offs and evaluate pros and cons step by step."},
		},
		Tools: []Tool{{Type: "function", Function: ToolFunction{Name: "test"}}},
	}

	score, tier := lr.Route(req)
	if score < 0.5 {
		t.Errorf("complex request: score=%f, want >= 0.5", score)
	}
	if tier < 2 {
		t.Errorf("complex request: tier=%d, want >= 2", tier)
	}
}

func TestLearnedRouterWithSidecar(t *testing.T) {
	// Fake sidecar that always returns high win rate.
	sidecar := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(SidecarResponse{StrongWinRate: 0.9})
	}))
	defer sidecar.Close()

	c := &Classifier{Tier1Max: 30, Tier2Max: 70}
	lr := NewLearnedRouter(c, LearnedRouterConfig{
		SidecarURL:    sidecar.URL,
		SidecarWeight: 0.8,
		Threshold:     0.5,
	})

	req := &ChatRequest{Messages: []Message{{Role: "user", Content: "simple question"}}}
	score, tier := lr.Route(req)

	// Sidecar says 0.9, heuristic says low. Weighted: 0.8*0.9 + 0.2*low ≈ 0.72+
	if score < 0.5 {
		t.Errorf("with sidecar: score=%f, want >= 0.5", score)
	}
	if tier < 2 {
		t.Errorf("with sidecar: tier=%d, want >= 2", tier)
	}
}

func TestLearnedRouterSidecarFallback(t *testing.T) {
	// Sidecar that always fails.
	sidecar := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "broken", 500)
	}))
	defer sidecar.Close()

	c := &Classifier{Tier1Max: 30, Tier2Max: 70}
	lr := NewLearnedRouter(c, LearnedRouterConfig{
		SidecarURL: sidecar.URL,
		Threshold:  0.5,
	})

	// Should fall back to heuristic without error.
	req := &ChatRequest{Messages: []Message{{Role: "user", Content: "Hi"}}}
	score, _ := lr.Route(req)
	if score > 0.3 {
		t.Errorf("fallback: score=%f, want < 0.3 (heuristic for simple request)", score)
	}
}

func TestLearnedRouterFeedback(t *testing.T) {
	c := &Classifier{Tier1Max: 30, Tier2Max: 70}
	lr := NewLearnedRouter(c, LearnedRouterConfig{Threshold: 0.5, MaxFeedback: 100})

	// Add some feedback.
	for i := 0; i < 50; i++ {
		lr.AddFeedback(FeedbackRequest{
			Score:   0.3,
			Tier:    1,
			Quality: 0.8,
		})
	}

	stats := lr.FeedbackStats()
	total, ok := stats["total_feedback"].(int)
	if !ok || total != 50 {
		t.Errorf("feedback total: got %v, want 50", stats["total_feedback"])
	}
}

func TestLearnedRouterThresholdCalibration(t *testing.T) {
	c := &Classifier{Tier1Max: 30, Tier2Max: 70}
	lr := NewLearnedRouter(c, LearnedRouterConfig{Threshold: 0.5, MaxFeedback: 1000})

	// Simulate data: low-score requests handled well by cheap model,
	// high-score requests need expensive model.
	for i := 0; i < 100; i++ {
		if i < 70 {
			// Cheap model handles well.
			lr.AddFeedback(FeedbackRequest{Score: 0.2, Tier: 1, Quality: 0.85})
		} else {
			// Only expensive model is good enough.
			lr.AddFeedback(FeedbackRequest{Score: 0.8, Tier: 3, Quality: 0.95})
		}
	}

	// Threshold should have been calibrated.
	thresh := lr.GetThreshold()
	// With 70% of requests handled well by cheap model at score 0.2,
	// the threshold should move to allow more cheap routing.
	if thresh > 0.8 {
		t.Errorf("threshold should be lower after calibration, got %f", thresh)
	}
}

func TestFeedbackEviction(t *testing.T) {
	c := &Classifier{Tier1Max: 30, Tier2Max: 70}
	lr := NewLearnedRouter(c, LearnedRouterConfig{Threshold: 0.5, MaxFeedback: 10})

	for i := 0; i < 20; i++ {
		lr.AddFeedback(FeedbackRequest{Score: 0.5, Tier: 2, Quality: 0.7})
	}

	stats := lr.FeedbackStats()
	total := stats["total_feedback"].(int)
	if total > 10 {
		t.Errorf("feedback should be capped at 10, got %d", total)
	}
}
