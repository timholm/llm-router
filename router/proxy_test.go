package router

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

func testConfig() *Config {
	return &Config{
		Models: []ModelConfig{
			{Name: "cheap", Provider: "openai", Model: "gpt-4o-mini", BaseURL: "", Tier: 1, CostPer1KIn: 0.00015, CostPer1KOut: 0.0006},
			{Name: "mid", Provider: "openai", Model: "gpt-4o", BaseURL: "", Tier: 2, CostPer1KIn: 0.005, CostPer1KOut: 0.015},
			{Name: "expensive", Provider: "openai", Model: "o1", BaseURL: "", Tier: 3, CostPer1KIn: 0.015, CostPer1KOut: 0.06},
		},
		Classifier: ClassifierConfig{Tier1Max: 30, Tier2Max: 70},
		Server:     ServerConfig{ReadTimeoutSec: 30, WriteTimeoutSec: 120},
	}
}

func TestSelectModelSimple(t *testing.T) {
	cfg := testConfig()
	srv := NewServer(cfg)

	req := &ChatRequest{Messages: []Message{{Role: "user", Content: "hi"}}}
	model := srv.selectModel(1, req)
	if model == nil {
		t.Fatal("selectModel returned nil")
	}
	if model.Name != "cheap" {
		t.Errorf("tier 1: got %q, want 'cheap'", model.Name)
	}
}

func TestSelectModelTier3(t *testing.T) {
	cfg := testConfig()
	srv := NewServer(cfg)

	req := &ChatRequest{Messages: []Message{{Role: "user", Content: "complex task"}}}
	model := srv.selectModel(3, req)
	if model == nil {
		t.Fatal("selectModel returned nil")
	}
	if model.Name != "expensive" {
		t.Errorf("tier 3: got %q, want 'expensive'", model.Name)
	}
}

func TestSelectModelToolUseUpgrade(t *testing.T) {
	cfg := testConfig()
	srv := NewServer(cfg)

	req := &ChatRequest{
		Messages: []Message{{Role: "user", Content: "call tool"}},
		Tools:    []Tool{{Type: "function", Function: ToolFunction{Name: "test"}}},
	}
	// Tier 1 request but with tools → should upgrade to tier 2.
	model := srv.selectModel(1, req)
	if model == nil {
		t.Fatal("selectModel returned nil")
	}
	if model.Tier < 2 {
		t.Errorf("tool use should force tier >= 2, got tier %d (%s)", model.Tier, model.Name)
	}
}

func TestSelectModelExplicitName(t *testing.T) {
	cfg := testConfig()
	srv := NewServer(cfg)

	req := &ChatRequest{
		Model:    "expensive",
		Messages: []Message{{Role: "user", Content: "hi"}},
	}
	model := srv.selectModel(1, req)
	if model == nil {
		t.Fatal("selectModel returned nil")
	}
	if model.Name != "expensive" {
		t.Errorf("explicit model: got %q, want 'expensive'", model.Name)
	}
}

func TestHealthEndpoint(t *testing.T) {
	cfg := testConfig()
	srv := NewServer(cfg)

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	srv.handleHealth(w, req)

	if w.Code != 200 {
		t.Errorf("health: status=%d, want 200", w.Code)
	}
}

func TestStatsEndpoint(t *testing.T) {
	cfg := testConfig()
	srv := NewServer(cfg)

	req := httptest.NewRequest("GET", "/stats", nil)
	w := httptest.NewRecorder()
	srv.handleStats(w, req)

	if w.Code != 200 {
		t.Errorf("stats: status=%d, want 200", w.Code)
	}

	var stats struct {
		TotalRequests int64 `json:"total_requests"`
	}
	if err := json.NewDecoder(w.Body).Decode(&stats); err != nil {
		t.Fatalf("decode stats: %v", err)
	}
	if stats.TotalRequests != 0 {
		t.Errorf("initial requests=%d, want 0", stats.TotalRequests)
	}
}

func TestChatEndpointRouting(t *testing.T) {
	// Set up a fake upstream LLM server.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Read the request to verify model was rewritten.
		body, _ := io.ReadAll(r.Body)
		var req ChatRequest
		json.Unmarshal(body, &req)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":      "test-1",
			"model":   req.Model,
			"choices": []map[string]any{{"message": map[string]string{"role": "assistant", "content": "hello"}}},
		})
	}))
	defer upstream.Close()

	cfg := testConfig()
	// Point all models at the fake upstream.
	for i := range cfg.Models {
		cfg.Models[i].BaseURL = upstream.URL
	}

	srv := NewServer(cfg)

	chatReq := ChatRequest{
		Messages: []Message{{Role: "user", Content: "What is 2+2?"}},
	}
	body, _ := json.Marshal(chatReq)

	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	srv.handleChat(w, req)

	if w.Code != 200 {
		t.Fatalf("chat: status=%d, body=%s", w.Code, w.Body.String())
	}

	// Check routing headers.
	if w.Header().Get("X-LLM-Router-Model") == "" {
		t.Error("missing X-LLM-Router-Model header")
	}
	if w.Header().Get("X-LLM-Router-Score") == "" {
		t.Error("missing X-LLM-Router-Score header")
	}

	// Stats should show 1 request.
	statsReq := httptest.NewRequest("GET", "/stats", nil)
	statsW := httptest.NewRecorder()
	srv.handleStats(statsW, statsReq)

	var stats struct {
		TotalRequests int64 `json:"total_requests"`
	}
	json.NewDecoder(statsW.Body).Decode(&stats)
	if stats.TotalRequests != 1 {
		t.Errorf("after 1 chat: requests=%d, want 1", stats.TotalRequests)
	}
}
