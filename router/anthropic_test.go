package router

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHandleMessagesRouting(t *testing.T) {
	// Fake Anthropic upstream.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify it's Anthropic format.
		if r.URL.Path != "/v1/messages" {
			t.Errorf("upstream path: got %q, want /v1/messages", r.URL.Path)
		}
		if r.Header.Get("x-api-key") == "" && r.Header.Get("Authorization") == "" {
			// OK — test may not have API key
		}
		if r.Header.Get("anthropic-version") == "" {
			t.Error("missing anthropic-version header")
		}

		var req AnthropicRequest
		json.NewDecoder(r.Body).Decode(&req)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id":   "msg_test",
			"type": "message",
			"role": "assistant",
			"content": []map[string]string{
				{"type": "text", "text": "Hello!"},
			},
			"model": req.Model,
			"usage": map[string]int{"input_tokens": 10, "output_tokens": 5},
		})
	}))
	defer upstream.Close()

	cfg := testConfig()
	// Override to use Anthropic provider pointing at fake upstream.
	cfg.Models = []ModelConfig{
		{Name: "haiku", Provider: "anthropic", Model: "claude-haiku", BaseURL: upstream.URL, Tier: 1, CostPer1KIn: 0.0008},
		{Name: "sonnet", Provider: "anthropic", Model: "claude-sonnet", BaseURL: upstream.URL, Tier: 2, CostPer1KIn: 0.003},
		{Name: "opus", Provider: "anthropic", Model: "claude-opus", BaseURL: upstream.URL, Tier: 3, CostPer1KIn: 0.015},
	}
	srv := NewServer(cfg)

	// Simple request → should route to haiku (tier 1).
	reqBody := AnthropicRequest{
		Model:     "claude-sonnet-4-6", // client requests sonnet, but router should override
		MaxTokens: 100,
		Messages: []AnthropicMsg{
			{Role: "user", Content: json.RawMessage(`"What is 2+2?"`)},
		},
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("anthropic-version", "2023-06-01")
	w := httptest.NewRecorder()

	srv.handleMessages(w, req)

	if w.Code != 200 {
		t.Fatalf("status=%d, body=%s", w.Code, w.Body.String())
	}

	// Check routing headers.
	if w.Header().Get("X-LLM-Router-Model") == "" {
		t.Error("missing X-LLM-Router-Model header")
	}
	if w.Header().Get("X-LLM-Router-Score") == "" {
		t.Error("missing X-LLM-Router-Score header")
	}

	// Verify the upstream received the rewritten model.
	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)
	if resp["model"] == "claude-sonnet-4-6" {
		t.Error("model should have been rewritten by router, but got original")
	}
}

func TestHandleMessagesWithTools(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"id": "msg_test", "type": "message", "role": "assistant",
			"content": []map[string]string{{"type": "text", "text": "done"}},
			"model": "claude-sonnet", "usage": map[string]int{"input_tokens": 10, "output_tokens": 5},
		})
	}))
	defer upstream.Close()

	cfg := testConfig()
	cfg.Models = []ModelConfig{
		{Name: "haiku", Provider: "anthropic", Model: "claude-haiku", BaseURL: upstream.URL, Tier: 1, CostPer1KIn: 0.0008},
		{Name: "sonnet", Provider: "anthropic", Model: "claude-sonnet", BaseURL: upstream.URL, Tier: 2, CostPer1KIn: 0.003},
	}
	srv := NewServer(cfg)

	// Request with tools → should route to tier 2+ (sonnet).
	reqBody := AnthropicRequest{
		Model:     "claude-haiku",
		MaxTokens: 100,
		Messages:  []AnthropicMsg{{Role: "user", Content: json.RawMessage(`"Use the tool"`)}},
		Tools:     json.RawMessage(`[{"name":"get_weather","description":"Get weather","input_schema":{"type":"object"}}]`),
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	srv.handleMessages(w, req)

	if w.Code != 200 {
		t.Fatalf("status=%d, body=%s", w.Code, w.Body.String())
	}

	// Tool use should have forced tier 2+ routing.
	routedModel := w.Header().Get("X-LLM-Router-Model")
	if routedModel == "haiku" {
		t.Error("tool use should force tier 2+, but got haiku")
	}
}

func TestExtractTextContent(t *testing.T) {
	// String content.
	s := extractTextContent(json.RawMessage(`"hello world"`))
	if s != "hello world" {
		t.Errorf("string: got %q", s)
	}

	// Array content blocks.
	blocks := `[{"type":"text","text":"hello"},{"type":"text","text":" world"}]`
	s = extractTextContent(json.RawMessage(blocks))
	if s != "hello\n world" {
		t.Errorf("blocks: got %q", s)
	}
}
