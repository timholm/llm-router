package router

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// AnthropicRequest is Anthropic's native /v1/messages request format.
// Claude Code uses this format — not the OpenAI format.
type AnthropicRequest struct {
	Model         string            `json:"model"`
	MaxTokens     int               `json:"max_tokens"`
	Messages      []AnthropicMsg    `json:"messages"`
	System        json.RawMessage   `json:"system,omitempty"`
	Stream        bool              `json:"stream,omitempty"`
	Temperature   *float64          `json:"temperature,omitempty"`
	TopP          *float64          `json:"top_p,omitempty"`
	TopK          *int              `json:"top_k,omitempty"`
	StopSequences []string          `json:"stop_sequences,omitempty"`
	Tools         json.RawMessage   `json:"tools,omitempty"`
	ToolChoice    json.RawMessage   `json:"tool_choice,omitempty"`
	Metadata      json.RawMessage   `json:"metadata,omitempty"`
}

// AnthropicMsg is a message in Anthropic format.
type AnthropicMsg struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"` // string or []ContentBlock
}

// extractTextContent extracts plain text from an Anthropic message content field.
// Content can be a string or an array of content blocks.
func extractTextContent(raw json.RawMessage) string {
	// Try as string first.
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}

	// Try as array of content blocks.
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &blocks); err == nil {
		var parts []string
		for _, b := range blocks {
			if b.Type == "text" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, "\n")
	}

	return string(raw)
}

// handleMessages handles Anthropic-native /v1/messages requests.
// This is the key integration point — Claude Code sends all API calls here
// when ANTHROPIC_BASE_URL points at the router.
//
// Flow: classify → pick model → rewrite → proxy → track costs
func (s *Server) handleMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	var req AnthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Convert Anthropic messages to our internal format for classification.
	var chatMsgs []Message
	// Include system prompt if present.
	if len(req.System) > 0 {
		sysText := extractTextContent(req.System)
		if sysText != "" {
			chatMsgs = append(chatMsgs, Message{Role: "system", Content: sysText})
		}
	}
	for _, m := range req.Messages {
		chatMsgs = append(chatMsgs, Message{
			Role:    m.Role,
			Content: extractTextContent(m.Content),
		})
	}

	chatReq := &ChatRequest{
		Messages: chatMsgs,
	}
	// Check if tools are present (affects classification).
	if len(req.Tools) > 0 {
		chatReq.Tools = []Tool{{Type: "function"}} // signal tool use to classifier
	}

	// Check semantic cache.
	if s.cfg.Cache.Enabled && !req.Stream {
		if cached, cachedModel, hit := s.cache.Get(chatMsgs); hit {
			log.Printf("anthropic: cache HIT → %s", cachedModel)
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-LLM-Router-Cache", "HIT")
			w.Header().Set("X-LLM-Router-Model", cachedModel)
			w.Write(cached)
			s.stats.mu.Lock()
			s.stats.TotalRequests++
			s.stats.ByModel[cachedModel]++
			s.stats.mu.Unlock()
			return
		}
	}

	// Classify using learned router.
	learnedScore, tier := s.learned.Route(chatReq)
	score := int(learnedScore * 100)

	// Select backend via health-aware pool.
	model := s.pool.SelectModel(tier, chatReq)
	if model == nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		json.NewEncoder(w).Encode(map[string]any{
			"type": "error", "error": map[string]string{
				"type": "overloaded_error", "message": "no model available",
			},
		})
		return
	}

	log.Printf("anthropic: score=%d tier=%d → %s", score, tier, model.Name)

	// Track in-flight.
	s.pool.TrackRequest(model.Name)
	reqStart := time.Now()

	// Rewrite model in the request.
	req.Model = model.Model
	rewritten, err := json.Marshal(req)
	if err != nil {
		s.pool.CompleteRequest(model.Name, time.Since(reqStart), err)
		http.Error(w, "failed to marshal request", http.StatusInternalServerError)
		return
	}

	// Forward to Anthropic API.
	upstreamURL := strings.TrimRight(model.BaseURL, "/") + "/v1/messages"
	upReq, err := http.NewRequestWithContext(r.Context(), "POST", upstreamURL, bytes.NewReader(rewritten))
	if err != nil {
		s.pool.CompleteRequest(model.Name, time.Since(reqStart), err)
		http.Error(w, "failed to create upstream request", http.StatusInternalServerError)
		return
	}

	// Anthropic uses x-api-key header, not Authorization Bearer.
	apiKey := ResolveAPIKey(*model)
	if apiKey != "" {
		upReq.Header.Set("x-api-key", apiKey)
	}
	upReq.Header.Set("Content-Type", "application/json")
	upReq.Header.Set("anthropic-version", "2023-06-01")
	// Forward any anthropic-beta header from the client.
	if beta := r.Header.Get("anthropic-beta"); beta != "" {
		upReq.Header.Set("anthropic-beta", beta)
	}

	// Proxy the response.
	resp, err := s.client.Do(upReq)
	if err != nil {
		s.pool.CompleteRequest(model.Name, time.Since(reqStart), err)
		s.pool.MarkDown(model.Name)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		json.NewEncoder(w).Encode(map[string]any{
			"type": "error", "error": map[string]string{
				"type": "api_error", "message": fmt.Sprintf("upstream: %v", err),
			},
		})
		return
	}
	defer resp.Body.Close()

	// Track latency and health.
	latency := time.Since(reqStart)
	var respErr error
	if resp.StatusCode >= 500 {
		respErr = fmt.Errorf("upstream %d", resp.StatusCode)
		s.pool.MarkDegraded(model.Name)
	} else {
		s.pool.MarkHealthy(model.Name)
	}
	s.pool.CompleteRequest(model.Name, latency, respErr)

	// Track costs.
	s.trackCost(model, score)

	// Buffer response for caching.
	respBody, _ := io.ReadAll(resp.Body)

	// Copy response headers.
	for k, vv := range resp.Header {
		for _, v := range vv {
			w.Header().Add(k, v)
		}
	}
	w.Header().Set("X-LLM-Router-Score", fmt.Sprintf("%d", score))
	w.Header().Set("X-LLM-Router-Tier", fmt.Sprintf("%d", tier))
	w.Header().Set("X-LLM-Router-Model", model.Name)
	w.Header().Set("X-LLM-Router-Cache", "MISS")

	w.WriteHeader(resp.StatusCode)
	w.Write(respBody)

	// Cache non-streaming successful responses.
	if s.cfg.Cache.Enabled && resp.StatusCode == http.StatusOK && !req.Stream {
		s.cache.Put(chatMsgs, respBody, model.Name)
	}
}
