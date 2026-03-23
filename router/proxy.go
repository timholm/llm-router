package router

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Server is the llm-router HTTP server.
type Server struct {
	cfg        *Config
	classifier *Classifier
	client     *http.Client
	stats      *Stats
	pool       *Pool          // health-aware backend pool (from kv.run concepts)
	dedup      *Deduplicator  // request deduplication (from ParrotServe concepts)
	cache      *SemanticCache // semantic response cache (from GPTCache concepts)
	learned    *LearnedRouter // ML-assisted routing (from RouteLLM concepts)
}

// Stats tracks cumulative cost savings.
type Stats struct {
	mu            sync.Mutex
	TotalRequests int64
	TotalSaved    float64 // dollars saved vs always using most expensive model
	TotalSpent    float64 // actual dollars spent
	ByModel       map[string]int64
}

// NewServer creates a new router server.
func NewServer(cfg *Config) *Server {
	return &Server{
		cfg: cfg,
		classifier: &Classifier{
			Tier1Max: cfg.Classifier.Tier1Max,
			Tier2Max: cfg.Classifier.Tier2Max,
		},
		client: &http.Client{Timeout: time.Duration(cfg.Server.WriteTimeoutSec) * time.Second},
		stats: &Stats{
			ByModel: make(map[string]int64),
		},
		pool:  NewPool(cfg.Models),
		dedup: NewDeduplicator(5 * time.Second),
		cache: NewSemanticCache(cfg.Cache),
		learned: NewLearnedRouter(&Classifier{
			Tier1Max: cfg.Classifier.Tier1Max,
			Tier2Max: cfg.Classifier.Tier2Max,
		}, cfg.Learned),
	}
}

// ListenAndServe starts the HTTP server.
func (s *Server) ListenAndServe(addr string) error {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	mux.HandleFunc("/v1/workflows", s.handleWorkflow)
	mux.HandleFunc("/v1/feedback", s.handleFeedback)
	mux.HandleFunc("/v1/backends", s.handleBackends)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/stats", s.handleStats)

	srv := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  time.Duration(s.cfg.Server.ReadTimeoutSec) * time.Second,
		WriteTimeout: time.Duration(s.cfg.Server.WriteTimeoutSec) * time.Second,
	}
	return srv.ListenAndServe()
}

// handleChat is the main routing endpoint. It accepts OpenAI-format requests,
// classifies them, picks the cheapest capable model, and proxies the request.
func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
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

	var req ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Check semantic cache first (GPTCache concept).
	if s.cfg.Cache.Enabled {
		if cached, cachedModel, hit := s.cache.Get(req.Messages); hit {
			log.Printf("cache hit: model=%s", cachedModel)
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

	// Classify using learned router (RouteLLM concept) with heuristic fallback.
	learnedScore, tier := s.learned.Route(&req)
	score := int(learnedScore * 100)

	// Select the best backend via the health-aware pool.
	model := s.pool.SelectModel(tier, &req)
	if model == nil {
		http.Error(w, "no model available for this request", http.StatusServiceUnavailable)
		return
	}

	log.Printf("route: score=%d tier=%d → %s (%s)", score, tier, model.Name, model.Provider)

	// Track in-flight request (kv.run lifecycle pattern).
	s.pool.TrackRequest(model.Name)
	reqStart := time.Now()

	// Rewrite the model field and proxy.
	req.Model = model.Model

	rewritten, err := json.Marshal(req)
	if err != nil {
		s.pool.CompleteRequest(model.Name, time.Since(reqStart), err)
		http.Error(w, "failed to marshal request", http.StatusInternalServerError)
		return
	}

	// Build the upstream request.
	upstreamURL := strings.TrimRight(model.BaseURL, "/") + "/v1/chat/completions"
	upReq, err := http.NewRequestWithContext(r.Context(), "POST", upstreamURL, bytes.NewReader(rewritten))
	if err != nil {
		s.pool.CompleteRequest(model.Name, time.Since(reqStart), err)
		http.Error(w, "failed to create upstream request", http.StatusInternalServerError)
		return
	}
	upReq.Header.Set("Content-Type", "application/json")
	if model.APIKey != "" {
		upReq.Header.Set("Authorization", "Bearer "+model.APIKey)
	}

	// Proxy the response (including streaming).
	resp, err := s.client.Do(upReq)
	if err != nil {
		s.pool.CompleteRequest(model.Name, time.Since(reqStart), err)
		// Mark backend as down if it's unreachable.
		s.pool.MarkDown(model.Name)
		http.Error(w, fmt.Sprintf("upstream error: %v", err), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Complete the request lifecycle tracking.
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

	// Copy response headers.
	for k, vv := range resp.Header {
		for _, v := range vv {
			w.Header().Add(k, v)
		}
	}
	// Add routing metadata headers.
	w.Header().Set("X-LLM-Router-Score", fmt.Sprintf("%d", score))
	w.Header().Set("X-LLM-Router-Tier", fmt.Sprintf("%d", tier))
	w.Header().Set("X-LLM-Router-Model", model.Name)
	w.Header().Set("X-LLM-Router-Cache", "MISS")

	// Buffer response for caching, then write to client.
	respBody, _ := io.ReadAll(resp.Body)

	w.WriteHeader(resp.StatusCode)
	w.Write(respBody)

	// Store in semantic cache for future similar queries.
	if s.cfg.Cache.Enabled && resp.StatusCode == http.StatusOK && !req.Stream {
		s.cache.Put(req.Messages, respBody, model.Name)
	}
}

// selectModel picks a model using the config-based fallback (used by tests that don't need the pool).
// The primary routing path uses pool.SelectModel instead.
func (s *Server) selectModel(tier int, req *ChatRequest) *ModelConfig {
	return s.pool.SelectModel(tier, req)
}

// trackCost records usage stats.
func (s *Server) trackCost(model *ModelConfig, score int) {
	s.stats.mu.Lock()
	defer s.stats.mu.Unlock()

	s.stats.TotalRequests++
	s.stats.ByModel[model.Name]++

	// Estimate savings: difference between this model's cost and the most expensive model.
	if len(s.cfg.Models) > 0 {
		maxModel := s.cfg.Models[len(s.cfg.Models)-1]
		savedPerKIn := maxModel.CostPer1KIn - model.CostPer1KIn
		savedPerKOut := maxModel.CostPer1KOut - model.CostPer1KOut
		// Rough estimate: assume 500 input tokens, 200 output tokens per request.
		s.stats.TotalSaved += (savedPerKIn * 0.5) + (savedPerKOut * 0.2)
		s.stats.TotalSpent += (model.CostPer1KIn * 0.5) + (model.CostPer1KOut * 0.2)
	}
}

// handleWorkflow executes a DAG workflow through the router.
// Inspired by Halo (arXiv:2509.02121) and Helium's workflow-as-query-plan model.
func (s *Server) handleWorkflow(w http.ResponseWriter, r *http.Request) {
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

	var req WorkflowRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if req.Workflow == "" {
		http.Error(w, "workflow field is required (inline YAML)", http.StatusBadRequest)
		return
	}

	wf, err := ParseWorkflow([]byte(req.Workflow))
	if err != nil {
		http.Error(w, "invalid workflow: "+err.Error(), http.StatusBadRequest)
		return
	}

	log.Printf("workflow: executing %d ops, input=%d chars", len(wf.Ops), len(req.Input))

	result, err := s.ExecuteWorkflow(r.Context(), wf, req.Input)
	if err != nil {
		http.Error(w, "workflow execution failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// handleFeedback accepts quality feedback for threshold calibration.
// POST /v1/feedback {"score": 0.3, "tier": 1, "quality": 0.85}
func (s *Server) handleFeedback(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var fb FeedbackRequest
	if err := json.NewDecoder(r.Body).Decode(&fb); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	s.learned.AddFeedback(fb)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s.learned.FeedbackStats())
}

// handleBackends returns the health and load state of all backends.
// Inspired by kv.run's worker status reporting.
func (s *Server) handleBackends(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s.pool.Snapshot())
}

// handleHealth returns a simple health check.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"status":"ok"}`))
}

// handleStats returns cumulative routing statistics.
func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	s.stats.mu.Lock()
	data := struct {
		TotalRequests int64            `json:"total_requests"`
		TotalSaved    float64          `json:"total_saved_usd"`
		TotalSpent    float64          `json:"total_spent_usd"`
		ByModel       map[string]int64 `json:"by_model"`
	}{
		TotalRequests: s.stats.TotalRequests,
		TotalSaved:    s.stats.TotalSaved,
		TotalSpent:    s.stats.TotalSpent,
		ByModel:       s.stats.ByModel,
	}
	s.stats.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}
