package router

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// BackendStatus represents the health state of a backend.
// Ported from kv.run's worker lifecycle (IDLE/RUNNING/FAILED).
type BackendStatus string

const (
	StatusHealthy  BackendStatus = "healthy"
	StatusDegraded BackendStatus = "degraded" // responding but slow
	StatusDown     BackendStatus = "down"
	StatusDraining BackendStatus = "draining" // no new requests, finishing in-flight
)

// Backend represents a live LLM inference endpoint with health and cost tracking.
// Inspired by kv.run's RedisWorker + Lifecycle pattern.
type Backend struct {
	Config      ModelConfig
	Status      BackendStatus
	LastSeen    time.Time     // last successful health check
	Latency     time.Duration // rolling average response time
	InFlight    int           // current in-flight requests
	TotalReqs   int64         // lifetime request count
	TotalErrors int64         // lifetime error count
	CostAccrued float64       // total cost accrued (USD)
	StartedAt   time.Time     // when this backend was registered
}

// Pool manages a set of backends with health checking and load-aware selection.
// Ported from kv.run's worker registration/heartbeat/lifecycle pattern.
type Pool struct {
	mu       sync.RWMutex
	backends map[string]*Backend // keyed by ModelConfig.Name
	checkInterval time.Duration
	healthTimeout time.Duration
	stopCh   chan struct{}
}

// NewPool creates a backend pool from config.
func NewPool(models []ModelConfig) *Pool {
	p := &Pool{
		backends:      make(map[string]*Backend, len(models)),
		checkInterval: 30 * time.Second,
		healthTimeout: 5 * time.Second,
		stopCh:        make(chan struct{}),
	}

	now := time.Now()
	for _, m := range models {
		p.backends[m.Name] = &Backend{
			Config:    m,
			Status:    StatusHealthy, // assume healthy until proven otherwise
			LastSeen:  now,
			StartedAt: now,
		}
	}

	return p
}

// SelectModel picks the best backend for the given tier.
// Selection criteria (in order):
//  1. Must be healthy or degraded (not down/draining)
//  2. Must meet minimum tier requirement
//  3. Prefer lowest in-flight count (load balancing)
//  4. Break ties by cheapest cost
//
// This implements kv.run's health-aware dispatch pattern.
func (p *Pool) SelectModel(tier int, req *ChatRequest) *ModelConfig {
	p.mu.RLock()
	defer p.mu.RUnlock()

	// If request specifies a model explicitly, use it (pass-through).
	if req != nil && req.Model != "" {
		for _, b := range p.backends {
			if (b.Config.Name == req.Model || b.Config.Model == req.Model) && b.Status != StatusDown {
				return &b.Config
			}
		}
	}

	// Tool use requires tier 2+.
	if req != nil && len(req.Tools) > 0 && tier < 2 {
		tier = 2
	}

	// Collect eligible backends.
	type candidate struct {
		backend *Backend
	}
	var candidates []candidate
	for _, b := range p.backends {
		if b.Status == StatusDown || b.Status == StatusDraining {
			continue
		}
		if b.Config.Tier >= tier {
			candidates = append(candidates, candidate{backend: b})
		}
	}

	if len(candidates) == 0 {
		// Fallback: any non-down backend.
		for _, b := range p.backends {
			if b.Status != StatusDown {
				return &b.Config
			}
		}
		return nil
	}

	// Sort: lowest in-flight first, then cheapest.
	sort.Slice(candidates, func(i, j int) bool {
		bi, bj := candidates[i].backend, candidates[j].backend
		if bi.InFlight != bj.InFlight {
			return bi.InFlight < bj.InFlight
		}
		return bi.Config.CostPer1KIn < bj.Config.CostPer1KIn
	})

	return &candidates[0].backend.Config
}

// TrackRequest increments in-flight count for a backend.
func (p *Pool) TrackRequest(modelName string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if b, ok := p.backends[modelName]; ok {
		b.InFlight++
		b.TotalReqs++
	}
}

// CompleteRequest decrements in-flight and updates latency.
func (p *Pool) CompleteRequest(modelName string, latency time.Duration, err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if b, ok := p.backends[modelName]; ok {
		if b.InFlight > 0 {
			b.InFlight--
		}
		// Exponential moving average for latency.
		if b.Latency == 0 {
			b.Latency = latency
		} else {
			b.Latency = (b.Latency*7 + latency*3) / 10
		}
		if err != nil {
			b.TotalErrors++
		}
	}
}

// MarkDown sets a backend as down.
func (p *Pool) MarkDown(modelName string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if b, ok := p.backends[modelName]; ok {
		b.Status = StatusDown
	}
}

// MarkHealthy sets a backend as healthy and updates LastSeen.
func (p *Pool) MarkHealthy(modelName string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if b, ok := p.backends[modelName]; ok {
		b.Status = StatusHealthy
		b.LastSeen = time.Now()
	}
}

// MarkDegraded sets a backend as degraded (slow but responding).
func (p *Pool) MarkDegraded(modelName string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if b, ok := p.backends[modelName]; ok {
		b.Status = StatusDegraded
		b.LastSeen = time.Now()
	}
}

// Snapshot returns a copy of all backend states for the /stats endpoint.
func (p *Pool) Snapshot() []BackendSnapshot {
	p.mu.RLock()
	defer p.mu.RUnlock()

	snaps := make([]BackendSnapshot, 0, len(p.backends))
	for _, b := range p.backends {
		snaps = append(snaps, BackendSnapshot{
			Name:        b.Config.Name,
			Model:       b.Config.Model,
			Provider:    b.Config.Provider,
			Tier:        b.Config.Tier,
			Status:      string(b.Status),
			InFlight:    b.InFlight,
			TotalReqs:   b.TotalReqs,
			TotalErrors: b.TotalErrors,
			LatencyMs:   b.Latency.Milliseconds(),
			CostPer1KIn: b.Config.CostPer1KIn,
			UptimeSec:   int64(time.Since(b.StartedAt).Seconds()),
		})
	}

	sort.Slice(snaps, func(i, j int) bool {
		return snaps[i].Name < snaps[j].Name
	})
	return snaps
}

// BackendSnapshot is the JSON-serializable view of a backend's state.
type BackendSnapshot struct {
	Name        string  `json:"name"`
	Model       string  `json:"model"`
	Provider    string  `json:"provider"`
	Tier        int     `json:"tier"`
	Status      string  `json:"status"`
	InFlight    int     `json:"in_flight"`
	TotalReqs   int64   `json:"total_requests"`
	TotalErrors int64   `json:"total_errors"`
	LatencyMs   int64   `json:"latency_ms"`
	CostPer1KIn float64 `json:"cost_per_1k_in"`
	UptimeSec   int64   `json:"uptime_sec"`
}

// String returns a human-readable summary.
func (b *Backend) String() string {
	return fmt.Sprintf("%s [%s] inflight=%d latency=%s reqs=%d errs=%d",
		b.Config.Name, b.Status, b.InFlight, b.Latency, b.TotalReqs, b.TotalErrors)
}
