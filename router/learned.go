package router

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"
)

// LearnedRouter implements RouteLLM's core concept: calculate_strong_win_rate(prompt) → float64.
// If the score >= threshold, route to the strong (expensive) model.
//
// Three modes of operation:
//  1. Heuristic-only (default): uses our built-in classifier
//  2. Sidecar: calls an external ML classifier (e.g., RouteLLM's BERT/MF model)
//  3. Hybrid: combines heuristic + sidecar scores with configurable weights
//
// The router also accepts feedback via /v1/feedback to calibrate thresholds
// over time, inspired by RouteLLM's threshold calibration on Arena data.
//
// Reference: "RouteLLM: Learning to Route LLMs with Preference Data" (LMSYS)
type LearnedRouter struct {
	mu             sync.RWMutex
	classifier     *Classifier    // heuristic fallback
	sidecarURL     string         // optional: URL of ML classifier sidecar
	sidecarWeight  float64        // 0-1: weight for sidecar score (rest goes to heuristic)
	threshold      float64        // 0-1: route to strong if score >= threshold
	client         *http.Client

	// Feedback accumulator for threshold calibration.
	feedback       []feedbackEntry
	maxFeedback    int
}

type feedbackEntry struct {
	Score     float64   `json:"score"`      // classifier score (0-1)
	Tier      int       `json:"tier"`       // which tier was used
	Quality   float64   `json:"quality"`    // user-reported quality (0-1)
	Timestamp time.Time `json:"timestamp"`
}

// FeedbackRequest is the API request for reporting response quality.
type FeedbackRequest struct {
	RequestHash string  `json:"request_hash"` // hash of the original request
	Quality     float64 `json:"quality"`      // 0-1 quality score
	Tier        int     `json:"tier"`         // which tier handled it
	Score       float64 `json:"score"`        // the classifier's score
}

// SidecarResponse is the expected response from an ML classifier sidecar.
type SidecarResponse struct {
	StrongWinRate float64 `json:"strong_win_rate"` // probability that the strong model wins
}

// LearnedRouterConfig holds configuration for the learned router.
type LearnedRouterConfig struct {
	SidecarURL    string  `yaml:"sidecar_url"`    // URL of ML classifier (empty = heuristic only)
	SidecarWeight float64 `yaml:"sidecar_weight"` // 0-1 (default: 0.7 if sidecar configured)
	Threshold     float64 `yaml:"threshold"`       // 0-1 (default: 0.5)
	MaxFeedback   int     `yaml:"max_feedback"`    // max stored feedback entries (default: 10000)
}

// NewLearnedRouter creates a learned router.
func NewLearnedRouter(classifier *Classifier, cfg LearnedRouterConfig) *LearnedRouter {
	threshold := cfg.Threshold
	if threshold == 0 {
		threshold = 0.5
	}
	sidecarWeight := cfg.SidecarWeight
	if sidecarWeight == 0 && cfg.SidecarURL != "" {
		sidecarWeight = 0.7
	}
	maxFeedback := cfg.MaxFeedback
	if maxFeedback == 0 {
		maxFeedback = 10000
	}

	return &LearnedRouter{
		classifier:    classifier,
		sidecarURL:    cfg.SidecarURL,
		sidecarWeight: sidecarWeight,
		threshold:     threshold,
		client:        &http.Client{Timeout: 2 * time.Second},
		maxFeedback:   maxFeedback,
	}
}

// CalculateStrongWinRate returns the probability (0-1) that the strong model
// should handle this request. This is the core RouteLLM interface.
func (r *LearnedRouter) CalculateStrongWinRate(req *ChatRequest) float64 {
	// Get heuristic score (0-100, normalized to 0-1).
	heuristicScore, _ := r.classifier.Classify(req)
	hNorm := float64(heuristicScore) / 100.0

	// If no sidecar, use heuristic only.
	if r.sidecarURL == "" {
		return hNorm
	}

	// Call sidecar for ML score.
	mlScore, err := r.callSidecar(req)
	if err != nil {
		// Fallback to heuristic on sidecar failure.
		return hNorm
	}

	// Weighted combination.
	return r.sidecarWeight*mlScore + (1-r.sidecarWeight)*hNorm
}

// Route returns the recommended tier (1-3) based on the win rate and threshold.
func (r *LearnedRouter) Route(req *ChatRequest) (score float64, tier int) {
	score = r.CalculateStrongWinRate(req)

	r.mu.RLock()
	thresh := r.threshold
	r.mu.RUnlock()

	// Map continuous score to 3 tiers using threshold bands.
	// score < threshold*0.6  → tier 1 (cheap)
	// score < threshold      → tier 2 (mid)
	// score >= threshold     → tier 3 (strong)
	if score < thresh*0.6 {
		tier = 1
	} else if score < thresh {
		tier = 2
	} else {
		tier = 3
	}
	return score, tier
}

// AddFeedback records a quality observation for threshold calibration.
func (r *LearnedRouter) AddFeedback(fb FeedbackRequest) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.feedback = append(r.feedback, feedbackEntry{
		Score:     fb.Score,
		Tier:      fb.Tier,
		Quality:   fb.Quality,
		Timestamp: time.Now(),
	})

	// Evict oldest if over capacity.
	if len(r.feedback) > r.maxFeedback {
		r.feedback = r.feedback[len(r.feedback)-r.maxFeedback:]
	}

	// Auto-calibrate threshold every 100 feedback entries.
	if len(r.feedback) > 0 && len(r.feedback)%100 == 0 {
		r.calibrateThreshold()
	}
}

// calibrateThreshold adjusts the routing threshold based on accumulated feedback.
// Goal: find the threshold that maximizes quality while minimizing cost.
// Ported from RouteLLM's calibrate_threshold.py concept.
func (r *LearnedRouter) calibrateThreshold() {
	if len(r.feedback) < 50 {
		return
	}

	// Find the score threshold where tier-1 quality drops below acceptable.
	// Acceptable = avg quality of tier-3 responses * 0.9 (90% of best quality).
	var tier3Quality, tier3Count float64
	for _, fb := range r.feedback {
		if fb.Tier == 3 {
			tier3Quality += fb.Quality
			tier3Count++
		}
	}
	if tier3Count == 0 {
		return
	}
	target := (tier3Quality / tier3Count) * 0.9

	// Binary search for optimal threshold.
	best := r.threshold
	bestSavings := 0.0

	for t := 0.1; t <= 0.9; t += 0.05 {
		var cheapOK, cheapTotal, expensiveTotal int
		for _, fb := range r.feedback {
			if fb.Score < t {
				cheapTotal++
				if fb.Quality >= target {
					cheapOK++
				}
			} else {
				expensiveTotal++
			}
		}

		if cheapTotal == 0 {
			continue
		}

		cheapQualityRate := float64(cheapOK) / float64(cheapTotal)
		savings := float64(cheapTotal) / float64(len(r.feedback))

		// Only accept thresholds where cheap model quality is acceptable.
		if cheapQualityRate >= 0.85 && savings > bestSavings {
			bestSavings = savings
			best = t
		}
	}

	r.threshold = best
}

// GetThreshold returns the current routing threshold.
func (r *LearnedRouter) GetThreshold() float64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.threshold
}

// FeedbackStats returns summary statistics about the feedback collected.
func (r *LearnedRouter) FeedbackStats() map[string]any {
	r.mu.RLock()
	defer r.mu.RUnlock()

	stats := map[string]any{
		"total_feedback": len(r.feedback),
		"threshold":      math.Round(r.threshold*1000) / 1000,
		"sidecar_url":    r.sidecarURL,
		"sidecar_weight": r.sidecarWeight,
	}

	if len(r.feedback) > 0 {
		var totalQuality float64
		tierCounts := map[int]int{}
		for _, fb := range r.feedback {
			totalQuality += fb.Quality
			tierCounts[fb.Tier]++
		}
		stats["avg_quality"] = math.Round((totalQuality/float64(len(r.feedback)))*1000) / 1000
		stats["by_tier"] = tierCounts
	}

	return stats
}

// callSidecar sends a request to the external ML classifier.
func (r *LearnedRouter) callSidecar(req *ChatRequest) (float64, error) {
	// Build the prompt from messages (RouteLLM uses the first user message).
	prompt := ""
	for _, m := range req.Messages {
		if m.Role == "user" {
			prompt = m.Content
			break
		}
	}

	body, _ := json.Marshal(map[string]string{"prompt": prompt})
	resp, err := r.client.Post(r.sidecarURL, "application/json", strings.NewReader(string(body)))
	if err != nil {
		return 0, fmt.Errorf("sidecar: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("sidecar: status %d", resp.StatusCode)
	}

	var result SidecarResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, fmt.Errorf("sidecar: decode: %w", err)
	}

	return result.StrongWinRate, nil
}
