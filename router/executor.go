package router

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// WorkflowRequest is the API request for executing a workflow.
type WorkflowRequest struct {
	Workflow   string `json:"workflow"`    // inline YAML or path (if server-side workflows enabled)
	Input      string `json:"input"`       // the user's query/content to process through the workflow
	Stream     bool   `json:"stream"`      // stream the final op's output
}

// WorkflowResponse is the API response from a workflow execution.
type WorkflowResponse struct {
	Outputs  map[string]string     `json:"outputs"`   // op_id → output text
	Routing  map[string]string     `json:"routing"`   // op_id → model used
	Costs    map[string]float64    `json:"costs"`     // op_id → estimated cost
	TotalMs  int64                 `json:"total_ms"`  // total execution time
	FinalOut string                `json:"final_output"` // last end_op's output
}

// ExecuteWorkflow runs a workflow DAG through the router, executing each op
// in topological order with dependency tracking.
//
// Key optimizations from Halo/Helium:
//   - Dependency-aware: ops only execute when all inputs are ready
//   - Cache-aware routing: same-tier consecutive ops prefer the same model
//   - Parallel execution: independent ops at the same level run concurrently
func (s *Server) ExecuteWorkflow(ctx context.Context, wf *Workflow, input string) (*WorkflowResponse, error) {
	start := time.Now()

	outputs := make(map[string]string)  // op_id → output
	routing := make(map[string]string)  // op_id → model used
	costs := make(map[string]float64)   // op_id → cost
	var mu sync.Mutex

	// Track which model was used for each tier (cache-aware routing).
	tierModelCache := make(map[int]*ModelConfig)

	// Execute in topological levels for maximum parallelism.
	levels := computeLevels(wf)

	for _, level := range levels {
		var wg sync.WaitGroup
		errs := make([]error, len(level))

		for i, op := range level {
			wg.Add(1)
			go func(idx int, op *Op) {
				defer wg.Done()

				// Build the prompt: op's system prompt + input/parent outputs.
				prompt := buildOpPrompt(op, input, outputs, &mu)

				// Route to cheapest capable model.
				mu.Lock()
				model := s.routeOp(op, prompt, tierModelCache)
				mu.Unlock()

				if model == nil {
					errs[idx] = fmt.Errorf("no model available for op %s (tier %d)", op.ID, op.Tier)
					return
				}

				// Call the LLM.
				result, err := s.callModel(ctx, model, prompt, op.MaxTokens)
				if err != nil {
					errs[idx] = fmt.Errorf("op %s: %w", op.ID, err)
					return
				}

				mu.Lock()
				outputs[op.ID] = result
				routing[op.ID] = model.Name
				// Estimate cost.
				inTokens := float64(len(prompt)) * 1.3 / 4.0 // rough estimate
				outTokens := float64(len(result)) * 1.3 / 4.0
				costs[op.ID] = (inTokens/1000)*model.CostPer1KIn + (outTokens/1000)*model.CostPer1KOut

				// Cache the model selection for this tier.
				if op.KeepCache {
					tierModelCache[op.Tier] = model
				}
				mu.Unlock()
			}(i, op)
		}

		wg.Wait()

		// Check for errors.
		for _, err := range errs {
			if err != nil {
				return nil, err
			}
		}
	}

	// Determine final output (last end op).
	finalOut := ""
	for _, endOp := range wf.EndOps {
		if out, ok := outputs[endOp.ID]; ok {
			finalOut = out
		}
	}

	// Track stats.
	totalCost := 0.0
	for _, c := range costs {
		totalCost += c
	}
	s.stats.mu.Lock()
	s.stats.TotalRequests++
	s.stats.TotalSpent += totalCost
	// Savings: compare vs running every op on the most expensive model.
	if len(s.cfg.Models) > 0 {
		maxModel := s.cfg.Models[len(s.cfg.Models)-1]
		maxCost := 0.0
		for opID := range outputs {
			inTokens := float64(len(buildOpPrompt(wf.Ops[opID], input, outputs, &mu))) * 1.3 / 4.0
			outTokens := float64(len(outputs[opID])) * 1.3 / 4.0
			maxCost += (inTokens/1000)*maxModel.CostPer1KIn + (outTokens/1000)*maxModel.CostPer1KOut
		}
		s.stats.TotalSaved += maxCost - totalCost
	}
	for _, model := range routing {
		s.stats.ByModel[model]++
	}
	s.stats.mu.Unlock()

	return &WorkflowResponse{
		Outputs:  outputs,
		Routing:  routing,
		Costs:    costs,
		TotalMs:  time.Since(start).Milliseconds(),
		FinalOut: finalOut,
	}, nil
}

// computeLevels groups ops into parallel execution levels.
// Ops in the same level have no dependencies on each other.
func computeLevels(wf *Workflow) [][]*Op {
	order := wf.TopologicalOrder()
	depth := make(map[string]int)

	for _, op := range order {
		maxDepth := 0
		for _, dep := range op.InputOps {
			if d, ok := depth[dep.ID]; ok && d+1 > maxDepth {
				maxDepth = d + 1
			}
		}
		depth[op.ID] = maxDepth
	}

	// Group by depth.
	maxLevel := 0
	for _, d := range depth {
		if d > maxLevel {
			maxLevel = d
		}
	}

	levels := make([][]*Op, maxLevel+1)
	for _, op := range order {
		d := depth[op.ID]
		levels[d] = append(levels[d], op)
	}
	return levels
}

// buildOpPrompt constructs the full prompt for an op: system prompt + parent outputs + input.
func buildOpPrompt(op *Op, input string, outputs map[string]string, mu *sync.Mutex) string {
	var parts []string

	if op.Prompt != "" {
		parts = append(parts, op.Prompt)
	}

	// Append parent outputs as context.
	mu.Lock()
	for _, parent := range op.InputOps {
		if out, ok := outputs[parent.ID]; ok {
			parts = append(parts, fmt.Sprintf("[%s output]: %s", parent.ID, out))
		}
	}
	mu.Unlock()

	// For start ops, append the user's input.
	if len(op.InputOps) == 0 {
		parts = append(parts, input)
	}

	return strings.Join(parts, "\n\n")
}

// routeOp selects a model for an op. Uses cache-aware routing:
// if a parent op used a model at the same tier, prefer it (KV cache reuse).
func (s *Server) routeOp(op *Op, prompt string, tierCache map[int]*ModelConfig) *ModelConfig {
	// Explicit model override.
	if op.Model != "" {
		for i := range s.cfg.Models {
			if s.cfg.Models[i].Name == op.Model || s.cfg.Models[i].Model == op.Model {
				return &s.cfg.Models[i]
			}
		}
	}

	tier := op.Tier
	if tier == 0 {
		// Auto-classify from prompt content.
		req := &ChatRequest{
			Messages: []Message{{Role: "user", Content: prompt}},
		}
		_, tier = s.classifier.Classify(req)
	}

	// Cache-aware: if we already used a model at this tier, reuse it.
	if cached, ok := tierCache[tier]; ok {
		return cached
	}

	// Find cheapest at or above tier.
	for i := range s.cfg.Models {
		if s.cfg.Models[i].Tier >= tier {
			return &s.cfg.Models[i]
		}
	}
	if len(s.cfg.Models) > 0 {
		return &s.cfg.Models[len(s.cfg.Models)-1]
	}
	return nil
}

// callModel sends a chat completion request to a specific model and returns the response text.
func (s *Server) callModel(ctx context.Context, model *ModelConfig, prompt string, maxTokens int) (string, error) {
	req := ChatRequest{
		Model: model.Model,
		Messages: []Message{
			{Role: "user", Content: prompt},
		},
	}
	if maxTokens > 0 {
		req.MaxTokens = &maxTokens
	}

	body, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("marshal: %w", err)
	}

	url := strings.TrimRight(model.BaseURL, "/") + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if model.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+model.APIKey)
	}

	resp, err := s.client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("upstream: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("upstream %d: %s", resp.StatusCode, truncateStr(string(respBody), 200))
	}

	// Parse OpenAI response format.
	var chatResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &chatResp); err != nil {
		return "", fmt.Errorf("parse response: %w", err)
	}
	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return chatResp.Choices[0].Message.Content, nil
}

func truncateStr(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
