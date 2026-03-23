package router

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestExecuteWorkflowEndToEnd(t *testing.T) {
	// Fake upstream that echoes the prompt back.
	callCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		var req ChatRequest
		json.NewDecoder(r.Body).Decode(&req)

		// Echo last message content as the response.
		content := "response"
		if len(req.Messages) > 0 {
			content = "processed: " + truncateStr(req.Messages[0].Content, 50)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"role": "assistant", "content": content}},
			},
		})
	}))
	defer upstream.Close()

	cfg := testConfig()
	for i := range cfg.Models {
		cfg.Models[i].BaseURL = upstream.URL
	}
	srv := NewServer(cfg)

	wfYAML := `
start_ops: [reason]
end_ops: [answer]
ops:
  reason:
    prompt: "Think step by step."
    tier: 1
    max_tokens: 512
    output_ops: [answer]
  answer:
    prompt: "Give a final answer."
    tier: 1
    max_tokens: 256
    input_ops: [reason]
`
	wf, err := ParseWorkflow([]byte(wfYAML))
	if err != nil {
		t.Fatalf("ParseWorkflow: %v", err)
	}

	result, err := srv.ExecuteWorkflow(context.Background(), wf, "What is 2+2?")
	if err != nil {
		t.Fatalf("ExecuteWorkflow: %v", err)
	}

	// Should have called upstream twice (reason + answer).
	if callCount != 2 {
		t.Errorf("upstream calls: got %d, want 2", callCount)
	}

	// Both ops should have output.
	if _, ok := result.Outputs["reason"]; !ok {
		t.Error("missing output for 'reason'")
	}
	if _, ok := result.Outputs["answer"]; !ok {
		t.Error("missing output for 'answer'")
	}

	// Should have routing info.
	if _, ok := result.Routing["reason"]; !ok {
		t.Error("missing routing for 'reason'")
	}

	// Final output should be from the end op.
	if result.FinalOut == "" {
		t.Error("final_output is empty")
	}

	// Timing should be positive.
	if result.TotalMs <= 0 {
		t.Errorf("total_ms=%d, want > 0", result.TotalMs)
	}
}

func TestExecuteWorkflowParallelBranches(t *testing.T) {
	callCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"role": "assistant", "content": "done"}},
			},
		})
	}))
	defer upstream.Close()

	cfg := testConfig()
	for i := range cfg.Models {
		cfg.Models[i].BaseURL = upstream.URL
	}
	srv := NewServer(cfg)

	wfYAML := `
start_ops: [a, b]
end_ops: [merge]
ops:
  a:
    prompt: "Perspective A."
    tier: 1
    output_ops: [merge]
  b:
    prompt: "Perspective B."
    tier: 1
    output_ops: [merge]
  merge:
    prompt: "Combine perspectives."
    tier: 2
    input_ops: [a, b]
`
	wf, err := ParseWorkflow([]byte(wfYAML))
	if err != nil {
		t.Fatalf("ParseWorkflow: %v", err)
	}

	result, err := srv.ExecuteWorkflow(context.Background(), wf, "Analyze this topic")
	if err != nil {
		t.Fatalf("ExecuteWorkflow: %v", err)
	}

	// 3 ops total.
	if callCount != 3 {
		t.Errorf("upstream calls: got %d, want 3", callCount)
	}
	if len(result.Outputs) != 3 {
		t.Errorf("outputs: got %d, want 3", len(result.Outputs))
	}
}

func TestWorkflowAPIEndpoint(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{"role": "assistant", "content": "answer"}},
			},
		})
	}))
	defer upstream.Close()

	cfg := testConfig()
	for i := range cfg.Models {
		cfg.Models[i].BaseURL = upstream.URL
	}
	srv := NewServer(cfg)

	reqBody := WorkflowRequest{
		Workflow: `
start_ops: [step1]
end_ops: [step1]
ops:
  step1:
    prompt: "Answer the question."
    tier: 1
    max_tokens: 256
`,
		Input: "What is Go?",
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/v1/workflows", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	srv.handleWorkflow(w, req)

	if w.Code != 200 {
		t.Fatalf("workflow API: status=%d, body=%s", w.Code, w.Body.String())
	}

	var result WorkflowResponse
	if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if result.FinalOut == "" {
		t.Error("final_output is empty")
	}
}
