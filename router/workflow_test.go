package router

import (
	"strings"
	"testing"
)

func TestParseWorkflowSimple(t *testing.T) {
	yaml := `
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
	wf, err := ParseWorkflow([]byte(yaml))
	if err != nil {
		t.Fatalf("ParseWorkflow: %v", err)
	}
	if len(wf.Ops) != 2 {
		t.Errorf("ops: got %d, want 2", len(wf.Ops))
	}
	if len(wf.StartOps) != 1 || wf.StartOps[0].ID != "reason" {
		t.Errorf("start_ops: got %v", wf.StartOps)
	}
	if len(wf.EndOps) != 1 || wf.EndOps[0].ID != "answer" {
		t.Errorf("end_ops: got %v", wf.EndOps)
	}
}

func TestParseWorkflowThreeStepChain(t *testing.T) {
	// Mirrors Halo's adv_reason_3 template.
	yaml := `
start_ops: [op0]
end_ops: [op2]
ops:
  op0:
    prompt: "Think step by step about this question."
    tier: 1
    max_tokens: 1024
    output_ops: [op1]
  op1:
    prompt: "Critique the reasoning above."
    tier: 2
    max_tokens: 1024
    input_ops: [op0]
    output_ops: [op2]
  op2:
    prompt: "Refine your answer considering the critique."
    tier: 2
    max_tokens: 256
    input_ops: [op1]
`
	wf, err := ParseWorkflow([]byte(yaml))
	if err != nil {
		t.Fatalf("ParseWorkflow: %v", err)
	}
	if len(wf.Ops) != 3 {
		t.Errorf("ops: got %d, want 3", len(wf.Ops))
	}

	// Check max distances.
	if wf.Ops["op0"].MaxDistance != 2 {
		t.Errorf("op0 max_distance: got %d, want 2", wf.Ops["op0"].MaxDistance)
	}
	if wf.Ops["op1"].MaxDistance != 1 {
		t.Errorf("op1 max_distance: got %d, want 1", wf.Ops["op1"].MaxDistance)
	}
	if wf.Ops["op2"].MaxDistance != 0 {
		t.Errorf("op2 max_distance: got %d, want 0", wf.Ops["op2"].MaxDistance)
	}
}

func TestParseWorkflowParallelBranches(t *testing.T) {
	// Two parallel branches merging at the end.
	yaml := `
start_ops: [a, b]
end_ops: [merge]
ops:
  a:
    prompt: "Analyze from perspective A."
    tier: 1
    output_ops: [merge]
  b:
    prompt: "Analyze from perspective B."
    tier: 1
    output_ops: [merge]
  merge:
    prompt: "Combine both perspectives."
    tier: 2
    input_ops: [a, b]
`
	wf, err := ParseWorkflow([]byte(yaml))
	if err != nil {
		t.Fatalf("ParseWorkflow: %v", err)
	}
	if len(wf.StartOps) != 2 {
		t.Errorf("start_ops: got %d, want 2", len(wf.StartOps))
	}

	// a and b should execute in parallel (same level).
	levels := computeLevels(wf)
	if len(levels) != 2 {
		t.Fatalf("levels: got %d, want 2", len(levels))
	}
	if len(levels[0]) != 2 {
		t.Errorf("level 0: got %d ops, want 2 (parallel a,b)", len(levels[0]))
	}
	if len(levels[1]) != 1 {
		t.Errorf("level 1: got %d ops, want 1 (merge)", len(levels[1]))
	}
}

func TestTopologicalOrder(t *testing.T) {
	yaml := `
start_ops: [a]
end_ops: [c]
ops:
  a:
    prompt: "first"
    tier: 1
    output_ops: [b]
  b:
    prompt: "second"
    tier: 1
    input_ops: [a]
    output_ops: [c]
  c:
    prompt: "third"
    tier: 1
    input_ops: [b]
`
	wf, err := ParseWorkflow([]byte(yaml))
	if err != nil {
		t.Fatalf("ParseWorkflow: %v", err)
	}

	order := wf.TopologicalOrder()
	if len(order) != 3 {
		t.Fatalf("topo order: got %d, want 3", len(order))
	}
	// a must come before b, b before c.
	idx := make(map[string]int)
	for i, op := range order {
		idx[op.ID] = i
	}
	if idx["a"] >= idx["b"] {
		t.Errorf("a should come before b: a=%d, b=%d", idx["a"], idx["b"])
	}
	if idx["b"] >= idx["c"] {
		t.Errorf("b should come before c: b=%d, c=%d", idx["b"], idx["c"])
	}
}

func TestCycleDetection(t *testing.T) {
	yaml := `
start_ops: [a]
end_ops: [b]
ops:
  a:
    prompt: "first"
    tier: 1
    output_ops: [b]
  b:
    prompt: "second"
    tier: 1
    input_ops: [a]
    output_ops: [a]
`
	_, err := ParseWorkflow([]byte(yaml))
	if err == nil {
		t.Error("expected cycle detection error, got nil")
	}
}

func TestParseWorkflowInvalidRef(t *testing.T) {
	yaml := `
start_ops: [a]
end_ops: [a]
ops:
  a:
    prompt: "test"
    tier: 1
    output_ops: [nonexistent]
`
	_, err := ParseWorkflow([]byte(yaml))
	if err == nil {
		t.Error("expected error for unknown op reference, got nil")
	}
}

func TestFormatDOT(t *testing.T) {
	yaml := `
start_ops: [a]
end_ops: [b]
ops:
  a:
    prompt: "first"
    tier: 1
    output_ops: [b]
  b:
    prompt: "second"
    tier: 2
    input_ops: [a]
`
	wf, err := ParseWorkflow([]byte(yaml))
	if err != nil {
		t.Fatalf("ParseWorkflow: %v", err)
	}
	dot := wf.FormatDOT()
	if dot == "" {
		t.Error("FormatDOT returned empty string")
	}
	if !strings.Contains(dot, "digraph") {
		t.Error("DOT output missing 'digraph'")
	}
}
