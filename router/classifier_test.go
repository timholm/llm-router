package router

import "testing"

func TestClassifySimpleRequest(t *testing.T) {
	c := &Classifier{Tier1Max: 30, Tier2Max: 70}

	req := &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "What is 2+2?"},
		},
	}

	score, tier := c.Classify(req)
	if tier != 1 {
		t.Errorf("simple request: tier=%d (score=%d), want tier=1", tier, score)
	}
	if score > 30 {
		t.Errorf("simple request: score=%d, want <= 30", score)
	}
}

func TestClassifyComplexReasoningRequest(t *testing.T) {
	c := &Classifier{Tier1Max: 30, Tier2Max: 70}

	req := &ChatRequest{
		Messages: []Message{
			{Role: "system", Content: "You are a senior software architect. Analyze the following codebase and provide a detailed architectural review with pros and cons of the current design patterns. Think step by step about each component."},
			{Role: "user", Content: "Here's the codebase:\n```go\npackage main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"hello\")\n}\n```\nPlease analyze this step by step and evaluate the trade-offs."},
		},
	}

	score, tier := c.Classify(req)
	// Should be tier 3: has reasoning cues, code, long system prompt.
	if tier < 2 {
		t.Errorf("complex request: tier=%d (score=%d), want tier >= 2", tier, score)
	}
}

func TestClassifyToolUseRequest(t *testing.T) {
	c := &Classifier{Tier1Max: 30, Tier2Max: 70}

	req := &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "What's the weather in SF?"},
		},
		Tools: []Tool{
			{
				Type: "function",
				Function: ToolFunction{
					Name:        "get_weather",
					Description: "Get weather for a city",
				},
			},
		},
	}

	score, _ := c.Classify(req)
	// Tool use adds 15 points to the score. The classifier returns the raw tier;
	// the actual routing upgrade (tool use → tier 2+) happens in selectModel.
	if score < 15 {
		t.Errorf("tool-use request: score=%d, want >= 15 (tool use adds 15)", score)
	}
}

func TestClassifyJSONModeRequest(t *testing.T) {
	c := &Classifier{Tier1Max: 30, Tier2Max: 70}

	req := &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "List 3 colors"},
		},
		ResponseFormat: &ResponseFormat{Type: "json_object"},
	}

	score, tier := c.Classify(req)
	// JSON mode adds 10, basic message is ~5 = 15 total → tier 1.
	if score < 10 {
		t.Errorf("json request: score=%d, want >= 10", score)
	}
	_ = tier
}

func TestClassifyMultiTurnConversation(t *testing.T) {
	c := &Classifier{Tier1Max: 30, Tier2Max: 70}

	req := &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hi"},
			{Role: "assistant", Content: "Hello!"},
			{Role: "user", Content: "Tell me about Go"},
			{Role: "assistant", Content: "Go is a programming language..."},
			{Role: "user", Content: "How does concurrency work?"},
			{Role: "assistant", Content: "Go uses goroutines..."},
			{Role: "user", Content: "Can you compare goroutines to threads?"},
		},
	}

	score, _ := c.Classify(req)
	// 7 messages → 15 points from depth alone.
	if score < 15 {
		t.Errorf("multi-turn: score=%d, want >= 15", score)
	}
}

func TestClassifyLongContext(t *testing.T) {
	c := &Classifier{Tier1Max: 30, Tier2Max: 70}

	// Build a long message (~20K chars ≈ ~6500 tokens).
	long := ""
	for i := 0; i < 5000; i++ {
		long += "word "
	}

	req := &ChatRequest{
		Messages: []Message{
			{Role: "user", Content: long},
		},
	}

	score, tier := c.Classify(req)
	if score < 30 {
		t.Errorf("long context: score=%d, want >= 30", score)
	}
	if tier < 2 {
		t.Errorf("long context: tier=%d, want >= 2", tier)
	}
}

func TestScoreCapsAt100(t *testing.T) {
	c := &Classifier{Tier1Max: 30, Tier2Max: 70}

	// Everything that adds score: long prompt, system prompt, tools, JSON, code, reasoning, multi-turn.
	long := ""
	for i := 0; i < 5000; i++ {
		long += "word "
	}
	req := &ChatRequest{
		Messages: []Message{
			{Role: "system", Content: "You are a very detailed analyst. " + long},
			{Role: "user", Content: "```go\nfunc main() {}\n```\nAnalyze step by step the pros and cons."},
			{Role: "assistant", Content: "ok"},
			{Role: "user", Content: "more"},
			{Role: "assistant", Content: "ok"},
			{Role: "user", Content: "more"},
			{Role: "assistant", Content: "ok"},
			{Role: "user", Content: "evaluate the trade-offs"},
		},
		Tools:          []Tool{{Type: "function", Function: ToolFunction{Name: "test"}}},
		ResponseFormat: &ResponseFormat{Type: "json_object"},
	}

	score, _ := c.Classify(req)
	if score > 100 {
		t.Errorf("score=%d, should be capped at 100", score)
	}
}
