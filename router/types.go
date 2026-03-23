package router

import "encoding/json"

// ChatRequest is the OpenAI-compatible chat completion request format.
// All major providers (OpenAI, Anthropic via proxy, Ollama) support this.
type ChatRequest struct {
	Model          string          `json:"model,omitempty"`
	Messages       []Message       `json:"messages"`
	Temperature    *float64        `json:"temperature,omitempty"`
	MaxTokens      *int            `json:"max_tokens,omitempty"`
	Stream         bool            `json:"stream,omitempty"`
	Tools          []Tool          `json:"tools,omitempty"`
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	Stop           json.RawMessage `json:"stop,omitempty"`
}

// Message is a single chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Tool represents a function the model can call.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a callable function.
type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ResponseFormat specifies the output format.
type ResponseFormat struct {
	Type string `json:"type"` // "text" or "json_object"
}

// Usage tracks token usage from a completion response.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// RoutingDecision records the classifier's output for logging/stats.
type RoutingDecision struct {
	Score           int     `json:"score"`
	Tier            int     `json:"tier"`
	SelectedModel   string  `json:"selected_model"`
	Provider        string  `json:"provider"`
	EstimatedCostIn float64 `json:"estimated_cost_in"`
	ActualCostIn    float64 `json:"actual_cost_in,omitempty"`
	ActualCostOut   float64 `json:"actual_cost_out,omitempty"`
	SavedVsMax      float64 `json:"saved_vs_max,omitempty"` // $ saved vs using most expensive model
}
