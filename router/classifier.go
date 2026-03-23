package router

import (
	"strings"
	"unicode/utf8"
)

// Classifier estimates request complexity and returns a score from 0-100.
// This is a heuristic classifier inspired by AMRO-S's intent classification
// approach but uses rule-based features instead of a fine-tuned SLM to keep
// the router dependency-free and fast (<1ms per classification).
//
// Signals used:
//   - Token count (proxy: word count * 1.3)
//   - System prompt complexity
//   - Presence of structured output requirements (JSON mode, function calling)
//   - Multi-turn conversation depth
//   - Code/reasoning indicators in the prompt
//   - Explicit complexity markers (chain-of-thought, step-by-step)
type Classifier struct {
	Tier1Max int
	Tier2Max int
}

// complexitySignals are the features extracted from a request.
type complexitySignals struct {
	estimatedTokens   int
	messageCount      int
	hasSystemPrompt   bool
	systemPromptLen   int
	hasJSON           bool
	hasTools          bool
	hasCodeIndicators bool
	hasReasoningCues  bool
	hasLongContext     bool
	maxMessageLen     int
}

// Classify returns a complexity score (0-100) and the recommended tier (1-3).
func (c *Classifier) Classify(req *ChatRequest) (score int, tier int) {
	sig := extractSignals(req)
	score = computeScore(sig)

	tier = 3
	if score <= c.Tier1Max {
		tier = 1
	} else if score <= c.Tier2Max {
		tier = 2
	}
	return score, tier
}

func extractSignals(req *ChatRequest) complexitySignals {
	var sig complexitySignals

	sig.messageCount = len(req.Messages)

	totalChars := 0
	for _, msg := range req.Messages {
		msgLen := utf8.RuneCountInString(msg.Content)
		totalChars += msgLen
		if msgLen > sig.maxMessageLen {
			sig.maxMessageLen = msgLen
		}

		if msg.Role == "system" {
			sig.hasSystemPrompt = true
			sig.systemPromptLen = msgLen
		}

		lower := strings.ToLower(msg.Content)

		// Code indicators.
		if containsAny(lower, "```", "func ", "def ", "class ", "import ", "const ", "SELECT ", "CREATE TABLE") {
			sig.hasCodeIndicators = true
		}

		// Reasoning cues.
		if containsAny(lower, "step by step", "chain of thought", "think carefully",
			"reason through", "analyze", "compare and contrast", "evaluate",
			"pros and cons", "trade-off", "tradeoff") {
			sig.hasReasoningCues = true
		}
	}

	// Estimate tokens (~1.3 tokens per word, ~4 chars per word).
	sig.estimatedTokens = (totalChars * 13) / (4 * 10)

	sig.hasLongContext = sig.estimatedTokens > 4000
	sig.hasJSON = req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object"
	sig.hasTools = len(req.Tools) > 0

	return sig
}

func computeScore(sig complexitySignals) int {
	score := 0

	// Token count contributes 0-35 points.
	switch {
	case sig.estimatedTokens < 200:
		score += 5
	case sig.estimatedTokens < 1000:
		score += 15
	case sig.estimatedTokens < 4000:
		score += 25
	default:
		score += 35
	}

	// Conversation depth: 0-15 points.
	switch {
	case sig.messageCount <= 2:
		score += 0
	case sig.messageCount <= 6:
		score += 8
	default:
		score += 15
	}

	// System prompt complexity: 0-10 points.
	if sig.hasSystemPrompt {
		if sig.systemPromptLen > 500 {
			score += 10
		} else {
			score += 5
		}
	}

	// Structured output: +10.
	if sig.hasJSON {
		score += 10
	}

	// Tool use: +15 (requires capable model).
	if sig.hasTools {
		score += 15
	}

	// Code: +10.
	if sig.hasCodeIndicators {
		score += 10
	}

	// Reasoning: +15.
	if sig.hasReasoningCues {
		score += 15
	}

	// Cap at 100.
	if score > 100 {
		score = 100
	}
	return score
}

func containsAny(s string, substrs ...string) bool {
	for _, sub := range substrs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}
