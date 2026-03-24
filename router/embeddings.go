package router

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"math"
	"net/http"
	"strings"
	"unicode"
)

// EmbeddingsRequest is the OpenAI-compatible embeddings request.
type EmbeddingsRequest struct {
	Input interface{} `json:"input"` // string or []string
	Model string      `json:"model"`
}

// EmbeddingsResponse is the OpenAI-compatible embeddings response.
type EmbeddingsResponse struct {
	Object string           `json:"object"`
	Data   []EmbeddingData  `json:"data"`
	Model  string           `json:"model"`
	Usage  EmbeddingsUsage  `json:"usage"`
}

// EmbeddingData holds one embedding vector.
type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float64 `json:"embedding"`
	Index     int       `json:"index"`
}

// EmbeddingsUsage tracks token usage.
type EmbeddingsUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// handleEmbeddings generates text embeddings using a fast local hash-based approach.
// This produces 1536-dim vectors suitable for cosine similarity search in pgvector.
// Not as accurate as neural embeddings but works instantly with zero external dependencies.
//
// The algorithm: hash each word to a position in the vector, accumulate weighted values.
// Similar texts produce similar vectors because they share words.
func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req EmbeddingsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	// Normalize input to []string
	var inputs []string
	switch v := req.Input.(type) {
	case string:
		inputs = []string{v}
	case []interface{}:
		for _, item := range v {
			if s, ok := item.(string); ok {
				inputs = append(inputs, s)
			}
		}
	default:
		http.Error(w, "input must be a string or array of strings", http.StatusBadRequest)
		return
	}

	data := make([]EmbeddingData, len(inputs))
	totalTokens := 0

	for i, text := range inputs {
		vec := hashEmbed(text, 1536)
		data[i] = EmbeddingData{
			Object:    "embedding",
			Embedding: vec,
			Index:     i,
		}
		totalTokens += len(strings.Fields(text))
	}

	resp := EmbeddingsResponse{
		Object: "list",
		Data:   data,
		Model:  "local-hash-1536",
		Usage: EmbeddingsUsage{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// hashEmbed produces a dense vector from text using feature hashing.
// Each word is hashed to multiple positions in the vector, weighted by TF.
// The result is L2-normalized so cosine similarity works correctly.
func hashEmbed(text string, dim int) []float64 {
	vec := make([]float64, dim)

	// Tokenize and count word frequencies
	words := tokenize(text)
	freq := make(map[string]int)
	for _, w := range words {
		freq[w]++
	}

	totalWords := float64(len(words))
	if totalWords == 0 {
		return vec
	}

	// Hash each unique word to multiple vector positions
	for word, count := range freq {
		tf := float64(count) / totalWords

		// Use multiple hash functions for better distribution
		for h := 0; h < 4; h++ {
			hash := sha256.Sum256([]byte(word + string(rune(h))))
			pos := int(binary.BigEndian.Uint64(hash[:8])) % dim
			if pos < 0 {
				pos = -pos
			}

			// Sign from second part of hash
			sign := 1.0
			if hash[8]&1 == 1 {
				sign = -1.0
			}

			vec[pos] += sign * tf
		}
	}

	// L2 normalize
	var norm float64
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}

	return vec
}

// tokenize splits text into lowercase words, removing punctuation and stop words.
func tokenize(text string) []string {
	var words []string
	for _, word := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if len(word) > 2 && !embeddingStopWords[word] {
			words = append(words, word)
		}
	}
	return words
}

var embeddingStopWords = map[string]bool{
	"the": true, "and": true, "for": true, "are": true, "but": true,
	"not": true, "you": true, "all": true, "can": true, "was": true,
	"one": true, "our": true, "out": true, "has": true, "have": true,
	"this": true, "that": true, "with": true, "from": true, "they": true,
	"been": true, "said": true, "each": true, "which": true, "their": true,
	"will": true, "other": true, "about": true, "many": true, "then": true,
	"them": true, "these": true, "some": true, "would": true, "make": true,
	"like": true, "into": true, "could": true, "what": true, "does": true,
	"just": true, "than": true, "when": true, "there": true, "also": true,
	"how": true, "more": true, "its": true, "over": true, "such": true,
	"after": true, "most": true, "only": true, "being": true, "where": true,
}
