package router

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// claudeCredentials represents the OAuth credentials stored by Claude Code.
type claudeCredentials struct {
	ClaudeAIOAuth *oauthToken `json:"claudeAiOauth"`
}

type oauthToken struct {
	AccessToken      string `json:"accessToken"`
	RefreshToken     string `json:"refreshToken"`
	ExpiresAt        int64  `json:"expiresAt"` // unix millis
	SubscriptionType string `json:"subscriptionType"`
	RateLimitTier    string `json:"rateLimitTier"`
}

// LoadClaudeOAuthToken reads the Claude Code OAuth access token from
// ~/.claude/.credentials.json. This allows llm-router to authenticate
// with Anthropic using Claude Code Max subscription credentials.
//
// Returns the access token string, or empty string if not found.
func LoadClaudeOAuthToken() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("get home dir: %w", err)
	}

	credPath := filepath.Join(home, ".claude", ".credentials.json")
	data, err := os.ReadFile(credPath)
	if err != nil {
		return "", fmt.Errorf("read credentials: %w", err)
	}

	var creds claudeCredentials
	if err := json.Unmarshal(data, &creds); err != nil {
		return "", fmt.Errorf("parse credentials: %w", err)
	}

	if creds.ClaudeAIOAuth == nil {
		return "", fmt.Errorf("no claudeAiOauth in credentials")
	}

	// Check if token is expired.
	expiresAt := time.UnixMilli(creds.ClaudeAIOAuth.ExpiresAt)
	if time.Now().After(expiresAt) {
		return "", fmt.Errorf("oauth token expired at %s (run 'claude login' to refresh)", expiresAt.Format(time.RFC3339))
	}

	return creds.ClaudeAIOAuth.AccessToken, nil
}

// ResolveAPIKey returns the API key for a model config.
// Priority:
//  1. Explicit api_key in config (may be $ENV_VAR expanded)
//  2. Claude Code OAuth token from ~/.claude/.credentials.json
//  3. Empty string (no auth)
func ResolveAPIKey(cfg ModelConfig) string {
	if cfg.APIKey != "" {
		return cfg.APIKey
	}

	// Try Claude Code OAuth for Anthropic provider.
	if cfg.Provider == "anthropic" {
		if token, err := LoadClaudeOAuthToken(); err == nil {
			return token
		}
	}

	return ""
}
