package router

import (
	"fmt"
	"os"
	"sort"

	"gopkg.in/yaml.v3"
)

// Config is the top-level configuration for llm-router.
type Config struct {
	Models     []ModelConfig       `yaml:"models"`
	Classifier ClassifierConfig    `yaml:"classifier"`
	Server     ServerConfig        `yaml:"server"`
	Stats      StatsConfig         `yaml:"stats"`
	Cache      CacheConfig         `yaml:"cache"`
	Learned    LearnedRouterConfig `yaml:"learned"`
}

// ModelConfig defines an LLM backend the router can send requests to.
type ModelConfig struct {
	Name          string  `yaml:"name"`           // human label, e.g. "gpt-4o-mini"
	Provider      string  `yaml:"provider"`       // "openai", "anthropic", "ollama"
	Model         string  `yaml:"model"`          // actual model ID sent to the API
	BaseURL       string  `yaml:"base_url"`       // API base URL
	APIKey        string  `yaml:"api_key"`        // API key (supports $ENV_VAR syntax)
	CostPer1KIn   float64 `yaml:"cost_per_1k_in"` // cost per 1K input tokens
	CostPer1KOut  float64 `yaml:"cost_per_1k_out"`// cost per 1K output tokens
	MaxTokens     int     `yaml:"max_tokens"`      // max context window
	Tier          int     `yaml:"tier"`            // capability tier: 1=basic, 2=mid, 3=advanced
}

// ClassifierConfig controls how request complexity is estimated.
type ClassifierConfig struct {
	// Thresholds for routing tiers based on estimated complexity score (0-100).
	Tier1Max int `yaml:"tier1_max"` // score <= this → tier 1 (cheapest)
	Tier2Max int `yaml:"tier2_max"` // score <= this → tier 2 (mid)
	// score > tier2_max → tier 3 (most capable)
}

// ServerConfig holds HTTP server settings.
type ServerConfig struct {
	ReadTimeoutSec  int `yaml:"read_timeout_sec"`
	WriteTimeoutSec int `yaml:"write_timeout_sec"`
}

// StatsConfig controls cost tracking.
type StatsConfig struct {
	LogPath string `yaml:"log_path"` // path to write cost savings log
}

// LoadConfig reads and validates a YAML config file.
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	// Expand environment variables in the YAML.
	expanded := os.ExpandEnv(string(data))

	var cfg Config
	if err := yaml.Unmarshal([]byte(expanded), &cfg); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	if len(cfg.Models) == 0 {
		return nil, fmt.Errorf("config: at least one model is required")
	}

	// Defaults.
	if cfg.Classifier.Tier1Max == 0 {
		cfg.Classifier.Tier1Max = 30
	}
	if cfg.Classifier.Tier2Max == 0 {
		cfg.Classifier.Tier2Max = 70
	}
	if cfg.Server.ReadTimeoutSec == 0 {
		cfg.Server.ReadTimeoutSec = 30
	}
	if cfg.Server.WriteTimeoutSec == 0 {
		cfg.Server.WriteTimeoutSec = 120
	}

	// Sort models by cost (cheapest first) within each tier.
	sort.Slice(cfg.Models, func(i, j int) bool {
		if cfg.Models[i].Tier != cfg.Models[j].Tier {
			return cfg.Models[i].Tier < cfg.Models[j].Tier
		}
		return cfg.Models[i].CostPer1KIn < cfg.Models[j].CostPer1KIn
	})

	return &cfg, nil
}
