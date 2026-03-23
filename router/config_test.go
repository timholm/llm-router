package router

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")

	yaml := `
models:
  - name: cheap
    provider: openai
    model: gpt-4o-mini
    base_url: https://api.openai.com
    api_key: sk-test
    cost_per_1k_in: 0.00015
    cost_per_1k_out: 0.0006
    max_tokens: 128000
    tier: 1
  - name: expensive
    provider: openai
    model: gpt-4o
    base_url: https://api.openai.com
    api_key: sk-test
    cost_per_1k_in: 0.005
    cost_per_1k_out: 0.015
    max_tokens: 128000
    tier: 3
classifier:
  tier1_max: 25
  tier2_max: 65
`
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if len(cfg.Models) != 2 {
		t.Fatalf("models: got %d, want 2", len(cfg.Models))
	}
	// Should be sorted by tier then cost — cheap first.
	if cfg.Models[0].Name != "cheap" {
		t.Errorf("first model: got %q, want 'cheap'", cfg.Models[0].Name)
	}
	if cfg.Classifier.Tier1Max != 25 {
		t.Errorf("tier1_max: got %d, want 25", cfg.Classifier.Tier1Max)
	}
}

func TestLoadConfigEnvExpansion(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")

	t.Setenv("TEST_API_KEY", "sk-from-env")

	yaml := `
models:
  - name: test
    provider: openai
    model: gpt-4o-mini
    base_url: https://api.openai.com
    api_key: $TEST_API_KEY
    cost_per_1k_in: 0.00015
    cost_per_1k_out: 0.0006
    tier: 1
`
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadConfig(cfgPath)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Models[0].APIKey != "sk-from-env" {
		t.Errorf("api_key: got %q, want 'sk-from-env'", cfg.Models[0].APIKey)
	}
}

func TestLoadConfigNoModels(t *testing.T) {
	dir := t.TempDir()
	cfgPath := filepath.Join(dir, "config.yaml")

	yaml := `models: []`
	if err := os.WriteFile(cfgPath, []byte(yaml), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadConfig(cfgPath)
	if err == nil {
		t.Error("expected error with empty models, got nil")
	}
}
