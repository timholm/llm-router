# llm-router

## Overview

llm-router is a cost-optimizing reverse proxy for LLM APIs written in Go. It classifies each incoming chat completion request by complexity using heuristic signals (token count, tool use, reasoning cues, conversation depth) and routes it to the cheapest model tier that can handle it. Drop-in replacement for the OpenAI API — just point your client at the router.

## Quick Start

```bash
go build -o bin/llm-router .
go test ./...
OPENAI_API_KEY=sk-... ./bin/llm-router --config config.yaml
```

## Key Files

- `main.go` — CLI entry point, flag parsing, starts server
- `router/config.go` — YAML config loader with env var expansion, model sorting
- `router/types.go` — OpenAI-compatible request/response types (ChatRequest, Message, Tool, etc.)
- `router/classifier.go` — Heuristic complexity scorer: extracts signals → computes score (0-100) → maps to tier (1-3)
- `router/proxy.go` — HTTP server: `/v1/chat/completions` (routing proxy), `/health`, `/stats`. Handles model selection, request rewriting, upstream proxying, cost tracking
- `config.yaml` — Example configuration with 3 OpenAI model tiers and pricing

## How to Extend

- **New classification signal**: Add field to `complexitySignals`, detect in `extractSignals()`, score in `computeScore()`
- **New provider**: Add to config.yaml — any OpenAI-compatible API works without code changes
- **New endpoint**: Register in `ListenAndServe()` mux, add handler to `Server` struct
- **Streaming support**: The proxy already passes through streaming responses from upstream

## Testing

```bash
go test ./...           # run all tests
go test ./router/...    # run router package tests only
```

Tests use `httptest.NewServer` for integration tests with a fake upstream. Classifier tests cover: simple requests, complex reasoning, tool use, JSON mode, multi-turn conversations, long context, score capping.
