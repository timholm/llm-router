# llm-router

## Build & Test

```bash
make build    # builds bin/llm-router
make test     # runs go test ./...
make clean    # removes bin/
```

## Architecture

Single Go binary, two packages:
- `main.go` — CLI entry point (flag parsing, config loading, server start)
- `router/` — all logic:
  - `config.go` — YAML config with env var expansion, model sorting by tier/cost
  - `types.go` — OpenAI-compatible chat request/response types
  - `classifier.go` — heuristic complexity scorer (0-100) using token count, tool use, reasoning cues, conversation depth
  - `proxy.go` — HTTP server with four endpoints: `/v1/chat/completions`, `/v1/workflows`, `/health`, `/stats`
  - `workflow.go` — DAG parser (YAML → Op graph), topological sort, cycle detection, max-distance computation (ported from Halo)
  - `executor.go` — workflow execution: level-based parallel dispatch, dependency tracking, cache-aware model selection, cost tracking

## Patterns

- OpenAI-compatible API format throughout — works as drop-in proxy
- Config supports `$ENV_VAR` expansion for secrets
- Models sorted by (tier, cost) at config load time — routing just picks first match
- Classifier is pure functions, no side effects, <1ms per request
- Stats use sync.Mutex for thread safety

## Common Tasks

- **Add a new classification signal**: Add detection in `extractSignals()`, add points in `computeScore()`, update `complexitySignals` struct
- **Add a new provider**: Add to config.yaml, no code change needed if the provider has an OpenAI-compatible API
- **Adjust routing thresholds**: Change `classifier.tier1_max` and `classifier.tier2_max` in config.yaml
- **Add a new endpoint**: Add handler in `ListenAndServe()` mux, implement handler method on `Server`
- **Create a new workflow**: Add a YAML file in `workflows/` — define ops with prompts, tiers, and edges
- **Add a new workflow feature**: Workflow parsing is in `workflow.go`, execution in `executor.go`. Levels are computed in `computeLevels()`
