# llm-router

A cost-optimizing reverse proxy for LLM APIs. Drop it between your app and your LLM providers — it routes each request to the cheapest model that can handle it, caches semantically similar responses, deduplicates concurrent identical requests, and executes multi-step agent workflows as optimized DAGs.

Single binary. Zero external dependencies. 57 tests. 9.2MB.

Built from techniques in 7 research papers and open-source systems:

| Source | Paper/Repo | What we took |
|--------|-----------|-------------|
| FleetOpt | [arXiv:2603.16514](https://arxiv.org/abs/2603.16514) | Compress-and-route cost optimization |
| AMRO-S | [arXiv:2603.12933](https://arxiv.org/abs/2603.12933) | Intent-based request classification |
| TARo | [arXiv:2603.18411](https://arxiv.org/abs/2603.18411) | Token-level adaptive routing |
| Halo | [arXiv:2509.02121](https://arxiv.org/abs/2509.02121) | DAG workflow orchestration |
| Helium | [github.com/mlsys-io/helium_demo](https://github.com/mlsys-io/helium_demo) | Workflow-as-query-plan, proactive caching |
| kv.run | [github.com/mlsys-io/kv.run](https://github.com/mlsys-io/kv.run) | Worker lifecycle, health-aware routing |
| ParrotServe | [OSDI'24](https://github.com/microsoft/ParrotServe) | Request deduplication/batching |
| RouteLLM | [github.com/lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM) | ML-trained quality-aware routing, threshold calibration |
| GPTCache | [github.com/zilliztech/GPTCache](https://github.com/zilliztech/GPTCache) | Semantic similarity caching |

---

## What it does

```
Your App ──→ llm-router ──→ cheapest Claude model that works
                │
                ├── Cache hit? Return instantly ($0)
                ├── Duplicate in-flight? Wait for that one
                ├── Simple question? → Claude Haiku ($0.80/M tokens)
                ├── Moderate task? → Claude Sonnet ($3/M tokens)
                └── Complex reasoning? → Claude Opus ($15/M tokens)
```

**Without llm-router:** Every request goes to Claude Opus. You pay $15/M input tokens for "what's 2+2?"

**With llm-router:** Each request is classified, routed to the cheapest Claude model that can handle it, and cached for similar future queries. Typical savings: **40-80% on API costs.**

---

## Install

```bash
go install github.com/timholm/llm-router@latest
```

Or build from source:

```bash
git clone https://github.com/timholm/llm-router.git
cd llm-router
make build    # → bin/llm-router
make test     # 57 tests
```

---

## Quick Start

```bash
# 1. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 2. Start the router
./bin/llm-router --config config.yaml --addr :8080

# 3. Send requests (OpenAI-compatible format)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}]}'
```

That's it. Works with the Anthropic SDK or any OpenAI-compatible client:

```python
import anthropic

# Use through the router
client = anthropic.Anthropic(base_url="http://localhost:8080")
```

---

## Features

### 1. Cost-Optimized Routing

Every request is scored (0-100) based on complexity signals, then routed to the cheapest model tier that can handle it.

| Signal | Points | Why it matters |
|--------|--------|----------------|
| Token count | 5-35 | Longer contexts need more capable models |
| Conversation depth | 0-15 | Multi-turn conversations are harder |
| System prompt length | 0-10 | Complex instructions need stronger models |
| JSON mode | +10 | Structured output is harder |
| Tool/function calling | +15 | Requires capable models (forces tier 2+) |
| Code in prompt | +10 | Code understanding needs capability |
| Reasoning cues | +15 | "step by step", "analyze", "trade-offs" |

Routing headers on every response:
```
X-LLM-Router-Score: 15
X-LLM-Router-Tier: 1
X-LLM-Router-Model: gpt-4o-mini
X-LLM-Router-Cache: MISS
```

### 2. Semantic Caching

Caches LLM responses and returns them for semantically similar (not just identical) queries. Uses Jaccard similarity on normalized word sets — no external embedding model needed.

```yaml
# config.yaml
cache:
  enabled: true
  max_size: 1000          # entries
  ttl_sec: 300            # 5 minutes
  similarity_thresh: 0.85 # 0-1, higher = stricter matching
```

- "What is the capital of France?" → cached
- "What's France's capital city?" → cache **HIT** (similar enough)
- "How do I write Go code?" → cache **MISS** (different topic)

### 3. Request Deduplication

If 5 identical requests arrive at the same time, only one upstream call is made. All 5 callers get the same response. Saves money on burst traffic patterns.

### 4. DAG Workflow Execution

Define multi-step LLM pipelines as YAML and execute them with dependency tracking, parallel branches, and per-step cost-optimized routing.

```bash
curl http://localhost:8080/v1/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "input": "How many rs are in strawberry?",
    "workflow": "start_ops: [reason]\nend_ops: [refine]\nops:\n  reason:\n    prompt: \"Think step by step.\"\n    tier: 1\n    max_tokens: 1024\n    output_ops: [critique]\n  critique:\n    prompt: \"Find errors in the reasoning.\"\n    tier: 2\n    max_tokens: 512\n    input_ops: [reason]\n    output_ops: [refine]\n  refine:\n    prompt: \"Give a correct final answer.\"\n    tier: 2\n    max_tokens: 512\n    input_ops: [critique]"
  }'
```

Response:
```json
{
  "outputs": {
    "reason": "Let me count: s-t-r-a-w-b-e-r-r-y...",
    "critique": "The count is correct — there are 3 r's...",
    "refine": "There are 3 r's in strawberry."
  },
  "routing": {"reason": "claude-haiku", "critique": "claude-sonnet", "refine": "claude-sonnet"},
  "costs": {"reason": 0.0002, "critique": 0.003, "refine": 0.002},
  "total_ms": 4200,
  "final_output": "There are 3 r's in strawberry."
}
```

**Workflow features:**
- **Parallel branches** — independent ops run concurrently
- **Dependency tracking** — ops wait for all inputs before executing
- **Per-op routing** — each step routes to cheapest model for its tier
- **Cache-aware** — same-tier consecutive ops prefer the same model (KV cache reuse)
- **Cycle detection** — validates DAG before execution

See `workflows/` for example templates.

### 5. Health-Aware Backend Pool

Tracks the health, load, and latency of every backend in real time. Routes away from unhealthy or overloaded backends automatically.

```bash
curl http://localhost:8080/v1/backends
```
```json
[
  {"name": "claude-haiku", "status": "healthy", "in_flight": 3, "latency_ms": 120, "total_requests": 891},
  {"name": "claude-sonnet", "status": "healthy", "in_flight": 1, "latency_ms": 340, "total_requests": 298},
  {"name": "claude-opus", "status": "degraded", "in_flight": 0, "latency_ms": 1200, "total_requests": 58}
]
```

- **Unreachable** → marked `down`, skipped entirely
- **5xx responses** → marked `degraded`, deprioritized
- **Load balancing** — prefers backends with fewer in-flight requests
- **Latency tracking** — exponential moving average per backend

### 6. ML-Assisted Routing with Feedback Loop

The router starts with heuristic classification but gets smarter over time. Report quality scores via `/v1/feedback`, and the routing threshold auto-calibrates.

```bash
# Report that a tier-1 response was good quality
curl http://localhost:8080/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"score": 0.2, "tier": 1, "quality": 0.9}'
```

**Optional ML sidecar:** For production deployments, run RouteLLM's BERT classifier as a Python sidecar and point the router at it:

```yaml
learned:
  sidecar_url: http://localhost:6060/classify
  sidecar_weight: 0.7    # 70% ML, 30% heuristic
  threshold: 0.5
```

### 7. Cost Tracking

```bash
curl http://localhost:8080/stats
```
```json
{
  "total_requests": 1247,
  "total_saved_usd": 18.42,
  "total_spent_usd": 3.15,
  "by_model": {"claude-haiku": 891, "claude-sonnet": 298, "claude-opus": 58}
}
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat proxy with automatic routing |
| `/v1/workflows` | POST | Execute a DAG workflow |
| `/v1/feedback` | POST | Report quality for threshold calibration |
| `/v1/backends` | GET | Backend health and load states |
| `/health` | GET | Health check |
| `/stats` | GET | Cost savings and request counts |

---

## Configuration

```yaml
# config.yaml — full reference

models:
  - name: claude-haiku
    provider: anthropic
    model: claude-haiku-4-5-20251001
    base_url: https://api.anthropic.com
    api_key: $ANTHROPIC_API_KEY       # supports env var expansion
    cost_per_1k_in: 0.0008
    cost_per_1k_out: 0.004
    max_tokens: 200000
    tier: 1                            # cheapest tier

  - name: claude-sonnet
    provider: anthropic
    model: claude-sonnet-4-6
    base_url: https://api.anthropic.com
    api_key: $ANTHROPIC_API_KEY
    cost_per_1k_in: 0.003
    cost_per_1k_out: 0.015
    max_tokens: 200000
    tier: 2

  - name: claude-opus
    provider: anthropic
    model: claude-opus-4-6
    base_url: https://api.anthropic.com
    api_key: $ANTHROPIC_API_KEY
    cost_per_1k_in: 0.015
    cost_per_1k_out: 0.075
    max_tokens: 200000
    tier: 3                            # most capable

  # Also works with Ollama, vLLM, or any OpenAI-compatible backend:
  # - name: llama-local
  #   provider: ollama
  #   model: llama3.2
  #   base_url: http://localhost:11434
  #   cost_per_1k_in: 0               # self-hosted = free
  #   cost_per_1k_out: 0
  #   tier: 1

classifier:
  tier1_max: 30       # score 0-30 → claude-haiku
  tier2_max: 70       # score 31-70 → claude-sonnet
                      # score 71+ → claude-opus

cache:
  enabled: true
  max_size: 1000
  ttl_sec: 300
  similarity_thresh: 0.85

learned:
  threshold: 0.5
  # sidecar_url: http://localhost:6060/classify
  # sidecar_weight: 0.7

server:
  read_timeout_sec: 30
  write_timeout_sec: 120
```

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              llm-router                  │
                        │                                          │
  Client Request ──────▶│  ┌──────────────┐                       │
                        │  │ Semantic Cache│ Hit? → Return ($0)    │
                        │  └──────┬───────┘                       │
                        │         │ Miss                           │
                        │  ┌──────▼───────┐                       │
                        │  │  Deduplicator │ Duplicate? → Wait     │
                        │  └──────┬───────┘                       │
                        │         │ Unique                         │
                        │  ┌──────▼───────┐                       │
                        │  │ ML Classifier │ Score 0-1             │
                        │  │ (+ heuristic) │ (RouteLLM concept)   │
                        │  └──────┬───────┘                       │
                        │         │                                │
                        │  ┌──────▼───────┐                       │
                        │  │ Backend Pool  │ Health + load aware   │
                        │  │ (kv.run)      │ Pick best backend     │
                        │  └──────┬───────┘                       │
                        │         │                                │
                        └─────────┼────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼              ▼
             Claude Haiku   Claude Sonnet   Claude Opus
              (tier 1)        (tier 2)       (tier 3)
              $0.80/M         $3/M           $15/M
```

**Key source files:**

| File | What it does |
|------|-------------|
| `router/classifier.go` | Heuristic complexity scorer (0-100) |
| `router/learned.go` | ML-assisted routing with feedback calibration |
| `router/cache.go` | Semantic cache with Jaccard similarity |
| `router/dedup.go` | In-flight request deduplication |
| `router/pool.go` | Health-aware backend pool with load balancing |
| `router/workflow.go` | DAG parser, topological sort, cycle detection |
| `router/executor.go` | Parallel workflow execution with dependency tracking |
| `router/proxy.go` | HTTP server, routing pipeline, stats |
| `router/config.go` | YAML config with env var expansion |
| `router/types.go` | OpenAI-compatible request/response types |

---

## Use Cases

**1. Drop-in proxy for Claude API**
Point your Anthropic client at `localhost:8080`. Done. Instant cost savings.

**2. Multi-model routing**
Route between Claude Haiku, Sonnet, and Opus — or add local models via Ollama. The router picks the cheapest healthy backend.

**3. Agent workflow optimization**
Define multi-step chains (reason → critique → refine) as YAML DAGs. Each step routes independently. Parallel branches run concurrently.

**4. API cost management**
Track spending per model, see savings in real time, set up per-model tiers based on your budget.

**5. Production resilience**
Backends go down? Router detects it automatically and shifts traffic to healthy alternatives. No manual intervention.

---

## License

MIT
