package router

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// WorkflowDef is a YAML-defined DAG of LLM operations.
// Inspired by Halo (arXiv:2509.02121) and Helium's workflow-as-query-plan model.
//
// Example YAML:
//
//	start_ops: [reason]
//	end_ops: [refine]
//	ops:
//	  reason:
//	    prompt: "Think step by step about this question."
//	    tier: 1
//	    max_tokens: 512
//	    output_ops: [critique]
//	  critique:
//	    prompt: "Critique the reasoning above. Find any errors."
//	    tier: 2
//	    max_tokens: 512
//	    input_ops: [reason]
//	    output_ops: [refine]
//	  refine:
//	    prompt: "Given the critique, provide a final correct answer."
//	    tier: 2
//	    max_tokens: 256
//	    input_ops: [critique]
type WorkflowDef struct {
	StartOps []string                `yaml:"start_ops"`
	EndOps   []string                `yaml:"end_ops"`
	Ops      map[string]OpDef        `yaml:"ops"`
}

// OpDef defines a single operation in the workflow DAG.
type OpDef struct {
	Prompt     string   `yaml:"prompt"`
	Tier       int      `yaml:"tier"`       // minimum model tier (0 = auto-classify)
	Model      string   `yaml:"model"`      // explicit model override (optional)
	MaxTokens  int      `yaml:"max_tokens"` // max output tokens
	InputOps   []string `yaml:"input_ops"`
	OutputOps  []string `yaml:"output_ops"`
	KeepCache  *bool    `yaml:"keep_cache"` // hint: keep KV cache for downstream reuse
}

// Op is a resolved operator in the execution DAG.
type Op struct {
	ID          string
	Prompt      string
	Tier        int
	Model       string
	MaxTokens   int
	InputOps    []*Op
	OutputOps   []*Op
	KeepCache   bool
	MaxDistance  int // longest path to any end op (for scheduling priority)
}

// Workflow is a parsed, validated, ready-to-execute DAG.
type Workflow struct {
	Ops      map[string]*Op
	StartOps []*Op
	EndOps   []*Op
}

// LoadWorkflow reads a workflow YAML file and builds the DAG.
func LoadWorkflow(path string) (*Workflow, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read workflow: %w", err)
	}
	return ParseWorkflow(data)
}

// ParseWorkflow parses a workflow from YAML bytes.
func ParseWorkflow(data []byte) (*Workflow, error) {
	var def WorkflowDef
	if err := yaml.Unmarshal(data, &def); err != nil {
		return nil, fmt.Errorf("parse workflow YAML: %w", err)
	}
	return buildWorkflow(&def)
}

// buildWorkflow validates the definition and constructs the Op graph.
func buildWorkflow(def *WorkflowDef) (*Workflow, error) {
	if len(def.Ops) == 0 {
		return nil, fmt.Errorf("workflow must have at least one op")
	}
	if len(def.StartOps) == 0 {
		return nil, fmt.Errorf("workflow must have at least one start_op")
	}
	if len(def.EndOps) == 0 {
		return nil, fmt.Errorf("workflow must have at least one end_op")
	}

	// Create op nodes.
	ops := make(map[string]*Op, len(def.Ops))
	for id, d := range def.Ops {
		maxTokens := d.MaxTokens
		if maxTokens == 0 {
			maxTokens = 256
		}
		keepCache := false
		if d.KeepCache != nil {
			keepCache = *d.KeepCache
		}
		ops[id] = &Op{
			ID:        id,
			Prompt:    d.Prompt,
			Tier:      d.Tier,
			Model:     d.Model,
			MaxTokens: maxTokens,
			KeepCache: keepCache,
		}
	}

	// Validate references and link edges.
	for id, d := range def.Ops {
		op := ops[id]
		for _, ref := range d.InputOps {
			target, ok := ops[ref]
			if !ok {
				return nil, fmt.Errorf("op %q references unknown input op %q", id, ref)
			}
			op.InputOps = append(op.InputOps, target)
		}
		for _, ref := range d.OutputOps {
			target, ok := ops[ref]
			if !ok {
				return nil, fmt.Errorf("op %q references unknown output op %q", id, ref)
			}
			op.OutputOps = append(op.OutputOps, target)
		}

		// Infer keep_cache: if any downstream op uses the same tier/model, cache reuse is valuable.
		if d.KeepCache == nil {
			for _, child := range op.OutputOps {
				if child.Tier == op.Tier || (child.Model != "" && child.Model == op.Model) {
					op.KeepCache = true
					break
				}
			}
		}
	}

	// Resolve start/end ops.
	startOps := make([]*Op, 0, len(def.StartOps))
	for _, id := range def.StartOps {
		op, ok := ops[id]
		if !ok {
			return nil, fmt.Errorf("unknown start_op %q", id)
		}
		startOps = append(startOps, op)
	}

	endOps := make([]*Op, 0, len(def.EndOps))
	for _, id := range def.EndOps {
		op, ok := ops[id]
		if !ok {
			return nil, fmt.Errorf("unknown end_op %q", id)
		}
		endOps = append(endOps, op)
	}

	// Compute max distance to end ops (for scheduling priority, ported from Halo).
	computeMaxDistances(ops, endOps)

	// Detect cycles.
	if err := detectCycles(ops); err != nil {
		return nil, err
	}

	return &Workflow{
		Ops:      ops,
		StartOps: startOps,
		EndOps:   endOps,
	}, nil
}

// computeMaxDistances annotates each op with the longest path (in edges) to any end op.
// Ported from Halo's _compute_max_distances.
func computeMaxDistances(ops map[string]*Op, endOps []*Op) {
	endSet := make(map[*Op]bool, len(endOps))
	for _, op := range endOps {
		endSet[op] = true
	}

	memo := make(map[*Op]int)
	var dfs func(op *Op) int
	dfs = func(op *Op) int {
		if v, ok := memo[op]; ok {
			return v
		}
		if endSet[op] {
			memo[op] = 0
			return 0
		}
		best := -1
		for _, child := range op.OutputOps {
			d := dfs(child)
			if d != -1 && d+1 > best {
				best = d + 1
			}
		}
		memo[op] = best
		return best
	}

	for _, op := range ops {
		op.MaxDistance = dfs(op)
	}
}

// detectCycles does a DFS cycle check on the op graph.
func detectCycles(ops map[string]*Op) error {
	const (
		white = 0 // unvisited
		gray  = 1 // in current path
		black = 2 // fully explored
	)
	color := make(map[*Op]int)

	var visit func(op *Op) error
	visit = func(op *Op) error {
		color[op] = gray
		for _, child := range op.OutputOps {
			switch color[child] {
			case gray:
				return fmt.Errorf("cycle detected: %s → %s", op.ID, child.ID)
			case white:
				if err := visit(child); err != nil {
					return err
				}
			}
		}
		color[op] = black
		return nil
	}

	for _, op := range ops {
		if color[op] == white {
			if err := visit(op); err != nil {
				return err
			}
		}
	}
	return nil
}

// TopologicalOrder returns ops in execution order (respecting dependencies).
func (w *Workflow) TopologicalOrder() []*Op {
	visited := make(map[*Op]bool)
	var order []*Op

	var visit func(op *Op)
	visit = func(op *Op) {
		if visited[op] {
			return
		}
		visited[op] = true
		for _, dep := range op.InputOps {
			visit(dep)
		}
		order = append(order, op)
	}

	for _, op := range w.StartOps {
		visit(op)
	}
	// Catch any ops not reachable from start (shouldn't happen in valid DAGs).
	for _, op := range w.Ops {
		visit(op)
	}
	return order
}

// FormatDOT returns a Graphviz DOT representation of the workflow.
func (w *Workflow) FormatDOT() string {
	var sb strings.Builder
	sb.WriteString("digraph workflow {\n")
	sb.WriteString("  rankdir=TB;\n")
	for _, op := range w.Ops {
		label := fmt.Sprintf("%s\\ntier=%d dist=%d", op.ID, op.Tier, op.MaxDistance)
		sb.WriteString(fmt.Sprintf("  %q [label=%q];\n", op.ID, label))
		for _, child := range op.OutputOps {
			sb.WriteString(fmt.Sprintf("  %q -> %q;\n", op.ID, child.ID))
		}
	}
	sb.WriteString("}\n")
	return sb.String()
}
