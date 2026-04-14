# CausalBench System Design Spec

## 1. Purpose

CausalBench is a benchmark for evaluating how well LLMs reason about causal inference. It generates causal graph problems with machine-verified ground truth, prompts an LLM, and scores the LLM's answers against that ground truth. The two core tasks are **identification** (can a causal effect be computed from observational data?) and **backdoor criterion** (does a given variable set block all confounding paths?).

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        GRAPH GENERATION                         │
│                                                                 │
│   admg_stress_test.py                                           │
│   ┌───────────────┐    ┌──────────────┐    ┌────────────────┐  │
│   │ Random ADMG    │───>│ ananke       │───>│ stress_test_   │  │
│   │ Generator      │    │ OneLineID    │    │ results.json   │  │
│   │ (seed-based)   │    │              │    │                │  │
│   └───────────────┘    └──────────────┘    └───────┬────────┘  │
│                                                     │           │
└─────────────────────────────────────────────────────┼───────────┘
                                                      │
                                                      v
┌─────────────────────────────────────────────────────────────────┐
│                       BENCHMARK RUNNER                          │
│                                                                 │
│   benchmark_runner.py                                           │
│                                                                 │
│   ┌────────────────┐                                            │
│   │ Load results   │ <── stress_test_results.json               │
│   │ or generate    │     (vertices, edges, treatment, outcome,  │
│   │ fresh ADMGs    │      is_identified, functional)            │
│   └───────┬────────┘                                            │
│           │                                                     │
│           v                                                     │
│   ┌────────────────┐    ┌──────────────────────────────┐       │
│   │ For each graph │───>│ backdoor_oracle.py            │       │
│   │                │    │                                │       │
│   │                │    │  1. DoWhy identify_effect      │       │
│   │                │    │     -> valid backdoor sets     │       │
│   │                │    │                                │       │
│   │                │    │  2. nx.is_d_separator          │       │
│   │                │    │     -> check candidate sets    │       │
│   │                │    │                                │       │
│   │                │    │  Output: list of questions     │       │
│   │                │    │  with ground-truth answers     │       │
│   │                │    └──────────────┬───────────────┘       │
│   │                │                   │                        │
│   │                │                   v                        │
│   │                │    ┌──────────────────────────────┐       │
│   │                │───>│ prompt_templates.py           │       │
│   │                │    │                                │       │
│   │                │    │  Formats graph + questions     │       │
│   │                │    │  into structured text prompt   │       │
│   │                │    └──────────────┬───────────────┘       │
│   │                │                   │                        │
│   │                │                   v                        │
│   │                │    ┌──────────────────────────────┐       │
│   │                │───>│ Anthropic API                 │       │
│   │                │    │  claude-sonnet-4-20250514     │       │
│   │                │    │  or any model string          │       │
│   │                │    │                                │       │
│   │                │    │  Input:  system + user prompt  │       │
│   │                │    │  Output: JSON with yes/no     │       │
│   │                │    │          answers + reasoning   │       │
│   │                │    └──────────────┬───────────────┘       │
│   │                │                   │                        │
│   │                │                   v                        │
│   │                │    ┌──────────────────────────────┐       │
│   │                │───>│ Scoring                       │       │
│   │                │    │                                │       │
│   │                │    │  Compare model yes/no against  │       │
│   │                │    │  ground truth bool per question│       │
│   │                │    │                                │       │
│   │                │    │  Tag each score with source:   │       │
│   │                │    │    ananke | dowhy | d-sep      │       │
│   └────────────────┘    └──────────────┬───────────────┘       │
│                                        │                        │
│                                        v                        │
│                          ┌──────────────────────────────┐       │
│                          │ benchmark_<task>_<model>.json │       │
│                          └──────────────────────────────┘       │
│                                        │                        │
│                                        v                        │
│                          ┌──────────────────────────────┐       │
│                          │ Summary stats                 │       │
│                          │  - accuracy by task            │       │
│                          │  - accuracy by graph size      │       │
│                          │  - accuracy by GT source       │       │
│                          │  - TP/TN breakdown             │       │
│                          └──────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Components

### 3.1 Graph Generator (`admg_stress_test.py`)

Existing code from ADMG-gen. Produces random Acyclic Directed Mixed Graphs deterministically from integer seeds.

**Input:** seed (int)

**Process:**
1. Sample node count uniformly from [3, 39]
2. Shuffle nodes into a topological order
3. For each forward pair (i, j) where i < j in the order, add directed edge with probability drawn from U(0.2, 0.7)
4. For each unordered pair, add bidirected edge with probability drawn from U(0.1, 0.5)
5. Remove isolated nodes
6. Pick treatment = random node with at least one descendant
7. Pick outcome = random descendant of treatment
8. If no valid pair exists, retry with seed + 1000

**Output per graph:**
```
vertices:      list[str]          ["V0", "V1", ...]
di_edges:      list[(str, str)]   directed edges
bi_edges:      list[(str, str)]   bidirected edges
treatment:     str                intervention target
outcome:       str                effect target
```

**Ground truth (ananke):**
```
is_identified: bool               OneLineID result
functional:    str | null          identifying expression if identified
```

All results are serialized to `stress_test_results.json`.

### 3.2 Backdoor Oracle (`backdoor_oracle.py`)

Computes ground truth for backdoor criterion questions. Two backends:

**Backend 1: DoWhy** (`find_backdoor_sets_dowhy`)
- Converts ADMG to GML with explicit unobserved nodes for each bidirected edge
- Calls `CausalModel.identify_effect()` to find valid adjustment sets
- Used for: "does any valid backdoor set exist?" and "what is a valid set?"

**Backend 2: networkx d-separation** (`satisfies_backdoor`)
- Converts ADMG to augmented DAG (bidirected -> latent common cause nodes)
- Builds mutilated graph (remove all edges out of treatment)
- Calls `nx.is_d_separator(mutilated_graph, {X}, {Y}, candidate_set)`
- Also checks descendant condition (no node in candidate is a descendant of X)
- Used for: checking specific candidate sets

**Graph conversion (shared by both backends):**
```
ADMG:  V0 -> V1, V0 <-> V1

Augmented DAG:
  V0 -> V1
  U_V0_V1 -> V0
  U_V0_V1 -> V1
  U_V0_V1 is marked observed="no" in GML (for DoWhy)
```

**Question generation** (`generate_backdoor_questions`):

For each graph, produces up to 6 questions:

| # | Candidate set | Why | Source |
|---|---|---|---|
| 1 | {} (empty) | Baseline: are there any backdoor paths? | d-sep |
| 2 | pa(X) | Natural first guess for adjustment | d-sep |
| 3 | DoWhy's set | Known-valid set (if different from above) | dowhy |
| 4 | A descendant of X | Should always fail condition 1 | d-sep |
| 5 | Non-descendant, non-parent | Tests nuanced reasoning | d-sep |
| 6 | "__any__" | Does any valid set exist? | dowhy or d-sep fallback |

**Fallback behavior:** If DoWhy is not installed, questions 3 and 6 fall back. Question 3 is skipped. Question 6 derives its answer from whether any of the other candidate sets passed.

### 3.3 Frontdoor Oracle (`frontdoor_oracle.py`)

Computes ground truth for front-door criterion questions using networkx.

**Three conditions checked on the augmented DAG:**

| Condition | What it checks | How |
|---|---|---|
| (i) Interception | M intercepts all directed paths from X to Y | Remove M from graph, check if X can still reach Y via `nx.has_path` |
| (ii) No X-to-M confounding | No unblocked backdoor path from X to M | Remove edges out of X, check `nx.is_d_separator(G_mut, {X}, M, {})` |
| (iii) M-to-Y blocked by X | All backdoor paths from M to Y blocked by X | Remove edges out of each m in M, check `nx.is_d_separator(G_mut, M, {Y}, {X})` |

**Question generation** (`generate_frontdoor_questions`):

For each graph, produces up to 5 questions:

| # | Candidate set | Why | Source |
|---|---|---|---|
| 1 | A valid front-door set | If one exists (found by enumeration up to size 3) | d-sep |
| 2 | A direct mediator | Child of X that is also parent of Y | d-sep |
| 3 | Child of X not on path to Y | Should fail condition (i) | d-sep |
| 4 | Non-child of X | Should fail condition (i) differently | d-sep |
| 5 | "__any__" | Does any valid front-door set exist? | d-sep |

**Valid set enumeration:** `find_frontdoor_sets` checks all subsets of eligible variables (excluding X and Y) up to `max_size=3`. For each subset it verifies all three conditions. This is exponential but bounded by the size cap and practical for graphs under ~20 nodes.

### 3.4 Prompt Templates (`prompt_templates.py`)

Converts graph specs + question lists into structured text prompts.

**System prompt:** Instructs the model to respond with JSON only. Defines notation (directed vs bidirected edges), states the backdoor and front-door criteria formally, and specifies the response schema:
```json
{
  "answers": [
    {"question_id": "a", "answer": "Yes", "reasoning": "..."}
  ]
}
```

**Five prompt modes:**

| Mode | Questions |
|---|---|
| `build_identification_prompt` | (a) Is P(Y\|do(X)) identifiable? (b) Does any backdoor set exist? |
| `build_backdoor_prompt` | (a-f) Backdoor criterion checks for specific candidate sets |
| `build_combined_prompt` | (a) Identification, then (b-g) all backdoor questions |
| `build_frontdoor_prompt` | (a-e) Front-door criterion checks for specific candidate sets |
| `build_full_prompt` | (a) Identification, (b-g) backdoor, (h-l) front-door |

**Graph rendering format:**
```
Vertices: V0, V1, V2, V3
Directed edges (parent -> child):
  V2 -> V0
  V0 -> V1
  V0 -> V3
Bidirected edges (latent common cause):
  V0 <-> V3
Treatment: V0
Outcome: V1
```

The `__any__` candidate set renders as: "Does any subset of observed variables (excluding X and Y) satisfy the backdoor criterion for (X, Y)?"

### 3.4 Benchmark Runner (`benchmark_runner.py`)

Orchestrates the full pipeline.

**Input modes:**
- `--results path.json` : load existing stress test output
- `--generate N` : generate N fresh ADMGs (requires ananke on PYTHONPATH)

**Filters:**
- `--max-nodes K` : only benchmark graphs with <= K nodes
- Graphs with `error != null` in the stress test are skipped

**Task modes:**
- `--task id` : identification questions only
- `--task backdoor` : backdoor questions only
- `--task combined` : both in one prompt

**API interaction:**
- Uses `anthropic.Anthropic()` client (reads `ANTHROPIC_API_KEY` from env)
- System prompt + user prompt per graph
- Retries up to 3 times on rate limits (exponential backoff)
- Retries on JSON parse failures
- 0.5s delay between calls

**Scoring per question:**
```python
{
    "question":      "identification" | "backdoor_b" | "frontdoor_f" | ...,
    "candidate_set": [] | ["V2"] | "__any__",
    "model_answer":  True | False,
    "ground_truth":  True | False,
    "correct":       True | False,
    "reasoning":     "model's stated reasoning",
    "source":        "ananke" | "dowhy" | "d-sep",
}
```

**Summary output:**
- Overall accuracy
- Identification accuracy with TP/TN breakdown
- Backdoor accuracy with per-source breakdown (ananke / dowhy / d-sep)
- Frontdoor accuracy
- Accuracy by graph size bucket

## 4. Data Flow

```
seed
  |
  v
generate_random_admg(seed)
  |
  v
ADMG spec: {vertices, di_edges, bi_edges, treatment, outcome}
  |
  ├──> ananke OneLineID ──> is_identified (bool), functional (str)
  |
  ├──> DoWhy identify_effect ──> valid backdoor sets (list[set])
  |
  ├──> nx.is_d_separator (per candidate) ──> satisfies_backdoor (bool)
  |
  ├──> frontdoor_oracle ──> satisfies_frontdoor (bool) per candidate
  |     (path interception + d-sep conditions i, ii, iii)
  |
  ├──> prompt_templates ──> prompt (str)
  |       |
  |       v
  |    Anthropic API ──> model response (JSON)
  |       |
  |       v
  |    parse yes/no per question
  |       |
  |       v
  └──> compare model answers vs ground truth
          |
          v
       scored results (JSON)
```

## 5. Ground Truth Sources

| Question | Library | Function | What it returns |
|---|---|---|---|
| Is P(Y\|do(X)) identifiable? | ananke | `OneLineID.id()` | bool |
| Identifying functional | ananke | `OneLineID.functional()` | str expression |
| Valid backdoor sets | DoWhy | `CausalModel.identify_effect()` | list of variable sets |
| Does {Z} satisfy backdoor? | networkx | `is_d_separator()` on mutilated graph | bool |
| Does {M} satisfy front-door? | networkx | path interception + `is_d_separator()` x2 | bool |

The pipeline never relies on the LLM for ground truth. Every answer is verified against library output before scoring.

## 6. File Layout

```
ADMG-gen/
├── admg_stress_test.py          # graph generator + ananke oracle
├── visualize_admgs.py           # graph visualization
├── stress_test_results.json     # generated graphs + ID ground truth
├── admg_grid.pdf                # visual grid of all graphs
│
└── causal_bench/
    ├── benchmark_runner.py      # orchestrator
    ├── backdoor_oracle.py       # DoWhy + networkx ground truth
    ├── frontdoor_oracle.py      # networkx ground truth
    ├── prompt_templates.py      # prompt formatting
    │
    └── benchmark_full_claude-sonnet-4-20250514.json  # output
```

## 7. Output Schema

```json
{
  "model": "claude-sonnet-4-20250514",
  "task": "combined",
  "n_graphs": 14,
  "max_nodes": 6,
  "results": [
    {
      "graph_id": 23,
      "seed": 23,
      "n_nodes": 4,
      "treatment": "V0",
      "outcome": "V1",
      "ground_truth_identified": true,
      "backdoor_ground_truth": [
        {
          "candidate_set": [],
          "satisfies_backdoor": true,
          "reason": "empty set",
          "source": "d-sep"
        },
        {
          "candidate_set": ["V2"],
          "satisfies_backdoor": true,
          "reason": "parents of treatment",
          "source": "d-sep"
        },
        {
          "candidate_set": "__any__",
          "satisfies_backdoor": true,
          "reason": "dowhy identification check",
          "source": "dowhy"
        }
      ],
      "raw_response": { "answers": [ ... ] },
      "scores": [
        {
          "question": "identification",
          "model_answer": true,
          "ground_truth": true,
          "correct": true,
          "reasoning": "V2 is a valid backdoor adjustment set...",
          "source": "ananke"
        },
        {
          "question": "backdoor_b",
          "candidate_set": [],
          "model_answer": true,
          "ground_truth": true,
          "correct": true,
          "reasoning": "No backdoor paths exist...",
          "source": "d-sep"
        }
      ]
    }
  ]
}
```

## 8. Dependencies

| Package | Version | Used by | Purpose |
|---|---|---|---|
| ananke | py3.13 fork | admg_stress_test.py | OneLineID identification |
| networkx | >= 3.0 | backdoor_oracle.py | d-separation, graph ops |
| numpy | any | admg_stress_test.py | random generation |
| pandas | any | backdoor_oracle.py | DoWhy requires it |
| dowhy | any | backdoor_oracle.py | backdoor set finding |
| anthropic | any | benchmark_runner.py | Claude API client |
| matplotlib | any | visualize_admgs.py | graph rendering |

DoWhy is optional. Without it, the oracle falls back to d-sep for all backdoor questions, and the "DoWhy recommended set" question is skipped.

## 9. Extending

**Add a new question type:**
1. Add ground truth computation to `backdoor_oracle.py` (or a new oracle file)
2. Add prompt formatting to `prompt_templates.py`
3. Add scoring logic to `benchmark_runner.py`

**Add a new model:**
```bash
python benchmark_runner.py --results ../stress_test_results.json --model claude-opus-4-20250514
```

**Front-door criterion:** Already implemented in `frontdoor_oracle.py`. Use `--task frontdoor` or `--task full`.

**Add z-identification:**
Requires experimental data P(V | do(Z)). The z-ID templates from the original doc define the question format. Ground truth would come from ananke or a dedicated z-ID algorithm implementation. The benchmark runner already supports `--task` flags, so adding `--task zid` is a matter of wiring in a new oracle and templates following the same pattern as backdoor/frontdoor.

**Add do-calculus reasoning:**
Present the model with a graph and a target query, ask it to derive the identifying expression step by step using the three rules of do-calculus. Score by comparing the final expression against ananke's `functional` field. This tests derivation, not just yes/no classification.

**Scale to more graphs:**
The generator supports arbitrary seed ranges. For large-scale runs:
```bash
python admg_stress_test.py --random  # 100 random graphs
```
Or modify `run_stress_test(n_trials=1000)` directly. For benchmarks exceeding ~200 graphs, consider batching API calls or using the Anthropic batch API.