# FLOWFIXER

A faithful, runnable reproduction of the algorithm in
**"Diagnosis-Driven Automatic Repair for Agentic Workflow via Symbolic Inference"**
(FLOWFIXER). This repo implements the full two-stage framework plus the RQ1–RQ4
evaluation harness. 


## What's implemented

| Paper section | Module |
|---|---|
| 3.2.1 Symbolic modeling (unified trace) | `flowfixer/symbolic.py`, `flowfixer/schema.py` |
| 3.2.2 Symbolic inference (behavioral assertions, Table 1 DSL) | `flowfixer/symbolic.py`, `flowfixer/dsl.py` |
| 3.3 Failure diagnosis (attribution + root-cause taxonomy, Fig. 4) | `flowfixer/diagnosis.py`, `flowfixer/taxonomy.py` |
| 3.4.1 Repair patch generation (5 atomic edits, R1–R7) | `flowfixer/repair.py` |
| 3.4.2 Pre-execution assessment (structural/semantic/consistency/offset) | `flowfixer/assessment.py` |
| 3.4.3 Dynamic verification (simulated exec + Multi-LLM judge + human hook) | `flowfixer/verification.py` |
| 3.5 Experience pool (online feedback + accumulated experience) | `flowfixer/experience.py` |
| Fig. 1 end-to-end diagnosis↔repair loop + RQ3 ablations | `flowfixer/pipeline.py` |
| All designed prompts | `flowfixer/prompts.py` |

## Setup

```bash
pip install -r requirements.txt
```

LLM access is centralized in `config.py`:

- `BASE_URL = https://api.vveai.com/v1` (OpenAI-compatible)
- `API_KEY`  — defaults to the provided key; override with `FLOWFIXER_API_KEY`
- backbone `gpt-5.2`, executor `gpt-5.2`, and four judge models
  (override via `FLOWFIXER_BACKBONE` / `FLOWFIXER_EXECUTOR` / `FLOWFIXER_JUDGES`)

Responses are disk-cached (`CACHE_DIR`) so repeated runs are cheap.

## Data

Bring your own AgentFail (Dify/Coze) and n8n dumps in the **canonical JSON
schema** documented in [`data/README.md`](data/README.md). Two runnable example
cases mirroring the paper's case studies live in `data/sample/`.

## Running the RQ harness

Every script accepts `--agentfail <path> --n8n <path>` (merged, per Sec. 4.2) and
falls back to the sample cases if neither is given. Results are averaged over
`NUM_RUNS=3` with a cold-started experience pool and randomized case order per run
(Sec. 4.5).

```bash
# RQ1 — overall repair effectiveness (RSR), Table 2
python experiments/run_rq1.py --agentfail data/agentfail.json --n8n data/n8n.json

# RQ2 — diagnosis accuracy (FAA / RCA), Table 2
python experiments/run_rq2.py --agentfail data/agentfail.json --n8n data/n8n.json

# RQ3 — ablation study, Table 3
python experiments/run_rq3.py --agentfail data/agentfail.json --n8n data/n8n.json

# RQ4 — generalization on 100 unseen inputs per workflow, Fig. 5
python experiments/run_rq4.py --agentfail data/agentfail.json --n8n data/n8n.json

# Everything at once -> Table 2, Table 3, and the Fig. 5 PNG
python experiments/report.py --agentfail data/agentfail.json --n8n data/n8n.json \
    --fig out/fig5_generalization.png --out out/report.json
```

Common flags: `--runs N`, `--limit N` (debug), `--seed S`, `--out results.json`.

## Offline smoke test (no API calls)

```bash
FLOWFIXER_MOCK=1 python tests/test_dsl.py
FLOWFIXER_MOCK=1 python tests/test_pipeline.py
FLOWFIXER_MOCK=1 python experiments/report.py --runs 1  
```

In MOCK mode the LLM layer returns deterministic, structurally-valid stubs, so
the full pipeline and all four RQ scripts execute without a network. The reported
numbers are placeholders — real metrics require the live endpoint.

## Metrics (Sec. 4.4)

- **RSR** — Repair Success Rate: fraction of cases whose repaired workflow passes
  dynamic verification.
- **FAA** — Failure Attribution Accuracy: responsible node located correctly.
- **RCA** — Root Cause Accuracy: correct taxonomy cause among correctly-attributed
  cases (taxonomy-aware matching).
- **Pass rate** (RQ4) — fraction of unseen inputs producing correct results,
  original vs. repaired, per platform.
