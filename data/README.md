# FLOWFIXER dataset format

Every case is a JSON object in the **canonical schema** below. You can supply:

- a single JSON file containing one object, **or a list** of objects, or
- a **directory** of `*.json` files (each one object or a list).

Point the experiment scripts at your files with `--agentfail <path>` and
`--n8n <path>` (see `experiments/`). The two datasets are merged for reporting,
as in the paper (Sec. 4.2).

## Canonical schema

```json
{
  "case_id": "iceland_travel_001",
  "platform": "dify",                 // dify | coze | n8n
  "task": "Plan a summer travel itinerary for Iceland, July 1-8.",
  "test_input": "I plan to travel to Iceland from July 1st to 8th, generate the travel plan.",

  "workflow": {                       // W = (N, E)   (Eq. 1)
    "nodes": [                        // n_i = (I_i, T_i, C_i)   (Eq. 2)
      {"id": "begin",   "type": "start", "config": {}},
      {"id": "decompose", "type": "llm", "config": {"prompt": "...", "model": "gpt-4o-mini"}}
    ],
    "edges": [["begin", "decompose"]] // E ⊆ N × N (control dependencies)
  },

  "trajectory": [                     // executed steps IN ORDER (Sec. 3.2.1)
    {
      "id": "begin", "type": "start",
      "input": "...", "output": "...",
      "status": "success",            // success | fail | error
      "config": {}
    }
  ],

  "final_output": "…the workflow's final (wrong) output…",

  // Expert annotations — ground truth for RQ2 (FAA / RCA):
  "gt_responsible_node": "decompose",
  "gt_root_cause": "Poor Prompt Design"   // one of the taxonomy names (see flowfixer/taxonomy.py)
}
```

### Notes

- `config` per node varies by type (LLM → `prompt`/`model`; tool → tool + params;
  code → `code`; retrieval → source/strategy), exactly as in Sec. 2.
- If `edges` are omitted, temporal/causal assertions have less structure to reason
  over; providing them improves attribution.
- `gt_root_cause` should match a taxonomy label so RCA scoring is exact; free-form
  labels are fuzzy-matched but exact strings are best.
- For RQ4, only `workflow` + `task` are needed on the *original* workflow; the
  script generates 100 unseen inputs per workflow.

The two files in `data/sample/` are runnable examples mirroring the paper's case
studies (Fig. 6 travel itinerary, Fig. 7 Q&A assistant).
