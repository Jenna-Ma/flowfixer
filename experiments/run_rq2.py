#!/usr/bin/env python3
"""RQ2 — Diagnosis accuracy (Sec. 4.6.2, Table 2).

Runs Stage 1 only (diagnose_case) and compares against expert annotations:
  * FAA — Failure Attribution Accuracy (responsible node located correctly)
  * RCA — Root Cause Accuracy (correct taxonomy cause among correctly-attributed
          cases)

Only cases carrying ``gt_responsible_node`` contribute to FAA; only those with
``gt_root_cause`` contribute to RCA.

    python experiments/run_rq2.py --agentfail data/agentfail.json --n8n data/n8n.json
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from tqdm import tqdm

import common  # noqa: F401  (adds project root to sys.path)
import metrics
from flowfixer.pipeline import diagnose_case


def _per_run(cases, run_idx: int) -> Dict[str, float]:
    pred_nodes: List[Optional[str]] = []
    gt_nodes: List[Optional[str]] = []
    pred_rcs: List[Optional[str]] = []
    gt_rcs: List[Optional[str]] = []
    for case in tqdm(cases, desc=f"RQ2 run {run_idx + 1}", leave=False):
        diagnosis = diagnose_case(case)
        pred_nodes.append(diagnosis.responsible_node)
        pred_rcs.append(diagnosis.root_cause)
        gt_nodes.append(case.gt_responsible_node)
        gt_rcs.append(case.gt_root_cause)
    return {
        "FAA": metrics.failure_attribution_accuracy(pred_nodes, gt_nodes),
        "RCA": metrics.root_cause_accuracy(pred_nodes, gt_nodes, pred_rcs, gt_rcs),
    }


def main() -> None:
    args = common.base_parser("RQ2: diagnosis accuracy (FAA / RCA).").parse_args()
    cases = common.load_dataset(args)
    annotated = sum(1 for c in cases if c.gt_responsible_node)
    print(f"[RQ2] {len(cases)} cases ({annotated} annotated), {args.runs} run(s)")
    results = common.run_multi(cases, _per_run, runs=args.runs, seed=args.seed)

    print("\n=== RQ2: Diagnosis accuracy ===")
    for key in ("FAA", "RCA"):
        r = results[key]
        print(f"  {key}  {common.fmt_pct(r['mean'])}  (±{common.fmt_pct(r['std'])})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
