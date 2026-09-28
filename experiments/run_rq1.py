#!/usr/bin/env python3
"""RQ1 — Overall repair effectiveness (Sec. 4.6.1, Table 2).

Runs the full FLOWFIXER pipeline (repair_case) over every failure case and
reports the Repair Success Rate (RSR), overall and per platform, averaged over
NUM_RUNS with a cold-started experience pool per run (Sec. 4.5).

    python experiments/run_rq1.py --agentfail data/agentfail.json --n8n data/n8n.json
    python experiments/run_rq1.py            # uses data/sample/sample_cases.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm

import common  # noqa: F401  (adds project root to sys.path)
import metrics
from flowfixer.experience import ExperiencePool
from flowfixer.pipeline import repair_case


def _per_run(cases, run_idx: int) -> Dict[str, float]:
    pool = ExperiencePool()                 # cold start each run (Sec. 3.5)
    flags: List[bool] = []
    by_platform: Dict[str, List[bool]] = defaultdict(list)
    for case in tqdm(cases, desc=f"RQ1 run {run_idx + 1}", leave=False):
        rec = repair_case(case, pool=pool)
        flags.append(rec.success)
        by_platform[case.platform].append(rec.success)
    out = {"RSR": metrics.repair_success_rate(flags)}
    for plat, pf in by_platform.items():
        out[f"RSR::{plat}"] = metrics.repair_success_rate(pf)
    return out


def main() -> None:
    args = common.base_parser("RQ1: overall repair effectiveness (RSR).").parse_args()
    cases = common.load_dataset(args)
    print(f"[RQ1] {len(cases)} cases, {args.runs} run(s)")
    results = common.run_multi(cases, _per_run, runs=args.runs, seed=args.seed)

    print("\n=== RQ1: Repair Success Rate ===")
    for key in sorted(results):
        r = results[key]
        print(f"  {key:16s}  {common.fmt_pct(r['mean'])}  (±{common.fmt_pct(r['std'])})")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
