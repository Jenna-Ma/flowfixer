#!/usr/bin/env python3
"""RQ3 — Ablation study (Sec. 4.6.3, Table 3).

Re-runs the full repair pipeline under each ablated configuration and reports the
RSR drop relative to the complete FLOWFIXER. Variants (see pipeline.variant):

    full          - complete FLOWFIXER
    w/o symbol    - no symbolic modeling & inference
    w/o taxonomy  - no root-cause taxonomy prior
    w/o repair    - no repair-strategy knowledge
    w/o knowledge - no taxonomy + no repair knowledge
    w/o online    - no online feedback
    w/o experience- no accumulated experience
    w/o pool      - no experience pool at all

    python experiments/run_rq3.py --agentfail data/agentfail.json --n8n data/n8n.json
"""
from __future__ import annotations

import json
from typing import Dict, List

from tqdm import tqdm

import common  # noqa: F401  (adds project root to sys.path)
import metrics
from flowfixer.experience import ExperiencePool
from flowfixer.pipeline import repair_case, variant

VARIANTS = [
    "full", "w/o symbol", "w/o taxonomy", "w/o repair", "w/o knowledge",
    "w/o online", "w/o experience", "w/o pool",
]


def _make_per_run(variant_name: str):
    ablation = variant(variant_name)

    def _per_run(cases, run_idx: int) -> Dict[str, float]:
        pool = ExperiencePool()
        flags: List[bool] = []
        for case in tqdm(cases, desc=f"RQ3 [{variant_name}] run {run_idx + 1}",
                         leave=False):
            rec = repair_case(case, pool=pool, ablation=ablation)
            flags.append(rec.success)
        return {"RSR": metrics.repair_success_rate(flags)}

    return _per_run


def main() -> None:
    args = common.base_parser("RQ3: ablation study (Table 3).").parse_args()
    cases = common.load_dataset(args)
    print(f"[RQ3] {len(cases)} cases, {args.runs} run(s), {len(VARIANTS)} variants")

    results: Dict[str, Dict] = {}
    for v in VARIANTS:
        results[v] = common.run_multi(cases, _make_per_run(v),
                                      runs=args.runs, seed=args.seed)["RSR"]

    full_rsr = results["full"]["mean"]
    print("\n=== RQ3: Ablation (RSR) ===")
    print(f"  {'variant':16s}  {'RSR':>8s}  {'Δ vs full':>10s}")
    for v in VARIANTS:
        m = results[v]["mean"]
        delta = "" if v == "full" else f"{100 * (m - full_rsr):+.1f}pt"
        print(f"  {v:16s}  {common.fmt_pct(m):>8s}  {delta:>10s}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
