#!/usr/bin/env python3
"""RQ4 — Generalization of repaired workflows (Sec. 4.6.4, Fig. 5).

For each workflow we synthesize RQ4_UNSEEN_PER_WORKFLOW new in-domain inputs
(Sec. 4.6.4), then measure the pass rate of the ORIGINAL vs the REPAIRED workflow
on those unseen inputs, broken down per platform (Dify / Coze / n8n).

Only workflows that FLOWFIXER successfully repairs contribute a repaired variant;
the original pass rate is reported over the same input set for comparison.

    python experiments/run_rq4.py --agentfail data/agentfail.json --n8n data/n8n.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm

import common  # noqa: F401  (adds project root to sys.path)
import config as C
import metrics
from flowfixer import llm, prompts, verification as verif
from flowfixer.experience import ExperiencePool
from flowfixer.pipeline import repair_case


def generate_unseen_inputs(task: str, n: int) -> List[str]:
    """Synthesize ``n`` in-domain unseen inputs for a workflow's task."""
    data = llm.chat_json(prompts.unseen_prompt(task, n),
                         system=prompts.EXEC_SYSTEM, model=C.EXECUTOR_MODEL,
                         default={"inputs": []})
    inputs = [str(x) for x in data.get("inputs", []) if str(x).strip()]
    return inputs[:n]


def _pass_rate_over(workflow, task, inputs) -> float:
    flags = []
    for inp in inputs:
        res = verif.verify(workflow, inp, task_spec=task)
        flags.append(res.passed)
    return metrics.pass_rate(flags)


def _per_run(cases, run_idx: int) -> Dict[str, float]:
    pool = ExperiencePool()
    orig_by_plat: Dict[str, List[float]] = defaultdict(list)
    rep_by_plat: Dict[str, List[float]] = defaultdict(list)
    n = C.RQ4_UNSEEN_PER_WORKFLOW

    for case in tqdm(cases, desc=f"RQ4 run {run_idx + 1}", leave=False):
        rec = repair_case(case, pool=pool)
        if not rec.success or rec.final_workflow is None:
            continue                         # only repaired workflows generalize
        inputs = generate_unseen_inputs(case.task, n)
        if not inputs:
            continue
        orig_by_plat[case.platform].append(
            _pass_rate_over(case.workflow, case.task, inputs))
        rep_by_plat[case.platform].append(
            _pass_rate_over(rec.final_workflow, case.task, inputs))

    out: Dict[str, float] = {}
    all_orig, all_rep = [], []
    for plat in set(orig_by_plat) | set(rep_by_plat):
        o = _mean(orig_by_plat.get(plat, []))
        r = _mean(rep_by_plat.get(plat, []))
        out[f"orig::{plat}"] = o
        out[f"repaired::{plat}"] = r
        all_orig += orig_by_plat.get(plat, [])
        all_rep += rep_by_plat.get(plat, [])
    out["orig::overall"] = _mean(all_orig)
    out["repaired::overall"] = _mean(all_rep)
    return out


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    args = common.base_parser("RQ4: generalization on unseen inputs (Fig. 5).").parse_args()
    cases = common.load_dataset(args)
    print(f"[RQ4] {len(cases)} cases, {C.RQ4_UNSEEN_PER_WORKFLOW} unseen inputs each, "
          f"{args.runs} run(s)")
    results = common.run_multi(cases, _per_run, runs=args.runs, seed=args.seed)

    print("\n=== RQ4: Pass rate on unseen inputs (original -> repaired) ===")
    plats = sorted({k.split("::", 1)[1] for k in results})
    for plat in plats:
        o = results.get(f"orig::{plat}", {}).get("mean", 0.0)
        r = results.get(f"repaired::{plat}", {}).get("mean", 0.0)
        print(f"  {plat:10s}  {common.fmt_pct(o)} -> {common.fmt_pct(r)}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
