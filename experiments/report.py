#!/usr/bin/env python3
"""Aggregate all RQs into the paper's tables and figure.

Runs RQ1-RQ4 on the loaded dataset and emits:
  * Table 2 — overall effectiveness (RSR) + diagnosis accuracy (FAA / RCA)
  * Table 3 — ablation study
  * Fig. 5 — per-platform pass rate on unseen inputs (matplotlib PNG)

    python experiments/report.py --agentfail data/agentfail.json --n8n data/n8n.json \
        --fig out/fig5_generalization.png

Because everything routes through the same LLM client, run offline first with
FLOWFIXER_MOCK=1 to smoke-test the harness without spending tokens.
"""
from __future__ import annotations

import json
import os

import common  # noqa: F401  (adds project root to sys.path)
import run_rq1
import run_rq2
import run_rq3
import run_rq4


def _table2(rq1, rq2) -> str:
    lines = ["Table 2 — Overall effectiveness & diagnosis accuracy",
             "-" * 52,
             f"  RSR (overall)   {common.fmt_pct(rq1['RSR']['mean'])}",
             f"  FAA             {common.fmt_pct(rq2['FAA']['mean'])}",
             f"  RCA             {common.fmt_pct(rq2['RCA']['mean'])}"]
    for key in sorted(k for k in rq1 if k.startswith("RSR::")):
        plat = key.split("::", 1)[1]
        lines.append(f"  RSR [{plat:6s}]  {common.fmt_pct(rq1[key]['mean'])}")
    return "\n".join(lines)


def _table3(rq3) -> str:
    lines = ["Table 3 — Ablation study (RSR)", "-" * 52]
    full = rq3["full"]["mean"]
    for v in run_rq3.VARIANTS:
        m = rq3[v]["mean"]
        delta = "" if v == "full" else f"  ({100 * (m - full):+.1f}pt)"
        lines.append(f"  {v:16s}  {common.fmt_pct(m)}{delta}")
    return "\n".join(lines)


def _figure5(rq4, path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"[report] matplotlib unavailable ({e}); skipping Fig. 5")
        return
    plats = sorted({k.split("::", 1)[1] for k in rq4})
    orig = [rq4.get(f"orig::{p}", {}).get("mean", 0.0) * 100 for p in plats]
    rep = [rq4.get(f"repaired::{p}", {}).get("mean", 0.0) * 100 for p in plats]
    x = range(len(plats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - w / 2 for i in x], orig, w, label="original")
    ax.bar([i + w / 2 for i in x], rep, w, label="repaired")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plats)
    ax.set_ylabel("Pass rate on unseen inputs (%)")
    ax.set_title("Fig. 5 — Generalization of repaired workflows")
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"[report] wrote {path}")


def main() -> None:
    parser = common.base_parser("Aggregate RQ1-RQ4 into tables + Fig. 5.")
    parser.add_argument("--fig", default="out/fig5_generalization.png",
                        help="output path for Fig. 5 PNG.")
    args = parser.parse_args()
    cases = common.load_dataset(args)
    print(f"[report] {len(cases)} cases, {args.runs} run(s)\n")

    rq1 = common.run_multi(cases, run_rq1._per_run, runs=args.runs, seed=args.seed)
    rq2 = common.run_multi(cases, run_rq2._per_run, runs=args.runs, seed=args.seed)
    rq3 = {v: common.run_multi(cases, run_rq3._make_per_run(v),
                               runs=args.runs, seed=args.seed)["RSR"]
           for v in run_rq3.VARIANTS}
    rq4 = common.run_multi(cases, run_rq4._per_run, runs=args.runs, seed=args.seed)

    print("\n" + _table2(rq1, rq2))
    print("\n" + _table3(rq3))
    _figure5(rq4, args.fig)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"rq1": rq1, "rq2": rq2, "rq3": rq3, "rq4": rq4}, f,
                      indent=2, ensure_ascii=False)
        print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
