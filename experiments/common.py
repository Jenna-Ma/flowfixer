"""Shared utilities for the RQ experiment scripts.

Handles: sys.path bootstrap, dataset loading (AgentFail + n8n merged, Sec. 4.2),
common CLI args, and the multi-run averaging protocol (Sec. 4.5): each case is
processed in randomized order, the run repeats NUM_RUNS times, and the experience
pool is cold-started per run (Sec. 3.5 "Initialization").
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from typing import Callable, Dict, List

# --- make the project root importable when run as `python experiments/run_rqX.py`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as C  # noqa: E402
from data import loaders  # noqa: E402
from flowfixer.schema import Case  # noqa: E402


DEFAULT_SAMPLE = os.path.join(_ROOT, "data", "sample", "sample_cases.json")


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #
def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--agentfail", default=None,
                   help="path to AgentFail dump (Dify/Coze).")
    p.add_argument("--n8n", default=None, help="path to n8n dump.")
    p.add_argument("--sample", default=DEFAULT_SAMPLE,
                   help="fallback sample cases if no dataset given.")
    p.add_argument("--runs", type=int, default=C.NUM_RUNS,
                   help="number of repeated runs to average (Sec. 4.5).")
    p.add_argument("--limit", type=int, default=None,
                   help="cap number of cases (debugging).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write JSON results here.")
    return p


def load_dataset(args) -> List[Case]:
    cases = loaders.load_all(args.agentfail, args.n8n, args.sample)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No cases loaded. Provide --agentfail/--n8n or a valid --sample.")
    return cases


# --------------------------------------------------------------------------- #
# Multi-run averaging                                                          #
# --------------------------------------------------------------------------- #
def run_multi(
    cases: List[Case],
    per_run: Callable[[List[Case], int], Dict[str, float]],
    runs: int,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Execute ``per_run`` ``runs`` times over shuffled cases; return mean/std
    for each metric key it produces."""
    accum: Dict[str, List[float]] = {}
    for r in range(runs):
        rng = random.Random(seed + r)
        shuffled = cases[:]
        rng.shuffle(shuffled)              # randomized order (Sec. 4.5)
        metrics = per_run(shuffled, r)
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v)
    out: Dict[str, Dict[str, float]] = {}
    for k, vals in accum.items():
        out[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "runs": vals,
        }
    return out


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"
