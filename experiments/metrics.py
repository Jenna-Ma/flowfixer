"""Evaluation metrics (Sec. 4.4)."""
from __future__ import annotations

from typing import List, Optional

from flowfixer import taxonomy


def repair_success_rate(passed_flags: List[bool]) -> float:
    """RSR: ratio of cases whose modified workflow passes dynamic verification."""
    return _ratio(sum(1 for p in passed_flags if p), len(passed_flags))


def failure_attribution_accuracy(pred_nodes: List[Optional[str]],
                                 gt_nodes: List[Optional[str]]) -> float:
    """FAA: ratio of cases where the responsible node is located correctly."""
    total = sum(1 for gt in gt_nodes if gt is not None)
    correct = sum(1 for p, gt in zip(pred_nodes, gt_nodes)
                  if gt is not None and _node_match(p, gt))
    return _ratio(correct, total)


def root_cause_accuracy(pred_nodes: List[Optional[str]], gt_nodes: List[Optional[str]],
                        pred_rcs: List[Optional[str]], gt_rcs: List[Optional[str]]) -> float:
    """RCA: among cases with correct attribution, ratio with correct root cause
    (Sec. 4.4)."""
    denom = 0
    correct = 0
    for pn, gn, prc, grc in zip(pred_nodes, gt_nodes, pred_rcs, gt_rcs):
        if gn is None or not _node_match(pn, gn):
            continue
        if grc is None:
            continue
        denom += 1
        if _rc_match(prc, grc):
            correct += 1
    return _ratio(correct, denom)


def pass_rate(passed_flags: List[bool]) -> float:
    """RQ4: proportion of unseen inputs producing correct results."""
    return _ratio(sum(1 for p in passed_flags if p), len(passed_flags))


# --------------------------------------------------------------------------- #
# matching helpers                                                             #
# --------------------------------------------------------------------------- #
def _node_match(pred: Optional[str], gt: Optional[str]) -> bool:
    if pred is None or gt is None:
        return False
    return _norm(pred) == _norm(gt)


def _rc_match(pred: Optional[str], gt: Optional[str]) -> bool:
    if pred is None or gt is None:
        return False
    if _norm(pred) == _norm(gt):
        return True
    # snap both to the taxonomy and compare canonical names
    return taxonomy.get_root_cause(pred).name == taxonomy.get_root_cause(gt).name


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0
