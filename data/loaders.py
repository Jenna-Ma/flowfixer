"""Dataset loaders for FLOWFIXER (Sec. 4.2).

The paper uses AgentFail (307 Dify/Coze cases) + 136 self-collected n8n cases.
Those are provided by the user; this module loads them into the canonical
``Case`` schema.  We accept two layouts:

  1. **Canonical FLOWFIXER JSON** (see data/README.md) — one JSON object per
     case, or a JSON list, or a directory of ``*.json`` files.
  2. **Best-effort AgentFail / n8n** — ``load_agentfail`` / ``load_n8n`` apply
     light field-name normalization on top of the canonical loader; adapt the
     ``_FIELD_ALIASES`` maps if your dump uses different keys.

Every loader returns ``List[Case]``.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List

from flowfixer.schema import Case, Workflow


# --------------------------------------------------------------------------- #
# Field aliases (extend these to match your raw dumps)                          #
# --------------------------------------------------------------------------- #
_FIELD_ALIASES = {
    "case_id": ["case_id", "id", "log_id", "uid"],
    "platform": ["platform", "source", "framework"],
    "task": ["task", "task_description", "query", "user_input", "goal", "input"],
    "workflow": ["workflow", "config", "workflow_config", "graph", "dsl"],
    "trajectory": ["trajectory", "trace", "execution_trace", "steps", "logs"],
    "final_output": ["final_output", "output", "result", "answer"],
    "test_input": ["test_input", "input", "query"],
    "gt_responsible_node": ["responsible_node", "failure_node", "root_node",
                            "gt_responsible_node", "faulty_node"],
    "gt_root_cause": ["root_cause", "gt_root_cause", "failure_cause",
                      "annotated_root_cause", "label"],
}


def _pick(d: Dict[str, Any], field: str, default=None):
    for key in _FIELD_ALIASES.get(field, [field]):
        if key in d and d[key] is not None:
            return d[key]
    return default


# --------------------------------------------------------------------------- #
# Workflow parsing                                                             #
# --------------------------------------------------------------------------- #
def _parse_workflow(raw: Any, platform: str, task: str) -> Workflow:
    if isinstance(raw, Workflow):
        return raw
    if not isinstance(raw, dict):
        return Workflow(platform=platform, task=task)
    wf = Workflow.from_dict(raw)
    if not wf.platform or wf.platform == "unknown":
        wf.platform = platform
    if not wf.task:
        wf.task = task
    # If edges are absent but the trajectory implies an order, leave empty here;
    # build_symbolic_trace fills upstream/downstream from whatever edges exist.
    return wf


def _parse_trajectory(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [s if isinstance(s, dict) else {"output": s} for s in raw]
    return []


# --------------------------------------------------------------------------- #
# Canonical case parsing                                                       #
# --------------------------------------------------------------------------- #
def parse_case(d: Dict[str, Any], idx: int = 0) -> Case:
    platform = str(_pick(d, "platform", "unknown"))
    task = str(_pick(d, "task", ""))
    workflow = _parse_workflow(_pick(d, "workflow", {}), platform, task)
    trajectory = _parse_trajectory(_pick(d, "trajectory", []))
    return Case(
        case_id=str(_pick(d, "case_id", f"case_{idx}")),
        platform=platform,
        task=task,
        workflow=workflow,
        trajectory=trajectory,
        final_output=_pick(d, "final_output"),
        test_input=_pick(d, "test_input", task),
        gt_responsible_node=(str(_pick(d, "gt_responsible_node"))
                             if _pick(d, "gt_responsible_node") is not None else None),
        gt_root_cause=(str(_pick(d, "gt_root_cause"))
                       if _pick(d, "gt_root_cause") is not None else None),
    )


# --------------------------------------------------------------------------- #
# Generic loader (file / dir / list)                                           #
# --------------------------------------------------------------------------- #
def load_cases(path: str, platform_filter: str = None) -> List[Case]:
    records: List[Dict[str, Any]] = []
    if os.path.isdir(path):
        for fp in sorted(glob.glob(os.path.join(path, "*.json"))):
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
            records.extend(obj if isinstance(obj, list) else [obj])
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        records = obj if isinstance(obj, list) else [obj]

    cases = [parse_case(d, i) for i, d in enumerate(records)]
    if platform_filter:
        cases = [c for c in cases if c.platform.lower() == platform_filter.lower()]
    return cases


# --------------------------------------------------------------------------- #
# Named loaders (Sec. 4.2)                                                      #
# --------------------------------------------------------------------------- #
def load_agentfail(path: str) -> List[Case]:
    """AgentFail dump (Dify + Coze). Adjust _FIELD_ALIASES for your schema."""
    cases = load_cases(path)
    for c in cases:
        if c.platform in ("unknown", ""):
            c.platform = "dify"        # AgentFail is Dify/Coze; default best-effort
    return cases


def load_n8n(path: str) -> List[Case]:
    cases = load_cases(path)
    for c in cases:
        if c.platform in ("unknown", ""):
            c.platform = "n8n"
    return cases


def load_all(agentfail_path: str = None, n8n_path: str = None,
             sample_path: str = None) -> List[Case]:
    """Merge datasets (Sec. 4.2 merges the two datasets for reporting)."""
    cases: List[Case] = []
    if agentfail_path and os.path.exists(agentfail_path):
        cases += load_agentfail(agentfail_path)
    if n8n_path and os.path.exists(n8n_path):
        cases += load_n8n(n8n_path)
    if not cases and sample_path and os.path.exists(sample_path):
        cases += load_cases(sample_path)
    return cases
