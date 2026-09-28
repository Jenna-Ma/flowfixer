"""Stage 2 (part 2): Multi-dimension Pre-execution Assessment (Sec. 3.4.2, Fig. 3).

Filters infeasible repair candidates before costly dynamic execution along four
dimensions:
  (a) Structural Correctness  — graph is well-formed (deterministic).
  (b) Semantic Correctness    — modified nodes' configs match what the task needs
                                (LLM, with modified configs masked then inferred).
  (c) Behavioral Consistency  — patch actually targets the diagnosed root cause
                                & violated specs (LLM).
  (d) Offset Rationality       — modification magnitude within a threshold
                                (deterministic).
The candidate passes only if all four pass.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import config as C
from . import llm, prompts
from .schema import (
    AssessmentResult, Diagnosis, Patch, Workflow,
)
from .schema import NodeCategory


# --------------------------------------------------------------------------- #
# (a) Structural Correctness                                                    #
# --------------------------------------------------------------------------- #
def _startish_ids(wf: Workflow) -> set:
    ids = set()
    for n in wf.nodes:
        indeg = len(wf.upstream(n.id))
        if indeg == 0 or n.category == NodeCategory.START_END and \
                any(k in (n.type + str(n.name)).lower() for k in ("start", "begin", "input")):
            ids.add(n.id)
    return ids


def _endish(n, wf: Workflow) -> bool:
    if n.category == NodeCategory.START_END:
        return True
    label = (n.type + str(n.name)).lower()
    return any(k in label for k in ("end", "answer", "output", "reply", "terminat"))


def assess_structural(
    original: Workflow, modified: Workflow
) -> Tuple[bool, str]:
    ids = set(modified.node_ids())
    # 1) edges reference existing nodes
    for (a, b) in modified.edges:
        if a not in ids or b not in ids:
            return False, f"edge ({a}->{b}) references a missing node"
    # 2) no duplicate node ids
    if len(ids) != len(modified.nodes):
        return False, "duplicate node ids after patch"
    orig_starts = _startish_ids(original) | _startish_ids(modified)
    # 3) dangling / broken-connection checks (catches the Fig. 7 violation:
    #    a node inserted but its successor not rewired)
    for n in modified.nodes:
        indeg = len(modified.upstream(n.id))
        outdeg = len(modified.downstream(n.id))
        is_start = n.id in orig_starts
        if indeg == 0 and not is_start:
            return False, f"node '{n.id}' has no upstream (broken connection)"
        if outdeg == 0 and not _endish(n, modified):
            return False, f"node '{n.id}' has no downstream (dangling node)"
    # 4) all nodes reachable from some start
    reachable = set(orig_starts)
    frontier = list(orig_starts)
    while frontier:
        cur = frontier.pop()
        for d in modified.downstream(cur):
            if d not in reachable:
                reachable.add(d)
                frontier.append(d)
    unreachable = ids - reachable
    if unreachable:
        return False, f"unreachable nodes: {sorted(unreachable)}"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Diffing to find modified nodes                                               #
# --------------------------------------------------------------------------- #
def modified_node_ids(original: Workflow, modified: Workflow) -> List[str]:
    orig = {n.id: n for n in original.nodes}
    changed: List[str] = []
    for n in modified.nodes:
        o = orig.get(n.id)
        if o is None or o.config != n.config or o.type != n.type:
            changed.append(n.id)
    return changed


# --------------------------------------------------------------------------- #
# (b) Semantic Correctness                                                      #
# --------------------------------------------------------------------------- #
def assess_semantic(
    original: Workflow, modified: Workflow, task: str
) -> Tuple[bool, str]:
    changed = modified_node_ids(original, modified)
    if not changed:
        return True, "no modified nodes"
    # render workflow with modified configs masked
    lines, actual = [], []
    for n in modified.nodes:
        if n.id in changed:
            lines.append(f"  {n.id} [{n.type}] config=<MASKED>")
            actual.append(f"  {n.id} [{n.type}] config={_short(n.config)}")
        else:
            lines.append(f"  {n.id} [{n.type}] config={_short(n.config)}")
    prompt = prompts.semantic_prompt(
        task=task, masked_workflow="\n".join(lines),
        edges=[list(e) for e in modified.edges],
        actual_configs="\n".join(actual),
    )
    data = llm.chat_json(prompt, system="You assess semantic correctness of "
                         "repaired agentic workflows.",
                         default={"pass": True, "reason": "unparsed->accept"})
    return bool(data.get("pass", True)), str(data.get("reason", ""))


# --------------------------------------------------------------------------- #
# (c) Behavioral Consistency                                                    #
# --------------------------------------------------------------------------- #
def assess_consistency(
    diagnosis: Diagnosis, patch: Patch
) -> Tuple[bool, str]:
    ver = diagnosis.verifications.get(diagnosis.responsible_node)
    violated = ([f"{r.assertion.code} -> {r.message}" for r in ver.violated_assertions()]
                if ver else [])
    prompt = prompts.consistency_prompt(
        root_cause=diagnosis.root_cause,
        violated_texts=violated,
        patch_summary=patch.summary() + " || contents: " + _patch_contents(patch),
    )
    data = llm.chat_json(prompt, system="You assess whether a repair patch "
                         "targets the diagnosed root cause.",
                         default={"pass": True, "reason": "unparsed->accept"})
    return bool(data.get("pass", True)), str(data.get("reason", ""))


# --------------------------------------------------------------------------- #
# (d) Offset Rationality                                                        #
# --------------------------------------------------------------------------- #
def assess_offset(original: Workflow, patch: Patch) -> Tuple[bool, str]:
    n_edits = len(patch)
    n_nodes = max(1, len(original.nodes))
    if n_edits < C.OFFSET_MIN_EDITS:
        return False, "modification too small (no effective edit)"
    ratio = n_edits / n_nodes
    if ratio > C.OFFSET_MAX_RATIO:
        return False, (f"modification too large: {n_edits} edits over {n_nodes} "
                       f"nodes (ratio {ratio:.2f} > {C.OFFSET_MAX_RATIO})")
    return True, f"magnitude ok (ratio {ratio:.2f})"


# --------------------------------------------------------------------------- #
# Full assessment                                                              #
# --------------------------------------------------------------------------- #
def assess(
    original: Workflow,
    modified: Workflow,
    patch: Patch,
    diagnosis: Diagnosis,
    task: str,
    enabled: Dict[str, bool] = None,
) -> AssessmentResult:
    enabled = enabled or {}
    structural = assess_structural(original, modified) if enabled.get("structural", True) else (True, "skipped")
    offset = assess_offset(original, patch) if enabled.get("offset", True) else (True, "skipped")
    # short-circuit the expensive LLM dims if a cheap deterministic one failed
    if not structural[0] or not offset[0]:
        semantic = (True, "skipped (deterministic dim failed)")
        consistency = (True, "skipped (deterministic dim failed)")
    else:
        semantic = assess_semantic(original, modified, task) if enabled.get("semantic", True) else (True, "skipped")
        consistency = assess_consistency(diagnosis, patch) if enabled.get("consistency", True) else (True, "skipped")
    passed = structural[0] and semantic[0] and consistency[0] and offset[0]
    return AssessmentResult(passed=passed, structural=structural, semantic=semantic,
                            consistency=consistency, offset=offset)


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #
def _short(obj, limit: int = 400) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "…"


def _patch_contents(patch: Patch) -> str:
    return " | ".join(f"{e.op.value}:{e.target}:{_short(e.content, 120)}" for e in patch.edits)
