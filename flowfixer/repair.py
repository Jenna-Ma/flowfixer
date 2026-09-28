"""Stage 2 (part 1): Repair Patch Generation & Application (Sec. 3.4.1).

* ``generate_patch`` — LLM produces a set of atomic edits, guided by the
  diagnosed root cause, its repair strategy, violated specs, and (optionally)
  retrieved experience.
* ``apply_patch``    — deterministically applies the edits (insert/remove/
  replace/append/swap over nodes and configs) to produce the modified workflow.
"""
from __future__ import annotations

import json
from typing import List, Optional

from . import llm, prompts
from .schema import (
    Diagnosis, Edit, EditOp, Node, Patch, SymbolicTrace, Workflow,
)
from . import taxonomy


# --------------------------------------------------------------------------- #
# Patch generation                                                              #
# --------------------------------------------------------------------------- #
def _workflow_repr(wf: Workflow) -> str:
    lines = []
    for n in wf.nodes:
        lines.append(f"  {n.id} [{n.type}] config={_short(n.config)}")
    return "\n".join(lines)


def _short(obj, limit: int = 400) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "…"


def generate_patch(
    workflow: Workflow,
    trace: SymbolicTrace,
    diagnosis: Diagnosis,
    experience_block: str = "",
    use_repair_knowledge: bool = True,
) -> Patch:
    node = trace.get(diagnosis.responsible_node) or workflow.get_node(diagnosis.responsible_node)
    ver = diagnosis.verifications.get(diagnosis.responsible_node)
    violated = ([f"{r.assertion.code} -> {r.message}" for r in ver.violated_assertions()]
                if ver else [])

    if use_repair_knowledge:
        strategy_id = diagnosis.repair_strategy
        strategy_desc = taxonomy.REPAIR_STRATEGIES.get(strategy_id, "")
    else:
        # ablation (w/o Repair): no strategy steer; let the model choose freely
        strategy_id, strategy_desc = "(none)", "no repair-strategy guidance"

    prompt = prompts.patch_prompt(
        task=trace.task,
        node=node,
        diagnosis=diagnosis,
        workflow_repr=_workflow_repr(workflow),
        edges=[list(e) for e in workflow.edges],
        violated_texts=violated,
        strategy_id=strategy_id,
        strategy_desc=strategy_desc,
        experience_block=experience_block,
    )
    data = llm.chat_json(prompt, system=prompts.PATCH_SYSTEM, default={"edits": []})
    return _parse_patch(data)


def _parse_patch(data) -> Patch:
    edits: List[Edit] = []
    for e in data.get("edits", []):
        op_raw = str(e.get("op", "")).lower().strip()
        try:
            op = EditOp(op_raw)
        except ValueError:
            continue
        edits.append(Edit(
            op=op,
            target=str(e.get("target", "")),
            content=e.get("content"),
            position=e.get("position"),
        ))
    return Patch(edits=edits)


# --------------------------------------------------------------------------- #
# Patch application                                                             #
# --------------------------------------------------------------------------- #
def apply_patch(workflow: Workflow, patch: Patch) -> Workflow:
    """Return a NEW workflow with the patch applied (original is untouched)."""
    wf = workflow.clone()
    for edit in patch.edits:
        try:
            _apply_edit(wf, edit)
        except Exception:
            # a malformed edit should not crash the pipeline; it will simply be
            # caught later by structural assessment.
            continue
    return wf


def _apply_edit(wf: Workflow, edit: Edit) -> None:
    if edit.op == EditOp.INSERT:
        _insert(wf, edit)
    elif edit.op == EditOp.APPEND:
        _append(wf, edit)
    elif edit.op == EditOp.REMOVE:
        _remove(wf, edit)
    elif edit.op == EditOp.REPLACE:
        _replace(wf, edit)
    elif edit.op == EditOp.SWAP:
        _swap(wf, edit)


# ---- helpers -------------------------------------------------------------- #
def _field_path(target: str):
    """Parse 'nodeid.config.field' -> (nodeid, field); else (target, None)."""
    if ".config." in target:
        nid, field = target.split(".config.", 1)
        return nid, field
    return target, None


def _make_node(content, fallback_id: str) -> Node:
    if isinstance(content, dict) and ("id" in content or "type" in content):
        return Node(
            id=str(content.get("id", fallback_id)),
            type=str(content.get("type", "code")),
            config=dict(content.get("config", {}) if isinstance(content.get("config"), dict) else {}),
            name=content.get("name"),
        )
    # content is a description string -> synthesize a code/validation node
    cfg = {"description": content} if isinstance(content, str) else {"content": content}
    return Node(id=fallback_id, type="code", config=cfg)


def _unique_id(wf: Workflow, base: str) -> str:
    if base not in wf.node_ids():
        return base
    i = 1
    while f"{base}_{i}" in wf.node_ids():
        i += 1
    return f"{base}_{i}"


def _insert(wf: Workflow, edit: Edit) -> None:
    nid, field = _field_path(edit.target)
    if field is not None:                         # insert a config field
        node = wf.get_node(nid)
        if node is not None:
            node.config[field] = edit.content
        return
    # insert a new node
    new_id = _unique_id(wf, str(edit.target) or "inserted_node")
    node = _make_node(edit.content, new_id)
    node.id = new_id
    pos = edit.position or ""
    if pos.startswith("between:"):
        _, a, b = pos.split(":", 2)
        wf.nodes.append(node)
        wf.edges = [e for e in wf.edges if not (e[0] == a and e[1] == b)]
        wf.edges.extend([(a, new_id), (new_id, b)])
    elif pos.startswith("after:"):
        x = pos.split(":", 1)[1]
        downs = wf.downstream(x)
        wf.nodes.append(node)
        wf.edges = [e for e in wf.edges if e[0] != x]
        wf.edges.append((x, new_id))
        wf.edges.extend([(new_id, d) for d in downs])
    elif pos.startswith("before:"):
        x = pos.split(":", 1)[1]
        ups = wf.upstream(x)
        wf.nodes.append(node)
        wf.edges = [e for e in wf.edges if e[1] != x]
        wf.edges.extend([(u, new_id) for u in ups])
        wf.edges.append((new_id, x))
    else:
        wf.nodes.append(node)                     # position unspecified: leave detached


def _append(wf: Workflow, edit: Edit) -> None:
    nid, field = _field_path(edit.target)
    node = wf.get_node(nid)
    if field is not None and node is not None:     # append text to a config field
        cur = node.config.get(field, "")
        if isinstance(cur, str):
            node.config[field] = (cur + "\n" + str(edit.content)).strip()
        elif isinstance(cur, list):
            cur.append(edit.content)
        else:
            node.config[field] = edit.content
        return
    # append a new node to the end of the workflow (after terminal nodes)
    new_id = _unique_id(wf, str(edit.target) or "appended_node")
    new_node = _make_node(edit.content, new_id)
    new_node.id = new_id
    terminals = [n.id for n in wf.nodes if not wf.downstream(n.id)]
    wf.nodes.append(new_node)
    wf.edges.extend([(t, new_id) for t in terminals])


def _remove(wf: Workflow, edit: Edit) -> None:
    nid, field = _field_path(edit.target)
    node = wf.get_node(nid)
    if field is not None and node is not None:      # remove a config field
        node.config.pop(field, None)
        return
    if node is None:
        return
    ups = wf.upstream(nid)
    downs = wf.downstream(nid)
    wf.nodes = [n for n in wf.nodes if n.id != nid]
    wf.edges = [e for e in wf.edges if e[0] != nid and e[1] != nid]
    # bypass: reconnect upstream directly to downstream
    for u in ups:
        for d in downs:
            if (u, d) not in wf.edges:
                wf.edges.append((u, d))


def _replace(wf: Workflow, edit: Edit) -> None:
    nid, field = _field_path(edit.target)
    node = wf.get_node(nid)
    if node is None:
        return
    if field is not None:                           # replace a config field value
        node.config[field] = edit.content
        return
    if isinstance(edit.content, dict):              # replace node type/config
        if "type" in edit.content:
            node.type = str(edit.content["type"])
        if isinstance(edit.content.get("config"), dict):
            node.config = dict(edit.content["config"])
        elif "config" not in edit.content:
            # treat remaining dict keys as config overrides
            for k, v in edit.content.items():
                if k not in ("id", "type", "name"):
                    node.config[k] = v
    else:
        node.config["content"] = edit.content


def _swap(wf: Workflow, edit: Edit) -> None:
    a = edit.target
    b = edit.content if isinstance(edit.content, str) else (
        edit.position.split(":", 1)[1] if edit.position and ":" in edit.position else edit.position)
    if not a or not b:
        return
    na, nb = wf.get_node(a), wf.get_node(b)
    if na is None or nb is None:
        return
    # swap by relabeling edges incident to a<->b
    def relabel(x):
        if x == a:
            return b
        if x == b:
            return a
        return x
    wf.edges = [(relabel(u), relabel(v)) for (u, v) in wf.edges]
