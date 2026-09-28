"""Symbolic Modeling & Inference (Sec. 3.2).

* ``build_symbolic_trace``  — normalizes a raw failed trajectory + workflow into
  the unified symbolic trace (Sec. 3.2.1).
* ``infer_specs``           — LLM-synthesizes node-wise behavioral specifications
  along the existence/temporal/causal dimensions (Sec. 3.2.2, Table 1, Fig. 2).
"""
from __future__ import annotations

from typing import Dict, List

from . import llm, prompts
from .dsl import validate_assertion, DSLValidationError
from .schema import (
    Assertion, NodeSpec, SpecDimension, SymbolicTrace, TraceNode, Workflow,
)


# --------------------------------------------------------------------------- #
# 3.2.1 Symbolic Modeling                                                       #
# --------------------------------------------------------------------------- #
def build_symbolic_trace(
    workflow: Workflow,
    raw_trajectory: List[Dict],
    task: str,
    final_output=None,
) -> SymbolicTrace:
    """Normalize a raw platform trajectory into a unified symbolic trace.

    ``raw_trajectory`` is a list of executed steps (in execution order); each is
    a dict with any subset of: id/node_id/name, type, input, output, status,
    config.  Missing structural info (upstream/downstream) is filled from the
    workflow graph so temporal/causal assertions can reason over it.
    """
    nodes: List[TraceNode] = []
    for step in raw_trajectory:
        nid = str(step.get("id") or step.get("node_id") or step.get("name") or f"node_{len(nodes)}")
        wf_node = workflow.get_node(nid)
        cfg = step.get("config")
        if cfg is None and wf_node is not None:
            cfg = wf_node.config
        ntype = step.get("type") or (wf_node.type if wf_node else "unknown")
        nodes.append(TraceNode(
            id=nid,
            type=ntype,
            input=step.get("input"),
            output=step.get("output"),
            status=step.get("status", "success"),
            config=dict(cfg or {}),
            name=step.get("name") or (wf_node.name if wf_node else nid),
            upstream=workflow.upstream(nid),
            downstream=workflow.downstream(nid),
        ))
    return SymbolicTrace(
        task=task, nodes=nodes, platform=workflow.platform,
        final_output=final_output,
    )


# --------------------------------------------------------------------------- #
# 3.2.2 Symbolic Inference                                                      #
# --------------------------------------------------------------------------- #
def infer_node_specs(node: TraceNode, task: str) -> NodeSpec:
    """Infer behavioral specifications for a single node via the LLM generator."""
    prompt = prompts.spec_prompt(node, task, node.upstream, node.downstream)
    data = llm.chat_json(
        prompt, system=prompts.SPEC_SYSTEM,
        default={"assertions": []},
    )
    assertions: List[Assertion] = []
    for item in data.get("assertions", []):
        code = (item.get("code") or "").strip()
        if not code:
            continue
        # normalize a bare predicate into an assert statement
        if not code.lstrip().startswith(("assert", "if", "for")):
            code = "assert " + code
        try:
            validate_assertion(code)      # keep only in-grammar assertions
        except DSLValidationError:
            continue
        dim = _parse_dim(item.get("dimension", "causal"))
        assertions.append(Assertion(
            node_id=node.id, dimension=dim, code=code,
            rationale=item.get("rationale", ""),
        ))
    return NodeSpec(node_id=node.id, assertions=assertions)


def infer_specs(trace: SymbolicTrace) -> Dict[str, NodeSpec]:
    """Infer specifications for every node in the trace."""
    return {n.id: infer_node_specs(n, trace.task) for n in trace.nodes}


def _parse_dim(raw: str) -> SpecDimension:
    r = (raw or "").lower()
    if "exist" in r:
        return SpecDimension.EXISTENCE
    if "temp" in r or "order" in r:
        return SpecDimension.TEMPORAL
    return SpecDimension.CAUSAL
