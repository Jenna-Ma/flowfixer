"""Stage 1: Failure Diagnosis (Sec. 3.3).

* ``attribute_failure`` — computes a suspicious score per node from (1) assertion
  violation rate and (2) structural propagation impact, and returns the
  failure-responsible node (Sec. 3.3.1).
* ``analyze_root_cause`` — maps the responsible node to a taxonomy root cause
  using its symbolic context + assertion evidence (Sec. 3.3.2).
* ``diagnose`` — the full stage; ``diagnose_raw`` is the w/o-symbol ablation
  baseline that reasons over the raw trajectory instead.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import config as C
from . import llm, prompts
from .dsl import evaluate_specs
from .schema import (
    Diagnosis, NodeSpec, NodeVerification, SymbolicTrace,
)
from . import taxonomy


# --------------------------------------------------------------------------- #
# Static verification: check inferred specs against observed outputs            #
# --------------------------------------------------------------------------- #
def verify_all(
    trace: SymbolicTrace, specs: Dict[str, NodeSpec]
) -> Dict[str, NodeVerification]:
    verifications: Dict[str, NodeVerification] = {}
    for node in trace.nodes:
        spec = specs.get(node.id)
        results = evaluate_specs(spec.assertions, node, trace) if spec else []
        verifications[node.id] = NodeVerification(node_id=node.id, results=results)
    return verifications


# --------------------------------------------------------------------------- #
# 3.3.1 Failure Attribution — suspicious score                                  #
# --------------------------------------------------------------------------- #
def _abnormality(node, verification: NodeVerification) -> float:
    """Behavioral abnormality from assertion violations, boosted by a
    non-success execution status."""
    rate = verification.violation_rate
    if str(getattr(node, "status", "success")).lower() not in ("success", "ok", "200"):
        rate = max(rate, 0.5)
    return rate


def _propagation_impact(trace: SymbolicTrace, node_id: str) -> float:
    """Fraction of the workflow a node can influence downstream (earlier nodes
    influence more, hence higher potential to be the failure origin)."""
    n = len(trace.nodes)
    if n <= 1:
        return 0.0
    # downstream reach in execution order (nodes after it that it feeds).
    idx = trace.index_of(node_id)
    downstream_positions = max(0, n - 1 - idx)
    return downstream_positions / (n - 1)


def suspicious_scores(
    trace: SymbolicTrace, verifications: Dict[str, NodeVerification]
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for node in trace.nodes:
        ver = verifications.get(node.id, NodeVerification(node.id, []))
        abn = _abnormality(node, ver)
        prop = _propagation_impact(trace, node.id)
        scores[node.id] = C.ATTRIBUTION_ALPHA * abn + (1 - C.ATTRIBUTION_ALPHA) * prop
    return scores


def attribute_failure(
    trace: SymbolicTrace, verifications: Dict[str, NodeVerification]
) -> str:
    """Return the failure-responsible node id (highest suspicious score).

    Ties are broken toward the *earliest* abnormal node, since it is more likely
    to be the origin rather than a propagated symptom."""
    scores = suspicious_scores(trace, verifications)
    if not scores:
        return trace.nodes[0].id if trace.nodes else ""
    # sort by score desc, then execution order asc
    order = {n.id: i for i, n in enumerate(trace.nodes)}
    best = max(scores.items(), key=lambda kv: (kv[1], -order.get(kv[0], 0)))
    return best[0]


# --------------------------------------------------------------------------- #
# 3.3.2 Root Cause Analysis                                                     #
# --------------------------------------------------------------------------- #
def analyze_root_cause(
    trace: SymbolicTrace,
    responsible_id: str,
    verification: NodeVerification,
    use_taxonomy: bool = True,
) -> Dict[str, str]:
    node = trace.get(responsible_id)
    violated = [f"{r.assertion.dimension.value}: {r.assertion.code} -> {r.message}"
                for r in verification.violated_assertions()]
    satisfied = [f"{r.assertion.dimension.value}: {r.assertion.code}"
                 for r in verification.satisfied_assertions()]

    prompt = prompts.root_cause_prompt(node, trace.task, violated, satisfied)
    if not use_taxonomy:
        # ablation: drop the taxonomy prior from the prompt
        prompt = prompt.split("ROOT CAUSE TAXONOMY")[0] + (
            'Return STRICT JSON:\n'
            '{"root_cause": "<free-form root cause>", '
            '"failure_description": "<1-2 sentences>"}\n'
        )
    data = llm.chat_json(
        prompt, system=prompts.ROOT_CAUSE_SYSTEM,
        default={"root_cause": "Unknown", "failure_description": ""},
    )
    raw_name = data.get("root_cause", "Unknown")
    # snap to taxonomy (unless ablated) so downstream strategy lookup is valid
    name = taxonomy.get_root_cause(raw_name).name if use_taxonomy else raw_name
    return {"root_cause": name,
            "failure_description": data.get("failure_description", "")}


# --------------------------------------------------------------------------- #
# Full Stage-1 diagnosis                                                        #
# --------------------------------------------------------------------------- #
def diagnose(
    trace: SymbolicTrace,
    specs: Dict[str, NodeSpec],
    use_taxonomy: bool = True,
) -> Diagnosis:
    verifications = verify_all(trace, specs)
    responsible = attribute_failure(trace, verifications)
    rc = analyze_root_cause(trace, responsible, verifications[responsible],
                            use_taxonomy=use_taxonomy)
    strategy = taxonomy.strategy_for(rc["root_cause"]) if use_taxonomy else "R1"
    return Diagnosis(
        responsible_node=responsible,
        root_cause=rc["root_cause"],
        repair_strategy=strategy,
        failure_description=rc["failure_description"],
        suspicious_scores=suspicious_scores(trace, verifications),
        verifications=verifications,
    )


# --------------------------------------------------------------------------- #
# w/o Symbol ablation: diagnose directly from the raw trajectory                #
# --------------------------------------------------------------------------- #
RAW_DIAGNOSE_PROMPT = """\
[ROOT_CAUSE]
You are diagnosing a failed agentic workflow from its RAW execution trajectory
(no symbolic specifications available).

TASK GOAL:
{task}

RAW TRAJECTORY (execution order):
{trajectory}

Identify (a) the single failure-responsible node id and (b) its root cause.
{taxonomy_block}
Return STRICT JSON:
{{"responsible_node": "<node id>",
  "root_cause": "<{rc_hint}>",
  "failure_description": "<1-2 sentences>"}}
"""


def diagnose_raw(trace: SymbolicTrace, use_taxonomy: bool = True) -> Diagnosis:
    """Ablation variant used by RQ3 (w/o Symbol)."""
    traj_lines = []
    for i, n in enumerate(trace.nodes):
        traj_lines.append(
            f"{i}. id={n.id} type={n.type} status={n.status} "
            f"input={prompts._trunc(n.input, 300)} output={prompts._trunc(n.output, 300)}"
        )
    tax_block = ("ROOT CAUSE TAXONOMY (choose one name):\n" + taxonomy.taxonomy_prompt_block()
                 if use_taxonomy else "")
    rc_hint = "one taxonomy name" if use_taxonomy else "free-form"
    prompt = RAW_DIAGNOSE_PROMPT.format(
        task=trace.task, trajectory="\n".join(traj_lines),
        taxonomy_block=tax_block, rc_hint=rc_hint,
    )
    data = llm.chat_json(prompt, system=prompts.ROOT_CAUSE_SYSTEM,
                         default={"responsible_node": trace.nodes[0].id if trace.nodes else "",
                                  "root_cause": "Unknown", "failure_description": ""})
    responsible = str(data.get("responsible_node") or (trace.nodes[0].id if trace.nodes else ""))
    if trace.get(responsible) is None and trace.nodes:
        responsible = trace.nodes[0].id
    raw_rc = data.get("root_cause", "Unknown")
    name = taxonomy.get_root_cause(raw_rc).name if use_taxonomy else raw_rc
    strategy = taxonomy.strategy_for(name) if use_taxonomy else "R1"
    empty_ver = {n.id: NodeVerification(n.id, []) for n in trace.nodes}
    return Diagnosis(
        responsible_node=responsible,
        root_cause=name,
        repair_strategy=strategy,
        failure_description=data.get("failure_description", ""),
        suspicious_scores={},
        verifications=empty_ver,
    )
