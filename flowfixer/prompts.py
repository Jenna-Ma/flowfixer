"""Prompt templates for every LLM-driven step in FLOWFIXER.

Each template embeds an ALL-CAPS marker (e.g. ``INFER_SPECS``) that the offline
mock in ``llm.py`` keys on.  Keeping prompts here makes them easy to inspect and
matches the paper's "designed prompts" replication artifact (Sec. 10).
"""
from __future__ import annotations

from typing import Any

from .taxonomy import taxonomy_prompt_block, strategy_prompt_block


# --------------------------------------------------------------------------- #
# 3.2.2 Symbolic Inference — behavioral specification generation               #
# --------------------------------------------------------------------------- #
SPEC_SYSTEM = (
    "You are a formal-specification generator for agentic-workflow nodes. "
    "You translate the expected behavior of a workflow node into executable "
    "assertions over its symbolic trace."
)

# BNF from Table 1, given to the model verbatim so generated code stays in-grammar.
DSL_BNF = """\
Grammar (assertions are Python-compatible; ONLY use these forms):
  Phi   ::= assert Pred                         # top-level assertion
  Pred  ::= Pred and Pred | Pred or Pred | not Pred | (Pred) | Expr comp Expr
  Expr  ::= value | var | var.attr | var.method(args)
  comp  ::= == | != | > | >= | < | <= | in | not in
  var   ::= identifier
  attr  ::= id | input | output | type | status | config
You may wrap an assertion in a single `if/else` or a `for` loop, and use the
builtins: len, any, all, range, enumerate, sum, min, max, sorted, isinstance.
Access the node under test as `node` (supports node.output and node["Output"]).
Access execution order via `trace.index(<node>)` and other nodes by their name.
Do NOT import anything, define functions, or use dunder attributes.
"""

SPEC_PROMPT = """\
[INFER_SPECS]
{bnf}

TASK GOAL:
{task}

NODE UNDER TEST:
  id: {node_id}
  type: {node_type}
  config: {node_config}
  observed input:  {node_input}
  observed output: {node_output}
  upstream nodes:   {upstream}
  downstream nodes: {downstream}

Infer behavioral specifications for this node along THREE dimensions:
  * existence — required fields/inputs/outputs/config are present & non-empty.
  * temporal  — this node executes in the semantically-correct order relative to
                other nodes (use trace.index comparisons).
  * causal    — variable consistency, interface compatibility, task-specific
                requirements, and upstream->downstream causal dependencies.

Return STRICT JSON:
{{"assertions": [
   {{"dimension": "existence|temporal|causal",
     "code": "<one assertion, python-compatible per grammar>",
     "rationale": "<short intent>"}}
]}}
Generate 2-6 assertions that are checkable against the observed output.
"""


def spec_prompt(node: Any, task: str, upstream, downstream) -> str:
    return SPEC_PROMPT.format(
        bnf=DSL_BNF,
        task=task,
        node_id=node.id,
        node_type=node.type,
        node_config=_trunc(node.config),
        node_input=_trunc(node.input),
        node_output=_trunc(node.output),
        upstream=upstream,
        downstream=downstream,
    )


# --------------------------------------------------------------------------- #
# 3.3.2 Root Cause Analysis                                                     #
# --------------------------------------------------------------------------- #
ROOT_CAUSE_SYSTEM = (
    "You are a failure-diagnosis expert for platform-orchestrated agentic "
    "workflows. You map a failing node to a single root cause from a fixed "
    "taxonomy, using the node's symbolic context and assertion-verification "
    "evidence."
)

ROOT_CAUSE_PROMPT = """\
[ROOT_CAUSE]
TASK GOAL:
{task}

FAILURE-RESPONSIBLE NODE (from failure attribution):
  id: {node_id}
  type: {node_type}
  config: {node_config}
  observed input:  {node_input}
  observed output: {node_output}

ASSERTION VERIFICATION EVIDENCE:
  violated assertions:
{violated}
  satisfied assertions:
{satisfied}

ROOT CAUSE TAXONOMY (choose EXACTLY one 'name'):
{taxonomy}

Return STRICT JSON:
{{"root_cause": "<one taxonomy name>",
  "failure_description": "<why this node caused the failure, 1-2 sentences>"}}
"""


def root_cause_prompt(node, task, violated_texts, satisfied_texts) -> str:
    return ROOT_CAUSE_PROMPT.format(
        task=task,
        node_id=node.id,
        node_type=node.type,
        node_config=_trunc(node.config),
        node_input=_trunc(node.input),
        node_output=_trunc(node.output),
        violated=_bullets(violated_texts) or "    (none)",
        satisfied=_bullets(satisfied_texts) or "    (none)",
        taxonomy=taxonomy_prompt_block(),
    )


# --------------------------------------------------------------------------- #
# 3.4.1 Repair Patch Generation                                                #
# --------------------------------------------------------------------------- #
PATCH_SYSTEM = (
    "You are a workflow-repair engine. You produce a minimal patch of atomic "
    "edit operators that fixes the diagnosed root cause while preserving the "
    "workflow's task objective and inter-node dependencies."
)

PATCH_PROMPT = """\
[GEN_PATCH]
TASK GOAL:
{task}

DIAGNOSIS:
  failure-responsible node: {node_id} (type: {node_type})
  root cause: {root_cause}
  chosen repair strategy: {strategy_id} - {strategy_desc}
  failure description: {failure_desc}
  violated specifications:
{violated}

CURRENT WORKFLOW (nodes and their config):
{workflow}
EDGES: {edges}

{experience}

Generate a repair patch as a set of atomic edits. Allowed operators:
  insert(target, content, position)  - add a node/config at a position
  remove(target, content, position)  - delete a node/config
  replace(target, content, position) - replace node/config with content
  append(target, content, position)  - add to the end / append config text
  swap(target, content, position)    - swap execution order of two nodes
For node insertion, use position like "after:<node_id>" or
"between:<a>:<b>"; for config edits, target is "<node_id>.config.<field>".

Return STRICT JSON:
{{"edits": [
   {{"op": "insert|remove|replace|append|swap",
     "target": "<node id or field path>",
     "content": <string or object>,
     "position": "<optional position spec or null>"}}
]}}
Keep the patch causally relevant to the root cause; do not make unrelated changes.
"""


def patch_prompt(task, node, diagnosis, workflow_repr, edges, violated_texts,
                 strategy_id, strategy_desc, experience_block="") -> str:
    return PATCH_PROMPT.format(
        task=task,
        node_id=node.id,
        node_type=node.type,
        root_cause=diagnosis.root_cause,
        strategy_id=strategy_id,
        strategy_desc=strategy_desc,
        failure_desc=diagnosis.failure_description,
        violated=_bullets(violated_texts) or "    (none)",
        workflow=workflow_repr,
        edges=edges,
        experience=experience_block or "(no prior experience)",
    )


# --------------------------------------------------------------------------- #
# 3.4.2 Pre-execution Assessment (semantic & behavioral-consistency dims)      #
# --------------------------------------------------------------------------- #
SEMANTIC_PROMPT = """\
[SEMANTIC_ASSESS]
TASK GOAL:
{task}

WORKFLOW (configs of MODIFIED nodes are MASKED as <MASKED>):
{masked_workflow}
EDGES: {edges}

For each MASKED node, infer the configuration it SHOULD have to satisfy both its
node-level function and the overall task, then compare with the ACTUAL config:

MODIFIED NODES (actual config after repair):
{actual_configs}

Decide whether the actual configs are semantically consistent with what the task
and workflow context require.

Return STRICT JSON: {{"pass": true|false, "reason": "<short>"}}
"""


def semantic_prompt(task, masked_workflow, edges, actual_configs) -> str:
    return SEMANTIC_PROMPT.format(
        task=task, masked_workflow=masked_workflow, edges=edges,
        actual_configs=actual_configs,
    )


CONSISTENCY_PROMPT = """\
[CONSISTENCY_ASSESS]
DIAGNOSED ROOT CAUSE: {root_cause}
VIOLATED BEHAVIORAL SPECIFICATIONS:
{violated}

REPAIR PATCH (high-level summary):
{patch_summary}

Does this patch directly address the diagnosed root cause and the violated
specifications, rather than introducing unrelated changes?

Return STRICT JSON: {{"pass": true|false, "reason": "<short>"}}
"""


def consistency_prompt(root_cause, violated_texts, patch_summary) -> str:
    return CONSISTENCY_PROMPT.format(
        root_cause=root_cause,
        violated=_bullets(violated_texts) or "    (none)",
        patch_summary=patch_summary,
    )


# --------------------------------------------------------------------------- #
# 3.4.3 Dynamic verification: simulated execution + Multi-LLM Judge            #
# --------------------------------------------------------------------------- #
EXEC_SYSTEM = (
    "You faithfully simulate the execution of an agentic workflow: given the "
    "node graph, each node's configuration, and a task input, you produce the "
    "final output the workflow would return, or report a runtime error."
)

EXEC_PROMPT = """\
[SIMULATE_EXEC]
TASK / INPUT:
{task_input}

WORKFLOW (execution order, node configs):
{workflow}
EDGES: {edges}

Simulate running this workflow on the input. Execute nodes in dependency order,
propagating outputs. If a structural/runtime error would occur, report it.

Return STRICT JSON:
{{"executed": true|false,
  "final_output": <the workflow's final output, or null>,
  "error": "<runtime error if executed=false, else empty>"}}
"""


def exec_prompt(task_input, workflow, edges) -> str:
    return EXEC_PROMPT.format(task_input=task_input, workflow=workflow, edges=edges)


JUDGE_SYSTEM = (
    "You are an impartial evaluator. Judge whether a workflow's output satisfies "
    "the task requirements, considering completeness, correctness, and explicit "
    "constraint compliance. Base your judgment ONLY on the task and the output."
)

JUDGE_PROMPT = """\
[JUDGE_TASK]
TASK SPECIFICATION:
{task}

WORKFLOW OUTPUT TO EVALUATE:
{output}

Assess three criteria:
  (1) completeness - does the output address the requested task?
  (2) correctness  - is it correct and does it contain the required information?
  (3) constraint compliance - does it satisfy the task's explicit constraints?

Return STRICT JSON: {{"correct": true|false, "rationale": "<short>"}}
"""


def judge_prompt(task, output) -> str:
    return JUDGE_PROMPT.format(task=task, output=_trunc(output, 2000))


# --------------------------------------------------------------------------- #
# 3.5 Experience Pool — reflection generation                                  #
# --------------------------------------------------------------------------- #
REFLECT_PROMPT = """\
[REFLECT]
A repair attempt just finished with outcome: {outcome}.

ROOT CAUSE: {root_cause}
REPAIR PATCH: {patch_summary}
VERIFICATION NOTE: {note}

Write a ONE-sentence transferable lesson: if outcome is success, give concise
advice on when this repair pattern works; if fail, give a reflection on why it
did not and what to try instead.

Return STRICT JSON: {{"reflection": "<one sentence>"}}
"""


def reflect_prompt(outcome, root_cause, patch_summary, note) -> str:
    return REFLECT_PROMPT.format(outcome=outcome, root_cause=root_cause,
                                 patch_summary=patch_summary, note=note)


# --------------------------------------------------------------------------- #
# RQ4 — unseen test-input generation                                           #
# --------------------------------------------------------------------------- #
UNSEEN_PROMPT = """\
[GEN_UNSEEN]
ORIGINAL TASK / INPUT of a workflow:
{task}

Generate {n} NEW test inputs for the SAME task and domain. Each must:
  (i) belong to the same task/domain as the original;
  (ii) conform to the input format and explicit constraints (be executable);
  (iii) cover diverse scenarios (different scales, boundary conditions, phrasings).

Return STRICT JSON: {{"inputs": ["<input 1>", "<input 2>", ...]}}
"""


def unseen_prompt(task, n) -> str:
    return UNSEEN_PROMPT.format(task=task, n=n)


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #
def _trunc(obj: Any, limit: int = 1200) -> str:
    s = obj if isinstance(obj, str) else _safe_str(obj)
    return s if len(s) <= limit else s[:limit] + " …[truncated]"


def _safe_str(obj: Any) -> str:
    import json
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _bullets(items) -> str:
    return "\n".join(f"    - {it}" for it in items) if items else ""
