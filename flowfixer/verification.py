"""Stage 2 (part 3): Dynamic Verification & Multi-LLM Judge (Sec. 3.4.3).

A candidate that passes pre-execution assessment is executed against the test
input, then judged at the *task level*:

  * Execution: if the dataset provides an executable runtime, use it; otherwise
    fall back to an LLM-simulated executor.
  * Multi-LLM Judge: four independent judges (config.JUDGE_MODELS) return binary
    correctness; majority vote decides provisional correctness.
  * Human Verification: an optional hook confirms provisionally-correct cases.
    By default it is auto-confirmed (with a flag) so the pipeline is fully
    automatic; supply a callback to plug in real annotation.

The judges assess correctness independently of FLOWFIXER's own assertions
(Sec. 3.4.3), avoiding circularity.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional

import config as C
from . import llm, prompts
from .schema import JudgeVote, VerificationResult, Workflow


# Optional pluggable real executor: fn(workflow, task_input) -> (executed, output)
Executor = Callable[[Workflow, str], "tuple[bool, Any]"]
# Optional human verification hook: fn(task, output) -> bool
HumanVerifier = Callable[[str, Any], bool]


# --------------------------------------------------------------------------- #
# Execution                                                                     #
# --------------------------------------------------------------------------- #
def simulate_execution(workflow: Workflow, task_input: str) -> "tuple[bool, Any]":
    """LLM-simulated execution when no real runtime is available."""
    lines = [f"  {n.id} [{n.type}] config={_short(n.config)}" for n in workflow.nodes]
    prompt = prompts.exec_prompt(
        task_input=task_input, workflow="\n".join(lines),
        edges=[list(e) for e in workflow.edges],
    )
    data = llm.chat_json(prompt, system=prompts.EXEC_SYSTEM,
                         model=C.EXECUTOR_MODEL,
                         default={"executed": False, "final_output": None,
                                  "error": "unparsed"})
    return bool(data.get("executed", False)), data.get("final_output")


# --------------------------------------------------------------------------- #
# Multi-LLM Judge                                                              #
# --------------------------------------------------------------------------- #
def judge(task: str, output: Any) -> List[JudgeVote]:
    votes: List[JudgeVote] = []
    prompt = prompts.judge_prompt(task, output)
    for model in C.JUDGE_MODELS:
        model = model.strip()
        if not model:
            continue
        data = llm.chat_json(prompt, system=prompts.JUDGE_SYSTEM, model=model,
                             default={"correct": False, "rationale": "unparsed"})
        votes.append(JudgeVote(model=model, correct=bool(data.get("correct", False)),
                               rationale=str(data.get("rationale", ""))))
    return votes


def majority_correct(votes: List[JudgeVote]) -> bool:
    if not votes:
        return False
    yes = sum(1 for v in votes if v.correct)
    return yes > len(votes) / 2      # strictly more than half (Sec. 3.4.3)


# --------------------------------------------------------------------------- #
# Full dynamic verification                                                    #
# --------------------------------------------------------------------------- #
def verify(
    workflow: Workflow,
    task_input: str,
    task_spec: Optional[str] = None,
    executor: Optional[Executor] = None,
    human_verifier: Optional[HumanVerifier] = None,
) -> VerificationResult:
    task_spec = task_spec or task_input
    # 1) execute
    exec_fn = executor or simulate_execution
    try:
        executed, output = exec_fn(workflow, task_input)
    except Exception as e:  # noqa: BLE001
        return VerificationResult(passed=False, executed=False, output=None,
                                  note=f"execution raised: {e}")
    if not executed:
        return VerificationResult(passed=False, executed=False, output=output,
                                  note="workflow failed to execute")
    # 2) multi-LLM judge (majority vote)
    votes = judge(task_spec, output)
    provisional = majority_correct(votes)
    if not provisional:
        return VerificationResult(passed=False, executed=True, votes=votes,
                                  output=output, note="failed majority judge vote")
    # 3) human verification of provisionally-correct cases
    if human_verifier is not None:
        confirmed = bool(human_verifier(task_spec, output))
    else:
        confirmed = True                      # auto-confirm (flagged)
    return VerificationResult(passed=confirmed, executed=True, votes=votes,
                              human_confirmed=confirmed, output=output,
                              note="verified" if confirmed else "human rejected")


def _short(obj, limit: int = 400) -> str:
    import json
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s if len(s) <= limit else s[:limit] + "…"
