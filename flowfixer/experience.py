"""Experience Pool (Sec. 3.5).

Two memories:
  * OnlineFeedback         — short-term, within the current failure case: past
    attempts (assessment results, applied edits, execution outcome) so the loop
    avoids repeating ineffective modifications.
  * AccumulatedExperience  — long-term, cross-case: structured records
    e = <phi_ctx, r, P, o, delta> keyed on failure signature (root cause + node
    symbolic context) and repair pattern, retrieved by similarity.

The pool is COLD-STARTED per evaluation run (no records seeded from the test set,
Sec. 3.5 "Initialization").
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config as C
from . import llm, prompts
from .schema import Diagnosis, Experience, Patch, VerificationResult


# --------------------------------------------------------------------------- #
# Online feedback (per-case short-term memory)                                 #
# --------------------------------------------------------------------------- #
@dataclass
class Attempt:
    root_cause: str
    patch_summary: str
    assessment_reasons: List[str]
    executed: bool
    passed: bool
    note: str = ""


@dataclass
class OnlineFeedback:
    attempts: List[Attempt] = field(default_factory=list)

    def record(self, attempt: Attempt) -> None:
        self.attempts.append(attempt)

    def render(self) -> str:
        if not self.attempts:
            return ""
        lines = ["PRIOR ATTEMPTS ON THIS CASE (avoid repeating failures):"]
        for i, a in enumerate(self.attempts, 1):
            status = "PASS" if a.passed else ("assessment-rejected" if not a.executed
                                              and a.assessment_reasons else "verify-failed")
            lines.append(f"  attempt {i}: rc={a.root_cause}; patch=[{a.patch_summary}]; "
                         f"result={status}; reasons={a.assessment_reasons or a.note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Accumulated experience (cross-case long-term memory)                         #
# --------------------------------------------------------------------------- #
def symbolic_context(diagnosis: Diagnosis, trace) -> str:
    """phi_ctx: compact symbolic context of the failure-responsible node."""
    node = trace.get(diagnosis.responsible_node)
    if node is None:
        return diagnosis.root_cause
    ver = diagnosis.verifications.get(diagnosis.responsible_node)
    violated = [r.assertion.dimension.value for r in ver.violated_assertions()] if ver else []
    return (f"type={node.type}; category={node.category.value}; "
            f"violated_dims={sorted(set(violated))}; status={node.status}")


class ExperiencePool:
    def __init__(self):
        self.records: List[Experience] = []

    # -- persistence ------------------------------------------------------- #
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.records], f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.records = [Experience(**d) for d in json.load(f)]

    # -- add --------------------------------------------------------------- #
    def add(self, exp: Experience) -> None:
        self.records.append(exp)

    def form_and_add(
        self, diagnosis: Diagnosis, trace, patch: Patch, outcome: str,
        note: str = "",
    ) -> Experience:
        reflection = _reflect(outcome, diagnosis.root_cause, patch.summary(), note)
        exp = Experience(
            phi_ctx=symbolic_context(diagnosis, trace),
            root_cause=diagnosis.root_cause,
            patch={"edits": [_edit_dict(e) for e in patch.edits]},
            outcome=outcome,
            reflection=reflection,
        )
        self.add(exp)
        return exp

    # -- retrieval (Sec. 3.5 "Retrieval") ---------------------------------- #
    def retrieve(self, diagnosis: Diagnosis, trace,
                 top_k: int = None, threshold: float = None) -> List[Experience]:
        top_k = top_k or C.EXPERIENCE_TOP_K
        threshold = C.EXPERIENCE_SIM_THRESHOLD if threshold is None else threshold
        if not self.records:
            return []
        query_ctx = symbolic_context(diagnosis, trace)
        query_rc = diagnosis.root_cause
        scored = []
        for e in self.records:
            rc_sim = 1.0 if e.root_cause == query_rc else _text_sim(e.root_cause, query_rc)
            ctx_sim = _text_sim(e.phi_ctx, query_ctx)
            sim = 0.5 * rc_sim + 0.5 * ctx_sim
            if sim >= threshold:
                scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def render_block(self, experiences: List[Experience]) -> str:
        if not experiences:
            return ""
        lines = ["RELEVANT PAST EXPERIENCE (similar failures & how they were handled):"]
        for e in experiences:
            edits = "; ".join(f"{ed['op']}({ed['target']})" for ed in e.patch.get("edits", []))
            lines.append(f"  - context: {e.phi_ctx}; root_cause: {e.root_cause}; "
                         f"patch: [{edits}]; outcome: {e.outcome}; lesson: {e.reflection}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #
def _edit_dict(e) -> Dict[str, Any]:
    return {"op": e.op.value, "target": e.target,
            "content": e.content, "position": e.position}


def _text_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def _reflect(outcome: str, root_cause: str, patch_summary: str, note: str) -> str:
    prompt = prompts.reflect_prompt(outcome, root_cause, patch_summary, note)
    data = llm.chat_json(prompt, system="You distill transferable repair lessons.",
                         default={"reflection": f"{outcome}: {root_cause}"})
    return str(data.get("reflection", ""))
