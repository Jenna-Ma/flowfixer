"""FLOWFIXER end-to-end pipeline: the diagnosis <-> repair loop (Fig. 1).

``repair_case`` runs Stage 1 (diagnosis) and Stage 2 (repair) with the
pre-execution assessment, dynamic verification, and experience pool, iterating
until a valid repair is found or the retry budget is exhausted (Sec. 3.4.3).

``AblationConfig`` toggles each designed component for RQ3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import config as C
from . import (assessment, diagnosis as diag, llm, repair, symbolic, taxonomy,
               verification as verif)
from .experience import (Attempt, ExperiencePool, OnlineFeedback)
from .schema import Case, Diagnosis, RepairRecord, Workflow


# --------------------------------------------------------------------------- #
# Ablation configuration (RQ3)                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class AblationConfig:
    use_symbol: bool = True            # symbolic modeling & inference
    use_taxonomy: bool = True          # root-cause taxonomy prior
    use_repair_knowledge: bool = True  # repair-strategy knowledge
    use_online_feedback: bool = True   # experience pool: online feedback
    use_experience: bool = True        # experience pool: accumulated experience
    # pre-execution assessment dimensions
    assess_structural: bool = True
    assess_semantic: bool = True
    assess_consistency: bool = True
    assess_offset: bool = True

    def assess_flags(self) -> Dict[str, bool]:
        return {
            "structural": self.assess_structural,
            "semantic": self.assess_semantic,
            "consistency": self.assess_consistency,
            "offset": self.assess_offset,
        }


# named variants used by RQ3 (Table 3)
def variant(name: str) -> AblationConfig:
    name = name.lower()
    cfg = AblationConfig()
    if name in ("full", "flowfixer"):
        return cfg
    if name == "w/o symbol":
        cfg.use_symbol = False
    elif name == "w/o taxonomy":
        cfg.use_taxonomy = False
    elif name == "w/o repair":
        cfg.use_repair_knowledge = False
    elif name == "w/o knowledge":            # both taxonomy + repair
        cfg.use_taxonomy = False
        cfg.use_repair_knowledge = False
    elif name == "w/o online":
        cfg.use_online_feedback = False
    elif name == "w/o experience":
        cfg.use_experience = False
    elif name == "w/o pool":                 # both online + accumulated
        cfg.use_online_feedback = False
        cfg.use_experience = False
    else:
        raise ValueError(f"unknown variant: {name}")
    return cfg


# --------------------------------------------------------------------------- #
# Stage-1 only (for RQ2 diagnosis evaluation)                                    #
# --------------------------------------------------------------------------- #
def diagnose_case(case: Case, ablation: Optional[AblationConfig] = None) -> Diagnosis:
    """Run only Stage 1 (failure diagnosis) on a case."""
    ablation = ablation or AblationConfig()
    trace = symbolic.build_symbolic_trace(
        case.workflow, case.trajectory, case.task, case.final_output)
    if ablation.use_symbol:
        specs = symbolic.infer_specs(trace)
        return diag.diagnose(trace, specs, use_taxonomy=ablation.use_taxonomy)
    return diag.diagnose_raw(trace, use_taxonomy=ablation.use_taxonomy)


# --------------------------------------------------------------------------- #
# Core loop                                                                     #
# --------------------------------------------------------------------------- #
def repair_case(
    case: Case,
    pool: Optional[ExperiencePool] = None,
    ablation: Optional[AblationConfig] = None,
    executor=None,
    human_verifier=None,
    retry_budget: Optional[int] = None,
) -> RepairRecord:
    ablation = ablation or AblationConfig()
    pool = pool if pool is not None else ExperiencePool()
    retry_budget = retry_budget if retry_budget is not None else C.REPAIR_RETRY_BUDGET
    llm.reset_tokens()

    # ---- Symbolic modeling (Sec. 3.2.1) ---------------------------------- #
    trace = symbolic.build_symbolic_trace(
        case.workflow, case.trajectory, case.task, case.final_output)

    record = RepairRecord(case_id=case.case_id, success=False)
    online = OnlineFeedback()

    # ---- Stage 1: diagnosis (re-run each outer iteration) ---------------- #
    def run_diagnosis() -> Diagnosis:
        if ablation.use_symbol:
            specs = symbolic.infer_specs(trace)
            return diag.diagnose(trace, specs, use_taxonomy=ablation.use_taxonomy)
        return diag.diagnose_raw(trace, use_taxonomy=ablation.use_taxonomy)

    diagnosis = run_diagnosis()
    record.diagnosis = diagnosis

    # retrieved accumulated experience (Sec. 3.5) — stable across inner retries
    exp_block = ""
    if ablation.use_experience:
        retrieved = pool.retrieve(diagnosis, trace)
        exp_block = pool.render_block(retrieved)

    iteration = 0
    while iteration < retry_budget:
        iteration += 1
        record.iterations = iteration

        # ---- experience context for patch generation --------------------- #
        blocks = [b for b in (exp_block,
                              online.render() if ablation.use_online_feedback else "")
                  if b]
        patch_experience = "\n\n".join(blocks)

        # ---- Stage 2: patch generation & application --------------------- #
        patch = repair.generate_patch(
            case.workflow, trace, diagnosis,
            experience_block=patch_experience,
            use_repair_knowledge=ablation.use_repair_knowledge,
        )
        modified = repair.apply_patch(case.workflow, patch)

        # ---- pre-execution assessment (Sec. 3.4.2) ---------------------- #
        assessment_result = assessment.assess(
            case.workflow, modified, patch, diagnosis, case.task,
            enabled=ablation.assess_flags())

        step = {"iteration": iteration, "root_cause": diagnosis.root_cause,
                "patch": patch.summary(),
                "assessment_passed": assessment_result.passed,
                "assessment_reasons": assessment_result.failure_reasons()}

        if not assessment_result.passed:
            if ablation.use_online_feedback:
                online.record(Attempt(
                    root_cause=diagnosis.root_cause, patch_summary=patch.summary(),
                    assessment_reasons=assessment_result.failure_reasons(),
                    executed=False, passed=False, note="assessment rejected"))
            step["result"] = "rejected_by_assessment"
            record.history.append(step)
            # re-generate patch next iteration; periodically re-diagnose
            if iteration % 2 == 0:
                diagnosis = run_diagnosis()
                record.diagnosis = diagnosis
            continue

        # ---- dynamic verification (Sec. 3.4.3) --------------------------- #
        vresult = verif.verify(
            modified, case.test_input, task_spec=case.task,
            executor=executor, human_verifier=human_verifier)
        step["executed"] = vresult.executed
        step["verified"] = vresult.passed
        step["note"] = vresult.note

        if ablation.use_online_feedback:
            online.record(Attempt(
                root_cause=diagnosis.root_cause, patch_summary=patch.summary(),
                assessment_reasons=[], executed=vresult.executed,
                passed=vresult.passed, note=vresult.note))

        if vresult.passed:
            record.success = True
            record.final_workflow = modified
            record.applied_patch = patch
            step["result"] = "success"
            record.history.append(step)
            break

        step["result"] = "verification_failed"
        record.history.append(step)
        # feed back into diagnosis for the next outer iteration
        diagnosis = run_diagnosis()
        record.diagnosis = diagnosis

    # ---- accumulate experience (success or fail) ------------------------ #
    outcome = "success" if record.success else "fail"
    last_patch = record.applied_patch or repair.Patch(edits=[])
    note = record.history[-1].get("note", "") if record.history else ""
    if ablation.use_experience:
        pool.form_and_add(record.diagnosis, trace, last_patch, outcome, note)

    record.tokens = llm.tokens_used()
    return record
