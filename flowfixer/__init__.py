"""FLOWFIXER: diagnosis-driven automatic repair for agentic workflows.

Reproduction of the framework described in "Diagnosis-Driven Automatic Repair
for Agentic Workflow via Symbolic Inference".

Public entry points:
  * pipeline.repair_case(case, pool, ablation, ...)   — end-to-end repair
  * pipeline.AblationConfig / pipeline.variant(name)  — RQ3 variants
  * experience.ExperiencePool                         — cross-case memory
"""
from . import (assessment, diagnosis, dsl, experience, llm, pipeline, prompts,
               repair, schema, symbolic, taxonomy, verification)

__all__ = [
    "assessment", "diagnosis", "dsl", "experience", "llm", "pipeline",
    "prompts", "repair", "schema", "symbolic", "taxonomy", "verification",
]

__version__ = "0.1.0"
