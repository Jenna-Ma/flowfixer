"""Root cause taxonomy and repair-strategy mapping (Fig. 4, Sec. 3.3.2 & 3.4.1).

The taxonomy is used as structured prior knowledge for root cause analysis, and
each root cause is mapped to a repair strategy (R1..R7) that constrains patch
generation to causally-relevant modifications.

NOTE ON COUNT: the paper's prose says "sixteen root cause types" while Fig. 4
enumerates fifteen named causes across three groups. We encode the fifteen shown
in the figure and keep the registry extensible; ``UNKNOWN`` is a catch-all used
only when the analyzer cannot map to a listed cause.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class RootCause:
    name: str
    group: str          # Node Capability | Node Orchestration | Node Execution
    strategy: str       # R1..R7
    description: str


# Repair strategies (Fig. 4, right column).
REPAIR_STRATEGIES: Dict[str, str] = {
    "R1": "Prompt constraint enforcement and specification",
    "R2": "Format validation and interface alignment",
    "R3": "Model capability upgrade or augmentation",
    "R4": "Encoding and content normalization",
    "R5": "Workflow structural refactoring",
    "R6": "Explicit context declaration and isolation",
    "R7": "Runtime robustness enhancement (e.g., retry)",
}


# The taxonomy exactly as grouped in Fig. 4.
TAXONOMY: List[RootCause] = [
    # ---- Node Capability -------------------------------------------------- #
    RootCause("Tool or Action Planning Error", "Node Capability", "R1",
              "The node plans an incorrect tool/action or ordering of actions."),
    RootCause("Response Format Error", "Node Capability", "R2",
              "The output does not conform to the required/agreed format."),
    RootCause("Response Content Deviation", "Node Capability", "R1",
              "Output content drifts from the requested intent or constraints."),
    RootCause("Knowledge Limitation", "Node Capability", "R3",
              "The model lacks the knowledge/capability the task requires."),
    RootCause("Poor Prompt Design", "Node Capability", "R1",
              "Prompt fails to specify constraints, leading to wrong behavior."),
    RootCause("Language or Encoding Issue", "Node Capability", "R4",
              "Language mismatch or character-encoding corruption in I/O."),
    RootCause("Tool Invocation or KB Retrieval Error", "Node Capability", "R2",
              "Tool call / knowledge-base retrieval is malformed or fails."),
    # ---- Node Orchestration ---------------------------------------------- #
    RootCause("Missing Input Verification", "Node Orchestration", "R2",
              "No validation of upstream/user input before it is consumed."),
    RootCause("Unreasonable Node Dependency", "Node Orchestration", "R5",
              "Data/control dependencies between nodes are wrong or missing."),
    RootCause("Loops and Deadlocks", "Node Orchestration", "R5",
              "Cyclic or stalled control flow prevents completion."),
    RootCause("Faulty Conditional Judgement", "Node Orchestration", "R5",
              "A branch/condition routes execution incorrectly."),
    RootCause("Improper Task Decomposition", "Node Orchestration", "R5",
              "The task is split into sub-tasks that omit requirements."),
    RootCause("Context Conflict", "Node Orchestration", "R6",
              "Conflicting or leaked context across nodes corrupts behavior."),
    # ---- Node Execution -------------------------------------------------- #
    RootCause("Network and Resource Fluctuation", "Node Execution", "R7",
              "Transient network/resource issues disrupt execution."),
    RootCause("Service Unavailability", "Node Execution", "R7",
              "An external service the node depends on is unavailable."),
]

# Catch-all (keeps the pipeline robust to un-mappable diagnoses).
UNKNOWN = RootCause("Unknown", "Node Capability", "R1",
                    "Root cause could not be mapped to a taxonomy entry.")

_BY_NAME: Dict[str, RootCause] = {rc.name.lower(): rc for rc in TAXONOMY}


def all_root_cause_names() -> List[str]:
    return [rc.name for rc in TAXONOMY]


def get_root_cause(name: str) -> RootCause:
    """Fuzzy-lookup a root cause by (possibly noisy) name."""
    if not name:
        return UNKNOWN
    key = name.strip().lower()
    if key in _BY_NAME:
        return _BY_NAME[key]
    # substring / token-overlap fallback
    best, best_score = UNKNOWN, 0.0
    key_tokens = set(key.replace("-", " ").split())
    for rc in TAXONOMY:
        rc_tokens = set(rc.name.lower().split())
        overlap = len(key_tokens & rc_tokens) / max(1, len(rc_tokens))
        if rc.name.lower() in key or key in rc.name.lower():
            overlap = max(overlap, 0.9)
        if overlap > best_score:
            best, best_score = rc, overlap
    return best if best_score >= 0.5 else UNKNOWN


def strategy_for(root_cause_name: str) -> str:
    return get_root_cause(root_cause_name).strategy


def taxonomy_prompt_block() -> str:
    """Render the taxonomy as prior knowledge for the root-cause analysis prompt."""
    lines = []
    cur_group = None
    for rc in TAXONOMY:
        if rc.group != cur_group:
            cur_group = rc.group
            lines.append(f"\n[{cur_group}]")
        lines.append(f"  - {rc.name} (strategy {rc.strategy}): {rc.description}")
    return "\n".join(lines)


def strategy_prompt_block() -> str:
    return "\n".join(f"  {k}: {v}" for k, v in REPAIR_STRATEGIES.items())
