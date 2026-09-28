"""Core data structures for FLOWFIXER.

These mirror the formalism in the paper:

  * A workflow ``W = (N, E)``               (Eq. 1, Sec. 2)
  * A node ``n_i = (I_i, T_i, C_i)``         (Eq. 2, Sec. 2)
  * A symbolic trace                         (Sec. 3.2.1)
  * Behavioral specifications / assertions   (Sec. 3.2.2, Table 1)
  * A repair patch = {edit_1..edit_k}        (Eq. 3-4, Sec. 3.4.1)
  * An accumulated experience e=<phi,r,P,o,d> (Sec. 3.5)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Node type categories (Sec. 2, six families)                                  #
# --------------------------------------------------------------------------- #
class NodeCategory(str, Enum):
    START_END = "Start/Termination"
    LLM_AGENT = "LLM/Agent"
    KNOWLEDGE = "Knowledge"
    LOGIC_CONTROL = "Logic/Control"
    CODE_TEMPLATE = "Code/Template"
    TOOL_INTEGRATION = "Tool/Integration"
    UNKNOWN = "Unknown"


# Heuristic mapping from raw platform node-type strings to the six families.
_TYPE_KEYWORDS: List[Tuple[NodeCategory, Tuple[str, ...]]] = [
    (NodeCategory.START_END, ("start", "begin", "end", "termination", "answer", "reply")),
    (NodeCategory.LLM_AGENT, ("llm", "agent", "chat", "assistant", "model")),
    (NodeCategory.KNOWLEDGE, ("knowledge", "retrieval", "rag", "vector", "kb", "search-kb")),
    (NodeCategory.LOGIC_CONTROL, ("if", "else", "condition", "branch", "loop",
                                  "iterate", "switch", "router", "classif")),
    (NodeCategory.CODE_TEMPLATE, ("code", "template", "script", "python",
                                  "javascript", "jinja", "transform")),
    (NodeCategory.TOOL_INTEGRATION, ("tool", "http", "api", "webhook",
                                     "integration", "plugin", "function")),
]


def classify_node_type(raw_type: str) -> NodeCategory:
    t = (raw_type or "").lower()
    for cat, kws in _TYPE_KEYWORDS:
        if any(kw in t for kw in kws):
            return cat
    return NodeCategory.UNKNOWN


# --------------------------------------------------------------------------- #
# Workflow structure                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class Node:
    """Static workflow node n_i = (I_i, T_i, C_i)."""
    id: str
    type: str                                  # raw platform type string
    config: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None                 # I_i (display name); defaults to id

    def __post_init__(self):
        if self.name is None:
            self.name = self.id

    @property
    def category(self) -> NodeCategory:
        return classify_node_type(self.type)


@dataclass
class Workflow:
    """W = (N, E)."""
    nodes: List[Node] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)   # E subset N x N
    platform: str = "unknown"
    task: str = ""

    # -- graph helpers ------------------------------------------------------ #
    def node_ids(self) -> List[str]:
        return [n.id for n in self.nodes]

    def get_node(self, node_id: str) -> Optional[Node]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None

    def upstream(self, node_id: str) -> List[str]:
        return [a for (a, b) in self.edges if b == node_id]

    def downstream(self, node_id: str) -> List[str]:
        return [b for (a, b) in self.edges if a == node_id]

    def reachable_downstream(self, node_id: str) -> set:
        """All nodes transitively reachable from ``node_id`` (excludes itself)."""
        seen: set = set()
        stack = list(self.downstream(node_id))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.downstream(cur))
        seen.discard(node_id)
        return seen

    def clone(self) -> "Workflow":
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "task": self.task,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [list(e) for e in self.edges],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Workflow":
        nodes = [
            Node(
                id=str(nd["id"]),
                type=str(nd.get("type", "unknown")),
                config=dict(nd.get("config", {})),
                name=nd.get("name"),
            )
            for nd in d.get("nodes", [])
        ]
        edges = [(str(a), str(b)) for a, b in d.get("edges", [])]
        return Workflow(
            nodes=nodes,
            edges=edges,
            platform=d.get("platform", "unknown"),
            task=d.get("task", ""),
        )


# --------------------------------------------------------------------------- #
# Execution trajectory / symbolic trace (Sec. 3.2.1)                           #
# --------------------------------------------------------------------------- #
@dataclass
class TraceNode:
    """One executed node in the unified symbolic trace.

    The paper's DSL accesses these via both attribute (``node.output``) and
    item (``node["Output"]``) style, so the accompanying ``SymNode`` wrapper in
    ``dsl.py`` exposes both.  This dataclass is the plain-data carrier.
    """
    id: str
    type: str
    input: Any = None
    output: Any = None
    status: str = "success"                    # success | fail | error
    config: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    upstream: List[str] = field(default_factory=list)
    downstream: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.name is None:
            self.name = self.id

    @property
    def category(self) -> NodeCategory:
        return classify_node_type(self.type)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SymbolicTrace:
    """Ordered sequence of executed nodes plus the global task context."""
    task: str
    nodes: List[TraceNode] = field(default_factory=list)
    platform: str = "unknown"
    final_output: Any = None

    def index_of(self, node_id: str) -> int:
        for i, n in enumerate(self.nodes):
            if n.id == node_id:
                return i
        return -1

    def get(self, node_id: str) -> Optional[TraceNode]:
        for n in self.nodes:
            if n.id == node_id:
                return n
        return None


# --------------------------------------------------------------------------- #
# Behavioral specifications (Sec. 3.2.2, Table 1)                              #
# --------------------------------------------------------------------------- #
class SpecDimension(str, Enum):
    EXISTENCE = "existence"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


@dataclass
class Assertion:
    """A single inferred behavioral assertion targeting one node."""
    node_id: str
    dimension: SpecDimension
    code: str                                  # DSL / python-syntax assertion block
    rationale: str = ""                        # natural-language intent


@dataclass
class AssertionResult:
    assertion: Assertion
    passed: bool
    message: str = ""                          # violation message or error
    error: bool = False                        # True if evaluation itself failed


@dataclass
class NodeSpec:
    """All inferred assertions for a single node."""
    node_id: str
    assertions: List[Assertion] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Diagnosis output (Sec. 3.3)                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class NodeVerification:
    node_id: str
    results: List[AssertionResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def violated(self) -> int:
        return sum(1 for r in self.results if not r.passed and not r.error)

    @property
    def violation_rate(self) -> float:
        return self.violated / self.total if self.total else 0.0

    def violated_assertions(self) -> List[AssertionResult]:
        return [r for r in self.results if not r.passed and not r.error]

    def satisfied_assertions(self) -> List[AssertionResult]:
        return [r for r in self.results if r.passed]


@dataclass
class Diagnosis:
    responsible_node: str
    root_cause: str                            # one of the taxonomy labels
    repair_strategy: str                       # R1..R7
    failure_description: str = ""
    suspicious_scores: Dict[str, float] = field(default_factory=dict)
    verifications: Dict[str, NodeVerification] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Repair patch (Sec. 3.4.1, Eq. 3-4)                                           #
# --------------------------------------------------------------------------- #
class EditOp(str, Enum):
    INSERT = "insert"
    REMOVE = "remove"
    REPLACE = "replace"
    APPEND = "append"
    SWAP = "swap"


@dataclass
class Edit:
    """edit = op(target, content, position)  (Eq. 4)."""
    op: EditOp
    target: str                                # node id / field being modified
    content: Any = None                        # new/replacement content
    position: Optional[str] = None             # e.g. "after:begin", "between:a:b"

    def summary(self) -> str:
        return f"{self.op.value}(target={self.target}, position={self.position})"


@dataclass
class Patch:
    """Patch = {edit_1, ..., edit_k}  (Eq. 3)."""
    edits: List[Edit] = field(default_factory=list)

    def __len__(self):
        return len(self.edits)

    def summary(self) -> str:
        return "; ".join(e.summary() for e in self.edits)


# --------------------------------------------------------------------------- #
# Assessment & verification results                                            #
# --------------------------------------------------------------------------- #
@dataclass
class AssessmentResult:
    passed: bool
    structural: Tuple[bool, str]
    semantic: Tuple[bool, str]
    consistency: Tuple[bool, str]
    offset: Tuple[bool, str]

    def failure_reasons(self) -> List[str]:
        out = []
        for name, (ok, msg) in [
            ("structural", self.structural),
            ("semantic", self.semantic),
            ("consistency", self.consistency),
            ("offset", self.offset),
        ]:
            if not ok:
                out.append(f"{name}: {msg}")
        return out


@dataclass
class JudgeVote:
    model: str
    correct: bool
    rationale: str = ""


@dataclass
class VerificationResult:
    passed: bool                               # final task-level correctness
    executed: bool                             # ran without runtime error
    votes: List[JudgeVote] = field(default_factory=list)
    human_confirmed: Optional[bool] = None
    output: Any = None
    note: str = ""


# --------------------------------------------------------------------------- #
# Experience pool (Sec. 3.5): e = <phi_ctx, r, P, o, delta>                    #
# --------------------------------------------------------------------------- #
@dataclass
class Experience:
    phi_ctx: str                               # symbolic context of resp. node
    root_cause: str                            # r
    patch: Dict[str, Any]                      # P (serialized edits)
    outcome: str                               # o in {success, fail}
    reflection: str                            # delta (advice or reflection)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# A single benchmark case (one failure log)                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Case:
    """One annotated failure case (AgentFail / n8n record)."""
    case_id: str
    platform: str
    task: str                                  # task description / original input
    workflow: Workflow
    trajectory: List[Dict[str, Any]] = field(default_factory=list)  # raw steps
    final_output: Any = None
    test_input: Optional[str] = None           # input to replay in verification
    # expert annotations (ground truth for RQ2)
    gt_responsible_node: Optional[str] = None
    gt_root_cause: Optional[str] = None

    def __post_init__(self):
        if self.test_input is None:
            self.test_input = self.task


# --------------------------------------------------------------------------- #
# End-to-end repair record (one case)                                          #
# --------------------------------------------------------------------------- #
@dataclass
class RepairRecord:
    case_id: str
    success: bool
    diagnosis: Optional[Diagnosis] = None
    final_workflow: Optional[Workflow] = None
    applied_patch: Optional[Patch] = None
    iterations: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    tokens: int = 0
