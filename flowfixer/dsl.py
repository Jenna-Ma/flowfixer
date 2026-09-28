"""Assertion DSL: validation and evaluation (Sec. 3.2.2, Table 1, Fig. 2).

Inferred behavioral specifications are expressed as assertions whose surface
syntax is a Python-compatible subset (Table 1 BNF).  Rather than build a bespoke
parser, we:

  1. **Validate** each assertion against an AST whitelist that realises the BNF
     (only the allowed node/operator/expression forms), rejecting anything with
     side effects (imports, defs, dunder access, ...).
  2. **Evaluate** it in a restricted sandbox where the DSL variables ``node`` and
     ``trace`` — plus every workflow node referenced by name/id — are bound, so
     expressions like ``node["Output"]``, ``node.output``,
     ``trace.index(travel_guide_node)`` and ``any("hotel" in s for s in
     node.Output)`` behave exactly as in Fig. 2.

An assertion that raises ``AssertionError`` is a *violation*; one that raises any
other error is flagged ``error=True`` and excluded from the violation count
(it is neither satisfied nor a genuine behavioral violation).
"""
from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

from .schema import Assertion, AssertionResult, SymbolicTrace, TraceNode


# --------------------------------------------------------------------------- #
# Case/format-tolerant node wrapper (attribute AND item access)                #
# --------------------------------------------------------------------------- #
class SymNode:
    """Wraps a ``TraceNode`` so DSL code can use ``node.output`` or
    ``node["Output"]`` interchangeably (case-insensitive keys)."""

    _CANON = {
        "id": "id", "name": "name", "type": "type", "status": "status",
        "input": "input", "output": "output", "config": "config",
        "upstream": "upstream", "downstream": "downstream",
    }

    def __init__(self, tn: TraceNode):
        object.__setattr__(self, "_tn", tn)

    def _resolve(self, key: str) -> Any:
        tn = object.__getattribute__(self, "_tn")
        k = self._CANON.get(str(key).lower())
        if k is not None:
            return getattr(tn, k)
        # Fall back to a key inside the node's structured output/config dicts.
        for container in (tn.output, tn.config, tn.input):
            if isinstance(container, dict):
                for ck, cv in container.items():
                    if str(ck).lower() == str(key).lower():
                        return cv
        raise KeyError(key)

    def __getitem__(self, key):
        return self._resolve(key)

    def __getattr__(self, key):
        # only called when normal attribute lookup fails
        try:
            return self._resolve(key)
        except KeyError as e:
            raise AttributeError(key) from e

    def __contains__(self, item):
        tn = object.__getattribute__(self, "_tn")
        if isinstance(tn.output, dict):
            return item in tn.output
        if tn.output is not None:
            try:
                return item in tn.output
            except TypeError:
                return False
        return False

    def __iter__(self):
        tn = object.__getattribute__(self, "_tn")
        out = tn.output
        if isinstance(out, dict):
            return iter(out.values())
        if isinstance(out, (list, tuple, set)):
            return iter(out)
        if out is None:
            return iter([])
        return iter([out])

    def __repr__(self):
        tn = object.__getattribute__(self, "_tn")
        return f"SymNode({tn.id})"


class TraceProxy:
    """Exposes ``trace.index(node)`` / ``len(trace)`` over the symbolic trace."""

    def __init__(self, trace: SymbolicTrace, sym_by_id: Dict[str, SymNode]):
        self._trace = trace
        self._sym_by_id = sym_by_id
        self._order = [n.id for n in trace.nodes]

    def index(self, node) -> int:
        nid = _node_id_of(node)
        if nid in self._order:
            return self._order.index(nid)
        raise ValueError(f"{node} not in trace")

    def __len__(self):
        return len(self._order)

    def __iter__(self):
        return iter(self._sym_by_id[i] for i in self._order)


def _node_id_of(node) -> Optional[str]:
    if isinstance(node, SymNode):
        return object.__getattribute__(node, "_tn").id
    if isinstance(node, str):
        return node
    return None


# --------------------------------------------------------------------------- #
# AST whitelist (realises the Table 1 grammar)                                 #
# --------------------------------------------------------------------------- #
_ALLOWED_NODES = {
    ast.Module, ast.Expr, ast.Assert, ast.If, ast.For,
    ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv,
    ast.Call, ast.Attribute, ast.Subscript, ast.Index if hasattr(ast, "Index") else ast.Slice,
    ast.Slice, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp, ast.comprehension,
    ast.Starred, ast.keyword, ast.arguments, ast.arg,
    ast.Pass,
}
# Older Pythons: ast.Str/ast.Num/ast.NameConstant
for _legacy in ("Str", "Num", "NameConstant", "Bytes", "Ellipsis"):
    if hasattr(ast, _legacy):
        _ALLOWED_NODES.add(getattr(ast, _legacy))

_FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "__import__", "globals",
                    "locals", "vars", "getattr", "setattr", "delattr", "input"}


class DSLValidationError(Exception):
    pass


def validate_assertion(code: str) -> None:
    """Raise ``DSLValidationError`` if ``code`` is outside the allowed grammar."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        raise DSLValidationError(f"syntax error: {e}")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise DSLValidationError(f"disallowed construct: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise DSLValidationError("dunder attribute access is not allowed")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise DSLValidationError(f"forbidden name: {node.id}")


# --------------------------------------------------------------------------- #
# Safe builtins available inside assertions                                    #
# --------------------------------------------------------------------------- #
_SAFE_BUILTINS = {
    "len": len, "any": any, "all": all, "range": range, "enumerate": enumerate,
    "zip": zip, "sum": sum, "min": min, "max": max, "sorted": sorted, "abs": abs,
    "str": str, "int": int, "float": float, "bool": bool, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "isinstance": isinstance,
    "True": True, "False": False, "None": None,
}


def _build_namespace(target: TraceNode, trace: SymbolicTrace) -> Dict[str, Any]:
    sym_by_id: Dict[str, SymNode] = {}
    name_bindings: Dict[str, SymNode] = {}
    for tn in trace.nodes:
        s = SymNode(tn)
        sym_by_id[tn.id] = s
        # bind by several normalized spellings so `Travel_guide_node`,
        # `travel_guide`, `TravelGuide` all resolve.
        for label in {tn.id, tn.name or tn.id}:
            for spelling in _spellings(label):
                name_bindings.setdefault(spelling, s)

    ns: Dict[str, Any] = dict(name_bindings)
    ns["node"] = sym_by_id[target.id]
    ns["trace"] = TraceProxy(trace, sym_by_id)
    ns["task"] = trace.task
    ns["__builtins__"] = _SAFE_BUILTINS
    return ns


def _spellings(label: str) -> List[str]:
    base = str(label)
    norm = base.replace(" ", "_").replace("-", "_")
    variants = {base, norm, norm.lower(), norm + "_node", norm.lower() + "_node"}
    return [v for v in variants if v.isidentifier()]


# --------------------------------------------------------------------------- #
# Evaluation                                                                   #
# --------------------------------------------------------------------------- #
def evaluate_assertion(
    assertion: Assertion, target: TraceNode, trace: SymbolicTrace
) -> AssertionResult:
    """Statically verify one assertion against the observed trace (Sec. 3.3.1)."""
    try:
        validate_assertion(assertion.code)
    except DSLValidationError as e:
        return AssertionResult(assertion, passed=False, message=f"invalid DSL: {e}",
                               error=True)
    ns = _build_namespace(target, trace)
    try:
        exec(compile(assertion.code, "<assertion>", "exec"), ns, ns)  # noqa: S102
        return AssertionResult(assertion, passed=True, message="")
    except AssertionError as e:
        return AssertionResult(assertion, passed=False, message=str(e) or "assertion violated")
    except Exception as e:  # noqa: BLE001 - unresolved names, type errors, ...
        return AssertionResult(assertion, passed=False,
                               message=f"eval error: {type(e).__name__}: {e}",
                               error=True)


def evaluate_specs(
    assertions: List[Assertion], target: TraceNode, trace: SymbolicTrace
) -> List[AssertionResult]:
    return [evaluate_assertion(a, target, trace) for a in assertions]
