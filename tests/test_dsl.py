"""DSL validation & evaluation tests (Sec. 3.2.2 / 3.3.1, Table 1).

Run with:  FLOWFIXER_MOCK=1 python -m pytest tests/ -q
(or plain `python tests/test_dsl.py` — a tiny runner is included at the bottom.)
"""
import os
import sys

os.environ.setdefault("FLOWFIXER_MOCK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flowfixer.dsl import (DSLValidationError, evaluate_assertion,
                           validate_assertion)
from flowfixer.schema import Assertion, SpecDimension, SymbolicTrace, TraceNode


def _trace():
    a = TraceNode(id="a", type="llm", output={"itinerary": "x", "hotel": None},
                  status="success")
    b = TraceNode(id="b", type="end", output=None, status="success")
    return SymbolicTrace(task="t", nodes=[a, b]), a


def test_validate_accepts_grammar():
    validate_assertion("assert node.status == 'success'")
    validate_assertion("assert len(node.output) > 0")
    validate_assertion("assert trace.index(node) >= 0")


def test_validate_rejects_import():
    for bad in ("import os", "assert __import__('os')", "assert (lambda: 1)()"):
        try:
            validate_assertion(bad)
        except DSLValidationError:
            continue
        raise AssertionError(f"should have rejected: {bad!r}")


def test_evaluate_pass_and_violation():
    trace, target = _trace()
    ok = evaluate_assertion(
        Assertion("a", SpecDimension.EXISTENCE, "assert node.status == 'success'"),
        target, trace)
    assert ok.passed and not ok.error

    # 'hotel' is None -> this required-field assertion should be violated
    bad = evaluate_assertion(
        Assertion("a", SpecDimension.CAUSAL,
                  "assert node.output['hotel'] is not None"),
        target, trace)
    assert not bad.passed and not bad.error


def test_evaluate_item_access_case_insensitive():
    trace, target = _trace()
    res = evaluate_assertion(
        Assertion("a", SpecDimension.EXISTENCE, "assert node['Output'] is not None"),
        target, trace)
    assert res.passed


def test_invalid_dsl_flagged_as_error():
    trace, target = _trace()
    res = evaluate_assertion(
        Assertion("a", SpecDimension.EXISTENCE, "import os"), target, trace)
    assert res.error and not res.passed


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
