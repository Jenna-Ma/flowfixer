"""End-to-end pipeline smoke tests in offline MOCK mode.

Run with:  FLOWFIXER_MOCK=1 python -m pytest tests/ -q
"""
import os
import sys

os.environ.setdefault("FLOWFIXER_MOCK", "1")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from data import loaders
from flowfixer.experience import ExperiencePool
from flowfixer.pipeline import diagnose_case, repair_case, variant
from flowfixer.schema import Diagnosis, RepairRecord

SAMPLE = os.path.join(_ROOT, "data", "sample", "sample_cases.json")


def _cases():
    return loaders.load_cases(SAMPLE)


def test_sample_loads():
    cases = _cases()
    assert len(cases) >= 2
    assert all(c.gt_responsible_node for c in cases)


def test_diagnose_case_returns_diagnosis():
    for case in _cases():
        d = diagnose_case(case)
        assert isinstance(d, Diagnosis)
        assert d.responsible_node
        assert d.root_cause


def test_repair_case_succeeds_in_mock():
    pool = ExperiencePool()
    for case in _cases():
        rec = repair_case(case, pool=pool)
        assert isinstance(rec, RepairRecord)
        assert rec.success                      # mock judge always passes
        assert rec.final_workflow is not None
        assert rec.iterations >= 1


def test_ablation_variants_run():
    for name in ("full", "w/o symbol", "w/o pool", "w/o knowledge"):
        cfg = variant(name)
        rec = repair_case(_cases()[0], pool=ExperiencePool(), ablation=cfg)
        assert isinstance(rec, RepairRecord)


def test_experience_pool_accumulates():
    pool = ExperiencePool()
    cases = _cases()
    repair_case(cases[0], pool=pool)
    assert len(pool.records) >= 1               # experience formed after a case


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
