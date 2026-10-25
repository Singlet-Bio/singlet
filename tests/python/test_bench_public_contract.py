"""The public benchmark results contract (bench/public/contract.py, CONTRACT.md)."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "bench" / "public"
_spec = importlib.util.spec_from_file_location("bench_public_contract", PUB / "contract.py")
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


@pytest.fixture
def example():
    return json.loads((PUB / "schema" / "example-capability.json").read_text(encoding="utf-8"))


def _metric(rule, s, comps=(), direction="higher"):
    return {"direction": direction, "singlet": {"value": s},
            "competitors": [{"tool": t, "value": v} for t, v in comps], "rule": rule}


def test_example_is_valid(example):
    assert contract.validate(example, contract.load_cohort(PUB / "cohort.tsv")) == []
    assert contract.capability_status(example)["status"] == "incomplete"  # draft never passes


def test_final_example_passes(example):
    example["status"] = "final"
    example["artifacts"][0]["sha256"] = "0" * 64
    assert contract.validate(example) == []
    assert contract.capability_status(example) == {"status": "pass", "n_metrics": 3, "n_pass": 3, "n_fail": 0, "n_na": 0}


@pytest.mark.parametrize("rule,s,comps,direction,want", [
    ({"type": "match_or_beat"}, 0.90, [("a", 0.91), ("b", 0.80)], "higher", "fail"),
    ({"type": "match_or_beat", "tolerance": {"abs": 0.01}}, 0.90, [("a", 0.91)], "higher", "pass"),
    ({"type": "match_or_beat", "tolerance": {"rel": 0.05}}, 104.0, [("a", 100.0)], "lower", "pass"),
    ({"type": "match_or_beat", "tolerance": {"rel": 0.05}}, 106.0, [("a", 100.0)], "lower", "fail"),
    ({"type": "match_or_beat", "against": ["b"]}, 0.85, [("a", 0.91), ("b", 0.80)], "higher", "pass"),
    ({"type": "match_or_beat"}, 0.5, [("a", None)], "higher", "na"),
    ({"type": "threshold", "value": 0.95}, 0.96, [], "higher", "pass"),
    ({"type": "threshold", "value": 0.01}, 0.02, [], "lower", "fail"),
    ({"type": "equivalence", "target": "a", "tolerance": {"rel": 0.05}}, 1.04, [("a", 1.0)], "none", "pass"),
    ({"type": "equivalence", "target": 1.0, "tolerance": {"rel": 0.05}}, 0.94, [], "none", "fail"),
    ({"type": "informational"}, 3, [], "none", "na"),
])
def test_rules(rule, s, comps, direction, want):
    assert contract.evaluate_metric(_metric(rule, s, comps, direction)) == want


def test_hand_written_pass_is_rejected(example):
    example["metrics"][0]["competitors"][0]["value"] = 0.999
    errs = contract.validate(example)
    assert any("rule gives 'fail'" in e for e in errs)


def test_loss_cannot_be_hidden_as_informational(example):
    m = example["metrics"][1]
    m["rule"] = {"type": "informational"}
    m["pass"] = "na"
    assert any("informational metric cannot carry competitor" in e for e in contract.validate(example))


def test_tolerance_needs_rationale(example):
    del example["metrics"][0]["rule"]["rationale"]
    assert any("rationale" in e for e in contract.validate(example))


def test_shape_errors(example):
    bad = copy.deepcopy(example)
    bad["conclusion"] = "word " * 101
    bad["capability"] = "teleportation"
    bad["singlet"]["commit"] = "abc"
    bad["code"][0]["url"] = "https://github.com/Singlet-Bio/singlet/blob/main/x.py"
    bad["metrics"][0]["competitors"][0]["tool"] = "undeclared"
    bad["datasets"][0]["id"] = "not-in-cohort"
    errs = " | ".join(contract.validate(bad, contract.load_cohort(PUB / "cohort.tsv")))
    for frag in ("101 words", "not registered", "40-hex", "pinned to a commit", "not declared in tools",
                 "not in the cohort"):
        assert frag in errs


def test_index_and_diff(tmp_path, example):
    example["status"] = "final"
    example["artifacts"][0]["sha256"] = "0" * 64
    contract.finalize(example)
    (tmp_path / "chromosomal-sex.json").write_text(json.dumps(example))
    idx = contract.build_index(tmp_path)
    by = {c["capability"]: c for c in idx["capabilities"]}
    assert by["chromosomal-sex"]["status"] == "pass"
    assert by["gex-counts"]["status"] == "missing"
    assert idx["overall"] == "incomplete" and len(by) == len(contract.CAPABILITIES)
    other = copy.deepcopy(example)
    other["metrics"][0]["singlet"]["value"] = 0.5
    assert contract.diff(example, example) == []
    assert len(contract.diff(example, other)) == 1


def test_published_results_are_valid():
    cohort = contract.load_cohort(PUB / "cohort.tsv")
    for p in sorted((PUB / "results").glob("*.json")):
        if p.name == "index.json":
            continue
        assert contract.validate(json.loads(p.read_text(encoding="utf-8")), cohort) == [], p.name
