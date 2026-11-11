"""Scoring helpers of bench/calls-sota/sex (gate: bench/calls-sota/gates/sex.md)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from singlet.celllabels import sex as SX

_p = Path(__file__).resolve().parents[2] / "bench" / "calls-sota" / "sex" / "score.py"
_spec = importlib.util.spec_from_file_location("calls_sota_sex_score", _p)
S = importlib.util.module_from_spec(_spec)
sys.modules["calls_sota_sex_score"] = S
_spec.loader.exec_module(S)


def test_rule2_two_gene():
    x = np.array([3, 0, 2, 0.0])
    y = np.array([0, 4, 5, 0.0])
    assert list(S.rule2(x, y)) == ["female", "male", None, None]


def test_unit_table_donor_call_and_abstention():
    df = pd.DataFrame({"dataset": "d", "unit": ["a"] * 4 + ["b"] * 4, "truth": ["female"] * 4 + ["male"] * 4,
                       "m": ["female", "female", None, "male", None, None, None, None]})
    u = S.unit_table(df, "m").set_index("unit")
    assert u.loc["a", "donor_ok"] and u.loc["a", "acc"] == 0.5 and u.loc["a", "cov"] == 0.75
    assert u.loc["b", "donor_abstain"] and not u.loc["b", "donor_ok"]


def test_singlet_deployed_mode_inherits_donor_sex():
    rng = np.random.default_rng(0)
    n = 300
    donors = ["f"] * n + ["m"] * n
    x = np.concatenate([rng.poisson(3, n), np.zeros(n)]).astype(float)
    y = np.concatenate([np.zeros(n), rng.poisson(2, n)]).astype(float)
    lab = S.singlet_calls(SX, x, y, donors)
    assert (lab[:n] == "female").mean() > 0.99 and (lab[n:] == "male").mean() > 0.99
