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


def _pool(female_silent_xist=True, n=220, seed=1):
    import scipy.sparse as sps
    rng = np.random.default_rng(seed)
    genes = ["XIST", "RPS4Y1", "KDM6A", "DDX3X", "ZFX", "KDM5C", "EIF1AX", "RPS4X"] + [f"G{i}" for i in range(20)]
    rows, grp = [], []
    for k in range(8):
        sex = "m" if k < 4 else "f"
        silent = female_silent_xist and k == 7
        for _ in range(n):
            r = np.zeros(len(genes))
            r[0] = 0 if sex == "m" or silent else rng.poisson(4)
            r[1] = rng.poisson(6) if sex == "m" else (rng.poisson(0.05))
            r[2:8] = rng.poisson(1.2 if sex == "f" else 0.6, 6)
            r[8:] = rng.poisson(20, 20)
            rows.append(r)
            grp.append(f"d{k}")
    return sps.csr_matrix(np.array(rows)), genes, grp


def test_pool_check_rescues_xist_silent_female():
    X, genes, grp = _pool()
    r = SX.call_sex(X, genes, species="human", groups=grp)
    assert r.groups["d7"]["call"] == "female" and r.groups["d7"].get("pool_check")
    assert r.groups["d0"]["call"] == "male" and r.groups["d5"]["call"] == "female"


def test_pool_check_inactive_without_enough_groups():
    X, genes, grp = _pool()
    keep = np.array([g in ("d0", "d5", "d7") for g in grp])
    r = SX.call_sex(X[keep], genes, species="human", groups=[g for g in grp if g in ("d0", "d5", "d7")])
    assert not any(v.get("pool_check") for v in r.groups.values())
