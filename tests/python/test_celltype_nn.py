# SPDX-License-Identifier: MIT
"""singlet.celltype_nn: ontology credit and roll-up, ensemble consensus, vocabulary projection, the sidecar guard, the loss."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

from singlet.celltype_nn.ensemble import SingleRRef, celltypist_crosswalk, refinement
from singlet.celltype_nn.features import GeneVocab
from singlet.celltype_nn.ontology import TermSpace, build_space, is_consistent
from singlet.celltype_nn.output import COLUMNS, PublicOutputError, assert_public_safe

T_CELL = "CL:0000084"
CD4 = "CL:0000624"  # CD4-positive, alpha-beta T cell
CD8 = "CL:0000625"  # CD8-positive, alpha-beta T cell
B_CELL = "CL:0000236"
LYMPH = "CL:0000542"  # lymphocyte
EPI = "CL:0000066"  # epithelial cell
AB_T = "CL:0000791"  # mature alpha-beta T cell (deepest common ancestor of CD4 and CD8 alpha-beta T cells)


@pytest.fixture(scope="module")
def space():
    return TermSpace([T_CELL, CD4, CD8, B_CELL, LYMPH, EPI])


def test_credit_finer_is_free_coarser_partial_other_branch_zero(space):
    c = space.credit(T_CELL)
    i = space.index
    assert c[i[T_CELL]] == 1 and c[i[CD4]] == 1 and c[i[CD8]] == 1
    assert c[i[B_CELL]] == 0
    c8 = space.credit(CD8)
    assert c8[i[CD8]] == 1 and c8[i[CD4]] == 0
    assert 0 < c8[i[LYMPH]] < c8[i[T_CELL]] < 1


def test_rollup_and_call(space):
    P = np.zeros((2, len(space)), np.float32)
    i = space.index
    P[0, i[CD4]], P[0, i[CD8]], P[0, i[B_CELL]] = 0.3, 0.3, 0.4  # spread over T subtypes: T cell is the call
    P[1, i[CD8]], P[1, i[T_CELL]] = 0.9, 0.1
    up = space.rollup(P)
    assert np.isclose(up[0, space.cindex[T_CELL]], 0.6) and np.isclose(up[0, space.cindex[AB_T]], 0.6)
    assert np.isclose(up[0, space.cindex[LYMPH]], 1.0)
    j, pj = space.call(P)
    calls = [space.closure[k] for k in j]
    assert calls == [AB_T, CD8]  # CD4 and CD8 are both alpha-beta T cells: the call is their closest common ancestor
    assert np.isclose(pj[0], 0.6) and np.isclose(pj[1], 0.9)


def test_build_space_folds_rare_terms():
    terms, fold = build_space({CD4: 100, CD8: 3, B_CELL: 80}, min_cells=50)
    assert CD8 not in terms and fold[CD8] in terms and CD8 != fold[CD8]
    assert is_consistent(TermSpace(terms).o, fold[CD8], CD8) == "coarse"


def test_refinement_needs_two_votes_below_author():
    assert refinement(T_CELL, [CD8, CD8, None]) == CD8
    assert refinement(T_CELL, [CD8, None, None]) is None
    assert refinement(T_CELL, [CD8, CD4, None]) is None or refinement(T_CELL, [CD8, CD4, None]) != T_CELL
    assert refinement(T_CELL, [B_CELL, B_CELL, B_CELL]) is None


def test_crosswalk_by_name_and_by_cover():
    labs = ["T cell"] * 5 + ["Tem/Effector helper T cells"] * 40
    auth = [T_CELL] * 5 + [CD4] * 36 + [CD8] * 4
    xw = celltypist_crosswalk(labs, auth, min_cells=30)
    assert xw["T cell"] == T_CELL
    assert xw["Tem/Effector helper T cells"] == CD4


def test_vocab_projection_maps_mouse_orthologs_and_ignores_versions():
    v = GeneVocab(["ENSG1", "ENSG2"], ["A", "B"], {"ENSMUSG9": "ENSG2"})
    X = sps.csr_matrix(np.array([[1, 2, 3]], np.float32))
    Ph = v.project(X, ["ENSG2.5", "ENSG1", "ENSG7"], "human").toarray()
    assert Ph.tolist() == [[2, 1]]
    Pm = v.project(X, ["ENSMUSG9", "ENSMUSG1", "ENSG1"], "mouse").toarray()
    assert Pm.tolist() == [[0, 1]]


def test_singler_reference_recovers_profiles():
    rng = np.random.default_rng(0)
    G, labels = 300, ["a", "b", "c"]
    base = rng.gamma(1, 1, (3, G))
    rows, lab, grp = [], [], []
    for d in range(3):
        for k, l in enumerate(labels):
            for _ in range(30):
                rows.append(rng.poisson(base[k] * 3))
                lab.append(l)
                grp.append(f"d{d}")
    X = sps.csr_matrix(np.log1p(np.array(rows, np.float32)))
    ref = SingleRRef.build(X, lab, grp, [f"g{i}" for i in range(G)])
    c = ref.score(X, exclude_group=grp)
    assert (c.cl == np.array(lab, object)).mean() > 0.95


def _table(n=3):
    d = {
        "gsm": ["GSM1"] * n,
        "col": np.arange(n, dtype=np.int64),
        "cell_type_cl": [T_CELL, None, CD8][:n],
        "cell_type_label": ["T cell", None, "CD8-positive, alpha-beta T cell"][:n],
        "cell_type_prob": np.array([0.7, np.nan, 0.9][:n], np.float32),
        "argmax_cl": [CD4, B_CELL, CD8][:n],
        "argmax_prob": np.array([0.4, 0.3, 0.8][:n], np.float32),
    }
    for i in range(1, 6):
        d[f"top{i}_cl"] = [CD4, B_CELL, CD8][:n]
    p = np.array([0.4, 0.3, 0.8][:n], np.float32)
    d["top1_prob"] = p
    for i in range(2, 6):
        d[f"top{i}_prob"] = (p * 0.05).astype(np.float32)
    d["model_version"] = ["celltype-nn=0.1.0;cl=2026-06-08"] * n
    return pd.DataFrame(d)[[c for c, _ in COLUMNS]]


def test_sidecar_guard_accepts_valid_and_rejects_extra_or_bad_values():
    assert_public_safe(_table())
    t = _table()
    t["barcode"] = "ACGT"
    with pytest.raises(PublicOutputError):
        assert_public_safe(t)
    t = _table()
    t.loc[0, "argmax_cl"] = "rs12345"
    with pytest.raises(PublicOutputError):
        assert_public_safe(t)
    t = _table()
    t.loc[1, "top2_prob"] = 0.9
    with pytest.raises(PublicOutputError):
        assert_public_safe(t)
    t = _table()
    t.loc[2, "col"] = 0
    with pytest.raises(PublicOutputError):
        assert_public_safe(t)


def test_credit_loss_prefers_deepest_supported_term(space):
    torch = pytest.importorskip("torch")
    from singlet.celltype_nn.model import credit_loss

    cr = torch.from_numpy(space.credit(CD8))[None]
    i = space.index
    deep = torch.full((1, len(space)), -10.0)
    deep[0, i[CD8]] = 10
    coarse = torch.full((1, len(space)), -10.0)
    coarse[0, i[T_CELL]] = 10
    wrong = torch.full((1, len(space)), -10.0)
    wrong[0, i[CD4]] = 10
    assert credit_loss(deep, cr) < credit_loss(coarse, cr) < credit_loss(wrong, cr)
    # a coarser author label: the finer prediction costs nothing
    crT = torch.from_numpy(space.credit(T_CELL))[None]
    assert float(credit_loss(deep, crT)) < 1e-3
