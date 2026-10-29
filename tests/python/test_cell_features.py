# SPDX-License-Identifier: MIT
"""singlet.features: rank statistic, null sets, sidecar guard and writer, registry and summary rows, end to end on a synthetic
sample built on the packaged human reference genes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest
import scipy.sparse as sps

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from singlet.features import genesets as GS  # noqa: E402
from singlet.features import output as O  # noqa: E402
from singlet.features import pipeline as P  # noqa: E402
from singlet.features import rank as R  # noqa: E402
from singlet.features import registry as REG  # noqa: E402


# ------------------------------------------------------------------------------------------------------------- rank statistic
def _brute_signal(row, max_rank):
    """Average ranks of a dense row by count descending, zeros after every detected gene."""
    from scipy.stats import rankdata

    r = rankdata(-row, method="average")
    s = np.maximum(0.0, 1.0 - (r - 1.0) / max_rank)
    s[row == 0] = 0.0
    return s


def test_rank_signal_matches_brute_force_with_ties():
    rng = np.random.default_rng(1)
    X = rng.poisson(0.6, size=(40, 300)).astype(np.float32)
    X[3] = 0  # an empty cell
    S = R.rank_signal(sps.csr_matrix(X), max_rank=50).toarray()
    for i in range(X.shape[0]):
        np.testing.assert_allclose(S[i], _brute_signal(X[i], 50), atol=1e-6)


def test_auc_is_shifted_ucell_and_bounded():
    rng = np.random.default_rng(2)
    X = sps.csr_matrix(rng.poisson(1.0, size=(30, 400)).astype(np.float32))
    M = 100
    sets = [np.arange(0, 20), np.arange(50, 60)]
    auc, _, _ = R.score_sets(X, sets, max_rank=M)
    assert auc.min() >= 0 and auc.max() <= 1
    from scipy.stats import rankdata

    for i in range(5):
        r = rankdata(-X[i].toarray().ravel(), method="average")
        r[X[i].toarray().ravel() == 0] = M + 1
        for j, g in enumerate(sets):
            n = len(g)
            rr = np.minimum(r[g], M + 1)
            ucell = 1 - (rr.sum() - n * (n + 1) / 2) / (n * M)
            mine = (ucell - (n - 1) / (2 * M)) / (1 - (n - 1) / (2 * M))
            assert auc[i, j] == pytest.approx(mine, abs=1e-5)


def test_top_genes_score_one_and_absent_score_zero():
    X = np.zeros((2, 100), np.float32)
    X[0, :10] = np.arange(20, 10, -1)  # set genes are the cell's 10 most abundant
    X[0, 10:60] = 1
    X[1, 50:90] = 3  # set genes undetected
    auc, _, _ = R.score_sets(sps.csr_matrix(X), [np.arange(10)], max_rank=30)
    assert auc[0, 0] == pytest.approx(1.0)
    assert auc[1, 0] == 0.0


def test_null_sets_deterministic_and_matched():
    ids = np.array([f"G{i}" for i in range(1000)], dtype=object)
    bins = R.abundance_bins(np.arange(1000, dtype=float), 50)
    members = [f"G{i}" for i in range(0, 1000, 50)]
    a = R.draw_null_sets(members, ids, bins, 4, seed=7)
    b = R.draw_null_sets(members, ids, bins, 4, seed=7)
    assert a == b
    pos = {g: i for i, g in enumerate(ids)}
    for s in a:
        assert not set(s) & set(members)
        assert sorted(bins[[pos[g] for g in s]].tolist()) == sorted(
            bins[[pos[g] for g in members]].tolist()
        )


# ---------------------------------------------------------------------------------------------------------- registry / data
def test_registry_names_and_rows():
    rows = REG.registry_rows()
    ids = [r["feature_id"] for r in rows]
    assert (
        len(ids) == len(set(ids)) == 53 and len(ids) <= 64
    )  # the cell index takes at most 64 features per version
    for r in rows:
        s, _, n = r["feature_id"].partition("/")
        assert re.match(r"^[a-z][a-z0-9_]{0,47}$", s) and re.match(r"^[a-z][a-z0-9_]{0,47}$", n)
        assert (
            r["storage"] == "shard"
            and r["comparable"] == "atlas"
            and r["kind"] in ("score", "category")
        )
        assert json.loads(r["organisms"]) == ["Homo sapiens", "Mus musculus"]
        assert r["licence"] and r["method_version"].startswith("singlet-features=")
    hyp = next(r for r in rows if r["feature_id"] == "hallmark/hypoxia")
    assert hyp["licence"] == "CC-BY-4.0" and "HALLMARK_HYPOXIA" in hyp["source_ref"]
    ph = next(r for r in rows if r["feature_id"] == "cell_cycle/phase")
    assert json.loads(ph["categories"]) == ["G1", "S", "G2M"]


def test_collections_map_most_genes():
    h = GS.collection("hallmark")
    assert h["licence"] == "CC-BY-4.0" and len(h["sets"]) == 50
    for org in ("human", "mouse"):
        n = [len(d["genes"][org]) for d in h["sets"].values()]
        assert min(n) >= 20
    cc = GS.collection("cell_cycle")
    assert len(cc["sets"]["s_phase"]["genes"]["human"]) >= 40


# --------------------------------------------------------------------------------------------------------- synthetic sample
@dataclass
class _Sample:
    gsm: str
    gse: str
    X: sps.csr_matrix
    gene_names: list
    col: np.ndarray
    species: Optional[str] = "human"
    meta: dict = field(default_factory=dict)


def _synthetic(n=400, seed=0):
    """Cells on the packaged human reference genes, counts drawn from the reference abundance; the first half has the hypoxia
    genes and the G2/M genes raised."""
    ids, syms, _ = GS.abundance("human")
    import pyarrow.parquet as pq_

    mean = np.asarray(
        pq_.read_table(GS.DATA / "abundance_human.parquet").column("mean").to_numpy(), np.float64
    )
    rng = np.random.default_rng(seed)
    lam = np.expm1(mean) * 0.25
    X = rng.poisson(lam, size=(n, len(ids))).astype(np.float32)
    hyp = {g for g, _ in GS.collection("hallmark")["sets"]["hypoxia"]["genes"]["human"]}
    g2m = {g for g, _ in GS.collection("cell_cycle")["sets"]["g2m_phase"]["genes"]["human"]}
    jh = [j for j, g in enumerate(ids) if g in hyp]
    jg = [j for j, g in enumerate(ids) if g in g2m]
    X[: n // 2][:, jh] += rng.poisson(6, size=(n // 2, len(jh)))
    X[: n // 2][:, jg] += rng.poisson(6, size=(n // 2, len(jg)))
    col = rng.permutation(n * 2)[:n].astype(np.int64)
    return _Sample("GSM1", "GSE1", sps.csr_matrix(X), list(syms), col)


def test_score_sample_separates_planted_program(tmp_path):
    s = _synthetic()
    res = P.score_sample(s)
    hyp = res["hallmark"].values["hypoxia"]
    sets = GS.collection("hallmark")["sets"]
    planted = {g for g, _ in sets["hypoxia"]["genes"]["human"]} | {
        g for g, _ in GS.collection("cell_cycle")["sets"]["g2m_phase"]["genes"]["human"]
    }
    control = min(
        (k for k in sets if k != "hypoxia"),
        key=lambda k: len(planted & {g for g, _ in sets[k]["genes"]["human"]}),
    )
    other = res["hallmark"].values[control]
    half = len(hyp) // 2
    assert np.nanmedian(hyp[:half]) > np.nanmedian(hyp[half:]) + 0.2
    assert abs(np.nanmedian(other[:half]) - np.nanmedian(other[half:])) < 0.1
    ph = res["cell_cycle"].values["phase"]
    assert (ph[:half] == 2).mean() > 0.9 and (ph[half:] == 0).mean() > 0.9
    rows = P.summaries(s, res)
    r = next(x for x in rows if x["item"] == "hallmark/hypoxia")
    assert r["n_cells"] == len(hyp) and r["n_hi1"] >= r["n_hi2"]
    g2m = next(x for x in rows if x["item"] == "cell_cycle/phase=G2M")
    assert g2m["n_cells"] == int((ph == 2).sum())


def test_low_depth_cells_and_other_species_get_na():
    s = _synthetic(n=50)
    X = s.X.tolil()
    X[0, :] = 0
    X[0, :20] = 1
    s.X = X.tocsr()
    res = P.score_sample(s)
    assert np.isnan(res["hallmark"].values["hypoxia"][0])
    assert res["cell_cycle"].values["phase"][0] == REG.CATEGORY_NA
    s.species = None
    res = P.score_sample(s)
    assert np.isnan(res["hallmark"].values["hypoxia"]).all()


# ------------------------------------------------------------------------------------------------------------- writer guard
def _table(s=None):
    s = s or _synthetic(n=60)
    res = P.score_sample(s)
    r = res["hallmark"]
    return s, r, O.build_table(s.gsm, s.col, r.values, r.features, r.method_version)


def test_write_roundtrip_and_metadata(tmp_path):
    s, r, t = _table()
    p = O.write_features(
        t, tmp_path / O.sidecar_name(s.gsm, "hallmark"), r.features, gene_symbols=s.gene_names
    )
    back = pq.read_table(p)
    md = {k.decode(): v.decode() for k, v in back.schema.metadata.items()}
    assert md["singlet_features_schema"] == "1" and md["feature_set"] == "hallmark"
    assert md["method_version"] == r.method_version
    assert back.column_names == ["gsm", "col"] + [f.name for f in r.features]
    assert (
        back.schema.field("col").type == pa.int64()
        and back.schema.field("hypoxia").type == pa.float32()
    )


def test_cell_index_reads_the_sidecar(tmp_path):
    atlas = pytest.importorskip("singlet.s1.atlas")
    if not hasattr(atlas, "normalize_feature_sidecar"):
        pytest.skip("cell index (Singlet-Bio/singlet#15) not on this branch")
    s, r, t = _table()
    p = O.write_features(t, tmp_path / "x.parquet", r.features)
    specs = atlas.load_features([f.cell_feature_row() for f in r.features])
    data, entry = atlas.normalize_feature_sidecar(
        p, gsm=s.gsm, specs=specs, n_columns=int(s.col.max()) + 1
    )
    assert entry["n_rows"] == len(s.col) and entry["method_version"] == r.method_version


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (
            lambda t: t.append_column("VEGFA", pa.array(np.zeros(t.num_rows, np.float32))),
            "columns must be exactly",
        ),
        (
            lambda t: t.set_column(
                2, t.column_names[2], pa.array(np.full(t.num_rows, 1.5, np.float32))
            ),
            "NaN or in",
        ),
        (lambda t: t.set_column(1, "col", pa.array(np.zeros(t.num_rows, np.int64))), "unique"),
        (
            lambda t: t.set_column(0, "gsm", pa.array(["GSM1"] * (t.num_rows - 1) + ["GSM2"])),
            "one sample",
        ),
        (
            lambda t: t.set_column(2, t.column_names[2], pa.array(np.zeros(t.num_rows))),
            "must be float",
        ),
        (
            lambda t: t.replace_schema_metadata(
                {
                    b"singlet_features_schema": b"1",
                    b"feature_set": b"hallmark",
                    b"method_version": b"x",
                    b"genes": b"VEGFA",
                }
            ),
            "metadata keys",
        ),
        (
            lambda t: t.replace_schema_metadata(
                {
                    b"singlet_features_schema": b"1",
                    b"feature_set": b"hallmark",
                    b"method_version": b"rs12345 chr1:12345",
                }
            ),
            "method_version",
        ),
    ],
)
def test_guard_rejects(mutate, msg):
    _, r, t = _table()
    with pytest.raises(O.PublicOutputError, match=msg):
        O.assert_public_safe(mutate(t), r.features)


def test_guard_rejects_gene_symbol_feature_names():
    _, r, t = _table()
    with pytest.raises(O.PublicOutputError, match="gene symbol"):
        O.assert_public_safe(t, r.features, gene_symbols=["HYPOXIA"])


def test_category_codes_checked():
    s = _synthetic(n=40)
    r = P.score_sample(s, ("cell_cycle",))["cell_cycle"]
    t = O.build_table(s.gsm, s.col, r.values, r.features, r.method_version)
    O.assert_public_safe(t, r.features)
    bad = t.set_column(2, "phase", pa.array(np.full(t.num_rows, 7, np.uint16)))
    with pytest.raises(O.PublicOutputError, match="categories"):
        O.assert_public_safe(bad, r.features)
