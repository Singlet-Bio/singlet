# SPDX-License-Identifier: MIT
"""singlet.adt QC, normalisation, hashtag calls and v2 members on a synthetic sample."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from singlet.adt import hto as H
from singlet.adt import normalize as N
from singlet.adt import panel as P
from singlet.adt import qc as Q
from singlet.adt import store as ST

RNG = np.random.default_rng(3)


def synth(n_cells=400, n_empty=1500, hashtags=4):
    """CD3/CD19/CD14 + an isotype + hashtags; RNA with matching CD3E/CD19/CD14."""
    types = RNG.integers(0, 3, n_cells)  # T, B, mono
    ab = ["CD3", "CD19", "CD14", "Mouse IgG1 Isotype"]
    k = len(ab) + hashtags
    X = np.zeros((k, n_cells + n_empty), dtype=np.int64)
    ambient = np.array([30, 20, 25, 5] + [10] * hashtags, float)
    X[:, n_cells:] = RNG.poisson(ambient[:, None] / ambient.sum() * 8, size=(k, n_empty))
    for j, t in enumerate(types):
        X[:len(ab), j] = RNG.poisson(ambient[:len(ab)] / 10)
        X[t, j] += RNG.poisson(300)
        if hashtags:
            X[len(ab):, j] = RNG.poisson(3, hashtags)
            X[len(ab) + j % hashtags, j] += RNG.poisson(200)
    genes = ["ENSG00000198851|CD3E", "ENSG00000177455|CD19", "ENSG00000170458|CD14", "ENSG00000000001|OTHER"]
    E = np.zeros((4, n_cells + n_empty), dtype=np.int64)
    E[3] = RNG.poisson(50, n_cells + n_empty)
    for j, t in enumerate(types):
        E[t, j] = RNG.poisson(5)
    names = ab + [f"Hashtag {i + 1}" for i in range(hashtags)]
    panel = pd.DataFrame({"name": names, "sequence": [f"{'ACGT'[i % 4] * 14}{'ACGT'[i // 4]}" for i in range(k)],
                          "kind": ["antibody"] * 3 + ["isotype"] + ["hashtag"] * hashtags,
                          "ensembl": ["ENSG00000198851", "ENSG00000177455", "ENSG00000170458", ""] + [""] * hashtags,
                          "barcode_id": [f"A{i:04d}" for i in range(k)]})
    called = np.zeros(n_cells + n_empty, bool)
    called[:n_cells] = True
    bcs = ["".join(RNG.choice(list("ACGT"), 16)) + "-1" for _ in range(n_cells + n_empty)]
    return sparse.csc_matrix(X.astype(np.uint32)), panel, called, sparse.csc_matrix(E.astype(np.uint32)), genes, \
        types, bcs


STATS = {"reads": 1_000_000, "reads_valid_barcode": 900_000, "reads_feature_exact": 800_000,
         "reads_feature_corrected": 20_000, "saturation": 0.6, "molecules": 300_000, "reads_usable": 780_000}


def test_library_qc_passes_and_concords():
    M, panel, called, E, genes, types, _ = synth()
    res = Q.library_qc(M, panel, called, STATS, np.zeros(len(panel)), E, genes)
    lib = res["library"]
    assert lib["cell_empty_ratio"] > 10
    assert lib["rna_concordance"] > 0.5 and abs(lib["rna_concordance_null"]) < 0.2
    assert lib["isotype_share_median"] < 0.1
    assert lib["hto_singlet_share"] > 0.9
    assert lib["verdict"] in ("pass", "warn") and not lib["failed"]
    df = Q.cell_table(M, panel, res["cells"], res["hto"])
    assert len(df) == called.sum() and set(df.columns) >= {"col", "adt_total", "hto_call", "isotype_share"}
    # hashtag calls are the designed hashtag
    assert (df.hto_call.values == np.arange(len(df)) % 4).mean() > 0.9


def test_unpaired_library_fails_pairing_check():
    M, panel, called, E, genes, _, _ = synth(hashtags=0)
    shuffled = RNG.permutation(len(called))  # ADT columns that do not belong to these cells
    res = Q.library_qc(M[:, shuffled], panel, called, STATS, np.zeros(len(panel)), E, genes)
    lib = res["library"]
    assert lib["rules"]["cell_empty_ratio"] == "fail"
    assert lib["verdict"] == "fail"


def test_clr_and_dsb_shapes_and_signal():
    M, panel, called, *_ = synth(hashtags=0)
    X = M.toarray()
    c = N.clr(X[:, called])
    assert c.shape == (4, called.sum()) and np.isfinite(c).all()
    emp = N.empty_droplets(X.sum(0), called)
    d = N.dsb(X[:, called], X[:, emp], isotype_rows=np.array([3]))
    assert d.shape == c.shape
    # positive population is far above background, negatives near 0
    assert np.median(d[0][X[0, called] > 100]) > 5
    assert abs(np.median(d[0][X[0, called] < 50])) < 2


def test_hashed_drops_doublets():
    amb = np.array([10, 10, 10.0])
    cells = np.array([[200, 3, 2], [180, 190, 2], [1, 2, 3]]).T  # singlet, doublet, negative (columns)
    cells = np.hstack([cells] + [np.array([[150 + i, 2, 3]]).T for i in range(30)])
    call, lt, ls = H.hashed_drops(cells, amb)
    assert call[0] == 0 and call[1] == -2 and call[2] == -1


def test_catalog_loads_and_has_hashtags_and_isotypes():
    cat = P.load_catalog()
    assert len(cat) > 600 and cat.sequence.str.len().eq(15).all()
    assert (cat.kind == "hashtag").sum() >= 30 and (cat.kind == "isotype").sum() >= 5
    assert cat.sequence.is_unique


def test_members_roundtrip():
    pytest.importorskip("zstandard")
    M, panel, called, E, genes, _, bcs = synth(n_cells=50, n_empty=100)
    res = Q.library_qc(M, panel, called, STATS, np.zeros(len(panel)), E, genes)
    mem = ST.build_members(M, panel, bcs, res["library"], np.arange(len(panel)), {"offset": 10})
    assert set(mem) == {"adt_counts.1pz", "hto_counts.1pz", "adt_panel.json"}
    buf = ST.build_sidecar("GSM0000001", mem)
    back = ST.read_sidecar(buf)
    A, names = back["adt"]
    assert names == list(panel.name[:4])
    assert (A.toarray() == M[:4].toarray()).all()
    Hm, hn = back["hto"]
    assert (Hm.toarray() == M[4:].toarray()).all()
    doc = back["panel"]
    assert doc["schema"] == ST.PANEL_SCHEMA and len(doc["features"]) == len(panel)
    assert json.dumps(doc)  # plain JSON


def test_hto_feature_sidecar():
    pytest.importorskip("pyarrow")
    call = np.array([0, 1, -1, -2])
    t, names = H.feature_sidecar("GSM1", np.array([3, 5, 8, 9]), call, np.array([3.0, 1.0, 0.2, 0.1]), ["H1", "H2"])
    assert t.column("hto/call").to_pylist() == [2, 3, 0, 1]
    m = t.column("hto/margin").to_pylist()
    assert m[2] == 0 and 0 < m[1] < m[0] < 1
    assert names == ["negative", "doublet", "H1", "H2"]
    assert t.schema.metadata[b"singlet_features_schema"] == b"1"


def test_classify_unmatched():
    pan = ["GTCAACTCTTTAGCG"]
    assert P.classify_unmatched("GGGGGGGGGGGGGGG", pan) == "low_complexity"
    assert P.classify_unmatched("TCAACTCTTTAGCGG", pan) == "shifted"
    assert P.classify_unmatched("GGTCAACTCTTTAGC", pan) == "shifted"
    assert P.classify_unmatched("ACGTTGCAACGTAGC", pan) == "other"


def test_discover_names_catalog_barcodes_and_flags_unknown():
    cat = P.load_catalog()
    a, b = cat.sequence.iloc[10], cat.sequence.iloc[20]
    unknown = "ACGTTGCAACGTAGC"
    assert unknown not in set(cat.sequence)
    feats = [a.encode()] * 900 + [b.encode()] * 500 + [b"G" * 15] * 50
    pan = P.discover(feats, cat)
    assert pan.status == "ok" and set(pan.features.sequence) == {a, b}
    pan2 = P.discover(feats + [unknown.encode()] * 400, cat)
    assert pan2.status == "incomplete" and pan2.notes["unexplained_top"][0][0] == unknown
    assert pan2.notes["low_complexity_share"] > 0


def test_discover_flags_reused_and_cross_species_barcodes():
    cat = P.load_catalog()
    reused = cat[cat.reused != ""].sequence.iloc[:3].tolist()
    feats = [q.encode() for q in reused for _ in range(300)]
    pan = P.discover(feats, cat, organism="Homo sapiens")
    assert pan.status == "ambiguous"
    assert pan.notes["reused_share"] > 0.9
    mouse_only = cat[(cat.reactivity == "Mouse") & (cat.reused == "") & (cat.kind == "antibody")].sequence.iloc[:2]
    pan = P.discover([q.encode() for q in mouse_only for _ in range(300)], cat, organism="Homo sapiens")
    assert pan.status == "ambiguous" and pan.notes["reactivity_mismatch_share"] > 0.9
    pan = P.discover([q.encode() for q in mouse_only for _ in range(300)], cat, organism="Mus musculus")
    assert pan.status == "ok"


def test_modalities_vocabulary_has_adt_and_hto():
    from singlet.v2 import client

    assert client.MODALITIES["adt"].members == (ST.MEMBERS[0],) and client.MODALITIES["adt"].slot == "obsm"
    assert client.MODALITIES["hto"].members == (ST.MEMBERS[1],)
    for alias, name in (("protein", "adt"), ("cite", "adt"), ("hashtag", "hto")):
        assert client._MODALITY_ALIASES[alias] == name


def test_reference_with_repeated_sequences_and_missing_ids(tmp_path):
    cat = P.load_catalog()
    h1 = cat[cat.kind == "hashtag"].iloc[0]
    p = tmp_path / "ref.csv"
    p.write_text("name,sequence\nControl 1-HTO,%s\nControl 2-HTO,%s\nCustomAb,ACGTTGCAACGTAGC\nCustomAb,TTTTGGGGCCCCAAA\n"
                 % (h1.sequence, h1.sequence))
    ref = P.annotate_from_catalog(P.read_feature_reference(str(p)), cat)
    assert ref.sequence.is_unique and ref.name.is_unique
    assert ref.name.iloc[0] == h1.specificity and "Control 2-HTO" in ref.aliases.iloc[0]


def test_mixture_calls_singlets_and_doublets():
    rng = np.random.default_rng(1)
    k, n = 6, 3000
    X = rng.poisson(rng.gamma(2, 5, size=(1, n)) * np.ones((k, 1)) * 0.6)  # cell-varying background
    true = rng.integers(0, k, n)
    X[true, np.arange(n)] += rng.poisson(150, n)
    d = rng.choice(n, 150, replace=False)
    X[(true[d] + 1) % k, d] += rng.poisson(150, len(d))
    call, score, post = H.mixture_calls(X)
    s = np.setdiff1d(np.arange(n), d)
    assert (call[s] == true[s]).mean() > 0.97 and (call[d] == -2).mean() > 0.9
    assert post.shape == X.shape and ((score >= 0) & (score <= 1)).all()
