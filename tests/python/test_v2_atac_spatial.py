# SPDX-License-Identifier: MIT
"""Chromatin (ATAC) and spatial members of a sample object (docs/format-v2.md section 14): a synthetic pipeline run directory ->
``modality_members.add_section14_members`` -> ``.s2`` object in the server layout -> ``modalities=["atac", "gene_activity",
"spatial"]`` on the read path.

Two samples: GSM9100001 (a Visium-like sample with ATAC members too; the run's matrices and positions list the barcodes in another
order than the cells table, one matrix barcode is not a cell, one cell is not in the matrices) and GSM9100002 (ATAC only, no
spatial members). The truth is the dense arrays the fixture was built from."""

from __future__ import annotations

import gzip
import json
import pathlib
import sys
import warnings

import numpy as np
import pytest

try:  # a source checkout without an installed package
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
pytest.importorskip("zstandard")

from singlet.v2 import cells as cells_mod  # noqa: E402
from singlet.v2 import (
    client,  # noqa: E402
    util,  # noqa: E402
)
from singlet.v2 import container as K  # noqa: E402
from singlet.v2 import manifest as M  # noqa: E402
from singlet.v2 import modality_members as MM  # noqa: E402

GSE = "GSE900002"
G1, G2 = "GSM9100001", "GSM9100002"
N = 40  # columns of the cells table
CALLED = np.arange(N) % 4 != 3  # 30 called


def _barcodes(prefix: str, n: int):
    return [f"{prefix}{i:04d}ACGTACGTAC" for i in range(n)]


def _write_mtx(path, dense):
    """Coordinate MatrixMarket gz of a features x barcodes dense integer array (the pipeline's write_mtx layout)."""
    r, c = np.nonzero(dense)
    lines = ["%%MatrixMarket matrix coordinate integer general", "%", f"{dense.shape[0]} {dense.shape[1]} {len(r)}"]
    # column-major, as the pipeline writes it
    for j in np.unique(c):
        for i in r[c == j]:
            lines.append(f"{i + 1} {j + 1} {int(dense[i, j])}")
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")


def _write_names(path, names):
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(names) + "\n")


class Truth:
    pass


def _make_run(run, cells_bc, seed, spatial: bool):
    """A run directory; returns the dense matrices re-laid on the cells table (what the member must hold)."""
    rng = np.random.default_rng(seed)
    run.mkdir()
    peaks = [f"chr1:{1000 * i}-{1000 * i + 500}" for i in range(1, 26)]
    genes = [f"ENSG{i:011d}_GENE{i}" for i in range(12)]
    # matrix barcodes: the cells in reverse order, minus one cell (never in the matrices), plus one stranger
    absent = cells_bc[5]
    mat_bc = [b for b in reversed(cells_bc) if b != absent] + ["Z0000NOTACELL"]
    out = {"peaks": peaks, "genes": genes, "absent": absent}
    for name, feats, hi in (("atac_peak_counts", peaks, 9), ("atac_gene_activity", genes, 30)):
        d = rng.integers(0, hi, size=(len(feats), len(mat_bc)))
        d[rng.random(d.shape) < 0.6] = 0
        _write_mtx(run / f"{name}.mtx.gz", d)
        _write_names(run / f"{name}_features.tsv.gz", feats)
        _write_names(run / f"{name}_barcodes.tsv.gz", mat_bc)
        truth = np.zeros((len(feats), len(cells_bc)), dtype=np.int64)
        for j, b in enumerate(mat_bc):
            if b in cells_bc:
                truth[:, cells_bc.index(b)] = d[:, j]
        out[name] = truth
    (run / "fragments.tsv.gz").write_bytes(gzip.compress(b"chr1\t100\t200\tAAAA\t1\n"))
    (run / "fragments.tsv.gz.tbi").write_bytes(b"TBI\x01fake-index")
    (run / "peaks.bed").write_text("\n".join(f"chr1\t{1000 * i}\t{1000 * i + 500}" for i in range(1, 26)) + "\n")
    (run / "atac_summary.json").write_text(json.dumps({"schema": "singlet-atac-summary/1", "qc": {"pass": True}}))
    if spatial:
        order = list(reversed(range(N)))  # positions listed in another order than the cells table
        rows = ["barcode\tspot\tarray_row\tarray_col\tin_tissue\ttissue_prob\tprimary\tumis"]
        for i in order:
            rows.append(f"{cells_bc[i]}\tS{i}\t{i // 8}\t{2 * (i % 8) + (i // 8) % 2}\t{int(i % 3 != 0)}\t0.9\t1\t{100 + i}")
        rows.append("Q0000EXTRAROW\tSX\t99\t99\t0\t0.1\t0\t1")  # a spot that is no column of the cells table
        (run / "spatial_positions.tsv").write_text("\n".join(rows) + "\n")
        (run / "spatial_summary.json").write_text(json.dumps({"schema": "singlet-spatial-summary/1", "tissue_spots": 27}))
    return out


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp("v2atac")
    runs = tmp_path_factory.mktemp("runs")
    samples, truths, specs_by = [], {}, {}
    for gsm, prefix, seed, spatial in ((G1, "A", 1, True), (G2, "C", 2, False)):
        bcs = _barcodes(prefix, N)
        truth = _make_run(runs / gsm, bcs, seed, spatial)
        specs = MM.add_section14_members(str(runs / gsm), bcs)
        specs_by[gsm] = specs
        w = K.S2Writer(gsm)
        ct = cells_mod.CellsTable(N)
        ct.add("barcode", cells_mod.DT_STR, bcs)
        ct.add("called", cells_mod.DT_BOOL, CALLED)
        w.add_cells("cells", ct.encode(), ct.digest(), {"n_rows": N}, flags=K.FLAG_ANNDATA)
        MM.add_to_writer(w, specs)
        obj = w.build()
        key = f"v2/s/{gsm}/{util.d16(obj).hex()}.s2"
        path = root.joinpath(*key.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(obj)
        members = {
            i.name: {"kind": i.kind, "flags": i.flags, "off": i.off, "len": i.len, "digest": i.digest.hex(), **i.meta.get("_", {})}
            for i in w.infos
        }
        from singlet import _pz2

        for i in w.infos:
            if i.kind == K.KIND_PZ2:
                h = _pz2.read_header(obj[i.off : i.off + 256])
                members[i.name].update(m=h["m"], n=h["n"], nnz=h["nnz"], codec=h["codec"], chunk_cols=h["chunk_cols"], n_chunks=h["n_chunks"])
        samples.append(M.sample_entry(gsm, key, obj, N, int(CALLED.sum()), members, {}))
        truth["bcs"] = bcs
        truth["obj"] = obj
        truths[gsm] = truth
    meta = {
        "schema_version": "1.0",
        "gse_id": GSE,
        "series_title": "synthetic atac",
        "gsm_meta": {g: {"gsm_id": g, "organism": "Homo sapiens", "protocol": "10xatac", "n_cells": int(CALLED.sum())} for g in (G1, G2)},
    }
    sfiles = {"study_meta.json": M.study_file_entry(json.dumps(meta).encode())}
    man = M.build_manifest(GSE, samples, sfiles, "GRCh38-2024-A", {"singlet_version": "test"}, {}, {}, "2026-10-25T00:00:00Z")
    mp = root / "v2" / "m" / f"{GSE}.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_bytes(M.dumps_manifest(man))
    return root, truths, specs_by


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("SINGLET_DATA_BASE", raising=False)
    monkeypatch.delenv("SINGLET_LEGACY_FALLBACK", raising=False)
    client.clear_memory_cache()
    yield tmp_path
    client.clear_memory_cache()


def _called_order(bcs):
    """Cell indices of the called cells in sorted-barcode order (the default ``cell_order``)."""
    called = np.flatnonzero(CALLED)
    return called[np.argsort(np.array(bcs, dtype=object)[called], kind="stable")]


# ============================================================================================================ member specs
def test_member_specs(study):
    _, truths, specs_by = study
    s1 = {s.name: s for s in specs_by[G1]}
    assert list(s1) == [
        "atac_fragments.tsv.gz",
        "atac_fragments.tsv.gz.tbi",
        "atac_peaks.bed",
        "atac_peak_counts.1pz",
        "atac_gene_activity.1pz",
        "atac_summary.json",
        "spatial_positions.tsv",
        "spatial_summary.json",
    ]
    assert all(s.flags == 0 for s in s1.values())  # bit 0 (legacy) clear
    assert s1["atac_fragments.tsv.gz"].as_written and s1["atac_fragments.tsv.gz.tbi"].as_written
    pc = s1["atac_peak_counts.1pz"].meta
    assert (pc["m"], pc["n"]) == (25, N)
    assert pc["n_dropped_barcodes"] == 1 and pc["n_empty_columns"] == 1
    assert pc["names"] == {"rows": "inline", "cols": "cells"}
    assert s1["spatial_positions.tsv"].meta == {"n_rows": N, "reordered": True, "n_dropped_rows": 1}
    # no run files, no members
    assert MM.add_section14_members(str(study[0]), truths[G1]["bcs"]) == []


def test_object_members_and_kinds(study):
    root, truths, specs_by = study
    r = K.S2Reader(truths[G1]["obj"])
    m = r.members
    assert m["atac_peak_counts.1pz"].kind == K.KIND_PZ2 and m["atac_gene_activity.1pz"].kind == K.KIND_PZ2
    for n in ("atac_fragments.tsv.gz", "atac_fragments.tsv.gz.tbi"):
        assert m[n].kind == K.KIND_RAW  # as written: byte-range readable bgzip
    assert m["spatial_positions.tsv"].kind in (K.KIND_STORED, K.KIND_RAW)
    assert all(not mi.legacy for n, mi in m.items() if n != "cells")
    # fragments bytes are exactly the run's
    mm = m["atac_fragments.tsv.gz"]
    assert truths[G1]["obj"][mm.off : mm.off + mm.len] == specs_by[G1][0].data


def test_incomplete_matrix_files_refuse(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    _write_mtx(run / "atac_peak_counts.mtx.gz", np.ones((2, 2), dtype=int))
    with pytest.raises(ValueError, match="missing"):
        MM.add_section14_members(str(run), ["AAA", "CCC"])


def test_positions_missing_cell_refuses(tmp_path):
    p = tmp_path / "spatial_positions.tsv"
    p.write_text("barcode\tarray_row\tarray_col\tin_tissue\nAAA\t0\t0\t1\n")
    with pytest.raises(ValueError, match="lacks 1 of the 2"):
        MM.aligned_positions(str(p), ["AAA", "CCC"])


def test_positions_already_aligned_are_stored_as_written(tmp_path):
    p = tmp_path / "spatial_positions.tsv"
    txt = "barcode\tarray_row\tarray_col\tin_tissue\nAAA\t0\t0\t1\nCCC\t1\t3\t0\n"
    p.write_bytes(txt.encode())
    data, info = MM.aligned_positions(str(p), ["AAA", "CCC"])
    assert data.decode() == txt and info["reordered"] is False


# ============================================================================================================ read path
def test_vocabulary():
    assert client.normalize_modalities(["peaks"]) == ("atac",)
    assert client.normalize_modalities("chromatin") == ("atac",)
    assert client.normalize_modalities(["spatial", "gene_activity", "atac"]) == ("atac", "gene_activity", "spatial")
    assert client.MODALITIES["atac"].members == ("atac_peak_counts.1pz",)
    assert client.MODALITIES["gene_activity"].members == ("atac_gene_activity.1pz",)
    assert client.MODALITIES["spatial"].members == ("spatial_positions.tsv",)


def test_read_back_all_three(study, env):
    root, truths, _ = study
    st = client.open_study(GSE, base=str(root))
    assert st.modalities(G1) == ["atac", "gene_activity", "spatial"]
    assert st.modalities(G2) == ["atac", "gene_activity"]
    t = truths[G1]
    ad = st.to_anndata([G1], modalities=["atac", "gene_activity", "spatial"])
    order = _called_order(t["bcs"])
    assert ad.n_obs == int(CALLED.sum()) and ad.n_vars == 0 and ad.X is None
    assert list(ad.obs_names) == [f"{G1}_{t['bcs'][i]}" for i in order]
    for k, key, feats in (("atac", "atac_peak_counts", t["peaks"]), ("gene_activity", "atac_gene_activity", t["genes"])):
        m = ad.obsm[k].toarray()
        assert m.shape == (len(order), len(feats))
        assert list(ad.uns["obsm_features"][k]) == feats
        np.testing.assert_array_equal(m, t[key][:, order].T)
    # the cell without a matrix column (cells-table order index 5) has an empty column if it is called
    if CALLED[5]:
        assert ad.obsm["atac"].toarray()[list(order).index(5)].sum() == 0
    # spatial
    sp = ad.obsm["spatial"]
    assert sp.shape == (len(order), 2)
    exp_row = order // 8
    exp_col = 2 * (order % 8) + (order // 8) % 2
    np.testing.assert_array_equal(sp[:, 0], exp_col)  # x = array_col
    np.testing.assert_array_equal(sp[:, 1], exp_row)  # y = array_row
    np.testing.assert_array_equal(ad.obs["array_row"].to_numpy(), exp_row)
    np.testing.assert_array_equal(ad.obs["array_col"].to_numpy(), exp_col)
    np.testing.assert_array_equal(ad.obs["in_tissue"].to_numpy(), order % 3 != 0)
    assert ad.uns["spatial"][G1] == {"schema": "singlet-spatial-summary/1", "tissue_spots": 27}
    assert ad.uns["singlet_v2"]["modalities"] == ["atac", "gene_activity", "spatial"]


def test_aliases_and_single_modality(study, env):
    root, truths, _ = study
    st = client.open_study(GSE, base=str(root))
    ad = st.to_anndata([G1], modalities="peaks")
    assert list(ad.obsm.keys()) == ["atac"] and "spatial" not in ad.uns and "in_tissue" not in ad.obs.columns
    sub = client.modality_anndata(ad, "chromatin")
    assert sub.shape == (ad.n_obs, 25)


def test_two_samples_one_without_spatial(study, env):
    root, truths, _ = study
    st = client.open_study(GSE, base=str(root))
    with pytest.warns(UserWarning, match="spatial: .*GSM9100002"):
        ad = st.to_anndata([G1, G2], modalities=["atac", "spatial"])
    n1 = int(CALLED.sum())
    assert ad.n_obs == 2 * n1
    sp = ad.obsm["spatial"]
    assert np.isfinite(sp[:n1]).all() and np.isnan(sp[n1:]).all()
    assert not ad.obs["in_tissue"].to_numpy()[n1:].any()
    assert set(ad.uns["spatial"]) == {G1}
    # peaks of both samples are the same names, first-seen order
    assert list(ad.uns["obsm_features"]["atac"]) == truths[G1]["peaks"]
    # G2 alone has no spatial: the explicit request is refused before any byte is read
    with pytest.raises(client.ModalityUnavailable):
        st.to_anndata([G2], modalities="spatial")


def test_only_selected_members_are_read(study, env, monkeypatch):
    root, truths, _ = study
    log = []
    orig = client._MmapSource.read

    def read(self, off, length, *, parallel=False):
        log.append((int(off), int(length)))
        return orig(self, off, length, parallel=parallel)

    monkeypatch.setattr(client._MmapSource, "read", read)
    st = client.open_study(GSE, base=pathlib.Path(root).as_uri())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # anndata still reads pandas options that pandas 3.1 deprecates; that third-party warning is not under test
        warnings.filterwarnings("ignore", message=r".*deprecated and will be removed in pandas.*")
        st.to_anndata([G1], modalities="gene_activity")
    sm = st.samples[G1]
    touched = set()
    for a, n in log:
        b = a + n
        touched |= {n for n, m in sm.members.items() if m.len and a < m.off + m.len and m.off < b}
    assert touched == {"cells", "atac_gene_activity.1pz"}, touched
