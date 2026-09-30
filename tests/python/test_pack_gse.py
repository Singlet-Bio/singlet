# SPDX-License-Identifier: MIT
"""Tests for singlet.bundle.pack_gse: hollow-sample screening and manifest fields.

pack_gse copies ``.1pz`` files verbatim and only reads their headers, so these
tests build synthetic pipeline output with hand-written TP1Z headers and need
no compiled codec.
"""

from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path

import pandas as pd
import pytest
import singlet
from singlet import bundle as bundle_mod

GSE = "GSE000077"
GOOD, GOOD2, HOLLOW = "GSM0000771", "GSM0000772", "GSM0000773"


def _pz_bytes(m: int, n: int, nnz: int, pad: int = 600) -> bytes:
    """A TP1Z file with the given header dims (the payload is never decoded)."""
    header = struct.pack("<IHBBIIQ", 0x5A315054, 1, 0, 0, m, n, nnz)
    return header + b"\0" * (96 - len(header)) + b"\0" * pad


def _write_sample(results: Path, gsm: str, *, n_called: int, exon: bytes | None) -> Path:
    out = results / gsm / "out"
    out.mkdir(parents=True)
    (out / "pileup_stats.json").write_text("{}")
    (out / "summary.json").write_text(
        json.dumps({"n_cells_called": n_called, "reference_build": "GRCh38", "status": "success"})
    )
    (out / "gene_expression.tsv").write_text("gene_id\tgene_name\nENSG01\tAAA\nENSG02\tBBB\n")
    (out / "cell_calls.tsv").write_text("barcode\tis_cell\nAAAC\tTrue\nAAAG\tFalse\n")
    if exon is not None:
        (out / "exon_counts.1pz").write_bytes(exon)
    return out


def _catalog(*gsms: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gse_id": [GSE] * len(gsms),
            "gsm_id": list(gsms),
            "organism": ["Homo sapiens"] * len(gsms),
        }
    )


def _manifest(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("manifest.json"))


@pytest.fixture
def results(tmp_path):
    r = tmp_path / "results"
    _write_sample(r, GOOD, n_called=1, exon=_pz_bytes(2, 2, 3))
    _write_sample(r, GOOD2, n_called=1, exon=_pz_bytes(2, 2, 1))
    _write_sample(r, HOLLOW, n_called=500, exon=_pz_bytes(0, 0, 0, pad=16))
    return r


class TestHollowScreening:
    def test_hollow_sample_is_excluded_and_recorded(self, results, tmp_path):
        out = tmp_path / f"{GSE}.singlet"
        with pytest.warns(UserWarning, match="hollow"):
            bundle_mod.pack_gse(
                GSE,
                results,
                "unused.parquet",
                out,
                verbose=False,
                _catalog_df=_catalog(GOOD, GOOD2, HOLLOW),
            )
        manifest = _manifest(out)
        assert manifest["gsm_ids"] == [GOOD, GOOD2]
        assert manifest["n_gsms"] == 2
        assert [e["gsm_id"] for e in manifest["excluded_samples"]] == [HOLLOW]
        assert "0x0" in manifest["excluded_samples"][0]["reason"]
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            study_meta = json.loads(zf.read("study_meta.json"))
        assert not any(n.startswith(f"samples/{HOLLOW}/") for n in names)
        assert f"samples/{GOOD}/exon_counts.1pz" in names
        assert HOLLOW not in study_meta["gsm_meta"]

    def test_clean_study_has_empty_excluded_list(self, results, tmp_path):
        out = tmp_path / "clean.singlet"
        bundle_mod.pack_gse(
            GSE, results, "unused.parquet", out, verbose=False, _catalog_df=_catalog(GOOD)
        )
        assert _manifest(out)["excluded_samples"] == []

    def test_strict_raises(self, results, tmp_path):
        with pytest.raises(ValueError, match=f"hollow.*{HOLLOW}"):
            bundle_mod.pack_gse(
                GSE,
                results,
                "unused.parquet",
                tmp_path / "x.singlet",
                verbose=False,
                strict=True,
                _catalog_df=_catalog(GOOD, HOLLOW),
            )
        assert not (tmp_path / "x.singlet").exists()

    def test_all_hollow_raises_even_without_strict(self, tmp_path):
        r = tmp_path / "results"
        _write_sample(r, GOOD, n_called=10, exon=None)
        _write_sample(r, HOLLOW, n_called=10, exon=_pz_bytes(0, 0, 0))
        with pytest.raises(ValueError, match="all 2 done GSMs are hollow"):
            bundle_mod.pack_gse(
                GSE,
                r,
                "unused.parquet",
                tmp_path / "x.singlet",
                verbose=False,
                _catalog_df=_catalog(GOOD, HOLLOW),
            )

    def test_manifest_records_real_singlet_version(self, results, tmp_path):
        out = tmp_path / "v.singlet"
        bundle_mod.pack_gse(
            GSE, results, "unused.parquet", out, verbose=False, _catalog_df=_catalog(GOOD)
        )
        assert _manifest(out)["singlet_version"] == singlet.__version__
        assert bundle_mod._singlet_version() == singlet.__version__

    def test_cli_strict_flag(self, results, tmp_path):
        cat = tmp_path / "catalog.parquet"
        _catalog(GOOD, HOLLOW).to_parquet(cat)
        out = tmp_path / "cli.singlet"
        argv = ["pack", "--gse", GSE, "--results", str(results), "--catalog", str(cat)]
        argv += ["--out", str(out), "--quiet"]
        with pytest.raises(ValueError, match="Refusing"):
            bundle_mod.main([*argv, "--strict"])
        with pytest.warns(UserWarning):
            assert bundle_mod.main(argv) == 0
        assert _manifest(out)["gsm_ids"] == [GOOD]

    def test_cli_without_command_prints_help(self, capsys):
        assert bundle_mod.main([]) == 1
        assert "pack" in capsys.readouterr().out


class TestHollowReason:
    def test_zero_called_cells_is_not_hollow(self, tmp_path):
        out = _write_sample(tmp_path, GOOD, n_called=0, exon=_pz_bytes(0, 0, 0))
        assert bundle_mod._hollow_reason(out) is None

    def test_missing_matrix(self, tmp_path):
        out = _write_sample(tmp_path, GOOD, n_called=5, exon=None)
        assert "missing" in bundle_mod._hollow_reason(out)

    def test_zero_by_zero(self, tmp_path):
        out = _write_sample(tmp_path, GOOD, n_called=5, exon=_pz_bytes(0, 0, 0))
        assert "0x0" in bundle_mod._hollow_reason(out)

    def test_all_zero_matrix(self, tmp_path):
        out = _write_sample(tmp_path, GOOD, n_called=5, exon=_pz_bytes(10, 20, 0))
        assert "no non-zero" in bundle_mod._hollow_reason(out)

    def test_tiny_unreadable_file(self, tmp_path):
        out = _write_sample(tmp_path, GOOD, n_called=5, exon=b"truncated")
        assert "9 bytes" in bundle_mod._hollow_reason(out)

    def test_large_non_tp1z_file_is_not_judged(self, tmp_path):
        out = _write_sample(tmp_path, GOOD, n_called=5, exon=b"1PZ02" + b"\0" * 2000)
        assert bundle_mod._hollow_reason(out) is None

    def test_good_matrix(self, tmp_path):
        out = _write_sample(tmp_path, GOOD, n_called=5, exon=_pz_bytes(10, 20, 30))
        assert bundle_mod._hollow_reason(out) is None

    def test_unreadable_summary_counts_as_no_cells(self, tmp_path):
        out = _write_sample(tmp_path, GOOD, n_called=5, exon=None)
        (out / "summary.json").write_text("{not json")
        assert bundle_mod._hollow_reason(out) is None


class TestHeaderAndCellCalls:
    def test_header_dims(self, tmp_path):
        p = tmp_path / "m.1pz"
        p.write_bytes(_pz_bytes(7, 11, 13))
        assert bundle_mod._pz_header_dims(p) == (7, 11, 13)

    def test_header_dims_rejects_other_files(self, tmp_path):
        p = tmp_path / "m.1pz"
        p.write_bytes(b"not a pz file at all, but long enough to hold a header")
        assert bundle_mod._pz_header_dims(p) is None
        assert bundle_mod._pz_header_dims(tmp_path / "missing.1pz") is None
        assert bundle_mod._pz_dims_from_bytes(b"TP1Z") is None

    @pytest.mark.parametrize("column", ["barcode", "cb", "cell_barcode", "CB"])
    def test_barcode_column_aliases(self, tmp_path, column):
        (tmp_path / "cell_calls.tsv").write_text(f"{column}\tis_cell\nA\ttrue\nB\tFALSE\nC\t1\n")
        assert bundle_mod._get_called_barcodes(tmp_path) == ["A", "C"]

    def test_without_is_cell_every_row_is_called(self, tmp_path):
        (tmp_path / "cell_calls.tsv").write_text("cb\tn_umi\nA\t5\nB\t9\n")
        assert bundle_mod._get_called_barcodes(tmp_path) == ["A", "B"]

    def test_unknown_column_falls_back_to_first(self, tmp_path):
        (tmp_path / "cell_calls.tsv").write_text("bc\tn_umi\nA\t5\n")
        assert bundle_mod._get_called_barcodes(tmp_path) == ["A"]

    def test_auto_barcodes_fallback(self, tmp_path):
        (tmp_path / "auto_barcodes.tsv").write_text("A\nB\n")
        assert bundle_mod._get_called_barcodes(tmp_path) == ["A", "B"]
        assert bundle_mod._get_called_barcodes(tmp_path / "nothing") == []

    def test_is_cell_mask_handles_bool_numeric_and_text(self):
        assert list(bundle_mod._is_cell_mask(pd.Series([True, False]))) == [True, False]
        assert list(bundle_mod._is_cell_mask(pd.Series([1, 0, 2]))) == [True, False, True]
        assert list(bundle_mod._is_cell_mask(pd.Series(["True", "False", "yes", None]))) == [
            True,
            False,
            True,
            False,
        ]

    def test_called_from_calls_without_barcode_column(self):
        df = pd.DataFrame({"bc": ["A"]})
        assert bundle_mod._called_from_calls(df) is None
        assert bundle_mod._called_from_calls(df, first_column_fallback=True) == ["A"]


class TestUnsSafe:
    def test_slash_keys_none_and_mixed_lists(self):
        out = bundle_mod._uns_safe(
            {
                "a/b": 1,
                "none": None,
                "empty": [],
                "strs": ["x", "y"],
                "nums": [1, 2.5],
                "mixed": [1, "x"],
                "records": [{"pmid": 1}],
                "nested": {"c/d": "v"},
                "obj": object,
            }
        )
        assert out["a__b"] == 1
        assert "none" not in out and "empty" not in out
        assert out["strs"] == ["x", "y"]
        assert out["nums"] == [1, 2.5]
        assert json.loads(out["mixed"]) == [1, "x"]
        assert json.loads(out["records"]) == [{"pmid": 1}]
        assert out["nested"] == {"c__d": "v"}
        assert isinstance(out["obj"], str)

    def test_manifest_checksums_become_columns(self):
        manifest = {
            "gse_id": GSE,
            "checksums": {f"samples/{GOOD}/exon_counts.1pz": "ab", "manifest.json": "cd"},
        }
        out = bundle_mod._manifest_for_uns(manifest)
        assert out["checksums"] == {
            "path": [f"samples/{GOOD}/exon_counts.1pz", "manifest.json"],
            "sha256": ["ab", "cd"],
        }
        assert "checksums" in manifest and "/" in next(iter(manifest["checksums"]))
