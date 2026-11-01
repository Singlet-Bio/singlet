# SPDX-License-Identifier: MIT
"""Companion libraries of a GEX sample: table lookup, read-file picking, and fetch with per-library failure recording."""

import gzip
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "atlas_rerun"))

import companions as CP  # noqa: E402


def _fq(path, length, n=5):
    with gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w") as f:
        for i in range(n):
            f.write("@r%d\n%s\n+\n%s\n" % (i, "ACGT" * (length // 4) + "A" * (length % 4), "I" * length))


def test_load_table_keeps_companion_types_and_runs(tmp_path):
    t = tmp_path / "c.tsv"
    t.write_text(
        "gex_gsm\tlib_type\tlib_gsm\tgse\tstem\tlib_runs\tlib_reads\n"
        "GSM1\tadt\tGSM2\tGSE1\tx\tSRR1,SRR2\t1000\n"
        "GSM1\tatac\tGSM3\tGSE1\tx\tSRR9\t5\n"
        "GSM1\tvdj\tGSM4\tGSE1\tx\t\t7\n"
        "GSM5\thto\tGSM6\tGSE2\ty\tSRR3\t2000\n"
    )
    tab = CP.load_table(str(t))
    assert tab["GSM1"] == [{"lib_type": "adt", "lib_gsm": "GSM2", "runs": ["SRR1", "SRR2"], "reads": 1000}]
    assert tab["GSM5"][0]["lib_type"] == "hto" and "GSM3" not in str(tab)


def test_pick_reads_drops_index_and_orders_r1_by_length(tmp_path):
    for name, ln in (("a_1.fastq", 8), ("a_2.fastq", 28), ("a_3.fastq", 15)):
        _fq(str(tmp_path / name), ln)
    r1, r2 = CP.pick_reads([str(tmp_path / n) for n in ("a_1.fastq", "a_2.fastq", "a_3.fastq")])
    assert os.path.basename(r1) == "a_2.fastq" and os.path.basename(r2) == "a_3.fastq"
    with pytest.raises(ValueError):
        CP.pick_reads([str(tmp_path / "a_1.fastq")])


def test_fetch_all_records_failures_and_counts_bytes(tmp_path, monkeypatch):
    tab = {"G": [{"lib_type": "adt", "lib_gsm": "L1", "runs": ["SRRA"], "reads": 10},
                 {"lib_type": "crispr", "lib_gsm": "L2", "runs": ["SRRB"], "reads": 99}]}

    def fake_fetch_run(acc, scr, log=None, **kw):
        if acc == "SRRB":
            raise RuntimeError("all_sources_failed")
        p = os.path.join(scr, acc + ".sra")
        open(p, "wb").write(b"x")
        return p, {}

    def fake_lib(lib, dest, scr, fetch_run, env, log, threads=2, **kw):
        fetch_run(lib["runs"][0], scr, log=log)
        os.makedirs(dest, exist_ok=True)
        _fq(os.path.join(dest, lib["runs"][0] + "_R1.fastq.gz"), 28)
        return [lib["runs"][0] + "_R1.fastq.gz"]

    real = CP.fetch_library
    monkeypatch.setattr(CP, "fetch_library", lambda lib, *a, **k: fake_lib(lib, *a, **k) if lib["lib_type"] == "adt" else real(lib, *a, **k))
    inc, scr = tmp_path / "inc", tmp_path / "scr"
    inc.mkdir()
    scr.mkdir()
    recs = CP.fetch_all(tab, "G", str(inc), str(scr), fake_fetch_run, {}, lambda *a: None)
    assert [r["status"] for r in recs] == ["ok", "failed"]
    assert recs[0]["bytes"] > 0 and "all_sources_failed" in recs[1]["reason"]
    assert not (inc / "companions" / "crispr" / "L2").exists()
    assert CP.pairs_total(recs) == 10
    assert CP.fetch_all(tab, "NOPE", str(inc), str(scr), fake_fetch_run, {}, lambda *a: None) == []
    json.dumps(recs)
