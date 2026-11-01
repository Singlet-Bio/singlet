# SPDX-License-Identifier: MIT
"""Per-sample private sidecar: what it selects, that it round-trips, deduplicates, and keeps public layers out."""

import gzip
import os
import sys
import tarfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "atlas_rerun"))

import private_sidecar as PS  # noqa: E402


def _w(root, rel, data):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)


@pytest.fixture()
def out(tmp_path):
    d = str(tmp_path / "out")
    sites = b"chr1\t100\tA\tG\n" * 5000
    _w(d, "private/snp_ad_features.tsv.gz", gzip.compress(sites))
    _w(d, "private/snp_dp_features.tsv.gz", gzip.compress(sites))        # identical content
    _w(d, "private/snp_ad.mtx.gz", gzip.compress(b"%%MatrixMarket\n1 1 1\n1 1 3\n"))
    _w(d, "private/donor/donor0.vcf", b"##fileformat=VCFv4.2\n" + b"chr1\t100\t.\tA\tG\n" * 1000)
    _w(d, "private/mt/donor0_mt_consensus.fa", b">chrM\nACGT\n")
    _w(d, "donor/donor0_coverage.tsv", b"site\tdp\n1\t5\n")
    _w(d, "donor_assignments.tsv", b"barcode\tdonor\nAAAC\td0\n")           # public
    _w(d, "ambient_profile.tsv", b"gene\tp\nA\t0.1\n")
    _w(d, "doublet_scores.tsv", b"barcode\tscore\nAAAC\t0.2\n")
    _w(d, "doublet_scores_engine.tsv", b"barcode\tscore\nAAAC\t0.1\n")
    _w(d, "sex_call.json", b'{"sex":"F"}')
    _w(d, "exon_counts.1pz", b"x" * 100)                                    # public
    _w(d, "cell_calls.tsv", b"barcode\tis_cell\nAAAC\ttrue\n")              # public
    _w(d, "star_Unmapped.out.mate1", b"@r\nACGT\n+\nIIII\n")                # reads: never
    _w(d, "nonhost_reads.fq", b"@r\nACGT\n+\nIIII\n")                       # reads: never
    return d


def test_select_takes_withheld_and_listed_layers_only(out):
    got = set(PS.select(out))
    assert "private/snp_ad.mtx.gz" in got and "donor/donor0_coverage.tsv" in got and "private/donor/donor0.vcf" in got
    assert {"ambient_profile.tsv", "doublet_scores.tsv", "doublet_scores_engine.tsv", "sex_call.json"} <= got
    for public in ("donor_assignments.tsv", "exon_counts.1pz", "cell_calls.tsv", "star_Unmapped.out.mate1", "nonhost_reads.fq"):
        assert public not in got


def test_roundtrip_dedup_and_size(out, tmp_path):
    dest = str(tmp_path / "side" / "GSE1" / "GSM1.private.tar")
    idx = PS.build(out, dest, "GSM1", "GSE1")
    assert PS.verify(dest) == []
    dup = [r for r, e in idx["files"].items() if "dup_of" in e]
    assert len(dup) == 1 and dup[0].startswith("private/snp_")                 # the two feature lists are stored once
    with tarfile.open(dest) as tf:
        assert not any(n.endswith(".gz") for n in tf.getnames())              # gz content is stored as xz
    back = str(tmp_path / "back")
    PS.extract(dest, back)
    for rel in PS.select(out):
        a, _ = PS._content(os.path.join(out, rel))
        b, _ = PS._content(os.path.join(back, rel))
        assert a == b, rel
    assert os.path.getsize(dest) < sum(e["bytes"] for e in idx["files"].values())


def test_verify_catches_corruption(out, tmp_path):
    dest = str(tmp_path / "x.tar")
    PS.build(out, dest, "GSM1")
    raw = bytearray(open(dest, "rb").read())
    # flip a byte inside the first stored blob (after its 512-byte tar header)
    raw[512 + 40] ^= 0xFF
    open(dest, "wb").write(bytes(raw))
    with pytest.raises(Exception):
        assert PS.verify(dest) == []


def test_collect_outputs_moves_stray_withheld_files_into_private(out, tmp_path):
    import types

    import stages

    _w(out, "mt/mt_events.1pz", b"e" * 50)           # restricted, but outside private/
    _w(out, "mt/mt_summary.tsv", b"a\tb\n")          # public
    ctx = types.SimpleNamespace(out_root=str(tmp_path / "o"), private_root=str(tmp_path / "p"))
    log = tmp_path / "log"
    log.write_text("x")
    final, priv, _sha, _man = stages.collect_outputs(ctx, "GSE1", "GSM1", out, str(tmp_path), str(log))
    pub = {os.path.relpath(os.path.join(d, f), final).replace(os.sep, "/") for d, _, fn in os.walk(final) for f in fn}
    assert "mt/mt_events.1pz" not in pub and "donor/donor0_coverage.tsv" not in pub
    assert "mt/mt_summary.tsv" in pub and "donor_assignments.tsv" in pub and "exon_counts.1pz" in pub
    assert os.path.exists(os.path.join(priv, "mt", "mt_events.1pz")) and os.path.exists(os.path.join(priv, "private", "snp_ad.mtx.gz")) is False
    assert os.path.exists(os.path.join(priv, "snp_ad.mtx.gz"))


def _mtx(rows, dims="5 4"):
    return ("%%MatrixMarket matrix coordinate integer general\n% c\n" + dims + " " + str(len(rows)) + "\n"
            + "".join("%d %d %d\n" % r for r in rows)).encode()


def test_mtx_coding_roundtrips_and_refuses_noncanonical():
    pytest.importorskip("pandas")
    rows = [(1, 1, 3), (4, 1, 1), (2, 2, 250), (5, 2, 70000), (3, 4, 9)]
    data = _mtx(rows)
    blob = PS.mtx_encode(data)
    assert blob is not None and PS.mtx_decode(blob) == data
    assert PS.mtx_encode(_mtx([(1, 2, 3), (1, 1, 3)])) is None           # columns decrease
    assert PS.mtx_encode(data.replace(b"\n2 2 250", b"\n2  2 250")) is None   # non-canonical spacing
    assert PS.mtx_encode(b"%%MatrixMarket matrix coordinate real general\n2 2 1\n1 1 0.5\n") is None


def test_big_matrix_uses_mtx_codec_and_verifies(out, tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    monkeypatch.setattr(PS, "MTX_MIN_BYTES", 100)
    rows = [(i % 5 + 1, i // 5 + 1, i % 7 + 1) for i in range(0, 400)]
    rows.sort(key=lambda r: (r[1], r[0]))
    _w(out, "private/snp_dp.mtx.gz", gzip.compress(_mtx(rows, "5 80")))
    dest = str(tmp_path / "s.tar")
    idx = PS.build(out, dest, "GSM1")
    assert idx["files"]["private/snp_dp.mtx.gz"]["codec"] == "mtx"
    assert PS.verify(dest) == []
    PS.extract(dest, str(tmp_path / "back"))
    a, _ = PS._content(os.path.join(out, "private/snp_dp.mtx.gz"))
    b, _ = PS._content(str(tmp_path / "back" / "private/snp_dp.mtx.gz"))
    assert a == b


REALISTIC = """ambient_contamination.json ambient_contamination.tsv ambient_corrected_counts.mtx.gz ambient_profile.tsv ancestry_call.json cell_calls.tsv
cell_cycle_scores.tsv cell_qc_metrics.tsv doublet_scores.tsv doublet_scores_engine.tsv doublet_stage.json donor/donor_assignments.tsv donor/donor0_coverage.tsv
exon_counts.1pz exon_counts.mtx.gz intron_counts.1pz mt/mt_events.1pz mt/mt_summary.tsv mt/donor0_mt_consensus.fa mt_heteroplasmy.1pz mt_heteroplasmy.mtx.gz
mt_heteroplasmy_barcodes.tsv.gz mt_reference.fa mt_variants.tsv nonhost/nonhost_per_cell.1pz nonhost/nonhost_per_cell.tsv nonhost/nonhost_summary.json
nonhost/nonhost_em_abundance.tsv pileup_stats.json polya_sites.tsv.gz private/snp_ad.mtx.gz private/snp_dp.mtx.gz private/snp_ad_features.tsv.gz
private/donor/donor0.vcf private/mt_allele_counts.mtx.gz private/ase_counts.tsv ase_counts.tsv saturation_curve.tsv sex_call.json sj_counts.1pz splice_events.tsv
splice_psi.1pz summary.json provenance.json star_SJ.out.tab star_Solo.out/Gene/raw/matrix.mtx""".split()


def test_every_policy_withheld_file_is_in_the_sidecar(tmp_path):
    """Whatever the release policy withholds from the public .s1 must be kept privately (the public copy is gone, and so are the reads)."""
    sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
    from singlet.s1 import policy

    d = str(tmp_path / "out")
    for rel in REALISTIC:
        _w(d, rel, b"x")
    _w(d, "possorted.bam", b"reads")
    got = set(PS.select(d))
    for rel in REALISTIC:
        if policy.omitted_layer(rel) is not None:
            assert rel in got, "%s is withheld by the policy (layer %s) but not in the sidecar" % (rel, policy.omitted_layer(rel))
    assert "possorted.bam" not in got                                            # reads are never kept
    assert {"mt_heteroplasmy.1pz", "mt/mt_summary.tsv"} <= got                  # the two files the first version missed
    for public in ("exon_counts.1pz", "cell_calls.tsv", "summary.json", "sj_counts.1pz", "nonhost/nonhost_summary.json"):
        assert public not in got, public
