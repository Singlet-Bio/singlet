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
    _w(d, "sex_call.json", b'{"sex":"F"}')
    _w(d, "exon_counts.1pz", b"x" * 100)                                    # public
    _w(d, "cell_calls.tsv", b"barcode\tis_cell\nAAAC\ttrue\n")              # public
    _w(d, "star_Unmapped.out.mate1", b"@r\nACGT\n+\nIIII\n")                # reads: never
    _w(d, "nonhost_reads.fq", b"@r\nACGT\n+\nIIII\n")                       # reads: never
    return d


def test_select_takes_withheld_and_listed_layers_only(out):
    got = set(PS.select(out))
    assert "private/snp_ad.mtx.gz" in got and "donor/donor0_coverage.tsv" in got and "private/donor/donor0.vcf" in got
    assert {"ambient_profile.tsv", "doublet_scores.tsv", "sex_call.json"} <= got
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
