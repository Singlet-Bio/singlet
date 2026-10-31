# SPDX-License-Identifier: MIT
"""Reads-per-UMI members in the bundle: the reads matrices stay in the run directory, the histogram is packed for the called cells only."""
import gzip

from singlet import bundle


def test_reads_matrices_are_not_packed():
    for rel in ("exon_reads.1pz", "intron_reads.1pz"):
        assert bundle._reads_member(rel, b"x", {"A"}) is None


def test_histogram_is_restricted_to_the_called_cells():
    text = "barcode\tn1\tn2\nA\t3\t1\nB\t2\t0\nC\t5\t5\n"
    out = bundle._reads_member("reads_per_umi_hist.tsv.gz", gzip.compress(text.encode()), {"A", "C"})
    assert gzip.decompress(out).decode() == "barcode\tn1\tn2\nA\t3\t1\nC\t5\t5\n"


def test_histogram_without_a_call_list_is_not_packed():
    assert bundle._reads_member("reads_per_umi_hist.tsv.gz", gzip.compress(b"barcode\tn1\n"), None) is None


def test_other_members_pass_through():
    assert bundle._reads_member("exon_counts.1pz", b"abc", {"A"}) == b"abc"


def test_reads_modalities_are_registered_after_the_counts():
    names = list(bundle.MODALITIES)
    i = names.index("intron_counts")
    assert names[i + 1 : i + 4] == ["exon_reads", "intron_reads", "reads_per_umi_hist"]
