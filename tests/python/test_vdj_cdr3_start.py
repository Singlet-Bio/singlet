"""The packaged human V(D)J reference starts the junction at the first of two adjacent Cys codons by default."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from singlet.vdj import reference as R  # noqa: E402


def _v(segs, name):
    return next(s for s in segs if s.name == name)


def test_default_is_first_cys():
    segs = R.load("human")
    for g in ("IGLV2-11", "IGLV2-23", "IGKV2-14"):
        s = _v(segs, g)
        assert s.seq[s.anchor : s.anchor + 3] in ("TGT", "TGC")
        assert s.seq[s.anchor + 3 : s.anchor + 6] in ("TGT", "TGC")  # the second Cys follows
    s = _v(segs, "IGLV2-14")  # a single Cys: the same anchor in both conventions
    assert s.seq[s.anchor : s.anchor + 3] in ("TGT", "TGC")


def test_second_cys_option():
    first, second = R.load("human"), R.load("human", cdr3_start="second")
    a, b = _v(first, "IGLV2-23"), _v(second, "IGLV2-23")
    assert b.anchor == a.anchor + 3
    assert _v(first, "IGLV2-14").anchor == _v(second, "IGLV2-14").anchor
    assert R.data_path("human").endswith("vdj_human_imgtcys.tsv.gz")
    assert R.data_path("human", "second").endswith("vdj_human.tsv.gz")
    assert R.data_path("mouse").endswith("vdj_mouse.tsv.gz")
