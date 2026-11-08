"""Cys104 from the read's Y-Y-C motif (ported from TRUST4's V motif) for reads with a J anchor and no V call."""
import numpy as np

from singlet.vdj import _kernels as K

ENC = {"A": 0, "C": 1, "G": 2, "T": 3}


def _codes(s):
    return np.array([ENC[c] for c in s], np.uint8)


def test_motif_places_cys_in_frame_upstream_of_phe():
    fr3 = "GACACGGCCGTGTATTACTGT"  # ... D T A V Y Y C
    cdr3 = "GCGAGAGATTACGGTATGGACGTC"  # A R D Y G M D V
    j = "TGGGGCCAAGGGACC"  # W G Q G T
    up = "ACGTCC" * 6
    s = up + fr3 + cdr3 + j
    codes = _codes(s)
    off = np.array([0, len(s)], np.int64)
    phe = len(up + fr3 + cdr3)
    cys = len(up + fr3) - 3
    out = K.cys_from_motif(codes, off, np.array([phe], np.int64), np.array([0], np.int64), 15, 120, 30)
    assert out[0] == cys
    # without enough read upstream of the motif, no anchor (a Cys further up could be missed)
    s2 = s[len(up) + 3:]
    out = K.cys_from_motif(_codes(s2), np.array([0, len(s2)], np.int64), np.array([phe - len(up) - 3], np.int64),
                           np.array([0], np.int64), 15, 120, 30)
    assert out[0] == -1
