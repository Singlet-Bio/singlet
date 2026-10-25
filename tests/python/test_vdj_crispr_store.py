# SPDX-License-Identifier: MIT
"""V(D)J and CRISPR members: round trip through a sidecar .s2 object, QC verdicts, clonotypes."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from singlet.crispr import guides as G
from singlet.crispr import qc as CQ
from singlet.crispr import store as CS
from singlet.vdj import clonotype as CL
from singlet.vdj import qc as VQ
from singlet.vdj import store as VS

pytest.importorskip("pyarrow")


def contigs():
    rows = []
    for i in range(6):
        bc = "ACGT"[i % 4] * 8 + "ACGT"[i // 4] * 8 + "-1"
        rows.append(dict(barcode=bc, contig_id=bc + "_contig_1", chain="TRB", v_gene="TRBV20-1", d_gene="", j_gene="TRBJ2-7",
                         c_gene="TRBC2", cdr3="CASSF", cdr3_nt="TGTGCCAGCAGCTTT" if i < 4 else "TGTGCCAGCAGCTTC",
                         productive=True, full_length=True, umis=5, junction_umis=3, reads=40, high_confidence=True, kept=True))
        rows.append(dict(rows[-1], contig_id=bc + "_contig_2", chain="TRA", v_gene="TRAV1-2", j_gene="TRAJ33", c_gene="TRAC",
                         cdr3="CAVF", cdr3_nt="TGTGCTGTGTTT"))
    return pd.DataFrame(rows)


def test_clonotypes_and_qc():
    c = contigs()
    cells = CL.assign_clonotypes(CL.cell_table(c))
    assert cells["paired"].all()
    assert cells["clonotype_size"].max() == 4
    st = {"read_pairs": 100, "mates_with_segment": 90, "mates_valid_barcode": 85, "orientation": {"r1_use": False}}
    q = VQ.library_qc(c, cells, st, gex_called=list(c.barcode.unique()), gex_columns=list(c.barcode.unique()) + ["N" * 16 + "-1"])
    assert q["verdict"] == "pass", q
    q2 = VQ.library_qc(c, cells, dict(st, mates_with_segment=5, mates_valid_barcode=5))
    assert "frac_on_target" in q2["failed"]


def test_vdj_members_roundtrip():
    c = contigs()
    gex = list(c.barcode.unique())[:4] + ["TTTTTTTTTTTTTTTT-1"]
    m = VS.build_members(c, gex, {"verdict": "pass"}, {"libraries": ["GSMx"]}, {"name": "vdj_human"})
    buf = VS.build_sidecar("GSM1", m)
    back = VS.read_members(buf)
    assert len(back["contigs"]) == len(c)
    assert (back["contigs"]["col"] >= 0).sum() == 8
    assert back["summary"]["schema"] == VS.SCHEMA


def test_crispr_members_roundtrip():
    lib = G.from_table([{"id": f"g{i}", "sequence": "ACGT" * 5 if i == 0 else "TTGCA" * 4, "target_gene": "NT" if i else "MYC"}
                        for i in range(2)], "t")
    M = sparse.csc_matrix(np.array([[3, 0, 1], [0, 7, 0]], dtype=np.uint32))
    m = CS.build_members(M, lib, ["AAAA-1", "CCCC-1", "GGGG-1"], np.array([2, 5]), "20 nt before GTTTAAGAGCTA", {}, {"verdict": "pass"})
    back = CS.read_members(CS.build_sidecar("GSM2", m))
    X, names = back["crispr"]
    assert names == ["g0", "g1"] and (X.toarray() == M.toarray()).all()
    assert back["guides"]["guides"][1]["kind"] == "non-targeting"
