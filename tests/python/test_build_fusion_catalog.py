# SPDX-License-Identifier: MIT
"""scripts/build_fusion_catalog.py: open sources only (COSMIC and gene-family pairs dropped), ordered CTAT pairs, unordered Mitelman pairs."""
import gzip
import importlib.util
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("build_fusion_catalog", ROOT / "scripts" / "build_fusion_catalog.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

DAT = (
    '#945\t.\t{"HGNC":"x"}\n'
    '+NUP214--ABL1\t["Mitelman"]\t{"Mitelman":"ALL=1"}\n'
    'BCR--ABL1\t["ChimerKB","Cosmic","Mitelman"]\t{}\n'
    'GENEX--GENEY\t["Cosmic"]\t{}\n'
    'FAM1--FAM2\t["HGNC_GENEFAM"]\t{}\n'
    'PAR1--PAR2\t["DGD_PARALOGS"]\t{}\n'
    'SAME--SAME\t["ChimerSeq"]\t{}\n'
)
MCGENE = "MolClin\tRefNo\tInvNo\tOrderNo\tPrefix\tGene\tSuffix\nM\t1\t1\t1\t\tABL1::BCR\t\nM\t1\t2\t1\t\tA::B::C\t\nC\t2\t1\t1\t\tX::Y\t+\nM\t3\t1\t1\t\tBCR\t+\n"


def test_catalog(tmp_path):
    dat = tmp_path / "fusion_lib.dat.gz"
    dat.write_bytes(gzip.compress(DAT.encode()))
    mz = tmp_path / "mitelman_db.zip"
    with zipfile.ZipFile(mz, "w") as z:
        z.writestr("MCGENE.TXT.DATA", MCGENE)
    rows = B.build(str(dat), str(mz))
    assert rows[("BCR", "ABL1")] == [{"ChimerKB", "Mitelman"}, 1]  # Cosmic dropped, Mitelman merged with the reversed pair
    assert ("ABL1", "BCR") not in rows
    assert rows[("PAR1", "PAR2")][0] == {"DGD_PARALOGS"}
    assert ("GENEX", "GENEY") not in rows and ("FAM1", "FAM2") not in rows and ("SAME", "SAME") not in rows
    assert rows[("A", "B")] == [{"Mitelman"}, 0] and rows[("B", "C")] == [{"Mitelman"}, 0]
    assert ("X", "Y") not in rows  # clinical association row
    assert rows[("+NUP214", "ABL1")][0] == {"Mitelman"}
    out = tmp_path / "cat.tsv.gz"
    B.main(["--ctat", str(dat), "--mitelman", str(mz), "--out", str(out)])
    text = gzip.decompress(out.read_bytes()).decode()
    lines = [l for l in text.splitlines() if not l.startswith("#")]
    assert not any("cosmic" in l.lower() for l in lines)
    assert lines[0] == "gene_a\tgene_b\tsources\tordered" and "BCR\tABL1\tChimerKB,Mitelman\t1" in lines
    out2 = tmp_path / "cat2.tsv.gz"
    B.main(["--ctat", str(dat), "--mitelman", str(mz), "--out", str(out2)])
    assert out.read_bytes() == out2.read_bytes()
