"""scripts/atlas_rerun/genotype_keep.py: the campaign keeps every genotype output under a genotypes/ directory, 0700, with a manifest."""
import json
import os
import stat
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "atlas_rerun"))
import genotype_keep as GK  # noqa: E402


def _w(p, s="x"):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(s)


def test_keep_copies_genotype_files(tmp_path):
    out = tmp_path / "out"
    for rel in ("private/genotype/donors.vcf.gz", "private/genotype/cellsnp/cellSNP.tag.AD.mtx", "private/donor/donor0.vcf",
                "private/snp_ad.1pz", "private/snp_dp.1pz", "private/mt_allele_counts.1pz", "donor_assignments.tsv", "cell_calls.tsv",
                "exon_counts.1pz"):
        _w(str(out / rel), rel)
    base = tmp_path / "genotypes" / "beta"
    rec = GK.keep(str(out), "GSE1", "GSM2", str(base))
    dest = base / "GSE1" / "GSM2"
    assert rec["status"] == "ok" and rec["path"] == str(dest)
    man = json.loads((dest / "manifest.json").read_text())
    assert set(man["files"]) == {"genotype/donors.vcf.gz", "genotype/cellsnp/cellSNP.tag.AD.mtx", "donor/donor0.vcf", "snp_ad.1pz",
                                 "snp_dp.1pz", "donor_assignments.tsv", "cell_calls.tsv"}
    assert stat.S_IMODE(os.stat(dest).st_mode) & 0o077 == 0 or os.name == "nt"
    assert not (dest.parent / "GSM2.tmp").exists()


def test_keep_without_snp_data(tmp_path):
    out = tmp_path / "out"
    _w(str(out / "cell_calls.tsv"))
    rec = GK.keep(str(out), "GSE1", "GSM3", str(tmp_path / "genotypes"))
    assert rec["status"] == "no_snp_data"


def test_keep_refuses_other_places(tmp_path):
    with pytest.raises(RuntimeError):
        GK.keep(str(tmp_path), "GSE1", "GSM4", str(tmp_path / "public"))
