"""The engine's informational species check line (singlet logs it when SINGLET_SPECIES_FILTERS is set) and how the workers read it."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "atlas_rerun"))

import common as C  # noqa: E402


def test_species_check_line_is_parsed(tmp_path):
    log = tmp_path / "singlet.log"
    log.write_text("[singlet] x\n[species-check] detected=Homo sapiens genome=GRCh38 confidence=0.8966 method=bloom reference=GRCh38-2024-A AGREE\n[singlet] y\n")
    sc = C.species_check(str(log))
    assert sc["verdict"] == "AGREE" and "GRCh38-2024-A" in sc["line"]
    log.write_text("[species-check] detected=Mus musculus genome=GRCm39 confidence=0.9 method=bloom reference=GRCh38-2024-A DISAGREE\n")
    assert C.species_check(str(log))["verdict"] == "DISAGREE"
    log.write_text("nothing here\n")
    assert C.species_check(str(log)) is None and C.species_check(str(tmp_path / "missing.log")) is None
