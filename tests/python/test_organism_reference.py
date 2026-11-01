"""Organism -> reference: the worker never falls back to species auto-detection (scripts/atlas_rerun)."""

import json
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "atlas_rerun"))

import common as C  # noqa: E402
import drain_worker as D  # noqa: E402
import stagequeue as Q  # noqa: E402

REFS = {"Homo sapiens": "GRCh38-2024-A", "Mus musculus": "GRCm39-2024-A", "Homo sapiens; Mus musculus": "GRCh38-GRCm39-barnyard", "*": "auto"}


def test_exact_keys_and_order_variants():
    assert C.resolve_reference("Homo sapiens", REFS) == "GRCh38-2024-A"
    assert C.resolve_reference("Mus musculus", REFS) == "GRCm39-2024-A"
    assert C.resolve_reference("Homo sapiens; Mus musculus", REFS) == "GRCh38-GRCm39-barnyard"
    assert C.resolve_reference("Mus musculus; Homo sapiens", REFS) == "GRCh38-GRCm39-barnyard"      # 7,801 catalogue samples: the same set in the other order


def test_noise_tokens_are_ignored_but_other_species_are_not_guessed():
    assert C.resolve_reference("blank sample; Homo sapiens", REFS) == "GRCh38-2024-A"
    assert C.resolve_reference("unidentified; Homo sapiens", REFS) == "GRCh38-2024-A"
    assert C.resolve_reference("Homo; Homo sapiens", REFS) == "GRCh38-2024-A"
    assert C.resolve_reference("synthetic construct; Homo sapiens; Mus musculus", REFS) == "GRCh38-GRCm39-barnyard"
    for o in ("Macaca mulatta; Homo sapiens", "Homo sapiens; Canis lupus familiaris; Mus musculus", "Danio rerio", "", None, "blank sample"):
        assert C.resolve_reference(o, REFS) is None            # the '*' default ("auto") is never returned


def test_singlet_cmd_passes_the_genome_and_unmapped_organisms_raise():
    a = types.SimpleNamespace(threads=8, engine="gpu", extra=[])
    ccfg = {"ref_base": "/ref", "binary": "/bin/singlet"}
    cfg = {"references": REFS, "pipeline": {}}
    cmd = D.singlet_cmd(a, ccfg, cfg, "/claim", {"runs": ["SRR1"], "organism": "Mus musculus; Homo sapiens"}, "/out")
    assert cmd[cmd.index("--genome-dir") + 1] == os.path.join("/ref", "GRCh38-GRCm39-barnyard", "star_2.7.11b")
    assert cmd[cmd.index("--exons") + 1] == os.path.join("/ref", "GRCh38-GRCm39-barnyard", "genes", "genes.gtf")
    with pytest.raises(C.OrganismUnmapped):
        D.singlet_cmd(a, ccfg, cfg, "/claim", {"runs": ["SRR1"], "organism": "Macaca mulatta; Homo sapiens"}, "/out")
    with pytest.raises(C.OrganismUnmapped):
        D.singlet_cmd(a, ccfg, cfg, "/claim", {"runs": ["SRR1"]}, "/out")


def test_compute_phase_fails_an_unmapped_sample_for_good(tmp_path):
    stage = str(tmp_path / "stage")
    Q.init(stage)
    d = os.path.join(stage, "ready", "GSM9")
    os.makedirs(d)
    json.dump({"pairs": 1000, "runs": ["SRR9"], "organism": "Danio rerio", "bytes": 1}, open(os.path.join(d, "manifest.json"), "w"))
    gsm, claim, man = Q.claim_ready(stage)
    man["gsm"] = gsm
    a = types.SimpleNamespace(stage=stage, scratch=str(tmp_path / "scr"), threads=8, engine="gpu", extra=[])
    cfg = {"references": REFS, "pipeline": {}}
    ccfg = {"ref_base": "/ref", "binary": "/bin/singlet", "root": str(tmp_path / "root")}
    msg, job = D.compute_phase(a, cfg, ccfg, dict(os.environ), gsm, claim, man)
    assert job is None and "organism unmapped" in msg
    assert os.listdir(os.path.join(stage, "ready")) == [] and os.listdir(os.path.join(stage, "claimed")) == []
    (failed,) = os.listdir(os.path.join(stage, "failed"))
    assert "organism_unmapped" in json.load(open(os.path.join(stage, "failed", failed, "manifest.json")))["failed_reason"]
    ev = [json.load(open(os.path.join(ccfg["root"], "inbox", f))) for f in os.listdir(os.path.join(ccfg["root"], "inbox"))]
    assert [e["reason"] for e in ev if e["event"] == "failed"] == ["organism_unmapped"]
