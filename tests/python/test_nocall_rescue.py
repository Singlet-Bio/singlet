"""S14: plan-time exclusions, the generous probe, and the low-confidence accept of a run the detector could not call."""
import json
import os
import subprocess
import sys
import types

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "atlas_rerun"))

import fetch_worker as F  # noqa: E402
import plan_filter as PF  # noqa: E402


def runs_table(rows):
    d = pd.DataFrame(rows, columns=["gsm", "run_accession", "library_strategy", "library_layout", "instrument_model"])
    d["read_count"] = 1000
    d["base_count"] = [50000 if lay == "SINGLE" else 150000 for lay in d.library_layout]    # SINGLE: 50 nt spots
    d.loc[d.run_accession == "R10", "base_count"] = 150000
    return d


def test_plan_filter_reasons_and_keeps_multiome_and_paired_10x():
    runs = runs_table([
        ("GSM1", "R1", "ATAC-seq", "PAIRED", "NovaSeq 6000"),
        ("GSM2", "R2", "RNA-Seq", "SINGLE", "NextSeq 500"),
        ("GSM3", "R3", "RNA-Seq", "PAIRED", "AB 5500xl Genetic Analyzer"),
        ("GSM4", "R4", "RNA-Seq", "PAIRED", "NovaSeq 6000"),
        ("GSM5", "R5", "RNA-Seq", "PAIRED", "NovaSeq 6000"), ("GSM5", "R6", "ATAC-seq", "PAIRED", "NovaSeq 6000"),     # multiome: an RNA run stays
        ("GSM6", "R7", "RNA-Seq", "SINGLE", "NextSeq 500"), ("GSM6", "R8", "RNA-Seq", "PAIRED", "NextSeq 500"),         # one paired run stays
        ("GSM9", "R11", "RNA-Seq", "SINGLE", "NovaSeq 6000"),                                                           # long single read (set below): stays
        ("GSM7", "R9", "ChIP-Seq", "PAIRED", "NovaSeq 6000"),
        ("GSM8", "R10", "RNA-Seq", "PAIRED", "HiSeq 3000"),
    ])
    runs.loc[runs.run_accession == "R11", "base_count"] = 150000
    u = pd.DataFrame({"gsm_id": ["GSM%d" % i for i in range(1, 10)], "gse_id": ["GSE1"] * 9})
    kept, ex = PF.apply(u, runs, {"GSM8": "cdna_read_all_N"})
    assert dict(zip(ex.gsm, ex.reason)) == {"GSM1": "non_rna_assay:ATAC-seq", "GSM2": "single_end", "GSM3": "plate_or_longread_instrument:AB 5500xl Genetic Analyzer",
                                           "GSM7": "non_rna_assay:ChIP-Seq", "GSM8": "unrecoverable:cdna_read_all_N"}
    assert list(kept.gsm_id) == ["GSM4", "GSM5", "GSM6", "GSM9"]
    assert PF.summary(ex, 9) == {"excluded": 5, "of": 9, "by_reason": {"non_rna_assay": 2, "single_end": 1, "plate_or_longread_instrument": 1, "unrecoverable": 1}}


def test_plan_filter_without_the_columns_excludes_nothing():
    u = pd.DataFrame({"gsm_id": ["GSM1"], "gse_id": ["GSE1"]})
    kept, ex = PF.apply(u, pd.DataFrame({"gsm": ["GSM1"], "run_accession": ["R1"]}), {})
    assert len(kept) == 1 and len(ex) == 0
    # SINGLE layout without read / base counts cannot be judged: kept
    kept, ex = PF.apply(u, pd.DataFrame({"gsm": ["GSM1"], "run_accession": ["R1"], "library_layout": ["SINGLE"]}), {})
    assert len(kept) == 1 and len(ex) == 0


def test_unrecoverable_list_names_the_two_all_n_samples():
    assert set(PF.read_unrecoverable()) == {"GSM4037986", "GSM7908441"}


def test_catalogue_tag():
    assert F.catalogue_tag("10xv3") == "10x-3p-v3" and F.catalogue_tag("10xv2") == "10x-3p-v2" and F.catalogue_tag("10xv3_5prime") == "10x-5p-v3"
    assert F.catalogue_tag("citeseq") is None and F.catalogue_tag("10x_atac") is None and F.catalogue_tag("10x_suspect") is None and F.catalogue_tag(None) is None


def test_probe_is_retried_once_with_a_longer_timeout(monkeypatch, tmp_path):
    seen = []

    def run(cmd, **kw):
        seen.append(kw["timeout"])
        if len(seen) == 1:
            raise subprocess.TimeoutExpired(cmd, kw["timeout"])
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"[singlet-gate] PASS_10X x tag=10x-3p-v3 confidence=3 wl_match_rate=0.78 gate_s=70\n[singlet-download] Protocol: 10x-3p-v3 (confidence: 3)\n")
    monkeypatch.setattr(F.subprocess, "run", run)
    assert F.probe_run_protocol("/bin/singlet", "SRR1", str(tmp_path), {}) == ("10x-3p-v3", 3, 0.78)
    assert seen == [1800, 3600]
    seen.clear()

    def always(cmd, **kw):
        seen.append(1)
        raise subprocess.TimeoutExpired(cmd, 1)
    monkeypatch.setattr(F.subprocess, "run", always)
    assert F.probe_run_protocol("/bin/singlet", "SRR1", str(tmp_path), {}) == (None, None, None) and len(seen) == 2


def fake_probe(monkeypatch, screen, dl_rc=0, make_out=True):
    def run(cmd, **kw):
        if "screen-runs" in cmd:
            json.dump({"runs": [screen]}, open(cmd[cmd.index("--json") + 1], "w"))
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        assert cmd[cmd.index("--protocol") + 1] and cmd[cmd.index("--max-reads") + 1] == "200000"
        if make_out:
            open(cmd[cmd.index("-o") + 1], "w").write("x")
        return types.SimpleNamespace(returncode=dl_rc, stdout=b"", stderr=b"[1fq-encode] ERROR: low barcode whitelist match in both orientations\n")
    monkeypatch.setattr(F.subprocess, "run", run)


def test_low_conf_probe_accepts_a_real_10x_run_and_rejects_the_rest(monkeypatch, tmp_path):
    good = {"keep": True, "reason": "", "chemistry": "10x-3p-v2", "best_whitelist": "737K-august-2016.txt", "r1_len_mode": 26,
            "whitelists": {"737K-august-2016.txt": {"exact": 0.952, "one_mm": 1.0}, "3M-february-2018.txt": {"exact": 0.07, "one_mm": 0.06}}}
    fake_probe(monkeypatch, good)
    r = F.low_conf_probe("/bin/singlet", "x.sra", "10x-3p-v2", str(tmp_path), {}, "/wl")
    assert r["ok"] and r["wl_exact"] == 0.952 and r["chemistry"] == "10x-3p-v2"
    assert not os.path.exists(str(tmp_path / "lowconf.1fq")) and not os.path.exists(str(tmp_path / "lowconf.json"))
    # the catalogue says v3 but the reads are v2: the geometry does not fit
    r = F.low_conf_probe("/bin/singlet", "x.sra", "10x-3p-v3", str(tmp_path), {}, "/wl")
    assert not r["ok"] and "geometry" in r["reason"]
    # the 3p/5p flag is not part of the geometry
    fake_probe(monkeypatch, dict(good, chemistry="10x-5p-v2"))
    assert F.low_conf_probe("/bin/singlet", "x.sra", "10x-3p-v2", str(tmp_path), {}, "/wl")["ok"]
    # too low an exact hit rate
    fake_probe(monkeypatch, dict(good, whitelists={"737K-august-2016.txt": {"exact": 0.2, "one_mm": 0.4}}))
    r = F.low_conf_probe("/bin/singlet", "x.sra", "10x-3p-v2", str(tmp_path), {}, "/wl")
    assert not r["ok"] and "0.200" in r["reason"]
    # the encoder aborts (0% in both orientations), no output file
    fake_probe(monkeypatch, good, dl_rc=1, make_out=False)
    r = F.low_conf_probe("/bin/singlet", "x.sra", "10x-5p-v3", str(tmp_path), {}, "/wl")
    assert not r["ok"] and "forced decode failed" in r["reason"]
    # the screen drops it (feature-barcode-like R2, ...)
    fake_probe(monkeypatch, dict(good, keep=False, reason="feature-barcode-like R2"))
    assert "screen dropped" in F.low_conf_probe("/bin/singlet", "x.sra", "10x-3p-v2", str(tmp_path), {}, "/wl")["reason"]
