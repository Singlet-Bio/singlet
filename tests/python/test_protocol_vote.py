"""One protocol per sample: the runs of a GSM are one library, so a 3'/5' disagreement between runs is detector noise (fetch_worker.vote_protocol)."""

import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "atlas_rerun"))

import fetch_worker as F  # noqa: E402


def test_parse_protocol_line():
    err = "x\n[singlet-download] Done: 1000000 reads, 3 blocks\n[singlet-download] Protocol: 10x-5p-v2 (confidence: 3)\n[singlet-download] Output: a\n"
    assert F.parse_protocol(err) == ("10x-5p-v2", 3)
    assert F.parse_protocol("nothing") == (None, None) and F.parse_protocol(None) == (None, None)


def test_family_ignores_only_the_3p_5p_flag():
    assert F.protocol_family("10x-3p-v2") == F.protocol_family("10x-5p-v2") == "10x-v2"
    assert F.protocol_family("10x-3p-v3") != F.protocol_family("10x-3p-v2")
    assert F.protocol_family("10x-arc-gex") == "10x-arc-gex" and F.protocol_family("celseq2") == "celseq2"


def test_agree_voted_incompatible_none():
    v = F.vote_protocol({"R1": "10x-3p-v3", "R2": "10x-3p-v3"}, {"R1": 5, "R2": 7})
    assert v["status"] == "agree" and v["winner"] == "10x-3p-v3"
    # GSM4987144: SRR13296299 called 3p-v2, the other three 5p-v2: 3'/5' only -> the majority by reads wins
    calls = {"SRR13296299": "10x-3p-v2", "SRR13296300": "10x-5p-v2", "SRR13296301": "10x-5p-v2", "SRR13296302": "10x-5p-v2"}
    w = {"SRR13296299": 77_297_667, "SRR13296300": 74_867_509, "SRR13296301": 79_590_901, "SRR13296302": 80_111_048}
    v = F.vote_protocol(calls, w)
    assert v["status"] == "voted" and v["winner"] == "10x-5p-v2" and v["family"] == "10x-v2" and v["weight"]["10x-5p-v2"] == 74_867_509 + 79_590_901 + 80_111_048
    # weights, not run counts: one deep 3' run outweighs two shallow 5' runs
    v = F.vote_protocol({"A": "10x-3p-v3", "B": "10x-5p-v3", "C": "10x-5p-v3"}, {"A": 100, "B": 10, "C": 10})
    assert v["winner"] == "10x-3p-v3"
    # a tie goes to the alphabetically first tag; unknown weights count as 1 read each
    assert F.vote_protocol({"A": "10x-5p-v2", "B": "10x-3p-v2"}, {})["winner"] == "10x-3p-v2"
    # different whitelist / geometry: different libraries (GEX + a feature library under one GSM): no winner
    v = F.vote_protocol({"A": "10x-3p-v3", "B": "10x-3p-v2"}, {"A": 1, "B": 1})
    assert v["status"] == "incompatible" and v["winner"] is None
    assert F.vote_protocol({"A": None, "B": None}, {})["status"] == "none"
    assert F.vote_protocol({"A": "10x-3p-v3", "B": None}, {"A": 1})["status"] == "agree"   # a run without a call does not vote


def test_decode_and_stage_commands_carry_the_forced_protocol(monkeypatch, tmp_path):
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(F.subprocess, "run", fake_run)
    F.decode_run("/bin/singlet", "x.sra", "o.1fq", "binned", 2, {}, str(tmp_path / "log"), protocol="10x-5p-v2")
    assert seen[-1][seen[-1].index("--protocol") + 1] == "10x-5p-v2"
    F.decode_run("/bin/singlet", "x.sra", "o.1fq", "binned", 2, {}, str(tmp_path / "log"))
    assert "--protocol" not in seen[-1]
    F.stage_run("/bin/singlet", "x.sra", str(tmp_path / "o"), "binned", 2, "/wl", {}, str(tmp_path), protocol="10x-5p-v2")
    assert seen[-1][seen[-1].index("--protocol") + 1] == "10x-5p-v2" and "--whitelist-dir" in seen[-1]
    F.stage_run("/bin/singlet", "x.sra", str(tmp_path / "o"), "binned", 2, "/wl", {}, str(tmp_path))
    assert "--protocol" not in seen[-1]


def test_head_screen_reports_the_call_and_passes_the_whitelist(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[1] == "download":
            open(cmd[cmd.index("-o") + 1], "w").write("x")
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"[singlet-download] Protocol: 10x-3p-v2 (confidence: 3)\n")
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr(F.subprocess, "run", fake_run)
    info = {}
    verdict, _ = F.head_screen("/bin/singlet", "SRR1", str(tmp_path), {}, info=info, whitelist_dir="/wl")
    assert verdict == "keep" and info == {"protocol": "10x-3p-v2", "confidence": 3}
    assert calls[0][calls[0].index("--whitelist-dir") + 1] == "/wl"
    F.head_screen("/bin/singlet", "SRR1", str(tmp_path), {})        # the old call still works
