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
            return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"[singlet-gate] PASS_10X x.sra tag=10x-3p-v2 confidence=3 wl_match_rate=0.9 gate_s=1\n[singlet-download] Protocol: 10x-3p-v2 (confidence: 3)\n")
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr(F.subprocess, "run", fake_run)
    info = {}
    verdict, _ = F.head_screen("/bin/singlet", "SRR1", str(tmp_path), {}, info=info, whitelist_dir="/wl")
    assert verdict == "keep" and info == {"protocol": "10x-3p-v2", "confidence": 3, "wl": 0.9}
    assert calls[0][calls[0].index("--whitelist-dir") + 1] == "/wl"
    F.head_screen("/bin/singlet", "SRR1", str(tmp_path), {})        # the old call still works


def test_catalogue_cross_check():
    assert F.catalogue_chemistry("10xv3") == ("3p", "v3") and F.catalogue_chemistry("10xv3_5prime") == ("5p", "v3")
    assert F.catalogue_chemistry("citeseq") == ("3p", None) and F.catalogue_chemistry("smartseq2") == (None, None) and F.catalogue_chemistry(None) == (None, None)
    assert F.check_against_catalogue("10x-5p-v3", "10xv3_5prime") == {"tag": "10x-5p-v3", "catalogue": "10xv3_5prime", "disagrees": False}
    assert F.check_against_catalogue("10x-3p-v3", "10xv3_5prime")["disagrees"] is True        # 3'/5' disagrees
    assert F.check_against_catalogue("10x-3p-v2", "10xv3")["disagrees"] is True               # version disagrees
    assert F.check_against_catalogue("10x-5p-v2", "10xv2")["disagrees"] is True               # catalogue says 3' v2, the call 5' v2
    assert F.check_against_catalogue("10x-3p-v3", "citeseq")["disagrees"] is False
    assert F.check_against_catalogue("celseq2", "10xv3") is None and F.check_against_catalogue("10x-3p-v3", "smartseq2") is None and F.check_against_catalogue(None, "10xv3") is None


def test_vote_runs_picks_the_largest_family_and_hands_the_rest_over():
    runs = ["R1", "R2", "R3", "R4"]
    calls = {"R1": ("10x-3p-v3", 3), "R2": ("10x-3p-v3", 3), "R3": ("10x-5p-v3", 2), "R4": ("10x-3p-v2", 3)}
    main, vote, others = F.vote_runs(runs, calls, [100, 100, 150, 30])
    assert main == ["R1", "R2", "R3"] and vote["status"] == "voted" and vote["winner"] == "10x-3p-v3"   # 200 reads of 3p-v3 vs 150 of 5p-v3
    assert vote["incompatible"] is True and vote["main_family"] == "10x-v3" and vote["other_runs"] == ["R4"]
    assert others == [{"lib_gsm": None, "lib_type": None, "status": "unclassified", "protocol_family": "10x-v2", "protocol_tags": {"R4": "10x-3p-v2"}, "runs": ["R4"], "reads": 30}]
    assert vote["calls"]["R4"] == ["10x-3p-v2", 3]
    # the GEX group is the one with the most reads, not the most runs
    main, vote, others = F.vote_runs(["A", "B", "C"], {"A": ("10x-3p-v3", 3), "B": ("10x-3p-v2", 3), "C": ("10x-3p-v2", 3)}, [900, 10, 10])
    assert main == ["A"] and others[0]["runs"] == ["B", "C"] and others[0]["reads"] == 20
    # one family: nothing handed over; a run without a call stays with the main group
    main, vote, others = F.vote_runs(["A", "B", "C"], {"A": ("10x-3p-v3", 3), "B": (None, None)}, [1, 1, 1])
    assert main == ["A", "B", "C"] and others == [] and vote["incompatible"] is False and vote["status"] == "agree"
    # no calls at all (e.g. every probe failed): everything stays, status none
    main, vote, others = F.vote_runs(["A", "B"], {}, [])
    assert main == ["A", "B"] and vote["status"] == "none" and others == []


def test_probe_run_protocol_without_a_head_screen(monkeypatch, tmp_path):
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        open(cmd[cmd.index("-o") + 1], "w").write("x")
        return types.SimpleNamespace(returncode=0, stdout=b"", stderr=b"[singlet-gate] PASS_10X x.sra tag=10x-5p-v2 confidence=3 wl_match_rate=0.88 gate_s=1\n[singlet-download] Protocol: 10x-5p-v2 (confidence: 3)\n")
    monkeypatch.setattr(F.subprocess, "run", fake_run)
    assert F.probe_run_protocol("/bin/singlet", "SRR1", str(tmp_path), {}, whitelist_dir="/wl") == ("10x-5p-v2", 3, 0.88)
    assert seen[0][seen[0].index("--max-reads") + 1] == "5000" and seen[0][seen[0].index("--whitelist-dir") + 1] == "/wl"
    assert not os.path.exists(str(tmp_path / "SRR1.probe.1fq"))      # the probe output is removed

    def timeout_run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)
    monkeypatch.setattr(F.subprocess, "run", timeout_run)
    assert F.probe_run_protocol("/bin/singlet", "SRR1", str(tmp_path), {}) == (None, None, None)


def test_plan_items_carry_run_reads_and_the_catalogue_protocol(tmp_path):
    import common as C
    import stagequeue as Q
    rows = [dict(priority=1, gsm="GSM1", gse="GSE1", organism="Homo sapiens", runs="SRR1,SRR2", run_reads="10,20", protocol_inferred="10xv3_5prime"),
            dict(priority=2, gsm="GSM2", gse="GSE1", organism="Homo sapiens", runs="SRR3", run_reads="5", protocol_inferred="")]
    plan = str(tmp_path / "plan.tsv.gz")
    C.write_plan(plan, rows, {"v": 1})
    items = Q.plan_items(plan)
    assert items[0] == {"gsm": "GSM1", "gse": "GSE1", "organism": "Homo sapiens", "runs": ["SRR1", "SRR2"], "run_reads": [10, 20], "protocol_inferred": "10xv3_5prime"}
    assert [i["gsm"] for i in Q.plan_items(plan, {"GSM2"})] == ["GSM2"] and len(Q.plan_items(plan, limit=1)) == 1
    out = str(tmp_path / "todo.jsonl")
    assert Q.main(["from-plan", plan, out, "--gsm", "GSM1"]) == 0
    import json
    assert json.loads(open(out).read())["protocol_inferred"] == "10xv3_5prime"


def test_low_confidence_calls_do_not_form_a_family():
    # census of the campaign's multi-run samples: the 'different protocol' runs are LOW calls (dropseq / marsseq2 / 10x-3p-v1 at whitelist rate ~0.05): a failed detection, not another library
    calls = {"A": ("10x-3p-v2", 3), "B": ("10x-3p-v2", 3), "C": ("10x-3p-v2", 3), "D": ("10x-3p-v1", 1)}
    main, vote, others = F.vote_runs(["A", "B", "C", "D"], calls, [10, 10, 10, 3])
    assert main == ["A", "B", "C", "D"] and others == [] and vote["incompatible"] is False
    assert vote["status"] == "agree" and vote["winner"] == "10x-3p-v2" and vote["low_confidence_runs"] == ["D"]
    # a confident call in another family is a different library
    calls["D"] = ("10x-3p-v3", 3)
    main, vote, others = F.vote_runs(["A", "B", "C", "D"], calls, [10, 10, 10, 3])
    assert main == ["A", "B", "C"] and vote["incompatible"] is True and others[0]["runs"] == ["D"] and vote["low_confidence_runs"] == []
    # nothing confident at all: no decision, every run stays (as before the vote existed)
    main, vote, others = F.vote_runs(["A", "B"], {"A": ("dropseq", 1), "B": ("celseq2", 1)}, [5, 5])
    assert main == ["A", "B"] and vote["status"] == "none" and others == [] and vote["low_confidence_runs"] == []


def test_parse_wl_rate_from_the_gate_line():
    assert F.parse_wl_rate("x\n[singlet-gate] PASS_10X /a/b.sra tag=10x-3p-v2 confidence=3 wl_match_rate=0.9012 gate_s=0.33\n") == 0.9012
    assert F.parse_wl_rate("[singlet-gate] UNCERTAIN /a.sra tag=dropseq confidence=1 wl_match_rate=0 r1=20bp r2=50bp gate_s=0.1 \u2014 passing through\n") == 0.0
    assert F.parse_wl_rate("[singlet-gate] REJECT a tag=splitseq confidence=1 wl_match_rate=0.001 r1=1bp") == 0.001
    assert F.parse_wl_rate("nothing") is None and F.parse_wl_rate(None) is None


def test_low_runs_follow_only_with_a_comparable_whitelist_rate():
    runs = ["A", "B", "C", "D"]
    base = {"A": ("10x-3p-v2", 3, 0.90), "B": ("10x-3p-v2", 3, 0.92), "C": ("10x-3p-v2", 3, 0.91)}
    # D: LOW call but a whitelist rate of 0.60 (>= half of ~0.91): a lane of the same library
    main, vote, others = F.vote_runs(runs, dict(base, D=("10x-3p-v1", 1, 0.60)), [10, 10, 10, 5])
    assert main == runs and vote["low_confidence_runs"] == ["D"] and vote["excluded_runs"] == [] and others == []
    # D at 0.05: another file layout or library: excluded, not pooled (5 of 35 reads = 14%: also offered to the companion classifier)
    main, vote, others = F.vote_runs(runs, dict(base, D=("10x-3p-v1", 1, 0.05)), [10, 10, 10, 5])
    assert main == ["A", "B", "C"] and vote["low_confidence_runs"] == []
    (ex,) = vote["excluded_runs"]
    assert ex["run"] == "D" and ex["wl_match_rate"] == 0.05 and ex["tag"] == "10x-3p-v1" and ex["reads"] == 5 and ex["main_wl_match_rate"] == 0.91
    assert vote["incompatible"] is True and others[0]["runs"] == ["D"] and others[0]["protocol_family"] is None and others[0]["wl_match_rate"] == 0.05
    # a small excluded run (1 of 31 reads = 3%) is excluded but not a companion candidate
    main, vote, others = F.vote_runs(runs, dict(base, D=("dropseq", 1, 0.0)), [10, 10, 10, 1])
    assert main == ["A", "B", "C"] and len(vote["excluded_runs"]) == 1 and others == [] and vote["incompatible"] is False
    # a run with no call at all (probe failed) or no rate follows
    main, vote, others = F.vote_runs(runs, dict(base, D=(None, None, None)), [10, 10, 10, 5])
    assert main == runs and vote["excluded_runs"] == []
    # nothing confident: no decision, nothing excluded
    main, vote, others = F.vote_runs(["A", "B"], {"A": ("dropseq", 1, 0.0), "B": ("celseq2", 1, 0.0)}, [5, 5])
    assert main == ["A", "B"] and vote["excluded_runs"] == [] and vote["status"] == "none"


def test_a_guessed_structure_never_outvotes_a_known_chemistry():
    # GSM9015940 (campaign census): a 205 M-read feature library called 'agnostic-bc6+umi7' (whitelist 0.40) next to a 123 M-read 10x v2 GEX run (whitelist 0.97).
    calls = {"G": ("10x-3p-v2", 3, 0.9704), "F": ("agnostic-bc6+umi7", 3, 0.3978)}
    main, vote, others = F.vote_runs(["G", "F"], calls, [123506128, 204953136])
    assert main == ["G"] and vote["winner"] == "10x-3p-v2" and vote["main_family"] == "10x-v2"
    assert others and others[0]["runs"] == ["F"] and others[0]["protocol_family"] == "agnostic-bc6+umi7"
    # among known chemistries the reads that hit the whitelist count: 100 M at 0.95 beat 150 M at 0.30
    main, vote, others = F.vote_runs(["A", "B"], {"A": ("10x-3p-v3", 3, 0.95), "B": ("10x-3p-v2", 3, 0.30)}, [100, 150])
    assert main == ["A"] and others[0]["runs"] == ["B"]
    # only guessed structures: the largest still wins
    main, vote, others = F.vote_runs(["A", "B"], {"A": ("agnostic-bc6+umi7", 3, 0.4), "B": ("agnostic-bc8+umi8", 3, 0.4)}, [10, 20])
    assert main == ["B"]
