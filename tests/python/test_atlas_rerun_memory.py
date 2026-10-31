"""Memory-aware GPU workers (scripts/atlas_rerun): the fitted memory model, size classes, class choice for new workers, the sample filter."""

import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "atlas_rerun"))

import common as C  # noqa: E402
import drain_worker as D  # noqa: E402
import launcher as L  # noqa: E402
import stagequeue as Q  # noqa: E402

CLASSES = [96, 176, 248, 344, 720]


def test_model_matches_the_measured_runs():
    # (reads, measured largest-process RSS in GB): S8, 1 H100 each
    for reads, gb in ((11.3e6, 22.8), (13.2e6, 22.1), (66.6e6, 43.5), (263.4e6, 120.4)):
        assert abs(C.gpu_peak_gb(reads) - gb) / gb < 0.04


def test_job_memory_rounds_up_and_grows_with_reads():
    assert C.gpu_job_mem_gb(66.6e6) == 80 and C.gpu_job_mem_gb(263.4e6) == 168
    assert all(C.gpu_job_mem_gb(r) % 8 == 0 for r in (1e6, 7e7, 3e8))
    assert C.gpu_job_mem_gb(1e8) < C.gpu_job_mem_gb(2e8) < C.gpu_job_mem_gb(6e8)
    # the launcher's old fixed 128 GB request did not hold the median sample of the atlas plan (250 M pairs)
    assert C.gpu_job_mem_gb(250e6) > 128


def test_class_is_the_smallest_that_holds_the_sample():
    assert L.mem_class(66e6, CLASSES) == 96
    assert L.mem_class(250e6, CLASSES) == 176
    assert L.mem_class(425e6, CLASSES) == 248
    assert L.mem_class(625e6, CLASSES) == 344
    assert L.mem_class(1.29e9, CLASSES) == 720
    assert L.mem_class(9e9, CLASSES) == 720    # nothing holds it: the largest, and the worker will wait / fail loudly rather than be killed silently
    for r in (3e7, 1.1e8, 2.9e8, 4.3e8, 6.4e8, 1.3e9):
        assert C.gpu_job_mem_gb(r) <= L.mem_class(r, CLASSES) or L.mem_class(r, CLASSES) == 720


def test_every_class_with_work_gets_a_worker_before_any_gets_a_second():
    assert L.pick_classes(3, {96: 1e9, 176: 3e9, 720: 1e9}, {}) == [176, 720, 96]
    assert L.pick_classes(2, {96: 1e9, 176: 3e9}, {176: 1}) == [96, 176]
    assert L.pick_classes(4, {176: 1e9}, {}) == [176, 176, 176, 176]
    assert L.pick_classes(3, {}, {}) == []


def test_plan_starts_workers_by_class_and_respects_the_gpu_cap():
    p = dict(L.DEFAULTS)
    assert p["max_h100"] == 3
    bl = {"pairs": 5e10, "samples": 100, "todo": 0, "fetching": 0, "by_class": {96: 1e10, 176: 3e10, 720: 1e10}}
    have = {"h100": {"run": 1, "pend": 0, "ages": []}, "h100_by_class": {176: {"run": 1, "pend": 0, "ages": []}},
            "v100": {"run": 0, "pend": 0, "ages": []}, "cpu": {"run": 0, "pend": 0, "ages": []}, "fetch": 0}
    d = L.plan(p, bl, have)
    assert d["h100"] == 2 and len(d["h100_classes"]) == 2 and 176 not in d["h100_classes"][:1]   # the busy class already has a worker
    assert set(d["h100_classes"]) == {96, 720}


def test_sbatch_script_has_the_class_memory_and_name():
    a = types.SimpleNamespace(log_dir="/tmp", stage="/stage", config="/cfg.json")
    s = L.sbatch_script("drain-h100", a, dict(L.DEFAULTS), 176)
    assert "#SBATCH --mem=176G" in s and "--mem-gb 176" in s and "#SBATCH --job-name=s9-drain-h100-c176" in s
    whole = L.sbatch_script("drain-h100", a, dict(L.DEFAULTS), 720)   # the too-big class: a whole node, alone
    assert "#SBATCH --exclusive" in whole and "#SBATCH --mem=0" in whole and "#SBATCH --job-name=s9-drain-h100-c720" in whole
    assert "--exclusive" not in s
    old = L.sbatch_script("drain-v100", a, dict(L.DEFAULTS))
    assert "#SBATCH --mem=128G" in old and "--mem-gb 128" in old


def test_worker_leaves_a_sample_it_cannot_hold():
    big = {"pairs": 625_000_000}
    small = {"pairs": 66_000_000}
    assert not D.accept_sample(big, mem_gb=128) and D.accept_sample(small, mem_gb=128)
    assert D.accept_sample(big, mem_gb=344) and not D.accept_sample(big, mem_gb=248)
    assert D.accept_sample(big, mem_gb=0)                       # memory unknown: no filter
    assert D.accept_sample({"pairs": 1_300_000_000}, mem_gb=720) and D.accept_sample({"pairs": 3_000_000_000}, mem_gb=720)   # the whole-node worker takes any sample
    assert not D.accept_sample({"pairs": 1_300_000_000}, mem_gb=344)   # and no shared worker takes a p99 sample
    assert D.accept_sample(big, mem_gb=128, engine="cpu")       # the CPU engine's memory is a different model
    assert not D.accept_sample({"pairs": 1, "format": "1fg2"}, mem_gb=128)
    assert D.accept_sample({"pairs": 1, "format": "1fg2"}, mem_gb=128, accept_1fg2=True)
    assert not D.accept_sample(small, mem_gb=128, min_pairs=70_000_000)


def test_backlog_by_class(tmp_path):
    stage = str(tmp_path / "stage")
    for d in ("todo", "fetching", "ready", "claimed"):
        os.makedirs(os.path.join(stage, d))
    for gsm, pairs in (("GSM1", 60_000_000), ("GSM2", 70_000_000), ("GSM3", 250_000_000), ("GSM4", 1_300_000_000)):
        os.makedirs(os.path.join(stage, "ready", gsm))
        json.dump({"pairs": pairs, "bytes": 1}, open(os.path.join(stage, "ready", gsm, "manifest.json"), "w"))
    bl = Q.backlog(stage, classify=lambda m: L.mem_class(m["pairs"], CLASSES))
    assert bl["by_class"] == {96: 130_000_000, 176: 250_000_000, 720: 1_300_000_000} and bl["pairs"] == 1_680_000_000
    assert "by_class" not in Q.backlog(stage)
