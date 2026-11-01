"""Multi-sample GPU drain worker (scripts/atlas_rerun/multi_drain.py): admission arithmetic with synthetic sizes, head-of-line protection,
cgroup limit, launcher classes, and an end-to-end drain with a fake engine (ok / failing / OOM-killed samples)."""

import json
import os
import stat
import subprocess
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "..", "scripts", "atlas_rerun")
sys.path.insert(0, SCRIPTS)

import common as C  # noqa: E402
import launcher as L  # noqa: E402
import multi_drain as M  # noqa: E402
import stagequeue as Q  # noqa: E402

M_ = 1_000_000


def test_models_have_both_routes_and_the_fq_default_is_the_fitted_line():
    assert abs(C.gpu_peak_gb(263.4e6) - 120.2) < 0.5 and C.gpu_peak_gb(263.4e6) == C.gpu_peak_gb(263.4e6, "fq")
    assert C.gpu_tmp_gb(100e6, "fq") > C.gpu_tmp_gb(100e6, "1fg2") >= 0    # the .1fg2 route writes no temp FASTQ
    assert abs((C.gpu_tmp_gb(1e9, "fq") - C.gpu_tmp_gb(0, "fq")) - 335.0) < 1   # 320 B/read of temp FASTQ + 15 B/read of outputs


def test_memory_model_against_the_measured_runs():
    # S8, 263.4 M reads, 1 H100: whole-job cgroup peak 100.6 GB on the .1fg2 route (anon 85.0 + page cache 15.4); 180.9 GB on the FASTQ route, which includes
    # the page cache of the temp FASTQ (reclaimable), so the model sits between the child's 120.3 GB and that figure
    assert abs(C.gpu_cgroup_gb(263.4e6, "1fg2") - 100.6) / 100.6 < 0.05
    assert 120.3 < C.gpu_cgroup_gb(263.4e6, "fq") < 180.9
    assert C.gpu_private_gb(100e6, "fq", shared=True) == C.gpu_cgroup_gb(100e6, "fq") - C.GENOME_SHARED_GB
    assert C.gpu_private_gb(100e6, "fq", shared=False) == C.gpu_cgroup_gb(100e6, "fq")


def test_budget_charges_the_shared_genome_once_and_stops_at_85_percent():
    b = M.Budget(370, 373, slots=8, shared_gb=C.GENOME_SHARED_GB)
    assert abs(b.cap_mem - 314.5) < 1e-9
    mem, tmp = M.estimate({"pairs": 150 * M_}, "fq", shared=True)      # 150 M pairs: private 3.8 + 66 = 69.8 GB each
    n = 0
    while b.fits(mem, tmp):
        b.add("G%d" % n, mem, tmp)
        n += 1
    assert n == 4 and abs(b.mem_used - (4 * mem + C.GENOME_SHARED_GB)) < 1e-9 and b.mem_used <= b.cap_mem     # a fifth would reach 367 GB
    plain = M.Budget(370, 373, slots=8)                                 # genome not shared: every process is charged in full
    mem0, _ = M.estimate({"pairs": 150 * M_}, "fq")
    assert mem0 > mem and plain.cap_mem - plain.mem_used >= mem0
    small = M.Budget(370, 373, slots=8, shared_gb=C.GENOME_SHARED_GB)
    for i in range(20):
        if small.fits(*M.estimate({"pairs": 66.6 * M_}, "fq", True)):
            small.add("S%d" % i, *M.estimate({"pairs": 66.6 * M_}, "fq", True))
    assert len(small.run) == 8                                          # pbmc1k-sized samples: the slot cap, not memory, stops them (8 x 33 + 17.7 = 282 GB)


def test_scratch_budget_limits_the_temp_fastq_route():
    b = M.Budget(370, 373, slots=8)
    mem, tmp = M.estimate({"pairs": 400 * M_}, "fq")           # 400 M reads -> 134 GB of temp FASTQ and outputs + 2
    assert b.fits(mem, tmp)
    b.add("A", mem, tmp)
    assert not b.fits(10, 230)                                  # 136 + 230 GB of scratch does not fit in 373 - 20
    assert b.fits(10, 100)
    assert M.estimate({"pairs": 600 * M_}, "1fg2")[1] == 2.0 + 0.015 * 600


def test_too_big_samples_belong_to_the_whole_node_class():
    b = M.Budget(370, 373, slots=8)
    assert b.ever_fits(M.estimate({"pairs": 640 * M_}, "fq")[0]) and not b.ever_fits(M.estimate({"pairs": 700 * M_}, "fq")[0])
    assert not b.ever_fits(M.estimate({"pairs": 1_290 * M_}, "fq")[0])           # p99
    assert b.ever_fits(M.estimate({"pairs": 1_000 * M_}, "1fg2")[0]) and not b.ever_fits(M.estimate({"pairs": 1_290 * M_}, "1fg2")[0])


def test_oom_floor_and_solo_retry():
    b = M.Budget(370, 373, slots=8)
    man = {"pairs": 250 * M_, "mem_floor_gb": 180.0, "solo": True}
    mem, tmp = M.estimate(man, "fq")
    assert mem == 180.0
    b.add("X", 10, 1)
    assert not b.fits(mem, tmp, solo=True)         # a solo retry waits for an empty worker
    b.drop("X")
    assert b.fits(mem, tmp, solo=True)


def adm(b, **kw):
    return M.Admitter(b, "fq", 373, {}, **kw)


def test_admitter_filters_static_and_dynamic():
    b = M.Budget(370, 373, slots=8)
    assert adm(b)({"pairs": 50 * M_}) and not adm(b)({"pairs": 900 * M_}) and not adm(b)({"pairs": 5, "format": "1fg2"})
    assert adm(b, accept_1fg2=True)({"pairs": 5, "format": "1fg2"})
    assert not adm(b, min_pairs=60 * M_)({"pairs": 50 * M_})


def test_head_of_line_protection(monkeypatch):
    b = M.Budget(370, 373, slots=8)
    b.add("R1", 200, 10)                              # 200 of 314 GB used: a 250 M sample (173 GB) does not fit, a small one (34 GB) does
    hol, now = {}, [1000.0]
    monkeypatch.setattr(M.time, "time", lambda: now[0])
    a = M.Admitter(b, "fq", 373, hol)
    assert not a({"pairs": 250 * M_}) and a({"pairs": 20 * M_}) and a.blocked      # overtaking is allowed at first
    now[0] += M.HOL_WAIT_S + 1
    a = M.Admitter(b, "fq", 373, hol)
    assert not a({"pairs": 250 * M_}) and not a({"pairs": 20 * M_})                # then the head blocks everything newer
    b.drop("R1")
    a = M.Admitter(b, "fq", 373, hol)
    assert a({"pairs": 250 * M_}) and "since" not in hol and not a.blocked


def test_cgroup_limit_v2_and_parent(tmp_path):
    root = tmp_path / "cg"
    (root / "job" / "step").mkdir(parents=True)
    (root / "memory.max").write_text("max\n")
    (root / "job" / "memory.max").write_text(str(370 * 10**9) + "\n")
    (root / "job" / "step" / "memory.max").write_text("max\n")
    pc = tmp_path / "proc_cgroup"
    pc.write_text("0::/job/step\n")
    assert abs(M.cgroup_limit_gb(str(root), str(pc)) - 370) < 1e-6
    (root / "job" / "memory.max").write_text("max\n")
    assert M.cgroup_limit_gb(str(root), str(pc)) is None


def test_launcher_shared_class_and_script():
    p = dict(L.DEFAULTS)
    assert p["multi_sample"] and p["max_h100"] == 3
    cls = p["multi_classes_gb"]
    assert L.mem_class(250e6, cls, 0.85) == 370 and L.mem_class(640e6, cls, 0.85) == 370
    assert L.mem_class(700e6, cls, 0.85) == 720 and L.mem_class(1.29e9, cls, 0.85) == 720
    a = types.SimpleNamespace(log_dir="/tmp", stage="/stage", config="/cfg.json")
    s = L.sbatch_script("drain-h100", a, p, 370)
    assert "multi_drain.py" in s and "#SBATCH --mem=370G" in s and "#SBATCH --gres=gpu:nvidia_h100_nvl:1" in s and "--concurrency 8" in s
    assert "--threads 12" in s and "--scratch /tmp" in s and "--exclusive" not in s and "cpus-per-task=28" in s
    whole = L.sbatch_script("drain-h100", a, p, 720)
    assert "drain_worker.py" in whole and "--exclusive" in whole and "--mem=0" in whole
    bl = {"pairs": 6e10, "samples": 100, "todo": 0, "fetching": 0, "by_class": {370: 5e10, 720: 1e10}}
    have = {"h100": {"run": 0, "pend": 0, "ages": []}, "h100_by_class": {}, "v100": {"run": 0, "pend": 0, "ages": []},
            "cpu": {"run": 0, "pend": 0, "ages": []}, "fetch": 0}
    d = L.plan(p, bl, have)
    assert d["h100"] == 3 and sorted(d["h100_classes"]) == [370, 370, 720]       # 6e10 pairs / 1.0 M/s = 16.7 h of work: all three GPUs


def test_workers_fail_fast_without_the_whitelists(tmp_path, monkeypatch):
    # the engine resolves <binary dir>/../whitelists; without it a run silently takes the sorted-BAM pileup (S8)
    monkeypatch.delenv("S9_NO_WL_CHECK", raising=False)
    (tmp_path / "bin").mkdir()
    binary = tmp_path / "bin" / "singlet"
    binary.write_text("x")
    with pytest.raises(SystemExit) as e:
        C.require_whitelists(str(binary))
    assert "3M-february-2018.txt" in str(e.value) and "sorted-BAM" in str(e.value)
    (tmp_path / "whitelists").mkdir()
    (tmp_path / "whitelists" / "3M-february-2018.txt").write_text("A\n")
    with pytest.raises(SystemExit):
        C.require_whitelists(str(binary))            # one list is not enough
    (tmp_path / "whitelists" / "737K-august-2016.txt").write_text("A\n")
    C.require_whitelists(str(binary))                # complete: fine
    monkeypatch.setenv("S9_NO_WL_CHECK", "1")
    C.require_whitelists(str(tmp_path / "elsewhere" / "singlet"))   # bypass for tests with a fake engine


def test_pileup_threads_reader(tmp_path):
    assert M.pileup_threads(str(tmp_path)) is None
    (tmp_path / "pileup_stats.json").write_text(json.dumps({"pileup_threads": 1}))
    assert M.pileup_threads(str(tmp_path)) == 1


# ── end to end with a fake engine ───────────────────────────────────────────────────────────────────────────────────

FAKE = """#!/usr/bin/env python3
import sys, os, time
a = sys.argv[1:]
src = open(a[0]).read().strip()
out = a[a.index("--out-prefix") + 1]
os.makedirs(out, exist_ok=True)
time.sleep(1.0)
if src == "oom":
    os.kill(os.getpid(), 9)
if src == "fail":
    sys.exit(3)
open(os.path.join(out, "counts.txt"), "w").write("counts of " + src)
"""


@pytest.mark.skipif(not os.path.exists("/usr/bin/time"), reason="needs GNU time (Linux)")
def test_drain_end_to_end(tmp_path):
    stage = str(tmp_path / "stage")
    Q.init(stage)
    root = str(tmp_path / "root")
    fake = tmp_path / "fake_singlet.py"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    cfg = tmp_path / "config.json"
    json.dump({"binary": {"path": str(fake)}, "clusters": {"clipper": {"root": root, "ref_base": str(tmp_path / "ref")}}}, open(cfg, "w"))
    samples = {"GSM1": ("ok", 20), "GSM2": ("ok", 250), "GSM3": ("fail", 30), "GSM4": ("oom", 40), "GSM5": ("ok", 900)}
    for gsm, (body, mp) in samples.items():
        d = os.path.join(stage, "ready", gsm)
        os.makedirs(d)
        open(os.path.join(d, "sample.1fq"), "w").write(body)
        json.dump({"pairs": mp * M_, "bytes": 10, "runs": ["sample"], "gse": "GSE1"}, open(os.path.join(d, "manifest.json"), "w"))
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "multi_drain.py"), "--stage", stage, "--config", str(cfg), "--concurrency", "3",
                        "--scratch", str(tmp_path / "scr"), "--mem-gb", "370", "--idle-exit-min", "0.05", "--publish", "none"],
                       capture_output=True, text=True, timeout=170, env=dict(os.environ, S9_NO_WL_CHECK="1"))
    assert r.returncode == 0, r.stdout + r.stderr
    log = r.stdout + r.stderr
    assert os.listdir(os.path.join(stage, "ready")) == ["GSM5"], log        # 900 M pairs: left for the whole-node class, not lost, not failed
    assert sorted(os.listdir(os.path.join(stage, "failed"))) == ["GSM3.2", "GSM4.2"]   # requeued once, then failed
    assert os.listdir(os.path.join(stage, "claimed")) == []
    for g in ("GSM3", "GSM4"):
        m = json.load(open(os.path.join(stage, "failed", g + ".2", "manifest.json")))
        assert m["failed_reason"] in ("rc_3", "oom") and m["attempts"] == 2
    ev = [json.load(open(os.path.join(root, "inbox", f))) for f in sorted(os.listdir(os.path.join(root, "inbox")))]
    assert sorted(e["gsm"] for e in ev if e["event"] == "done") == ["GSM1", "GSM2"]
    assert [e for e in ev if e["gsm"] == "GSM4" and e["event"] == "requeued"][0]["reason"] == "oom"
    assert {e["gsm"] for e in ev if e["event"] == "failed" and e.get("final")} == {"GSM3", "GSM4"}
    assert not os.listdir(str(tmp_path / "scr"))                       # scratch cleaned
