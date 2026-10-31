"""Atlas re-run orchestrator (scripts/atlas_rerun): plan assignment, ledger fold, controller lifecycle on a fake Slurm.

No cluster and no binary are needed: job-side events are written with common.emit exactly as the stages write them.
"""

import json
from pathlib import Path
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "atlas_rerun"))

import common as C  # noqa: E402
import orchestrator as O  # noqa: E402
from ledger import Ledger  # noqa: E402
from slurm import FakeSlurm, parse_elapsed, parse_mem_gb, su_of_job  # noqa: E402


def make_cfg(tmp_path, **clipper):
    cfg = json.load(open(os.path.join(HERE, "..", "..", "scripts", "atlas_rerun", "config.json")))
    cfg["clusters"]["clipper"]["root"] = str(tmp_path / "clipper")
    cfg["clusters"]["anvil"]["root"] = str(tmp_path / "anvil")
    cfg["clusters"]["clipper"].update(clipper)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return C.load_config(str(p))


def plan_rows(n_studies=3, per=2, wave="a_empty", cluster="clipper", reads=1e8):
    rows, pri = [], 0
    for i in range(n_studies):
        for j in range(per):
            rows.append(
                {
                    "priority": pri,
                    "gsm": "GSM%d%02d" % (i + 1, j),
                    "gse": "GSE%d" % (i + 1),
                    "wave": wave,
                    "study_wave": wave,
                    "study_rank": i,
                    "sample_rank": j,
                    "organism": "Homo sapiens",
                    "reference": "GRCh38-2024-A",
                    "reads_available": int(reads),
                    "depth_unknown": 0,
                    "bases": int(reads * 300),
                    "n_runs": 1,
                    "runs": "SRR%d%02d" % (i + 1, j),
                    "run_reads": str(int(reads)),
                    "est_wall_s": C.compute_wall_s(reads),
                    "est_core_h": C.compute_core_h(reads),
                    "est_su": C.anvil_su(reads),
                    "est_sra_bytes": int(C.sra_bytes(reads, reads * 300)),
                    "size_class": "S",
                    "cluster": cluster,
                }
            )
            pri += 1
    return rows


def write_plan(cfg, rows, cluster="clipper"):
    root = cfg["clusters"][cluster]["root"]
    os.makedirs(os.path.join(root, "plan"), exist_ok=True)
    path = os.path.join(root, "plan", "plan.tsv.gz")
    C.write_plan(path, rows, {"test": True})
    return path


def test_cost_model_matches_plan_numbers():
    # 250M pairs, screen on, x1.54 mapping: (300 + (16.8 - 10.8 * (1 - 1/1.54)) * 250) * 1.15
    w = C.compute_wall_s(250e6)
    assert abs(w - (300 + (16.8 - 10.8 * (1 - 1 / 1.54)) * 250) * 1.15) < 1e-6
    assert C.compute_wall_s(250e6, threads=8) < w
    assert C.compute_wall_s(None) == C.compute_wall_s(C.MEDIAN_READS)


def test_plan_roundtrip(tmp_path):
    rows = plan_rows()
    p = str(tmp_path / "p.tsv.gz")
    meta = C.write_plan(p, rows, {"x": 1})
    with open(p, "rb") as fh:
        assert fh.read(2) == b"\x1f\x8b"  # really gzip
    back = C.read_plan(p)
    assert [r["gsm"] for r in back] == [r["gsm"] for r in rows]
    assert back[0]["runs"] == ["SRR100"] and back[0]["reads_available"] == 100000000
    assert C.read_plan_meta(p)["plan_id"] == meta["plan_id"]


def test_assign_whole_studies_to_anvil_within_budget():
    studies = [
        {
            "gse": "G%d" % i,
            "wave": "d_rest" if i >= 2 else "a_empty",
            "n": 3,
            "est_su": 100.0,
            "max_reads": 5e8,
        }
        for i in range(6)
    ]
    studies[3]["max_reads"] = 2e9  # needs a 5-day partition: stays on Clipper
    a = O.assign_clusters(studies, {"anvil_su_budget": 250})
    assert [a["G%d" % i] for i in range(6)] == [
        "clipper",
        "clipper",
        "anvil",
        "clipper",
        "anvil",
        "clipper",
    ]
    assert set(O.assign_clusters(studies, {"anvil_su_budget": 0}).values()) == {"clipper"}


def emit_ok(root, gsm, attempt=0, reads=1e8):
    C.emit(root, "downloading", gsm, attempt=attempt, stage="io")
    C.emit(
        root,
        "screened",
        gsm,
        attempt=attempt,
        stage="io",
        runs_kept=["SRR1"],
        runs_dropped=[],
        runs=[{"run": "SRR1", "keep": True, "reason": "ok"}],
    )
    C.emit(
        root,
        "downloaded",
        gsm,
        attempt=attempt,
        stage="io",
        sra_bytes=13e9,
        io_s=900,
        runs_kept=["SRR1"],
    )
    C.emit(root, "running", gsm, attempt=attempt, stage="compute", threads=4)
    C.emit(
        root,
        "done",
        gsm,
        attempt=attempt,
        stage="compute",
        compute_s=1500,
        out_dir="/x/" + gsm,
        manifest_sha256="ab" * 32,
        reads_used=int(reads),
        binary_sha256="cd" * 32,
    )


def test_controller_lifecycle_retry_park_handoff(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg["clusters"]["clipper"]["io"]["max_inflight"] = 4
    rows = plan_rows(n_studies=3, per=2)
    path = write_plan(cfg, rows)
    root = cfg["clusters"]["clipper"]["root"]
    fake = FakeSlurm()

    ctl = O.Controller(cfg, "clipper", fake, plan_path=path, log=lambda m: None)
    ctl.tick()
    # 4 samples fit the I/O cap: one io array + one compute array (aftercorr)
    assert len(fake.submitted) == 2
    io_cmd, cmp_cmd = fake.submitted[0]["cmd"], fake.submitted[1]["cmd"]
    assert "--array=0-3%4" in io_cmd
    assert (
        any(x.startswith("--dependency=aftercorr:") for x in cmp_cmd)
        and "--kill-on-invalid-dep=yes" in cmp_cmd
    )
    L = Ledger(root)
    assert sum(1 for s in L.samples.values() if s.get("io_job")) == 4

    # sample 1 succeeds, sample 2 fails in compute twice (retried once from the staged reads, then parked),
    # sample 3 parks itself in the I/O stage (no GEX run)
    emit_ok(root, "GSM100")
    emit_ok(root, "GSM101")
    C.emit(root, "downloading", "GSM200", attempt=0, stage="io")
    C.emit(root, "downloaded", "GSM200", attempt=0, stage="io", sra_bytes=1e9)
    C.emit(root, "failed", "GSM200", attempt=0, stage="compute", reason="oom")
    C.emit(root, "parked", "GSM201", attempt=0, stage="io", reason="no_gex_runs")
    staged = os.path.join(root, "stage", "GSM200")
    os.makedirs(staged)
    open(os.path.join(staged, "sample.1fq"), "w").close()

    ctl = O.Controller(cfg, "clipper", fake, plan_path=path, log=lambda m: None)
    st = ctl.tick()
    s = ctl.L.samples
    assert s["GSM100"]["state"] == "done" and s["GSM201"]["state"] == "parked"
    assert s["GSM200"]["attempt"] == 1 and s["GSM200"]["state"] == "downloaded"
    # its compute-only retry went out with more memory (oom escalation) and no I/O job
    retry = [x for x in fake.submitted[2:] if "ar-cmp-clipper-s4+mem" in " ".join(x["cmd"])]
    assert (
        retry
        and "--mem=144G" in retry[0]["cmd"]
        and not any("aftercorr" in c for c in retry[0]["cmd"])
    )
    assert os.path.exists(os.path.join(root, "handoff", "GSE1.json"))
    h = json.load(open(os.path.join(root, "handoff", "GSE1.json")))
    assert h["ready_for_packing"] and [x["gsm"] for x in h["samples"]] == ["GSM100", "GSM101"]
    assert "never be uploaded" in h["private"]["note"]
    assert st["waves"]["a_empty"]["done"] == 2

    # a late event of attempt 0 must not override attempt 1
    C.emit(root, "done", "GSM200", attempt=0, stage="compute")
    C.emit(root, "running", "GSM200", attempt=1, stage="compute")
    C.emit(root, "failed", "GSM200", attempt=1, stage="compute", reason="oom")
    ctl = O.Controller(cfg, "clipper", fake, plan_path=path, log=lambda m: None)
    ctl.tick()
    assert ctl.L.samples["GSM200"]["state"] == "parked"
    assert "retries_exhausted" in ctl.L.samples["GSM200"]["last_reason"]

    # the fold is reproducible from the log alone
    L2 = Ledger(root)
    L2.rebuild(C.read_plan(path, cluster="clipper"))
    for g in ("GSM100", "GSM200", "GSM201"):
        assert L2.samples[g]["state"] == ctl.L.samples[g]["state"]

    # snapshots and status
    O.cmd_status(type("A", (), {"ledger": [os.path.join(root, "ledger")], "json": False})())


def test_pause_blocks_submission_and_staging_cap(tmp_path):
    cfg = make_cfg(tmp_path)
    path = write_plan(cfg, plan_rows())
    root = cfg["clusters"]["clipper"]["root"]
    os.makedirs(root, exist_ok=True)
    open(os.path.join(root, "PAUSE"), "w").close()
    fake = FakeSlurm()
    O.Controller(cfg, "clipper", fake, plan_path=path, log=lambda m: None).tick()
    assert fake.submitted == []
    os.remove(os.path.join(root, "PAUSE"))
    cfg["clusters"]["clipper"]["staging_cap_tb"] = 0
    O.Controller(cfg, "clipper", fake, plan_path=path, log=lambda m: None).tick()
    assert fake.submitted == []


def test_class_choice_and_escalation(tmp_path):
    cfg = make_cfg(tmp_path)
    path = write_plan(cfg, plan_rows())
    ctl = O.Controller(cfg, "clipper", FakeSlurm(), plan_path=path, log=lambda m: None)

    def pick(reads, esc=None):
        return ctl.choose_class({"reads_available": reads}, {"escalate": esc})["name"]

    assert pick(1e8) == "s4"
    assert pick(8e8) in ("m8", "l8")
    assert pick(2e9) == "xl16"
    assert pick(1e8, "timeout") == "m8"
    assert pick(1e8, "oom") == "s4+mem"


def test_anvil_is_inert_until_approved(tmp_path):
    cfg = make_cfg(tmp_path)
    rows = plan_rows(cluster="anvil", wave="d_rest")
    path = write_plan(cfg, rows, cluster="anvil")
    fake = FakeSlurm()
    O.Controller(cfg, "anvil", fake, plan_path=path, log=lambda m: None).tick()
    assert fake.submitted == []  # enabled: false
    cfg["clusters"]["anvil"]["enabled"] = True  # still su_cap 0
    ctl = O.Controller(cfg, "anvil", fake, plan_path=path, log=lambda m: None)
    ctl.tick()
    assert fake.submitted == [] and ctl.stats["budget_stop"] == 1


def test_anvil_node_mode_packs_staged_samples(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg["clusters"]["anvil"].update(enabled=True, su_cap=10000)
    rows = plan_rows(n_studies=4, per=8, cluster="anvil", wave="d_rest", reads=3e8)
    path = write_plan(cfg, rows, cluster="anvil")
    root = cfg["clusters"]["anvil"]["root"]
    fake = FakeSlurm()
    O.Controller(cfg, "anvil", fake, plan_path=path, log=lambda m: None).tick()
    io = [x for x in fake.submitted if "ar-io-anvil" in " ".join(x["cmd"])]
    assert io and "--partition=shared" in io[0]["cmd"] and "--mem=1946M" in io[0]["cmd"]
    for r in rows:
        C.emit(root, "downloading", r["gsm"], attempt=0, stage="io")
        C.emit(root, "downloaded", r["gsm"], attempt=0, stage="io", sra_bytes=4e10)
    n0 = len(fake.submitted)
    ctl = O.Controller(cfg, "anvil", fake, plan_path=path, log=lambda m: None)
    ctl.tick()
    node = [x for x in fake.submitted[n0:] if "ar-node-anvil" in " ".join(x["cmd"])]
    assert node and "--exclusive" in node[0]["cmd"] and "--partition=wholenode" in node[0]["cmd"]
    in_node = [s for s in ctl.L.samples.values() if s.get("node_job")]
    cap = 112 * 24 * 0.8
    assert 0 < sum(C.compute_core_h(3e8) for _ in in_node) <= cap


class ScriptedSlurm(FakeSlurm):
    """FakeSlurm whose queue is empty and whose sacct answers from a dict."""

    def __init__(self, acct):
        FakeSlurm.__init__(self)
        self.dry = False
        self._acct = acct

    def acct(self, job_ids):
        return {j: self._acct[j] for j in job_ids if j in self._acct}


def test_cancelled_chained_compute_is_resubmitted_not_failed(tmp_path):
    # Slurm marks an I/O job OUT_OF_MEMORY when a child was OOM-killed even if the stage recovered and staged the reads;
    # aftercorr then cancels the compute task. The controller must resubmit compute alone, without a failure.
    cfg = make_cfg(tmp_path)
    path = write_plan(cfg, plan_rows(n_studies=1, per=1))
    root = cfg["clusters"]["clipper"]["root"]
    fake = FakeSlurm()
    O.Controller(cfg, "clipper", fake, plan_path=path, log=lambda m: None).tick()
    C.emit(root, "downloading", "GSM100", attempt=0, stage="io")
    C.emit(root, "downloaded", "GSM100", attempt=0, stage="io", sra_bytes=1e9)
    open(
        os.path.join(root, "bin", "singlet-" + cfg["binary"]["sha256"][:12]), "w"
    ).close()  # already installed
    s = ScriptedSlurm({"900000002_0": {"state": "CANCELLED", "elapsed_s": 0}})
    for _ in range(2):  # one tick of grace, then the release
        ctl = O.Controller(cfg, "clipper", s, plan_path=path, log=lambda m: None)
        ctl.tick()
    g = ctl.L.samples["GSM100"]
    assert g["failures"] == 0 and g["state"] == "downloaded"
    assert any(
        "ar-cmp-clipper" in " ".join(x["cmd"]) and not any("aftercorr" in c for c in x["cmd"])
        for x in s.submitted
    )


def test_slurm_parsers():
    assert parse_elapsed("1-02:03:04") == 93784 and parse_elapsed("03:59:00") == 14340
    assert parse_mem_gb("96G") == 96 and abs(parse_mem_gb("7600M") - 7.6) < 1e-9
    assert (
        su_of_job(
            {"elapsed_s": 3600, "alloc_cpus": 128, "req_mem_gb": 257, "partition": "wholenode"}
        )
        == 128
    )
    assert (
        abs(
            su_of_job(
                {"elapsed_s": 3600, "alloc_cpus": 1, "req_mem_gb": 3.8, "partition": "shared"}
            )
            - 2
        )
        < 1e-9
    )


def test_strand_contradiction_exit_8_reruns_on_the_sorted_path():
    import stages as S

    cmd = ["singlet", "s.1fq", "--pipeline", "--genome-shared", "off"]
    assert S.STRAND_CONTRADICTION == 8
    assert S.strand_retry_cmd(cmd) == cmd + ["--sorted-bam"]
    assert S.strand_retry_cmd(cmd + ["--sorted-bam"]) == cmd + ["--sorted-bam"]  # never doubled
    assert S.strand_retry_cmd(cmd) is not cmd and cmd[-1] == "off"  # the first command is untouched
    # still contradicting after the sorted-path retry: reported as such, not as a generic exit
    assert S.classify_failure(8, "...") == "strand_contradiction"
    assert S.classify_failure(137, "") == "oom"


def _compute(tmp_path, monkeypatch, staged, rcs, pooled=False):
    """Run compute_stage with the engine replaced by a recorder that answers `rcs` in turn."""
    import stages as S

    tmp_path.mkdir(exist_ok=True)
    cfg = make_cfg(tmp_path)
    cfg["pipeline"]["pooled_input"] = pooled
    ccfg = cfg["clusters"]["clipper"]
    ccfg.update(scratch=str(tmp_path / "scr"), stage_root=str(tmp_path / "stage"))
    sdir = tmp_path / "stage" / "GSM1"
    sdir.mkdir(parents=True)
    for name in staged:
        (sdir / name).write_text("x")
    calls = []

    def fake_run(cmd, log, env=None, cwd=None, timeout=None):
        calls.append(list(cmd))
        return rcs[len(calls) - 1], 1.0, "out"

    monkeypatch.setattr(S, "run", fake_run)
    monkeypatch.setattr(S, "verify_binary", lambda ctx: None)
    monkeypatch.setattr(S, "collect_outputs", lambda *a: (str(tmp_path / "o"), None, "sha", 1))
    ctx = S.Ctx(cfg, ccfg)
    rc = S.compute_stage(ctx, {"gsm": "GSM1", "gse": "GSE1"}, 0, 4)
    return rc, calls


def test_multi_run_sample_gets_its_per_run_files_not_a_pool(tmp_path, monkeypatch):
    rc, calls = _compute(tmp_path / "a", monkeypatch, ["SRR1.1fq", "SRR2.1fq", "screen.json"], [0])
    cmd = calls[0]
    assert rc == 0 and cmd[cmd.index("--runs-dir") + 1].endswith("GSM1")
    assert "--run-screen" in cmd and not any(x.endswith("sample.1fq") for x in cmd)
    # the legacy pooled sample (and a single-run sample) is still one positional file
    rc, calls = _compute(tmp_path / "b", monkeypatch, ["sample.1fq", "screen.json"], [0])
    assert calls[0][calls[0].index("--ref-base") - 1].endswith("sample.1fq") and "--runs-dir" not in calls[0]


def test_per_run_files_without_a_screen_fail_early(tmp_path, monkeypatch):
    rc, calls = _compute(tmp_path / "a", monkeypatch, ["SRR1.1fq", "SRR2.1fq"], [0])
    assert rc == 1 and not calls


def test_exit_8_in_compute_stage_reruns_once_on_the_sorted_path(tmp_path, monkeypatch):
    rc, calls = _compute(tmp_path / "a", monkeypatch, ["sample.1fq"], [8, 0])
    assert rc == 0 and len(calls) == 2 and "--sorted-bam" in calls[1] and "--sorted-bam" not in calls[0]
    rc, calls = _compute(tmp_path / "b", monkeypatch, ["sample.1fq"], [8, 8])
    assert rc == 1 and len(calls) == 2  # a second exit 8 is not retried again


def test_job_slots_from_node_memory():
    # deep GSM7036403 with the shared index: 33.4 GB per job, 17.8 GB index once per node (S1's measurement)
    assert C.job_slots(512, 128, 4) == 13
    assert C.job_slots(768, 128, 8) == 16  # cores bind first
    assert C.job_slots(768, 256, 4) == 20  # memory binds
    assert C.job_slots(64, 128, 4) == 1  # never zero
    assert C.job_slots(0, 128, 4) == 32  # memory unknown: cores only


def test_beta_target_guard():
    import beta

    ok = {"base": "https://beta.singlet.bio", "bucket": "singlet-beta-data", "database": "singlet-beta-catalog"}
    assert beta.check_target(ok) == "https://beta.singlet.bio"
    for bad in ({"base": "https://singlet.bio"}, {"base": "https://data.singlet.bio"}, {"bucket": "singlet-data"},
                {"database": "singlet-catalog"}, {"bucket": None}, {"base": "https://evil.example@singlet.bio"}):
        try:
            beta.check_target({**ok, **bad})
        except beta.BetaRefused:
            continue
        raise AssertionError("accepted %r" % bad)


def test_beta_manifest_refuses_private_keys():
    import beta

    good = {"samples": [{"key": "s1/s/GSM1/" + "a" * 32 + ".s1"}]}
    assert beta.check_manifest_keys(good) == [good["samples"][0]["key"]]
    for k in ("private/GSE1/GSM1/snp.tsv", "s1/s/GSM1/private/x.s1", "restricted/GSE1/x.singlet", "GSM1.bam"):
        try:
            beta.check_manifest_keys({"samples": [{"key": k}]})
        except beta.BetaRefused:
            continue
        raise AssertionError(k)


def test_beta_publish_dry_run_and_real_run_with_a_stand_in_uploader(tmp_path):
    import hashlib

    import beta

    cfg = make_cfg(tmp_path)
    ccfg = cfg["clusters"]["clipper"]
    root = ccfg["root"]
    os.makedirs(os.path.join(root, "handoff"))
    open(os.path.join(root, "handoff", "GSE1.json"), "w").write("{}")
    body = b"s1-object"
    sha = hashlib.sha256(body).hexdigest()
    key = "s1/s/GSM1/%s.s1" % sha[:32]
    manifest = {"gse": "GSE1", "rev": "r", "samples": [{"gsm": "GSM1", "key": key, "bytes": len(body), "sha256": sha}]}
    producer = tmp_path / "producer.py"
    producer_src = (
        "import json, os, sys\n"
        "work = sys.argv[1]\n"
        "os.makedirs(os.path.join(work, 's1', 's', 'GSM1'), exist_ok=True)\n"
        "open(os.path.join(work, %r), 'wb').write(BODY)\n"
        "json.dump(%r, open(os.path.join(work, 'GSE1.manifest.json'), 'w'))\n" % (key, manifest))
    producer.write_text("BODY = %r\n" % body + producer_src)
    uploader = tmp_path / "beta_ingest.py"
    uploader.write_text(
        "import hashlib, json, re, sys\n"
        "DIGEST_RE = re.compile(r'/([0-9a-f]{32})\\.')\n"
        "def sha256_file(p):\n    return hashlib.sha256(open(p, 'rb').read()).hexdigest()\n"
        "class Client:\n    def __init__(s, b, t): pass\n    def call(s, *a, **k): return 404, {}\n"
        "if __name__ == '__main__':\n"
        "    print(json.dumps({'gse': 'GSE1', 'uploaded': 1, 'bytes_sent': 9, 'd1_rows_written': 6})); sys.exit(0)\n")
    cfg["beta"].update(enabled=True, uploader=str(uploader), retries=2, retry_wait_s=0,
                       producer=[sys.executable, str(producer), "{work}"])
    # no token: a dry run still verifies the objects locally; a real run refuses to start
    r = beta.publish_study(cfg, ccfg, "GSE1", dry_run=True, env={})
    assert r["status"] == "dry_run_ok" and r["objects"] == 1 and r["present"] is None
    assert beta.publish_study(cfg, ccfg, "GSE1", env={})["status"] == "failed"
    r = beta.publish_study(cfg, ccfg, "GSE1", env={"SINGLET_INGEST_TOKEN": "t"})
    assert r["status"] == "published" and r["d1_rows_written"] == 6
    # no producer configured: nothing is uploaded; a production target is refused before anything runs
    keep = cfg["beta"]["producer"]
    cfg["beta"]["producer"] = None
    assert beta.publish_study(cfg, ccfg, "GSE1", env={})["status"] == "no_producer"
    cfg["beta"]["bucket"] = "singlet-data"
    assert beta.publish_study(cfg, ccfg, "GSE1", env={"SINGLET_INGEST_TOKEN": "t"})["status"] == "refused"
    # an object that differs from the manifest is refused locally
    cfg["beta"].update(bucket="singlet-beta-data", producer=keep)
    producer.write_text(producer.read_text().replace(repr(body), repr(b"tampered")))
    assert beta.publish_study(cfg, ccfg, "GSE1", env={"SINGLET_INGEST_TOKEN": "t"})["status"] == "refused"


def test_s1_producer_copies_only_the_manifest_and_its_objects(tmp_path):
    import hashlib

    import s1_producer as P

    ho = tmp_path / "ho.json"
    out = tmp_path / "o"
    out.mkdir()
    ho.write_text(json.dumps({"samples": [{"gsm": "GSM1", "state": "done", "out_dir": str(out)},
                                          {"gsm": "GSM2", "state": "parked", "out_dir": str(out)}]}))
    res = tmp_path / "res"
    res.mkdir()
    try:
        assert P.build_results(str(ho), str(res)) == 1 and sorted(os.listdir(res)) == ["GSM1"]
    except OSError:  # no symlinks on this machine
        return
    sha = hashlib.sha256(b"x").hexdigest()
    key, lab = "s1/s/GSM1/%s.s1" % sha[:32], "s1/a/GSM1/%s.parquet" % sha[:32]
    store = tmp_path / "store"
    for k, data in ((key, b"x"), (lab, b"y"), ("m/GSE1.json", json.dumps({"samples": [{"key": key, "labels": lab, "n_columns": 5, "n_called": 2, "stats": {"a": 1}}]}).encode()),
                    ("s/GSM9/other.s1", b"z")):
        p = store / (k[3:] if k.startswith("s1/") else k)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    work = tmp_path / "work"
    assert P.copy_store(str(store), "GSE1", str(work)) == 2
    man = json.loads((work / "GSE1.manifest.json").read_text())
    assert man["n_samples"] == 1 and (work / "s1" / "s" / "GSM1" / (sha[:32] + ".s1")).exists()
    assert not (work / "s1" / "s" / "GSM9").exists()


def test_beta_policy_withholds_what_alpha_withholds_and_reaches_s1_publish():
    """The producer's default policy loads and removes every withheld layer; s1_publish passes it on to the transcoder."""
    sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
    sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
    from singlet.s1 import policy as POL

    pol = POL.Policy.load(os.path.join(HERE, "..", "..", "scripts", "atlas_rerun", "beta-policy.json"))
    assert pol.version.startswith("beta-") and pol.manifest_block()["omitted_layers"] == list(POL.OMITTED_LAYERS)
    for rel in ("mt_heteroplasmy.1pz", "mt_variants.tsv", "mt/mt_summary.tsv", "ancestry_call.json", "donor/snp_counts.tsv",
                "snp_allele_counts.tsv", "x.vcf.gz", "possorted.bam", "ambient_profile.tsv", "doublet_scores.tsv"):
        assert POL.omitted_layer(rel), rel
    assert POL.omitted_layer("exon_counts.1pz") is None and POL.omitted_layer("donor/donor_assignments.tsv") is None
    import argparse

    import s1_publish as PUB

    ns = argparse.Namespace(apply=True, gse="G", io_threads=1, blas_threads=1, migrate_timeout=1, verify_timeout=1,
                            rows_dir="r", ledger_dir="l", readback="none", public="b", commit="", policy="/p/beta-policy.json", study_log_dir=None,
                            migrate_cmd=None, verify_cmd=None, pythonpath=None, local_root="x", part_size_mib=None,
                            retry_sleep=None)
    args = PUB.run_args(ns, "G", Path("/tmp/x"))
    assert args[args.index("--alpha-policy") + 1] == "/p/beta-policy.json"


def test_s1_producer_refuses_a_hollow_object(tmp_path):
    import s1_producer as P

    store = tmp_path / "s"
    (store / "m").mkdir(parents=True)
    (store / "m" / "GSE1.json").write_text(json.dumps({"samples": [{"gsm": "GSM1", "key": "s1/s/GSM1/a.s1", "n_columns": 0,
                                                                   "n_called": 0, "stats": {}}]}))
    try:
        P.copy_store(str(store), "GSE1", str(tmp_path / "w"))
    except SystemExit as e:
        assert "hollow" in str(e)
    else:
        raise AssertionError("published a hollow object")
