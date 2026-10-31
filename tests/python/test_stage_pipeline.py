"""Decoupled fetch -> stage -> drain pipeline (scripts/atlas_rerun: stagequeue, launcher.plan, fetch_worker.RangeFetcher).

No cluster, no binary, no network beyond a localhost range server: the queue is exercised with real renames in a tmp dir.
"""

import http.client
import http.server
import json
import os
import sys
import threading

import pytest
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "atlas_rerun"))

import stagequeue as Q  # noqa: E402
import launcher as L  # noqa: E402


def todo(stage, gsm, **kw):
    with open(os.path.join(stage, "todo", gsm + ".json"), "w") as f:
        json.dump(dict(gsm=gsm, runs=["SRR1"], **kw), f)


def stage_sample(stage, gsm, nbytes=1000, pairs=10_000_000):
    c = Q.claim_todo(stage, "w")
    assert c and c[0] == gsm
    inc = Q.new_incoming(stage, gsm, "w")
    with open(os.path.join(inc, "SRR1.1fq"), "wb") as f:
        f.write(b"x" * nbytes)
    return Q.publish(stage, gsm, inc, c[1], {"pairs": pairs, "runs": ["SRR1"]})


def test_claim_is_exclusive_and_finish_deletes(tmp_path):
    s = Q.init(str(tmp_path / "stage"))
    for g in ("GSM1", "GSM2"):
        todo(s, g)
        stage_sample(s, g)
    a = Q.claim_ready(s, "a")
    b = Q.claim_ready(s, "b")
    assert {a[0], b[0]} == {"GSM1", "GSM2"} and Q.claim_ready(s, "c") is None
    Q.finish(s, a[0], a[1])
    assert not os.path.exists(a[1]) and os.listdir(os.path.join(s, "deleting")) == []
    assert Q.backlog(s)["claimed"] == 1


def test_crash_recovery_and_attempt_limit(tmp_path):
    s = Q.init(str(tmp_path / "stage"))
    todo(s, "GSM1")
    stage_sample(s, "GSM1")
    for attempt in (1, 2):
        g, claim, man = Q.claim_ready(s, "dead")
        assert man["attempts"] == attempt
        # the worker dies: nobody heartbeats; a live claim is not reaped, a stale one is
        assert Q.reap(s, log=lambda *_: None)["claimed"] == 0
        Q.reap(s, now=time.time() + Q.HB_STALE_S + 5, log=lambda *_: None)
        assert os.path.isdir(os.path.join(s, "ready", "GSM1"))
    g, claim, man = Q.claim_ready(s, "dead")  # third attempt
    assert man["attempts"] == 3
    n = Q.reap(s, now=time.time() + Q.HB_STALE_S + 5, log=lambda *_: None)
    assert n["failed"] == 1 and not os.listdir(os.path.join(s, "ready"))


def test_fetch_crash_returns_todo_and_clears_incoming(tmp_path):
    s = Q.init(str(tmp_path / "stage"))
    todo(s, "GSM1")
    gsm, claim, item = Q.claim_todo(s, "dead")
    Q.new_incoming(s, gsm, "dead")
    assert not os.listdir(os.path.join(s, "todo"))
    later = time.time() + Q.HB_STALE_S + 5
    Q.reap(s, now=later, log=lambda *_: None)
    assert os.listdir(os.path.join(s, "todo")) == ["GSM1.json"] and not os.listdir(os.path.join(s, "fetching"))
    assert not os.listdir(os.path.join(s, "incoming"))


def test_backpressure_counts_ready_claimed_incoming(tmp_path):
    s = Q.init(str(tmp_path / "stage"))
    todo(s, "GSM1")
    assert stage_sample(s, "GSM1", nbytes=5000) >= 5000
    todo(s, "GSM2")
    c = Q.claim_todo(s, "w")
    inc = Q.new_incoming(s, "GSM2", "w")
    open(os.path.join(inc, "x"), "wb").write(b"y" * 3000)
    assert Q.staged_bytes(s) >= 8000
    assert Q.wait_for_room(s, 6000, poll_s=0.05, log=lambda *_: None, max_wait_s=0.3) is False  # full: the fetch worker pauses
    g, claim, _ = Q.claim_ready(s, "d")
    Q.finish(s, g, claim)  # the drain worker finished GSM1: room again
    assert Q.wait_for_room(s, 6000, poll_s=0.05, log=lambda *_: None, max_wait_s=2) is True
    assert c is not None


def J(run=0, pend=0, ages=()):
    return {"run": run, "pend": pend, "ages": list(ages)}


def test_launcher_policy():
    p = dict(L.DEFAULTS, gpu_Mpairs_per_s=10, gpu_start_h=1.0, max_h100=6, max_v100=4, v100_rel=0.3, overflow_h=3.0, fallback_wait_h=1.0,
             cpu_cores=16, cpu_Mpairs_per_core_s=0.025, cpu_job_h=3.9, max_cpu=8, fetch_jobs=6)
    none = {"h100": J(), "v100": J(), "cpu": J(), "fetch": 0}
    H = 10e6 * 3600  # pairs of one H100-hour
    bl = lambda pairs, todo=0, fetching=0: dict(samples=1, pairs=pairs, bytes=0, claimed=0, todo=todo, fetching=fetching)  # noqa: E731
    # under X: nothing (a GPU job would idle at the tail)
    assert L.plan(p, bl(0.5 * H), none)["h100"] == 0
    # 2.5 H100-hours: two workers, each with >= X hours; no fallback pools
    d = L.plan(p, bl(2.5 * H), none)
    assert (d["h100"], d["v100"], d["cpu"]) == (2, 0, 0)
    # already two pending (young): nothing more
    assert L.plan(p, bl(2.5 * H), dict(none, h100=J(pend=2, ages=[60, 90])))["h100"] == 0
    # capped at max_h100 (3 nodes x 2 GPUs)
    assert L.plan(p, bl(50 * H), none)["h100"] == 6
    # scavenger starvation: both pending jobs have waited > 1 h and 30 h of work is waiting -> V100S then CPU take the excess
    starved = dict(none, h100=J(pend=6, ages=[7200] * 6))
    d = L.plan(p, bl(30 * H), starved)
    assert d["h100"] == 0 and d["v100"] == 4 and d["cpu"] >= 1
    # the same backlog with young pending jobs: wait, no fallback (CPU overflow only if the H100s cannot clear it in overflow_h)
    young = dict(none, h100=J(pend=6, ages=[60] * 6))
    d = L.plan(p, bl(30 * H), young)
    assert d["v100"] == 0
    # fetch workers topped up to fetch_jobs while there is todo
    assert L.plan(p, bl(0, todo=100), dict(none, fetch=2))["fetch"] == 4
    assert L.plan(p, bl(0, todo=0), none)["fetch"] == 0


def test_preempted_claim_returns_with_attempt_refunded(tmp_path):
    s = Q.init(str(tmp_path / "stage"))
    todo(s, "GSM1")
    stage_sample(s, "GSM1")
    g, claim, man = Q.claim_ready(s, "w")
    assert man["attempts"] == 1
    Q.release(s, g, claim, refund=True)  # what drain_worker does on SIGTERM
    g, claim, man = Q.claim_ready(s, "w2")
    assert man["attempts"] == 1  # preemption does not eat the 3 attempts


def test_global_rate_cap_is_shared(tmp_path):
    import ratecap
    stage = str(tmp_path / "stage")
    os.makedirs(stage)
    ratecap.write_cap_mbps(stage, 100)
    a, b = ratecap.SharedRate(stage, "a-1"), ratecap.SharedRate(stage, "b-2")
    a.set_active(True)
    b.set_active(True)
    a._tick()
    assert abs(a.share_bps - 50e6) < 1e6 and abs(b.share_bps - 50e6) < 1e6  # two active workers split the cap
    b.set_active(False)  # b is decoding: a gets it all
    a._tick()
    assert abs(a.share_bps - 100e6) < 1e6
    ratecap.write_cap_mbps(stage, 900)  # raising the cap is a config edit
    a._tick()
    assert abs(a.share_bps - 900e6) < 1e6
    # the bucket paces: 20 MB at 900 MB/s with 0.25 s credit finishes fast; at 10 MB/s it takes ~2 s
    ratecap.write_cap_mbps(stage, 10)
    a._tick()
    t0 = time.time()
    for _ in range(20):
        a.acquire(1_000_000)
    assert 1.5 < time.time() - t0 < 3.5
    a.close()
    b.close()


def test_sdl_prefers_full_s3_and_skips_lite_by_default(monkeypatch):
    import fetch_worker as F
    doc = {"result": [{"files": [
        {"name": "SRRX", "size": "100", "md5": "m1", "locations": [{"service": "s3", "link": "https://sra-pub-run-odp.s3.amazonaws.com/sra/SRRX/SRRX"}]},
        {"name": "SRRX.lite", "size": "60", "md5": "m2", "locations": [
            {"service": "ncbi", "link": "https://sra-downloadb.be-md.ncbi.nlm.nih.gov/x.lite.1"},
            {"service": "s3", "link": "https://sra-pub-zq-3.s3.amazonaws.com/SRRX/SRRX.lite.1"},
            {"service": "gs", "link": "https://storage.googleapis.com/sra-pub-zq-100/SRRX/SRRX.lite.1"}]}]}]}

    class R:
        def read(self, *a):
            return json.dumps(doc).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(F.urllib.request, "urlopen", lambda *a, **k: R())
    full_only = F.sdl_locations("SRRX")
    assert [(x["kind"], x["service"]) for x in full_only] == [("full", "s3")]
    with_lite = F.sdl_locations("SRRX", allow_lite=True)
    assert [(x["kind"], x["service"]) for x in with_lite] == [("full", "s3"), ("lite", "ncbi")]  # requester-pays copies never listed


@pytest.mark.skipif(not hasattr(os, "pwrite"), reason="pwrite/pread are POSIX-only (the fetcher runs on Linux)")
def test_range_fetcher_md5_and_retry(tmp_path):
    import hashlib
    import fetch_worker as F
    data = os.urandom(3 * F.BLOCK + 12345)
    fails = {"n": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            a, b = self.headers["Range"].split("=")[1].split("-")
            a, b = int(a), int(b)
            if a == F.BLOCK and fails["n"] < 2:  # the second block fails twice before it works
                fails["n"] += 1
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Length", str(b - a + 1))
            self.end_headers()
            self.wfile.write(data[a:b + 1])

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    class Plain(F.RangeFetcher):  # the server is plain HTTP
        def _conn(self):
            return http.client.HTTPConnection(self.u.netloc, timeout=10)

    out = str(tmp_path / "f")
    f = Plain("http://127.0.0.1:%d/x" % srv.server_address[1], len(data), out, 4, md5=True, retries=4)
    assert f.run() == hashlib.md5(data).hexdigest() and open(out, "rb").read() == data and fails["n"] == 2
    srv.shutdown()


# ── per-sample Beta publish (sample_publish.py) ─────────────────────────────────────────────────────────────────────

def _one_sample_manifest(gse, gsm, key_digest):
    sf = {"feature_vocab.json": {"size": 1, "crc32": 1, "sha256": "ab" * 32}}
    key = "s1/s/%s/%s.s1" % (gsm, key_digest)
    return {"format": "singlet-s1", "gse": gse, "rev": "x", "study_files": sf, "samples": [{"gsm": gsm, "key": key, "bytes": 3, "sha256": "0" * 64}]}


def test_manifest_merge_replaces_and_rev_matches_singlet():
    import sample_publish as SP
    a = _one_sample_manifest("GSE1", "GSM1", "a" * 32)
    m1 = SP.merge_manifests(None, a)
    b = _one_sample_manifest("GSE1", "GSM2", "b" * 32)
    m2 = SP.merge_manifests(m1, b)
    assert [s["gsm"] for s in m2["samples"]] == ["GSM1", "GSM2"] and m2["n_samples"] == 2 and m2["rev"] != m1["rev"]
    # re-running GSM2 with the same key is idempotent; with a new digest it replaces the old object in the manifest
    assert SP.merge_manifests(m2, b)["rev"] == m2["rev"]
    b2 = _one_sample_manifest("GSE1", "GSM2", "c" * 32)
    m3 = SP.merge_manifests(m2, b2)
    assert m3["n_samples"] == 2 and m3["samples"][1]["key"].endswith("c" * 32 + ".s1")
    # same algorithm as the manifest builder of the s1 package
    sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
    try:
        from singlet.s1 import manifest as M
    except Exception:  # noqa: BLE001 - optional compiled dependencies not built here
        pytest.skip("singlet.s1.manifest not importable")
    assert SP.compute_rev([s["key"] for s in m2["samples"]], m2["study_files"]) == M.compute_rev([s["key"] for s in m2["samples"]], m2["study_files"])


def test_study_lock_is_exclusive_and_stale_locks_expire(tmp_path):
    import sample_publish as SP
    with SP.StudyLock(str(tmp_path), "GSE1"):
        with pytest.raises(TimeoutError):
            SP.StudyLock(str(tmp_path), "GSE1", wait_s=0.5).__enter__()
        os.utime(os.path.join(str(tmp_path), "beta", "locks", "GSE1"), (0, 0))  # holder died long ago
        with SP.StudyLock(str(tmp_path), "GSE1", wait_s=2):
            pass
    assert not os.path.exists(os.path.join(str(tmp_path), "beta", "locks", "GSE1"))


def test_publish_refuses_a_non_beta_target(tmp_path):
    import sample_publish as SP
    r = SP.publish_sample({"beta": {"base": "https://singlet.bio", "bucket": "singlet-data", "database": "singlet-catalog"}},
                          {"root": str(tmp_path)}, "GSE1", "GSM1", str(tmp_path), str(tmp_path / "s"), dry_run=False)
    assert r["status"] == "refused"
