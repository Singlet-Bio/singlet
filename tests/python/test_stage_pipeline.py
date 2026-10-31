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


def test_launcher_policy():
    p = dict(L.DEFAULTS, gpu_Mpairs_per_s=40, gpu_start_h=1.0, max_gpu=1, cpu_overflow_h=3.0, cpu_cores=16, cpu_Mpairs_per_core_s=0.06,
             cpu_job_h=3.9, max_cpu=8, fetch_jobs=4)
    none = {"gpu": 0, "cpu": 0, "fetch": 0}
    bl = lambda pairs, todo=0, fetching=0: dict(samples=1, pairs=pairs, bytes=0, claimed=0, todo=todo, fetching=fetching)  # noqa: E731
    # 0.5 GPU-h waiting: no start (a GPU job would idle at the tail)
    assert L.plan(p, bl(0.5 * 40e6 * 3600), none)["gpu"] == 0
    # 1.2 GPU-h: start the one GPU worker; no CPU overflow
    d = L.plan(p, bl(1.2 * 40e6 * 3600), none)
    assert d["gpu"] == 1 and d["cpu"] == 0
    # 10 GPU-h waiting and a GPU already running: overflow beyond 3 h goes to CPU workers
    d = L.plan(p, bl(10 * 40e6 * 3600), dict(gpu=1, cpu=0, fetch=0))
    assert d["gpu"] == 0 and 1 <= d["cpu"] <= 8
    # fetch workers topped up to fetch_jobs while there is todo
    assert L.plan(p, bl(0, todo=100), dict(gpu=0, cpu=0, fetch=1))["fetch"] == 3
    assert L.plan(p, bl(0, todo=0), none)["fetch"] == 0


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
