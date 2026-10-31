#!/usr/bin/env python3
"""Fetch worker (S9): download SRA runs at line rate, decode locally, stage compactly, never wait for compute.

    fetch_worker.py --stage DIR --cap-tb 20 [--streams 64] [--idle-exit-min 30] [--config config.json] [--cluster clipper]

Loop: claim a todo item (stagequeue) -> wait for room under the stage cap (backpressure) -> per run: NCBI SDL (with alternate
locations) -> SRA Lite on NCBI's host, else full .sra on public S3 -> parallel ranged GETs straight into a node-local /tmp file
(pwrite, keep-alive connections, per-block retries, a hasher thread md5s the contiguous prefix while the rest downloads and the
result is checked against the SDL md5) -> `singlet download` decodes the local file to a .1fq with binned qualities (2.2 core-s per M
spots) -> raw file deleted -> optional `screen-runs` -> the sample directory is renamed into ready/.

Only the staged .1fq (and screen.json) ever reach the shared stage; the .sra lives in node-local /tmp for the duration of its decode.
A fetch job holds a few runs in /tmp at once at most (--runs-in-flight), so a 2 TB NVMe is not a constraint.
Standard library only; events go to the orchestrator inbox when --config is given, so the ledger sees the same states as before.
"""

import argparse
import hashlib
import http.client
import json
import os
import re
try:
    import resource
except ImportError:  # not POSIX: the fetcher only runs on Linux; tests import the module
    resource = None
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratecap  # noqa: E402
import stagequeue as Q  # noqa: E402

BLOCK = 8 << 20
SDL = "https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve?acc=%s&filetype=run&accept-alternate-locations=yes"


class FetchError(Exception):
    pass


Preempted = Q.Preempted


def sdl_locations(acc, allow_lite=False, tries=4):
    """-> candidate sources, best first: [{'kind': 'full'|'lite', 'service', 'url', 'size', 'md5'}].
    Order (decision 2026-10-30: the SNP/mtDNA callers need real qualities): full-quality `.sra` on public S3 (`sra-pub-run-odp`), then a
    full `.sra` on NCBI's own host if the SDL lists one, then (only with allow_lite) SRA Lite on NCBI's host. The S3/GCS Lite copies
    are requester-pays and never used."""
    err = None
    for i in range(tries):
        try:
            d = json.load(urllib.request.urlopen(SDL % acc, timeout=60))
            break
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(3 * (i + 1))
    else:
        raise FetchError("sdl_failed: %r" % err)
    res = (d.get("result") or [{}])[0]
    if res.get("status") not in (None, 200) and not res.get("files"):
        raise FetchError("sdl_no_files: %s" % res.get("message"))
    out = []
    for f in res.get("files", []):
        lite = f["name"].endswith(".lite")
        for loc in f.get("locations", []):
            svc, link = loc["service"], loc["link"]
            if not lite and svc == "s3" and "sra-pub-run-odp" in link:
                rank = 0
            elif not lite and svc == "ncbi":
                rank = 1
            elif lite and svc == "ncbi" and allow_lite:
                rank = 2
            else:
                continue
            out.append((rank, {"kind": "lite" if lite else "full", "service": svc, "url": link, "size": int(f["size"]), "md5": f.get("md5")}))
    out.sort(key=lambda x: x[0])
    if not out:
        raise FetchError("no_open_location")
    return [x[1] for x in out]


class RangeFetcher:
    """N threads, each with its own keep-alive HTTPS connection, take 8 MiB blocks off a shared counter and pwrite them."""

    def __init__(self, url, size, path, streams, md5=None, retries=6, limiter=None):
        self.limiter = limiter
        self.url, self.size, self.path, self.streams, self.retries = url, size, path, streams, retries
        self.nblocks = (size + BLOCK - 1) // BLOCK
        self.next, self.lock = 0, threading.Lock()
        self.done = bytearray(self.nblocks)
        self.err = None
        self.bytes = 0
        self.md5 = hashlib.md5() if md5 else None
        self.u = urllib.parse.urlparse(url)

    def _conn(self):
        return http.client.HTTPSConnection(self.u.netloc, timeout=60, context=ssl.create_default_context())

    def _get(self, conn, fd, i):
        a, b = i * BLOCK, min((i + 1) * BLOCK, self.size) - 1
        path = self.u.path + ("?" + self.u.query if self.u.query else "")
        conn.request("GET", path, headers={"Range": "bytes=%d-%d" % (a, b), "Connection": "keep-alive"})
        r = conn.getresponse()
        if r.status not in (200, 206):
            r.read()
            raise FetchError("http_%d" % r.status)
        if r.status == 200 and self.nblocks > 1:
            raise FetchError("range_ignored")
        buf, off = bytearray(1 << 20), a
        mv = memoryview(buf)
        while off <= b:
            n = r.readinto(mv[: min(len(buf), b - off + 1)])
            if not n:
                raise FetchError("short_read")
            if self.limiter:
                self.limiter.acquire(n)
            os.pwrite(fd, mv[:n], off)
            off += n
        with self.lock:
            self.bytes += b - a + 1

    def _worker(self):
        try:
            self._work()
        except Exception as e:  # noqa: BLE001 - a dying thread must stop the others and the hasher, never leave them waiting
            self.err = self.err or FetchError("worker: %r" % e)

    def _work(self):
        fd = os.open(self.path, os.O_WRONLY)
        conn = self._conn()
        try:
            while self.err is None:
                with self.lock:
                    i = self.next
                    self.next += 1
                if i >= self.nblocks:
                    return
                for t in range(self.retries):
                    try:
                        self._get(conn, fd, i)
                        self.done[i] = 1
                        break
                    except (OSError, http.client.HTTPException, FetchError) as e:
                        conn.close()
                        conn = self._conn()
                        if t == self.retries - 1:
                            self.err = FetchError("block %d: %r" % (i, e))
                            return
                        time.sleep(min(2 ** t, 20))
        finally:
            os.close(fd)

    def _hasher(self):
        i, fd = 0, os.open(self.path, os.O_RDONLY)
        try:
            while i < self.nblocks and self.err is None:
                if not self.done[i]:
                    time.sleep(0.05)
                    continue
                self.md5.update(os.pread(fd, BLOCK, i * BLOCK))
                i += 1
        finally:
            os.close(fd)

    def run(self):
        with open(self.path, "wb") as f:
            f.truncate(self.size)
        ts = [threading.Thread(target=self._worker, daemon=True) for _ in range(min(self.streams, self.nblocks))]
        h = threading.Thread(target=self._hasher, daemon=True) if self.md5 else None
        for t in ts + ([h] if h else []):
            t.start()
        for t in ts + ([h] if h else []):
            t.join()
        if self.err:
            raise self.err
        return self.md5.hexdigest() if self.md5 else None


def fetch_run(acc, dest_dir, streams, log, limiter=None, allow_lite=False):
    """Download one run to dest_dir (node-local). -> (path, info). Falls back from Lite to the full .sra on any failure."""
    last = None
    for loc in sdl_locations(acc, allow_lite):
        path = os.path.join(dest_dir, acc + (".sralite" if loc["kind"] == "lite" else ".sra"))
        t0 = time.time()
        try:
            got = RangeFetcher(loc["url"], loc["size"], path, streams, md5=loc.get("md5"), limiter=limiter).run()
            if loc.get("md5") and got != loc["md5"]:
                raise FetchError("md5_mismatch %s != %s" % (got, loc["md5"]))
            if os.path.getsize(path) != loc["size"]:
                raise FetchError("size_mismatch")
            dt = time.time() - t0
            log("[fetch] %s %s %.2f GB in %.1f s (%.0f MB/s)" % (acc, loc["kind"], loc["size"] / 1e9, dt, loc["size"] / dt / 1e6))
            return path, {"run": acc, "kind": loc["kind"], "bytes": loc["size"], "md5": loc.get("md5"), "fetch_s": round(dt, 1)}
        except (FetchError, OSError) as e:
            last = e
            log("[fetch] %s %s failed: %s" % (acc, loc["kind"], e))
            try:
                os.remove(path)
            except OSError:
                pass
    raise FetchError("all_sources_failed: %r" % last)


def decode_run(binary, sra, out, quality, vdb_threads, env, log_path):
    cmd = [binary, "download", sra, "--vdb-threads", str(vdb_threads), "--no-profile", "-o", out]
    if quality and quality != "none":
        cmd += ["--quality", quality]
    t0 = time.time()
    with open(log_path, "a") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return rc, time.time() - t0


def pairs_in(binary, onefq, env):
    """Read-pair count from the .1fq footer (`singlet decode --header`, as stages.read_header_reads)."""
    try:
        out = subprocess.run([binary, "decode", onefq, "--header"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                             timeout=300).stdout
        m = re.search(r"Reads: (\d+)", out)
        return int(m.group(1)) if m else 0
    except Exception:  # noqa: BLE001
        return 0


def process_item(a, stage, gsm, claim, item, env, binary, log):
    worker = "%s-%d" % (os.uname().nodename.split(".")[0], os.getpid())
    inc = Q.new_incoming(stage, gsm, worker)
    scr = os.path.join(a.scratch, "s9-fetch-%s-%d" % (gsm, os.getpid()))
    shutil.rmtree(scr, ignore_errors=True)
    os.makedirs(scr)
    t_start = time.time()
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN) if resource else None
    runs_info, raw_total = [], 0
    try:
        for r in item["runs"]:
            Q.heartbeat(claim)
            if a.limiter:
                a.limiter.set_active(True)
            try:
                sra, info = fetch_run(r, scr, a.streams, log, a.limiter, a.allow_lite)
            finally:
                if a.limiter:
                    a.limiter.set_active(False)  # decoding: this worker's share goes to the others
            Q.heartbeat(claim)
            out = os.path.join(inc, r + ".1fq")
            rc, dsec = decode_run(binary, sra, out, a.quality, a.decode_threads, env, os.path.join(scr, "decode.log"))
            os.remove(sra)
            if rc not in (0, 2):  # 2 = the download's non-10x gate: a verdict, not an error; the run is simply not staged
                raise FetchError("decode_failed rc=%s %s" % (rc, open(os.path.join(scr, "decode.log")).read()[-400:]))
            info.update(decode_s=round(dsec, 1), gated=(rc == 2))
            if rc == 0:
                info.update(staged_bytes=os.path.getsize(out), pairs=pairs_in(binary, out, env))
                raw_total += info["bytes"]
            runs_info.append(info)
        kept = [x["run"] for x in runs_info if not x["gated"]]
        if not kept:
            Q.give_back_todo(stage, gsm, claim, retry=False)
            return "parked_non10x"
        if a.screen:
            rc = subprocess.run([binary, "screen-runs", "--json", os.path.join(inc, "screen.json")] +
                                [os.path.join(inc, r + ".1fq") for r in kept], env=env, capture_output=True).returncode
            if rc == 6:
                Q.give_back_todo(stage, gsm, claim, retry=False)
                return "parked_no_gex"
        pairs = sum(x.get("pairs", 0) for x in runs_info)
        ru1 = resource.getrusage(resource.RUSAGE_CHILDREN) if resource else None
        child_cpu = ((ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)) if resource else 0.0  # decode + screen (the fetch threads are ours, small)
        nbytes = Q.publish(stage, gsm, inc, claim, {
            "gse": item.get("gse"), "organism": item.get("organism"), "runs": kept, "run_info": runs_info, "pairs": pairs,
            "raw_bytes": raw_total, "decode_cpu_s": round(child_cpu, 1),
            "decode_core_s_per_Mpair": round(child_cpu / (pairs / 1e6), 2) if pairs else None, "fetch_total_s": round(time.time() - t_start, 1), "quality": a.quality,
            "bytes_per_pair": round(sum(x.get("staged_bytes", 0) for x in runs_info) / pairs, 2) if pairs else None,
            "stage_ratio": round(sum(x.get("staged_bytes", 0) for x in runs_info) / raw_total, 3) if raw_total else None})
        log("[stage] %s ready: %.2f GB staged from %.2f GB raw, %d pairs, %.1f B/pair, decode+screen %.2f core-s/M, %.0f s total" % (
            gsm, nbytes / 1e9, raw_total / 1e9, pairs, nbytes / max(pairs, 1), child_cpu / max(pairs / 1e6, 1e-9), time.time() - t_start))
        return "ready"
    except Preempted:
        shutil.rmtree(inc, ignore_errors=True)
        Q.give_back_todo(stage, gsm, claim, retry=True)  # not a failure: the item goes back untouched, attempts not counted
        log("[fetch] %s preempted: item returned to todo" % gsm)
        raise
    except Exception as e:  # noqa: BLE001
        shutil.rmtree(inc, ignore_errors=True)
        log("[fetch] %s FAILED: %r" % (gsm, e))
        n = item.get("fetch_attempts", 0) + 1
        item["fetch_attempts"] = n
        json.dump(item, open(os.path.join(claim, "item.json"), "w"))
        Q.give_back_todo(stage, gsm, claim, retry=n < 3)
        return "failed"
    finally:
        shutil.rmtree(scr, ignore_errors=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--cap-tb", type=float, default=20.0, help="stop fetching while staged + incoming bytes are over this")
    ap.add_argument("--room-need-gb", type=float, default=30.0, help="free space required below the cap before taking a new item")
    ap.add_argument("--streams", type=int, default=64, help="ranged connections per run (Phase 1: ~64 saturates one 10G node)")
    ap.add_argument("--quality", default="binned4", help="`singlet download --quality` (none|binned4|full)")
    ap.add_argument("--decode-threads", type=int, default=2)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--config", help="campaign config.json: supplies the pinned binary and whitelist dir of --cluster")
    ap.add_argument("--cluster", default="clipper")
    ap.add_argument("--binary", default=os.environ.get("SINGLET_BIN", "singlet"))
    ap.add_argument("--whitelist-dir", default=os.environ.get("SINGLET_WHITELIST_DIR"))
    ap.add_argument("--allow-lite", action="store_true", help="also accept SRA Lite (simplified qualities) when no full .sra is reachable")
    ap.add_argument("--screen", action="store_true", default=True)
    ap.add_argument("--idle-exit-min", type=float, default=30.0)
    ap.add_argument("--max-samples", type=int, default=0)
    a = ap.parse_args()
    if a.config:
        import common as C
        cfg = C.load_config(a.config)
        ccfg = C.cluster_cfg(cfg, a.cluster)
        a.binary = ccfg.get("binary") or cfg["binary"]["path"]
        a.whitelist_dir = a.whitelist_dir or ccfg.get("whitelist_dir")
    env = dict(os.environ)
    if a.whitelist_dir:
        env["SINGLET_WHITELIST_DIR"] = a.whitelist_dir

    def log(m):
        print("%s %s" % (time.strftime("%H:%M:%S"), m), flush=True)

    Q.init(a.stage)
    Q.sweep_scratch(a.scratch, "s9-fetch-", log)
    a.limiter = ratecap.SharedRate(a.stage, "%s-%d" % (os.uname().nodename.split(".")[0], os.getpid()))
    log("[fetch] global rate cap %.0f MB/s (stage.json rate_cap_MBps), shared by all fetch workers" % a.limiter.cap)

    Q.install_sigterm()
    cap, idle_since, done = int(a.cap_tb * 1e12), time.time(), 0
    while True:
        Q.reap(a.stage, log=log)
        Q.wait_for_room(a.stage, cap, log=log, need=int(a.room_need_gb * 1e9))
        c = Q.claim_todo(a.stage)
        if not c:
            if time.time() - idle_since > a.idle_exit_min * 60:
                log("[fetch] no todo for %.0f min: exit" % a.idle_exit_min)
                return 0
            time.sleep(20)
            continue
        idle_since = time.time()
        gsm, claim, item = c
        log("[fetch] %s: %s" % (gsm, process_item(a, a.stage, gsm, claim, item, env, a.binary, log)))
        done += 1
        if a.max_samples and done >= a.max_samples:
            return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Preempted:
        print("[fetch] SIGTERM: exiting (claim already returned)", flush=True)
        sys.exit(143)
