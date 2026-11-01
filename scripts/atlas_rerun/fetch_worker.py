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
import companions as CP  # noqa: E402
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


PROTO_LINE = re.compile(r"\[singlet-download\] Protocol: (\S+) \(confidence: (\d+)\)")


def parse_protocol(text):
    """-> (tag, confidence) from the 'Protocol:' line `singlet download` prints, or (None, None)."""
    m = PROTO_LINE.search(text or "")
    return (m.group(1), int(m.group(2))) if m else (None, None)


GATE_LINE = re.compile(r"\[singlet-gate\] (\w+) .*?wl_match_rate=([0-9.eE+-]+)")


def parse_wl_rate(text):
    """-> the whitelist hit rate of the gate's protocol call ('[singlet-gate] PASS_10X|UNCERTAIN|REJECT ... wl_match_rate=0.9'), or None. A true lane of a 10x library has about the
    same rate as its sibling lanes (~0.9); a rate ~0.05 means another file layout or another library."""
    m = GATE_LINE.search(text or "")
    try:
        return float(m.group(2)) if m else None
    except ValueError:
        return None


def protocol_family(tag):
    """'10x-3p-v2' and '10x-5p-v2' are one family: same whitelist, same barcode / UMI geometry, they differ only in which end of the transcript read 2 starts at
    (a heuristic of the detector: poly(A) vs poly(T) at the start of the cDNA read)."""
    return re.sub(r"-(3p|5p)-", "-", tag or "")


def vote_protocol(calls, weights):
    """One protocol per sample. calls: {run: tag}; weights: {run: reads}. -> dict(status, winner, tags, family, weight):
      agree        every run has the same tag: nothing to do;
      voted        the tags differ only in the 3'/5' flag (one family): the tag with the most reads wins (ties: the alphabetically first), and every run is decoded with it;
      incompatible the runs differ in whitelist or geometry (different families): genuinely different libraries under one GSM, never pooled by guessing: winner None;
      none         no run reported a tag."""
    tags = {r: t for r, t in calls.items() if t}
    if not tags:
        return {"status": "none", "winner": None, "tags": {}, "family": None, "weight": {}}
    fams = {protocol_family(t) for t in tags.values()}
    w = {}
    for r, t in tags.items():
        w[t] = w.get(t, 0) + max(int(weights.get(r, 0) or 0), 1)
    if len(set(tags.values())) == 1:
        return {"status": "agree", "winner": next(iter(tags.values())), "tags": tags, "family": next(iter(fams)), "weight": w}
    if len(fams) == 1:
        return {"status": "voted", "winner": sorted(w, key=lambda t: (-w[t], t))[0], "tags": tags, "family": next(iter(fams)), "weight": w}
    return {"status": "incompatible", "winner": None, "tags": tags, "family": None, "weight": w}


def catalogue_chemistry(protocol_inferred):
    """-> (end, version) from the catalogue's `protocol_inferred` ('10xv3' -> ('3p', 'v3'), '10xv3_5prime' -> ('5p', 'v3'), 'citeseq' -> ('3p', None)); (None, None) if it says nothing."""
    p = (protocol_inferred or "").lower()
    if not p:
        return None, None
    m = re.search(r"v(\d)", p)
    ver = "v" + m.group(1) if m else None
    if "5prime" in p or "5p" in p:
        return "5p", ver
    if p.startswith("10xv") or p == "citeseq":
        return "3p", ver
    return None, None


def check_against_catalogue(tag, protocol_inferred):
    """Cross-check a sample's protocol call (the vote's winner, or the one run's call) with the catalogue's guess. -> dict(tag, catalogue, disagrees) or None when either side
    says nothing. The call decides; a disagreement is only recorded and flagged."""
    end, ver = catalogue_chemistry(protocol_inferred)
    m = re.match(r"10x-(3p|5p)-(v\d)$", tag or "")
    if not m or (end is None and ver is None):
        return None
    bad = (end is not None and end != m.group(1)) or (ver is not None and ver != m.group(2))
    return {"tag": tag, "catalogue": protocol_inferred, "disagrees": bool(bad)}


MIN_VOTE_CONFIDENCE = 2   # Confidence::MEDIUM: a LOW / NONE call (whitelist hit rate ~0, e.g. 'dropseq' or '10x-3p-v1' at 5%) is a failed detection of a run, not a different library


EXCLUDE_BELOW = 0.5        # a run without a confident call follows the sample only if its whitelist rate is at least this fraction of the main group's
COMPANION_READ_FRAC = 0.10  # an excluded run holding at least this fraction of the sample's reads is also listed in companion_candidates


def vote_runs(runs, calls, run_reads):
    """The sample-level protocol decision for a multi-run sample. calls: {run: (tag, confidence[, whitelist rate])}; run_reads: reads per run in `runs` order (may be empty).
    -> (main_runs, vote, other_groups). Only calls at MEDIUM or better take part in the vote. They are grouped by protocol FAMILY (same whitelist and geometry; the 3p/5p flag is not
    part of it); the group with the most reads is the GEX sample. A run WITHOUT a confident call follows it (`low_confidence_runs`, decoded with the sample's protocol) only when its
    whitelist rate is at least half the main group's: a true lane of a 10x library hits the whitelist about as often as its sibling lanes (~0.9), a run at ~0.05 is another file layout
    (barcode read missing or truncated, index reads as R1, ...) or another library, and decoding it with the sample's protocol would drop 95% of its reads or add mismatched barcodes.
    Those runs are EXCLUDED (`vote["excluded_runs"]`: run, call, whitelist rate, reads) and, when they hold >= 10% of the sample's reads, also returned as `other_groups`. Other
    families with a confident call are different libraries under one GSM (a feature library next to the GEX, for instance): not decoded here, returned as `other_groups`
    (companion_candidates in the manifest). A run whose whitelist rate is unknown (no gate line) follows. Runs excluded or in another family are never in `main_runs`."""
    nreads = {r: int(n or 0) for r, n in zip(runs, run_reads or [])}
    w = {r: max(n, 1) for r, n in nreads.items()}
    conf = {r: calls[r] for r in runs if calls.get(r) and calls[r][0] and (calls[r][1] or 0) >= MIN_VOTE_CONFIDENCE}
    tags = {r: c[0] for r, c in conf.items()}
    wl = lambda r: calls[r][2] if calls.get(r) and len(calls[r]) > 2 else None
    groups = {}
    for r, t in tags.items():
        groups.setdefault(protocol_family(t), []).append(r)
    # The GEX sample is the family that is a KNOWN chemistry (a guessed 'agnostic-...' / UNKNOWN structure is a last resort: GSM9015940's 205 M reads of a 6+7 nt feature library, whitelist
    # rate 0.40, outweighed the 123 M reads of its 10x v2 GEX run, whitelist rate 0.97, and the sample was parked as having no GEX) and, among those, the one with the most reads that
    # hit the whitelist (reads x whitelist rate; an unknown rate counts as 1).
    known = lambda f: not (f.startswith("agnostic") or f == "UNKNOWN")
    score = lambda f: sum(w.get(r, 1) * (wl(r) if wl(r) is not None else 1.0) for r in groups[f])
    main_fam = max(groups, key=lambda f: (known(f), score(f), f)) if groups else None
    main_conf = [r for r in groups.get(main_fam, [])]
    rates = [(wl(r), w.get(r, 1)) for r in main_conf if wl(r) is not None]
    main_wl = sum(x * y for x, y in rates) / sum(y for _, y in rates) if rates else None
    followers, excluded = [], []
    for r in runs:
        if r in tags:
            continue
        rate = wl(r)
        if tags and rate is not None and main_wl is not None and rate < EXCLUDE_BELOW * main_wl:
            excluded.append({"run": r, "tag": calls[r][0] if calls.get(r) else None, "confidence": calls[r][1] if calls.get(r) else None, "wl_match_rate": rate,
                             "main_wl_match_rate": round(main_wl, 3), "reads": nreads.get(r, 0), "reason": "no confident call and a whitelist rate below half the main group's"})
        else:
            followers.append(r)
    main = [r for r in runs if (r in groups.get(main_fam, [])) or r in followers] if tags else list(runs)
    vote = vote_protocol({r: tags[r] for r in main if r in tags}, w)
    others = []
    for f in sorted(groups):
        if f == main_fam:
            continue
        rs = [r for r in runs if r in groups[f]]
        others.append({"lib_gsm": None, "lib_type": None, "status": "unclassified", "protocol_family": f, "protocol_tags": {r: tags[r] for r in rs}, "runs": rs,
                       "reads": sum(nreads.get(r, 0) for r in rs)})
    total = sum(nreads.values())
    for e in excluded:
        if total and e["reads"] >= COMPANION_READ_FRAC * total:
            others.append({"lib_gsm": None, "lib_type": None, "status": "unclassified", "protocol_family": None, "protocol_tags": {e["run"]: e["tag"]}, "runs": [e["run"]],
                           "reads": e["reads"], "reason": e["reason"], "wl_match_rate": e["wl_match_rate"]})
    vote["incompatible"] = bool(others)
    vote["low_confidence_runs"] = [r for r in followers] if tags else []
    vote["excluded_runs"] = excluded
    if others:
        vote["main_family"] = main_fam
        vote["other_runs"] = [r for o in others for r in o["runs"]]
    vote["calls"] = {r: list(calls[r]) for r in runs if calls.get(r)}
    return main, vote, others


def probe_run_protocol(binary, run, scr, env, whitelist_dir=None, spots=5000, vdb_threads=2, timeout=600):
    """The first `spots` spots of a run through `singlet download` (its gate and protocol detection probe 5000 spots) without fetching the run: -> (tag, confidence, wl_match_rate) or (None, None, None).
    Used for the sample vote when the head screen is off."""
    out = os.path.join(scr, run + ".probe.1fq")
    try:
        pr = subprocess.run([binary, "download", run, "--max-reads", str(spots), "--vdb-threads", str(vdb_threads), "--no-profile", "-o", out] +
                            (["--whitelist-dir", whitelist_dir] if whitelist_dir else []), env=env, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, None, None
    finally:
        try:
            os.remove(out)
        except OSError:
            pass
    txt = pr.stderr.decode("utf-8", "replace")
    return parse_protocol(txt) + (parse_wl_rate(txt),)


def decode_run(binary, sra, out, quality, vdb_threads, env, log_path, protocol=None):
    cmd = [binary, "download", sra, "--vdb-threads", str(vdb_threads), "--no-profile", "-o", out]
    if protocol:   # the sample-level vote (vote_protocol): decode this run with the sample's protocol, not its own call
        cmd += ["--protocol", protocol]
    if quality and quality != "none":
        cmd += ["--quality", quality]
    t0 = time.time()
    with open(log_path, "a") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return rc, time.time() - t0


def head_screen(binary, run, scr, env, reads=1_000_000, vdb_threads=2, timeout=600, keep_json=None, info=None, whitelist_dir=None):
    """Screen the first `reads` pairs of a run BEFORE fetching it: VDB-streams just that head (`singlet download --max-reads`), then
    `screen-runs`. -> ('keep'|'non10x'|'no_gex'|'unknown', seconds). 'unknown' (stream failed/timed out) means: fetch and let the full-run
    screen decide. SRA files are columnar, so a byte range cannot give the head; this costs one short VDB stream, about the price
    of 1-2% of a typical run (docs/research-beta/download-and-staging.md §3)."""
    t0 = time.time()
    out = os.path.join(scr, run + ".head.1fq")
    try:
        pr = subprocess.run([binary, "download", run, "--max-reads", str(reads), "--vdb-threads", str(vdb_threads), "--no-profile", "-o", out] +
                            (["--whitelist-dir", whitelist_dir] if whitelist_dir else []), env=env, capture_output=True, timeout=timeout)
        rc = pr.returncode
        if info is not None:
            txt = pr.stderr.decode("utf-8", "replace")
            info["protocol"], info["confidence"] = parse_protocol(txt)
            info["wl"] = parse_wl_rate(txt)
    except subprocess.TimeoutExpired:
        return "unknown", time.time() - t0
    if rc == 2:
        return "non10x", time.time() - t0
    if rc != 0 or not os.path.exists(out):
        return "unknown", time.time() - t0
    rc = subprocess.run([binary, "screen-runs", "--json", out + ".json", out], env=env, capture_output=True).returncode
    if keep_json and rc == 0 and os.path.exists(out + ".json"):
        shutil.copyfile(out + ".json", keep_json)   # the .1fg2 route has no full .1fq to screen: this 1 M-pair screen is the run's screen.json
    for f in (out, out + ".json"):
        try:
            os.remove(f)
        except OSError:
            pass
    return ("no_gex" if rc == 6 else "keep" if rc == 0 else "unknown"), time.time() - t0


def stage_run(binary, sra, out, quality, vdb_threads, whitelist_dir, env, scr, part=False, protocol=None):
    """`singlet download <local .sra> --stage-out OUT`: SRA -> .1fg2 in one pass (no .1fq, no barcode sort). -> (rc, seconds, report dict|None).
    rc 2 = the non-10x gate (a verdict); rc 6 = protocol not stageable (no universal R1 whitelist): the caller falls back to the .1fq path."""
    cmd = [binary, "download", sra, "--quality", quality, "--vdb-threads", str(vdb_threads), "--no-profile", "--stage-out", out,
           "-o", os.path.join(scr, "unused.1fq")] + (["--stage-part"] if part else []) + (["--whitelist-dir", whitelist_dir] if whitelist_dir else []) + (["--protocol", protocol] if protocol else [])
    t0 = time.time()
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    rep = None
    if p.returncode == 0 and part:
        import struct
        with open(out, "rb") as fh:   # PartHdr (src/stage/stage_encoder.h): n_stored, n_in at byte 16
            n_stored, n_in = struct.unpack("<QQ", fh.read(32)[16:32])
        rep = {"n_reads": n_stored, "n_in": n_in}
    elif p.returncode == 0:
        try:
            rep = json.loads(p.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            rep = None
    elif p.returncode not in (2, 6):
        with open(os.path.join(scr, "decode.log"), "a") as fh:
            fh.write(p.stderr[-2000:])
    return p.returncode, time.time() - t0, rep


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
    fmt, staged_files, parts, fallback_1fq = "1fq", [], [], False
    protocol_vote, force_protocol, companion_candidates = None, None, []
    calls = {}
    try:
        if a.head_screen:
            verdicts = {}
            for r in item["runs"]:
                hinfo = {}
                verdicts[r], hs = head_screen(binary, r, scr, env, keep_json=os.path.join(scr, "head_screen.json") if a.format == "1fg2" else None, info=hinfo,
                                              whitelist_dir=a.whitelist_dir)
                calls[r] = (hinfo.get("protocol"), hinfo.get("confidence"), hinfo.get("wl"))
                Q.heartbeat(claim)
                log("[head] %s %s %.0f s" % (r, verdicts[r], hs))
            keep_runs = [r for r in item["runs"] if verdicts[r] in ("keep", "unknown")]
            if not keep_runs:
                Q.give_back_todo(stage, gsm, claim, retry=False)
                return "parked_head_" + ("non10x" if "non10x" in verdicts.values() else "no_gex")
            item = dict(item, runs=keep_runs, run_reads=[n for r, n in zip(item["runs"], item.get("run_reads") or []) if r in keep_runs])
        elif len(item["runs"]) > 1:   # no head screen: the vote still needs each run's call, from a 5000-spot probe
            for r in item["runs"]:
                calls[r] = probe_run_protocol(binary, r, scr, env, whitelist_dir=a.whitelist_dir)
                Q.heartbeat(claim)
        if len(item["runs"]) > 1:
            main, vote, others = vote_runs(item["runs"], calls, item.get("run_reads") or [])
            protocol_vote = vote
            if others or vote["excluded_runs"]:
                companion_candidates = others
                log("[vote] %s: runs %s: the GEX sample is %s; handed to the companion classifier: %s; excluded from pooling: %s" % (
                    gsm, vote["calls"], main, vote.get("other_runs", []), [e["run"] for e in vote["excluded_runs"]]))
                item = dict(item, runs=main)
            if vote["status"] == "voted":
                force_protocol = vote["winner"]
                log("[vote] %s: runs disagree on the 3'/5' flag %s -> every run decoded as %s" % (gsm, vote["tags"], force_protocol))
            elif vote["status"] == "agree" and vote["low_confidence_runs"]:
                force_protocol = vote["winner"]   # the runs the detector could not call are lanes of the same library: decode them as the others
                log("[vote] %s: runs %s have no confident call -> decoded as %s like the rest" % (gsm, vote["low_confidence_runs"], force_protocol))
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
            rc, dsec, staged = None, 0.0, False
            if a.format == "1fg2" and not fallback_1fq:
                # SRA -> .1fg2 directly. A single-run sample goes straight to sample.1fg2; the runs of a multi-run sample become parts that are merged below into ONE file
                # (one exact-barcode histogram, one distinct-sequence table: the pooled R11 correction of the .1fq route)
                multi = len(item["runs"]) > 1
                g = os.path.join(scr, "part_" + r) if multi else os.path.join(inc, "sample.1fg2")
                rc, dsec, rep = stage_run(binary, sra, g, a.quality, a.decode_threads, a.whitelist_dir, env, scr, part=multi, protocol=force_protocol)
                if rc == 0 and rep:
                    staged, fmt = True, "1fg2"
                    info.update(pairs=rep["n_reads"])
                    if multi:
                        parts.append(g)
                    else:
                        staged_files = ["sample.1fg2"]
                        info.update(staged_bytes=os.path.getsize(g), n_unique=rep["n_unique"], max_len=rep["max_len"])
                elif rc == 0:
                    raise FetchError("stage encoder gave no report for %s" % r)
                elif rc == 6:
                    if parts:
                        raise FetchError("runs of one sample disagree on stageability (protocol): %s" % r)
                    log("[stage] %s: protocol not stageable, .1fq route" % r)
                    fallback_1fq, rc = True, None
                    try:
                        os.remove(g)
                    except OSError:
                        pass
            if rc is None:
                rc, dsec = decode_run(binary, sra, out, a.quality, a.decode_threads, env, os.path.join(scr, "decode.log"), protocol=force_protocol)
            os.remove(sra)
            if rc not in (0, 2):  # 2 = the download's non-10x gate: a verdict, not an error; the run is simply not staged
                raise FetchError("decode_failed rc=%s %s" % (rc, open(os.path.join(scr, "decode.log")).read()[-400:]))
            info.update(decode_s=round(dsec, 1), gated=(rc == 2))
            if rc == 0:
                if not staged:
                    info.update(staged_bytes=os.path.getsize(out), pairs=pairs_in(binary, out, env))
                raw_total += info["bytes"]
            runs_info.append(info)
        if parts:   # multi-run sample, parts -> one pooled .1fg2
            g = os.path.join(inc, "sample.1fg2")
            t_m = time.time()
            pm = subprocess.run([a.stage_bin or os.path.join(os.path.dirname(binary), "singlet-stage"), "--whitelist-dir", a.whitelist_dir or os.environ.get("SINGLET_WHITELIST_DIR", ""),
                                 "--scratch", scr, "-o", g, "--merge-parts"] + parts, env=env, capture_output=True, text=True)
            if pm.returncode != 0:
                raise FetchError("merge of the run parts failed rc=%s %s" % (pm.returncode, pm.stderr[-300:]))
            rep = json.loads(pm.stdout.strip().splitlines()[-1])
            staged_files = ["sample.1fg2"]
            for x in runs_info:
                if not x["gated"]:
                    x.update(staged_bytes=0, merge_s=round(time.time() - t_m, 1))
            first = next(x for x in runs_info if not x["gated"])
            first.update(staged_bytes=os.path.getsize(g), n_unique=rep["n_unique"], max_len=rep["max_len"])
            for pth in parts:
                os.remove(pth)
        kept = [x["run"] for x in runs_info if not x["gated"]]
        if not kept:
            Q.give_back_todo(stage, gsm, claim, retry=False)
            return "parked_non10x"
        if fmt == "1fg2":
            hj = os.path.join(scr, "head_screen.json")
            if os.path.exists(hj):
                shutil.copyfile(hj, os.path.join(inc, "screen.json"))
        elif a.screen:
            rc = subprocess.run([binary, "screen-runs", "--json", os.path.join(inc, "screen.json")] +
                                [os.path.join(inc, r + ".1fq") for r in kept], env=env, capture_output=True).returncode
            if rc == 6:
                Q.give_back_todo(stage, gsm, claim, retry=False)
                return "parked_no_gex"
        pairs = sum(x.get("pairs", 0) for x in runs_info)
        ru1 = resource.getrusage(resource.RUSAGE_CHILDREN) if resource else None
        child_cpu = ((ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)) if resource else 0.0  # decode + screen (the fetch threads are ours, small)
        comp = []
        if getattr(a, "companion_table", None):   # companion libraries (ADT/HTO/CRISPR/V(D)J): a failure is recorded, never fails the GEX sample
            comp = CP.fetch_all(a.companion_table, gsm, inc, scr, fetch_run, env, log, heartbeat=lambda: Q.heartbeat(claim), threads=a.decode_threads,
                                streams=a.streams, limiter=a.limiter, allow_lite=a.allow_lite)
        man = {
            "gse": item.get("gse"), "organism": item.get("organism"), "runs": kept, "run_info": runs_info, "pairs": pairs,
            "format": fmt, "files": staged_files,
            "raw_bytes": raw_total, "decode_cpu_s": round(child_cpu, 1),
            "decode_core_s_per_Mpair": round(child_cpu / (pairs / 1e6), 2) if pairs else None, "fetch_total_s": round(time.time() - t_start, 1), "quality": a.quality,
            "bytes_per_pair": round(sum(x.get("staged_bytes", 0) for x in runs_info) / pairs, 2) if pairs else None,
            "stage_ratio": round(sum(x.get("staged_bytes", 0) for x in runs_info) / raw_total, 3) if raw_total else None}
        if protocol_vote:   # absent for single-run samples and without --head-screen: the default manifest is unchanged
            man["protocol_vote"] = protocol_vote
            if companion_candidates:   # runs of another protocol family under this GSM: not staged here; the interface to the companion classification
                man["companion_candidates"] = companion_candidates
            if protocol_vote.get("excluded_runs"):   # runs with no confident call and a whitelist rate far below the siblings': not pooled, not fetched
                man["excluded_runs"] = protocol_vote["excluded_runs"]
        if calls:           # catalogue cross-check of the sample's call (the vote's winner, else the single kept run's call); only when the item carries protocol_inferred
            call = (protocol_vote or {}).get("winner") or next((calls[r][0] for r in kept if calls.get(r) and calls[r][0]), None)
            chk = check_against_catalogue(call, item.get("protocol_inferred"))
            if chk:
                man["protocol_check"] = chk
                if chk["disagrees"]:
                    log("[vote] %s: FLAG protocol %s disagrees with the catalogue (%s)" % (gsm, chk["tag"], chk["catalogue"]))
        if getattr(a, "companion_table", None):   # absent otherwise: the default manifest is unchanged
            man.update(companions=comp, companion_pairs=CP.pairs_total(comp))
        nbytes = Q.publish(stage, gsm, inc, claim, man)
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
    ap.add_argument("--quality", default="binned", help="`singlet download --quality` (none|binned4|full)")
    ap.add_argument("--decode-threads", type=int, default=2)
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--config", help="campaign config.json: supplies the pinned binary and whitelist dir of --cluster")
    ap.add_argument("--cluster", default="clipper")
    ap.add_argument("--binary", default=os.environ.get("SINGLET_BIN", "singlet"))
    ap.add_argument("--whitelist-dir", default=os.environ.get("SINGLET_WHITELIST_DIR"))
    ap.add_argument("--head-screen", action="store_true",
                    help="screen 1 M pairs of every run by a short VDB stream BEFORE downloading it (drops non-10x / non-GEX runs at ~no cost)")
    ap.add_argument("--format", choices=("1fq", "1fg2"), default="1fq",
                    help="staged form: 1fq (barcode-sorted .1fq per run) or 1fg2 (src/stage/fmt2.h, single-run samples straight from the SRA; others fall back to 1fq)")
    ap.add_argument("--stage-bin", help="singlet-stage binary for merging the parts of a multi-run sample (default: next to --binary)")
    ap.add_argument("--companions", help="companions.tsv (companion_libs.py): also fetch the ADT/HTO/CRISPR/V(D)J libraries of each sample as FASTQ into its ready dir")
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
    a.companion_table = CP.load_table(a.companions) if a.companions else None
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
