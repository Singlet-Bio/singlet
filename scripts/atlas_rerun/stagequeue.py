"""Bounded on-disk stage with an atomic claim queue (S9). Standard library only; every state change is one rename().

    <stage>/todo/<GSM>.json         work still to fetch (written by the planner/launcher, claimed by fetch workers)
    <stage>/fetching/<GSM>.<w>/     a fetch worker's claim on a todo item (the JSON moved in), heartbeat file HB
    <stage>/incoming/<GSM>.<w>/     being written; never visible to drain workers
    <stage>/ready/<GSM>/            complete, immutable: manifest.json + the staged reads (sample.1fq or run .1fq files)
    <stage>/claimed/<GSM>.<w>/      a drain worker owns it (rename from ready/), heartbeat file HB
    <stage>/failed/<GSM>.<n>/       gave up after MAX_ATTEMPTS claims; kept for a human
    <stage>/stats/                  bytes accounting is the sum of manifest.json["bytes"], never a du walk

Guarantees (rename within one directory tree is atomic on the cluster's NFS/Lustre project space):
  * exactly one worker wins a claim (the loser's rename raises FileNotFoundError and tries the next item);
  * a sample reaches ready/ only after its last byte is written and fsynced: a crash mid-write leaves an incoming/ dir that
    reap() deletes; a crash mid-drain leaves a claimed/ dir with a stale heartbeat that reap() returns to ready/ (attempt+1);
  * a finished sample is removed with finish(): rename claimed/ -> deleting/ first, then rmtree, so a crash never leaves a
    half-deleted dir that looks claimable;
  * no sample is processed twice: after finish() the only record is the ledger event; a reaped claim can only repeat work
    that did not finish, and only after HB_STALE_S without a heartbeat.
"""

import json
import os
import shutil
import socket
import time

HB_STALE_S = int(os.environ.get("S9_HB_STALE_S", 300))  # tests shorten it
MAX_ATTEMPTS = 3
class Preempted(BaseException):
    """Raised by the SIGTERM handler of a worker (Slurm preemption or time limit). BaseException so no `except Exception` swallows it."""


def install_sigterm():
    import signal

    def _term(*_):
        raise Preempted()
    signal.signal(signal.SIGTERM, _term)


DIRS = ("todo", "fetching", "incoming", "ready", "claimed", "failed", "deleting")


def _w():
    return "%s-%d" % (socket.gethostname().split(".")[0], os.getpid())


def init(stage):
    for d in DIRS:
        os.makedirs(os.path.join(stage, d), exist_ok=True)
    return stage


def _hb(path):
    with open(os.path.join(path, "HB"), "w") as f:
        f.write(str(time.time()))


def heartbeat(path):
    """Touch the claim's heartbeat; workers call this at least every few minutes."""
    try:
        _hb(path)
    except OSError:
        pass


def _fsync_tree(path):
    for r, _, names in os.walk(path):
        for n in names:
            fd = os.open(os.path.join(r, n), os.O_RDONLY)
            try:
                os.fsync(fd)
            except OSError:  # read-only fsync is refused on some platforms (Windows test runs); Linux accepts it
                pass
            finally:
                os.close(fd)


# ── fetch side ──────────────────────────────────────────────────────────────────────────────────────────────────────

def claim_todo(stage, worker=None):
    """Claim one todo item -> (gsm, claim_dir, item dict) or None. Item order = lexical order of file names (the planner names
    them so)."""
    worker = worker or _w()
    td = os.path.join(stage, "todo")
    for name in sorted(os.listdir(td)):
        if not name.endswith(".json"):
            continue
        gsm = name[:-5]
        cdir = os.path.join(stage, "fetching", "%s.%s" % (gsm, worker))
        os.makedirs(cdir, exist_ok=True)
        try:
            os.rename(os.path.join(td, name), os.path.join(cdir, "item.json"))
        except FileNotFoundError:
            shutil.rmtree(cdir, ignore_errors=True)
            continue
        _hb(cdir)
        return gsm, cdir, json.load(open(os.path.join(cdir, "item.json")))
    return None


def new_incoming(stage, gsm, worker=None):
    d = os.path.join(stage, "incoming", "%s.%s" % (gsm, worker or _w()))
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def publish(stage, gsm, incoming, fetch_claim, manifest):
    """Finish a fetch: write manifest.json (with the byte total) into the incoming dir, fsync, rename into ready/."""
    nbytes = sum(os.path.getsize(os.path.join(r, n)) for r, _, ns in os.walk(incoming) for n in ns if n != "HB")
    manifest = dict(manifest, gsm=gsm, bytes=nbytes, attempts=0, staged_at=time.time())
    with open(os.path.join(incoming, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    _fsync_tree(incoming)
    os.rename(incoming, os.path.join(stage, "ready", gsm))
    shutil.rmtree(fetch_claim, ignore_errors=True)
    return nbytes


def give_back_todo(stage, gsm, fetch_claim, retry=True):
    """A fetch failed: return the item to todo (retry) or park it in failed/."""
    item = os.path.join(fetch_claim, "item.json")
    if retry and os.path.exists(item):
        os.rename(item, os.path.join(stage, "todo", gsm + ".json"))
    else:
        os.rename(fetch_claim, os.path.join(stage, "failed", "%s.fetch" % gsm))
        return
    shutil.rmtree(fetch_claim, ignore_errors=True)


def staged_bytes(stage):
    """Bytes held by ready/ + claimed/ (manifest totals) plus what the incoming/ files have so far (stat only)."""
    total = 0
    for d in ("ready", "claimed"):
        base = os.path.join(stage, d)
        for n in os.listdir(base):
            try:
                total += json.load(open(os.path.join(base, n, "manifest.json")))["bytes"]
            except (OSError, ValueError, KeyError):
                pass
    base = os.path.join(stage, "incoming")
    for n in os.listdir(base):
        for r, _, ns in os.walk(os.path.join(base, n)):
            for x in ns:
                try:
                    total += os.path.getsize(os.path.join(r, x))
                except OSError:
                    pass
    return total


def wait_for_room(stage, cap_bytes, poll_s=30, log=print, need=0, max_wait_s=None):
    """Backpressure: block until staged + incoming bytes (+ the `need` the next item will add) are under the cap.
    Returns False if `max_wait_s` ran out first."""
    warned, t0 = False, time.time()
    while staged_bytes(stage) + need > cap_bytes:
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            return False
        if not warned:
            log("[stage] over cap (%.2f TB of %.2f TB): fetch paused" % (staged_bytes(stage) / 1e12, cap_bytes / 1e12))
            warned = True
        time.sleep(poll_s)
    return True


# ── drain side ──────────────────────────────────────────────────────────────────────────────────────────────────────

def claim_ready(stage, worker=None, accept=None):
    """Claim one ready sample -> (gsm, claim_dir, manifest) or None. `accept(manifest)` may veto (e.g. CPU workers skip
    samples above a size). Oldest first so staged data never ages."""
    worker = worker or _w()
    rd = os.path.join(stage, "ready")
    names = []
    for n in os.listdir(rd):
        try:
            names.append((os.path.getmtime(os.path.join(rd, n)), n))
        except OSError:
            pass
    for _, gsm in sorted(names):
        try:
            man = json.load(open(os.path.join(rd, gsm, "manifest.json")))
        except (OSError, ValueError):
            continue
        if accept and not accept(man):
            continue
        cdir = os.path.join(stage, "claimed", "%s.%s" % (gsm, worker))
        try:
            os.rename(os.path.join(rd, gsm), cdir)
        except (FileNotFoundError, OSError):
            continue
        _hb(cdir)
        man["attempts"] = man.get("attempts", 0) + 1
        with open(os.path.join(cdir, "manifest.json"), "w") as f:
            json.dump(man, f)
        return gsm, cdir, man
    return None


def finish(stage, gsm, claim_dir):
    """Drained and uploaded: delete the staged reads (rename away first so a crash cannot leave a claimable half-dir)."""
    gone = os.path.join(stage, "deleting", os.path.basename(claim_dir))
    os.rename(claim_dir, gone)
    shutil.rmtree(gone, ignore_errors=True)


def release(stage, gsm, claim_dir, failed=False, refund=False):
    """Hand a claim back to ready/ (e.g. worker is shutting down) or to failed/. `refund`: the claim was lost to preemption, not to a
    failure, so it does not count as an attempt."""
    man = json.load(open(os.path.join(claim_dir, "manifest.json")))
    if refund:
        man["attempts"] = max(0, man.get("attempts", 1) - 1)
        with open(os.path.join(claim_dir, "manifest.json"), "w") as f:
            json.dump(man, f)
    dst = os.path.join(stage, "failed" if failed or man.get("attempts", 0) >= MAX_ATTEMPTS else "ready",
                       gsm if not failed and man.get("attempts", 0) < MAX_ATTEMPTS else "%s.%d" % (gsm, man.get("attempts", 0)))
    try:
        os.remove(os.path.join(claim_dir, "HB"))
    except OSError:
        pass
    os.rename(claim_dir, dst)


def reap(stage, now=None, log=print):
    """Crash recovery, safe to run from any worker or the launcher: stale fetching/incoming dirs are removed (the todo item
    goes back), stale claimed dirs return to ready/ or, after MAX_ATTEMPTS, to failed/. Returns counts."""
    now = now or time.time()
    n = {"fetch": 0, "incoming": 0, "claimed": 0, "failed": 0}

    def stale(path):
        try:
            return now - float(open(os.path.join(path, "HB")).read()) > HB_STALE_S
        except (OSError, ValueError):
            return now - os.stat(path).st_ctime > HB_STALE_S  # renamed in a moment ago, HB not yet written

    base = os.path.join(stage, "fetching")
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if stale(p) and os.path.exists(os.path.join(p, "item.json")):
            gsm = name.rsplit(".", 1)[0]
            try:
                os.rename(os.path.join(p, "item.json"), os.path.join(stage, "todo", gsm + ".json"))
            except OSError:
                continue
            shutil.rmtree(p, ignore_errors=True)
            n["fetch"] += 1
    base = os.path.join(stage, "incoming")
    live = {x.split(".", 1)[1] for x in os.listdir(os.path.join(stage, "fetching")) if "." in x}
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if name.split(".", 1)[-1] not in live and now - os.path.getmtime(p) > HB_STALE_S:
            shutil.rmtree(p, ignore_errors=True)
            n["incoming"] += 1
    base = os.path.join(stage, "claimed")
    for name in os.listdir(base):
        p = os.path.join(base, name)
        if not stale(p):
            continue
        gsm = name.rsplit(".", 1)[0]
        try:
            man = json.load(open(os.path.join(p, "manifest.json")))
        except (OSError, ValueError):
            continue
        bad = man.get("attempts", 1) >= MAX_ATTEMPTS
        dst = os.path.join(stage, "failed", "%s.%d" % (gsm, man.get("attempts", 1))) if bad else os.path.join(stage, "ready", gsm)
        try:
            os.rename(p, dst)
        except OSError:
            continue
        n["failed" if bad else "claimed"] += 1
        log("[stage] reaped stale claim %s -> %s" % (name, "failed" if bad else "ready"))
    return n


def backlog(stage):
    """-> dict(samples, pairs, bytes) of ready/ (what drain workers can start now) and the todo/fetching counts."""
    ready = {"samples": 0, "pairs": 0, "bytes": 0}
    rd = os.path.join(stage, "ready")
    for n in os.listdir(rd):
        try:
            m = json.load(open(os.path.join(rd, n, "manifest.json")))
        except (OSError, ValueError):
            continue
        ready["samples"] += 1
        ready["pairs"] += m.get("pairs", 0)
        ready["bytes"] += m.get("bytes", 0)
    ready["claimed"] = len(os.listdir(os.path.join(stage, "claimed")))
    ready["todo"] = len(os.listdir(os.path.join(stage, "todo")))
    ready["fetching"] = len(os.listdir(os.path.join(stage, "fetching")))
    return ready


# ── node-local scratch left by a killed worker ──────────────────────────────────────────────────────────────────────

def sweep_scratch(base, prefix, log=print):
    """A SIGKILLed worker cannot clean /tmp. Directories named `<prefix>-<tag>-<pid>` whose pid is gone on this host are removed.
    (Node-local /tmp: only the worker's own node can see them, so every worker sweeps at start-up.)"""
    n = 0
    try:
        names = os.listdir(base)
    except OSError:
        return 0
    for name in names:
        if not name.startswith(prefix):
            continue
        try:
            pid = int(name.rsplit("-", 1)[1])
            os.kill(pid, 0)
        except (ValueError, IndexError, ProcessLookupError):
            shutil.rmtree(os.path.join(base, name), ignore_errors=True)
            n += 1
        except OSError:  # EPERM: the pid exists
            pass
    if n:
        log("[stage] swept %d stale scratch dir(s) %s* in %s" % (n, prefix, base))
    return n


# ── operator CLI ────────────────────────────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="stage queue: status | enqueue | reap | drain-stub")
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("status", "reap"):
        sp.add_parser(name).add_argument("stage")
    e = sp.add_parser("enqueue", help="add todo items from a JSONL file ({gsm, gse, organism, runs[]} per line)")
    e.add_argument("stage")
    e.add_argument("jsonl")
    d = sp.add_parser("drain-stub", help="test aid: claim ready samples and delete them after --delay s (no compute)")
    d.add_argument("stage")
    d.add_argument("--delay", type=float, default=3.0)
    d.add_argument("--idle-exit-s", type=float, default=120)
    a = ap.parse_args(argv)
    init(a.stage)
    if a.cmd == "status":
        b = backlog(a.stage)
        b["staged_bytes"] = staged_bytes(a.stage)
        b["failed"] = len(os.listdir(os.path.join(a.stage, "failed")))
        print(json.dumps(b))
    elif a.cmd == "reap":
        print(json.dumps(reap(a.stage)))
    elif a.cmd == "enqueue":
        n = 0
        for line in open(a.jsonl):
            if line.strip():
                it = json.loads(line)
                tmp = os.path.join(a.stage, "todo", ".%s.tmp" % it["gsm"])
                json.dump(it, open(tmp, "w"))
                os.rename(tmp, os.path.join(a.stage, "todo", it["gsm"] + ".json"))
                n += 1
        print("enqueued", n)
    else:
        idle = time.time()
        while time.time() - idle < a.idle_exit_s:
            c = claim_ready(a.stage, "stub-%d" % os.getpid())
            if not c:
                time.sleep(2)
                continue
            idle = time.time()
            time.sleep(a.delay)
            finish(a.stage, c[0], c[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
