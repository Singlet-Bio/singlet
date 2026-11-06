#!/usr/bin/env python3
"""Co-located fetch tier inside a GPU job (bench/colo_fetch/README.md).

colo_fetch.py OUT D SCRATCH RUNLIST

D slots run side by side. Each slot loops over RUNLIST (lines: run url size reads, cycled from a shared counter):
ranged GET of the full .sra into SCRATCH with STREAMS threads and pwrite, size check, then
`nice -n NICE singlet download <sra> --quality binned` to a .1fq in SCRATCH, then both files are deleted.
This is the fetch tier's work (scripts/atlas_rerun/fetch_worker.py) without its queue.

Writes OUT/runs.jsonl (one line per run: done, failed or stopped) and OUT/net.tsv (time, cumulative bytes, every 2 s).
Stops when OUT/STOP exists: the transfer is abandoned and the decoder is killed. Exit 0.

Environment: COLO_DEC (decoder binary), COLO_STREAMS (8), COLO_SLOT_MBPS (per-slot rate cap, 100),
COLO_VDB_THREADS (2), COLO_NICE (19).
"""
import json, os, subprocess, sys, threading, time, urllib.request

OUT, D, SC, RL = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
DEC = os.environ.get("COLO_DEC", "/mnt/projects/debruinz_project/debruinz/singlet/jobs/s9/bin/singlet-cpu")
STREAMS = int(os.environ.get("COLO_STREAMS", "8"))
SLOT_BPS = float(os.environ.get("COLO_SLOT_MBPS", "100")) * 1e6
VT = os.environ.get("COLO_VDB_THREADS", "2")
NICE = os.environ.get("COLO_NICE", "19")
BLK = 8 << 20
os.makedirs(OUT, exist_ok=True); os.makedirs(SC, exist_ok=True)
RUNS = [l.split() for l in open(RL) if l.strip() and not l.startswith("#")]
LOCK = threading.Lock()
STATE = {"next": 0, "bytes": 0}
STOP = os.path.join(OUT, "STOP")


def stopped():
    return os.path.exists(STOP)


def take():
    with LOCK:
        n = STATE["next"]; STATE["next"] += 1
    return RUNS[n % len(RUNS)]


def record(d):
    with LOCK:
        with open(os.path.join(OUT, "runs.jsonl"), "a") as f:
            f.write(json.dumps(d) + "\n")


def fetch(url, size, path):
    """Ranged GET into path with STREAMS threads. Returns 'ok', 'stopped' or an error string."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.ftruncate(fd, size)
    offs = list(range(0, size, BLK)); oi = {"i": 0}; err = []; got = {"b": 0}; t0 = time.time()

    def worker():
        while not err and not stopped():
            with LOCK:
                if oi["i"] >= len(offs):
                    return
                o = offs[oi["i"]]; oi["i"] += 1
            e = min(o + BLK, size) - 1
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={o}-{e}", "User-Agent": "singlet-colo-bench"})
                    pos = o
                    with urllib.request.urlopen(req, timeout=60) as r:
                        while True:
                            b = r.read(1 << 20)
                            if not b:
                                break
                            os.pwrite(fd, b, pos); pos += len(b)
                            with LOCK:
                                got["b"] += len(b); STATE["bytes"] += len(b)
                                ahead = got["b"] / SLOT_BPS - (time.time() - t0)   # per-slot token bucket
                            if ahead > 0:
                                time.sleep(ahead)
                    if pos != e + 1:
                        raise IOError(f"short range {o}-{e}: got {pos - o}")
                    break
                except Exception as x:
                    if attempt == 3:
                        err.append(str(x)[:200])
                    else:
                        time.sleep(2 * (attempt + 1))

    th = [threading.Thread(target=worker, daemon=True) for _ in range(STREAMS)]
    for t in th: t.start()
    for t in th: t.join()
    os.close(fd)
    if err:
        return "error: " + err[0]
    if stopped():
        return "stopped"
    return "ok" if os.path.getsize(path) == size else "error: size"


def slot(s):
    while not stopped():
        run, url, size, reads = take(); size = int(size); reads = int(reads)
        d = os.path.join(SC, f"s{s}"); os.makedirs(d, exist_ok=True)
        sra, fq, tf = os.path.join(d, run + ".sra"), os.path.join(d, run + ".1fq"), os.path.join(d, "time.txt")
        rec = {"slot": s, "run": run, "size": size, "reads": reads, "t0": time.time()}
        st = fetch(url, size, sra); rec["t_fetch"] = time.time()
        if st != "ok":
            rec.update(status="stopped" if st == "stopped" else "failed", why=st)
        else:
            cmd = ["nice", "-n", NICE, "/usr/bin/time", "-f", "%e %U %S %M", "-o", tf, DEC, "download", sra, "--quality", "binned",
                   "--vdb-threads", VT, "--no-profile", "-o", fq]
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open(os.path.join(d, "decode.log"), "w"))
            while p.poll() is None:
                if stopped():
                    p.kill(); p.wait(); break
                time.sleep(0.5)
            rec["t_dec"] = time.time(); rec["rc"] = p.returncode
            try:
                e, u, sy, m = open(tf).read().split()[-4:]
                rec.update(wall=float(e), user=float(u), sys=float(sy), maxrss_kb=int(m))
            except Exception:
                pass
            rec["status"] = "stopped" if stopped() else ("done" if p.returncode == 0 else "failed")
            if rec["status"] == "failed":
                rec["why"] = open(os.path.join(d, "decode.log")).read()[-300:]
        record(rec)
        for f in (sra, fq, fq + ".json", tf):
            try:
                os.remove(f)
            except FileNotFoundError:
                pass


def net():
    with open(os.path.join(OUT, "net.tsv"), "w") as f:
        while not stopped():
            with LOCK:
                b = STATE["bytes"]
            f.write(f"{time.time():.1f}\t{b}\n"); f.flush(); time.sleep(2)


threading.Thread(target=net, daemon=True).start()
ts = [threading.Thread(target=slot, args=(s,)) for s in range(D)]
for t in ts: t.start()
for t in ts: t.join()
