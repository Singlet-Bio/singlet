#!/usr/bin/env python3
"""S9 Phase 1: measure download throughput from SRA sources on one Clipper node (or in step with other nodes).

Stdlib only. Every test is a fixed time window: start N streams of one tool against one source, poll the allocated bytes of
the node-local output files once a second, kill the streams, delete the files. Rate = bytes gained between `skip` seconds and
the end of the window, so connection setup and TCP ramp-up are excluded (the full-window rate is reported too).

  dlbench.py prep   --runs runs.json [--n 80]        resolve candidate runs (ENA search + NCBI SDL) -> runs.json
  dlbench.py matrix --runs runs.json --out r.jsonl --spec ncbi:curl:1,4,8 --spec s3:aria2:16 [--mode single|multi]
                    [--start-at EPOCH] [--tag name]

Sources: ncbi (SRA Lite on NCBI's own host), s3 (full .sra, sra-pub-run-odp), gcs (anonymous only; skipped on 403).
Tools: curl (N ranged processes), aria2 (one aria2c, -x16 per URI, URI repeated so N streams are possible), s5cmd (s3 only,
--concurrency N parts), prefetch (N processes, multi mode only: one run each).
Modes: single = all streams read disjoint ranges of ONE large file; multi = each stream reads a different run (the pipeline's case).
"""
import argparse, json, os, shutil, signal, socket, subprocess, time, urllib.parse, urllib.request

WORKDIR = os.path.join("/tmp", "%s-%s-dlb" % (os.environ.get("USER", "u"), os.environ.get("SLURM_JOB_ID", str(os.getpid()))))
DLENV = "/mnt/projects/debruinz_project/debruinz/singlet/tools/dl/bin"
os.environ["PATH"] = DLENV + ":" + os.environ["PATH"]
RANGE_LEN = 2 * 1000**3


def get_json(url, tries=3):
    err = None
    for i in range(tries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=60))
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(2 * (i + 1))
    raise err


def cmd_prep(a):
    q = urllib.parse.quote('library_source="TRANSCRIPTOMIC SINGLE CELL" AND instrument_platform="ILLUMINA" AND library_strategy="RNA-Seq"')
    url = ("https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&query=%s&fields=run_accession,base_count,read_count&format=json&limit=4000" % q)
    rows = get_json(url)
    rows = [r for r in rows if r.get("read_count") and 40e6 < int(r["read_count"]) < 250e6]
    rows.sort(key=lambda r: r["run_accession"])
    step = max(1, len(rows) // (a.n * 3))
    out = []
    for r in rows[::step]:
        acc = r["run_accession"]
        try:
            d = get_json("https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve?acc=%s&filetype=run&accept-alternate-locations=yes" % acc)
        except Exception:  # noqa: BLE001
            continue
        ent = {"run": acc, "reads": int(r["read_count"])}
        for f in d["result"][0].get("files", []):
            for loc in f.get("locations", []):
                svc = loc.get("service")
                lite = f["name"].endswith(".lite") or "lite" in f["name"]
                key = ("lite_" if lite else "full_") + svc if f.get("type", "sra") == "sra" else None
                if key and key not in ent:
                    ent[key] = loc["link"]
                    ent[key + "_size"] = f["size"]
        if "lite_ncbi" in ent and "full_s3" in ent:
            out.append(ent)
        if len(out) >= a.n:
            break
    out.sort(key=lambda e: -e["lite_ncbi_size"])
    json.dump(out, open(a.runs, "w"), indent=1)
    print("prepared %d runs; lite sizes GB: %s" % (len(out), [round(e["lite_ncbi_size"] / 1e9, 1) for e in out[:10]]))


def alloc_bytes(paths):
    t = 0
    for p in paths:
        try:
            t += os.stat(p).st_blocks * 512
        except OSError:
            pass
    return t


def url_for(source, e):
    if source == "ncbi":
        return e.get("lite_ncbi"), e.get("lite_ncbi_size")
    if source == "s3":
        return e.get("full_s3"), e.get("full_s3_size")
    if source == "gcs":
        return e.get("lite_gs"), e.get("lite_gs_size")
    return None, None


def build(tool, source, n, mode, runs, d):
    """-> (list of argv, list of output paths)"""
    os.makedirs(d, exist_ok=True)
    cmds, outs = [], []
    if mode == "single":
        url, size = url_for(source, runs[0])
        span = max(size - RANGE_LEN, 0)
        if tool == "curl":
            for i in range(n):
                off = int(span * i / max(n, 1))
                o = os.path.join(d, "p%d" % i)
                cmds.append(["curl", "-sS", "--fail", "--retry", "2", "-r", "%d-%d" % (off, min(off + RANGE_LEN, size) - 1), "-o", o, url])
                outs.append(o)
        elif tool == "aria2":
            o = os.path.join(d, "f")
            conn = min(n, 16)
            nuri = (n + 15) // 16
            cmds.append(["aria2c", "-q", "--file-allocation=none", "--max-connection-per-server=%d" % conn, "--split=%d" % n,
                         "--min-split-size=1M", "-k", "1M", "--summary-interval=0", "--timeout=30", "--allow-overwrite=true",
                         "-d", d, "-o", "f"] + [url] * nuri)
            outs.append(o)
        elif tool == "s5cmd" and source == "s3":
            key = url.split(".amazonaws.com/")[1]
            o = os.path.join(d, "f")
            cmds.append(["s5cmd", "--no-sign-request", "--endpoint-url", "https://s3.amazonaws.com", "cp", "--concurrency", str(n),
                         "--part-size", "16", "s3://sra-pub-run-odp/" + key, o])
            outs.append(o)
        else:
            return None, None
    else:  # multi: one run per stream
        for i in range(n):
            e = runs[i % len(runs)]
            url, size = url_for(source, e)
            o = os.path.join(d, "r%d" % i)
            if tool == "curl":
                cmds.append(["curl", "-sS", "--fail", "--retry", "2", "-o", o, url])
            elif tool == "aria2":
                cmds.append(["aria2c", "-q", "--file-allocation=none", "-x", "4", "-s", "4", "-k", "4M", "--timeout=30", "-d", d,
                             "-o", "r%d" % i, url])
            elif tool == "s5cmd" and source == "s3":
                cmds.append(["s5cmd", "--no-sign-request", "--endpoint-url", "https://s3.amazonaws.com", "cp", "--concurrency", "4",
                             "--part-size", "16", "s3://sra-pub-run-odp/" + url.split(".amazonaws.com/")[1], o])
            elif tool == "prefetch":
                o = os.path.join(d, "pf%d" % i)
                cmds.append(["prefetch", "--max-size", "u", "-q", "-O", o, e["run"]])
            else:
                return None, None
            outs.append(o)
    return cmds, outs


def tree_files(outs):
    fs = []
    for o in outs:
        if os.path.isdir(o):
            for r, _, names in os.walk(o):
                fs += [os.path.join(r, x) for x in names]
        else:
            fs.append(o)
    return fs


def probe_ok(source, runs):
    url, _ = url_for(source, runs[0])
    if not url:
        return False, "no url"
    try:
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-r", "0-99", url], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() in ("200", "206"), r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return False, repr(e)


def one_test(source, tool, n, mode, runs, seconds, skip):
    d = os.path.join(WORKDIR, "t")
    shutil.rmtree(d, ignore_errors=True)
    cmds, outs = build(tool, source, n, mode, runs, d)
    if cmds is None:
        return {"skipped": "unsupported combination"}
    ok, code = probe_ok(source, runs)
    if not ok:
        return {"skipped": "probe http %s" % code}
    procs = [subprocess.Popen(c, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True) for c in cmds]
    t0 = time.time()
    series = []
    while time.time() - t0 < seconds:
        time.sleep(1.0)
        series.append((time.time() - t0, alloc_bytes(tree_files(outs))))
        if all(p.poll() is not None for p in procs):
            break
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
    errs = [e for e in (p.stderr.read().decode()[:200] for p in procs) if e.strip()]
    elapsed = series[-1][0] if series else 0
    total = series[-1][1] if series else 0
    w = [s for s in series if s[0] >= skip]
    steady = (w[-1][1] - w[0][1]) / (w[-1][0] - w[0][0]) / 1e6 if len(w) >= 3 and w[-1][0] > w[0][0] else None
    shutil.rmtree(d, ignore_errors=True)
    return {"bytes": total, "secs": round(elapsed, 1), "mbps_full": round(total / max(elapsed, 1e-9) / 1e6, 1),
            "mbps_steady": round(steady, 1) if steady else None, "finished_early": elapsed < seconds - 1.5,
            "errors": errs[:2] or None, "series": [(round(t, 1), b) for t, b in series[::3]]}


def cmd_matrix(a):
    runs = json.load(open(a.runs))
    os.makedirs(WORKDIR, exist_ok=True)
    node = socket.gethostname()
    if a.start_at:
        time.sleep(max(0, a.start_at - time.time()))
    total = 0
    for spec in a.spec:
        source, tool, ns = spec.split(":")
        for n in [int(x) for x in ns.split(",")]:
            rec = {"tag": a.tag, "node": node, "source": source, "tool": tool, "streams": n, "mode": a.mode,
                   "start": time.strftime("%Y-%m-%dT%H:%M:%S"), "node_count": a.node_count}
            rec.update(one_test(source, tool, n, a.mode, runs, a.seconds, a.skip))
            total += rec.get("bytes", 0)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("%s %s %s n=%d -> %s MB/s steady, %s full, %.1f GB %s" % (
                node, source, tool, n, rec.get("mbps_steady"), rec.get("mbps_full"), rec.get("bytes", 0) / 1e9, rec.get("skipped", "")), flush=True)
            time.sleep(a.gap)
    print("TOTAL_GB %.1f" % (total / 1e9))
    shutil.rmtree(WORKDIR, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="c", required=True)
    p = sp.add_parser("prep")
    p.add_argument("--runs", required=True)
    p.add_argument("--n", type=int, default=80)
    m = sp.add_parser("matrix")
    m.add_argument("--runs", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--spec", action="append", required=True)
    m.add_argument("--mode", default="single")
    m.add_argument("--seconds", type=float, default=25)
    m.add_argument("--skip", type=float, default=8)
    m.add_argument("--gap", type=float, default=3)
    m.add_argument("--start-at", type=float, default=0)
    m.add_argument("--tag", default="")
    m.add_argument("--node-count", type=int, default=1)
    args = ap.parse_args()
    cmd_prep(args) if args.c == "prep" else cmd_matrix(args)
