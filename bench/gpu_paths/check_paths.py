#!/usr/bin/env python3
"""Checks of the code-path corpus (bench/gpu_paths/README.md). Helper of jobs/paths.sbatch.

  check_paths.py ids [A,B,..]                       ids of the manifest, optionally restricted
  check_paths.py get ID COLUMN                      one manifest field (@W@ replaced by $W)
  check_paths.py keepref ID OUT REFD EXIT WALL RSS COMMIT [CELLREADS]
        keep a CPU run as the oracle: output without the read files, sorted read hashes, invariants, BAM digest, exit status
  check_paths.py verify ID OUT REFD LOG EXIT WALL RSS RES GB OE [CELLREADS]
        compare a GPU run with the oracle and check the target path and the invariants. Writes RES/ID.json
  check_paths.py table RES [...]                    RES/verdicts.tsv from the RES/*.json files

Per input the verdict has four parts. Any failed part fails the input.
  identity    CPU oracle against GPU: out_equiv.py (its DIFF files must be the run-dependent set), the 10 STAR Solo files, the mapping
              statistics, SJ.out.tab, the three read files by sorted hash, CellReads.stats by sorted rows, fieldcheck.py (unexplained
              fields), the sorted BAM by record hash when both runs left one. When the CPU run failed, the GPU run must fail with the same exit status.
  path        the manifest's `must` expressions all match the GPU run log, no `must_not` expression matches, `files_must` exist.
  invariants  pileup reads are at least the uniquely mapped reads, called cells do not exceed the barcodes of cell_calls.tsv,
              the fraction of reads in cells lies in [0, 1], no .gpu_handoff* file is left in the output. Checked on the CPU run and the GPU run.
  fallback    strict inputs need 0 disallowed lines of fallback.py. Info inputs only report them.
"""
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "corpus", "manifest.tsv")
READF = ["nonhost_reads.fq", "star_Unmapped.out.mate1", "star_Unmapped.out.mate2"]
EXPECT_DIFF = {"pileup_stats.json", "read_stats.tsv", "star_Log.std.out", "summary.json"}
IGNORE = re.compile(r"(\.bam$|\.sam$|\.cram$|\.fq|\.fastq|\.1fq$|^\.gpu_handoff|Unmapped|^_STARtmp|\.reads$|\.reads\.q$)")
SOLO = ["Barcodes.stats", "Gene/Features.stats", "Gene/Summary.csv", "Gene/UMIperCellSorted.txt", "Gene/raw/barcodes.tsv",
        "Gene/raw/features.tsv", "Gene/raw/matrix.mtx", "Gene/filtered/barcodes.tsv", "Gene/filtered/features.tsv", "Gene/filtered/matrix.mtx"]
BAMNAME = "star_Aligned.sortedByCoord.out.bam"


def manifest():
    with open(MANIFEST, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def row(i):
    for r in manifest():
        if r["id"] == i:
            return r
    sys.exit("no such id " + i)


def ignore_fn(d, names):
    return [n for n in names if IGNORE.search(n)]


def read_hash(out):
    res = {}
    for f in READF:
        p = os.path.join(out, f)
        if not os.path.isfile(p):
            res[f] = ["absent", 0]
            continue
        with open(p, "rb") as fh:
            lines = fh.read().split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        recs = sorted(b"\t".join(lines[i:i + 4]) for i in range(0, len(lines), 4))
        h = hashlib.sha256(b"\n".join(recs)).hexdigest()[:16]
        res[f] = [h, len(recs)]
    return res


def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def final_num(out, key):
    p = os.path.join(out, "star_Log.final.out")
    if not os.path.isfile(p):
        return None
    for l in open(p, errors="replace"):
        if l.split("|")[0].strip() == key:
            try:
                return int(l.split("|")[1].strip())
            except ValueError:
                return None
    return None


def invariants(out):
    v = []
    ps = jload(os.path.join(out, "pileup_stats.json")) or {}
    uniq = final_num(out, "Uniquely mapped reads number")
    total = ps.get("total_reads")
    if total is not None and uniq is not None and total < uniq:
        v.append("pileup reads %d are fewer than the uniquely mapped reads %d" % (total, uniq))
    cc = os.path.join(out, "cell_calls.tsv")
    cells = ps.get("n_called_cells")
    if cells is not None and os.path.isfile(cc):
        with open(cc, errors="replace") as f:
            nb = max(0, sum(1 for _ in f) - 1)
        if cells > nb:
            v.append("called cells %d exceed the %d barcodes of cell_calls.tsv" % (cells, nb))
    sm = jload(os.path.join(out, "summary.json")) or {}
    fr = sm.get("fraction_reads_in_cells")
    if isinstance(fr, (int, float)) and not (0 <= fr <= 1):
        v.append("fraction_reads_in_cells %s outside [0, 1]" % fr)
    left = sorted(n for n in os.listdir(out) if n.startswith(".gpu_handoff")) if os.path.isdir(out) else []
    if left:
        v.append("hand-off side files left in the output: " + ", ".join(left))
    return v


def cellreads_hash(path):
    """header plus the sorted rows of STAR's CellReads.stats (the file order is run-dependent)"""
    if not path or path == "-" or not os.path.isfile(path):
        return None
    nl = bytes([10])
    lines = open(path, "rb").read().split(nl)
    if lines and lines[-1] == b"":
        lines.pop()
    if not lines:
        return ["empty", 0]
    h = hashlib.sha256(lines[0] + nl + nl.join(sorted(lines[1:]))).hexdigest()[:16]
    return [h, len(lines) - 1]


def bam_digest(out):
    p = os.path.join(out, BAMNAME)
    if not os.path.isfile(p):
        return None
    sys.path.insert(0, os.environ.get("OE_DIR", HERE))
    try:
        import out_equiv
        return out_equiv.bam_body_digest(p)[0]
    except Exception as e:  # noqa: BLE001
        return "ERR " + str(e)[:80]


def keepref(i, out, refd, ex, wall, rss, commit, crs="-"):
    tmp = refd + ".new"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    meta = dict(id=i, exit=int(ex), wall_s=float(wall), maxrss_gb=float(rss), commit=commit,
                readhash=read_hash(out) if os.path.isdir(out) else {}, invariants=invariants(out) if os.path.isdir(out) else [],
                bam=bam_digest(out) if os.path.isdir(out) else None, cellreads=cellreads_hash(crs))
    if os.path.isdir(out):
        shutil.copytree(out, os.path.join(tmp, "out"), ignore=ignore_fn)
    json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"), indent=1)
    shutil.rmtree(refd, ignore_errors=True)
    os.rename(tmp, refd)
    print("reference kept: %s exit %s wall %.1f s invariants %s" % (refd, ex, float(wall), meta["invariants"] or "ok"))


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def identity(a, b, ra, rb, gb, oe):
    """a = oracle out dir, b = copy of the GPU out dir. Returns (identical, dict)"""
    d = {}
    ok = True
    r = run([sys.executable, oe, a, b])
    txt = r.stdout + r.stderr
    m = re.search(r"files: (\d+) identical, (\d+) differ only in timings/paths, (\d+) differ", txt)
    d["out_equiv"] = "/".join(m.groups()) if m else "FAILED"
    unx = []
    for f in re.findall(r"^  DIFF ([^ :]+)", txt, re.M):
        if f not in EXPECT_DIFF:
            unx.append(f)
    if re.search(r"^  DIFF provenance.json: /references/whitelist/name: [^;]*$", txt, re.M) and "provenance.json" in unx:
        unx.remove("provenance.json")
    unx += ["only:" + x for x in re.findall(r"^  only in [ab]: (.+)$", txt, re.M)]
    if not m:
        unx.append("out_equiv-failed")
    d["unexpected_diff"] = unx
    ok &= not unx
    solo_bad = [f for f in SOLO if not (os.path.isfile(os.path.join(a, "star_Solo.out", f)) and os.path.isfile(os.path.join(b, "star_Solo.out", f))
                                       and open(os.path.join(a, "star_Solo.out", f), "rb").read() == open(os.path.join(b, "star_Solo.out", f), "rb").read())]
    d["solo_differ"] = solo_bad
    ok &= not solo_bad

    def tail(p):
        try:
            return open(p, errors="replace").read().splitlines()[4:]
        except OSError:
            return None
    d["final_out_same"] = tail(os.path.join(a, "star_Log.final.out")) == tail(os.path.join(b, "star_Log.final.out"))
    ok &= d["final_out_same"]
    sj = [os.path.join(x, "star_SJ.out.tab") for x in (a, b)]
    d["sj_same"] = all(os.path.isfile(p) for p in sj) and open(sj[0], "rb").read() == open(sj[1], "rb").read()
    ok &= d["sj_same"]
    d["readfiles_differ"] = [f for f in READF if ra.get("readhash", {}).get(f) != rb.get("readhash", {}).get(f)]
    ok &= not d["readfiles_differ"]
    ca, cb = ra.get("cellreads"), rb.get("cellreads")
    d["cellreads_same"] = (ca is None and cb is None) or ca == cb
    ok &= d["cellreads_same"]
    fc = run([sys.executable, os.path.join(gb, "fieldcheck.py"), a, b])
    mm = re.search(r"FIELD CHECK: (\d+) unexplained", fc.stdout)
    d["field_unexplained"] = int(mm.group(1)) if mm else -1
    d["fielddiff"] = re.findall(r"^FIELDDIFF .*$", fc.stdout, re.M)[:40]
    ok &= d["field_unexplained"] == 0
    return ok, d


def verify(i, out, refd, log, ex, wall, rss, res, gb, oe, crs="-"):
    r = row(i)
    meta = jload(os.path.join(refd, "meta.json")) or {}
    v = dict(id=i, kind=r["kind"], target=r["target"], reads=r["n"], gpu_exit=int(ex), gpu_wall_s=float(wall), gpu_maxrss_gb=float(rss),
             cpu_exit=meta.get("exit"), cpu_wall_s=meta.get("wall_s"), cpu_maxrss_gb=meta.get("maxrss_gb"))
    text = open(log, errors="replace").read() if os.path.isfile(log) else ""
    # path assertions
    must = [x.strip() for x in r["must"].split(";;") if x.strip()]
    mustnot = [x.strip() for x in r["must_not"].split(";;") if x.strip()]
    miss = [x for x in must if not re.search(x, text)]
    hit = [x for x in mustnot if re.search(x, text)]
    nofile = [f for f in [x.strip() for x in r["files_must"].split(";;") if x.strip()] if not os.path.exists(os.path.join(out, f))]
    v["path_missing"], v["path_forbidden_seen"], v["files_missing"] = miss, hit, nofile
    v["path"] = "PASS" if not (miss or hit or nofile) else "FAIL"
    mw = re.findall(r"low-quality masks for (\d+) reads(?:, (\d+) words each)?", text)
    v["mask_line"] = ("%s reads, %s words" % (mw[-1][0], mw[-1][1] or "2 (not logged)")) if mw else "none"
    v["restarted"] = "restarting on the sorted path" in text
    lv = re.findall(r"\[engine_gpu\] level (\d) \(requested (\d)\)", text)
    v["level"] = ("%s of %s" % (lv[-1][0], lv[-1][1])) if lv else "none"
    v["ingest"] = "used" if re.search(r"\.1fq scanned: STAR's prefill decodes", text) else "not used"
    # invariants
    inv = invariants(out) if os.path.isdir(out) else ["no output directory"]
    v["invariant_violations_gpu"] = inv
    v["invariant_violations_cpu"] = meta.get("invariants", [])
    v["invariants"] = "PASS" if not (inv or v["invariant_violations_cpu"]) else "FAIL"
    # fallback lines
    fb = run([sys.executable, os.path.join(gb, "fallback.py"), res + "/" + i]) if os.path.isdir(res + "/" + i) else None
    nbad = 0
    bad_lines = []
    if fb is not None:
        for l in fb.stdout.splitlines():
            if l.startswith("SUMMARY"):
                nbad = int(l.split("\t")[1])
            elif l.startswith("DISALLOWED"):
                bad_lines.append(l.split("\t", 2)[-1][:300])
    v["fallback_disallowed"] = nbad
    v["fallback_lines"] = bad_lines
    v["fallback"] = "PASS" if (nbad == 0 or r["fb"] == "info") else "FAIL"
    # identity
    cpu_ok = meta.get("exit") == 0
    if not meta:
        v["identity"] = "NO ORACLE"
    elif not cpu_ok:
        v["identity"] = "SAME FAILURE" if int(ex) == meta.get("exit") else "DIFFERENT EXIT (cpu %s, gpu %s)" % (meta.get("exit"), ex)
    elif int(ex) != 0:
        v["identity"] = "GPU FAILED (exit %s)" % ex
    else:
        cmp_dir = os.path.join(res, i, "cmp")
        shutil.rmtree(cmp_dir, ignore_errors=True)
        shutil.copytree(out, cmp_dir, ignore=ignore_fn)
        rb = dict(readhash=read_hash(out), cellreads=cellreads_hash(crs))
        ra = dict(readhash=meta.get("readhash", {}), cellreads=meta.get("cellreads"))
        okid, d = identity(os.path.join(refd, "out"), cmp_dir, ra, rb, gb, oe)
        v.update(d)
        v["identity"] = "IDENTICAL" if okid else "NOT IDENTICAL"
        shutil.rmtree(cmp_dir, ignore_errors=True)
        bg, bc = bam_digest(out), meta.get("bam")
        v["bam"] = "not written" if (bg is None and bc is None) else ("same" if bg == bc else "DIFFERENT (cpu %s, gpu %s)" % (bc, bg))
        if v["bam"].startswith("DIFFERENT"):
            v["identity"] = "NOT IDENTICAL"
    v.setdefault("bam", "not written" if bam_digest(out) is None else "gpu only")
    v["overall"] = "PASS" if (v["identity"] in ("IDENTICAL", "SAME FAILURE") and v["path"] == "PASS" and v["invariants"] == "PASS"
                              and v["fallback"] == "PASS") else "FAIL"
    json.dump(v, open(os.path.join(res, i + ".json"), "w"), indent=1)
    print("VERDICT %s %s (identity %s, path %s, invariants %s, fallback %s)" % (i, v["overall"], v["identity"], v["path"], v["invariants"], v["fallback"]))
    for k in ("path_missing", "path_forbidden_seen", "files_missing", "invariant_violations_gpu", "invariant_violations_cpu", "unexpected_diff", "solo_differ", "fielddiff"):
        if v.get(k):
            print("  %s: %s" % (k, v[k]))
    for l in bad_lines:
        print("  DISALLOWED " + l)


def table(res):
    cols = ["id", "kind", "reads", "overall", "identity", "path", "invariants", "fallback", "fallback_disallowed", "out_equiv", "field_unexplained",
            "level", "ingest", "mask_line", "restarted", "bam", "cpu_exit", "gpu_exit", "cpu_wall_s", "gpu_wall_s", "cpu_maxrss_gb", "gpu_maxrss_gb"]
    rows = []
    order = [r["id"] for r in manifest()]
    for i in order:
        p = os.path.join(res, i + ".json")
        if os.path.isfile(p):
            rows.append(json.load(open(p)))
    with open(os.path.join(res, "verdicts.tsv"), "w") as f:
        f.write("\t".join(cols) + "\n")
        for v in rows:
            f.write("\t".join(str(v.get(c, "")) for c in cols) + "\n")
    n_fail = sum(1 for v in rows if v["overall"] != "PASS")
    print("CORPUS: %d inputs, %d pass, %d fail" % (len(rows), len(rows) - n_fail, n_fail))
    return n_fail


def main(a):
    cmd = a[1]
    if cmd == "ids":
        only = set(a[2].split(",")) if len(a) > 2 and a[2] else None
        print(" ".join(r["id"] for r in manifest() if only is None or r["id"] in only))
    elif cmd == "get":
        print(row(a[2])[a[3]].replace("@W@", os.environ.get("W", "")))
    elif cmd == "keepref":
        keepref(*a[2:10])
    elif cmd == "verify":
        verify(*a[2:13])
    elif cmd == "table":
        return 1 if table(a[2]) else 0
    else:
        sys.exit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
