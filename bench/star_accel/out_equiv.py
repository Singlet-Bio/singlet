"""Are two Singlet runs' outputs the same? File by file, and STAR BAMs record by record.

    python out_equiv.py <run_a> <run_b> [--bam A.bam B.bam ...pairs]

run_a / run_b: the `out/` directories of two runs of the same input (bench/atlas_rerun/deep_perf/run_deep.sbatch).
Every file is compared by sha256. Files that differ are listed; JSON files are then compared key by key with timing,
path, host and version fields ignored, so a run whose matrices and stats agree and whose logs differ reads as
"identical outside timings/paths". BAM pairs are compared record by record (pysam, all fields and tags, in file
order) with the header's @PG/@CO lines (command lines, temp paths) ignored.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

VOLATILE_KEY = re.compile(r"(time|_s$|seconds|elapsed|wall|date|host|node|path|dir|prefix|file|command|cmd|pid|rss|"
                          r"version|sha|commit|started|finished|tmp|threads|input)", re.I)
SKIP_FILE = re.compile(r"(\.log$|Log\.out$|Log\.progress\.out$|\.bam$|\.bai$|time\.txt$)")
# STAR's Log.final.out carries the mapping statistics the pipeline consumes (summary.json mapping rate, the catalog's star_log member): compared
# with its date and speed lines removed.
LOG_FINAL = re.compile(r"Log\.final\.out$")
LOG_FINAL_VOLATILE = ("Started job on", "Started mapping on", "Finished on", "Mapping speed")


def normalize_log_final(text: str) -> list:
    return [l.rstrip() for l in text.splitlines() if not l.strip().startswith(LOG_FINAL_VOLATILE)]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if not VOLATILE_KEY.search(k)}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def json_diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'b' if k not in a else 'a'}")
            else:
                out += json_diff(a[k], b[k], f"{path}/{k}")
    elif a != b:
        out.append(f"{path}: {str(a)[:80]} != {str(b)[:80]}")
    return out


def compare_dirs(a: Path, b: Path) -> int:
    fa = {p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file()}
    fb = {p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file()}
    bad = 0
    for f in sorted(fa ^ fb):
        if not SKIP_FILE.search(f):
            print(f"  only in {'a' if f in fa else 'b'}: {f}")
            bad += 1
    same = diff_soft = 0
    for f in sorted(fa & fb):
        if SKIP_FILE.search(f):
            continue
        if sha(a / f) == sha(b / f):
            same += 1
            continue
        if LOG_FINAL.search(f):
            la, lb = normalize_log_final((a / f).read_text(errors="replace")), normalize_log_final((b / f).read_text(errors="replace"))
            if la == lb:
                diff_soft += 1
                continue
            nd = [i for i in range(min(len(la), len(lb))) if la[i] != lb[i]]
            print(f"  DIFF {f}: {len(nd)} of {len(la)}/{len(lb)} lines differ" +
                  "".join(f"\n    a: {la[i][:160]}\n    b: {lb[i][:160]}" for i in nd[:4]))
            bad += 1
            continue
        if f.endswith(".json"):
            try:
                ja, jb = json.loads((a / f).read_text()), json.loads((b / f).read_text())
                d = json_diff(strip(ja), strip(jb))
                if not d:
                    diff_soft += 1
                    continue
                print(f"  DIFF {f}: " + "; ".join(d[:6]) + (f" (+{len(d) - 6})" if len(d) > 6 else ""))
            except ValueError:
                print(f"  DIFF {f} (unparsable json)")
        else:
            print(f"  DIFF {f} ({(a / f).stat().st_size} vs {(b / f).stat().st_size} bytes)")
            if f.endswith((".tsv", ".csv", ".txt")) and (a / f).stat().st_size < 2_000_000:
                la, lb = (a / f).read_text(errors="replace").splitlines(), (b / f).read_text(errors="replace").splitlines()
                nd = [i for i in range(min(len(la), len(lb))) if la[i] != lb[i]]
                print(f"    {len(nd)} of {len(la)}/{len(lb)} lines differ" +
                      "".join(f"\n    a: {la[i][:160]}\n    b: {lb[i][:160]}" for i in nd[:2]))
        bad += 1
    print(f"files: {same} identical, {diff_soft} differ only in timings/paths, {bad} differ")
    return bad


def bam_body_digest(path: str):
    """sha256 of the uncompressed BAM after its header (records only, in file order), the reference list, and the
    header text without @PG/@CO lines. Independent of BGZF block layout and compression level."""
    import struct
    import zlib

    h = hashlib.sha256()
    buf = bytearray()
    state = {"skip": None, "refs": None, "text": None}

    def feed(data: bytes):
        buf.extend(data)
        if state["skip"] is None:
            if len(buf) < 12:
                return
            l_text = struct.unpack_from("<i", buf, 4)[0]
            if len(buf) < 12 + l_text:
                return
            n_ref = struct.unpack_from("<i", buf, 8 + l_text)[0]
            off = 12 + l_text
            refs = []
            for _ in range(n_ref):
                if len(buf) < off + 4:
                    return
                l_name = struct.unpack_from("<i", buf, off)[0]
                if len(buf) < off + 4 + l_name + 4:
                    return
                refs.append((bytes(buf[off + 4:off + 3 + l_name]), struct.unpack_from("<i", buf, off + 4 + l_name)[0]))
                off += 8 + l_name
            text = bytes(buf[8:8 + l_text]).decode(errors="replace").rstrip("\0")
            state["text"] = [l for l in text.splitlines() if not l.startswith(("@PG", "@CO"))]
            state["refs"] = refs
            state["skip"] = off
            del buf[:off]
        h.update(buf)
        buf.clear()

    with open(path, "rb") as f:
        d = zlib.decompressobj(31)
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            while chunk:
                feed(d.decompress(chunk))
                if d.eof:
                    chunk = d.unused_data
                    d = zlib.decompressobj(31)
                else:
                    chunk = b""
    return h.hexdigest(), state["refs"], state["text"]


def compare_bams(pa: str, pb: str) -> int:
    da, db = bam_body_digest(pa), bam_body_digest(pb)
    if da == db:
        print(f"BAM {Path(pa).name}: header and all records identical (sha256 of the uncompressed records {da[0][:16]})")
        return 0
    print(f"BAM {Path(pa).name}: uncompressed records differ (refs equal {da[1] == db[1]}, header equal {da[2] == db[2]})"
          "; record-level diff:")
    import pysam

    with pysam.AlignmentFile(pa, "rb", check_sq=False) as A, pysam.AlignmentFile(pb, "rb", check_sq=False) as B:
        ha = [l for l in str(A.header).splitlines() if not l.startswith(("@PG", "@CO"))]
        hb = [l for l in str(B.header).splitlines() if not l.startswith(("@PG", "@CO"))]
        bad = 0 if ha == hb else 1
        if bad:
            print(f"  BAM header differs ({len(ha)} vs {len(hb)} lines)")
        n = diff = 0
        ia, ib = A.fetch(until_eof=True), B.fetch(until_eof=True)
        while True:
            ra, rb = next(ia, None), next(ib, None)
            if ra is None and rb is None:
                break
            n += 1
            if ra is None or rb is None or ra.to_string() != rb.to_string():
                diff += 1
                if diff <= 5:
                    print(f"  record {n}: {ra.to_string()[:150] if ra else None}\n            {rb.to_string()[:150] if rb else None}")
                if diff >= 1000:
                    print("  (stopped after 1000 differing records)")
                    break
        print(f"BAM {Path(pa).name}: {n} records, {diff} differ")
        return bad + diff


def compare_matrices(a: Path, b: Path) -> None:
    """For runs written with --output-format mtx: per member, entries that differ after aligning features/barcodes."""
    import gzip

    import scipy.io
    import scipy.sparse as sp

    def lines(p):
        with gzip.open(p, "rt") as f:
            return [x.rstrip("\n") for x in f]

    for mtx in sorted(a.glob("*.mtx.gz")):
        m = mtx.name[:-len(".mtx.gz")]
        if not (b / mtx.name).exists():
            continue
        try:
            xa = sp.csr_matrix(scipy.io.mmread(str(mtx)))
            xb = sp.csr_matrix(scipy.io.mmread(str(b / mtx.name)))
            fa, fb = lines(a / f"{m}_features.tsv.gz"), lines(b / f"{m}_features.tsv.gz")
            ba, bb = lines(a / f"{m}_barcodes.tsv.gz"), lines(b / f"{m}_barcodes.tsv.gz")
        except (OSError, ValueError):
            continue
        fs, bs = sorted(set(fa) & set(fb)), sorted(set(ba) & set(bb))
        ia = {f: i for i, f in enumerate(fa)}; ib = {f: i for i, f in enumerate(fb)}
        ja = {c: i for i, c in enumerate(ba)}; jb = {c: i for i, c in enumerate(bb)}
        A = xa[[ia[f] for f in fs]][:, [ja[c] for c in bs]]
        B = xb[[ib[f] for f in fs]][:, [jb[c] for c in bs]]
        d = abs(A - B)
        print(f"  matrix {m}: barcodes {len(ba)}/{len(bb)} shared {len(bs)}, total {xa.sum():.0f}/{xb.sum():.0f}, "
              f"nnz {A.nnz}/{B.nnz}, entries differing {d.nnz}, sum |diff| {d.sum():.0f}")


def main(argv):
    a, b = Path(argv[1]), Path(argv[2])
    print(f"a = {a}\nb = {b}")
    bad = compare_dirs(a, b)
    compare_matrices(a, b)
    if "--bam" in argv:
        rest = argv[argv.index("--bam") + 1:]
        for i in range(0, len(rest) - 1, 2):
            bad += compare_bams(rest[i], rest[i + 1])
    print("RESULT:", "IDENTICAL" if bad == 0 else f"{bad} differences")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
