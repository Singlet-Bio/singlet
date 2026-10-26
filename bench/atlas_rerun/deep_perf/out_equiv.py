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
SKIP_FILE = re.compile(r"(\.log$|Log\.out$|Log\.progress\.out$|Log\.final\.out$|\.bam$|\.bai$|time\.txt$)")


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
        bad += 1
    print(f"files: {same} identical, {diff_soft} differ only in timings/paths, {bad} differ")
    return bad


def compare_bams(pa: str, pb: str) -> int:
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
        print(f"BAM {Path(pa).name}: {n} records, {diff} differ")
        return bad + diff


def main(argv):
    a, b = Path(argv[1]), Path(argv[2])
    print(f"a = {a}\nb = {b}")
    bad = compare_dirs(a, b)
    if "--bam" in argv:
        rest = argv[argv.index("--bam") + 1:]
        for i in range(0, len(rest) - 1, 2):
            bad += compare_bams(rest[i], rest[i + 1])
    print("RESULT:", "IDENTICAL" if bad == 0 else f"{bad} differences")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
