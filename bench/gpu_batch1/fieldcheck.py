#!/usr/bin/env python3
"""Field-level check of the output files that the gate excuses by name.

Usage: fieldcheck.py OUT_A OUT_B [--show-allowed]
Files: read_stats.tsv, pileup_stats.json, summary.json, provenance.json, ambient_contamination.json.
Timing fields are ignored (leaf name ends in seconds, _s or _ms, or the field sits under timings, or is a timestamp).
Every other difference is printed as
    FIELDDIFF <file> <field> <a> <b>
unless an entry of ALLOW or a derived rule explains it. Last line:  FIELD CHECK: <n> unexplained
Allowed differences are counted on the line before it. --show-allowed prints each one as FIELDOK.
The check never changes the gate's verdict line.
"""
import hashlib
import json
import os
import re
import sys

FILES = ["pileup_stats.json", "summary.json", "provenance.json", "ambient_contamination.json"]

# (file, regex on the JSON path, id, reason). Keep this list short. Each entry needs a reason that can be stated.
ALLOW = [
    ("summary.json", r"^/memory_tier$", "memory_tier",
     "memory_tier is the tier of the measured peak RSS. The GPU path peaks higher, for example 84 GB against 35 GB on nuclei2k_v2."),
    ("summary.json", r"^/peak_rss_gb$", "peak_rss",
     "peak_rss_gb is a measurement of the run, not an output of the pipeline."),
    ("pileup_stats.json", r"^/provenance/clip5p/probe/n_reads$", "clip5p_probe_n",
     "the clip5p probe reads the first reads it sees. The file path of the GPU run and the CPU stream see a different count (E11 section 2a, noted in the gate)."),
    ("provenance.json", r"^/host/", "host",
     "node name of the machine, and ram_gb, which holds the run's peak RSS (26.4 against 103.6 on pbmc2m, the same as summary.json peak_rss_gb)."),
    ("provenance.json", r"^/references/whitelist/name$", "whitelist_path",
     "the whitelist name is the path that the binary resolved from its own directory."),
]
# Derived rules, applied in code:
#  bytes_public, bytes_private and outputs/files/<path>/bytes may differ only when some listed output file differs in content
#  between the two runs (the sizes of files that carry timings).
#  read_stats.tsv: the wall_seconds column is ignored. The encode row exists only on a run that decoded the .1fq (decode_time > 0 in singlet.cpp).
#  align reads_in may differ only when exactly one run has an encode row and its align reads_in equals that row's reads_out (singlet.cpp: decode_reads_out).

TIMING_LEAF = re.compile(r"(seconds|_s|_ms)$")
TIMING_PART = {"timings", "timestamp"}
FILES_KEY = re.compile(r"^(/provenance)?/outputs/files/(.+)/(bytes|visibility)$")
TOTAL_KEY = re.compile(r"^(/provenance)?/outputs/bytes_(public|private)$")


def is_timing(path):
    parts = [p for p in path.split("/") if p]
    return bool(parts) and (TIMING_LEAF.search(parts[-1]) is not None or any(p in TIMING_PART for p in parts))


def flatten(o, path=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flatten(v, f"{path}/{k}"))
    elif isinstance(o, list):
        if o and all(isinstance(e, dict) and "path" in e for e in o):
            for e in o:
                out.update(flatten({k: v for k, v in e.items() if k != "path"}, f"{path}/{e['path']}"))
        else:
            for i, v in enumerate(o):
                out.update(flatten(v, f"{path}/{i}"))
    else:
        out[path or "/"] = o
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def short(v):
    s = str(v)
    return s if len(s) <= 60 else s[:57] + "..."


def main(a, b, show):
    unexplained, allowed = [], {}
    cache = {}

    def diff(f, field, va, vb):
        unexplained.append(f"FIELDDIFF {f} {field} {short(va)} {short(vb)}")

    def differs_in_content(rel):
        pa, pb = os.path.join(a, rel), os.path.join(b, rel)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            return None
        if rel not in cache:
            cache[rel] = sha(pa) != sha(pb)
        return cache[rel]

    for f in FILES:
        pa, pb = os.path.join(a, f), os.path.join(b, f)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            if os.path.exists(pa) != os.path.exists(pb):
                diff(f, "(file)", "present" if os.path.exists(pa) else "missing", "present" if os.path.exists(pb) else "missing")
            continue
        try:
            fa, fb = flatten(json.load(open(pa))), flatten(json.load(open(pb)))
        except ValueError:
            diff(f, "(unparsable json)", "", "")
            continue
        listed = {FILES_KEY.match(x).group(2) for x in list(fa) + list(fb) if FILES_KEY.match(x)}
        for k in sorted(set(fa) | set(fb)):
            va, vb = fa.get(k, "<absent>"), fb.get(k, "<absent>")
            if va == vb or is_timing(k):
                continue
            hit = next(((i, why) for ff, rx, i, why in ALLOW if ff == f and re.search(rx, k)), None)
            if hit:
                allowed.setdefault(hit[0], []).append((f, k, va, vb, hit[1]))
                continue
            m = FILES_KEY.match(k)
            if (m and m.group(3) == "bytes" and differs_in_content(m.group(2)) is True) or \
               (TOTAL_KEY.match(k) and any(differs_in_content(r) is True for r in listed)):
                allowed.setdefault("bytes", []).append((f, k, va, vb, "an output file that differs in content (timings) has another size."))
                continue
            diff(f, k, va, vb)

    pa, pb = os.path.join(a, "read_stats.tsv"), os.path.join(b, "read_stats.tsv")
    if os.path.exists(pa) and os.path.exists(pb):
        def rows(p):
            L = [l.rstrip("\n").split("\t") for l in open(p) if l.strip()]
            return L[0], {r[0]: dict(zip(L[0], r)) for r in L[1:]}
        ha, ra = rows(pa)
        hb, rb = rows(pb)
        if ha != hb:
            diff("read_stats.tsv", "(header)", ",".join(ha), ",".join(hb))
        for st in sorted(set(ra) | set(rb)):
            if st not in ra or st not in rb:
                if st == "encode":
                    allowed.setdefault("encode_row", []).append(("read_stats.tsv", "encode", "present" if st in ra else "absent", "present" if st in rb else "absent",
                                                                  "the encode row is written only on a run that decoded the .1fq (decode_time > 0)."))
                else:
                    diff("read_stats.tsv", st, "present" if st in ra else "absent", "present" if st in rb else "absent")
                continue
            for col in ha:
                if col in ("stage", "wall_seconds"):
                    continue
                va, vb = ra[st].get(col), rb[st].get(col)
                if va == vb:
                    continue
                if st == "align" and col == "reads_in":
                    ea, eb = ra.get("encode"), rb.get("encode")
                    if (ea and not eb and ea["reads_out"] == va) or (eb and not ea and eb["reads_out"] == vb):
                        allowed.setdefault("align_reads_in", []).append(("read_stats.tsv", "align.reads_in", va, vb,
                                                                          "on a decoded run align reads_in is the decoded read count (singlet.cpp: decode_reads_out)."))
                        continue
                diff("read_stats.tsv", f"{st}.{col}", va, vb)

    for line in unexplained:
        print(line)
    if allowed:
        print("FIELD ALLOWED: " + ", ".join(f"{i} x{len(v)}" for i, v in sorted(allowed.items())) + f" ({sum(len(v) for v in allowed.values())} fields)")
        if show:
            for i, v in sorted(allowed.items()):
                for f, k, va, vb, why in v:
                    print(f"FIELDOK {f} {k} {short(va)} {short(vb)} [{i}]")
    print(f"FIELD CHECK: {len(unexplained)} unexplained")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--show-allowed" in sys.argv)
