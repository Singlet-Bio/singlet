#!/usr/bin/env python
"""Keep notes/track_a.md current: every block between <!-- TABLE_X --> and <!-- /TABLE_X --> is regenerated from the
result TSVs (a bare TABLE_X token is converted into such a block on first use).

    python -P tools/track_a_notes.py            # refresh all blocks that have data
"""
import csv, io, os, re, sys
from collections import defaultdict
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import track_a_tables as T

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NOTES = os.path.join(ROOT, "notes", "track_a.md")
R = os.path.join(ROOT, "results")


def capture(fn, *a, **k):
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*a, **k)
    return buf.getvalue().strip()


def have(name):
    p = os.path.join(R, name)
    return os.path.exists(p) and os.path.getsize(p) > 0


def container():
    rows = T.load("track_a_container.tsv")
    print("| file | original | re-emitted (zstd 1.5.5) | delta | sections | byte-identical | same content, other zstd bytes | chunks orig / re | meta orig / re | perm orig / re |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %s %s | %s | %s | %+d | %s | %s | %s | %s / %s | %s / %s | %s / %s |" % (
            r["sample"], r["member"].replace("_counts.1pz", ""), "{:,}".format(int(r["orig_bytes"])), "{:,}".format(int(r["re_bytes"])), int(r["delta_bytes"]),
            r["sections"], r["sec_identical"], r["sec_zstd_only_diff"], "{:,}".format(int(r["chunks_orig"])), "{:,}".format(int(r["chunks_re"])),
            "{:,}".format(int(r["meta_orig"])), "{:,}".format(int(r["meta_re"])), "{:,}".format(int(r["perm_orig"])), "{:,}".format(int(r["perm_re"]))))


def sweep_all():
    rows = []
    for fn in ("track_a_sweep_exon.tsv", "track_a_sweep_intron.tsv"):
        if have(fn):
            rows += T.load(fn)
    if not rows:
        print("(sweep still running)")
        return
    coders = []
    for r in rows:
        if r["coder"] not in coders and not r["coder"].startswith("parts") and r["coder"] != "non_entropy_decode":
            coders.append(r["coder"])
    by = defaultdict(dict)
    files = []
    for r in rows:
        key = (r["sample"], r["member"].replace("_counts.1pz", ""))
        if key not in files:
            files.append(key)
        by[(key, r["chunk_cols"])][r["coder"]] = r
    label = {"z1": "zstd-1", "z3": "zstd-3 (today)", "z6": "zstd-6", "z9": "zstd-9", "z12": "zstd-12", "z15": "zstd-15", "z19": "zstd-19", "z22": "zstd-22",
             "z22l": "zstd-22 w27+LDM", "xz": "xz -9e", "br": "brotli-11 w24", "bz2": "bzip2 -9", "zd3": "zstd-3 + dict", "zd19": "zstd-19 + dict"}
    print("**(a) Payload bits/nnz at chunk_cols = 1024 (the production chunking), per coder.** Payload = sum of the entropy-coder outputs; add 24 B per chunk for headers. Dictionary rows are blank where no out-of-study training data with the same vocabulary exists (mouse).\n")
    print("| file | nnz | packed bytes in | " + " | ".join(label.get(c, c) for c in coders) + " |")
    print("|---|---|---|" + "---|" * len(coders))
    for f in files:
        d = by.get((f, "1024"))
        if not d:
            continue
        any_ = next(iter(d.values()))
        print("| %s %s | %s | %s | %s |" % (f[0], f[1], "{:,}".format(int(any_["nnz"])), "{:,}".format(int(any_["packed_bytes"])),
                                          " | ".join(("%.2f" % float(d[c]["bits_per_nnz_payload"])) if c in d else "" for c in coders)))
    print("\n**(b) Relative payload size vs zstd-3 (today), chunk_cols 1024.** Mean over the files that have the coder.\n")
    print("| coder | " + " | ".join("%s %s" % f for f in files) + " | mean |")
    print("|---|" + "---|" * (len(files) + 1))
    for c in coders:
        vals = []
        cells = []
        for f in files:
            d = by.get((f, "1024"))
            if d and c in d and "z3" in d:
                v = float(d[c]["payload_bytes"]) / float(d["z3"]["payload_bytes"])
                vals.append(v)
                cells.append("%.3f" % v)
            else:
                cells.append("")
        print("| %s | %s | %s |" % (label.get(c, c), " | ".join(cells), ("%.3f" % (sum(vals) / len(vals))) if vals else ""))
    print("\n**(c) Entropy-stage decode speed (Mnnz/s of the decompression alone, chunk_cols 1024, single thread, min of 3, noisy machine) and encode time (ms).**\n")
    print("| file | " + " | ".join(label.get(c, c) for c in coders) + " |")
    print("|---|" + "---|" * len(coders))
    for f in files:
        d = by.get((f, "1024"))
        if not d:
            continue
        print("| %s %s | %s |" % (f[0], f[1], " | ".join(("%.0f / %.0f ms" % (float(d[c]["dec_entropy_Mnnz_s"]), float(d[c]["enc_ms"]))) if c in d else "" for c in coders)))
    print("\n**(d) Chunk size: payload bits/nnz by chunk_cols for zstd-3, zstd-19 and xz (0 = whole matrix in one chunk), and the coder-independent decode work (ms) at each chunk size.**\n")
    print("| file | 256 z3 | 1024 z3 | 4096 z3 | whole z3 | 256 z19 | 1024 z19 | 4096 z19 | whole z19 | 256 xz | 1024 xz | 4096 xz | whole xz | non-entropy decode ms 256 / 1024 / 4096 / whole |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in files:
        cells = []
        for c in ("z3", "z19", "xz"):
            for cc in ("256", "1024", "4096", "0"):
                d = by.get((f, cc))
                cells.append(("%.2f" % float(d[c]["bits_per_nnz_payload"])) if d and c in d else "")
        ne = []
        for cc in ("256", "1024", "4096", "0"):
            d = by.get((f, cc))
            ne.append(("%.0f" % float(d["non_entropy_decode"]["dec_full_ms_est"])) if d and "non_entropy_decode" in d else "")
        print("| %s %s | %s | %s |" % (f[0], f[1], " | ".join(cells), " / ".join(ne)))
    print("\n**(e) What is inside the prefilter bytes** (chunk_cols 1024; each logical stream compressed alone with zstd-19): varint metadata (group counts, values, run lengths), the 8 bit planes of the low gap byte, the upper gap byte(s).\n")
    print("| file | meta raw -> z19 | bit planes raw -> z19 | upper bytes raw -> z19 | sum of parts (bits/nnz) | joint zstd-19 (bits/nnz) |")
    print("|---|---|---|---|---|---|")
    for f in files:
        d = by.get((f, "1024"))
        if not d or "parts_z19" not in d:
            continue
        p = dict(kv.split("=") for kv in d["parts_z19"]["model_bytes"].split(";"))
        print("| %s %s | %s | %s | %s | %.2f | %s |" % (f[0], f[1], "{:,}".format(int(p["meta"])), "{:,}".format(int(p["planes"])), "{:,}".format(int(p["upper"])),
                                                       float(d["parts_z19"]["bits_per_nnz_payload"]), ("%.2f" % float(d["z19"]["bits_per_nnz_payload"])) if "z19" in d else ""))


def bundle_whatif():
    if not have("track_a_bundle_whatif.tsv"):
        print("(bundle what-if still running)")
        return
    rows = T.load("track_a_bundle_whatif.tsv")
    tot = defaultdict(lambda: defaultdict(int))
    rh = defaultdict(set)
    for r in rows:
        kind = r["file"].split("__")[1]
        t = tot[kind]
        t["files"] += 1
        for k in ("bytes", "perm_z", "ptr_z", "colsums_z", "meta_z", "rowcounts_best", "whitelist_idx_bytes", "nnz"):
            t[k] += int(r[k])
        rh[kind].add(r["rownames_hash"])
    print("| member | files | bytes today | perm today | row counts instead | ptr+colsums today (derivable) | metadata today | distinct row-name lists | barcodes as whitelist index (once per sample) | bytes if names->model, perm->counts, ptr/colsums dropped | of today |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    gtot = gnew = 0
    for kind, t in tot.items():
        new = t["bytes"] - t["perm_z"] - t["ptr_z"] - t["colsums_z"] - t["meta_z"] + t["rowcounts_best"] + t["whitelist_idx_bytes"] + 8 * t["files"]
        if len(rh[kind]) > 1:   # per-sample row names (sj/psi): keep them, best generic coder is about 0.57x of zstd-3 (section 2.2)
            pass
        gtot += t["bytes"]; gnew += new
        print("| %s | %d | %s | %s | %s | %s | %s | %d | %s | %s | %.1f%% |" % (
            kind, t["files"], "{:,}".format(t["bytes"]), "{:,}".format(t["perm_z"]), "{:,}".format(t["rowcounts_best"]), "{:,}".format(t["ptr_z"] + t["colsums_z"]),
            "{:,}".format(t["meta_z"]), len(rh[kind]), "{:,}".format(t["whitelist_idx_bytes"]), "{:,}".format(new), 100.0 * new / t["bytes"]))
    print("| **all matrices** | | %s | | | | | | | %s | %.1f%% |" % ("{:,}".format(gtot), "{:,}".format(gnew), 100.0 * gnew / gtot))


BLOCKS = {
    "TABLE_ACCOUNT": lambda: capture(T.account),
    "TABLE_NAMES_ROWS": lambda: capture(T.names).split("### Barcodes")[0].split("\n", 2)[2].strip(),
    "TABLE_NAMES_COLS": lambda: capture(T.names).split("### Barcodes")[1].split("\n", 2)[2].strip(),
    "TABLE_PERM": lambda: capture(T.perm),
    "TABLE_CONTAINER": lambda: capture(container),
    "TABLE_PROFILE": lambda: capture(T.profile),
    "TABLE_SWEEP": lambda: capture(sweep_all),
    "TABLE_RUN": lambda: capture(T.run) if have("track_a_run.tsv") else "(codec run not finished)",
    "TABLE_CODEC_SUMMARY": lambda: capture(T.codec_summary),
    "TABLE_SPEED": lambda: capture(T.speed),
    "TABLE_BUNDLE_WHATIF": lambda: capture(bundle_whatif),
}


def main():
    text = open(NOTES, encoding="utf-8").read()
    for name, fn in BLOCKS.items():
        try:
            body = fn()
        except Exception as e:
            body = "(not available yet: %s)" % e
        block = "<!-- %s -->\n%s\n<!-- /%s -->" % (name, body, name)
        pat = re.compile(r"<!-- %s -->.*?<!-- /%s -->" % (name, name), re.S)
        if pat.search(text):
            text = pat.sub(lambda m: block, text)
        elif re.search(r"^%s$" % name, text, re.M):
            text = re.sub(r"^%s$" % name, lambda m: block, text, flags=re.M)
    open(NOTES, "w", encoding="utf-8").write(text)
    print("notes refreshed")


if __name__ == "__main__":
    main()
