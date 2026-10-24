#!/usr/bin/env python
"""Summarise results/track_a_sweep_*.tsv.

    python -P tools/track_a_sweep_summary.py [bits|dec|parts|best|chunks]
"""
import collections, glob, os, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
HDR = "sample member chunk_cols coder nnz prefilter packed payload hdr bpn enc dece decf decfM decM ok model".split()
CODERS = ["z1", "z3", "z6", "z9", "z12", "z15", "z19", "z22", "z22l", "xz", "br", "bz2", "zd3", "zd19"]


def load():
    rows = []
    for fn in sorted(glob.glob(os.path.join(R, "track_a_sweep_*.tsv"))):
        for i, l in enumerate(open(fn)):
            if i == 0:
                continue
            f = l.rstrip("\n").split("\t")
            if len(f) == len(HDR):
                rows.append(dict(zip(HDR, f)))
    return rows


def key(r):
    return (r["sample"], r["member"].replace("_counts.1pz", ""))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "bits"
    rows = load()
    d = collections.defaultdict(dict)
    for r in rows:
        d[(key(r), r["chunk_cols"])][r["coder"]] = r
    cc_list = sorted({r["chunk_cols"] for r in rows}, key=lambda x: (x == "0", int(x)))
    if mode in ("bits", "dec"):
        field = "bpn" if mode == "bits" else "decfM"
        print("payload bits/nnz" if mode == "bits" else "estimated full-decode Mnnz/s (entropy stage of that coder + coder-independent decode work)")
        for cc in cc_list:
            print("\nchunk_cols =", "whole matrix" if cc == "0" else cc)
            print("| file | " + " | ".join(CODERS) + " |")
            print("|---|" + "---|" * len(CODERS))
            for (k, c), v in sorted(d.items()):
                if c != cc:
                    continue
                print("| %s %s | %s |" % (k[0], k[1], " | ".join(("%.3f" % float(v[x][field])) if x in v else "" for x in CODERS)))
    elif mode == "chunks":
        # for the best-per-file coder at each chunk size relative to 1024 (z3 and z19, xz)
        print("payload bytes relative to chunk_cols=1024, per coder\n")
        for coder in ("z3", "z19", "xz"):
            print("\ncoder", coder)
            print("| file | " + " | ".join("cc=" + ("whole" if c == "0" else c) for c in cc_list) + " | bytes@1024 |")
            print("|---|" + "---|" * (len(cc_list) + 1))
            for k in sorted({kk for kk, _ in d}):
                base = d.get((k, "1024"), {}).get(coder)
                if not base:
                    continue
                cells = []
                for c in cc_list:
                    r = d.get((k, c), {}).get(coder)
                    cells.append("%+.1f%%" % (100.0 * (int(r["payload"]) / int(base["payload"]) - 1)) if r else "")
                print("| %s %s | %s | %s |" % (k[0], k[1], " | ".join(cells), "{:,}".format(int(base["payload"]))))
    elif mode == "parts":
        for r in rows:
            if r["coder"] == "parts_z19" and r["chunk_cols"] == "1024":
                print(key(r), r["model"], "total", r["payload"], "bits/nnz", r["bpn"])
    elif mode == "best":
        print("best coder by payload bytes per file/chunk size (and its gain over z3)")
        for (k, c), v in sorted(d.items()):
            cand = {x: int(v[x]["payload"]) for x in v if x in CODERS and not x.startswith("zd")}
            if not cand or "z3" not in cand:
                continue
            b = min(cand, key=cand.get)
            print("%-22s %-6s cc=%-6s best=%-5s %9d  vs z3 %9d  (%+.1f%%)" % (k[0], k[1], c, b, cand[b], cand["z3"], 100.0 * (cand[b] / cand["z3"] - 1)))


if __name__ == "__main__":
    main()
