#!/usr/bin/env python3
"""Score B20-at-scale calls (calls.tsv of run_scale.sbatch) against the corpus's stated assay.

Prints, per binary, agreement by stated assay, and for every disagreement the evidence and a suggested verdict
(detector wrong, metadata wrong, unclear), which a reader confirms from the evidence.
"""
import re
import sys
from collections import defaultdict

EXPECT = {
    "10x_3p_v3": r"^10x-3p-v[34]$", "10x_3p_v4": r"^10x-3p-v[34]$", "10x_3p_v2": r"^10x-3p-v2$", "10x_3p_v1": r"^10x-3p-v1$",
    "10x_5p": r"^10x-5p-", "10x_multiome": r"^10x-arc-gex$", "visium": r"^10x-visium$", "dropseq": r"^(dropseq|seqwell)$",
    "seqwell": r"^(dropseq|seqwell)$", "bd_rhapsody": r"^bd-rhapsody$", "indrop": r"^indrop$", "dnbelab_c4": r"^dnbelab-c4$",
    "singleron": r"^singleron",
}


def ev(s):
    """'R1 5'-TSO a / polyT b R2 3'-TSO c polyA d polyT e of n spots' -> dict of rates"""
    m = re.search(r"5'-TSO (\d+) / polyT (\d+) R2 3'-TSO (\d+) polyA (\d+) polyT (\d+) of (\d+)", s or "")
    if not m:
        return None
    a = [int(x) for x in m.groups()]
    n = max(a[5], 1)
    return {"r1tso": a[0], "r1polyt": a[1], "tso": a[2] / n, "polya": a[3] / n, "polyt": a[4] / n, "n": n}


def verdict(assay, tag, wl, e):
    three, five = tag.startswith("10x-3p"), tag.startswith("10x-5p")
    if assay == "10x_5p" and three and e:
        return "metadata wrong (3' TSO in R2)" if e["tso"] >= 0.005 else "detector wrong (no 3' TSO)"
    if assay.startswith("10x_3p") and five and e:
        return "detector wrong (3' TSO present)" if e["tso"] >= 0.002 else "unclear (no 3' TSO: 5' library or trimmed R2)"
    if assay.startswith("10x_") and (three or five) and wl not in ("-", "") and int(wl) >= 50:
        return "metadata wrong (whitelist and R1 length fit the call)"
    if tag in ("-", "") or tag == "unknown":
        return "unclear (no call)"
    return "unclear"


def main(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path)][1:]
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    dis = defaultdict(list)
    for r in rows:
        run, sample, study, assay, b, tag, conf, wl, enc, flen, pairs, evs = (r + ["-"] * 12)[:12]
        if b == "-":
            by["(no pair)"][assay][1] += 1
            continue
        ok = bool(re.search(EXPECT.get(assay, "^$"), tag.replace("gate:", "")))
        by[b][assay][0] += ok
        by[b][assay][1] += 1
        if not ok:
            e = ev(evs)
            dis[b].append((assay, run, study, tag, conf, wl, enc, flen, evs, verdict(assay, tag, wl, e)))
    for b in sorted(by):
        print("== %s" % b)
        tot = [0, 0]
        for a in sorted(by[b]):
            k, n = by[b][a]
            tot[0] += k; tot[1] += n
            print("  %-14s %3d / %3d agree" % (a, k, n))
        print("  %-14s %3d / %3d agree" % ("all", *tot))
    for b in sorted(dis):
        print("\n== disagreements, %s (%d)" % (b, len(dis[b])))
        print("assay\trun\tstudy\tcall\tconf\twl%\tencoded R1/R2\tfile R1/R2\tevidence\tsuggested verdict")
        for d in sorted(dis[b]):
            print("\t".join(str(x) for x in d))
        v = defaultdict(lambda: defaultdict(int))
        for d in dis[b]:
            fam = "5p" if d[0] == "10x_5p" else "3p" if d[0].startswith("10x_3p") else "other"
            v[fam][d[9].split(" (")[0]] += 1
        for fam in ("3p", "5p", "other"):
            n = sum(by[b][a][1] for a in by[b] if (a == "10x_5p") == (fam == "5p") and (a.startswith("10x_3p")) == (fam == "3p"))
            print("%s: %d samples; suggested verdicts %s" % (fam, n, dict(v[fam])))


if __name__ == "__main__":
    main(sys.argv[1])
