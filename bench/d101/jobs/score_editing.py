"""Score per-site A-to-I editing levels from single-cell reads against bulk RNA-seq of the same donor.

Level at a + strand site = G / (A + G); at a - strand site = C / (T + C) (REDIportal sites, genome strand). Singlet's level comes from
its UMI counts (edited_umis / (ref_umis + edited_umis)); REDItools3 levels come from read BaseCount [A, C, G, T]. Sites are scored when
bulk has >= --min-bulk reads and the single-cell method has >= --min-sc; the shared set is where both single-cell methods qualify.
Reports Pearson r, Spearman rho and mean absolute error for each method on its own qualifying set and on the shared set.
"""
import argparse
import gzip
import json
import math


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2.0
        i = j + 1
    return r


def stats(pairs):
    if not pairs:
        return {"sites": 0}
    x = [a for a, _ in pairs]
    y = [b for _, b in pairs]
    return {"sites": len(pairs), "pearson": pearson(x, y), "spearman": pearson(ranks(x), ranks(y)),
            "mae": sum(abs(a - b) for a, b in pairs) / len(pairs), "mean_sc": sum(x) / len(x), "mean_bulk": sum(y) / len(y)}


def read_singlet(path):
    out = {}
    with gzip.open(path, "rt") as f:
        h = f.readline().rstrip("\n").split("\t")
        ix = {k: i for i, k in enumerate(h)}
        for line in f:
            v = line.rstrip("\n").split("\t")
            chrom, pos, _ = v[ix["site"]].split(":")
            ref, alt = int(v[ix["ref_umis"]]), int(v[ix["edited_umis"]])
            if ref + alt:
                out[(chrom, int(pos))] = (alt / (ref + alt), ref + alt, v[ix["strand"]])
    return out


def read_reditools(path, strand_of):
    out = {}
    with open(path) as f:
        h = f.readline().rstrip("\n").split("\t")
        ix = {k: i for i, k in enumerate(h)}
        for line in f:
            v = line.rstrip("\n").split("\t")
            key = (v[ix["Region"]], int(v[ix["Position"]]))
            s = strand_of.get(key)
            if s is None:
                continue
            a, c, g, t = (int(x) for x in v[ix["BaseCount[A,C,G,T]"]].strip("[]").split(","))
            ref, alt = (a, g) if s == "+" else (t, c)
            if ref + alt:
                out[key] = (alt / (ref + alt), ref + alt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singlet", required=True)
    ap.add_argument("--sc", required=True)
    ap.add_argument("--bulk", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--min-bulk", type=int, default=10)
    ap.add_argument("--min-sc", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sg = read_singlet(a.singlet)
    strand = {k: v[2] for k, v in sg.items()}
    rs = read_reditools(a.sc, strand)
    bk = read_reditools(a.bulk, strand)
    good_bulk = {k: v[0] for k, v in bk.items() if v[1] >= a.min_bulk}
    s_pairs = {k: (sg[k][0], good_bulk[k]) for k in good_bulk if k in sg and sg[k][1] >= a.min_sc}
    r_pairs = {k: (rs[k][0], good_bulk[k]) for k in good_bulk if k in rs and rs[k][1] >= a.min_sc}
    shared = sorted(set(s_pairs) & set(r_pairs))
    res = {
        "donor": a.label, "min_bulk_reads": a.min_bulk, "min_sc_depth": a.min_sc,
        "sites_singlet_covered": len(sg), "sites_bulk_qualifying": len(good_bulk),
        "singlet": stats(list(s_pairs.values())), "reditools3_sc": stats(list(r_pairs.values())),
        "shared": {"singlet": stats([s_pairs[k] for k in shared]), "reditools3_sc": stats([r_pairs[k] for k in shared])},
        "level_definition": "edited / (ref + edited); Singlet in UMIs, REDItools3 in reads; + sites A>G, - sites T>C",
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
