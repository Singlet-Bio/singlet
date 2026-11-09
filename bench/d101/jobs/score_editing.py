"""Score per-site A-to-I editing levels from single-cell reads against bulk RNA-seq of the same donor.

Level at a + strand site = G / (A + G); at a - strand site = C / (T + C) (REDIportal sites, genome strand). Singlet's level comes from
its UMI counts (edited_umis / (ref_umis + edited_umis)); REDItools3 levels come from read BaseCount [A, C, G, T]. Sites are scored when
bulk has >= --min-bulk reads and the single-cell method has >= --min-sc; the shared set is where both single-cell methods qualify.
Reports Pearson r, Spearman rho and mean absolute error for each method on its own qualifying set and on the shared set.

With --sites (chrom pos strand bulk_depth sc_depth from reditools_neutral.sbatch) the primary verdict is on a neutral set chosen
independently of both tools: bulk REDItools3 >= --min-bulk reads and raw single-cell BAM depth >= --min-sc. Each tool's coverage of that
set (its own depth >= --min-sc) and its levels on the covered part and on the part both tools cover are reported. The Singlet-selected
lines above stay as a secondary result.
"""
import argparse
import gzip
import json
import math
import random


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
                out[key] = (alt / (ref + alt), ref + alt, alt)
    return out


DEPTH_BINS = [(5, 10), (10, 20), (20, 50), (50, None)]


def downsample(alt, depth, n, rng):
    """Edited count after drawing n of depth reads without replacement (hypergeometric)."""
    if n >= depth:
        return alt
    a, d, k = alt, depth, 0
    for _ in range(n):
        if rng.random() < a / d:
            k += 1
            a -= 1
        d -= 1
    return k


def matched_depth(both, sg, rs, bulk, seed=1):
    """Both-covered sites at matched depth: REDItools3 reads downsampled to Singlet's UMI count per site, and both tools
    stratified by Singlet UMI depth, so reads-versus-UMIs depth does not decide the comparison."""
    rng = random.Random(seed)
    ds = []
    for k in both:
        n = sg[k][1]
        e = downsample(rs[k][2], rs[k][1], n, rng)
        ds.append((e / min(n, rs[k][1]), bulk[k]))
    bins = {}
    for lo, hi in DEPTH_BINS:
        ks = [k for k in both if sg[k][1] >= lo and (hi is None or sg[k][1] < hi)]
        bins[f"{lo}-{hi - 1}" if hi else f"{lo}+"] = {"singlet": stats([(sg[k][0], bulk[k]) for k in ks]),
                                                      "reditools3_sc": stats([(rs[k][0], bulk[k]) for k in ks])}
    out = {"reditools3_downsampled_to_singlet_umis": stats(ds), "singlet": stats([(sg[k][0], bulk[k]) for k in both]),
           "by_singlet_umi_depth": bins, "seed": seed}
    out["pearson_difference_singlet_minus_reditools3"] = paired_bootstrap([(sg[k][0], d[0], bulk[k]) for k, d in zip(both, ds)], seed)
    return out


def paired_bootstrap(rows, seed, n_boot=500):
    """Singlet minus REDItools3 Pearson r with bulk, resampling sites (paired). rows = (singlet level, reditools level, bulk level)."""
    try:
        import numpy as np
    except ImportError:
        return None
    if len(rows) < 100:
        return None
    a = np.array(rows, dtype=float)
    r = np.random.default_rng(seed)

    def corr(x, y):
        x = x - x.mean(); y = y - y.mean()
        d = math.sqrt((x * x).sum() * (y * y).sum())
        return float((x * y).sum() / d) if d > 0 else 0.0
    diff = corr(a[:, 0], a[:, 2]) - corr(a[:, 1], a[:, 2])
    bs = []
    for _ in range(n_boot):
        ix = r.integers(0, len(a), len(a)); b = a[ix]
        bs.append(corr(b[:, 0], b[:, 2]) - corr(b[:, 1], b[:, 2]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"difference": diff, "ci95": [float(lo), float(hi)], "bootstrap": n_boot, "sites": len(a)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--singlet", required=True)
    ap.add_argument("--sc", required=True)
    ap.add_argument("--bulk", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--min-bulk", type=int, default=10)
    ap.add_argument("--min-sc", type=int, default=5)
    ap.add_argument("--sites", default=None, help="neutral site table (chrom pos strand bulk_depth sc_depth), gzip")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sg = read_singlet(a.singlet)
    strand = {k: v[2] for k, v in sg.items()}
    neutral_dp = {}
    if a.sites:
        with gzip.open(a.sites, "rt") as f:
            for line in f:
                c, pos, st, bd, sd = line.rstrip("\n").split("\t")
                strand[(c, int(pos))] = st
                neutral_dp[(c, int(pos))] = int(sd)
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
        "shared": {"singlet": stats([s_pairs[k] for k in shared]), "reditools3_sc": stats([r_pairs[k] for k in shared]),
                   "matched_depth": matched_depth(shared, sg, rs, good_bulk)},
        "level_definition": "edited / (ref + edited); Singlet in UMIs, REDItools3 in reads; + sites A>G, - sites T>C",
    }
    if a.sites:
        neutral = sorted(k for k in good_bulk if neutral_dp.get(k, 0) >= a.min_sc)
        s_cov = [k for k in neutral if k in sg and sg[k][1] >= a.min_sc]
        r_cov = [k for k in neutral if k in rs and rs[k][1] >= a.min_sc]
        both = sorted(set(s_cov) & set(r_cov))
        n = len(neutral)
        res["primary_neutral"] = {
            "definition": f"REDIportal panel sites with bulk REDItools3 >= {a.min_bulk} reads and raw single-cell BAM depth "
                          f"(called cells, samtools depth) >= {a.min_sc}; chosen independently of Singlet and REDItools3",
            "sites": n,
            "singlet": {"coverage": len(s_cov) / n if n else None, **stats([(sg[k][0], good_bulk[k]) for k in s_cov])},
            "reditools3_sc": {"coverage": len(r_cov) / n if n else None, **stats([(rs[k][0], good_bulk[k]) for k in r_cov])},
            "both_covered": {"singlet": stats([(sg[k][0], good_bulk[k]) for k in both]),
                             "reditools3_sc": stats([(rs[k][0], good_bulk[k]) for k in both])},
            "both_covered_matched_depth": matched_depth(both, sg, rs, good_bulk),
        }
        sel = {k for k in good_bulk if k in sg and sg[k][1] >= a.min_sc}
        res["sample_membership"] = {
            "sampled_sites_with_bulk_output": len(bk), "bulk_qualifying": len(good_bulk),
            "neutral": n, "singlet_selected": len(sel), "in_both_sets": len(sel & set(neutral)),
            "note": "the verdict uses the neutral set only; the Singlet-selected set is the secondary line; the union is never scored"}
        res["secondary_note"] = "singlet / reditools3_sc / shared above are on sites Singlet selected (Singlet depth >= min_sc)"
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
