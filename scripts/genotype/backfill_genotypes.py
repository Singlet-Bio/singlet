#!/usr/bin/env python3
"""Backfill the D-52 genotype outputs of an existing engine run, without rerunning the engine.

Writes, from <run>/private/snp_{ad,dp}.1pz + <run>/cell_calls.tsv + <run>/donor_assignments.tsv, the same content that
include/singlet/pileup/genotype_export.h writes into <run>/private/genotype/ (keep the two in step):

  <dest>/donors.vcf.gz     multi-sample VCF, a column per donor, GT:GQ:AD:DP:PL, called cells with prob_max >= 0.9
  <dest>/cellsnp/          cellsnp-lite layout of the called cells at the callable SNPs (DP over called cells >= 20)

and copies the engine's own genotype files beside them (private/donor/*, private/snp_{ad,dp}.1pz, private/ase_counts.tsv), with a
manifest.json of sha256 sums. gzip bytes differ from zlib's (header), so compare decompressed content with the engine's files.

Private data: run only on Clipper, write only under $WORK/genotypes/. Never copy the outputs anywhere else.

usage: backfill_genotypes.py --run <engine out dir> --dest <dir> --sample <id> [--source <label>]
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np

MIN_COUNT = 20
MIN_PROB = np.float32(0.9)
ERR = 0.01
POST = 0.99


def read_snp(priv, which):
    """private/snp_<which> as (SNPs x barcodes CSC, snp names, barcodes), from .1pz or from --output-format mtx files."""
    p = os.path.join(priv, "snp_%s.1pz" % which)
    if os.path.exists(p):
        return read_1pz(p)
    import scipy.io
    m = scipy.io.mmread(os.path.join(priv, "snp_%s.mtx.gz" % which)).tocsc()
    names = [l.rstrip("\n").split("\t")[0] for l in gzip.open(os.path.join(priv, "snp_%s_features.tsv.gz" % which), "rt")]
    bcs = [l.rstrip("\n").split("\t")[0] for l in gzip.open(os.path.join(priv, "snp_%s_barcodes.tsv.gz" % which), "rt")]
    return m, names, bcs


def read_1pz(path):
    try:
        from singlet._pz import read_1pz as rd
    except ImportError:  # pragma: no cover
        from singlet._io import _read_pz_native
        mat, rows, cols, _ = _read_pz_native(path)
        return mat.tocsc(), list(rows), list(cols)
    import scipy.sparse as sp
    r = rd(path)
    return sp.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(r["m"], r["n"])), list(r["rownames"]), list(r["colnames"])


def parse_snp(name):
    """'chr:pos:R>A' -> (chrom, pos, ref, alt), as parse_snp_name."""
    p = name.split(":")
    chrom, pos = p[0], int(p[1])
    ra = p[2] if len(p) > 2 else "N>N"
    return chrom, pos, ra[0] if ra else "N", ra[2] if len(ra) > 2 else "N"


def donor_fields(ad, dp):
    """genotype_export.h donor_fields, the same expressions in the same order."""
    if dp == 0:
        return "./."
    ad = min(ad, dp)
    a, r = float(ad), float(dp - ad)
    l = [r * math.log1p(-ERR) + a * math.log(ERR), float(dp) * math.log(0.5), a * math.log1p(-ERR) + r * math.log(ERR)]
    m = max(l[0], max(l[1], l[2]))
    e = [math.exp(l[0] - m), math.exp(l[1] - m), math.exp(l[2] - m)]
    z = e[0] + e[1] + e[2]
    best = 0
    for g in (1, 2):
        if e[g] > e[best]:
            best = g
    gt = ("0/0", "0/1", "1/1")[best] if e[best] / z >= POST else "./."
    rest = 0.0
    for g in range(3):
        if g != best:
            rest += e[g]
    q = rest / z
    gq = 99 if q <= 0.0 else int(math.floor(-10.0 * math.log10(q) + 0.5))
    gq = min(gq, 99)
    pl = [int(math.floor(-10.0 * (l[g] - m) / math.log(10.0) + 0.5)) for g in range(3)]
    return "%s:%d:%d,%d:%d:%d,%d,%d" % (gt, gq, dp - ad, ad, dp, pl[0], pl[1], pl[2])


def contig_header(names, rows):
    h, last, seen = [], None, set()
    for s in rows:
        c = parse_snp(names[s])[0]
        if c == last:
            continue
        last = c
        if c in seen:
            continue
        seen.add(c)
        h.append("##contig=<ID=%s>\n" % c)
    return "".join(h)


def called_columns(run, barcodes):
    idx = {b: i for i, b in enumerate(barcodes)}
    cols = set()
    with open(os.path.join(run, "cell_calls.tsv")) as f:
        head = f.readline().rstrip("\n").split("\t")
        ci = head.index("is_cell")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[ci] == "TRUE" and p[0] in idx:
                cols.add(idx[p[0]])
    return sorted(cols)


def write_cellsnp(dest, ad, dp, cells, barcodes, names):
    os.makedirs(dest, exist_ok=True)
    adc, dpc = ad[:, cells].tocsr(), dp[:, cells].tocsr()
    agg_dp = np.asarray(dpc.sum(axis=1), dtype=np.uint64).ravel()
    agg_ad = np.asarray(adc.sum(axis=1), dtype=np.uint64).ravel()
    rows = [s for s in np.flatnonzero(agg_dp >= MIN_COUNT).tolist() if s < len(names)]
    with gzip.open(os.path.join(dest, "cellSNP.base.vcf.gz"), "wt", compresslevel=6) as f:
        f.write("##fileformat=VCFv4.2\n##source=singlet genotype_export (cellsnp-lite layout)\n"
                "##singlet_callable=DP summed over called cells >= %d\n" % MIN_COUNT)
        f.write(contig_header(names, rows))
        f.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="total counts for ALT and REF">\n'
                '##INFO=<ID=AD,Number=1,Type=Integer,Description="total counts for ALT">\n'
                '##INFO=<ID=OTH,Number=1,Type=Integer,Description="total counts for other bases from REF and ALT">\n'
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for s in rows:
            c, p, r, a = parse_snp(names[s])
            f.write("%s\t%d\t.\t%s\t%s\t.\tPASS\tAD=%d;DP=%d;OTH=0\n" % (c, p, r, a, agg_ad[s], agg_dp[s]))
    with open(os.path.join(dest, "cellSNP.samples.tsv"), "w") as f:
        for c in cells:
            f.write(barcodes[c] + "\n")
    sel = np.asarray(rows, dtype=np.int64)
    for tag, m in (("AD", adc), ("DP", dpc)):
        sub = m[sel, :].tocoo() if len(rows) else None
        with open(os.path.join(dest, "cellSNP.tag.%s.mtx" % tag), "w") as f:
            if sub is None:
                f.write("%%%%MatrixMarket matrix coordinate integer general\n%%\n0\t%d\t0\n" % len(cells))
                continue
            keep = sub.data > 0
            r, c, v = sub.row[keep], sub.col[keep], sub.data[keep]
            o = np.lexsort((c, r))
            f.write("%%%%MatrixMarket matrix coordinate integer general\n%%\n%d\t%d\t%d\n" % (len(rows), len(cells), len(o)))
            f.writelines("%d\t%d\t%d\n" % (r[i] + 1, c[i] + 1, v[i]) for i in o)
    with open(os.path.join(dest, "cellSNP.tag.OTH.mtx"), "w") as f:
        f.write("%%%%MatrixMarket matrix coordinate integer general\n%%\n%d\t%d\t0\n" % (len(rows), len(cells)))
    return len(rows)


def write_donors(path, run, ad, dp, cells, barcodes, names):
    lab = {}
    with open(os.path.join(run, "donor_assignments.tsv")) as f:
        head = f.readline().rstrip("\n").split("\t")
        i_lab, i_p, i_st = head.index("donor_id"), head.index("prob_max"), head.index("cell_status")
        for line in f:
            p = line.rstrip("\n").split("\t")
            lab[p[0]] = (p[i_lab], np.float32(float(p[i_p])), p[i_st])
    donors = sorted({int(v[0][5:]) for v in lab.values() if v[0].startswith("donor") and v[0][5:].isdigit()})
    if not donors:
        return 0, 0
    K = max(donors) + 1
    use = []
    for c in cells:
        v = lab.get(barcodes[c])
        if v is None or v[2] != "cell" or not v[0].startswith("donor") or not v[0][5:].isdigit():
            continue
        if v[1] < MIN_PROB:
            continue
        use.append((c, int(v[0][5:])))
    if not use:
        return 0, K
    n = dp.shape[0]
    sdp = np.zeros((n, K), dtype=np.int64)
    sad = np.zeros((n, K), dtype=np.int64)
    for k in range(K):
        cs = [c for c, kk in use if kk == k]
        if cs:
            sdp[:, k] = np.asarray(dp[:, cs].sum(axis=1)).ravel()
            sad[:, k] = np.asarray(ad[:, cs].sum(axis=1)).ravel()
    rows = [s for s in np.flatnonzero(sdp.sum(axis=1) > 0).tolist() if s < len(names)]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", compresslevel=6) as f:
        f.write("##fileformat=VCFv4.2\n##source=singlet genotype_export\n"
                "##singlet_genotype_model=binomial on donor-summed UMI counts, error 0.01, flat prior; GT when the posterior >= 0.99\n"
                "##singlet_cells=called cells (cell_status cell) with donor prob_max >= 0.9\n")
        f.write(contig_header(names, rows))
        f.write('##FILTER=<ID=PASS,Description="All filters passed">\n'
                '##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of donors with coverage at the site">\n'
                '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype; ./. when the best posterior is below 0.99">\n'
                '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Phred-scaled 1 - posterior of the most likely genotype, capped at 99">\n'
                '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Ref and alt UMI counts summed over the donor\'s cells">\n'
                '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Ref + alt UMI count summed over the donor\'s cells">\n'
                '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Phred-scaled genotype likelihoods, normalized to the best">\n'
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
        f.write("".join("\tdonor%d" % k for k in range(K)) + "\n")
        for s in rows:
            c, p, r, a = parse_snp(names[s])
            d, x = sdp[s], sad[s]
            ns = int((d > 0).sum())
            f.write("%s\t%d\t%s\t%s\t%s\t.\tPASS\tNS=%d\tGT:GQ:AD:DP:PL\t%s\n" % (
                c, p, names[s], r, a, ns, "\t".join(donor_fields(int(x[k]), int(d[k])) for k in range(K))))
    return len(rows), K


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--prewritten", default="", help="stats JSON of genotype_backfill: its outputs are already in --dest")
    a = ap.parse_args()
    if "/genotypes/" not in os.path.abspath(a.dest) + "/":
        sys.exit("refusing: genotype outputs go only under $WORK/genotypes/")
    priv = os.path.join(a.run, "private")
    os.makedirs(a.dest, exist_ok=True)
    os.chmod(a.dest, 0o700)
    if a.prewritten:   # donors.vcf.gz + cellsnp/ already written by genotype_backfill (the engine's writers): only copy and checksum
        with open(a.prewritten) as f:
            st = json.loads(f.read().strip().splitlines()[-1])
        writer = "scripts/genotype/genotype_backfill.cpp (genotype_export.h)"
    else:
        ad, names, barcodes = read_snp(priv, "ad")
        dp, names2, barcodes2 = read_snp(priv, "dp")
        assert names == names2 and barcodes == barcodes2 and ad.shape == dp.shape
        cells = called_columns(a.run, barcodes)
        n_callable = write_cellsnp(os.path.join(a.dest, "cellsnp"), ad, dp, cells, barcodes, names)
        n_sites, K = write_donors(os.path.join(a.dest, "donors.vcf.gz"), a.run, ad, dp, cells, barcodes, names)
        st = {"n_barcodes": len(barcodes), "n_called_cells": len(cells), "n_panel_snps": len(names), "n_callable_sites": n_callable,
              "n_donors": K, "n_donor_sites": n_sites}
        writer = "scripts/genotype/backfill_genotypes.py"
    eng = os.path.join(a.dest, "engine")
    for rel in ("donor", "snp_ad.1pz", "snp_dp.1pz", "ase_counts.tsv", "snp_ad.mtx.gz", "snp_dp.mtx.gz", "snp_ad_features.tsv.gz",
                "snp_ad_barcodes.tsv.gz", "snp_dp_features.tsv.gz", "snp_dp_barcodes.tsv.gz"):
        src = os.path.join(priv, rel)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(eng, rel), dirs_exist_ok=True)
        elif os.path.exists(src):
            os.makedirs(eng, exist_ok=True)
            shutil.copy2(src, os.path.join(eng, rel))
    for rel in ("donor_assignments.tsv", "cell_calls.tsv", "summary.json", "sex_call.json", "ancestry_call.json"):
        if os.path.exists(os.path.join(a.run, rel)):
            os.makedirs(eng, exist_ok=True)
            shutil.copy2(os.path.join(a.run, rel), os.path.join(eng, rel))
    files = {}
    for dpath, _d, fns in os.walk(a.dest):
        for fn in fns:
            if fn == "manifest.json":
                continue
            p = os.path.join(dpath, fn)
            files[os.path.relpath(p, a.dest)] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    man = {"sample": a.sample, "source": a.source, "run": os.path.abspath(a.run), **st, "writer": writer, "files": dict(sorted(files.items()))}
    with open(os.path.join(a.dest, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps({k: v for k, v in man.items() if k != "files"}))


if __name__ == "__main__":
    main()
