#!/usr/bin/env python3
"""D-101 donor genotypes of a one-donor library against the donor's 1000 Genomes genotypes (counts only; nothing per site leaves).

  score_genotypes.py <sample id> <1000G dir> <singlet out dir> <cellsnp-lite de novo dir> <out.json> [bcftools]

Calls compared, all on the same reads and the engine's called cells:
  singlet_panel    private/genotype/donors.vcf.gz, donor0 (fixed 1000G panel, AF >= 5%)
  singlet_allsite  private/genotype/donors_allsite.vcf.gz, donor0 (sites with ALT evidence in the sample)
  singlet_union    both; a position in both keeps the panel call
  cellsnp_denovo   cellsnp-lite mode 2 (no panel) on the engine's tagged BAM, AD / DP summed over the called cells, genotyped with the
                   engine's model (binomial, error 0.01, flat prior, GT when the posterior >= 0.99): the "cellsnp-pileup" comparator
Truth: the 1000 Genomes GRCh38 genotypes of the donor (biallelic SNVs, autosomes). A position absent from the 1000 Genomes file is
hom-ref in the truth (absent_* counts how many calls rest on that). Metrics per stratum of donor depth (DP >= 10, >= 20):
concordance of the genotype class, false-alt rate (non-ref calls at truth 0/0), non-ref concordance, and the number of truth non-ref
sites called correctly.
"""
import gzip, json, math, os, subprocess, sys
from collections import defaultdict
import numpy as np
import scipy.io

AUTO = {f"chr{i}" for i in range(1, 23)}


def gt_index(ad, dp, err=0.01, post=0.99):
    if dp <= 0: return -1
    ad = min(ad, dp); r = dp - ad
    l = [r * math.log1p(-err) + ad * math.log(err), dp * math.log(0.5), ad * math.log1p(-err) + r * math.log(err)]
    m = max(l); e = [math.exp(x - m) for x in l]; b = int(np.argmax(e))
    return b if e[b] / sum(e) >= post else -1


def vcf_calls(path, col=9):
    out = {}
    if not os.path.exists(path): return out
    with gzip.open(path, "rt") as f:
        for l in f:
            if l[0] == "#": continue
            t = l.rstrip("\n").split("\t")
            if t[0] not in AUTO: continue
            fmt = t[8].split(":"); v = t[col].split(":")
            gt = v[fmt.index("GT")]; dp = int(v[fmt.index("DP")]) if v[fmt.index("DP")] not in (".", "") else 0
            g = {"0/0": 0, "0/1": 1, "1/1": 2}.get(gt, -1)
            out[(t[0], int(t[1]))] = (t[4], g, dp)
    return out


def cellsnp_calls(d):
    base = []
    with gzip.open(os.path.join(d, "cellSNP.base.vcf.gz"), "rt") as f:
        for l in f:
            if l[0] != "#": t = l.split("\t", 6); base.append((t[0], int(t[1]), t[4]))
    AD = scipy.io.mmread(os.path.join(d, "cellSNP.tag.AD.mtx")).tocsr()
    DP = scipy.io.mmread(os.path.join(d, "cellSNP.tag.DP.mtx")).tocsr()
    ad = np.asarray(AD.sum(1)).ravel(); dp = np.asarray(DP.sum(1)).ravel()
    return {(c, p): (a, gt_index(int(x), int(y)), int(y)) for (c, p, a), x, y in zip(base, ad, dp) if c in AUTO}


def truth(sample, tdir, positions, bcftools, tmp):
    by = defaultdict(list)
    for c, p in positions: by[c].append(p)
    out = {}
    for c, ps in by.items():
        f = [os.path.join(tdir, x) for x in os.listdir(tdir) if x.startswith(f"ALL.{c}.") and x.endswith(".vcf.gz")]
        if not f: continue
        f = f[0]
        hdr = subprocess.run([bcftools, "view", "-h", f], capture_output=True, text=True).stdout
        name = c if f"##contig=<ID={c}," in hdr or f"##contig=<ID={c}>" in hdr else c[3:]
        reg = os.path.join(tmp, f"reg.{c}.tsv")
        with open(reg, "w") as w:
            for p in sorted(set(ps)): w.write(f"{name}\t{p}\n")
        q = subprocess.run([bcftools, "query", "-s", sample, "-R", reg, "-i", 'TYPE="snp"', "-f", "%POS\t%REF\t%ALT\t[%GT]\n", f],
                           capture_output=True, text=True)
        for l in q.stdout.splitlines():
            pos, ref, alt, gt = l.split("\t")
            if "," in alt: continue
            a = gt.replace("|", "/")
            g = {"0/0": 0, "0/1": 1, "1/0": 1, "1/1": 2}.get(a)
            if g is not None: out[(c, int(pos))] = (alt, g)
    return out


def score(calls, T, min_dp):
    n = conc = ref_n = ref_fa = nr_n = nr_ok = absent = absent_alt = mism = 0
    for k, (alt, g, dp) in calls.items():
        if g < 0 or dp < min_dp: continue
        if k in T:
            ta, tg = T[k]
            if ta != alt and tg != 0: mism += 1; continue   # another ALT allele in the truth
            if ta != alt: tg = 0
        else:
            tg = 0; absent += 1; absent_alt += g > 0
        n += 1; conc += g == tg
        if tg == 0: ref_n += 1; ref_fa += g > 0
        else: nr_n += 1; nr_ok += g == tg
    f = lambda a, b: a / b if b else None
    return {"sites": n, "concordance": f(conc, n), "false_alt": f(ref_fa, ref_n), "truth_ref_sites": ref_n, "nonref_concordance": f(nr_ok, nr_n),
            "truth_nonref_sites": nr_n, "truth_nonref_correct": nr_ok, "absent_from_1000G": absent, "absent_nonref_calls": absent_alt,
            "other_alt_in_truth": mism}


def main(sample, tdir, sdir, cdir, out, bcftools="bcftools"):
    tmp = os.path.dirname(os.path.abspath(out))
    P = vcf_calls(os.path.join(sdir, "private/genotype/donors.vcf.gz"))
    A = vcf_calls(os.path.join(sdir, "private/genotype/donors_allsite.vcf.gz"))
    U = dict(A); U.update(P)
    C = cellsnp_calls(cdir) if cdir != "-" else {}
    T = truth(sample, tdir, set(U) | set(C), bcftools, tmp)
    res = {"sample": sample, "truth_positions_found": len(T)}
    for name, calls in [("singlet_panel", P), ("singlet_allsite", A), ("singlet_union", U), ("cellsnp_denovo", C)]:
        res[name] = {f"dp{d}": score(calls, T, d) for d in (10, 20)}
        res[name]["called_any_depth"] = sum(1 for v in calls.values() if v[1] >= 0)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
