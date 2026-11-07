"""Score HLA genotype calls against molecular typing at 2-field resolution (the metric of the T1K paper, Song et al. 2023).

Per sample and gene the two truth alleles are paired with the two called alleles in the order that matches more; a homozygous call
(T1K's second allele ".") counts twice, and a missing call scores 0 of 2. Accuracy = matched alleles / truth alleles.

  geuvadis : truth = Abi-Rached 2018 panel (truth.txt), calls = <res>/<sample>/{t1k,singlet}_genotype.tsv
  nih      : truth = Scisco typing in hla_benchmark's compiled_nih_3p_genotypes.csv ("Ground truth" rows, by donor letter);
             calls = Singlet's private/hla/hla_genotype.tsv per library, plus the published arcasHLA / OptiType / PHLAT / HLAminer rows
             for the same libraries from the same file
Prints one JSON object with per-gene and overall accuracy per method.
"""
import argparse
import csv
import glob
import json
import os
import re

GENES5 = ["A", "B", "C", "DQB1", "DRB1"]


def two_field(a):
    a = (a or "").strip()
    if not a or a in (".", "NA", "-", "None"):
        return None
    a = a.replace("HLA-", "")
    if "*" in a:
        a = a.split("*", 1)[1]
    m = re.match(r"(\d+):(\d+)", a)
    return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}" if m else None


def alts(a):
    """A truth cell may list ambiguous alleles separated by '/'."""
    return {x for x in (two_field(p) for p in (a or "").split("/")) if x}


def match(truth, call):
    """truth: [set, set]; call: [str|None, str|None] -> matched count (0..2)."""
    def m(t, c):
        return 1 if c is not None and c in t else 0
    return max(m(truth[0], call[0]) + m(truth[1], call[1]), m(truth[0], call[1]) + m(truth[1], call[0]))


def read_t1k(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            v = line.rstrip("\n").split("\t")
            if len(v) < 6:
                continue
            g = v[0].replace("HLA-", "")
            a1, a2 = two_field(v[2]), two_field(v[5])
            if a1 is None and a2 is None:
                continue
            out[g] = [a1, a2 if a2 is not None else a1]
    return out


def score(truth, calls, genes):
    per = {g: [0, 0] for g in genes}
    for s, tg in truth.items():
        if s not in calls:
            continue
        for g in genes:
            if g not in tg or not tg[g][0] or not tg[g][1]:
                continue
            per[g][0] += match(tg[g], calls[s].get(g, [None, None]))
            per[g][1] += 2
    tot = [sum(v[0] for v in per.values()), sum(v[1] for v in per.values())]
    return {"samples": sum(1 for s in truth if s in calls),
            "per_gene": {g: (round(v[0] / v[1], 4) if v[1] else None) for g, v in per.items()},
            "per_gene_counts": per, "overall": round(tot[0] / tot[1], 4) if tot[1] else None, "alleles": tot[1]}


def geuvadis(a):
    truth = {}
    with open(a.truth) as f:
        r = csv.reader(f, delimiter="\t")
        h = next(r)
        si = h.index("Sample ID")
        for v in r:
            tg = {}
            for g in GENES5:
                i1, i2 = h.index(f"HLA-{g} 1"), h.index(f"HLA-{g} 2")
                tg[g] = [alts(v[i1]), alts(v[i2])]
            truth[v[si]] = tg
    methods = {}
    for m in ("t1k", "singlet"):
        calls = {}
        for p in glob.glob(os.path.join(a.res, "*", f"{m}_genotype.tsv")):
            calls[os.path.basename(os.path.dirname(p))] = read_t1k(p)
        methods[m] = score(truth, calls, GENES5)
    same = 0
    n = 0
    for d in glob.glob(os.path.join(a.res, "*")):
        p1, p2 = os.path.join(d, "t1k_genotype.tsv"), os.path.join(d, "singlet_genotype.tsv")
        if os.path.exists(p1) and os.path.exists(p2):
            n += 1
            same += read_t1k(p1) == read_t1k(p2)
    return {"set": "Geuvadis RNA-seq (E-GEUV-1) vs Abi-Rached 2018", "methods": methods, "identical_2field_genotypes": [same, n]}


def nih(a):
    rows = list(csv.DictReader(open(a.compiled)))
    genes = sorted({k[:-2] for k in rows[0] if k.endswith("_1")})
    truth_by_donor = {}
    for r in rows:
        if r["genotyper"] == "Ground truth":
            truth_by_donor[r["sample"]] = {g: [alts(r[f"{g}_1"]), alts(r[f"{g}_2"])] for g in genes}
    libs = sorted({r["sample"] for r in rows if r["genotyper"] != "Ground truth"})
    donor_of = {s: s if s in truth_by_donor else s[0] for s in libs}
    truth = {s: truth_by_donor[donor_of[s]] for s in libs if donor_of[s] in truth_by_donor}
    methods = {}
    for gt in sorted({r["genotyper"] for r in rows} - {"Ground truth"}):
        calls = {r["sample"]: {g: [two_field(r[f"{g}_1"]), two_field(r[f"{g}_2"]) or two_field(r[f"{g}_1"])] for g in genes}
                 for r in rows if r["genotyper"] == gt}
        methods[gt] = calls
    sg = {}
    for lib, path in (x.split("=", 1) for x in a.singlet):
        sg[lib] = read_t1k(path)
    methods["Singlet (T1K)"] = sg
    out = {"set": "Oetjen 2018 bone marrow scRNA-seq (GSE120221) vs Scisco typing", "genes_scored": GENES5,
           "libraries_singlet": sorted(sg)}
    # Score every method on the libraries Singlet typed so the comparison is like for like
    sub = {s: t for s, t in truth.items() if s in sg}
    out["methods"] = {m: score(sub, c, GENES5) for m, c in methods.items()}
    out["methods_all_loci"] = {m: score(sub, c, [g for g in genes if g in ("A", "B", "C", "DPA1", "DPB1", "DQA1", "DQB1", "DRB1")])
                               for m, c in methods.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="mode", required=True)
    g = sp.add_parser("geuvadis")
    g.add_argument("--truth", required=True)
    g.add_argument("--res", required=True)
    n = sp.add_parser("nih")
    n.add_argument("--compiled", required=True)
    n.add_argument("--singlet", nargs="+", required=True, help="LIB=path/to/hla_genotype.tsv (LIB as in the compiled file)")
    a = ap.parse_args()
    print(json.dumps(geuvadis(a) if a.mode == "geuvadis" else nih(a), indent=1))


if __name__ == "__main__":
    main()
