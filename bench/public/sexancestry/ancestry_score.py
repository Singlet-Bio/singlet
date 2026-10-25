#!/usr/bin/env python3
"""Genetic-ancestry (and bulk sex) benchmark, scoring step: every tool on the same MAGE alignments, at every depth.

    python ancestry_score.py --mage $BP/work/ancestry/mage --af $BP/work/ancestry/af_leaveout.parquet \
        --excluded $BP/work/ancestry/excluded.txt --somalier-dir $WORK/tools/bpub/somalier --gtf GENES.gtf \
        --artifacts $BP/artifacts/genetic-ancestry --bench-commit SHA

Tools:
- Singlet: singlet.celllabels.ancestry.infer_ancestry on the private depth tables, with leave-out frequencies (test
  individuals removed). The published label is the super-population with q >= 0.8, otherwise 'admixed'. `top` is the
  argmax of q, for a like-for-like comparison with tools that never abstain.
- somalier 0.3.6 `ancestry`: labelled background = 1000 Genomes somalier files minus the test individuals.
- peddy 0.4.8 on a multi-sample VCF (bcftools call at peddy's GRCh38 sites). Its built-in background includes the test
  individuals, which favours peddy.
Bulk sex (per individual, 8M read pairs): Singlet's sex model on the sample's XIST / chrY gene counts, peddy sex_check,
and the XIST/chrY rule.

Only labels, proportions and site counts are written (artifacts are public); depth tables never leave the cluster.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))
LEVELS = ["0.5", "2", "8"]


def singlet_calls(mage: Path, af_path: Path, subset: pd.DataFrame) -> tuple[list[dict], float]:
    from singlet.celllabels import ancestry as A

    af = A.AFIndex(pd.read_parquet(af_path))
    rows, t = [], 0.0
    for ind in subset["sample_id_1kg"]:
        for L in LEVELS:
            p = mage / "private" / ind / L / "sites.parquet"
            if not p.exists():
                continue
            sites = pd.read_parquet(p)
            t0 = time.perf_counter()
            ad, dp, F = af.align(sites)
            c = A.infer_ancestry(ad, dp, F)
            t += time.perf_counter() - t0
            top = max(c.q, key=c.q.get) if c.q else None
            rows.append(dict(individual=ind, level=L, tool="singlet", call=c.group, top=top,
                             q_max=c.prob, n_sites=c.n_sites, note=c.reason))
    return rows, t


def somalier_calls(mage: Path, sdir: Path, excluded: set, work: Path) -> tuple[list[dict], float]:
    lab = pd.read_csv(sdir / "ancestry-labels-1kg.tsv", sep="\t")
    lab = lab[~lab.iloc[:, 0].isin(excluded)]
    work.mkdir(parents=True, exist_ok=True)
    lab.to_csv(work / "labels.tsv", sep="\t", index=False)
    bg = [f for f in glob.glob(str(sdir / "1kg-somalier" / "*.somalier")) if Path(f).stem not in excluded]
    rows, t = [], 0.0
    for L in LEVELS:
        q = sorted(glob.glob(str(mage / "somalier" / L / "*.somalier")))
        if not q:
            continue
        out = work / f"L{L}"
        t0 = time.perf_counter()
        subprocess.run([str(sdir / "somalier"), "ancestry", "-o", str(out), "--labels", str(work / "labels.tsv"), *bg, "++", *q],
                       check=True, capture_output=True)
        t += time.perf_counter() - t0
        res = pd.read_csv(f"{out}.somalier-ancestry.tsv", sep="\t")
        res = res[res.iloc[:, 0].isin([Path(x).stem for x in q])]
        probc = [c for c in res.columns if c.endswith("_prob")]
        for _, r in res.iterrows():
            pr = {c[:-5]: r[c] for c in probc}
            rows.append(dict(individual=r.iloc[0], level=L, tool="somalier", call=r["predicted_ancestry"],
                             top=r["predicted_ancestry"], q_max=float(max(pr.values())) if pr else None, n_sites=None, note=None))
    return rows, t


def peddy_calls(mage: Path, subset: pd.DataFrame, work: Path, bcftools: str) -> tuple[list[dict], list[dict], float]:
    rows, sexrows, t = [], [], 0.0
    work.mkdir(parents=True, exist_ok=True)
    for L in LEVELS:
        vcfs = sorted(glob.glob(str(mage / "peddy" / L / "*.vcf.gz")))
        if len(vcfs) < 2:
            continue
        merged = work / f"L{L}.vcf.gz"
        subprocess.run([bcftools, "merge", "-m", "all", "-Oz", "-o", str(merged), *vcfs], check=True)
        subprocess.run([bcftools, "index", "-t", "-f", str(merged)], check=True)
        ped = work / f"L{L}.ped"
        sx = dict(zip(subset["sample_id_1kg"], subset["sex"]))
        with open(ped, "w") as fh:
            for v in vcfs:
                ind = Path(v).name.replace(".vcf.gz", "")
                fh.write(f"{ind}\t{ind}\t0\t0\t{1 if sx.get(ind) == 'male' else 2}\t-9\n")
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-m", "peddy", "-p", "8", "--sites", "hg38", "--prefix", str(work / f"L{L}"),
                        str(merged), str(ped)], check=True, capture_output=True)
        t += time.perf_counter() - t0
        het = pd.read_csv(work / f"L{L}.het_check.csv")
        for _, r in het.iterrows():
            rows.append(dict(individual=r["sample_id"], level=L, tool="peddy", call=r["ancestry-prediction"],
                             top=r["ancestry-prediction"], q_max=float(r["ancestry-prob"]), n_sites=None, note=None))
        if L == "8":
            sc = pd.read_csv(work / f"L{L}.sex_check.csv")
            for _, r in sc.iterrows():
                sexrows.append(dict(individual=r["sample_id"], tool="peddy", call=str(r["predicted_sex"]).lower()))
    return rows, sexrows, t


def gene_names(gtf: Path) -> dict:
    m = {}
    with open(gtf) as fh:
        for line in fh:
            if "\tgene\t" not in line:
                continue
            gid = re.search(r'gene_id "([^"]+)"', line).group(1)
            nm = re.search(r'gene_name "([^"]+)"', line)
            m[gid] = nm.group(1) if nm else gid
    return m


def bulk_sex(mage: Path, subset: pd.DataFrame, gtf: Path) -> tuple[list[dict], float]:
    import scipy.sparse as sps
    from singlet.celllabels import sex as SX

    names = gene_names(gtf)
    msy = ("AMELY DAZ1 PRKY RBMY1A1 RBMY1HP RPS4Y1 SRY TSPY1 UTY ZFY KDM5D USP9Y DDX3Y PRY XKRY BPY2 VCY CDY1 EIF1AY TMSB4Y "
           "CDY2A NLGN4Y PCDH11Y HSFY1 TGIF2LY TBL1Y RPS4Y2 HSFY2 CDY2B TXLNGY CDY1B DAZ3 DAZ2 DAZ4").split()
    rows, t = [], 0.0
    for ind in subset["sample_id_1kg"]:
        p = mage / "genecounts" / f"{ind}.tsv"
        if not p.exists():
            continue
        gc = pd.read_csv(p, sep="\t", header=None, skiprows=4, names=["gene", "unstranded", "s1", "s2"])
        gc["name"] = gc["gene"].map(names)
        cnt = dict(zip(gc["name"], gc["unstranded"]))
        genes = ["XIST", *SX.CHRY["human"]]
        X = sps.csr_matrix(np.array([[cnt.get(g, 0) for g in genes]], dtype=np.float64))
        t0 = time.perf_counter()
        call = SX.call_sex(X, genes, species="human")
        t += time.perf_counter() - t0
        rows.append(dict(individual=ind, tool="singlet", call=str(call.label[0])))
        x, y = cnt.get("XIST", 0), sum(cnt.get(g, 0) for g in msy)
        rows.append(dict(individual=ind, tool="xist-chrY-rule",
                         call="female" if x > 0 and y == 0 else ("male" if y > 0 and x == 0 else None)))
        rows.append(dict(individual=ind, tool="_counts", call=f"{x}:{y}"))
    return rows, t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mage", type=Path, required=True)
    ap.add_argument("--af", type=Path, required=True)
    ap.add_argument("--excluded", type=Path, required=True)
    ap.add_argument("--somalier-dir", type=Path, required=True)
    ap.add_argument("--gtf", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--bcftools", default="bcftools")
    ap.add_argument("--bench-commit", required=True)
    a = ap.parse_args(argv)
    a.artifacts.mkdir(parents=True, exist_ok=True)
    subset = pd.read_csv(REPO / "bench/public/sexancestry/mage_subset.tsv", sep="\t")
    excluded = set(a.excluded.read_text().split())
    work = a.mage.parent / "score"
    srows, ts = singlet_calls(a.mage, a.af, subset)
    mrows, tm = somalier_calls(a.mage, a.somalier_dir, excluded, work / "somalier")
    try:
        prows, psex, tp = peddy_calls(a.mage, subset, work / "peddy", a.bcftools)
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError) as e:
        print("peddy failed:", e)
        prows, psex, tp = [], [], None
    sex_rows, tsx = bulk_sex(a.mage, subset, a.gtf)
    truth = subset.set_index("sample_id_1kg")
    calls = pd.DataFrame(srows + mrows + prows)
    calls.insert(0, "dataset", "mage150")
    calls["population"] = calls["individual"].map(truth["population"])
    calls["super_population"] = calls["individual"].map(truth["super_population"])
    calls["sex"] = calls["individual"].map(truth["sex"])
    calls.to_csv(a.artifacts / "ancestry_calls.tsv", sep="\t", index=False)
    sx = pd.DataFrame(sex_rows + psex)
    sx.insert(0, "dataset", "mage150")
    sx["sex_truth"] = sx["individual"].map(truth["sex"])
    sx.to_csv(a.artifacts / "sex_bulk_calls.tsv", sep="\t", index=False)
    # timing of the shared inputs (download, STAR, per-depth pileup / extract / calling) from the mage jobs
    lv = pd.concat([pd.read_csv(f, sep="\t", header=None, names=["individual", "level", "pileup_s", "somalier_s", "peddy_s"])
                    for f in glob.glob(str(a.mage / "timing" / "*.levels.tsv"))]) if glob.glob(str(a.mage / "timing" / "*.levels.tsv")) else None
    n = len(set(calls["individual"]))
    res = [{"tool": "singlet", "dataset": "mage150", "wall_s": round(ts + (lv["pileup_s"].sum() if lv is not None else 0), 1),
            "max_rss_gb": 0.0, "threads": 1, "notes": f"panel pileup (bcftools) + infer_ancestry, {n} individuals x {len(LEVELS)} depths"},
           {"tool": "somalier", "dataset": "mage150", "wall_s": round(tm + (lv["somalier_s"].sum() if lv is not None else 0), 1),
            "max_rss_gb": 0.0, "threads": 1, "notes": "somalier extract + ancestry"}]
    if tp is not None:
        res.append({"tool": "peddy", "dataset": "mage150", "wall_s": round(tp + (lv["peddy_s"].sum() if lv is not None else 0), 1),
                    "max_rss_gb": 0.0, "threads": 8, "notes": "bcftools call at peddy sites + peddy"})
    afj = json.loads((a.af.parent / "af_leaveout.json").read_text()) if (a.af.parent / "af_leaveout.json").exists() else {}
    som_v = subprocess.run([str(a.somalier_dir / "somalier")], capture_output=True, text=True).stderr[:80]
    prov = {"bench_commit": a.bench_commit, "status": "draft", "resources": res,
            "af_leaveout": {k: v for k, v in afj.items() if k != "excluded"}, "n_excluded": len(excluded),
            "somalier_banner": som_v.strip()}
    (a.artifacts / "provenance_scoring.json").write_text(json.dumps(prov, indent=1))
    print(calls.groupby(["tool", "level"]).apply(lambda d: (d["top"] == d["super_population"]).mean()).round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
