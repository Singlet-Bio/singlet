#!/usr/bin/env python3
"""End-to-end single-cell test on 1000 Genomes cell lines: Singlet 1.0.0 output -> donors -> ancestry and sex.

    python lcl_score.py --bp $BP --af af_leaveout.parquet --somalier-dir DIR --excluded excluded.txt --bench-commit SHA

Datasets (harness runs in $BP/runs/<dataset>/singlet/, made with --snps and --n-donors 2):
- lcl_gse230239: reads of two libraries (HG00362, FIN, female; HG01277, CLM, male) run as ONE library, an in-silico pool.
  Each cell's true individual is the library its barcode's reads came from (barcodes in both libraries are dropped).
- osorio2019_lcl_mix: a real 1:1 pool of GM12878 (CEU, female) and GM18502 (YRI, female). No per-cell truth, so the donor
  test is that the two donors get the two ancestries.

Writes, to $BP/artifacts/genetic-ancestry/ and $BP/artifacts/chromosomal-sex/:
- lcl_ancestry_calls.tsv: per Singlet donor: dataset, donor, individual (lcl_gse230239), truth super-population, tool,
  call, top, q_max, n_sites. Competitor: somalier on the donor's reads (split from Singlet's tagged BAM).
- lcl_sex_cells.parquet: per cell of lcl_gse230239, in the sex artifact schema, with truth_donor_sex. Tools: Singlet's
  sex model with no donor grouping (the pool is one group, the hard case), cellXY, the XIST/chrY rule.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
IND = {"lcl_gse230239": {"GSM7208744": ("HG00362", "EUR", "female"), "GSM7208745": ("HG01277", "AMR", "male")},
       "osorio2019_lcl_mix": {}}
OSORIO_TRUTH = {"EUR", "AFR"}


def barcodes_by_run(data: Path, runs: dict, cb_len: int = 16, max_reads: int = 20_000_000) -> dict:
    """Barcode -> GSM, from the barcode reads of each run (first max_reads per run)."""
    owner, seen = {}, Counter()
    for gsm, run_list in runs.items():
        for r in run_list:
            f = sorted(glob.glob(str(data / r / "*_R1_*.fastq.gz")))
            if not f:
                continue
            with gzip.open(f[0], "rt") as fh:
                for i, line in enumerate(fh):
                    if i % 4 == 1:
                        bc = line[:cb_len]
                        if owner.get(bc, gsm) != gsm:
                            seen[bc] += 1
                        owner.setdefault(bc, gsm)
                    if i >= 4 * max_reads:
                        break
    for bc in seen:  # barcodes present in both libraries: no single truth
        owner[bc] = None
    return owner


def read_counts(run: Path):
    import scipy.io
    import scipy.sparse as sps

    def one(stem):
        X = scipy.io.mmread(gzip.open(run / f"{stem}.mtx.gz")).tocsr()
        bc = pd.read_csv(run / f"{stem}_barcodes.tsv.gz", header=None)[0].astype(str).tolist()
        ft = pd.read_csv(run / f"{stem}_features.tsv.gz", header=None, sep="\t")
        return X, bc, ft

    Xe, bc, ft = one("exon_counts")
    Xi, bci, _ = one("intron_counts")
    if Xe.shape[0] != len(bc):  # features x cells on disk
        Xe, Xi = Xe.T.tocsr(), Xi.T.tocsr()
    assert bc == bci
    names = ft[1].astype(str).tolist() if ft.shape[1] > 1 else ft[0].astype(str).tolist()
    return (Xe + Xi).tocsr(), bc, names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bp", type=Path, required=True)
    ap.add_argument("--af", type=Path, required=True)
    ap.add_argument("--somalier-dir", type=Path, required=True)
    ap.add_argument("--excluded", type=Path, required=True)
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--samtools", default="samtools")
    ap.add_argument("--bench-commit", required=True)
    a = ap.parse_args(argv)
    from singlet.celllabels import ancestry as A
    from singlet.celllabels import sex as SX
    from gsm_runs_lookup import runs_of

    af = A.AFIndex(pd.read_parquet(a.af))
    excluded = set(a.excluded.read_text().split())
    rows, sex_parts = [], []
    for ds in ("lcl_gse230239", "osorio2019_lcl_mix"):
        run = a.bp / "runs" / ds / "singlet"
        if not (run / ".done").exists():
            print(f"{ds}: no finished Singlet run")
            continue
        asg = pd.read_csv(run / "donor" / "donor_assignments.tsv", sep="\t") if (run / "donor" / "donor_assignments.tsv").exists() \
            else pd.read_csv(run / "donor_assignments.tsv", sep="\t")
        cell_ind = None
        if IND[ds]:
            owner = barcodes_by_run(a.bp / "data" / ds, {g: runs_of(g) for g in IND[ds]})
            asg["gsm"] = asg["cell"].str[:16].map(owner)
            cell_ind = asg.set_index("cell")["gsm"]
        # ---- ancestry per Singlet donor
        som_q = []
        for cov in sorted(glob.glob(str(run / "donor" / "donor*_coverage.tsv"))):
            donor = Path(cov).name.split("_")[0]
            t = pd.read_csv(cov, sep="\t", usecols=["chrom", "pos", "ref", "alt", "ad", "dp"])
            ad, dp, F = af.align(t)
            c = A.infer_ancestry(ad, dp, F)
            ind, truth = None, None
            if IND[ds]:
                g = asg.loc[asg["donor_id"] == donor, "gsm"].dropna()
                if len(g):
                    top_gsm, n = Counter(g).most_common(1)[0]
                    if n / len(g) >= 0.9:
                        ind, truth, _ = IND[ds][top_gsm]
            rows.append(dict(dataset=ds, donor=donor, individual=ind, super_population=truth, tool="singlet", call=c.group,
                             top=max(c.q, key=c.q.get) if c.q else None, q_max=c.prob, n_sites=c.n_sites,
                             n_cells=int((asg["donor_id"] == donor).sum())))
            # somalier on the same donor's reads
            cells = asg.loc[asg["donor_id"] == donor, "cell"]
            lst = run / f"{donor}.cells.txt"
            lst.write_text("\n".join(cells) + "\n")
            dbam = run / f"{donor}.bam"
            subprocess.run([a.samtools, "view", "-b", "-@", "8", "-D", f"CB:{lst}", "-o", str(dbam), str(run / "possorted.bam")], check=True)
            subprocess.run([a.samtools, "addreplacerg", "-r", f"ID:{ds}_{donor}\tSM:{ds}_{donor}", "-o", str(dbam) + ".rg.bam", str(dbam)], check=True)
            subprocess.run([a.samtools, "index", str(dbam) + ".rg.bam"], check=True)
            od = a.bp / "work" / "ancestry" / "lcl_somalier"
            od.mkdir(parents=True, exist_ok=True)
            subprocess.run([str(a.somalier_dir / "somalier"), "extract", "-d", str(od), "-s", str(a.somalier_dir / "sites.hg38.vcf.gz"),
                            "-f", str(a.fasta), str(dbam) + ".rg.bam"], check=True, capture_output=True)
            som_q.append((f"{ds}_{donor}", donor, ind, truth))
            for p in (dbam, Path(str(dbam) + ".rg.bam"), Path(str(dbam) + ".rg.bam.bai"), lst):
                p.unlink(missing_ok=True)
        if som_q:
            od = a.bp / "work" / "ancestry" / "lcl_somalier"
            lab = pd.read_csv(a.somalier_dir / "ancestry-labels-1kg.tsv", sep="\t")
            lab[~lab.iloc[:, 0].isin(excluded)].to_csv(od / "labels.tsv", sep="\t", index=False)
            bg = [f for f in glob.glob(str(a.somalier_dir / "1kg-somalier" / "*.somalier")) if Path(f).stem not in excluded]
            q = [str(od / f"{s}.somalier") for s, *_ in som_q]
            subprocess.run([str(a.somalier_dir / "somalier"), "ancestry", "-o", str(od / ds), "--labels", str(od / "labels.tsv"),
                            *bg, "++", *q], check=True, capture_output=True)
            res = pd.read_csv(od / f"{ds}.somalier-ancestry.tsv", sep="\t").set_index("#sample_id" if "#sample_id" in
                              pd.read_csv(od / f"{ds}.somalier-ancestry.tsv", sep="\t", nrows=0).columns else "sample_id")
            for s, donor, ind, truth in som_q:
                if s in res.index:
                    r = res.loc[s]
                    probs = [c for c in res.columns if c.endswith("_prob")]
                    rows.append(dict(dataset=ds, donor=donor, individual=ind, super_population=truth, tool="somalier",
                                     call=r["predicted_ancestry"], top=r["predicted_ancestry"],
                                     q_max=float(max(r[c] for c in probs)) if probs else None, n_sites=None,
                                     n_cells=int((asg["donor_id"] == donor).sum())))
        # ---- per-cell sex on the in-silico pool
        if cell_ind is not None:
            X, bc, genes = read_counts(run)
            from sex_extract import CELLXY_X, CELLXY_Y, _rows
            up = np.array([g.upper() for g in genes], dtype=object)
            keep = np.array([cell_ind.get(b) is not None and not pd.isna(cell_ind.get(b)) for b in bc])
            X = X[keep]
            bck = [b for b, k in zip(bc, keep) if k]
            call = SX.call_sex(X, genes, species="human")  # the pool as one group: no donor information
            x, y = SX.sex_evidence(X, genes, "human")
            colsum = lambda idx: np.asarray(X[:, idx].sum(1)).ravel()
            df = pd.DataFrame({"dataset": ds, "gse": "GSE230239", "gsm": "pool", "species": "human",
                               "col": np.arange(len(bck)), "donor": None,
                               "total_umi": np.asarray(X.sum(1)).ravel().astype(np.int64), "n_genes": np.diff(X.indptr),
                               "xist": np.asarray(x, np.int64), "y_singlet": np.asarray(y, np.int64),
                               "super_x": colsum(_rows(up, CELLXY_X)).astype(np.int64),
                               "super_y": colsum(_rows(up, CELLXY_Y)).astype(np.int64),
                               "singlet_label": call.label.astype(str), "singlet_p_male": call.p_male,
                               "truth_donor_sex": [IND[ds][cell_ind[b]][2] for b in bck], "barcode": bck})
            sex_parts.append(df)
            # matrix for cellXY (sex.sbatch cellxy picks up every mtx/<name>/)
            import scipy.io
            d = a.bp / "work" / "sex" / "mtx" / f"{ds}_pool"
            d.mkdir(parents=True, exist_ok=True)
            with gzip.open(d / "matrix.mtx.gz", "wb") as fh:
                scipy.io.mmwrite(fh, X.T.tocoo().astype(np.int32))
            (d / "genes.tsv").write_text("\n".join(genes) + "\n")
            (d / "cells.tsv").write_text("\n".join(map(str, range(len(bck)))) + "\n")
            (d / "species.txt").write_text("human\n")
    out_a = a.bp / "artifacts" / "genetic-ancestry"
    out_a.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_a / "lcl_ancestry_calls.tsv", sep="\t", index=False)
    if sex_parts:
        out_s = a.bp / "work" / "sex"
        pd.concat(sex_parts).to_parquet(out_s / "lcl_sex_cells.parquet", index=False)
    print(pd.DataFrame(rows).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
