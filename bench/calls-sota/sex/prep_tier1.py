#!/usr/bin/env python3
"""Tier 1 development libraries: evidence tables and cellXY input matrices.

lcl_pool: in-silico pool of HG00362 (female) and HG01277 (male); truth per cell = library of origin of the barcode.
osorio:   real 1:1 pool of GM12878 and GM18502, both female (1000 Genomes); truth = female for every cell.
Writes <out>/evidence/<name>.parquet and <out>/work/mtx/<name>/ (genes x cells) for cellxy.R.
"""
import glob, gzip, importlib.util, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sps

W = Path("/mnt/projects/debruinz_project/debruinz/singlet")
OUT = W / "bench/calls-sota/sex"
spec = importlib.util.spec_from_file_location("sexmod", OUT / "sexmod.py")
SX = importlib.util.module_from_spec(spec); sys.modules["sexmod"] = SX; spec.loader.exec_module(SX)

RUNS = {"HG01277": ("male", ["SRR24236331", "SRR24236332"]), "HG00362": ("female", ["SRR24236333", "SRR24236334"])}


def owners(data, max_reads=20_000_000, cb=16):
    own, dup = {}, set()
    for ind, (sex, runs) in RUNS.items():
        for r in runs:
            f = sorted(glob.glob(str(data / r / "*_R1_*.fastq.gz")))[0]
            with gzip.open(f, "rt") as fh:
                for i, line in enumerate(fh):
                    if i % 4 == 1:
                        bc = line[:cb]
                        if own.get(bc, ind) != ind:
                            dup.add(bc)
                        own.setdefault(bc, ind)
                    if i >= 4 * max_reads:
                        break
    for b in dup:
        own[b] = None
    return own


def read_run(run):
    """Gene-level counts of the called cells (filtered_feature_bc_matrix of the Singlet run): cells x genes."""
    f = run / "filtered_feature_bc_matrix"
    X = scipy.io.mmread(gzip.open(f / "matrix.mtx.gz")).tocsr()
    bc = [b.split("-")[0] for b in pd.read_csv(f / "barcodes.tsv.gz", header=None)[0].astype(str)]
    ft = pd.read_csv(f / "features.tsv.gz", header=None, sep="	")
    names = ft[1].astype(str).tolist()
    print("matrix", X.shape, "barcodes", len(bc), "features", len(names), flush=True)
    if X.shape[0] != len(names):
        X = X.T.tocsr()
    return X.T.tocsr(), bc, names


def main():
    (OUT / "evidence").mkdir(parents=True, exist_ok=True)
    for name, ds in (("lcl_pool", "lcl_gse230239"), ("osorio", "osorio2019_lcl_mix")):
        run = W / "bpub/runs" / ds / "singlet"
        X, bc, names = read_run(run)
        asg = pd.read_csv(run / "donor/donor_assignments.tsv", sep="\t") if (run / "donor/donor_assignments.tsv").exists() else pd.read_csv(run / "donor_assignments.tsv", sep="\t")
        asg = asg.set_index("cell")
        st = asg["cell_status"] if "cell_status" in asg else None
        keep = np.array([(b in asg.index) and (st is None or st.loc[b] == "cell") for b in bc])
        print(name, "barcodes", len(bc), "kept (called cells in donor table)", keep.sum(), flush=True)
        X = X[np.flatnonzero(keep)]; bc = [b for b, k in zip(bc, keep) if k]
        donor = np.array([str(asg.loc[b, "donor_id"]) for b in bc], dtype=object)
        donor[np.isin(donor, ["unassigned", "doublet", "nan"])] = ""
        x, y = SX.sex_evidence(X, names, "human")
        df = pd.DataFrame({"dataset": name, "tier": 1, "col": np.arange(len(bc)), "barcode": bc, "demux": donor,
                           "total_umi": np.asarray(X.sum(1)).ravel(), "x": x, "y": y})
        if name == "lcl_pool":
            own = owners(W / "bpub/data/lcl_gse230239")
            df["unit"] = [own.get(b[:16]) for b in bc]
            df["truth"] = df["unit"].map({k: v[0] for k, v in RUNS.items()})
            df = df[df["unit"].notna()].reset_index(drop=True)
            df["col"] = np.arange(len(df)); keepidx = np.flatnonzero(np.isin(np.array(bc), df["barcode"].values))
            X = X[keepidx]
        else:
            df["unit"] = "demux_" + df["demux"].replace("", "none"); df["truth"] = "female"
        print(name, df.groupby(["unit", "truth"]).size().to_dict(), flush=True)
        # demux vs truth purity (lcl_pool)
        if name == "lcl_pool":
            print(pd.crosstab(df["unit"], df["demux"]), flush=True)
        df.to_parquet(OUT / "evidence" / f"{name}.parquet", index=False)
        d = OUT / "work/mtx" / name; d.mkdir(parents=True, exist_ok=True)
        scipy.io.mmwrite(str(d / "matrix.mtx"), sps.csr_matrix(X.T.astype(np.int32)))
        import subprocess; subprocess.run(["gzip", "-f", str(d / "matrix.mtx")], check=True)
        (d / "genes.tsv").write_text("\n".join(names) + "\n")
        (d / "cells.tsv").write_text("\n".join(str(i) for i in range(len(df))) + "\n")
        (d / "species.txt").write_text("human\n")
    print("done")


main()
