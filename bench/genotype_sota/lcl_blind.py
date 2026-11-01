"""Blind check of the ancestry label rule: the four scRNA-seq LCL donors (HG00362 FIN, HG01277 CLM; GM12878 CEU, GM18502 YRI) are in the
benchmark's excluded list, so neither the leave-out frequencies nor the label rule's reference individuals include them.
Applies the old rule and the shipped rule to each donor's coverage table. Prints q, calls and the label probability (counts and
proportions only). Usage: lcl_blind.py ANCESTRY_PY AF_PARQUET BP_RUNS"""
import glob, importlib.util, sys, os
from pathlib import Path
import numpy as np, pandas as pd
sp = importlib.util.spec_from_file_location("ancestry", sys.argv[1]); A = importlib.util.module_from_spec(sp); sys.modules["ancestry"] = A; sp.loader.exec_module(A)
af = A.AFIndex(pd.read_parquet(sys.argv[2]))
expect = {"lcl_gse230239": {"EUR", "AMR"}, "osorio2019_lcl_mix": {"EUR", "AFR"}}
for ds in expect:
    run = Path(sys.argv[3]) / ds / "singlet"
    got = []
    for cov in sorted(glob.glob(str(run / "private" / "donor" / "donor*_coverage.tsv"))):
        t = pd.read_csv(cov, usecols=["chrom", "pos", "ref", "alt", "ad", "dp"], sep="\t")
        ad, dp, F = af.align(t)
        old = A.infer_ancestry(ad, dp, F, n_boot=0, use_label_model=False); new = A.infer_ancestry(ad, dp, F, n_boot=0)
        p = A.label_probabilities(np.array([new.q[k] for k in A.POPS])) if new.q else None
        got.append(new.group)
        print(ds, Path(cov).name.split("_")[0], "sites", new.n_sites, "q", {k: round(v, 3) for k, v in new.q.items()}, "old", old.group,
              "new", new.group, "p_label", None if p is None else round(float(p.max()), 3), "nn_l1",
              None if not new.q else round(A._nn_l1(np.array([new.q[k] for k in A.POPS]), A.label_model()), 3))
    print(ds, "expected set", sorted(expect[ds]), "called set", sorted(g for g in got if g), "MATCH" if set(got) == expect[ds] else "MISMATCH")
