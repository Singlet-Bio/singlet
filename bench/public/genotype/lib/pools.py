"""Pool definitions and truth tables for the donor-demux / donor-snps / mtdna benchmarks.

A *pool* is one real scRNA-seq library with known donors or cell lines. No simulated or in-silico pools are used:
published numbers come from real libraries with real truth only.

  Kang 2018 (GSE96583) lanes, K = 8 (batch 2 ctrl [development lane], batch 2 stim, batch 1 C) and K = 4 (batch 1 A, B).
      Truth is the authors' demuxlet call (genotype-based): `ind` = donor, `multiplets` = singlet / doublet / ambs.
      No donor genotypes are public for this study.
  MAESTER (Miller 2022, GSE182685) 10x 3' libraries:
      maester_k562_bt142_3p  K = 2 real cell-line mixture (K562 + BT142); truth = the authors' cell labels (expression and
                             mt-variant labels must agree).
      maester_bpdcn712_3p    K = 1 (one patient's bone marrow); every called cell is one donor.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEV = "kang2018_ctrl"            # batch 2 control lane 2.1: development data (PR #36 was tuned here)
KANG = {
    # id: (lane label in the truth table, truth file, column holding the lane)
    "kang2018_ctrl": ("ctrl", "batch2", "stim"),
    "kang2018_stim": ("stim", "batch2", "stim"),
    "kang2018_b1_a": ("A", "batch1", "batch"),
    "kang2018_b1_b": ("B", "batch1", "batch"),
    "kang2018_b1_c": ("C", "batch1", "batch"),
}
MAESTER = ["maester_k562_bt142_3p", "maester_bpdcn712_3p"]
SINGLE_DONOR = {"maester_bpdcn712_3p"}
# replicate lane holding the same donors, used as independent genotype / haplotype truth (donor-snps, mtdna)
REPLICATES = {
    "kang2018_ctrl": ["kang2018_stim"],
    "kang2018_stim": ["kang2018_ctrl"],
    "kang2018_b1_a": ["kang2018_b1_c"],
    "kang2018_b1_b": ["kang2018_b1_c"],
    "kang2018_b1_c": ["kang2018_b1_a", "kang2018_b1_b"],
}
K_TRUE = {"kang2018_ctrl": 8, "kang2018_stim": 8, "kang2018_b1_a": 4, "kang2018_b1_b": 4, "kang2018_b1_c": 8,
          "maester_k562_bt142_3p": 2, "maester_bpdcn712_3p": 1}


# run directories keep the names they were created with
RUNDIR = {"kang2018_ctrl": "kang2018_b2_ctrl", "kang2018_stim": "kang2018_b2_stim"}


def rundir(runs, pool):
    return Path(runs) / RUNDIR.get(pool, pool)


def all_pools() -> list[str]:
    return list(KANG) + MAESTER


def load_truth(truth_dir: str | Path, pool: str) -> pd.DataFrame | None:
    """Per-barcode truth: index = barcode without '-1'; columns ind (str), multiplets. None for a single-donor pool
    (every called cell is the donor). `truth_dir` is the Kang truth directory; MAESTER truth sits next to it."""
    if pool in KANG:
        lane, batch, col = KANG[pool]
        t = pd.read_csv(Path(truth_dir) / f"GSE96583_{batch}.total.tsne.df.tsv.gz", sep="\t", index_col=0)
        t = t[t[col] == lane].copy()
        t.index = [i.split("-")[0] for i in t.index]
        t["ind"] = t["ind"].map(lambda v: "<NA>" if pd.isna(v) else str(int(v)))
        return t[["ind", "multiplets"]]
    if pool == "maester_k562_bt142_3p":
        t = pd.read_csv(Path(truth_dir).parent / "maester_truth" / "celllinemix_cells.tsv", sep="\t").set_index("barcode")
        t = t[(t["line_rna"] == t["line_mt"]) & t["line_rna"].isin(["K562", "BT142"])]
        return pd.DataFrame({"ind": t["line_rna"], "multiplets": "singlet"}, index=t.index)
    if pool in SINGLE_DONOR:
        return None
    raise KeyError(pool)


def pool_k(truth_dir: str | Path, pool: str) -> int:
    return K_TRUE[pool]


if __name__ == "__main__":  # python pools.py TRUTH_DIR : print the pool table
    import sys
    rows = []
    for p in all_pools():
        t = load_truth(sys.argv[1], p)
        rows.append(dict(pool=p, k=K_TRUE[p], truth_cells=None if t is None else len(t),
                         truth_singlets=None if t is None else int((t["multiplets"] == "singlet").sum())))
    print(pd.DataFrame(rows).to_string())
