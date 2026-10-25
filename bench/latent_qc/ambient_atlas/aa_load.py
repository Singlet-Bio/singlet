"""Load the per-sample records written by extract.py into aligned arrays (shared by the analysis scripts)."""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

HB = {"HBB", "HBA1", "HBA2", "HBD", "HBG1", "HBG2", "HBM", "HBQ1", "HBZ", "HBE1", "AHSP"}
HB3 = {"HBB", "HBA1", "HBA2"}


def gene_class(sym, chrom):
    s = sym.upper()
    if chrom in ("chrM", "MT", "M") or s.startswith("MT-"):
        return "MT"
    if s in HB:
        return "HB"
    if s == "MALAT1":
        return "MALAT1"
    if re.match(r"^RP[LS]\d+[AXY]?$|^RPLP\d$|^RPSA$|^MRP[LS]", s):
        return "RP"
    if re.match(r"^IG[HKL][VDJC]|^JCHAIN$|^IGLL|^IGHV|^IGKV", s):
        return "IG"
    return ""


def load_records(rec_dir, meta_tsv, org="human", min_called=50, min_pool_umi=2000):
    """Return (S dataframe, mats dict of (nsamp x ncore) arrays, genes, symbols, gclass). Core genes = present in >=95% of samples."""
    meta = pd.read_csv(meta_tsv, sep="\t").drop_duplicates("gsm").set_index("gsm")
    recs = []
    for f in sorted(glob.glob(f"{rec_dir}/GSM*.npz")):
        gsm = os.path.basename(f)[:-4]
        if gsm not in meta.index or meta.loc[gsm, "org"] != org:
            continue
        z = np.load(f, allow_pickle=False)
        st = json.loads(str(z["stat"]))
        if st["n_called"] < min_called:
            continue
        recs.append((gsm, st, z))
    S = pd.DataFrame([r[1] for r in recs])
    S = S.join(meta[["tissue_coarse", "tissue_label", "disease_label", "sample_kind"]], on="gsm")
    # core genes
    cnt = {}
    sym, chrom = {}, {}
    for gsm, st, z in recs:
        for g, s, c in zip(z["genes"], z["symbols"], z["chroms"]):
            cnt[g] = cnt.get(g, 0) + 1
            sym.setdefault(g, str(s)); chrom.setdefault(g, str(c))
    n = len(recs)
    core = sorted(g for g, c in cnt.items() if c >= 0.95 * n)
    gi = {g: i for i, g in enumerate(core)}
    names = ["called", "pool1", "pool2", "pool3", "poolA", "poolB", "called_i", "pool1_i"]
    mats = {k: np.zeros((n, len(core)), np.float32) for k in names}
    for r, (gsm, st, z) in enumerate(recs):
        idx = np.array([gi.get(g, -1) for g in z["genes"]])
        ok = idx >= 0
        for k in names:
            mats[k][r, idx[ok]] = z[k][ok]
    symbols = np.array([sym[g] for g in core])
    gcl = np.array([gene_class(sym[g], chrom[g]) for g in core])
    S["n_core"] = len(core)
    S["core_frac_pool"] = mats["pool1"].sum(1) / np.maximum(S.pool1_umi.values, 1)
    S["core_frac_called"] = mats["called"].sum(1) / np.maximum(S.called_umi.values, 1)
    S["p_side"] = np.where(S.protocol.fillna("").str.contains("5"), "5p", "3p")
    S["chem"] = S.protocol.fillna("NA")
    S["pool_ok"] = S.pool1_umi >= min_pool_umi
    return S, mats, np.array(core), symbols, gcl
