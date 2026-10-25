"""Enrichment of a named panel of expected ambient genes, overall and in their tissue. usage: targeted.py REC_DIR SAMPLES_TSV OUT_CSV"""
import sys

import numpy as np
import pandas as pd

from aa_load import load_records

S, M, genes, sym, gcl = load_records(sys.argv[1], sys.argv[2])
ok = S.pool_ok.values
S = S[ok].reset_index(drop=True)
A, C = M["pool1"][ok].astype(float), M["called"][ok].astype(float)
Na, Nc = A.sum(1, keepdims=True), C.sum(1, keepdims=True)
PANEL = {"pancreas": ["PRSS1", "PRSS2", "CPA1", "CELA3A", "CTRB1", "PNLIP", "REG1A", "INS", "GCG"], "liver": ["ALB", "APOA1", "APOA2", "APOC3", "TTR", "SERPINA1", "FGB"],
         "lung": ["SFTPC", "SFTPB", "SFTPA1", "NAPSA", "SCGB1A1"], "blood": ["HBB", "HBA1", "HBA2", "PPBP", "S100A8", "S100A9", "IGKC", "JCHAIN"],
         "bone_marrow": ["HBB", "HBA1", "HBA2", "AHSP", "HBD", "MPO", "ELANE"], "all": ["MALAT1", "MT-CO1", "MT-ND1", "MT-ATP6", "HBB", "HBA1", "HBA2", "IGKC", "IGHG1", "JCHAIN", "ALB", "APOE", "SPARCL1", "FTL", "B2M", "ACTB"],
         "gut": ["MUC2", "TFF3", "FABP2", "REG4", "LYZ"], "heart": ["MYH7", "TNNT2", "NPPA", "MB"], "brain": ["PLP1", "MBP", "SNAP25", "GFAP"],
         "kidney": ["UMOD", "SLC34A1", "LRP2", "NPHS2"], "skin": ["KRT1", "KRT10", "DCD", "KRT5"]}
rows = []
for tissue, gl in PANEL.items():
    idx = np.arange(len(S)) if tissue == "all" else np.flatnonzero(S.tissue_coarse.values == tissue)
    for g in gl:
        j = np.flatnonzero(sym == g)
        if len(j) == 0 or len(idx) == 0:
            continue
        j = j[0]
        e = np.log2(((A[idx, j] + 0.5) / Na[idx, 0]) / ((C[idx, j] + 0.5) / Nc[idx, 0]))
        ev = np.maximum(A[idx, j], Na[idx, 0] * C[idx, j] / Nc[idx, 0]) >= 5
        rows.append(dict(panel=tissue, gene=g, n_samples=len(idx), n_eval=int(ev.sum()), median_log2enr=np.median(e[ev]) if ev.any() else np.nan,
                         frac_gt2x=float((e[ev] > 1).mean()) if ev.any() else np.nan,
                         pooled_log2enr=np.log2((A[idx, j] / Na[idx, 0]).mean() / (C[idx, j] / Nc[idx, 0]).mean()),
                         amb_cpm=(A[idx, j] / Na[idx, 0]).mean() * 1e6, cell_cpm=(C[idx, j] / Nc[idx, 0]).mean() * 1e6))
pd.DataFrame(rows).to_csv(sys.argv[3], index=False, float_format="%.4g")
print(pd.DataFrame(rows).round(2).to_string())
