#!/usr/bin/env python3
"""Chromosomal-sex benchmark, step 1: per-cell evidence tables from Singlet's counts (one Clipper task per study).

For every sample of a study that has sex in its GEO metadata (``--sex-labels``, {GSM: female|male}), it writes:

- ``<out>/cells/<GSM>.parquet``: one row per called cell. Columns: gsm, gse, species, col, donor, total_umi, n_genes, xist,
  y_singlet (the non-PAR chrY genes Singlet uses), super_x and super_y (cellXY's escape-gene and MSY sets), and Singlet's call
  (singlet_label, singlet_p_male), made by ``singlet.celllabels.sex.call_sex`` exactly as the atlas labels are. Only these
  aggregates of a few public genes are kept. They are the artifact the notebook recomputes every metric from.
- ``<out>/mtx/<GSM>/``: the full gene x cell count matrix (matrix.mtx.gz, genes.tsv, cells.tsv). cellXY (step 2,
  ``cellxy.R``) runs on it. It stays on Clipper.

Counts are exon + intron at gene level, as served. Legacy bundles are read with ``singlet.celllabels.io.read_legacy``,
store studies with ``read_s1``: the same readers the atlas annotation used.

    python bench/public/sexancestry/sex_extract.py --gse GSE145370 --sex-labels sex_labels.json --out $BP/work/sex --scratch $TMP
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

# cellXY's gene sets (phipsonlab/cellXY R/preprocess.R; matched on upper-cased symbols, for mouse too)
CELLXY_X = ("ARHGAP4 STS ARSD ARSL AVPR2 BRS3 S100G CHM CLCN4 DDX3X EIF1AX EIF2S3 GPM6B GRPR HCFC1 L1CAM MAOA MYCLP1 NAP1L3 "
            "GPR143 CDK16 PLXNB3 PRKX RBBP7 RENBP RPS4X TRAPPC2 SH3BGRL TBL1X UBA1 KDM6A XG XIST ZFX PUDP PNPLA4 USP9X KDM5C "
            "SMC1A NAA10 OFD1 IKBKG PIR INE2 INE1 AP1S2 GYG2 MED14 RAB9A ITM2A MORF4L2 CA5B SRPX2 GEMIN8 CTPS2 CLTRN NLGN4X "
            "DUSP21 ALG13 SYAP1 SYTL4 FUNDC1 GAB3 RIBC1 FAM9C CA5BP1").split()
CELLXY_Y = ("AMELY DAZ1 PRKY RBMY1A1 RBMY1HP RPS4Y1 SRY TSPY1 UTY ZFY KDM5D USP9Y DDX3Y PRY XKRY BPY2 VCY CDY1 EIF1AY TMSB4Y "
            "CDY2A NLGN4Y PCDH11Y HSFY1 TGIF2LY TBL1Y RPS4Y2 HSFY2 CDY2B TXLNGY CDY1B DAZ3 DAZ2 DAZ4").split()


def _rows(genes_upper: np.ndarray, wanted) -> np.ndarray:
    w = set(wanted)
    return np.flatnonzero([g in w for g in genes_upper])


def evidence_table(s, sex_call) -> "object":
    import pandas as pd
    import scipy.sparse as sps
    from singlet.celllabels import sex as SX

    X = sps.csr_matrix(s.X)  # cells x genes
    genes = np.asarray(s.gene_names, dtype=object)
    up = np.array([str(g).upper() for g in genes], dtype=object)
    colsum = lambda idx: np.asarray(X[:, idx].sum(1)).ravel() if len(idx) else np.zeros(X.shape[0])
    x_s, y_s = SX.sex_evidence(X, s.gene_names, s.species)
    return pd.DataFrame({
        "gsm": s.gsm, "gse": s.gse, "species": s.species,
        "col": np.asarray(s.col, np.int64),
        "donor": pd.array([d or None for d in s.donor] if s.donor is not None else [None] * X.shape[0], dtype="string"),
        "total_umi": np.asarray(X.sum(1)).ravel().astype(np.int64),
        "n_genes": np.diff(X.indptr).astype(np.int32),
        "xist": np.asarray(x_s).astype(np.int64),
        "y_singlet": np.asarray(y_s).astype(np.int64),
        "super_x": colsum(_rows(up, CELLXY_X)).astype(np.int64),
        "super_y": colsum(_rows(up, CELLXY_Y)).astype(np.int64),
        "singlet_label": pd.array(sex_call.label, dtype="string"),
        "singlet_p_male": sex_call.p_male.astype(np.float32),
    })


def write_mtx(s, d: Path) -> None:
    """Gene x cell matrix (matrix.mtx.gz, genes.tsv, cells.tsv = col, species.txt) for cellxy.R."""
    import scipy.io
    import scipy.sparse as sps

    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "matrix.mtx.gz", "wb") as fh:
        scipy.io.mmwrite(fh, sps.csr_matrix(s.X).T.tocoo().astype(np.int32))  # genes x cells, as cellXY expects
    (d / "genes.tsv").write_text("\n".join(map(str, s.gene_names)) + "\n")
    (d / "cells.tsv").write_text("\n".join(str(int(c)) for c in s.col) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gse", required=True)
    ap.add_argument("--sex-labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--gsm", nargs="*", help="restrict to these samples (default: every labelled sample of the study)")
    a = ap.parse_args(argv)

    from annotate_campaign import DATA_BASE, _download, _has_store
    from singlet.celllabels import io as IO
    from singlet.celllabels import sex as SX

    truth = json.loads(a.sex_labels.read_text())
    (a.out / "cells").mkdir(parents=True, exist_ok=True)
    work = a.scratch / a.gse
    work.mkdir(parents=True, exist_ok=True)
    v2 = _has_store(a.gse)
    bundle = None
    try:
        if v2:
            from singlet.s1 import client as C
            gsms = [g for g in C.open_study(a.gse).samples if g in truth]
        else:
            bundle = work / f"{a.gse}.singlet"
            _download(f"{DATA_BASE}/data/{a.gse}/{a.gse}.singlet", bundle)
            from singlet.bundle import SingletBundle
            b = SingletBundle(bundle)
            gsms = [g for g in b.gsm_ids if g in truth]
            b.close()
        if a.gsm:
            gsms = [g for g in gsms if g in set(a.gsm)]
        for gsm in gsms:
            t0 = time.time()
            try:
                s = IO.read_s1(a.gse, gsm) if v2 else IO.read_legacy(bundle, gsm)
            except Exception as e:  # noqa: BLE001  (one bad sample must not stop the study)
                (a.out / "cells" / f"{gsm}.error.txt").write_text(f"{type(e).__name__}: {e}\n")
                print(f"{gsm}: read failed: {e}", flush=True)
                continue
            if s.species not in ("human", "mouse") or len(s.col) == 0:
                print(f"{gsm}: skipped (species {s.species}, {len(s.col)} cells)", flush=True)
                continue
            t1 = time.time()
            call = SX.call_sex(s.X, s.gene_names, species=s.species, groups=s.donor)
            t_sex = time.time() - t1
            df = evidence_table(s, call)
            df.to_parquet(a.out / "cells" / f"{gsm}.parquet", index=False)
            write_mtx(s, a.out / "mtx" / gsm)
            rec = dict(gse=a.gse, gsm=gsm, truth=truth[gsm], species=s.species, n_cells=int(len(df)),
                       source="v2" if v2 else "legacy", singlet_sex_s=round(t_sex, 4), read_s=round(t1 - t0, 2),
                       n_donors=int(len({d for d in (s.donor if s.donor is not None else []) if d})))
            with open(a.out / "samples.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
