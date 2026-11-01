# SPDX-License-Identifier: MIT
"""One V(D)J library of the atlas -> contigs, per-cell labels, QC verdict and the s1 members of its GEX sample.

    python -m singlet.vdj.library --r1 R1.fastq.gz [...] --r2 R2.fastq.gz [...] --gex-columns cols.txt
        --gex-called called.txt --species human --whitelist 737K-august-2016.txt --gsm GSM... --out DIR

``--gex-columns``: the GEX sample's ``cells`` table barcodes in ``col`` order; ``--gex-called``: its called cells. Writes
``contigs.csv``, ``cells.csv``, ``qc.json`` and, when the library passes QC, ``<GSM>.vdj.s2`` (a sidecar object holding
``vdj_contigs.parquet`` and ``vdj_summary.json``) and ``<GSM>.features.vdj.parquet``. Nothing is uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, Optional, Sequence

import numpy as np

from . import assemble as A
from . import clonotype as CL
from . import qc as Q
from . import members as ST


def process_library(
    r1: Sequence[str],
    r2: Sequence[str],
    *,
    gex_columns: Sequence[str],
    gex_called: Sequence[str],
    whitelist_codes: np.ndarray,
    species: str = "human",
    ref: Optional[A.Reference] = None,
    umi_len: int = 10,
    params: Optional[A.Params] = None,
    libraries: Sequence[str] = (),
    log=print,
) -> Dict:
    ref = ref or A.Reference.load(species)
    res = A.run_library(
        r1,
        r2,
        ref=ref,
        cb_len=16,
        umi_len=umi_len,
        whitelist_codes=whitelist_codes,
        params=params or A.Params(),
        log=log,
    )
    called = set(gex_called)
    contigs = res.contigs
    # V(D)J cells are called GEX cells; contigs elsewhere stay in the member (col -1 or an uncalled column)
    cells = CL.assign_clonotypes(CL.cell_table(contigs[contigs["barcode"].isin(called)]))
    qc = Q.library_qc(
        contigs, cells, res.stats, gex_called=list(called), gex_columns=list(gex_columns)
    )
    ref_path = os.path.join(
        os.path.dirname(os.path.abspath(A.R.__file__)), "data", f"vdj_{species}.tsv.gz"
    )
    ref_info = {"name": f"vdj_{species}", "segments": len(ref.segs)}
    if os.path.exists(ref_path):
        ref_info["sha256"] = hashlib.sha256(open(ref_path, "rb").read()).hexdigest()
    counting = {"libraries": list(libraries), "stats": res.stats, "umi_len": umi_len}
    members = (
        ST.build_members(contigs, gex_columns, qc, counting, ref_info)
        if qc["verdict"] != "fail"
        else None
    )
    labels = ST.cell_labels(cells, gex_columns) if len(cells) else None
    return {
        "contigs": contigs,
        "cells": cells,
        "qc": qc,
        "members": members,
        "labels": labels,
        "stats": res.stats,
        "timings": res.timings,
    }


def _lines(path):
    with open(path) as f:
        return [x.strip().split(",")[0] for x in f if x.strip()]


def main(argv=None):
    from singlet.adt import count as AC
    from singlet.adt import seq as S

    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", nargs="+", required=True)
    ap.add_argument("--r2", nargs="+", required=True)
    ap.add_argument("--gex-columns", required=True)
    ap.add_argument("--gex-called", required=True)
    ap.add_argument("--whitelist", required=True)
    ap.add_argument("--species", default="human")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--umi-len", type=int, default=10)
    ap.add_argument("--gsm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    wl = AC.load_barcode_list(a.whitelist)
    codes, bad = S.encode_strings(wl, 16)
    ref = A.Reference.load(a.ref or a.species)
    out = process_library(
        a.r1,
        a.r2,
        gex_columns=_lines(a.gex_columns),
        gex_called=_lines(a.gex_called),
        whitelist_codes=np.sort(codes[bad == 0]),
        species=a.species,
        ref=ref,
        umi_len=a.umi_len,
        params=A.Params(threads=a.threads),
        libraries=[os.path.basename(p) for p in a.r1],
    )
    out["contigs"].to_csv(os.path.join(a.out, "contigs.csv"), index=False)
    out["cells"].to_csv(os.path.join(a.out, "cells.csv"), index=False)
    with open(os.path.join(a.out, "qc.json"), "w") as f:
        json.dump(out["qc"], f, indent=1, default=float)
    if out["members"] is not None:
        with open(os.path.join(a.out, f"{a.gsm}.vdj.s2"), "wb") as f:
            f.write(ST.build_sidecar(a.gsm, out["members"]))
    if out["labels"] is not None:
        out["labels"].to_parquet(os.path.join(a.out, f"{a.gsm}.features.vdj.parquet"), index=False)
    print(
        json.dumps(
            {
                k: out["qc"][k]
                for k in ("verdict", "failed", "warned", "vdj_cells", "pairing_rate")
                if k in out["qc"]
            },
            default=float,
        )
    )


if __name__ == "__main__":
    main()
