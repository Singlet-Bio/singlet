#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the 1000 Genomes super-population allele-frequency table that singlet.annotate.ancestry uses.

Source: the 1000 Genomes phase 3 call set on GRCh38 (``20190312_biallelic_SNV_and_INDEL``), whose sites VCF carries AFR_AF,
AMR_AF, EAS_AF, EUR_AF and SAS_AF in INFO. These are population summaries (no individual genotypes). Kept: biallelic SNVs on
chr1-22 whose allele frequency is between ``--min-af`` and ``1 - --min-af`` in at least one super-population; with ``--panel``
(the pipeline's SNP panel VCF, e.g. ``genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz``) only the panel's sites.

  python scripts/annotate_build_af.py --vcf ALL.wgs.shapeit2_integrated_snvindels_v2a.GRCh38.27022019.sites.vcf.gz \
      --out 1kgp_superpop_af.parquet [--panel genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz]

Output columns: chrom (no 'chr'), pos (1-based), ref, alt, AFR, AMR, EAS, EUR, SAS (float32).
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

import numpy as np

POPS = ("AFR", "AMR", "EAS", "EUR", "SAS")
_AF = {p: re.compile(rf"(?:^|;){p}_AF=([0-9.eE+-]+)") for p in POPS}
_AUTOSOMES = {str(i) for i in range(1, 23)}


def _open(p: Path):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p)


def panel_keys(p: Path) -> set:
    keys = set()
    with _open(p) as f:
        for line in f:
            if line.startswith("#"):
                continue
            c, pos, _, ref, alt = line.split("\t", 5)[:5]
            keys.add((c.removeprefix("chr"), int(pos), ref, alt))
    return keys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vcf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panel", type=Path)
    ap.add_argument("--min-af", type=float, default=0.01)
    a = ap.parse_args(argv)

    import pandas as pd

    keep = panel_keys(a.panel) if a.panel else None
    if keep is not None:
        print(f"panel: {len(keep):,} sites", flush=True)
    cols = {k: [] for k in ("chrom", "pos", "ref", "alt", *POPS)}
    n = 0
    with _open(a.vcf) as f:
        for line in f:
            if line.startswith("#"):
                continue
            n += 1
            c, pos, _, ref, alt, _, _, info = line.split("\t", 8)[:8]
            c = c.removeprefix("chr")
            if (
                c not in _AUTOSOMES
                or len(ref) != 1
                or len(alt) != 1
                or ref not in "ACGT"
                or alt not in "ACGT"
            ):
                continue
            p = int(pos)
            if keep is not None and (c, p, ref, alt) not in keep and (c, p, alt, ref) not in keep:
                continue
            afs = []
            for k in POPS:
                m = _AF[k].search(info)
                afs.append(float(m.group(1)) if m else np.nan)
            if not np.isfinite(afs).all() or not any(a.min_af <= x <= 1 - a.min_af for x in afs):
                continue
            cols["chrom"].append(c)
            cols["pos"].append(p)
            cols["ref"].append(ref)
            cols["alt"].append(alt)
            for k, x in zip(POPS, afs):
                cols[k].append(x)
            if n % 5_000_000 == 0:
                print(f"{n:,} records read, {len(cols['pos']):,} kept", flush=True)
    df = pd.DataFrame(cols)
    for k in POPS:
        df[k] = df[k].astype(np.float32)
    df["pos"] = df["pos"].astype(np.int64)
    df.to_parquet(a.out, index=False, compression="zstd")
    print(f"wrote {a.out}: {len(df):,} sites from {n:,} records", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
