# SPDX-License-Identifier: MIT
"""Synthetic cell-index partition for the format benchmark (docs/format-store-cell-index.md, "Format choice").

The pilot study has 41,614 called cells, too few to measure a shard format; this generator produces one realistic partition
(one organism x one coarse cell-type group) with the per-cell distributions of 10x data: log-normal depth, feature counts
that follow depth sub-linearly, a Zipf mix of fine cell types, per-sample constant tissue / sex / ancestry with a few
per-cell exceptions, called cells concentrated in the first matrix columns. It is a stand-in for measuring bytes and scan
speed only; nothing here is used by the builder.

    python bench/cidx/synth.py --cells 4000000 --out <dir>      # writes <dir>/part.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make(n_cells: int, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    # samples: cells per sample in this partition ~ log-normal around 1,500, ordinals grouped by study (10 samples/study)
    sizes = []
    while sum(sizes) < n_cells:
        sizes.append(int(np.clip(rng.lognormal(np.log(1500), 0.9), 20, 40000)))
    sizes[-1] -= sum(sizes) - n_cells
    sizes = [s for s in sizes if s > 0]
    n_samples = len(sizes)
    first_ord = 1000
    ordinals = first_ord + np.sort(rng.choice(70000, n_samples, replace=False)).astype(np.uint32)
    tissue_of = rng.zipf(1.6, n_samples).clip(1, 400).astype(np.uint16)  # UBERON dictionary codes
    sex_of = rng.choice([1, 2, 0], n_samples, p=[0.45, 0.45, 0.10]).astype(np.uint8)
    anc_of = rng.choice(
        np.arange(1, 9), n_samples, p=[0.5, 0.15, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02]
    ).astype(np.uint8)
    depth_of = rng.lognormal(np.log(3000), 0.6, n_samples)

    cols, samp, tis, sex, anc = [], [], [], [], []
    for k, n in enumerate(sizes):
        n_cols = int(n / rng.uniform(0.02, 0.4)) + n  # raw barcodes (called + uncalled)
        # called cells sit mostly in the first columns (the matrix is roughly UMI-ordered)
        c = np.unique((rng.beta(0.7, 6.0, int(n * 1.3)) * n_cols).astype(np.int64))[:n]
        while len(c) < n:
            c = np.unique(np.concatenate([c, rng.integers(0, n_cols, n)]))[:n]
        cols.append(np.sort(c).astype(np.uint32))
        samp.append(np.full(n, ordinals[k], np.uint32))
        tis.append(np.full(n, tissue_of[k], np.uint16))
        s = np.full(n, sex_of[k], np.uint8)
        s[rng.random(n) < 0.04] = 3  # ambiguous
        sex.append(s)
        anc.append(np.full(n, anc_of[k], np.uint8))
    col = np.concatenate(cols)
    sample = np.concatenate(samp)
    tissue = np.concatenate(tis)
    depth = np.repeat(depth_of, sizes)
    umi_exon = np.clip(rng.lognormal(np.log(depth), 0.8), 200, 4e5).astype(np.uint32)
    umi_intron = (umi_exon * rng.beta(2, 5, n_cells)).astype(np.uint32)
    feat_exon = np.clip(
        umi_exon**0.78 * rng.lognormal(np.log(1.3), 0.15, n_cells), 50, 30000
    ).astype(np.uint32)
    feat_intron = np.clip(umi_intron**0.8 * rng.lognormal(0.0, 0.2, n_cells), 0, 30000).astype(
        np.uint32
    )
    ct = (
        rng.zipf(1.5, n_cells).clip(1, 60).astype(np.uint16) + 100
    )  # fine cell-type dictionary codes
    ct_score = (np.clip(rng.beta(6, 2, n_cells), 0, 1) * 255).round().astype(np.uint8)
    anc_prob = (np.clip(rng.beta(20, 2, n_cells), 0, 1) * 255).round().astype(np.uint8)
    mt = (np.clip(rng.beta(1.2, 25, n_cells), 0, 1) * 255).round().astype(np.uint8)
    doublet = (np.clip(rng.beta(1, 12, n_cells), 0, 1) * 255).round().astype(np.uint8)
    out = dict(
        sample=sample,
        col=col,
        tissue=tissue,
        umi_exon=umi_exon,
        umi_intron=umi_intron,
        feat_exon=feat_exon,
        feat_intron=feat_intron,
        ct=ct,
        ct_score=ct_score,
        sex=np.concatenate(sex),
        anc=np.concatenate(anc),
        anc_prob=anc_prob,
        mt=mt,
        doublet=doublet,
    )
    # the builder's sort order inside a partition: tissue, sample, col
    order = np.lexsort((out["col"], out["sample"], out["tissue"]))
    return {k: v[order] for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=4_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = make(a.cells)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez(Path(a.out) / "part.npz", **d)
    print(
        f"{len(d['sample']):,} cells, {len(np.unique(d['sample'])):,} samples -> {a.out}/part.npz"
    )


if __name__ == "__main__":
    main()
