#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/marker_score_decoupler_reference.py
#
# Reference driver: run DecoupleR marker scoring on a dense float32 matrix
# (genes × cells) with a gene-set DB (TSV), then write activity scores to .npz.
#
# Environment: pip install decoupler-py anndata scanpy numpy scipy
#
# Usage:
#   python marker_score_decoupler_reference.py \
#       --matrix   <matrix.npy>      # float32 (genes × cells), shape (G, C)
#       --geneset  <geneset.tsv>     # TSV: set_name, gene_idx, weight
#       --output   <scores.npz>      # scores + metadata
#       --method   mlm|ulm|wsum      # default: mlm
#
# Output .npz keys:
#   scores       : float32 (n_sets × n_cells) — activity score matrix
#   set_names    : str array (n_sets,)         — gene set names (sorted)
#   n_sets       : int scalar
#   n_cells      : int scalar
#
# The gene-set TSV has columns: set_name (str), gene_idx (int), weight (float).
# The matrix npy is row=genes, col=cells (G × C).
#
# DecoupleR expects AnnData with cells as rows (C × G). We transpose internally.
# Gene names are stringified integer indices ("0", "1", …).

import argparse
import sys
import numpy as np
import pandas as pd


def load_geneset_tsv(path: str):
    """Return (net DataFrame with source/target/weight, sorted set_names list)."""
    df = pd.read_csv(path, sep="\t", header=None, names=["set_name", "gene_idx", "weight"],
                     dtype={"set_name": str, "gene_idx": int, "weight": float})
    # DecoupleR 'net' format: source, target, weight
    net = pd.DataFrame({
        "source": df["set_name"].astype(str),
        "target": df["gene_idx"].astype(str),   # must match adata.var_names
        "weight": df["weight"].astype(np.float32),
    })
    set_names = sorted(df["set_name"].unique().tolist())
    return net, set_names


def run_reference(matrix_path: str,
                  geneset_path: str,
                  output_path: str,
                  method: str) -> None:
    import anndata
    import decoupler as dc

    mat = np.load(matrix_path)             # (G, C) float32
    if mat.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got shape {mat.shape}")
    n_genes, n_cells = mat.shape
    print(f"[ref] matrix: {n_genes} genes × {n_cells} cells", file=sys.stderr)

    net, set_names = load_geneset_tsv(geneset_path)
    print(f"[ref] gene sets: {len(set_names)}, method={method}", file=sys.stderr)

    # Build AnnData: cells × genes (transpose).
    gene_names = [str(i) for i in range(n_genes)]
    cell_names = [str(j) for j in range(n_cells)]
    adata = anndata.AnnData(
        X=mat.T.astype(np.float32),        # (C, G)
        obs={"cell": cell_names},
        var={"gene": gene_names},
    )
    adata.obs_names = cell_names
    adata.var_names = gene_names

    # Run the chosen method.
    if method == "mlm":
        acts, _ = dc.run_mlm(mat=adata, net=net, source="source",
                              target="target", weight="weight",
                              verbose=False, use_raw=False)
    elif method == "ulm":
        acts, _ = dc.run_ulm(mat=adata, net=net, source="source",
                              target="target", weight="weight",
                              verbose=False, use_raw=False)
    elif method == "wsum":
        acts, _, _ = dc.run_wsum(mat=adata, net=net, source="source",
                                 target="target", weight="weight",
                                 verbose=False, use_raw=False)
    else:
        raise ValueError(f"Unknown method: {method}")

    # acts is an AnnData: obs=cells, var=gene_sets.
    # Re-order columns to sorted set_names for deterministic comparison.
    available = [s for s in set_names if s in acts.var_names]
    acts_ordered = acts[:, available].X  # (C, n_sets) dense
    if hasattr(acts_ordered, "toarray"):
        acts_ordered = acts_ordered.toarray()
    scores = acts_ordered.T.astype(np.float32)   # (n_sets, n_cells)

    print(f"[ref] scores shape: {scores.shape}", file=sys.stderr)
    np.savez(output_path,
             scores=scores,
             set_names=np.array(available, dtype=object),
             n_sets=np.int32(len(available)),
             n_cells=np.int32(n_cells))
    print(f"[ref] Saved to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="DecoupleR marker score reference for singlet-gpu anno tests")
    parser.add_argument("--matrix",  required=True,
                        help="float32 .npy gene×cell matrix, shape (G, C)")
    parser.add_argument("--geneset", required=True,
                        help="TSV: set_name, gene_idx, weight")
    parser.add_argument("--output",  required=True,
                        help="Output .npz path")
    parser.add_argument("--method",  default="mlm",
                        choices=["mlm", "ulm", "wsum"],
                        help="DecoupleR method (default: mlm)")
    args = parser.parse_args()

    run_reference(args.matrix, args.geneset, args.output, args.method)


if __name__ == "__main__":
    main()
