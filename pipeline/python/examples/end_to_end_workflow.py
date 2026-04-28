"""
End-to-end singlify pipeline-output workflow example.

Demonstrates the typical user journey from a singlify pipeline output
directory to a scanpy-ready AnnData with QC, normalization, and
preliminary clustering. This script is meant to be readable as
documentation as much as it is to be executed.

Usage::

    python end_to_end_workflow.py <pipeline_output_dir>

If no path is given, the script picks one of the canonical small
fixtures on the Clipper NFS tree (GSM7103327, ~410 cells).

Requirements: numpy, scipy, anndata, scanpy, singlify (with the
_pz_io extension built — see the package README for install).
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pipeline_dir",
        nargs="?",
        default=None,
        help="Singlify pipeline output directory (one sample). "
             "Default: a small fixture on Clipper.",
    )
    args = parser.parse_args()

    if args.pipeline_dir is None:
        # Default fixture — a 410-cell GSM that's small enough to run
        # this whole workflow in under 5 seconds.
        args.pipeline_dir = (
            "/mnt/projects/debruinz_project/singlify_pipeline/"
            "quant/scrna/GSE227/GSE227136/GSM7103327"
        )

    sample_dir = pathlib.Path(args.pipeline_dir)
    if not sample_dir.is_dir():
        print(f"error: {sample_dir} is not a directory", file=sys.stderr)
        return 1

    print(f"=== singlify end-to-end workflow ===")
    print(f"sample: {sample_dir}\n")

    # ------------------------------------------------------------------
    # Step 1: Inspect the pipeline directory (optional, fast)
    # ------------------------------------------------------------------
    print("Step 1: inspecting pipeline directory")
    from singlify.io import open_pz, read_metadata

    gene_counts_path = sample_dir / "gene_counts.1pz"
    if not gene_counts_path.is_file():
        # Tier 1 drop — gene_counts is verified redundant in newer pipelines.
        # Fall back to spliced.1pz, which is what the AnnData adapter does.
        gene_counts_path = sample_dir / "spliced.1pz"

    with open_pz(gene_counts_path) as pz:
        print(f"  shape:       {pz.header.m} × {pz.header.n}")
        print(f"  vt_code:     {pz.header.vt_code.name}")
        print(f"  nnz:         {pz.header.nnz:,}")
        print(f"  num_chunks:  {pz.header.num_chunks}")
        meta = read_metadata(pz)
        print(f"  gsm_id:      {meta.user_kv.get('gsm_id')}")
        print(f"  gse_id:      {meta.user_kv.get('gse_id')}")
        print(f"  organism:    {meta.user_kv.get('organism')}")
        print(f"  protocol:    {meta.user_kv.get('protocol')}")
        print(f"  pipeline:    {meta.user_kv.get('singlify_version')} "
              f"({meta.user_kv.get('pipeline_date')})")
    print()

    # ------------------------------------------------------------------
    # Step 2: Read into a scvelo-ready AnnData
    # ------------------------------------------------------------------
    print("Step 2: reading into AnnData (scvelo-ready)")
    import singlify.interop.scanpy as slsc

    t0 = time.perf_counter()
    adata = slsc.read(sample_dir)
    elapsed = time.perf_counter() - t0
    print(f"  loaded {adata.n_obs} cells × {adata.n_vars} genes in {elapsed*1000:.0f}ms")
    print(f"  layers:  {sorted(adata.layers.keys())}")
    print(f"  obsm:    {sorted(adata.obsm.keys())}")
    print(f"  obs columns: {len(adata.obs.columns)} (from auto-loaded sidecars)")
    print(f"  uns keys: {sorted(adata.uns.keys())}")
    derived = adata.uns.get("singlify_derived_layers", [])
    if derived:
        print(f"  *** derived layers from per-feature matrices: {derived}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Quick QC + normalize + log
    # ------------------------------------------------------------------
    print("Step 3: quick QC + normalize_total + log1p")
    n_before = adata.n_obs
    slsc.quick_qc(adata, min_genes=50, min_cells=2, max_mt_pct=30, min_umis=50)
    print(f"  cells: {n_before} -> {adata.n_obs}")
    print(f"  genes: -> {adata.n_vars}")

    if adata.n_obs < 5 or adata.n_vars < 100:
        print("  (sample too small for downstream analysis — stopping at QC)")
        print()
        print("=== done ===")
        return 0

    slsc.normalize_log(adata)
    print(f"  X.max after log1p: {adata.X.max():.2f}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Standard scanpy embedding + cluster
    # ------------------------------------------------------------------
    print("Step 4: scanpy HVG + PCA + UMAP + Leiden")
    try:
        import scanpy as sc
    except ImportError:
        print("  scanpy not installed — install with `pip install scanpy`")
        return 0

    sc.pp.highly_variable_genes(adata, n_top_genes=min(500, adata.n_vars - 1))
    n_hvg = int(adata.var["highly_variable"].sum())
    print(f"  HVG: {n_hvg}")

    sc.tl.pca(adata, n_comps=min(20, adata.n_obs - 1, n_hvg))
    print(f"  PCA: {adata.obsm['X_pca'].shape}")

    sc.pp.neighbors(adata, n_neighbors=min(15, adata.n_obs - 1))
    sc.tl.umap(adata)
    print(f"  UMAP: {adata.obsm['X_umap'].shape}")

    try:
        # Newer scanpy needs igraph for the default flavor; fall back to
        # leidenalg if available, otherwise skip clustering gracefully.
        sc.tl.leiden(adata, resolution=0.5, flavor="leidenalg", directed=False)
        n_clusters = adata.obs["leiden"].nunique()
        print(f"  Leiden: {n_clusters} clusters")
    except (ImportError, ValueError) as e:
        print(f"  Leiden skipped: {type(e).__name__}: {e}")
    print()

    # ------------------------------------------------------------------
    # Step 5: Show the GEO context survives end-to-end
    # ------------------------------------------------------------------
    print("Step 5: GEO context preserved")
    print(f"  adata.uns['singlify']['gsm_id']:  {adata.uns['singlify']['gsm_id']}")
    print(f"  adata.uns['singlify']['protocol']: {adata.uns['singlify']['protocol']}")
    print(f"  adata.uns['singlify']['pipeline_date']: {adata.uns['singlify']['pipeline_date']}")
    print()

    # ------------------------------------------------------------------
    # Step 6: Show the cohort-level read API
    # ------------------------------------------------------------------
    # The parent directory of `sample_dir` is the GSE root in the
    # canonical NFS layout (.../GSEnnnnnn/GSMnnnnn). Walk it to find
    # all sibling samples, then bulk-load their spliced matrices.
    print("Step 6: cohort-level read")
    from singlify.io import read_cohort

    gse_root = sample_dir.parent
    sibling_dirs = sorted(d for d in gse_root.iterdir() if d.is_dir())
    n_siblings = len(sibling_dirs)
    print(f"  GSE root: {gse_root}")
    print(f"  found {n_siblings} sibling samples")

    if n_siblings >= 2:
        # Cap at 5 to keep the demo fast
        cohort = read_cohort(sibling_dirs[: min(5, n_siblings)],
                             matrix_name="spliced")
        print(f"  loaded {len(cohort)} samples from the cohort:")
        for mat, meta in cohort:
            gsm = meta.user_kv.get("gsm_id", "?")
            print(f"    {gsm}: {mat.shape[0]:>6}x{mat.shape[1]:>6}  nnz={mat.nnz:,}")

        if len(cohort) >= 2:
            import scipy.sparse as sp
            shapes = set(m.shape[0] for m, _ in cohort)
            if len(shapes) == 1:
                merged = sp.hstack([m for m, _ in cohort])
                print(f"  hstack merged: {merged.shape[0]} genes × "
                      f"{merged.shape[1]} cells, nnz={merged.nnz:,}")
    else:
        print("  (no sibling samples to demonstrate cohort merge)")
    print()

    print("=== done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
