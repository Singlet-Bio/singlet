Interop with AnnData and Scanpy
===============================

The :mod:`singlify.interop` subpackage adapts singlify pipeline outputs
into bioinformatics-ecosystem-native objects. Everything delegates to
:mod:`singlify.io`; the adapters just reshape and attach metadata.

AnnData
-------

:func:`singlify.interop.anndata.read_anndata` is the single entry point.
It takes a pipeline output directory and returns a fully-populated
:class:`anndata.AnnData`:

.. code-block:: python

    from singlify.interop.anndata import read_anndata

    adata = read_anndata("quant/scrna/GSE174/GSE174399/GSM5293863")
    print(adata)
    # AnnData object with n_obs × n_vars = 16079 × 38606
    #     obs: cell_qc_metrics__total_umis, ..., doublet_scores__doublet_score, ...
    #     var: feature_id
    #     uns: 'singlify', 'singlify_source_path', 'singlify_primary_assay'
    #     obsm: 'exon_counts', 'intron_counts', 'sj_counts', 'splice_psi', ...
    #     layers: 'unspliced', 'ambiguous', 'gene_counts_em'

Layout:

* ``adata.X`` — the primary per-gene count matrix (``spliced`` by default,
  falling back to ``gene_counts`` then ``gene_counts_em``). Oriented
  ``cells × genes`` per AnnData convention. Narrow dtype matching
  the ``.1pz`` file.
* ``adata.layers["unspliced"]`` — unspliced counts (velocity input)
* ``adata.layers["ambiguous"]`` — ambiguous reads (zero-nnz on 3'/5' 10x
  and drop-seq, but the layer exists for API consistency)
* ``adata.layers["gene_counts_em"]`` — EM-rescued multi-mapper counts
* ``adata.obsm["exon_counts"]`` — per-exon matrix (310797 features; a
  different feature axis than ``.X``, so it lives in ``obsm``)
* ``adata.obsm["intron_counts"]`` — per-intron matrix (272191 features)
* ``adata.obsm["sj_counts"]`` — per-splice-junction matrix
* ``adata.obsm["splice_psi"]`` — per-event PSI matrix
* ``adata.obsm["mt_heteroplasmy"]`` — per-mitochondrial-site matrix
* ``adata.obsm["vdj_gene_usage"]`` — per-VDJ-gene matrix
* ``adata.obs`` — per-cell sidecars loaded from
  ``cell_qc_metrics.tsv``, ``cell_cycle_scores.tsv``,
  ``doublet_scores.tsv``, ``read_stats.tsv``,
  ``ambient_contamination.tsv``. Columns are prefixed by the source
  filename so different sidecars never collide.
* ``adata.var["feature_id"]`` — Ensembl gene IDs (for the 2024-A GRCh38
  reference the pipeline uses).
* ``adata.uns["singlify"]`` — the full embedded GEO context:
  ``gsm_id``, ``gse_id``, ``srr_ids``, ``organism``, ``protocol``,
  ``singlify_version``, ``pipeline_date``, ``read_count``.

Choose a different primary assay with the ``primary_assay`` argument:

.. code-block:: python

    adata = read_anndata(path, primary_assay="gene_counts_em")

Scanpy
------

:mod:`singlify.interop.scanpy` provides small shortcuts that operate on
the AnnData produced above:

.. code-block:: python

    import singlify.interop.scanpy as slsc
    import scanpy as sc

    # One-step read that guarantees adata.layers["spliced"] / ["unspliced"]
    # exist — scvelo's minimum contract.
    adata = slsc.read("quant/scrna/GSE174/GSE174399/GSM5293863")

    # QC using the pipeline's pre-computed mt_pct / total_umis — no need
    # to recompute with calculate_qc_metrics.
    slsc.quick_qc(adata, min_genes=200, max_mt_pct=20, min_umis=500)

    # Standard normalize_total + log1p in one call
    slsc.normalize_log(adata, target_sum=1e4)

    # From here it's pure scanpy:
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)

scVelo
------

Because :func:`read_anndata` already populates ``layers["unspliced"]``
and ``layers["spliced"]``, scVelo workflows run without any renaming:

.. code-block:: python

    import singlify.interop.scanpy as slsc
    import scvelo as scv

    adata = slsc.read("quant/scrna/GSE174/GSE174399/GSM5293863")
    scv.pp.filter_and_normalize(adata)
    scv.pp.moments(adata)
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)

Cross-sample / cohort analysis
------------------------------

Each GSM carries its own GEO context in ``adata.uns["singlify"]``, so a
cohort merge is straightforward:

.. code-block:: python

    import anndata as ad
    from pathlib import Path
    from singlify.interop.anndata import read_anndata

    gse_dir = Path("quant/scrna/GSE174/GSE174399")
    samples = [read_anndata(sd) for sd in gse_dir.iterdir() if sd.is_dir()]
    for a in samples:
        a.obs["gsm_id"] = a.uns["singlify"]["gsm_id"]

    merged = ad.concat(samples, join="outer", label="gsm_id")
    print(merged)

Seurat (via R)
--------------

For R users, the sister package ``singlify`` in this repo has a native
``as_seurat()`` adapter that produces a Seurat object with
scvelo-compatible assays. Use :func:`rpy2 <rpy2.robjects>` or
``anndata2ri`` if you need a cross-language bridge from the Python
side — but the preferred workflow for Seurat users is to use the R
package directly.

See the R package README at ``singlify/r/README.md`` in the repository.
