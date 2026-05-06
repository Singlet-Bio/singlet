"""
singlet — Python client for the Singlet single-cell atlas.

Browse catalog (works offline):
    singlet.catalog()                      Browse all datasets
    singlet.catalog("lung")                Search by keyword
    singlet.info("GSE264667")              Dataset metadata
    singlet.sample_index()                 Full sample index DataFrame
    singlet.species()                      Species breakdown
    singlet.tissues()                      Tissue/source breakdown
    singlet.protocols()                    Protocol breakdown
    singlet.datasets(organism="Homo sapiens", min_cells=100000)
    singlet.samples(status="SUCCESS")      Filter samples
    singlet.samples(tissue="brain")        Filter by tissue/source
    singlet.samples(protocol="dropseq")    Filter by protocol
    singlet.top_series(n=10)               Largest series
    singlet.quality_tiers()                Quality tier breakdown (gold/silver/bronze)
    singlet.failure_categories()           Pipeline failure analysis
    singlet.cell_types()                   Cell type annotations (50% coverage, 42 categories)
    singlet.summary()                      Atlas overview

Load data:
    singlet.load("GSE264667")              Load from local catalog or Zenodo → AnnData
    singlet.load("path/to/counts.1pz")     Load local .1pz file
    singlet.load_dir("/path/to/quant/GSM") Load singlify output directory → AnnData
    singlet.load_sample("GSM3308814")      Load single sample (column-range read)

Format I/O:
    singlet.read_1pz("file.1pz")           Read .1pz → AnnData (preferred)
    singlet.write_1pz(adata, "out.1pz")    Write AnnData → .1pz (preferred)
    singlet.read_kraken2("gse_dir/")       Read kraken2 microbiome matrix
    singlet.read_spz("file.spz")           Read .spz → AnnData (legacy)
    singlet.write_spz(adata, "out.spz")    Write AnnData → .spz (legacy)
    singlet.read_matrix("file")            Auto-detect .spz or .1pz

Configuration:
    singlet.set_catalog_dir("/path/to/catalog")  Set local catalog path

Cell type annotation (free, local):
    singlet.gene_programs("Homo sapiens")  Download NMF gene programs (W matrix)
    singlet.project(adata)                 Project cells → gene program space (H matrix)
    singlet.annotate(adata)                Annotate cells with types (NMF-based)

Exploration:
    singlet.describe(adata)                Quick summary stats (sparsity, counts, genes)

QC filtering:
    singlet.filter_cells(adata, min_genes=200)  Filter low-quality cells
    singlet.filter_genes(adata, min_cells=3)    Filter rarely-detected genes

Preprocessing:
    singlet.normalize(adata)                    Library-size normalize + log1p
    singlet.highly_variable_genes(adata)        Select top variable genes
    singlet.pca(adata)                          PCA dimensionality reduction
    singlet.harmony(adata, "batch")             Batch correction (Harmony)

Token-priced (requires API key):
    singlet.login(key)                     Authenticate
    singlet.query(...)                     Cross-atlas query → AnnData
    singlet.search(text)                   Natural-language search → AnnData
"""

__version__ = "2.0.0"

from singlet._annotate import annotate, gene_programs, project
from singlet._auth import login
from singlet._catalog import (
    catalog,
    cell_types,
    datasets,
    failure_categories,
    info,
    protocols,
    quality_tiers,
    refresh,
    sample_index,
    samples,
    set_catalog_dir,
    species,
    summary,
    tissues,
    top_series,
)
from singlet._combat import combat
from singlet._concat import concatenate
from singlet._de import rank_genes_groups, rank_genes_groups_df
from singlet._dendrogram import dendrogram
from singlet._diffmap import diffmap
from singlet._dpt import dpt
from singlet._describe import describe
from singlet._embedding_density import embedding_density
from singlet._enrichr import enrichr, enrichr_from_de
from singlet._filter import filter_cells, filter_genes
from singlet._harmony import harmony
from singlet._ingest import ingest
from singlet._hvg import highly_variable_genes
from singlet._hvg_seurat import highly_variable_genes_seurat_v3
from singlet._io import (
    info_1pz,
    read_1pz,
    read_kraken2,
    read_matrix,
    read_spz,
    spz_info,
    write_1pz,
    write_spz,
)
from singlet._leiden import leiden
from singlet._louvain import louvain
from singlet._loader import download, load, load_dir, load_sample
from singlet._marker_overlap import marker_gene_overlap
from singlet._neighbors import neighbors
from singlet._paga import paga
from singlet._normalize import normalize
from singlet._pca import pca
from singlet._plot import plot_dotplot, plot_umap, plot_violin
from singlet._plot_paga import plot_paga
from singlet._plot_heatmap import plot_heatmap
from singlet._plot_markers import rank_genes_groups_dotplot
from singlet._plot_scatter import plot_scatter
from singlet._plot_stacked_violin import plot_stacked_violin
from singlet._qc import calculate_qc_metrics
from singlet._query import query, search
from singlet._regress import regress_out
from singlet._scale import scale
from singlet._score import score_genes
from singlet._subsample import subsample
from singlet._tsne import tsne
from singlet._umap import umap
from singlet._versions import show_versions
from singlet.convert import (
    from_h5ad,
    from_mtx,
    from_tiledb,
    from_zarr,
    to_csc,
    to_h5ad,
    to_mtx,
    to_tiledb,
    to_zarr,
)

__all__ = [
    # Browse
    "catalog",
    "info",
    "species",
    "tissues",
    "protocols",
    "datasets",
    "sample_index",
    "set_catalog_dir",
    "summary",
    "samples",
    "top_series",
    "refresh",
    "quality_tiers",
    "failure_categories",
    "cell_types",
    # Load
    "load",
    "load_sample",
    "load_dir",
    "download",
    # Token-priced
    "login",
    "query",
    "search",
    # Annotation (free, local)
    "gene_programs",
    "project",
    "annotate",
    # I/O
    "read_1pz",
    "write_1pz",
    "info_1pz",
    "read_kraken2",
    "read_matrix",
    "read_spz",
    "write_spz",
    "spz_info",
    # Conversions
    "to_h5ad",
    "to_zarr",
    "to_mtx",
    "to_csc",
    "to_tiledb",
    "from_h5ad",
    "from_zarr",
    "from_mtx",
    "from_tiledb",
    # Exploration
    "describe",
    # QC filtering
    "calculate_qc_metrics",
    "filter_cells",
    "filter_genes",
    "subsample",
    "concatenate",
    # Preprocessing
    "normalize",
    "highly_variable_genes",
    "highly_variable_genes_seurat_v3",
    "scale",
    "regress_out",
    "pca",
    "neighbors",
    "leiden",
    "louvain",
    "tsne",
    "diffmap",
    "dpt",
    "umap",
    "paga",
    # Differential expression
    "rank_genes_groups",
    "rank_genes_groups_df",
    "rank_genes_groups_dotplot",
    "score_genes",
    "dendrogram",
    "enrichr",
    "enrichr_from_de",
    # Marker overlap
    "marker_gene_overlap",
    # Embedding density
    "embedding_density",
    # Batch correction
    "harmony",
    "combat",
    # Plotting
    "plot_umap",
    "plot_violin",
    "plot_dotplot",
    "plot_heatmap",
    "plot_scatter",
    "plot_stacked_violin",
    "plot_paga",
    # Integration
    "ingest",
    # Diagnostics
    "show_versions",
]
