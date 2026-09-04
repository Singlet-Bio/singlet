# SPDX-License-Identifier: MIT
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

Find data (natural language):
    singlet.find("human lung fibroblasts")     Search → list of accessions
    singlet.find("melanoma T cells", level="gse")
    singlet.find_load("human pancreas islets") Search + load → AnnData

Load data:
    singlet.load("GSE149298")              Load an accession → AnnData (free download)
    singlet.load("path/to/data.singlet")   Load a local .singlet file → AnnData
    singlet.load(["GSE149298", "b.singlet"]) Load + concatenate several → one AnnData
    singlet.load_dir("/path/to/quant/GSM") Load a pipeline output directory → AnnData
    singlet.load_sample("GSM3308814")      Load single sample (column-range read)

Format I/O:
    singlet.read_kraken2("gse_dir/")       Read kraken2 microbiome matrix
    singlet.read_matrix("file")            Auto-detect single-block count matrix

Canonical v2 samples (multi-block, from the pipeline):
    sample = singlet.SingletSample("sample_dir/")
    counts = sample.counts.adata()         Spliced+unspliced AnnData

Pipeline (process raw reads → canonical sample):
    singlet.run_pipeline("SRR12345678", out_dir="results/")
    singlet.run_pipeline("https://.../SRR12345678", reference="GRCh38")
    singlet.transcode_v1_to_v2(old_dir, new_dir)  Migrate legacy outputs
    singlet.validate_sample(sample_dir)            Manifest + structure check

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
"""

__version__ = "2.0.0"

from singlet._aggregate import aggregate
from singlet._ambient_rna_score import ambient_rna_score
from singlet._annotate import annotate, gene_programs, project
from singlet._annotate_cell_types import annotate_cell_types
from singlet._augur_prioritize import augur_prioritize
from singlet._auth import login
from singlet._batch_evaluation import batch_evaluation
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
from singlet._cca import cca
from singlet._cell_communication import cell_communication
from singlet._cell_cycle import score_cell_cycle
from singlet._cell_cycle_regression import cell_cycle_regression
from singlet._cell_distances import cell_distances
from singlet._cell_level_de import cell_level_de
from singlet._cell_type_proportions import cell_type_proportions
from singlet._cluster_stability import cluster_stability
from singlet._coexpression_modules import coexpression_modules
from singlet._combat import combat
from singlet._composition_analysis import composition_analysis
from singlet._concat import concatenate
from singlet._connectivity_score import connectivity_score
from singlet._consensus_clustering import consensus_clustering
from singlet._correlation import correlation_matrix
from singlet._cross_species_correlation import cross_species_correlation
from singlet._cv_resolution import cross_validate_resolution
from singlet._de import rank_genes_groups, rank_genes_groups_df
from singlet._dendrogram import dendrogram
from singlet._denoise import denoise
from singlet._describe import describe
from singlet._differential_abundance import differential_abundance
from singlet._differential_test import differential_test
from singlet._diffmap import diffmap
from singlet._doublet_score_hybrid import doublet_score_hybrid
from singlet._downsample import downsample_counts
from singlet._dpt import dpt
from singlet._draw_graph import draw_graph
from singlet._embedding_density import embedding_density
from singlet._enrichr import enrichr, enrichr_from_de
from singlet._entropy_score import entropy_score
from singlet._export_df import obs_df, to_df, var_df
from singlet._fate_probabilities import fate_probabilities
from singlet._filter import filter_cells, filter_genes
from singlet._filter_de import filter_rank_genes_groups
from singlet._find_all_markers import find_all_markers
from singlet._gene_activity_score import gene_activity_score
from singlet._gene_correlation_network import gene_correlation_network
from singlet._gene_importance import gene_importance
from singlet._gene_module_score import gene_module_score
from singlet._gene_set_enrichment import gene_set_enrichment
from singlet._gene_set_variation import gene_set_variation
from singlet._gene_space_embedding import gene_space_embedding
from singlet._gene_trend_clustering import gene_trend_clustering
from singlet._harmony import harmony
from singlet._hotspot_genes import hotspot_genes
from singlet._hvg import highly_variable_genes
from singlet._hvg_cell_ranger import highly_variable_genes_cell_ranger
from singlet._hvg_seurat import highly_variable_genes_seurat_v3
from singlet._ica import ica
from singlet._identify_bipotent_cells import identify_bipotent_cells
from singlet._infer_grn import infer_grn
from singlet._ingest import ingest
from singlet._io import (
    info_1pz,
    read_1pz,
    read_kraken2,
    read_matrix,
    write_1pz,
)
from singlet.io.sample import (
    SingletCounts,
    SingletMt,
    SingletNonhost,
    SingletSample,
    SingletSnp,
    open_sample as open,
)
from singlet.fetch import fetch, default_cache_dir, default_base_url
from singlet.pipeline import PipelineError, Run, run as run_pipeline
from singlet.transcode import transcode_v1_to_v2
from singlet.manifest import validate_sample
from singlet.bundle import SingletBundle, pack_gse
from singlet.find import find, find_load
from singlet.views import gene_counts as view_gene_counts
from singlet.views import psi as view_psi
from singlet.views import usa as view_usa
from singlet._knn_impute import knn_impute
from singlet._leiden import leiden
from singlet._loader import download, load, load_dir, load_sample
from singlet._louvain import louvain
from singlet._magic import magic
from singlet._marker_overlap import marker_gene_overlap
from singlet._marker_specificity import marker_specificity
from singlet._metacell import metacell
from singlet._metrics import calinski_harabasz_score, silhouette_score
from singlet._mnn import mnn_correct
from singlet._morans_i import morans_i
from singlet._multiome_factor_analysis import multiome_factor_analysis
from singlet._neighbors import neighbors
from singlet._nmf import nmf
from singlet._normalize import normalize
from singlet._optimal_transport import optimal_transport
from singlet._paga import paga
from singlet._palantir_pseudotime import palantir_pseudotime
from singlet._pca import pca
from singlet._perturbation_signature import perturbation_signature
from singlet._phate import phate
from singlet._phenograph import phenograph
from singlet._plot import plot_dotplot, plot_umap, plot_violin
from singlet._plot_embedding import plot_embedding
from singlet._plot_genes_in_groups import plot_genes_in_groups
from singlet._plot_heatmap import plot_heatmap
from singlet._plot_markers import rank_genes_groups_dotplot
from singlet._plot_matrix import rank_genes_groups_matrixplot
from singlet._plot_paga import plot_paga
from singlet._plot_ranking import plot_ranking
from singlet._plot_scatter import plot_scatter
from singlet._plot_stacked_violin import plot_stacked_violin
from singlet._plot_tracks import rank_genes_groups_tracksplot
from singlet._predict_cell_type import predict_cell_type
from singlet._pseudobulk import pseudobulk
from singlet._qc import calculate_qc_metrics
from singlet._qc_summary import qc_summary
from singlet._query import query, search
from singlet._rank_genes import rank_genes
from singlet._recipes import recipe_seurat, recipe_zheng17
from singlet._regress import regress_out
from singlet._reprogramming_score import reprogramming_score
from singlet._scale import scale
from singlet._scanorama_integrate import scanorama_integrate
from singlet._score import score_genes
from singlet._scrublet import scrublet
from singlet._sparse_pca import sparse_pca
from singlet._spatial import spatial_neighbors
from singlet._spectral_clustering import spectral_clustering
from singlet._splicing_ratio import splicing_ratio
from singlet._subcluster import leiden_subclustering
from singlet._subsample import subsample
from singlet._subsample_balanced import subsample_balanced
from singlet._topic_model import topic_model
from singlet._trajectory_genes import trajectory_genes
from singlet._transfer_labels import transfer_labels
from singlet._transforms import expm1, log1p, sqrt_transform
from singlet._tsne import tsne
from singlet._umap import umap
from singlet._variance_partition import variance_partition
from singlet._variational_inference import variational_inference
from singlet._velocity_pseudotime import velocity_pseudotime
from singlet._versions import show_versions
from singlet._weighted_nearest_neighbors import weighted_nearest_neighbors
from singlet._wishart import wishart_test
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
    # Find (natural-language search)
    "find",
    "find_load",
    # Load
    "load",
    "load_sample",
    "load_dir",
    "download",
    "SingletBundle",
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
    # Canonical v2 readers
    "SingletSample",
    "SingletCounts",
    "SingletSnp",
    "SingletMt",
    "SingletNonhost",
    "open",
    "fetch",
    "default_cache_dir",
    "default_base_url",
    "view_gene_counts",
    "view_usa",
    "view_psi",
    # Pipeline
    "run_pipeline",
    "Run",
    "PipelineError",
    "transcode_v1_to_v2",
    "validate_sample",
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
    # DataFrame export
    "to_df",
    "obs_df",
    "var_df",
    # QC filtering
    "calculate_qc_metrics",
    "qc_summary",
    "filter_cells",
    "filter_genes",
    "subsample",
    "downsample_counts",
    "concatenate",
    "aggregate",
    "pseudobulk",
    # Preprocessing
    "normalize",
    "highly_variable_genes",
    "highly_variable_genes_seurat_v3",
    "highly_variable_genes_cell_ranger",
    "scale",
    "regress_out",
    "nmf",
    "pca",
    "sparse_pca",
    "ica",
    "magic",
    "neighbors",
    "leiden",
    "leiden_subclustering",
    "louvain",
    "tsne",
    "diffmap",
    "dpt",
    "umap",
    "paga",
    # Graph layout
    "draw_graph",
    # Differential expression
    "rank_genes_groups",
    "rank_genes_groups_df",
    "filter_rank_genes_groups",
    "rank_genes_groups_dotplot",
    "rank_genes_groups_matrixplot",
    "rank_genes_groups_tracksplot",
    "score_genes",
    "score_cell_cycle",
    "rank_genes",
    "dendrogram",
    "enrichr",
    "enrichr_from_de",
    "gene_set_enrichment",
    # Marker overlap
    "marker_gene_overlap",
    # Embedding density
    "embedding_density",
    # Batch correction
    "harmony",
    "combat",
    "mnn_correct",
    "scanorama_integrate",
    # Doublet detection
    "scrublet",
    "doublet_score_hybrid",
    # Correlation
    "correlation_matrix",
    # Connectivity
    "connectivity_score",
    # Recipes
    "recipe_seurat",
    "recipe_zheng17",
    # Spatial
    "spatial_neighbors",
    # Plotting
    "plot_umap",
    "plot_violin",
    "plot_dotplot",
    "plot_embedding",
    "plot_genes_in_groups",
    "plot_heatmap",
    "plot_ranking",
    "plot_scatter",
    "plot_stacked_violin",
    "plot_paga",
    # Integration
    "ingest",
    "transfer_labels",
    # Cell type annotation
    "annotate_cell_types",
    # Diagnostics
    "show_versions",
    # Transforms
    "log1p",
    "expm1",
    "sqrt_transform",
    # Clustering metrics
    "silhouette_score",
    "calinski_harabasz_score",
    "cross_validate_resolution",
    "wishart_test",
    # Pseudotime
    "velocity_pseudotime",
    # Palantir pseudotime
    "palantir_pseudotime",
    # Trajectory analysis
    "trajectory_genes",
    # Cell fate probabilities
    "fate_probabilities",
    # Denoising
    "denoise",
    # Metacell
    "metacell",
    # Gene correlation network
    "gene_correlation_network",
    # Gene regulatory network
    "infer_grn",
    # PhenoGraph clustering
    "phenograph",
    # Comprehensive marker finding
    "find_all_markers",
    # Cell composition
    "composition_analysis",
    # Cell type proportions
    "cell_type_proportions",
    # Splicing
    "splicing_ratio",
    # Neighborhood entropy
    "entropy_score",
    # kNN imputation
    "knn_impute",
    # Balanced subsampling
    "subsample_balanced",
    # Spatial statistics
    "morans_i",
    # Ambient RNA
    "ambient_rna_score",
    # Batch evaluation
    "batch_evaluation",
    # Canonical Correlation Analysis
    "cca",
    # Differential abundance
    "differential_abundance",
    # Gene module scoring
    "gene_module_score",
    # Gene activity/pathway scoring
    "gene_activity_score",
    # Cross-species correlation
    "cross_species_correlation",
    # Cluster stability
    "cluster_stability",
    # Bipotent cell identification
    "identify_bipotent_cells",
    # Variance partitioning
    "variance_partition",
    # Cell distances
    "cell_distances",
    # Optimal transport
    "optimal_transport",
    # Hotspot genes
    "hotspot_genes",
    # Cell-level DE
    "cell_level_de",
    # Weighted Nearest Neighbors
    "weighted_nearest_neighbors",
    # Gene Set Variation Analysis
    "gene_set_variation",
    # Cell-cell communication
    "cell_communication",
    # Consensus clustering
    "consensus_clustering",
    # Marker specificity scoring
    "marker_specificity",
    # Gene-space embedding
    "gene_space_embedding",
    # Reprogramming potential
    "reprogramming_score",
    # Distribution-level differential testing
    "differential_test",
    # Multi-modal factor analysis
    "multiome_factor_analysis",
    # Cell type prioritization
    "augur_prioritize",
    # PHATE embedding
    "phate",
    # Spectral clustering
    "spectral_clustering",
    # Gene trend clustering
    "gene_trend_clustering",
    # Perturbation signature
    "perturbation_signature",
    # Topic modeling
    "topic_model",
    # Cell type prediction
    "predict_cell_type",
    # Variational inference
    "variational_inference",
    # Coexpression modules
    "coexpression_modules",
    # Gene importance
    "gene_importance",
    # Cell cycle regression
    "cell_cycle_regression",
]
