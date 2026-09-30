API reference
=============

The most commonly used entry points, grouped by task. Every function is
also listed alphabetically in the :ref:`full-api-reference` below, with a
generated page giving its complete signature and docstring.

Browse the catalog
-------------------

Lookups against the catalog snapshot bundled with the package. They work
offline but lag the live catalog at https://singlet.bio; :func:`singlet.info`
falls back to the live REST API for accessions the snapshot does not know,
and ``python -m singlet`` prints live totals.

.. autosummary::
   :toctree: generated
   :nosignatures:

   singlet.catalog
   singlet.info
   singlet.sample_index
   singlet.species
   singlet.tissues
   singlet.protocols
   singlet.datasets
   singlet.samples
   singlet.top_series
   singlet.quality_tiers
   singlet.failure_categories
   singlet.cell_types
   singlet.summary

Find data
---------

Natural-language search over the live catalog. ``find`` returns study
(``GSE``) accessions by default; ``find_load`` loads the top three.

.. autosummary::
   :toctree: generated
   :nosignatures:

   singlet.find
   singlet.find_load
   singlet.set_api_key

Load data
---------

.. autosummary::
   :toctree: generated
   :nosignatures:

   singlet.load
   singlet.download
   singlet.open_bundle
   singlet.SingletBundle
   singlet.load_dir
   singlet.SingletSample

Format I/O
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   singlet.read_matrix
   singlet.read_kraken2
   singlet.read_1pz
   singlet.write_1pz
   singlet.info_1pz
   singlet.from_h5ad
   singlet.from_mtx
   singlet.from_zarr
   singlet.from_tiledb
   singlet.to_h5ad
   singlet.to_mtx
   singlet.to_csc
   singlet.to_zarr
   singlet.to_tiledb

Pipeline
--------

Process raw reads (FASTQ / SRA / ENA accessions) into a canonical
``.singlet`` sample.

.. autosummary::
   :toctree: generated
   :nosignatures:

   singlet.run_pipeline
   singlet.transcode_v1_to_v2
   singlet.validate_sample

Cell type annotation
---------------------

Local annotation from marker genes or a labelled reference — no external
API calls. (The former NMF gene-program functions ``gene_programs``,
``project`` and ``annotate`` depended on a model host that is not in service
and now raise :class:`NotImplementedError`.)

.. autosummary::
   :toctree: generated
   :nosignatures:

   singlet.annotate_cell_types
   singlet.predict_cell_type
   singlet.transfer_labels

QC and preprocessing
---------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   singlet.describe
   singlet.calculate_qc_metrics
   singlet.qc_summary
   singlet.filter_cells
   singlet.filter_genes
   singlet.normalize
   singlet.highly_variable_genes
   singlet.scale
   singlet.regress_out

Configuration
-------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   singlet.set_catalog_dir

.. _full-api-reference:

Full API reference
-------------------

Every public name in ``singlet.__all__``, alphabetically.

.. autosummary::
   :toctree: generated
   :nosignatures:

    singlet.aggregate
    singlet.ambient_rna_score
    singlet.annotate_cell_types
    singlet.augur_prioritize
    singlet.batch_evaluation
    singlet.calculate_qc_metrics
    singlet.calinski_harabasz_score
    singlet.catalog
    singlet.cca
    singlet.cell_communication
    singlet.cell_cycle_regression
    singlet.cell_distances
    singlet.cell_level_de
    singlet.cell_type_proportions
    singlet.cell_types
    singlet.cluster_stability
    singlet.coexpression_modules
    singlet.combat
    singlet.composition_analysis
    singlet.concatenate
    singlet.connectivity_score
    singlet.consensus_clustering
    singlet.correlation_matrix
    singlet.cross_species_correlation
    singlet.cross_validate_resolution
    singlet.datasets
    singlet.default_base_url
    singlet.default_cache_dir
    singlet.dendrogram
    singlet.denoise
    singlet.describe
    singlet.differential_abundance
    singlet.differential_test
    singlet.diffmap
    singlet.doublet_score_hybrid
    singlet.download
    singlet.downsample_counts
    singlet.dpt
    singlet.draw_graph
    singlet.embedding_density
    singlet.enrichr
    singlet.enrichr_from_de
    singlet.entropy_score
    singlet.expm1
    singlet.failure_categories
    singlet.fate_probabilities
    singlet.fetch
    singlet.filter_cells
    singlet.filter_genes
    singlet.filter_rank_genes_groups
    singlet.find
    singlet.find_all_markers
    singlet.find_load
    singlet.from_h5ad
    singlet.from_mtx
    singlet.from_tiledb
    singlet.from_zarr
    singlet.gene_activity_score
    singlet.gene_correlation_network
    singlet.gene_importance
    singlet.gene_module_score
    singlet.gene_set_enrichment
    singlet.gene_set_variation
    singlet.gene_space_embedding
    singlet.gene_trend_clustering
    singlet.harmony
    singlet.highly_variable_genes
    singlet.highly_variable_genes_cell_ranger
    singlet.highly_variable_genes_seurat_v3
    singlet.hotspot_genes
    singlet.ica
    singlet.identify_bipotent_cells
    singlet.infer_grn
    singlet.info
    singlet.info_1pz
    singlet.ingest
    singlet.knn_impute
    singlet.leiden
    singlet.leiden_subclustering
    singlet.load
    singlet.load_dir
    singlet.log1p
    singlet.louvain
    singlet.magic
    singlet.marker_gene_overlap
    singlet.marker_specificity
    singlet.metacell
    singlet.mnn_correct
    singlet.morans_i
    singlet.multiome_factor_analysis
    singlet.neighbors
    singlet.nmf
    singlet.normalize
    singlet.obs_df
    singlet.open
    singlet.open_bundle
    singlet.optimal_transport
    singlet.paga
    singlet.palantir_pseudotime
    singlet.pca
    singlet.perturbation_signature
    singlet.phate
    singlet.phenograph
    singlet.PipelineError
    singlet.plot_dotplot
    singlet.plot_embedding
    singlet.plot_genes_in_groups
    singlet.plot_heatmap
    singlet.plot_paga
    singlet.plot_ranking
    singlet.plot_scatter
    singlet.plot_stacked_violin
    singlet.plot_umap
    singlet.plot_violin
    singlet.predict_cell_type
    singlet.protocols
    singlet.pseudobulk
    singlet.qc_summary
    singlet.quality_tiers
    singlet.rank_genes
    singlet.rank_genes_groups
    singlet.rank_genes_groups_df
    singlet.rank_genes_groups_dotplot
    singlet.rank_genes_groups_matrixplot
    singlet.rank_genes_groups_tracksplot
    singlet.read_1pz
    singlet.read_kraken2
    singlet.read_matrix
    singlet.recipe_seurat
    singlet.recipe_zheng17
    singlet.refresh
    singlet.regress_out
    singlet.reprogramming_score
    singlet.Run
    singlet.run_pipeline
    singlet.sample_index
    singlet.samples
    singlet.scale
    singlet.scanorama_integrate
    singlet.score_cell_cycle
    singlet.score_genes
    singlet.scrublet
    singlet.set_api_key
    singlet.set_catalog_dir
    singlet.show_versions
    singlet.silhouette_score
    singlet.SingletBundle
    singlet.SingletCounts
    singlet.SingletMt
    singlet.SingletNonhost
    singlet.SingletSample
    singlet.SingletSnp
    singlet.sparse_pca
    singlet.spatial_neighbors
    singlet.species
    singlet.spectral_clustering
    singlet.splicing_ratio
    singlet.sqrt_transform
    singlet.subsample
    singlet.subsample_balanced
    singlet.summary
    singlet.tissues
    singlet.to_csc
    singlet.to_df
    singlet.to_h5ad
    singlet.to_mtx
    singlet.to_tiledb
    singlet.to_zarr
    singlet.top_series
    singlet.topic_model
    singlet.trajectory_genes
    singlet.transcode_v1_to_v2
    singlet.transfer_labels
    singlet.tsne
    singlet.umap
    singlet.validate_sample
    singlet.var_df
    singlet.variance_partition
    singlet.variational_inference
    singlet.velocity_pseudotime
    singlet.view_gene_counts
    singlet.view_psi
    singlet.view_usa
    singlet.weighted_nearest_neighbors
    singlet.wishart_test
    singlet.write_1pz
