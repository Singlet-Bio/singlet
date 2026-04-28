Vignette: Multi-Omic Feature Extraction
========================================

This vignette demonstrates singlify's Python API for extracting multi-omic
features from a 10x Chromium single-cell RNA-seq BAM file.

Setup
-----

.. code-block:: python

   import singlify
   import scipy.sparse as sp
   import numpy as np

   # Paths to data
   BAM = "path/to/Aligned.sortedByCoord.out.bam"
   BARCODES = "path/to/barcodes.tsv"
   GTF = "path/to/genes.gtf.gz"
   VCF = "path/to/common_snps.vcf.gz"

Step 1: Basic Gene Expression
------------------------------

Extract exon and intron counts in one pass:

.. code-block:: python

   result = singlify.pileup(
       bam_path=BAM,
       barcode_path=BARCODES,
       exon_gtf_path=GTF,
   )

   print(result)
   # PileupResult(reads=72,000,000, exons=33,538×5,000, introns=33,538×5,000)

   # Access the matrices
   exon_counts = result.exons       # scipy.sparse.csc_matrix, features × barcodes
   intron_counts = result.introns
   sj_counts = result.splice_junctions

   # Read-level QC
   print(f"Total reads: {result.stats.total_reads:,}")
   print(f"Mapped: {result.stats.mapped_reads:,}")
   print(f"Barcoded: {result.stats.barcoded_reads:,}")
   print(f"UMI unique: {result.stats.umi_unique:,}")
   print(f"UMI duplicate: {result.stats.umi_duplicate:,}")

Step 2: RNA Velocity via Introns
---------------------------------

Intron counts enable RNA velocity analysis. singlify provides both exon
(spliced) and intron (unspliced) counts from a single pass:

.. code-block:: python

   # Gene-level spliced / unspliced counts
   spliced = result.exons     # features × barcodes
   unspliced = result.introns

   # Compute unspliced fraction per cell
   unspliced_frac = np.array(unspliced.sum(axis=0)).flatten() / (
       np.array(spliced.sum(axis=0)).flatten() +
       np.array(unspliced.sum(axis=0)).flatten() + 1e-8
   )
   print(f"Median unspliced fraction: {np.median(unspliced_frac):.3f}")

Step 3: SNP Genotyping for Demultiplexing
------------------------------------------

Adding a VCF file enables per-cell SNP genotyping (useful for donor
demultiplexing in pooled experiments):

.. code-block:: python

   result = singlify.pileup(
       bam_path=BAM,
       barcode_path=BARCODES,
       exon_gtf_path=GTF,
       snp_path=VCF,
       count_mt=True,  # Also collect chrM data
   )

   # SNP matrices
   snp_ad = result.snp_ad  # Alternate allele depth
   snp_dp = result.snp_dp  # Total depth

   # Allele frequency per cell
   with np.errstate(divide="ignore", invalid="ignore"):
       af = snp_ad.toarray() / np.maximum(snp_dp.toarray(), 1)

   print(f"SNPs genotyped: {snp_ad.shape[0]:,}")
   print(f"Median SNP depth per cell: {np.median(np.array(snp_dp.sum(axis=0))):.0f}")

Step 4: AnnData Integration
----------------------------

Convert results to scanpy-compatible AnnData:

.. code-block:: python

   adata = result.to_anndata()

   # adata.X = exon counts (barcodes × features, CSR format)
   # adata.layers["introns"] = intron counts
   # adata.layers["splice_junctions"] = SJ counts
   # adata.obs_names = barcode strings
   # adata.var_names = feature (gene) names

   print(adata)
   # AnnData object with n_obs × n_vars = 5000 × 33538
   #     obs: 'barcode'
   #     var: 'feature_name'
   #     layers: 'introns', 'splice_junctions'

   # Continue with standard scanpy workflow
   import scanpy as sc
   sc.pp.filter_cells(adata, min_genes=200)
   sc.pp.filter_genes(adata, min_cells=3)
   sc.pp.normalize_total(adata, target_sum=1e4)
   sc.pp.log1p(adata)

Step 5: Performance Tuning
---------------------------

For large BAMs (>100M reads), increase threads:

.. code-block:: python

   result = singlify.pileup(
       bam_path=BAM,
       barcode_path=BARCODES,
       exon_gtf_path=GTF,
       threads=8,  # BAM decompression threads
   )

   print(f"Wall time: {result.stats.wall_time_s:.1f}s")
   reads_per_sec = result.stats.total_reads / result.stats.wall_time_s
   print(f"Throughput: {reads_per_sec / 1e6:.1f}M reads/s")
