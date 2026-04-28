singlify
========

Streaming BAM pileup engine for single-cell multi-omics feature extraction.

**singlify** processes a BAM file in a single pass, extracting per-barcode counts
for exon/intron expression, splice junctions, SNP genotypes, and chrM alleles.
The C++ engine returns results directly as scipy sparse matrices.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   api
   architecture
   vignette

Installation
------------

.. code-block:: bash

   cd singlify/python
   pip install -e .

Quick Example
-------------

.. code-block:: python

   import singlify

   result = singlify.pileup(
       bam_path="sample.bam",
       barcode_path="barcodes.tsv",
       exon_gtf_path="genes.gtf.gz",
   )

   # Access as scipy sparse matrices
   print(result.exons.shape)      # (n_features, n_barcodes)
   print(result.stats.total_reads) # 72,000,000

   # Convert to AnnData for scanpy
   adata = result.to_anndata()
