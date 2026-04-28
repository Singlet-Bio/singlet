Quick Start
===========

Installation
------------

From the singlify repository:

.. code-block:: bash

   cd singlify/python
   pip install -e .

Requirements:

- Python ≥ 3.9
- htslib (≥ 1.17, typically via conda: ``conda install -c bioconda htslib``)
- C++17 compiler (GCC ≥ 10 or Clang ≥ 12)

Basic Usage
-----------

The main entry point is :func:`singlify.pileup`:

.. code-block:: python

   import singlify

   result = singlify.pileup(
       bam_path="Aligned.sortedByCoord.out.bam",
       barcode_path="barcodes.tsv",
       exon_gtf_path="genes.gtf.gz",
   )

This returns a :class:`singlify.PileupResult` with scipy sparse matrices:

.. code-block:: python

   result.exons            # scipy.sparse.csc_matrix (features × barcodes)
   result.introns          # scipy.sparse.csc_matrix or None
   result.splice_junctions # scipy.sparse.csc_matrix or None
   result.stats            # PileupStats with read counts
   result.barcodes         # list of barcode strings
   result.exon_names       # list of feature names

SNP Genotyping
--------------

To also extract per-cell SNP genotypes, provide a VCF file:

.. code-block:: python

   result = singlify.pileup(
       bam_path="sample.bam",
       barcode_path="barcodes.tsv",
       exon_gtf_path="genes.gtf.gz",
       snp_path="common_snps.vcf.gz",
   )

   result.snp_ad  # Alternate allele depth (n_snps × n_barcodes)
   result.snp_dp  # Total depth (n_snps × n_barcodes)

Parameters
----------

Control engine behavior with keyword arguments:

.. code-block:: python

   result = singlify.pileup(
       bam_path="sample.bam",
       barcode_path="barcodes.tsv",
       exon_gtf_path="genes.gtf.gz",
       threads=8,           # BAM decompression threads
       min_mapq=30,         # Stricter mapping quality
       count_introns=False,  # Disable intron counting
       count_mt=True,        # Enable chrM allele pileup
   )

AnnData Integration
-------------------

Convert results directly to scanpy-compatible AnnData:

.. code-block:: python

   adata = result.to_anndata()
   # adata.X = exon counts (barcodes × features, CSR)
   # adata.layers["introns"] = intron counts
   # adata.obs_names = barcode strings
   # adata.var_names = feature names
