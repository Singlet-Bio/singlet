API Reference
=============

Main Function
-------------

.. autofunction:: singlify.pileup

Result Container
----------------

.. autoclass:: singlify.PileupResult
   :members:
   :undoc-members:

Statistics
----------

.. autoclass:: singlify.PileupStats
   :members:
   :undoc-members:

   .. attribute:: total_reads
      :type: int

      Total reads seen in BAM.

   .. attribute:: mapped_reads
      :type: int

      Reads passing alignment filter.

   .. attribute:: barcoded_reads
      :type: int

      Reads with valid cell barcodes.

   .. attribute:: snp_hits
      :type: int

      Reads overlapping SNP positions.

   .. attribute:: exon_hits
      :type: int

      Reads assigned to exons.

   .. attribute:: intron_hits
      :type: int

      Reads assigned to introns.

   .. attribute:: sj_hits
      :type: int

      Splice junction observations.

   .. attribute:: chrm_reads
      :type: int

      Reads on chrM.

   .. attribute:: mt_pileup_bases
      :type: int

      Total bases piled up on chrM.

   .. attribute:: low_mapq
      :type: int

      Reads filtered by MAPQ threshold.

   .. attribute:: secondary_reads
      :type: int

      Secondary alignments.

   .. attribute:: no_barcode
      :type: int

      Reads without valid barcode.

   .. attribute:: no_umi
      :type: int

      Reads without UMI tag.

   .. attribute:: umi_unique
      :type: int

      Unique UMI observations.

   .. attribute:: umi_duplicate
      :type: int

      Duplicate UMIs deduplicated.

   .. attribute:: multimapper_reads
      :type: int

      Multi-mapped reads (NH > 1).

   .. attribute:: multigene_reads
      :type: int

      Reads mapping to multiple genes.

   .. attribute:: wrong_strand
      :type: int

      Reads on wrong strand (filtered).

   .. attribute:: wall_time_s
      :type: float

      Wall-clock time in seconds.
