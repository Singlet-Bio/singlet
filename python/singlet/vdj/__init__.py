# SPDX-License-Identifier: MIT
"""V(D)J (TCR / BCR) libraries as a Singlet modality.

* :mod:`singlet.vdj.reference`: V / D / J / C segments with their CDR3 anchors, built from a genome and its Ensembl /
  GENCODE annotation (no IMGT or vendor reference).
* :mod:`singlet.vdj.assemble`: reads -> corrected barcodes and UMIs -> junction contigs per cell -> annotated contigs.
* :mod:`singlet.vdj.clonotype`: per-cell chains, pairing and clonotypes.
* :mod:`singlet.vdj.qc`: library QC with pass/fail thresholds.
* :mod:`singlet.vdj.members`: the members (``vdj_contigs.parquet``, ``vdj_summary.json``) aligned to the GEX ``cells`` table.

Design and benchmark: ``docs/vdj-crispr-modalities.md``.
"""

VDJ_SCHEMA = "1"
