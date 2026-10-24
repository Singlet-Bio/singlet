# SPDX-License-Identifier: MIT
"""Antibody-derived tags (ADT / CITE-seq) and cell hashtags (HTO) as a Singlet modality.

* :mod:`singlet.adt.seq`: 2-bit sequence codes and Hamming-1 neighbourhoods (vectorised).
* :mod:`singlet.adt.chemistry`: read geometry per protocol (10x 3' v2/v3/v4, 5'), whitelists and the 3' v3 feature
  barcode translation.
* :mod:`singlet.adt.panel`: the TotalSeq catalog, study feature references, panel discovery from the reads.
* :mod:`singlet.adt.count`: counting UMIs per (cell barcode, antibody) with Cell Ranger feature-barcoding logic.
* :mod:`singlet.adt.qc`: library and cell QC, pass/fail thresholds.
* :mod:`singlet.adt.normalize`: derived layers (DSB with empty droplets, CLR); raw counts stay the stored truth.
* :mod:`singlet.adt.hto`: hashtag demultiplexing calls.
* :mod:`singlet.adt.store`: the v2 members (``adt_counts.1pz`` / ``hto_counts.1pz`` pz2 matrices aligned to the GEX ``cells`` table,
  ``adt_panel.json``).

Design and QC report: ``docs/adt-modality.md``.
"""

ADT_SCHEMA = "1"
