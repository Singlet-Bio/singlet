Architecture
============

singlify is a two-layer system:

1. **C++ engine** (singlet-pileup): Header-only library that does all computation
2. **Python wrapper**: Thin pybind11 bindings + scipy result conversion

.. code-block:: text

   ┌─────────────────────────────────────────────┐
   │  Python: singlify.pileup()                  │
   │    ├─ Validates parameters                  │
   │    ├─ Calls C++ _core.pileup()              │
   │    └─ Wraps results as PileupResult         │
   │         ├─ scipy.sparse.csc_matrix          │
   │         ├─ PileupStats                      │
   │         └─ Feature/barcode name lists       │
   ├─────────────────────────────────────────────┤
   │  C++ _core: pybind11 module                 │
   │    ├─ Creates PileupEngine with config      │
   │    ├─ Loads references (barcodes, GTF, VCF) │
   │    ├─ Runs pileup (GIL released)            │
   │    └─ Returns CSC arrays as numpy views     │
   ├─────────────────────────────────────────────┤
   │  C++ singlet-pileup (header-only)           │
   │    ├─ pileup_engine.h  — Core BAM consumer  │
   │    ├─ gene_model.h     — GTF/BED parser     │
   │    ├─ sparse_accumulator.h — COO→CSC        │
   │    ├─ interval_tree.h  — Exon overlap       │
   │    ├─ umi_dedup.h      — FNV-1a hash dedup  │
   │    ├─ pz_writer.h      — .1pz format        │
   │    ├─ mtx_writer.h     — Matrix Market I/O  │
   │    ├─ mt_heteroplasmy.h — chrM analysis     │
   │    ├─ donor_demux.h    — VB Beta-Binomial   │
   │    └─ export.h         — Shared export logic │
   └─────────────────────────────────────────────┘

Performance Notes
-----------------

- The BAM pileup runs entirely in C++ with the GIL released — Python overhead is negligible
- COO → CSC conversion uses counting sort (O(nnz + n) time) for each accumulator
- Multiple CSC conversions run in parallel threads
- The engine processes ~2.7M reads/second on a single core
- Total wall time ≈ STAR alignment time + ~3s export overhead

Data Flow
---------

.. code-block:: text

   BAM file ──read──► PileupEngine.run()
                         │
                         ├─ SNP binary search → snp_ad, snp_dp (COO)
                         ├─ Exon overlap      → exon_acc (COO)
                         ├─ Intron overlap     → intron_acc (COO)
                         ├─ CIGAR N-ops        → sj_acc (COO)
                         └─ chrM bases         → mt_acc (COO)
                         │
                      to_csc() (parallel)
                         │
                         ▼
                    scipy.sparse.csc_matrix (Python)
