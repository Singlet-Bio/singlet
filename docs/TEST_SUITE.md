# Singlify Test Suite

## Quick reference

Build and run all droplet-hardening tests:
```bash
source /opt/rh/gcc-toolset-13/enable
cd singlify && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
for t in test_summary_json test_ws2_protocol_hardening test_cell_calling \
         test_whitelist_integrity test_ws4_crash_diagnostics test_mega_sort_params; do
    ./build/$t
done
```

## Droplet-hardening suites (990 tests)

| Suite | Tests | Focus |
|-------|-------|-------|
| `test_summary_json` | 270 | classify_outcome thresholds, json_escape, PipelineSummary serialization, STAR log parsing, species mismatch warnings, mapping rate edge cases, data_incomplete paths, metrics formatting, saturation formula, assay type classification, JSON edge cases, structural JSON validity |
| `test_ws2_protocol_hardening` | 157 | Protocol detection, known_protocols() registry integrity, linker detection, geometry validation, tag normalization, alias resolution, confidence ordering, candidate scoring |
| `test_whitelist_integrity` | 107 | Whitelist file existence, barcode length validation, sorted order, duplicate detection |
| `test_mega_sort_params` | 87 | BAM sort parameter computation for various memory tiers |
| `test_ws4_crash_diagnostics` | 81 | Crash diagnostics, guard clause coverage, graceful exit paths |
| `test_1fq_header` | 68 | .1fq header struct layout, magic validation, field offsets, stream_lengths semantics, binary roundtrip, DecodedBlock accessors, quality guard |
| `test_cell_calling` | 67 | EmptyDrops Monte Carlo, chi-squared p-values, BH FDR, knee fallback, spurious knee guard, lgamma_nr, AliasTable sampling, compute_deviance, gammaq, mc_null_deviances, mc_emptydrops_pvalues, bimodal/exponential/step knee distributions |
| `test_sex_calling` | 21 | Sex inference from XIST/chrY markers, ambiguous signal resolution, threshold boundaries, JSON output, multi-cell aggregation, UMI counting |

## Additional suites (compiled independently)

| Suite | Tests | Focus |
|-------|-------|-------|
| `test_barcode_rank` | 41 | Barcode rank plot generation |
| `test_wl_ambient` | 13 | Whitelist ambient profiling, WL+bc_index supplement, ceil filtering, stress test (1000 genes) |
| `test_adt_counter` | 15 | ADT UMI deduplication, multi-tag, empty finalize, large cell index |
| `test_adt_matcher` | 13 | ADT tag matching, Hamming-1, case sensitivity, thread safety |
| `test_interval_tree` | 16 | Interval queries, nested, large coordinates, empty tree |
| `test_nonhost` | 10 | MinSketch index build, classify, batch vs sequential, edge cases |
| `test_nonhost_host_filter` | 12 | Bloom filter FPR, host subtraction, batch consistency |
| `test_nonhost_em` | 9 | EM deconvolution, multi-species, ambiguous reads |
| `test_nonhost_unmapped_capture` | 7 | Unmapped read capture, revcomp, homopolymer |
| `test_nonhost_db_build` | 7 | Database construction, merge, empty build |
| `test_sparse_accumulator` | 19 | CSC matrix construction, large-N, single barcode, tall matrix, uint32 range, column/row totals, empty CSC, uint16 max, indptr monotonicity |
| `test_complex_whitelist` | 49 | Multi-segment whitelist parsing (BD Rhapsody, SPLiT-seq) |
| `test_pz_reader` | 45 | .1pz VOCSC round-trip (uint8/16/32, multi-chunk, empty matrix, corrupt header, metadata) |
| `test_tiny_dataset_guard` | 51 | Read thresholds, RAM tiers, boundary precision, constants, monotonicity |
| `test_hto_demux` | 11 | HTO CLR demux, singlet/doublet/negative, large panel, TSV output |
| `test_cellplex` | 11 | CellPlex CMO Otsu demux, multiplet detection, 4-CMO panel, sample_idx |
| `test_alignment_qc` | 28 | Mapping rate, duplication, Lander-Waterman, 5'/3' ratio, gene body coverage |
| `test_atac_qc` | 17 | ATAC fragment QC, mito fraction, TSS enrichment, multi-cell, histogram |
| `test_combinatorial_barcode` | ~10 | Combinatorial barcode factories |

## Test architecture

Tests are standalone C++ executables. No test framework dependency (no gtest/catch2).
Each test prints `PASS:` or `FAIL:` per check and returns exit code 0/1.

Header-only modules (summary_json.h, cell_calling.h, protocol.h) are tested without
linking against htslib/STAR. This keeps compile times <2s per suite.

## Key thresholds tested

- **Mapping rate**: 50% (scrna), 30% (ATAC)
- **Cell count**: 0 → align_zero_cells, <10 → align_low_cells
- **Median genes**: <200 → align_low_genes
- **Species mismatch**: mismatch_rate >5% AND MR <30% → warning
- **Ambient coverage**: <40% → knee fallback (not EmptyDrops)
