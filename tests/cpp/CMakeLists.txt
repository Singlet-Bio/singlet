# C++ unit tests — header-only pileup/pz/fq tests
# Requires: zstd, zlib, htslib, OpenMP, pthreads
# Optional: HDF5 >= 1.12 (for h5ad/loom/molecule_info tests; the writers use
# H5Treclaim, which replaced H5Dvlen_reclaim in 1.12 -- older HDF5 such as
# Ubuntu 24.04's 1.10.x is skipped rather than failing the build)

include(FindPkgConfig)
pkg_check_modules(HTSLIB REQUIRED htslib>=1.10)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(HDF5 1.12 COMPONENTS C)
if(HDF5_FOUND)
    message(STATUS "HDF5 ${HDF5_VERSION} found -- building h5ad/loom/molecule_info tests")
else()
    message(STATUS "HDF5 >= 1.12 not found -- skipping h5ad/loom/molecule_info tests")
endif()

enable_testing()

set(PILEUP_TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/singlet/pileup
    ${CMAKE_SOURCE_DIR}/include/singlet
    ${HTSLIB_INCLUDE_DIRS}
)

set(PILEUP_TEST_LINK_LIBS
    ${ZSTD_LIBRARY}
    ZLIB::ZLIB
    ${HTSLIB_LIBRARIES}
    OpenMP::OpenMP_CXX
    Threads::Threads
)

# Helper macro for adding test executables
macro(add_pileup_test name)
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE ${PILEUP_TEST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE ${PILEUP_TEST_LINK_LIBS})
    target_link_directories(${name} PRIVATE ${HTSLIB_LIBRARY_DIRS})
    target_compile_options(${name} PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)
    add_test(NAME ${name} COMMAND ${name})
endmacro()

# Core codec tests
add_pileup_test(test_pz_reader)
add_pileup_test(test_pz_hardening)

# pz2 (v2 matrix codec). Header-only C++17, runtime ISA dispatch: deliberately built WITHOUT -march flags.
# test_pz2_roundtrip also reads the legacy corpora under bench/compression (skips when they are absent).
macro(add_pz2_test name)
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tests/cpp)
    target_link_libraries(${name} PRIVATE ${ZSTD_LIBRARY} Threads::Threads)
    target_compile_definitions(${name} PRIVATE PZ2_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    target_compile_options(${name} PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)
    add_test(NAME ${name} COMMAND ${name})
endmacro()
add_pz2_test(test_pz2_api)
add_pz2_test(test_pz2_edge)
add_pz2_test(test_pz2_roundtrip)
# The corpus round trip decodes every matrix on every supported ISA in both modes: about an hour for the full corpus. Set
# PZ2_TEST_MAX_MB=8 (environment) for a quick run; it skips quietly when bench/compression/corpus* are absent.
set_tests_properties(test_pz2_roundtrip PROPERTIES TIMEOUT 14400)
add_executable(test_pz2_multi_tu test_pz2_multi_tu.cpp test_pz2_multi_tu_b.cpp)   # two TUs include the header-only codec
target_include_directories(test_pz2_multi_tu PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_pz2_multi_tu PRIVATE ${ZSTD_LIBRARY} Threads::Threads)
add_test(NAME test_pz2_multi_tu COMMAND test_pz2_multi_tu)

# Cell calling / QC
add_pileup_test(test_cell_calling)
add_pileup_test(test_called_v2)   # C++ port of python/singlet/s1/cellcall.py vs its output (called_v2_fixture.h)
target_include_directories(test_called_v2 PRIVATE ${CMAKE_SOURCE_DIR}/tests/cpp)
target_compile_definitions(test_called_v2 PRIVATE CV2_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/data/called_ed_golden")
add_pileup_test(test_barcode_rank)
add_pileup_test(test_cell_qc)
add_pileup_test(test_tiny_dataset_guard)
add_pileup_test(test_barcode_metrics)

# Pipeline features
add_pileup_test(test_mega_sort_params)
add_pileup_test(test_bus_writer)
add_pileup_test(test_transcript_compat)
add_pileup_test(test_barnyard)
add_pileup_test(test_barnyard_route)
add_pileup_test(test_doublet_scores)
add_pileup_test(test_saturation_curve)
add_pileup_test(test_splice_psi)
add_pileup_test(test_em_rescue)
add_pileup_test(test_molecule_labels)
add_pileup_test(test_mt_numt_tie)
add_pileup_test(test_strand_detect)
add_pileup_test(test_run_screen)
add_pileup_test(test_tpm_fpkm)
add_pileup_test(test_atac_peaks)
add_pileup_test(test_protocol_confidence_override)

# ATAC tests
add_pileup_test(test_atac_fragment)
add_pileup_test(test_atac_peak_caller)
add_pileup_test(test_atac_metrics)
add_pileup_test(test_atac_bin_counter)
add_pileup_test(test_atac_qc)
add_pileup_test(test_atac_cell_caller)
add_pileup_test(test_atac_donor_demux)
add_pileup_test(test_atac_ancestry_sex)
add_pileup_test(test_bulk_atac)
add_pileup_test(test_chipseq)
add_pileup_test(test_cuttag)

# ADT / HTO / feature barcodes
add_pileup_test(test_adt_counter)
add_pileup_test(test_adt_matcher)
add_pileup_test(test_hto_demux)
add_pileup_test(test_cellplex)
add_pileup_test(test_crispr_guide)

# Species / alignment
add_pileup_test(test_species_detect)
add_pileup_test(test_species_registry)
add_pileup_test(test_alignment_qc)
add_pileup_test(test_sex_calling)
add_pileup_test(test_rrna_detect)
add_pileup_test(test_txome_aligner)

# Nonhost detection
add_pileup_test(test_nonhost)
add_pileup_test(test_nonhost_aligner)
add_pileup_test(test_nonhost_cell_matrix)
add_pileup_test(test_nonhost_db_build)
add_pileup_test(test_nonhost_em)
add_pileup_test(test_nonhost_host_filter)
add_pileup_test(test_nonhost_unmapped_capture)

# Export formats
add_pileup_test(test_mtx_export)
add_pileup_test(test_velocity_export)
add_pileup_test(test_fragments_writer)
add_pileup_test(test_tagged_bam)
add_pileup_test(test_web_summary)
add_pileup_test(test_summary_json)

# Doublet / ambient / dedup
add_pileup_test(test_doublet_detect_v2)
add_pileup_test(test_doublet_calibration)   # v7 Scrublet-style scorer: planted doublets, rate-calibrated calls, determinism
add_pileup_test(test_ambient_profile)
add_pileup_test(test_read_dedup_stats)
add_pileup_test(test_donor_demux_perf)
add_pileup_test(test_wl_ambient)
add_pileup_test(test_ambient_realistic)
add_pileup_test(test_ambient_decontx)   # gene-level cluster-EM ambient estimator + integer corrected matrix
add_pileup_test(test_sparse_accumulator)
add_pileup_test(test_pseudocount_sensitivity)
add_pileup_test(test_gray_zone)
add_pileup_test(test_mc_pvalue_calibration)

# Modality-specific
add_pileup_test(test_mt_events)
add_pileup_test(test_mt_caller)   # reference-based, strand-aware chrM caller + private haplotype outputs
add_pileup_test(test_rna_variant_caller)
add_pileup_test(test_longread)
add_pileup_test(test_methyl)
add_pileup_test(test_mirna)
add_pileup_test(test_slamseq)
add_pileup_test(test_te_classifier)
add_pileup_test(test_te_counter)
add_pileup_test(test_vdj_assembly)
add_pileup_test(test_scdna_cnv)
add_pileup_test(test_hic_contacts)
add_pileup_test(test_wgs_assembly)

# Spatial
add_pileup_test(test_visium_spatial)
add_pileup_test(test_visium_qc)
add_pileup_test(test_visium_tissue)
add_pileup_test(test_spatial_barcodes)
add_pileup_test(test_spatial_multiome)

# Protocol detection / barcode handling
add_pileup_test(test_combinatorial_barcode)
add_pileup_test(test_complex_whitelist)
add_pileup_test(test_indrop_detect)
add_pileup_test(test_seqwell_detect)
add_pileup_test(test_cascade_router)
add_pileup_test(test_multiome_router)
add_pileup_test(test_multiome_cell_calling)
add_pileup_test(test_interop)

# Reference / fq
add_pileup_test(test_ref_fetch)

# Misc
add_pileup_test(test_logging)
add_pileup_test(test_interval_tree)
add_pileup_test(test_benchmark_suite)

# UMI / dedup / velocity / bloom / saturation / read_stats / provenance / minimizer / cascade / pz_writer / ancestry / ase / mtx_writer
add_pileup_test(test_umi_dedup)
add_pileup_test(test_bloom_filter)
add_pileup_test(test_velocity)
add_pileup_test(test_saturation)
add_pileup_test(test_read_stats)
add_pileup_test(test_provenance)
add_pileup_test(test_minimizer_index)
add_pileup_test(test_cascade_stats_writer)
add_pileup_test(test_pz_writer)
add_pileup_test(test_ancestry)
add_pileup_test(test_ase)
add_pileup_test(test_mtx_writer)

# Ported from singlify (2026-05-26): TE k-mer index, txome gene index, sylph nonhost,
# 1fq header hardening, cascade stats v2, donor k=1 workaround, feature barcode detect,
# RAM tier, species short R2, whitelist integrity, WS2/WS4 protocol+crash hardening
add_pileup_test(test_te_kmer_index)
add_pileup_test(test_te_kmer_integration)
add_pileup_test(test_txome_gene_index)
add_pileup_test(test_nonhost_sylph)
add_pileup_test(test_1fq_header)
add_pileup_test(test_cascade_stats)
add_pileup_test(test_donor_k1_workaround)
add_pileup_test(test_demux_doublet_autok)   # genotype doublets + held-out auto-K on synthetic pools
add_pileup_test(test_prefix_probe)   # 5' constant-prefix probe on reads spread through the file (no false clip on sorted duplicates)
add_pileup_test(test_polya_sites)    # poly(A) site detection/clustering, STAR CellReads.stats parsing, sj_features
add_pileup_test(test_feature_barcode_detect)
add_pileup_test(test_ram_tier)
add_pileup_test(test_species_short_r2)
add_pileup_test(test_whitelist_integrity)
# The real barcode whitelist files ship with the Python package, not under a
# repo-root "whitelists/" dir (the compiled-in fallback the test falls back
# to when this isn't set).
target_compile_definitions(test_whitelist_integrity PRIVATE
    WHITELIST_DIR="${CMAKE_SOURCE_DIR}/python/singlet/data/whitelists")
add_pileup_test(test_ws2_protocol_hardening)
add_pileup_test(test_ws4_crash_diagnostics)

# HDF5-dependent tests (h5ad, loom, molecule_info, and anything else that
# transitively includes export.h -> h5ad_writer.h -> <hdf5.h>)
if(HDF5_FOUND)
    macro(add_hdf5_test name)
        add_executable(${name} ${name}.cpp)
        target_include_directories(${name} PRIVATE ${PILEUP_TEST_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})
        target_link_libraries(${name} PRIVATE ${PILEUP_TEST_LINK_LIBS} ${HDF5_LIBRARIES})
        target_link_directories(${name} PRIVATE ${HTSLIB_LIBRARY_DIRS})
        target_compile_options(${name} PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare)
        add_test(NAME ${name} COMMAND ${name})
    endmacro()

    add_hdf5_test(test_h5ad_writer)
    add_hdf5_test(test_loom_writer)
    add_hdf5_test(test_molecule_info)
    # Includes singlet/pileup/export.h, which unconditionally includes
    # h5ad_writer.h -> <hdf5.h>; needs the HDF5 include/link wiring above,
    # not the plain add_pileup_test() macro.
    add_hdf5_test(test_metrics_summary)
endif()

