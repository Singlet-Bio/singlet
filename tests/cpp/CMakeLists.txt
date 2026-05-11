# C++ unit tests — header-only pileup/pz/fq tests
# Requires: zstd, zlib, htslib, OpenMP, pthreads
# Optional: HDF5 (for h5ad/loom/molecule_info tests)

include(FindPkgConfig)
pkg_check_modules(HTSLIB REQUIRED htslib>=1.10)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(HDF5 COMPONENTS C)

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

# Cell calling / QC
add_pileup_test(test_cell_calling)
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
add_pileup_test(test_metrics_summary)
add_pileup_test(test_tpm_fpkm)
add_pileup_test(test_atac_peaks)
add_pileup_test(test_protocol_confidence_override)

# ATAC tests
add_pileup_test(test_atac_fragment)
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
add_pileup_test(test_ambient_profile)
add_pileup_test(test_read_dedup_stats)
add_pileup_test(test_donor_demux_perf)
add_pileup_test(test_wl_ambient)
add_pileup_test(test_sparse_accumulator)
add_pileup_test(test_pseudocount_sensitivity)
add_pileup_test(test_gray_zone)
add_pileup_test(test_mc_pvalue_calibration)

# Modality-specific
add_pileup_test(test_mt_events)
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
add_pileup_test(test_spatial_barcodes)
add_pileup_test(test_spatial_multiome)

# Protocol detection / barcode handling
add_pileup_test(test_combinatorial_barcode)
add_pileup_test(test_complex_whitelist)
add_pileup_test(test_indrop_detect)
add_pileup_test(test_seqwell_detect)
add_pileup_test(test_cascade_router)
add_pileup_test(test_multiome_router)
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

# HDF5-dependent tests (h5ad, loom, molecule_info)
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
endif()

