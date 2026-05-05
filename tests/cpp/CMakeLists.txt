# C++ unit tests — header-only pileup/pz/fq tests
# These tests require only zstd and zlib (no VDB)
# Some tests transitively include htslib headers via pileup_engine.h

include(FindPkgConfig)
pkg_check_modules(HTSLIB htslib>=1.10)

enable_testing()

set(PILEUP_TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/singlet/pileup
    ${HTSLIB_INCLUDE_DIRS}
)

set(PILEUP_TEST_LINK_LIBS ${ZSTD_LIBRARY} ZLIB::ZLIB)
if(HTSLIB_FOUND)
    list(APPEND PILEUP_TEST_LINK_LIBS ${HTSLIB_LIBRARIES})
endif()

# Helper macro for adding test executables
macro(add_pileup_test name)
    add_executable(${name} ${name}.cpp)
    target_include_directories(${name} PRIVATE ${PILEUP_TEST_INCLUDE_DIRS})
    target_link_libraries(${name} PRIVATE ${PILEUP_TEST_LINK_LIBS})
    target_compile_options(${name} PRIVATE -O2)
    if(HTSLIB_FOUND)
        target_link_directories(${name} PRIVATE ${HTSLIB_LIBRARY_DIRS})
    endif()
    add_test(NAME ${name} COMMAND ${name})
endmacro()

# Core codec tests
add_pileup_test(test_pz_reader)

# Cell calling / QC
add_pileup_test(test_cell_calling)
add_pileup_test(test_barcode_rank)
add_pileup_test(test_cell_qc)
add_pileup_test(test_tiny_dataset_guard)

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
add_pileup_test(test_tpm_fpkm)
add_pileup_test(test_atac_peaks)
add_pileup_test(test_protocol_confidence_override)

# Tests that need htslib at link time
if(HTSLIB_FOUND)
    add_pileup_test(test_metrics_summary)
endif()
