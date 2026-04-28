cmake_minimum_required(VERSION 3.14)
project(singlify VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG" CACHE STRING "Release C++ flags")

# ── PGO (Profile-Guided Optimization) support ──
# Two-phase build:
#   Phase 1: cmake -DSINGLIFY_PGO=generate .. && make star_pgo_trainer && run training
#   Phase 2: cmake -DSINGLIFY_PGO=use      .. && make singlify
#
# PGO is applied ONLY to star_objects (the STAR hot path).
# singlify.cpp and other targets are NOT PGO-optimized because
# mismatched profile data causes pessimization of non-trained code.
set(SINGLIFY_PGO "" CACHE STRING "PGO mode: '' (off), 'generate', or 'use'")
set(SINGLIFY_PGO_DIR "${CMAKE_BINARY_DIR}/pgo_profile" CACHE PATH "PGO profile data directory")

# PGO flags are applied per-target below (star_objects only), not globally.
# The link options still need to be global for LTO compatibility.
if(SINGLIFY_PGO STREQUAL "generate")
  message(STATUS "PGO: generating profile data → ${SINGLIFY_PGO_DIR}")
  file(MAKE_DIRECTORY ${SINGLIFY_PGO_DIR})
  add_link_options(-fprofile-generate=${SINGLIFY_PGO_DIR})
elseif(SINGLIFY_PGO STREQUAL "use")
  message(STATUS "PGO: using profile data ← ${SINGLIFY_PGO_DIR}")
  add_link_options(-fprofile-use=${SINGLIFY_PGO_DIR})
endif()

# ── Find dependencies ──
find_package(PkgConfig REQUIRED)
pkg_check_modules(HTSLIB REQUIRED htslib>=1.10)
pkg_check_modules(ZSTD REQUIRED libzstd)
pkg_check_modules(LZ4 liblz4)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenMP REQUIRED)  # STAR requires OpenMP for thread parallelism

# ── HDF5 C library (for h5ad_writer — no pkg-config available on Clipper) ──
# Use file-existence check to bypass cmake cache persistence issues.
if(EXISTS "/usr/lib64/libhdf5.so" AND EXISTS "/usr/include/hdf5.h")
  set(HDF5_C_LIB   "/usr/lib64/libhdf5.so")
  set(HDF5_C_INCLUDE "/usr/include")
  set(HDF5_AVAILABLE TRUE)
  message(STATUS "Found HDF5: ${HDF5_C_LIB}")
else()
  set(HDF5_AVAILABLE FALSE)
  message(WARNING "HDF5 not found — h5ad export support disabled")
endif()

# ── NCBI VDB library for native SRA reading ──
find_library(NCBI_VDB_LIB ncbi-vdb
  HINTS $ENV{CONDA_PREFIX}/lib ENV LD_LIBRARY_PATH)
find_path(NCBI_VDB_INCLUDE vdb/manager.h
  HINTS $ENV{CONDA_PREFIX}/include)

if(NOT NCBI_VDB_LIB OR NOT NCBI_VDB_INCLUDE)
  message(FATAL_ERROR "libncbi-vdb not found. Install sra-tools or set CONDA_PREFIX.")
endif()
message(STATUS "Found ncbi-vdb: ${NCBI_VDB_LIB}")
message(STATUS "Found ncbi-vdb headers: ${NCBI_VDB_INCLUDE}")

# ── Include directories (self-contained) ──
set(SINGLIFY_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(STAR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/star")

# ══════════════════════════════════════════════════
# STAR singlet-lite — compiled as an OBJECT library
#
# STAR source lives in src/star/ (copied from the singlet-lite branch of
# alexdobin/STAR). The library is linked into the singlify binary so that
# singlify can call star_main_impl() directly in a forked child process
# instead of exec-ing an external STAR binary.
#
# Compile flags match the STAR_production_v3 build:
#   -O3 -march=native           — parity with standalone production build
#   -DSA_BOUNDARY_PREFETCH      — singlet-lite optimization ~1.5%
#   -DSA_LAZY_WINBIN             — singlet-lite optimization ~7%
#   -fopenmp                    — STAR's internal thread management
#   -std=c++14                  — STAR uses C++11/14 features
#
# PGO+LTO is applied at the top-level singlify binary, not the object
# library, when CMake is invoked with -DSTAR_PGO=ON.
# ══════════════════════════════════════════════════

# Explicit source list derived from STAR Makefile OBJECTS.
# Using file(GLOB) is wrong here: several .cpp files in the source tree are
# code-fragment files designed to be #include-d by other translation units
# (ReadAlign_CIGAR.cpp, stitchGapIndel.cpp, Transcript_variationOutput.cpp,
# ReadAlign_outputVariation.cpp).  Compiling them standalone produces errors.
set(STAR_CPP_SOURCES
  ${STAR_SOURCE_DIR}/BAMbinSortByCoordinate.cpp
  ${STAR_SOURCE_DIR}/BAMbinSortUnmapped.cpp
  ${STAR_SOURCE_DIR}/BAMfunctions.cpp
  ${STAR_SOURCE_DIR}/BAMoutput.cpp
  ${STAR_SOURCE_DIR}/Chain.cpp
  ${STAR_SOURCE_DIR}/ChimericAlign.cpp
  ${STAR_SOURCE_DIR}/ChimericAlign_chimericBAMoutput.cpp
  ${STAR_SOURCE_DIR}/ChimericAlign_chimericJunctionOutput.cpp
  ${STAR_SOURCE_DIR}/ChimericAlign_chimericStitching.cpp
  ${STAR_SOURCE_DIR}/ChimericDetection.cpp
  ${STAR_SOURCE_DIR}/ChimericDetection_chimericDetectionMult.cpp
  ${STAR_SOURCE_DIR}/ChimericSegment.cpp
  ${STAR_SOURCE_DIR}/ClipCR4.cpp
  ${STAR_SOURCE_DIR}/ClipMate_clip.cpp
  ${STAR_SOURCE_DIR}/ClipMate_clipChunk.cpp
  ${STAR_SOURCE_DIR}/ClipMate_initialize.cpp
  ${STAR_SOURCE_DIR}/ErrorWarning.cpp
  ${STAR_SOURCE_DIR}/GTF.cpp
  ${STAR_SOURCE_DIR}/GTF_superTranscript.cpp
  ${STAR_SOURCE_DIR}/GTF_transcriptGeneSJ.cpp
  ${STAR_SOURCE_DIR}/Genome.cpp
  ${STAR_SOURCE_DIR}/Genome_genomeGenerate.cpp
  ${STAR_SOURCE_DIR}/Genome_genomeLoad.cpp
  ${STAR_SOURCE_DIR}/Genome_genomeOutLoad.cpp
  ${STAR_SOURCE_DIR}/Genome_insertSequences.cpp
  ${STAR_SOURCE_DIR}/Genome_transformGenome.cpp
  ${STAR_SOURCE_DIR}/GlobalVariables.cpp
  ${STAR_SOURCE_DIR}/InOutStreams.cpp
  ${STAR_SOURCE_DIR}/OutSJ.cpp
  ${STAR_SOURCE_DIR}/PackedArray.cpp
  ${STAR_SOURCE_DIR}/Parameters.cpp
  ${STAR_SOURCE_DIR}/ParametersChimeric_initialize.cpp
  ${STAR_SOURCE_DIR}/ParametersClip_initialize.cpp
  ${STAR_SOURCE_DIR}/ParametersGenome.cpp
  ${STAR_SOURCE_DIR}/ParametersSolo.cpp
  ${STAR_SOURCE_DIR}/Parameters_closeReadsFiles.cpp
  ${STAR_SOURCE_DIR}/Parameters_openReadsFiles.cpp
  ${STAR_SOURCE_DIR}/Parameters_readFilesInit.cpp
  ${STAR_SOURCE_DIR}/Parameters_readSAMheader.cpp
  ${STAR_SOURCE_DIR}/Parameters_samAttributes.cpp
  ${STAR_SOURCE_DIR}/Quantifications.cpp
  ${STAR_SOURCE_DIR}/ReadAlign.cpp
  ${STAR_SOURCE_DIR}/ReadAlignChunk.cpp
  ${STAR_SOURCE_DIR}/ReadAlignChunk_mapChunk.cpp
  ${STAR_SOURCE_DIR}/ReadAlignChunk_processChunks.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_alignBAM.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_assignAlignToWindow.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_calcCIGAR.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_chimericDetection.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_chimericDetectionOld.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_chimericDetectionOldOutput.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_chimericDetectionPEmerged.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_createExtendWindowsWithAlign.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_mapOneRead.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_mapOneReadSpliceGraph.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_mappedFilter.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_maxMappableLength2strands.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_multMapSelect.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_oneRead.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_outputAlignments.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_outputSpliceGraphSAM.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_outputTranscriptCIGARp.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_outputTranscriptSAM.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_outputTranscriptSJ.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_peOverlapMergeMap.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_quantTranscriptome.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_stitchPieces.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_stitchWindowSeeds.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_storeAligns.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_transformGenome.cpp
  ${STAR_SOURCE_DIR}/ReadAlign_waspMap.cpp
  ${STAR_SOURCE_DIR}/STAR.cpp
  ${STAR_SOURCE_DIR}/SequenceFuns.cpp
  ${STAR_SOURCE_DIR}/SharedMemory.cpp
  ${STAR_SOURCE_DIR}/Solo.cpp
  ${STAR_SOURCE_DIR}/SoloBarcode.cpp
  ${STAR_SOURCE_DIR}/SoloBarcode_extractBarcode.cpp
  ${STAR_SOURCE_DIR}/SoloFeature.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_addBAMtags.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_cellFiltering.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_collapseUMI_Graph.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_collapseUMIall.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_countCBgeneUMI.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_countSmartSeq.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_countVelocyto.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_emptyDrops_CR.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_loadRawMatrix.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_outputResults.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_processRecords.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_quantTranscript.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_redistributeReadsByCB.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_statsOutput.cpp
  ${STAR_SOURCE_DIR}/SoloFeature_sumThreads.cpp
  ${STAR_SOURCE_DIR}/SoloRead.cpp
  ${STAR_SOURCE_DIR}/SoloReadBarcode.cpp
  ${STAR_SOURCE_DIR}/SoloReadBarcode_getCBandUMI.cpp
  ${STAR_SOURCE_DIR}/SoloReadFeature.cpp
  ${STAR_SOURCE_DIR}/SoloReadFeature_inputRecords.cpp
  ${STAR_SOURCE_DIR}/SoloReadFeature_record.cpp
  ${STAR_SOURCE_DIR}/SoloRead_record.cpp
  ${STAR_SOURCE_DIR}/SpliceGraph.cpp
  ${STAR_SOURCE_DIR}/SpliceGraph_findSuperTr.cpp
  ${STAR_SOURCE_DIR}/SpliceGraph_swScoreSpliced.cpp
  ${STAR_SOURCE_DIR}/SpliceGraph_swTraceBack.cpp
  ${STAR_SOURCE_DIR}/Stats.cpp
  ${STAR_SOURCE_DIR}/SuffixArrayFuns.cpp
  ${STAR_SOURCE_DIR}/SuperTranscriptome.cpp
  ${STAR_SOURCE_DIR}/ThreadControl.cpp
  ${STAR_SOURCE_DIR}/TimeFunctions.cpp
  ${STAR_SOURCE_DIR}/Transcript.cpp
  ${STAR_SOURCE_DIR}/Transcript_alignScore.cpp
  ${STAR_SOURCE_DIR}/Transcript_convertGenomeCigar.cpp
  ${STAR_SOURCE_DIR}/Transcript_generateCigarP.cpp
  ${STAR_SOURCE_DIR}/Transcript_transformGenome.cpp
  ${STAR_SOURCE_DIR}/Transcript_variationAdjust.cpp
  ${STAR_SOURCE_DIR}/Transcriptome.cpp
  ${STAR_SOURCE_DIR}/Transcriptome_alignExonOverlap.cpp
  ${STAR_SOURCE_DIR}/Transcriptome_classifyAlign.cpp
  ${STAR_SOURCE_DIR}/Transcriptome_geneCountsAddAlign.cpp
  ${STAR_SOURCE_DIR}/Transcriptome_geneFullAlignOverlap.cpp
  ${STAR_SOURCE_DIR}/Transcriptome_geneFullAlignOverlap_ExonOverIntron.cpp
  ${STAR_SOURCE_DIR}/Transcriptome_quantAlign.cpp
  ${STAR_SOURCE_DIR}/Variation.cpp
  ${STAR_SOURCE_DIR}/bamRemoveDuplicates.cpp
  ${STAR_SOURCE_DIR}/bamSortByCoordinate.cpp
  ${STAR_SOURCE_DIR}/binarySearch2.cpp
  ${STAR_SOURCE_DIR}/blocksOverlap.cpp
  ${STAR_SOURCE_DIR}/extendAlign.cpp
  ${STAR_SOURCE_DIR}/funCompareUintAndSuffixes.cpp
  ${STAR_SOURCE_DIR}/funCompareUintAndSuffixesMemcmp.cpp
  ${STAR_SOURCE_DIR}/funPrimaryAlignMark.cpp
  ${STAR_SOURCE_DIR}/genomeParametersWrite.cpp
  ${STAR_SOURCE_DIR}/genomeSAindex.cpp
  ${STAR_SOURCE_DIR}/genomeScanFastaFiles.cpp
  ${STAR_SOURCE_DIR}/insertSeqSA.cpp
  ${STAR_SOURCE_DIR}/mapThreadsSpawn.cpp
  ${STAR_SOURCE_DIR}/outputSJ.cpp
  ${STAR_SOURCE_DIR}/readLoad.cpp
  ${STAR_SOURCE_DIR}/samHeaders.cpp
  ${STAR_SOURCE_DIR}/serviceFuns.cpp
  ${STAR_SOURCE_DIR}/signalFromBAM.cpp
  ${STAR_SOURCE_DIR}/sjAlignSplit.cpp
  ${STAR_SOURCE_DIR}/sjdbBuildIndex.cpp
  ${STAR_SOURCE_DIR}/sjdbInsertJunctions.cpp
  ${STAR_SOURCE_DIR}/sjdbLoadFromFiles.cpp
  ${STAR_SOURCE_DIR}/sjdbLoadFromStream.cpp
  ${STAR_SOURCE_DIR}/sjdbPrepare.cpp
  ${STAR_SOURCE_DIR}/soloInputFeatureUMI.cpp
  ${STAR_SOURCE_DIR}/stitchAlignToTranscript.cpp
  ${STAR_SOURCE_DIR}/stitchWindowAligns.cpp
  ${STAR_SOURCE_DIR}/streamFuns.cpp
  ${STAR_SOURCE_DIR}/stringSubstituteAll.cpp
  ${STAR_SOURCE_DIR}/sysRemoveDir.cpp
  ${STAR_SOURCE_DIR}/systemFunctions.cpp
  ${STAR_SOURCE_DIR}/twoPassRunPass1.cpp
)

# bam_cat has a .c extension but includes <cstring> (a C++ header) —
# STAR's Makefile compiles it with the C++ compiler.  Match that here.
set_source_files_properties(${STAR_SOURCE_DIR}/bam_cat.c PROPERTIES LANGUAGE CXX)

add_library(star_objects OBJECT
  ${STAR_CPP_SOURCES}
  ${STAR_SOURCE_DIR}/bam_cat.c
  ${STAR_SOURCE_DIR}/opal/opal.cpp
)

string(TIMESTAMP STAR_BUILD_DATE "%Y-%m-%dT%H:%M:%S")
set(STAR_COMPILE_TIME_PLACE "${STAR_BUILD_DATE} singlify-bundled")

target_compile_options(star_objects PRIVATE
  -std=c++14
  -O3
  -march=native
  -DSA_BOUNDARY_PREFETCH
  -DSA_INIT_GENOME_PREFETCH
  -DSA_LAZY_WINBIN
  # -DMINIMIZER_SA_INDEX  # S3 dead end: correctness failure (read minimizer ≠ genome minimizer)
  -fopenmp
  # Suppress STAR's routine warnings so singlify build output stays clean
  -w
)

# Apply PGO compile flags to star_objects ONLY (not singlify.cpp)
if(SINGLIFY_PGO STREQUAL "generate")
  target_compile_options(star_objects PRIVATE -fprofile-generate=${SINGLIFY_PGO_DIR})
elseif(SINGLIFY_PGO STREQUAL "use")
  target_compile_options(star_objects PRIVATE
    -fprofile-use=${SINGLIFY_PGO_DIR}
    -fprofile-correction
    -Wno-missing-profile
  )
endif()

# STAR's Makefile injects these via -D; reproduce them here
target_compile_definitions(star_objects PRIVATE
  "COMPILATION_TIME_PLACE=\"${STAR_COMPILE_TIME_PLACE}\""
  "GIT_BRANCH_COMMIT_DIFF=\"singlify-bundled\""
  STAR_VERSION="2.7.11b-singlet-lite"
)

target_include_directories(star_objects PRIVATE
  # STAR source dir: resolves #include "IncludeDefine.h" etc.
  ${STAR_SOURCE_DIR}
  # htslib bundled with STAR: STAR uses #include "htslib/htslib/sam.h"
  # The headers live at src/star/htslib/htslib/*.h wrapped in the extra dir.
  ${STAR_SOURCE_DIR}/htslib
  # opal SIMD alignment
  ${STAR_SOURCE_DIR}/opal
  # ZLIB (used by STAR for BAM)
  ${ZLIB_INCLUDE_DIRS}
  # singlify includes: minimizer_index.h used for S3 SA pre-screening
  ${SINGLIFY_INCLUDE}
)

# Propagate OpenMP include paths needed by STAR headers
target_link_libraries(star_objects PRIVATE OpenMP::OpenMP_CXX)

# ── Common link/include helpers ──
set(PILEUP_LINK_LIBS
  ${HTSLIB_LIBRARIES}
  ${ZSTD_LIBRARIES}
  ZLIB::ZLIB
  Threads::Threads
)
set(PILEUP_LINK_DIRS
  ${HTSLIB_LIBRARY_DIRS}
  ${ZSTD_LIBRARY_DIRS}
)
set(PILEUP_INCLUDE_DIRS
  ${SINGLIFY_INCLUDE}
  ${HTSLIB_INCLUDE_DIRS}
  ${ZSTD_INCLUDE_DIRS}
)

# ══════════════════════════════════════════════════
# singlify — unified SRA→.1fq→STAR→pileup→.1pz binary
#
# Includes the STAR singlet-lite aligner as a compiled-in object library.
# The singlify binary is completely self-contained: no external STAR
# executable required.  singlify forks a child that calls star_main_impl()
# directly, writes BAM to a pipe, and the parent runs the singlet-pileup
# engine on the stream, producing .1pz output.
# ══════════════════════════════════════════════════
add_executable(singlify src/singlify.cpp)

target_include_directories(singlify PRIVATE
  ${PILEUP_INCLUDE_DIRS}
  ${NCBI_VDB_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  # singlify/include/star_api.h
  ${SINGLIFY_INCLUDE}
)

target_link_libraries(singlify PRIVATE
  $<TARGET_OBJECTS:star_objects>  # STAR singlet-lite compiled in
  ${PILEUP_LINK_LIBS}
  ${NCBI_VDB_LIB}
  OpenMP::OpenMP_CXX
  ZLIB::ZLIB
)

if(HDF5_AVAILABLE)
  target_include_directories(singlify PRIVATE ${HDF5_C_INCLUDE})
  target_link_libraries(singlify PRIVATE ${HDF5_C_LIB})
endif()

target_link_directories(singlify PRIVATE ${PILEUP_LINK_DIRS})

if(LZ4_FOUND)
  target_compile_definitions(singlify PRIVATE HAS_LZ4)
  target_link_libraries(singlify PRIVATE ${LZ4_LIBRARIES})
endif()

install(TARGETS singlify RUNTIME DESTINATION bin)

# ══════════════════════════════════════════════════
# pileup — standalone BAM pileup CLI (debug/interop only)
#
# The pileup engine is designed to run integrated with the aligner inside
# singlify, not as a standalone tool.  This target exists only for
# development debugging with pre-existing BAM files.  It is NOT installed.
# ══════════════════════════════════════════════════
add_executable(pileup EXCLUDE_FROM_ALL src/pileup.cpp)

target_include_directories(pileup PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(pileup PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(pileup PRIVATE ${PILEUP_LINK_DIRS})

if(LZ4_FOUND)
  target_compile_definitions(pileup PRIVATE HAS_LZ4)
  target_link_libraries(pileup PRIVATE ${LZ4_LIBRARIES})
endif()

# Not installed — use singlify for production

# ══════════════════════════════════════════════════
# 1fq — .1fq format CLI tool (encode/decode/inspect)
# ══════════════════════════════════════════════════
add_executable(1fq src/1fq.cpp)

target_include_directories(1fq PRIVATE
  ${SINGLIFY_INCLUDE}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${ZSTD_INCLUDE_DIRS}
  ${NCBI_VDB_INCLUDE}
)

target_link_libraries(1fq PRIVATE
  ${ZSTD_LIBRARIES}
  ${NCBI_VDB_LIB}
  Threads::Threads
)

if(LZ4_FOUND)
  target_compile_definitions(1fq PRIVATE HAS_LZ4)
  target_include_directories(1fq PRIVATE ${LZ4_INCLUDE_DIRS})
  target_link_libraries(1fq PRIVATE ${LZ4_LIBRARIES})
endif()

target_link_directories(1fq PRIVATE ${ZSTD_LIBRARY_DIRS})

install(TARGETS 1fq RUNTIME DESTINATION bin)

# ══════════════════════════════════════════════════
# star_pgo_trainer — standalone STAR binary for PGO training
#
# The integrated singlify binary uses fork()+star_main_impl() which crashes
# under -fprofile-generate (double-free from forked OpenMP + gcov counters).
# This standalone binary calls star_main_impl() directly in main(), letting
# GCC collect clean profile data for all STAR hot paths.
#
# PGO workflow:
#   1. cmake -DSINGLIFY_PGO=generate .. && make star_pgo_trainer
#   2. ./star_pgo_trainer <STAR args> --readMapNumber 5000000
#   3. cmake -DSINGLIFY_PGO=use .. && make singlify
# ══════════════════════════════════════════════════
add_executable(star_pgo_trainer EXCLUDE_FROM_ALL src/star_pgo_trainer.cpp)

target_include_directories(star_pgo_trainer PRIVATE
  ${SINGLIFY_INCLUDE}
  ${STAR_SOURCE_DIR}
  ${STAR_SOURCE_DIR}/htslib
  ${STAR_SOURCE_DIR}/opal
  ${ZLIB_INCLUDE_DIRS}
)

target_link_libraries(star_pgo_trainer PRIVATE
  $<TARGET_OBJECTS:star_objects>
  ${HTSLIB_LIBRARIES}
  OpenMP::OpenMP_CXX
  ZLIB::ZLIB
  Threads::Threads
)

target_link_directories(star_pgo_trainer PRIVATE
  ${HTSLIB_LIBRARY_DIRS}
)

# ══════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════
enable_testing()

# lib1fq tests
add_executable(test_lib1fq src/test_lib1fq.cpp)
target_include_directories(test_lib1fq PRIVATE ${SINGLIFY_INCLUDE} ${ZSTD_INCLUDE_DIRS})
target_link_libraries(test_lib1fq PRIVATE ${ZSTD_LIBRARIES})
if(LZ4_FOUND)
  target_compile_definitions(test_lib1fq PRIVATE HAS_LZ4)
  target_include_directories(test_lib1fq PRIVATE ${LZ4_INCLUDE_DIRS})
  target_link_libraries(test_lib1fq PRIVATE ${LZ4_LIBRARIES})
endif()
target_link_directories(test_lib1fq PRIVATE ${ZSTD_LIBRARY_DIRS})
add_test(NAME lib1fq COMMAND test_lib1fq)

# pileup engine tests
add_executable(test_interval_tree test/test_interval_tree.cpp)
target_include_directories(test_interval_tree PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME interval_tree COMMAND test_interval_tree)

add_executable(test_sparse_accumulator test/test_sparse_accumulator.cpp)
target_include_directories(test_sparse_accumulator PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME sparse_accumulator COMMAND test_sparse_accumulator)

# N18: CRISPR guide capture unit tests
add_executable(test_crispr_guide test/test_crispr_guide.cpp)
target_include_directories(test_crispr_guide PRIVATE ${SINGLIFY_INCLUDE})
# The PGO profile was generated for singlify with its specific call patterns.
# Overriding with an empty path + correction flag prevents mismatched-counter
# crashes when SparseAccumulator<uint16_t>::to_csc() is profiled for 60K×5K
# matrices but called with 2×3 in the unit test.
if(SINGLIFY_PGO STREQUAL "use")
  target_link_options(test_crispr_guide PRIVATE
    "-fprofile-use=/dev/null" -fprofile-correction -Wno-missing-profile)
endif()
add_test(NAME crispr_guide COMMAND test_crispr_guide)

# T1: ADT antibody barcode matcher unit tests
add_executable(test_adt_matcher test/test_adt_matcher.cpp)
target_include_directories(test_adt_matcher PRIVATE ${SINGLIFY_INCLUDE})
if(SINGLIFY_PGO STREQUAL "use")
  target_link_options(test_adt_matcher PRIVATE
    "-fprofile-use=/dev/null" -fprofile-correction -Wno-missing-profile)
endif()
add_test(NAME adt_matcher COMMAND test_adt_matcher)

add_executable(test_pileup_integration test/test_pileup_integration.cpp)
target_include_directories(test_pileup_integration PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_pileup_integration PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_pileup_integration PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME pileup_integration COMMAND test_pileup_integration)
set_tests_properties(pileup_integration PROPERTIES
    ENVIRONMENT "PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$ENV{PATH}")

add_executable(test_species_detect test/test_species_detect.cpp)
target_include_directories(test_species_detect PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_species_detect PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_species_detect PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME species_detect COMMAND test_species_detect)

add_executable(test_cell_calling test/test_cell_calling.cpp)
target_include_directories(test_cell_calling PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME cell_calling COMMAND test_cell_calling)

# N22: full-whitelist ambient profiling for EmptyDrops
add_executable(test_wl_ambient test/test_wl_ambient.cpp)
target_include_directories(test_wl_ambient PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME wl_ambient COMMAND test_wl_ambient)

# N14: sex calling
add_executable(test_sex_calling test/test_sex_calling.cpp)
target_include_directories(test_sex_calling PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_sex_calling PRIVATE ${PILEUP_LINK_LIBS} z)
target_link_directories(test_sex_calling PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME sex_calling COMMAND test_sex_calling)

add_executable(test_atac_fragment test/test_atac_fragment.cpp)
target_include_directories(test_atac_fragment PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_atac_fragment PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_atac_fragment PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME atac_fragment COMMAND test_atac_fragment)

add_executable(test_atac_bin_counter test/test_atac_bin_counter.cpp)
target_include_directories(test_atac_bin_counter PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_atac_bin_counter PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_atac_bin_counter PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME atac_bin_counter COMMAND test_atac_bin_counter)

add_executable(test_atac_qc test/test_atac_qc.cpp)
target_include_directories(test_atac_qc PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_atac_qc PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_atac_qc PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME atac_qc COMMAND test_atac_qc)

# T2: ADT UMI dedup + counting
add_executable(test_adt_counter test/test_adt_counter.cpp)
target_include_directories(test_adt_counter PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME adt_counter COMMAND test_adt_counter)

# T3: HTO demultiplexing
add_executable(test_hto_demux test/test_hto_demux.cpp)
target_include_directories(test_hto_demux PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME hto_demux COMMAND test_hto_demux)

# V1: Visium spatial coordinate parser
add_executable(test_visium_spatial test/test_visium_spatial.cpp)
target_include_directories(test_visium_spatial PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME visium_spatial COMMAND test_visium_spatial)

# V2+V3: Visium per-spot pileup integration + spatial QC metrics
add_executable(test_visium_qc test/test_visium_qc.cpp)
target_include_directories(test_visium_qc PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME visium_qc COMMAND test_visium_qc)

# SS2+B2: Alignment QC — gene body coverage, 5'/3' ratio, Lander-Waterman
add_executable(test_alignment_qc test/test_alignment_qc.cpp)
target_include_directories(test_alignment_qc PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME alignment_qc COMMAND test_alignment_qc)

# SS3+B3: RNA variant caller — de novo SNV discovery from aligned BAM
add_executable(test_rna_variant_caller test/test_rna_variant_caller.cpp)
target_include_directories(test_rna_variant_caller PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_rna_variant_caller PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_rna_variant_caller PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME rna_variant_caller COMMAND test_rna_variant_caller)

# B4: Read-level dedup stats — PCR + optical duplicate marking, library complexity
add_executable(test_read_dedup_stats test/test_read_dedup_stats.cpp)
target_include_directories(test_read_dedup_stats PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_read_dedup_stats PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_read_dedup_stats PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME read_dedup_stats COMMAND test_read_dedup_stats)

# A4: ATAC donor demux — fragment-based SNP genotyping
add_executable(test_atac_donor_demux test/test_atac_donor_demux.cpp)
target_include_directories(test_atac_donor_demux PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME atac_donor_demux COMMAND test_atac_donor_demux)

# A7: ATAC cell calling — TSS enrichment + fragment count + FRIP thresholding
add_executable(test_atac_cell_caller test/test_atac_cell_caller.cpp)
target_include_directories(test_atac_cell_caller PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME atac_cell_caller COMMAND test_atac_cell_caller)

# A5: ATAC ancestry + sex calling
add_executable(test_atac_ancestry_sex test/test_atac_ancestry_sex.cpp)
target_include_directories(test_atac_ancestry_sex PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME atac_ancestry_sex COMMAND test_atac_ancestry_sex)

# G-EXPORT: CellRanger-compatible MTX export
add_executable(test_mtx_export test/test_mtx_export.cpp)
target_include_directories(test_mtx_export PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_mtx_export PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_mtx_export PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME mtx_export COMMAND test_mtx_export)

# G-SATCURVE: Sequencing saturation downsampling curves
add_executable(test_saturation_curve test/test_saturation_curve.cpp)
target_include_directories(test_saturation_curve PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_saturation_curve PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_saturation_curve PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME saturation_curve COMMAND test_saturation_curve)

# G-PSI: Per-cell splice junction PSI
add_executable(test_splice_psi test/test_splice_psi.cpp)
target_include_directories(test_splice_psi PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME splice_psi COMMAND test_splice_psi)

# G-QC: Per-cell QC metrics (MT%, ribo%, intronic fraction)
add_executable(test_cell_qc test/test_cell_qc.cpp)
target_include_directories(test_cell_qc PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_cell_qc PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_cell_qc PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME cell_qc COMMAND test_cell_qc)

# G-EM: Equivalence-class EM multi-mapper rescue
add_executable(test_em_rescue test/test_em_rescue.cpp)
target_include_directories(test_em_rescue PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME em_rescue COMMAND test_em_rescue)

# G-METRICS + G-PERMITLIST: metrics_summary.csv and forced/expect-cells modes
add_executable(test_metrics_summary test/test_metrics_summary.cpp)
target_include_directories(test_metrics_summary PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_metrics_summary PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_metrics_summary PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME metrics_summary COMMAND test_metrics_summary)

# G-BARNYARD: Per-cell species classification from dual-genome
add_executable(test_barnyard test/test_barnyard.cpp)
target_include_directories(test_barnyard PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME barnyard COMMAND test_barnyard)

# G-CELLPLEX: 10x CellPlex / CMO multiplexing demultiplex
add_executable(test_cellplex test/test_cellplex.cpp)
target_include_directories(test_cellplex PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME cellplex COMMAND test_cellplex)

# G-TXLEVEL: Transcript-level equivalence class counting (TCC)
add_executable(test_transcript_compat test/test_transcript_compat.cpp)
target_include_directories(test_transcript_compat PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_transcript_compat PRIVATE z)
add_test(NAME transcript_compat COMMAND test_transcript_compat)

# G-ATAC-PEAKS: Native ATAC peak calling (Poisson enrichment)
add_executable(test_atac_peaks test/test_atac_peaks.cpp)
target_include_directories(test_atac_peaks PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME atac_peaks COMMAND test_atac_peaks)

# G-LOGGING: Structured multi-level logging with JSON Lines output
add_executable(test_logging test/test_logging.cpp)
target_include_directories(test_logging PRIVATE ${SINGLIFY_INCLUDE} include)
target_link_libraries(test_logging PRIVATE pthread)
add_test(NAME logging COMMAND test_logging)

# G-MULTIOME-ROUTE: 10x Multiome auto-detect + split/route
add_executable(test_multiome_router test/test_multiome_router.cpp)
target_include_directories(test_multiome_router PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME multiome_router COMMAND test_multiome_router)

# G-BARNYARD-ROUTE: Per-cell species routing for barnyard mixture experiments
add_executable(test_barnyard_route test/test_barnyard_route.cpp)
target_include_directories(test_barnyard_route PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME barnyard_route COMMAND test_barnyard_route)

# G-SPATIAL-MULTIOME: Spatial multiome router (Visium HD + spatial RNA/ATAC)
add_executable(test_spatial_multiome test/test_spatial_multiome.cpp)
target_include_directories(test_spatial_multiome PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME spatial_multiome COMMAND test_spatial_multiome)

# G-VELOCITY: Spliced/unspliced/ambiguous per-gene velocity matrices
add_executable(test_velocity_export test/test_velocity_export.cpp)
target_include_directories(test_velocity_export PRIVATE ${SINGLIFY_INCLUDE})
if(SINGLIFY_PGO STREQUAL "use")
  target_link_options(test_velocity_export PRIVATE
    "-fprofile-use=/dev/null" -fprofile-correction -Wno-missing-profile)
endif()
add_test(NAME velocity_export COMMAND test_velocity_export)

# G-VDJ: V(D)J CDR3 clonotype assembly
add_executable(test_vdj_assembly test/test_vdj_assembly.cpp)
target_include_directories(test_vdj_assembly PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME vdj_assembly COMMAND test_vdj_assembly)

# G-SLAMSEQ: SLAM-seq / TT-seq / scEU-seq metabolic labeling T→C conversion counting
add_executable(test_slamseq test/test_slamseq.cpp)
target_include_directories(test_slamseq PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME slamseq COMMAND test_slamseq)

# G-SHARESEQ: SHARE-seq / PAIRED-seq / scifi-RNA-seq combinatorial barcode decoder
add_executable(test_combinatorial_barcode test/test_combinatorial_barcode.cpp)
target_include_directories(test_combinatorial_barcode PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME combinatorial_barcode COMMAND test_combinatorial_barcode)

# G-TPM: TPM and FPKM normalized gene expression output
add_executable(test_tpm_fpkm test/test_tpm_fpkm.cpp)
target_include_directories(test_tpm_fpkm PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_tpm_fpkm PRIVATE z)
add_test(NAME tpm_fpkm COMMAND test_tpm_fpkm)

# G-BULK-ATAC: Bulk ATAC-seq auto-detection and QC
add_executable(test_bulk_atac test/test_bulk_atac.cpp)
target_include_directories(test_bulk_atac PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_bulk_atac PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_bulk_atac PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME bulk_atac COMMAND test_bulk_atac)

add_executable(test_cuttag test/test_cuttag.cpp)
target_include_directories(test_cuttag PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_cuttag PRIVATE ${PILEUP_LINK_LIBS})
target_link_directories(test_cuttag PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME cuttag COMMAND test_cuttag)

# G-CHIPSEQ: Bulk ChIP-seq QC metrics (NSC/RSC, PBC1/PBC2, Lorenz curve)
add_executable(test_chipseq test/test_chipseq.cpp)
target_include_directories(test_chipseq PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME chipseq COMMAND test_chipseq)

# G-SCDNA: Single-cell DNA CNV bin counting and QC (MAPD, ploidy, GC correction)
add_executable(test_scdna_cnv test/test_scdna_cnv.cpp)
target_include_directories(test_scdna_cnv PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME scdna_cnv COMMAND test_scdna_cnv)

# G-LONGREAD: Long-read scRNA-seq (PacBio MAS-seq / ONT) support
add_executable(test_longread test/test_longread.cpp)
target_include_directories(test_longread PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME longread COMMAND test_longread)

# G-HIC: Hi-C / Micro-C contact pair extraction and QC
add_executable(test_hic_contacts test/test_hic_contacts.cpp)
target_include_directories(test_hic_contacts PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME hic_contacts COMMAND test_hic_contacts)

# G-SPATIAL-BARCODES: Spatial transcriptomics barcode decoders
add_executable(test_spatial_barcodes test/test_spatial_barcodes.cpp)
target_include_directories(test_spatial_barcodes PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME spatial_barcodes COMMAND test_spatial_barcodes)

# G-MIRNA: miRNA quantification from small RNA-seq data
add_executable(test_mirna test/test_mirna.cpp)
target_include_directories(test_mirna PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME mirna COMMAND test_mirna)

# G-RRNA: rRNA contamination detection via hard-coded diagnostic k-mers
add_executable(test_rrna_detect test/test_rrna_detect.cpp)
target_include_directories(test_rrna_detect PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME rrna_detect COMMAND test_rrna_detect)

# G-METHYL: bisulfite/methylation analysis (WGBS, RRBS, scBS, NOMe-seq)
add_executable(test_methyl test/test_methyl.cpp)
target_include_directories(test_methyl PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME methyl COMMAND test_methyl)

# G-WGS: WGS/de novo assembly QC, coverage analysis, and assembler subprocess management
add_executable(test_wgs_assembly test/test_wgs_assembly.cpp)
target_include_directories(test_wgs_assembly PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME wgs_assembly COMMAND test_wgs_assembly)

# SPECIES-KMER-DB: Compile-time species registry (37 species, Ensembl URLs, lookup helpers)
add_executable(test_species_registry test/test_species_registry.cpp)
target_include_directories(test_species_registry PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME species_registry COMMAND test_species_registry)

# REF-FETCH-CMD: Reference download and STAR build plan builder
add_executable(test_ref_fetch test/test_ref_fetch.cpp)
target_include_directories(test_ref_fetch PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME ref_fetch COMMAND test_ref_fetch)

# G-INTEROP: Interoperability compliance validators (MTX, fragments, pairs, BED, CSV, VCF)
add_executable(test_interop test/test_interop.cpp)
target_include_directories(test_interop PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME interop COMMAND test_interop)

# VAL1-INDROP-VARIANT: InDrop version detection and layout (fixes SRR5945694 cblen/umilen overflow)
add_executable(test_indrop_detect test/test_indrop_detect.cpp)
target_include_directories(test_indrop_detect PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME indrop_detect COMMAND test_indrop_detect)

# VAL1-SEQWELL-DETECT: Seq-Well protocol tag detection and CBlen validation (fixes SRR32182924)
add_executable(test_seqwell_detect test/test_seqwell_detect.cpp)
target_include_directories(test_seqwell_detect PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME seqwell_detect COMMAND test_seqwell_detect)

# G-BENCHMARK-SUITE: Per-modality benchmark registry (speed + accuracy vs SOTA tools)
add_executable(test_benchmark_suite test/test_benchmark_suite.cpp)
target_include_directories(test_benchmark_suite PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME benchmark_suite COMMAND test_benchmark_suite)

# VAL1-TINY-CRASH: Guard against STAR crashes on tiny datasets (e.g. SRR23027738, 110K reads)
add_executable(test_tiny_dataset_guard test/test_tiny_dataset_guard.cpp)
target_include_directories(test_tiny_dataset_guard PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME tiny_dataset_guard COMMAND test_tiny_dataset_guard)

# AUTOFIX-MEGA-SORT-RSS-OVERAGE: STAR BAM-sort tier selection for 500M+ read samples
add_executable(test_mega_sort_params test/test_mega_sort_params.cpp)
target_include_directories(test_mega_sort_params PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME mega_sort_params COMMAND test_mega_sort_params)

# G-WEBSUMMARY: Self-contained HTML report generator (Cell Ranger-style web_summary.html)
add_executable(test_web_summary test/test_web_summary.cpp)
target_include_directories(test_web_summary PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME web_summary COMMAND test_web_summary)

# G-RAWMATRIX + G-SUMMARYJSON: raw matrix export and structured summary JSON
add_executable(test_summary_json test/test_summary_json.cpp)
target_include_directories(test_summary_json PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME summary_json COMMAND test_summary_json)


# G-TAGGED-BAM: Cell Ranger-compatible BAM output with CB/UB/GX/GN tags
add_executable(test_tagged_bam test/test_tagged_bam.cpp)
target_include_directories(test_tagged_bam PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_tagged_bam PRIVATE ${PILEUP_LINK_LIBS} ${CMAKE_DL_LIBS} Threads::Threads)
target_link_directories(test_tagged_bam PRIVATE ${PILEUP_LINK_DIRS})
if(SINGLIFY_PGO STREQUAL "use")
  target_link_options(test_tagged_bam PRIVATE
    "-fprofile-use=/dev/null" -fprofile-correction -Wno-missing-profile)
endif()
add_test(NAME tagged_bam COMMAND test_tagged_bam)

# G-H5AD: AnnData .h5ad export
if(HDF5_AVAILABLE)
  add_executable(test_h5ad_writer test/test_h5ad_writer.cpp)
  target_include_directories(test_h5ad_writer PRIVATE ${SINGLIFY_INCLUDE} ${HDF5_C_INCLUDE})
  target_link_libraries(test_h5ad_writer PRIVATE ${HDF5_C_LIB} ${CMAKE_DL_LIBS})
  add_test(NAME h5ad_writer COMMAND test_h5ad_writer)
endif()

# G-LOOM: Loom v3 format export (velocyto/scVelo compatible)
if(HDF5_AVAILABLE)
  add_executable(test_loom_writer test/test_loom_writer.cpp)
  target_include_directories(test_loom_writer PRIVATE ${SINGLIFY_INCLUDE} ${HDF5_C_INCLUDE})
  target_link_libraries(test_loom_writer PRIVATE ${HDF5_C_LIB} ${CMAKE_DL_LIBS})
  add_test(NAME loom_writer COMMAND test_loom_writer)
endif()

# G-BUS: BUS format (kallisto/bustools compatible) — no external dependencies
add_executable(test_bus_writer test/test_bus_writer.cpp)
target_include_directories(test_bus_writer PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME bus_writer COMMAND test_bus_writer)

# G-MOLINFO: Molecule info HDF5 (Cell Ranger compatible)
if(HDF5_AVAILABLE)
  add_executable(test_molecule_info test/test_molecule_info.cpp)
  target_include_directories(test_molecule_info PRIVATE ${SINGLIFY_INCLUDE} ${HDF5_C_INCLUDE})
  target_link_libraries(test_molecule_info PRIVATE ${HDF5_C_LIB} ${CMAKE_DL_LIBS})
  add_test(NAME molecule_info COMMAND test_molecule_info)
endif()

# G-FRAGMENTS: Standard fragments.tsv.gz writer (10x/ArchR/Signac compatible)
add_executable(test_fragments_writer test/test_fragments_writer.cpp)
target_include_directories(test_fragments_writer PRIVATE ${PILEUP_INCLUDE_DIRS})
target_link_libraries(test_fragments_writer PRIVATE ${PILEUP_LINK_LIBS} ${CMAKE_DL_LIBS} Threads::Threads)
target_link_directories(test_fragments_writer PRIVATE ${PILEUP_LINK_DIRS})
add_test(NAME fragments_writer COMMAND test_fragments_writer)

# G-BARCODE-METRICS: Per-barcode metrics TSV
add_executable(test_barcode_metrics test/test_barcode_metrics.cpp)
target_include_directories(test_barcode_metrics PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME barcode_metrics COMMAND test_barcode_metrics)

# G-AMBIENT: Ambient RNA profile export (SoupX/CellBender compatible)
add_executable(test_ambient_profile test/test_ambient_profile.cpp)
target_include_directories(test_ambient_profile PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME ambient_profile COMMAND test_ambient_profile)

# G-DOUBLET-EXPORT: Per-cell doublet score TSV export
add_executable(test_doublet_scores test/test_doublet_scores.cpp)
target_include_directories(test_doublet_scores PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME doublet_scores COMMAND test_doublet_scores)

# G-BARCODE-RANK: Barcode rank table for knee plot generation
add_executable(test_barcode_rank test/test_barcode_rank.cpp)
target_include_directories(test_barcode_rank PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME barcode_rank COMMAND test_barcode_rank)

# AUTOFIX-DEMUX-K-SWEEP-SLOW: donor demux performance + correctness
add_executable(test_donor_demux_perf test/test_donor_demux_perf.cpp)
target_include_directories(test_donor_demux_perf PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_donor_demux_perf PRIVATE Threads::Threads)
add_test(NAME donor_demux_perf COMMAND test_donor_demux_perf)

# AUTOFIX-PROTOCOL-CONFIDENCE-OVERRIDE: low-confidence detection overridden by catalog metadata
add_executable(test_protocol_confidence_override test/test_protocol_confidence_override.cpp)
target_include_directories(test_protocol_confidence_override PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME protocol_confidence_override COMMAND test_protocol_confidence_override)

# AUTOFIX-BD-RHAPSODY-WHITELIST: per-segment whitelist discovery for CB_UMI_Complex protocols
add_executable(test_complex_whitelist test/test_complex_whitelist.cpp)
target_include_directories(test_complex_whitelist PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME complex_whitelist COMMAND test_complex_whitelist)

# NONHOST-SYLPH-PORT: MinSketch index + NonHostScreener for viral/bacterial/fungal screening
add_executable(test_nonhost test/test_nonhost.cpp)
target_include_directories(test_nonhost PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_nonhost PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME nonhost COMMAND test_nonhost)

# ── NONHOST-VIRALDB / NONHOST-MICROBIALDB: build-nonhost-db workflow tests ──
add_executable(test_nonhost_db_build test/test_nonhost_db_build.cpp)
target_include_directories(test_nonhost_db_build PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_nonhost_db_build PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME nonhost_db_build COMMAND test_nonhost_db_build)

# ── NONHOST-UNMAPPED-CAPTURE: classify pipeline tests ──
add_executable(test_nonhost_unmapped_capture test/test_nonhost_unmapped_capture.cpp)
target_include_directories(test_nonhost_unmapped_capture PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_nonhost_unmapped_capture PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME nonhost_unmapped_capture COMMAND test_nonhost_unmapped_capture)

# ── NONHOST-EM: EM abundance deconvolution tests ──────────────────────────────
add_executable(test_nonhost_em test/test_nonhost_em.cpp)
target_include_directories(test_nonhost_em PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_nonhost_em PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME nonhost_em COMMAND test_nonhost_em)

# -- NONHOST-HOST-SUBTRACT: host k-mer Bloom filter tests ---------------------
add_executable(test_nonhost_host_filter test/test_nonhost_host_filter.cpp)
target_include_directories(test_nonhost_host_filter PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_nonhost_host_filter PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME nonhost_host_filter COMMAND test_nonhost_host_filter)

# -- NONHOST-SECONDARY-ALIGN: seed-chain-extend aligner tests ----------------
add_executable(test_nonhost_aligner test/test_nonhost_aligner.cpp)
target_include_directories(test_nonhost_aligner PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_nonhost_aligner PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME nonhost_aligner COMMAND test_nonhost_aligner)

# -- NONHOST-CB-SPECIES-MATRIX: per-cell non-host species count matrix tests --
add_executable(test_nonhost_cell_matrix test/test_nonhost_cell_matrix.cpp)
target_include_directories(test_nonhost_cell_matrix PRIVATE ${SINGLIFY_INCLUDE})
target_link_libraries(test_nonhost_cell_matrix PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME nonhost_cell_matrix COMMAND test_nonhost_cell_matrix)

# -- DOUBLET-DETECT-V2: Scrublet-style simulation kNN doublet detection tests --
add_executable(test_doublet_detect_v2 test/test_doublet_detect_v2.cpp)
target_include_directories(test_doublet_detect_v2 PRIVATE ${SINGLIFY_INCLUDE})
add_test(NAME doublet_detect_v2 COMMAND test_doublet_detect_v2)
