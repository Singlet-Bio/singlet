#!/usr/bin/env bash
set -euo pipefail

# Build STAR with PGO+LTO for ~5% speedup over plain -O3 -march=native
# Requires: gcc-toolset-13, 5M-read benchmark FASTQs, genome index
# Produces: STAR_pgo_lto binary in source/

SRCDIR=/mnt/home/debruinz/Singlet-AI/STAR/source
PROFILE_DIR=/mnt/home/debruinz/Singlet-AI/STAR/pgo_profile
GENOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
WL=/mnt/home/debruinz/Singlet-AI/STAR/experiments/learned_cache/correctness_test/whitelist.txt
R1=/mnt/home/debruinz/Singlet-AI/STAR/experiments/learned_cache/bench_3way_results/sub_R1.fastq.gz
R2=/mnt/home/debruinz/Singlet-AI/STAR/experiments/learned_cache/bench_3way_results/sub_R2.fastq.gz
TRAINDIR=/dev/shm/pgo_train

source /opt/rh/gcc-toolset-13/enable

SA_FLAGS="-march=native -DSA_BOUNDARY_PREFETCH -DSA_LAZY_WINBIN"

echo "=== Phase 1: Instrumented build ==="
cd "$SRCDIR"
mkdir -p "$PROFILE_DIR"
rm -f "$PROFILE_DIR"/*.gcda
make clean
make -j8 STAR \
    CXXFLAGSextra="$SA_FLAGS -fprofile-generate=$PROFILE_DIR" \
    LDFLAGSextra="-fprofile-generate"
cp STAR STAR_pgo_generate

echo "=== Phase 2: Profile collection (5M reads) ==="
rm -rf "$TRAINDIR"; mkdir -p "$TRAINDIR"
cat "$GENOME/Genome" "$GENOME/SA" "$GENOME/SAindex" > /dev/null 2>&1
./STAR_pgo_generate \
    --runThreadN 8 --genomeDir "$GENOME" \
    --readFilesIn "$R2" "$R1" --readFilesCommand zcat \
    --soloType CB_UMI_Simple --soloCBwhitelist "$WL" \
    --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 \
    --outSAMtype BAM SortedByCoordinate \
    --outSAMattributes NH HI nM AS CR UR CB UB GX GN sS sQ sM \
    --readMapNumber 500000 --outFileNamePrefix "$TRAINDIR/"

NPROFILES=$(ls "$PROFILE_DIR"/*.gcda 2>/dev/null | wc -l)
echo "Collected $NPROFILES profile files"

echo "=== Phase 3: PGO+LTO optimized build ==="
make clean
make -j8 STAR \
    CXXFLAGSextra="$SA_FLAGS -fprofile-use=$PROFILE_DIR -fprofile-correction -Wno-missing-profile -Wno-coverage-mismatch -flto" \
    LDFLAGSextra="-fprofile-use=$PROFILE_DIR -flto -fopenmp" \
    CXXFLAGS="-pipe -w"
cp STAR STAR_pgo_lto_sa

echo "=== Done: STAR_pgo_lto_sa ready ==="
ls -la STAR_pgo_lto_sa
