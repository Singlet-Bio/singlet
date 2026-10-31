#!/bin/bash
#SBATCH --job-name=cpuperf-flags
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=03:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build one compiler-flag variant of the CPU-only singlet for the flags comparison (docs/research-beta/cpu-engine-perf.md, section 7).
# Env: PIN (build this commit instead of the branch tip; keeps variants of one source in one output dir), BRANCH (branch of the bare repo to build, default claude/cpu-perf-flags), VARIANT = <march>[-lto][-starpgo|-pgo][-g] (-g: add debug info and build bin/libsamprof.so), where
#   <march> = v3 (-march=x86-64-v3) | zn3 (-march=znver3; Clipper has only Intel nodes, so zn3 builds cannot run there);  lto = -flto=auto on all host code;
#   starpgo = profile-guided STAR objects only (the recipe of src/star/pgo/pgo_build.sh);  pgo = whole-program PGO (pipeline, pileup, export, STAR).
# PGO variants build an instrumented binary, train it on the .1fq files in TRAIN (default pbmc1k_v3 neuron1k_v3), and rebuild with the profile.
# Output: $W/jobs/cpu-perf/<COMMIT8>/bin/singlet-<VARIANT> (whitelists link as in build_cpu.sbatch).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
BARE=$W/repos/singlet-py.git; BRANCH=${BRANCH:-claude/cpu-perf-flags}; VARIANT=${VARIANT:?VARIANT}; TRAIN=${TRAIN:-"pbmc1k_v3 neuron1k_v3"}
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BRANCH:refs/cpuperf/flags
COMMIT=$(git -C $BARE rev-parse ${PIN:-refs/cpuperf/flags}); OUTD=$W/jobs/cpu-perf/${COMMIT:0:8}; mkdir -p $OUTD/bin
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive $COMMIT | tar -x -C $SCR/src; S=$SCR/src
grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || \
  sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
case $VARIANT in zn3*) MARCH=-march=znver3;; *) MARCH=-march=x86-64-v3;; esac
LTO=OFF; case $VARIANT in *-lto*) LTO=ON;; esac
GF=""; case $VARIANT in *-g) GF="-DCMAKE_CXX_FLAGS=-g -DCMAKE_C_FLAGS=-g";; esac   # -g: debug info for the samprof sampling profiler (no codegen change)
MODE=none; case $VARIANT in *-starpgo*) MODE=star;; *-pgo*) MODE=whole;; esac
cat > $SCR/build.sh <<EOS
set -euo pipefail
export PATH=$ENV/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib" SINGLET_REF_BASE=$REF SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600
B=$SCR/b
cfg() { cmake -S $S -B \$B -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$COMMIT -DSINGLET_ENGINE_GPU=OFF \
        -DSINGLET_MARCH_FLAG=$MARCH -DSINGLET_LTO=$LTO $GF "\$@" > $SCR/cfg.log 2>&1 || { tail -40 $SCR/cfg.log; exit 1; }; }
bld() { cmake --build \$B --target singlet -j 24 > $SCR/bld.log 2>&1 || { grep -m30 -B3 -A8 "error" $SCR/bld.log || tail -40 $SCR/bld.log; exit 1; }; }
train() {
  for t in $TRAIN; do
    mkdir -p $SCR/t/\$t; s=\$(date +%s)
    \$B/src/pipeline/singlet $PB/data/\$t/\$t.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $SCR/t/\$t/ --threads 8 --pipeline --output-format 1pz --engine cpu > $SCR/t/\$t.log 2>&1 || { tail -30 $SCR/t/\$t.log; exit 1; }
    echo "[train] \$t \$((\$(date +%s)-s)) s"; rm -rf $SCR/t/\$t
  done
}
s_none() { cfg; bld; }
s_star_gen() { cfg "-DSINGLET_STAR_EXTRA_FLAGS=-fprofile-generate -fprofile-update=atomic" -DCMAKE_EXE_LINKER_FLAGS=-fprofile-generate; bld; }
s_star_use() { cfg "-DSINGLET_STAR_EXTRA_FLAGS=-fprofile-use -fprofile-partial-training -Wno-missing-profile -Wno-error=coverage-mismatch" -DCMAKE_EXE_LINKER_FLAGS=; bld; }
s_whole_gen() { cfg -DSINGLET_PGO=generate -DSINGLET_PGO_DIR=\$B/pgo; bld; }
s_whole_use() { echo "[pgo] \$(find \$B/pgo -name '*.gcda' | wc -l) profiles"; cfg -DSINGLET_PGO=use -DSINGLET_PGO_DIR=\$B/pgo; bld; }
final() { $ENV/bin/x86_64-conda-linux-gnu-gcc -O2 -fPIC -shared -o $OUTD/bin/libsamprof.so $S/scripts/cpu_perf/samprof/samprof.c; cp \$B/src/pipeline/singlet $OUTD/bin/singlet-$VARIANT.new && mv $OUTD/bin/singlet-$VARIANT.new $OUTD/bin/singlet-$VARIANT; }
"\$@"
EOS
inc() { apptainer exec -B /mnt/projects/debruinz_project -B $SCR $IMG bash $SCR/build.sh "$@"; }
case $MODE in
  none) inc s_none;;
  star) inc s_star_gen; bash $SCR/build.sh train; inc s_star_use;;
  whole) inc s_whole_gen; bash $SCR/build.sh train; inc s_whole_use;;
esac
inc final
ln -sfn $PB/whitelists $OUTD/whitelists; echo $COMMIT > $OUTD/commit
sha256sum $OUTD/bin/singlet-$VARIANT; echo "[build] ok $VARIANT $MARCH lto=$LTO mode=$MODE"
