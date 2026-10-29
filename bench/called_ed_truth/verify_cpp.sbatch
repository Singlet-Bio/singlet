#!/bin/bash
#SBATCH --job-name=cv2fix
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# called_v2 2.1 in C++ (docs/called-ed.md "Fix: version 2.1") on Clipper.
#
#   MODE=build  : fetch BR (default prod/atlas-rerun) into its own worktree, build the pipeline as
#                 pipeline-benchmark/bin/singlet-prod-cv2 (x86-64-v3, as build_prod2), run test_called_v2, build cv2_stored
#   MODE=stored : array 0-5, cv2_stored on the run the truth-check script used, compared with its recommended call
#                 (state/called_v2_truth/datasets/<name>.barcodes.tsv.gz) by check_cpp.py
#   MODE=e2e    : array 0-1 (pbmc1k_v3, hgmm1k_v3), full pipeline with singlet-prod-cv2 at 8 threads, then
#                 equiv_cpp.py against python/singlet/s1/cellcall.py (2.1) and cv2_stored on the run's own stored counts
# Results: $W/state/called_v2_truth/cpp/
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark; MB=$W/modality-bench
BARE=$W/repos/singlet-py.git; BR=${BR:-prod/atlas-rerun}; SRC=$W/src/singlet-py/cv2fix__${BR//\//__}
OUT=$W/state/called_v2_truth/cpp; TRUTH=$W/state/called_v2_truth/datasets; DRV=$PB/bin/cv2_stored-2.1
REF=/mnt/projects/debruinz_project/cellarium/reference
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib SINGLET_STALL_TIMEOUT=7200
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} NUMBA_CACHE_DIR=$W/cache/numba-cv2fix
TASK=${SLURM_ARRAY_TASK_ID:-0}; MODE=${MODE:-build}
SCR=/tmp/$USER-${SLURM_JOB_ID:-x}-$TASK; mkdir -p $SCR $OUT; trap 'rm -rf $SCR' EXIT
echo "[cv2fix] mode=$MODE task=$TASK node=$(hostname) $(date -Is)"

case "$MODE" in
  build)
    git -C $BARE fetch -q --update-head-ok origin +refs/heads/$BR:refs/heads/$BR
    [ -d $SRC ] || git -C $BARE worktree add -q --detach $SRC refs/heads/$BR
    git -C $SRC checkout -q --detach refs/heads/$BR; git -C $SRC reset -q --hard
    COMMIT=$(git -C $SRC rev-parse HEAD); git -C $SRC log --oneline -1
    export CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
    export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ CMAKE_POLICY_VERSION_MINIMUM=3.5
    rsync -a --exclude .git --exclude build $SRC/ $SCR/src/
    grep -rl --include=CMakeLists.txt -e "-march=native" $SCR/src | xargs -r sed -i 's/-march=native/-march=x86-64-v3/'
    sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $SCR/src/src/star/bam_cat.c
    $CXX -std=c++17 -O2 -march=x86-64-v3 -pthread -I$SCR/src/include $SCR/src/bench/called_ed_truth/cv2_stored.cpp -o $DRV
    cmake -S $SCR/src -B $SCR/t -DSINGLET_BUILD_TESTS=ON -DSINGLET_BUILD_PIPELINE=OFF > $SCR/tc.log 2>&1
    cmake --build $SCR/t --target test_called_v2 -j 8 > $SCR/tb.log 2>&1 || { grep -m30 -A8 error $SCR/tb.log; exit 1; }
    echo "== test_called_v2"; $SCR/t/tests/cpp/test_called_v2 | grep -v "PASS" | tail -20
    cmake -S $SCR/src -B $SCR/b -DSINGLET_BUILD_PIPELINE=ON -DCMAKE_BUILD_TYPE=Release -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/c.log 2>&1 || { tail -30 $SCR/c.log; exit 1; }
    cmake --build $SCR/b --target singlet -j 16 > $SCR/b.log 2>&1 || { grep -m30 -B2 -A8 "error" $SCR/b.log; exit 1; }
    cp $SCR/b/src/pipeline/singlet $PB/bin/singlet-prod-cv2
    sha256sum $PB/bin/singlet-prod-cv2 $DRV
    echo "$COMMIT" > $OUT/build_commit.txt ;;
  stored)
    NAMES=(kang hgmm1k_v3 pbmc1k_v3 neuron1k_v3 nuclei2k_v2 pbmc5p_v1)
    N=${NAMES[$TASK]}
    RUN=$PB/runs/$N/singlet-modfix-t8; [ $N = kang ] && RUN=$MB/genotype_pool/singlet_modfix
    /usr/bin/time -f "[cv2fix] cv2_stored %e s %M KB" $DRV $RUN $OUT/$N.cpp.tsv 16 > $OUT/$N.cpp_params.json
    $W/envs/gpool/venv/bin/python -P $SRC/bench/called_ed_truth/check_cpp.py $N $OUT/$N.cpp.tsv $TRUTH/$N.barcodes.tsv.gz \
      --json $OUT/$N.check.json || echo "[cv2fix] $N NOT IDENTICAL" ;;
  e2e)
    DSS=(pbmc1k_v3 hgmm1k_v3); ID=${DSS[$TASK]}
    IFS=$'\t' read -r _ SPECIES CHEM _ _ _ < <(awk -F'\t' -v id="$ID" '$1==id' $SRC/bench/pipeline/datasets.tsv)
    GENOME=$REF/$SPECIES/star_2.7.11b; GTF=$REF/$SPECIES/genes/genes.gtf; D=$PB/data/$ID
    cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz
    cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
    BIN=$PB/bin/singlet-prod-cv2; O=$PB/runs/$ID/singlet-prod-cv2-t8; rm -rf $O; mkdir -p $O; cd $SCR
    $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/r.1fq > $O/encode.log 2>&1
    /usr/bin/time -v -o $O/time.txt $BIN $SCR/r.1fq --ref-base $REF --genome-dir $GENOME --exons $GTF \
      --out-prefix $O/ --threads 8 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "[cv2fix] run exit $?"
    grep -E "Elapsed \(wall" $O/time.txt
    grep -h "called_v2" $O/run.log | head -5
    cd $SRC
    python -P bench/called_ed/equiv_cpp.py $O --json $OUT/$ID.e2e_equiv.json \
      || echo "[cv2fix] equiv FAILED"
    $DRV $O $OUT/$ID.e2e_stored.tsv 8 > $OUT/$ID.e2e_stored_params.json
    python -P -c "
import pandas as pd
a = pd.read_csv('$O/cell_calls.tsv', sep='\t', dtype={'barcode': str}); b = pd.read_csv('$OUT/$ID.e2e_stored.tsv', sep='\t', dtype={'barcode': str})
m = a.merge(b, on='barcode', suffixes=('_pipe', '_stored'))
print('[cv2fix] $ID pipeline called_v2', int(a.called_v2.sum()), 'cv2_stored on its counts', int(b.called_v2.sum()),
      'differ', int((m.called_v2_pipe != m.called_v2_stored).sum()))
" ;;
esac
