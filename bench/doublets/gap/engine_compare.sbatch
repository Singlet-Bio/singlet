#!/bin/bash
#SBATCH --job-name=dbl-engine
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Before/after of the doublet change, on the real engine (CPU path), one whole node: build the base (gpu/frontier tip) and the branch,
# run both on the development libraries, compare every output file, then score doublet_scores.tsv against truth.
#   BASE_REF / NEW_REF are git refs of the bare clone (fetched below). Development libraries only: gate set and Kang lanes.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; G=$W/jobs/dblfix/engine; PB=$W/pipeline-benchmark; RG4=$W/release-gate4
REFB=/mnt/projects/debruinz_project/cellarium/reference; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git
BASE_BRANCH=${BASE_BRANCH:-gpu/frontier}; NEW_BRANCH=${NEW_BRANCH:-fix/doublet-expression-score}
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR $G/runs $G/bin; trap 'rm -rf $SCR' EXIT
NC=$(nproc); echo "[eng] node=$(hostname) cpus=$NC $(date -Is)"
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
build() {  # label branch
  L=$1; BR=$2
  git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/dbleng/$L || return 1
  COMMIT=$(git -C $BARE rev-parse refs/dbleng/$L); OUTD=$G/bin/$L-${COMMIT:0:8}; BIN=$OUTD/singlet-cpu
  echo "$L $COMMIT" >> $G/bin/commits.txt
  [ -x $BIN ] && { ln -sfn $OUTD $G/bin/$L; return 0; }
  mkdir -p $SCR/src-$L; git -C $BARE archive $COMMIT | tar -x -C $SCR/src-$L; S=$SCR/src-$L
  grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
  grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
  cat > $SCR/build-$L.sh <<EOS
set -euo pipefail
export PATH=$ENV/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
cmake -S $S -B $SCR/b-$L -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_GIT_COMMIT=$COMMIT -DSINGLET_ENGINE_GPU=OFF > $SCR/c-$L.log 2>&1 || { tail -40 $SCR/c-$L.log; exit 1; }
cmake --build $SCR/b-$L --target singlet -j $NC > $SCR/b-$L.log 2>&1 || { grep -m30 -B3 -A8 "error" $SCR/b-$L.log || tail -40 $SCR/b-$L.log; exit 1; }
mkdir -p $OUTD; cp $SCR/b-$L/src/pipeline/singlet $BIN
EOS
  apptainer exec -B $W -B $SCR $IMG bash $SCR/build-$L.sh || { echo "[build] $L FAILED"; return 1; }
  echo $COMMIT > $OUTD/commit; ln -sfn $OUTD $G/bin/$L; rm -rf $S $SCR/b-$L
}
build base $BASE_BRANCH & build new $NEW_BRANCH & wait
[ -x $G/bin/base/singlet-cpu ] && [ -x $G/bin/new/singlet-cpu ] || { echo "[eng] build failed"; exit 3; }
[ -e $G/bin/whitelists ] || ln -s $RG4/bin/whitelists $G/bin/whitelists
for L in base new; do [ -e $G/bin/$L/../whitelists ] || true; done
echo "[eng] builds ok $(date -Is)"

GP=$W/modality-bench/genotype_pool
cat > $SCR/libs.txt <<EOF
kang2018_ctrl GRCh38-2024-A pair:$GP/fastq/R1.fastq.gz,$GP/fastq/R2.fastq.gz
kang2018_b1_a GRCh38-2024-A pair:$GP/kang2018_b1_a/fastq/R1.fastq.gz,$GP/kang2018_b1_a/fastq/R2.fastq.gz
kang2018_b1_b GRCh38-2024-A pair:$GP/kang2018_b1_b/fastq/R1.fastq.gz,$GP/kang2018_b1_b/fastq/R2.fastq.gz
kang2018_b1_c GRCh38-2024-A pair:$GP/kang2018_b1_c/fastq/R1.fastq.gz,$GP/kang2018_b1_c/fastq/R2.fastq.gz
pbmc5p_v1 GRCh38-2024-A fq:$PB/data/pbmc5p_v1/gex_fastqs.txt
hgmm1k_v3 GRCh38-GRCm39-barnyard fq:$PB/data/hgmm1k_v3/gex_fastqs.txt
hgmm1k_v2 GRCh38-GRCm39-barnyard fq:$RG4/ambient/data/hgmm1k_v2/gex_fastqs.txt
neuron1k_v3 GRCm39-2024-A fq:$PB/data/neuron1k_v3/gex_fastqs.txt
nuclei2k_v2 GRCm39-2024-A fq:$PB/data/nuclei2k_v2/gex_fastqs.txt
pbmc1k_v3 GRCh38-2024-A fq:$PB/data/pbmc1k_v3/gex_fastqs.txt
pbmc1k_v2 GRCh38-2024-A fq:$PB/data/pbmc1k_v2/gex_fastqs.txt
pbmc1k_cite_v3 GRCh38-2024-A fq:$PB/data/pbmc1k_cite_v3/gex_fastqs.txt
EOF
# encode once per library (the encoder is shared), then the two engines run the same .1fq
run_lib() {
  ID=$1; SP=$2; SRC=$3; S=$SCR/run-$ID; mkdir -p $S
  case $SRC in
    fq:*) L=${SRC#fq:}; cat $(grep "_R1_" $L) > $S/R1.fastq.gz; cat $(grep "_R2_" $L) > $S/R2.fastq.gz; R1=$S/R1.fastq.gz; R2=$S/R2.fastq.gz ;;
    pair:*) P=${SRC#pair:}; R1=${P%,*}; R2=${P#*,} ;;
  esac
  $G/bin/new/singlet-cpu encode --reads $R1 $R2 -o $S/s.1fq > $S/encode.log 2>&1
  rm -f $S/R1.fastq.gz $S/R2.fastq.gz
  for L in base new; do
    O=$G/runs/$L/$ID; [ -s $O/exit.txt ] && grep -q "^exit 0" $O/exit.txt && continue
    rm -rf $O; mkdir -p $O
    CMD=($G/bin/$L/singlet-cpu $S/s.1fq --ref-base $REFB --genome-dir $REFB/$SP/star_2.7.11b --exons $REFB/$SP/genes/genes.gtf --out-prefix $O/ --threads 6 --pipeline --output-format mtx --raw-matrix)
    echo "${CMD[*]}" > $O/command.txt
    (cd $S && SINGLET_STALL_TIMEOUT=10800 /usr/bin/time -v -o $O/time.txt "${CMD[@]}" > $O/run.log 2>&1); echo "exit $?" > $O/exit.txt
    grep -q "BAM: unsorted STAR stream into the pileup" $O/run.log && echo stream > $O/stream_path.txt || echo NOT-stream > $O/stream_path.txt
    echo "[run] $L $ID $(cat $O/exit.txt) $(date -Is)"
  done
  rm -rf $S
}
export -f run_lib; export G SCR REFB
# whitelists must sit next to the binary's parent (stream path); both bins live under $G/bin/<label>-<sha>, so link them there
for d in $G/bin/*-*/; do [ -e $d/../whitelists ] || true; done
xargs -P 6 -L 1 bash -c 'run_lib "$@"' _ < $SCR/libs.txt
echo "[eng] runs done $(date -Is)"

# ---- compare + score ----
$ENV/bin/python -I $G/../engine_compare.py $G $W/jobs/dblfix/data $W/jobs/gap134/amb/work $W/modality-bench/genotype_pool/truth > $G/compare.log 2>&1 || echo "[eng] compare FAILED"
tail -60 $G/compare.log
echo "[eng] end $(date -Is)"
