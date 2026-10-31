#!/bin/bash
#SBATCH --job-name=w3-equiv
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=110G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch w3_equiv.sbatch <dataset> <binA> <binB> [tag]
# Same-node identity check of two engine binaries (names in $PB/bin) on one real sample, in the campaign's compute
# configuration (orchestrator s4 class, as deep_perf/run_deep.sbatch: --pipeline, --bam-sort-mode disk, private genome,
# production whitelists, non-host screen, mtx + 1pz outputs with the raw matrix). Both sides add --tagged-bam (the
# pinned binary has no SINGLET_KEEP_STAR_BAM), so STAR's sorted BAM survives for the record-level comparison.
# A runs first, then B. ENV_A / ENV_B: extra environment ("VAR=x ..."). Threads: T (default $SLURM_CPUS_PER_TASK).
# <dataset>: a benchmark dataset ($PB/data/<ds>/<ds>.1fq; mouse names use GRCm39) or a GSM staged under
#   $W/data/atlas-rerun/perf/<GSM>/sample.1fq (GRCh38; multi-run samples are the pooled sample.1fq).
# Output: $W/w3-equiv/<dataset>/<tag>/ {equiv.txt, stages_a.json, stages_b.json, run_*.log, time_*.txt}
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
DS=$1; A=$2; B=$3; TAG=${4:-$A-vs-$B}; T=${T:-$SLURM_CPUS_PER_TASK}
REF=/mnt/projects/debruinz_project/cellarium/reference
HERE=$W/src/singlet-py/perf__w3-speedups/bench/star_accel
case $DS in
  GSM*) IN=$W/data/atlas-rerun/perf/$DS/sample.1fq; SPECIES=GRCh38-2024-A ;;
  neuron*|nuclei*) IN=$PB/data/$DS/$DS.1fq; SPECIES=GRCm39-2024-A ;;
  hgmm*) IN=$PB/data/$DS/$DS.1fq; SPECIES=GRCh38-GRCm39-barnyard ;;
  *) IN=$PB/data/$DS/$DS.1fq; SPECIES=GRCh38-2024-A ;;
esac
G=$REF/$SPECIES
O=$W/w3-equiv/$DS/$TAG; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-w3e-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
cp $IN $SCR/s.1fq
echo "[w3] $DS species=$SPECIES T=$T A=$A ($(sha256sum $PB/bin/$A | cut -c1-16)) B=$B ($(sha256sum $PB/bin/$B | cut -c1-16))" \
  "node=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p') $(date -Is)" | tee $O/node.txt
run_side() {  # side bin env
  local side=$1 bin=$2 envs=$3
  mkdir -p $SCR/$side/out; cd $SCR/$side
  env SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF $envs \
    /usr/bin/time -v -o $O/time_$side.txt $PB/bin/$bin $SCR/s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b \
    --exons $G/genes/genes.gtf --out-prefix $SCR/$side/out/ --threads $T --pipeline --output-format mtx --raw-matrix \
    --bam-sort-mode disk --genome-shared off --tagged-bam > $O/run_$side.log 2>&1 || echo "exit $?" >> $O/run_$side.log
  cp $SCR/$side/out/star_Log.out $O/star_Log_$side.out 2>/dev/null || true
  cp $SCR/$side/out/star_Log.final.out $O/star_Log.final_$side.out 2>/dev/null || true
  $W/envs/pbench/bin/python $HERE/stage_times.py $O/run_$side.log $O/star_Log_$side.out $O/time_$side.txt \
    > $O/stages_$side.json || true
  echo "[$side] $(tr -d '\n ' < $O/stages_$side.json)"
}
run_side a $A "${ENV_A:-}"
run_side b $B "${ENV_B:-}"
BAMS=()
for f in star_Aligned.sortedByCoord.out.bam possorted_genome_bam.bam; do
  [ -f $SCR/a/out/$f ] && [ -f $SCR/b/out/$f ] && BAMS+=($SCR/a/out/$f $SCR/b/out/$f)
done
mkdir -p $SCR/cmp; for s in a b; do rsync -a --exclude '*.bam' --exclude '*.bai' --exclude '*Unmapped.out.mate*' \
  --exclude '*.fq' --exclude '*_STARtmp*' --exclude 'star_tmp*' $SCR/$s/out/ $SCR/cmp/$s/; done
# Keep both output trees (no BAMs or reads) for follow-up diffs.
rsync -a $SCR/cmp/ $O/out/
PYTHONPATH=$W/envs/pylib-pysam $W/envs/pbench/bin/python $HERE/out_equiv.py $SCR/cmp/a $SCR/cmp/b \
  ${BAMS:+--bam "${BAMS[@]}"} 2>&1 | tee $O/equiv.txt
