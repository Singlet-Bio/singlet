#!/bin/bash
#SBATCH --job-name=dp-equiv
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch equiv_pair.sbatch <GSM> <binA> <binB> [tag]
# Runs binary A and binary B (names in pipeline-benchmark/bin) on the same sample, one after the other on this node,
# with --tagged-bam and SINGLET_KEEP_STAR_BAM=1, then compares every output file and both BAMs record by record
# (out_equiv.py). A side: ENV_A="VAR=x ..." / INPUTS_A="a.1fq b.1fq" (relative to the sample dir), same for B.
# Default input: <sample dir>/sample.1fq. Threads: T (default 8).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
GSM=$1; A=$2; B=$3; TAG=${4:-$A-vs-$B}
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
D=$W/data/atlas-rerun/perf/$GSM
O=$W/deep-perf/equiv/$GSM/$TAG; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-dpe-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "[equiv] $GSM A=$A B=$B node=$(hostname) $(date -Is)" | tee $O/node.txt
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
run_side() {  # side bin env inputs
  local side=$1 bin=$2 envs=$3 inputs=$4 in=()
  if [ -n "$inputs" ]; then for f in $inputs; do in+=($D/$f); done; else in=($D/sample.1fq); fi
  mkdir -p $SCR/$side/out; cd $SCR/$side
  env SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF SINGLET_KEEP_STAR_BAM=1 $envs \
    /usr/bin/time -v -o $O/time_$side.txt $PB/bin/$bin "${in[@]}" --ref-base $REF --genome-dir $G/star_2.7.11b \
    --exons $G/genes/genes.gtf --out-prefix $SCR/$side/out/ --threads ${T:-8} --pipeline --output-format 1pz \
    --bam-sort-mode disk --genome-shared off --tagged-bam "${SCREEN[@]}" > $O/run_$side.log 2>&1 || echo "exit $?" >> $O/run_$side.log
  grep -E "Elapsed|Maximum resident" $O/time_$side.txt | sed "s/^/[$side] /"
  grep -E "started sorting|finished success|STAR completed|indexed in|refresh|multi-run|Pileup:|\[singlet\] Total" $O/run_$side.log | sed "s/^/[$side] /"
  cp $SCR/$side/out/star_Log.out $O/star_Log_$side.out 2>/dev/null || true
}
run_side a $A "${ENV_A:-}" "${INPUTS_A:-}"
run_side b $B "${ENV_B:-}" "${INPUTS_B:-}"
BAMS=()
for f in star_Aligned.sortedByCoord.out.bam possorted_genome_bam.bam; do
  [ -f $SCR/a/out/$f ] && [ -f $SCR/b/out/$f ] && BAMS+=($SCR/a/out/$f $SCR/b/out/$f)
done
ls -la $SCR/a/out/*.bam $SCR/b/out/*.bam 2>/dev/null | awk '{print $5, $9}'
mkdir -p $SCR/cmp; for s in a b; do rsync -a --exclude '*.bam' --exclude '*.bai' --exclude '*Unmapped.out.mate*' \
  --exclude '*.fq' --exclude '.multi_run' $SCR/$s/out/ $SCR/cmp/$s/; done
PYTHONPATH=$W/envs/pylib-pysam $W/envs/pbench/bin/python -I ${EQUIV_PY:-$W/jobs/deep-perf/out_equiv.py} \
  $SCR/cmp/a $SCR/cmp/b ${BAMS:+--bam "${BAMS[@]}"} 2>&1 | tee $O/equiv.txt
