#!/bin/bash
#SBATCH --job-name=txf-pair
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch txf_pair.sbatch <dataset> [reps=2] [first rep index - 1, default 0]
# Same-node A/B of SINGLET_STAR_FASTASSIGN: off/on alternated <reps> times with bin/singlet-txf (perf_run's t4
# config: 4 threads, private genome, sorted BAM, parallel pileup) on the dataset's own reference. STAR's BAM is kept
# (SINGLET_KEEP_STAR_BAM=1) for the record-level comparison (txf_equiv.py).
# Output: $PB/perf/<dataset>_txf/{off,on}_r<k>/
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
DS=$1; REPS=${2:-2}
case $DS in
  hgmm*) SPECIES=GRCh38-GRCm39-barnyard ;;
  neuron*|nuclei*) SPECIES=GRCm39-2024-A ;;
  pbmc*) SPECIES=GRCh38-2024-A ;;
  *) echo "no reference for $DS"; exit 2 ;;
esac
REF=/mnt/projects/debruinz_project/cellarium/reference/$SPECIES; GENOME=$REF/star_2.7.11b
BIN=$PB/bin/${PERF_BIN:-singlet-txf}
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
cp $PB/data/$DS/$DS.1fq $SCR/s.1fq
OUT=$PB/perf/${DS}_txf; mkdir -p $OUT
echo "[txf] $DS species=$SPECIES node=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p') bin=$(sha256sum $BIN | cut -c1-16) $(date -Is)"
export SINGLET_STALL_TIMEOUT=10800 SINGLET_KEEP_STAR_BAM=1
R0=${3:-0}
for r in $(seq $((R0+1)) $((R0+REPS))); do
  for mode in off on; do
    O=$OUT/${mode}_r$r; rm -rf $O; mkdir -p $O; cd $SCR
    if [ $mode = off ]; then export SINGLET_STAR_FASTASSIGN=0; else export SINGLET_STAR_FASTASSIGN=1; fi
    /usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
      --genome-dir $GENOME --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 4 --pipeline \
      --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
    L=$O/star_Log.out
    g=$(grep -m1 "^Finished loading the genome" $L | sed 's/^Finished loading the genome: //')
    m=$(grep -m1 "finished mapping" $L | sed 's/ \.\.\..*//')
    map=$(( $(date -d "$m" +%s) - $(date -d "$g" +%s) ))
    wall=$(grep "Elapsed" $O/time.txt | sed 's/.*: //')
    echo "[txf] $mode r$r map=${map}s wall=$wall $(grep -h FASTASSIGN $L) $(grep -h 'SEQCACHE: lookups' $L)"
  done
done
