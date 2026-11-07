#!/bin/bash
#SBATCH --job-name=w3-time
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch w3_timing.sbatch <dataset> <reps> <bin1> [bin2 ...]
# Stage timings of several binaries on one sample, interleaved on the same node (bin1 bin2 ... per rep), in the
# campaign's compute configuration (4 threads, as w3_equiv.sbatch but without --tagged-bam). The job asks for 8 CPUs
# so that the 4-thread run has its cores to itself. Output: $W/w3-timing/<dataset>/<bin>_r<k>.json (stage_times.py).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
DS=$1; REPS=$2; shift 2
REF=/mnt/projects/debruinz_project/cellarium/reference
HERE=$W/src/singlet-py/perf__w3-speedups/bench/star_accel
case $DS in
  GSM*) IN=$W/data/atlas-rerun/perf/$DS/sample.1fq; SPECIES=GRCh38-2024-A ;;
  neuron*|nuclei*) IN=$PB/data/$DS/$DS.1fq; SPECIES=GRCm39-2024-A ;;
  *) IN=$PB/data/$DS/$DS.1fq; SPECIES=GRCh38-2024-A ;;
esac
G=$REF/$SPECIES; O=$W/w3-timing/$DS; mkdir -p $O
SCR=/tmp/$USER-w3t-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
cp $IN $SCR/s.1fq
echo "[w3t] $DS node=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p') $(date -Is)"
squeue -w $(hostname -s) -h -o "%u %C %j" | sort | uniq -c
for r in $(seq 1 $REPS); do
  for bin in "$@"; do
    rm -rf $SCR/out; mkdir -p $SCR/out; cd $SCR
    env SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF \
      /usr/bin/time -v -o $O/${bin}_r$r.time $PB/bin/$bin $SCR/s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b \
      --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads 4 --pipeline --output-format mtx --raw-matrix \
      --bam-sort-mode disk --genome-shared off > $O/${bin}_r$r.log 2>&1 || echo "exit $?" >> $O/${bin}_r$r.log
    $W/envs/pbench/bin/python $HERE/stage_times.py $O/${bin}_r$r.log $SCR/out/star_Log.out $O/${bin}_r$r.time \
      > $O/${bin}_r$r.json || true
    echo "[w3t] $bin r$r $(tr -d '\n ' < $O/${bin}_r$r.json)"
  done
done
