#!/bin/bash
#SBATCH --job-name=gsota-gatebs
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=18
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# PR star-bamsort-budget gate: base (prod/campaign 1b7bf90d) vs new binary, CPU engine, stream path and --sorted-bam path; outputs compared byte for byte.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
SBASE=$W/jobs/gsota-r12/bin_bsbase; NEW=$W/jobs/gsota-r12/bin_bs
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
s=(pbmc1k_v3 neuron1k_v3); s=${s[$SLURM_ARRAY_TASK_ID]}
G=$REF/GRCh38-2024-A; [ $s = neuron1k_v3 ] && G=$REF/GRCm39-2024-A
FQ=$PB/data/$s/$s.1fq; [ -e $FQ ] || { D=$W/data/w11/$s; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$s; FQ=$D/sample.1fq; }
R=$W/jobs/gsota-gatebs/$s; rm -rf $R; mkdir -p $R
echo "base $(cat $SBASE/commit) new $(cat $NEW/commit)"
for mode in stream sorted; do for v in base new; do
  O=$R/$mode-$v; mkdir -p $O; B=$SBASE; [ $v = new ] && B=$NEW
  X=""; [ $mode = sorted ] && X="--sorted-bam"
  $B/singlet-gpu $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $O/ --threads 18 --pipeline $X --output-format 1pz > $O/run.log 2>&1 || echo "EXIT $? $mode-$v"
  grep -a "unsorted STAR stream\|BAM sort\|\[star\] BAM sort" $O/run.log | head -3
done; done
for mode in stream sorted; do echo "== $s $mode"; python3 $W/jobs/gsota-det/det_compare.py $R/$mode-base $R/$mode-new 2>&1 | head -30; done
