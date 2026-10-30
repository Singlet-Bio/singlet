#!/bin/bash
#SBATCH --job-name=w11-m6-dump
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# W11 M6 truth data: run the dev-build CPU engine (tuple_dump.patch) on one real sample; the tuple file (CB index / gene / UMI
# ids, no sequences) and the CPU's final counts go to $W/jobs/w11/m6/data/<label>.{tuples,tuples.truth} (deleted when M6 is done).
#   sbatch --array=0-2%1 m6_dump.sbatch     (0 pbmc2m test, 1 GSM4006843, 2 pbmc1k_v3)
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; M=$W/jobs/w11/m6
BIN=$M/bin/singlet-m6dump; T=${T:-16}
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
TASKS=(
"pbmc2m|$PB/data/pbmc2m/pbmc2m.1fq"
"GSM4006843|$W/data/bam-stream/GSM4006843/sample.1fq"
"pbmc1k_v3|$PB/data/pbmc1k_v3/pbmc1k_v3.1fq"
)
IFS='|' read -r LABEL F <<< "${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}"
mkdir -p $M/data $M/results
SCR=/tmp/$USER-m6-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
export SINGLET_M6_TUPLES=$SCR/$LABEL.tuples
echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK label=$LABEL file=$F T=$T $(date -Is)"; sha256sum $BIN; cat $M/bin/singlet-m6dump.commit
$PB/bin/singlet-w3 decode "$F" --header 2>&1 | head -20
cd $SCR
S0=$(date +%s)
/usr/bin/time -v -o $M/results/$LABEL.time.txt $BIN $F --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
  --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off --parallel-pileup \
  > $M/results/$LABEL.run.log 2>&1 || echo "engine exit $?"
echo "engine wall $(( $(date +%s) - S0 )) s"
grep -E "m6dump|Pileup:|Directional dedup|Molecule labelling|cross-worker|umi_dedup|strand" $M/results/$LABEL.run.log | head -30
ls -la $SCR/*.tuples* && mv $SCR/$LABEL.tuples $SCR/$LABEL.tuples.truth $M/data/
# small copies of CPU-engine outputs for context (count matrices stay in the job scratch and are discarded)
ls -la $SCR/out | head -40
grep -E "Elapsed|Maximum resident" $M/results/$LABEL.time.txt
ls -la $M/data
