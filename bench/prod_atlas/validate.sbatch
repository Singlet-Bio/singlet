#!/bin/bash
#SBATCH --job-name=ar-valprod
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --array=0-4%3
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Validate bin/singlet-prod (prod/atlas-rerun) against bin/singlet-fixed-det (#22) on the benchmark datasets:
#   * gene matrices identical at 4 and 8 threads (except where #25 changes strand handling), prod 4 vs 8 identical;
#   * pbmc5p_v1 (5') counted reverse-strand without --reverse-strand;
#   * called_v2 from the binary == python/singlet/s1/cellcall.py on the binary's stored counts (bench/called_ed/equiv_cpp.py);
#   * wall time per run (time.txt).
# Output: $PB/runs/<dataset>/singlet-{prod,det}-t{4,8}/ and $PB/results/prod_atlas/<dataset>.*.json
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; ENV=$W/envs/pbench
SRC=$W/src/singlet-py/prod__atlas-rerun
REF=/mnt/projects/debruinz_project/cellarium/reference
export PATH=$ENV/bin:$PATH SINGLET_STALL_TIMEOUT=7200
# scripts come from the worktree as build.sbatch (or sync.sbatch) left it; the binary is bin/singlet-prod
DSS=(pbmc1k_v3 pbmc5p_v1 nuclei2k_v2 neuron1k_v3 hgmm1k_v3)
ID=${1:-${DSS[$SLURM_ARRAY_TASK_ID]}}
IFS=$'\t' read -r _ SPECIES CHEM _ _ _ < <(awk -F'\t' -v id="$ID" '$1==id' $SRC/bench/pipeline/datasets.tsv)
GENOME=$REF/$SPECIES/star_2.7.11b; GTF=$REF/$SPECIES/genes/genes.gtf; D=$PB/data/$ID
RES=$PB/results/prod_atlas; mkdir -p $RES
SCR=/tmp/$USER-$SLURM_JOB_ID-${SLURM_ARRAY_TASK_ID:-0}; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "[val] $ID species=$SPECIES chem=$CHEM node=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p') $(date -Is)"
cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz
cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
for B in det prod; do
  BIN=$PB/bin/singlet-fixed-det; [ $B = prod ] && BIN=$PB/bin/singlet-prod
  $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/$B.1fq > $SCR/encode_$B.log 2>&1
  for T in 4 8; do
    O=$PB/runs/$ID/singlet-$B-t$T; rm -rf $O; mkdir -p $O; cd $SCR
    /usr/bin/time -v -o $O/time.txt $BIN $SCR/$B.1fq --ref-base $REF --genome-dir $GENOME --exons $GTF \
      --out-prefix $O/ --threads $T --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "[val] $B t$T exit $?"
    cp $SCR/encode_$B.log $O/encode.log
    grep -E "Elapsed \(wall" $O/time.txt | sed "s/^/[val] $B t$T /"
  done
done
rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz
cd $SRC
for T in 4 8; do
  python -P bench/prod_atlas/compare_runs.py $PB/runs/$ID/singlet-det-t$T $PB/runs/$ID/singlet-prod-t$T --json $RES/$ID.det_vs_prod_t$T.json | cut -c1-1500
done
python -P bench/prod_atlas/compare_runs.py $PB/runs/$ID/singlet-prod-t4 $PB/runs/$ID/singlet-prod-t8 --json $RES/$ID.prod_t4_vs_t8.json | cut -c1-1500
python -P -c "import numpy, scipy, numba; print('[val] pbench env: numpy', numpy.__version__, 'scipy', scipy.__version__, 'numba', numba.__version__)"
python -P bench/called_ed/equiv_cpp.py $PB/runs/$ID/singlet-prod-t4 $PB/runs/$ID/singlet-prod-t8 --json $RES/$ID.equiv_pbench.json || echo "[val] equiv (pbench env) FAILED"
# the pinned runtime venv (numpy 2.2 / scipy 1.15 / numba 0.64), as the v2 migration uses
if [ -f $W/src/singlet-agents/main/hpc/clipper/in_runtime.sh ]; then
  source $W/src/singlet-agents/main/hpc/clipper/in_runtime.sh
  in_runtime bash -c "cd $SRC && python -P bench/called_ed/equiv_cpp.py $PB/runs/$ID/singlet-prod-t4 --json $RES/$ID.equiv_venv.json" || echo "[val] equiv (venv) FAILED"
fi
grep -h "called_v2:" $PB/runs/$ID/singlet-prod-t4/run.log | head -2
grep -h "Strand\|strand" $PB/runs/$ID/singlet-prod-t4/run.log | head -8
