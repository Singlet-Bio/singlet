#!/bin/bash
#SBATCH --job-name=mf-val
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --array=0-5%6
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# bin/singlet-modfix (prod/modality-fixes) on the 5 pipeline benchmark datasets, 8 threads, same flags as the
# singlet-prod-t8 reference runs (bench/prod_atlas/validate.sbatch); the default SNP panel now applies (--pipeline).
#   * gene / exon / intron / sj matrices vs singlet-prod-t8 (compare_runs.py + modfix_compare.py)
#   * wall time and peak RSS (time.txt)
# Task 5 = pbmc5p_v1 with --no-adapter-clip (is the 5' TSO clip worth its STAR time?).
# Output: $PB/runs/<dataset>/singlet-modfix-t8[-noadapter]/ and $PB/results/modfix/<dataset>*.json
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; ENV=$W/envs/pbench
SRC=$W/src/singlet-py/prod__modality-fixes
REF=/mnt/projects/debruinz_project/cellarium/reference
export PATH=$ENV/bin:$PATH SINGLET_STALL_TIMEOUT=7200
DSS=(pbmc1k_v3 pbmc5p_v1 nuclei2k_v2 neuron1k_v3 hgmm1k_v3 pbmc5p_v1)
T=${SLURM_ARRAY_TASK_ID:-0}; ID=${1:-${DSS[$T]}}
EXTRA=""; TAG=singlet-modfix-t8
if [ "$T" = 5 ]; then EXTRA="--no-adapter-clip"; TAG=singlet-modfix-t8-noadapter; fi
IFS=$'\t' read -r _ SPECIES CHEM _ _ _ < <(awk -F'\t' -v id="$ID" '$1==id' $SRC/bench/pipeline/datasets.tsv)
GENOME=$REF/$SPECIES/star_2.7.11b; GTF=$REF/$SPECIES/genes/genes.gtf; D=$PB/data/$ID
RES=$PB/results/modfix; mkdir -p $RES
SCR=/tmp/$USER-$SLURM_JOB_ID-$T; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
BIN=$PB/bin/singlet-modfix
echo "[val] $ID $TAG species=$SPECIES node=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p') $(date -Is) bin=$(sha256sum $BIN | cut -c1-16)"
cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz
cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
$BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/s.1fq > $SCR/encode.log 2>&1
rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz
O=$PB/runs/$ID/$TAG; rm -rf $O; mkdir -p $O; cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $REF --genome-dir $GENOME --exons $GTF \
  --out-prefix $O/ --threads 8 --pipeline --output-format mtx --raw-matrix $EXTRA > $O/run.log 2>&1 || echo "[val] exit $?"
cp $SCR/encode.log $O/encode.log
grep -E "Elapsed \(wall|Maximum resident" $O/time.txt | sed "s/^/[val] /"
cd $SRC
python -P bench/prod_atlas/compare_runs.py $PB/runs/$ID/singlet-prod-t8 $O --json $RES/$ID.$TAG.prod_vs_modfix.json | cut -c1-2500
python -P bench/modalities/prod_modfix/modfix_compare.py $PB/runs/$ID/singlet-prod-t8 $O --json $RES/$ID.$TAG.modfix_compare.json | cut -c1-5000
grep -h -E "^\[(singlet|doublet|ambient|demux|donor|mt_|polya|sj_features|cell_qc|nonhost|barnyard|export)\]" $O/run.log | grep -v -E "^\[export\] (CSC|Standard)" | head -60
ls -la $O $O/private 2>/dev/null | head -80
if [ "$ID" = hgmm1k_v3 ]; then
  $W/envs/ambient/bin/python -P bench/modalities/prod_modfix/hgmm_e2e.py $O $RES/hgmm1k_v3.e2e_ambient_doublets.json | head -80
fi
