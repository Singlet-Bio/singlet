#!/bin/bash
#SBATCH --job-name=pub-singlet
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Singlet 1.0.0 public benchmark: the campaign binary (prod/campaign 02aa9b9, PGO, sha256 261d3a50...) on every cohort
# dataset, from the FASTQs: `singlet encode` then the campaign flags, with the dataset's own reference.
#   sbatch --array=0-7 singlet_public.sbatch
# Output: $PB/runs/<dataset>/singlet-public/ (+ encode.log, encode.time.txt, time.txt, command.txt). The run directory also
# holds private/ (donor genotypes). It stays on Clipper and is never read by the benchmark extractors.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=$PB/bin/singlet-campaign
EXPECT=261d3a50ed624572418d30489dfd58904323f6e0d8820bd5260dfe7fb5038e6c
[ "$(sha256sum $BIN | cut -d' ' -f1)" = "$EXPECT" ] || { echo "binary sha mismatch"; exit 3; }
DSS=(pbmc1k_v3 pbmc1k_v2 neuron1k_v3 hgmm1k_v3 pbmc1k_cite_v3 nuclei2k_v2 pbmc5p_v1 kang2018_ctrl)
ID=${DSS[$SLURM_ARRAY_TASK_ID]}; T=$SLURM_CPUS_PER_TASK
case $ID in
  hgmm1k_v3) SPECIES=GRCh38-GRCm39-barnyard ;;
  neuron1k_v3|nuclei2k_v2) SPECIES=GRCm39-2024-A ;;
  *) SPECIES=GRCh38-2024-A ;;
esac
SCR=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
O=$PB/runs/$ID/singlet-public; rm -rf $O; mkdir -p $O
echo "[pub-singlet] $ID species=$SPECIES node=$(hostname) $(date -Is)"
echo "node: $(hostname); cpu: $(lscpu | sed -n 's/^Model name: *//p')" > $O/node.txt
if [ $ID = kang2018_ctrl ]; then
  R1=$W/modality-bench/genotype_pool/fastq/R1.fastq.gz; R2=$W/modality-bench/genotype_pool/fastq/R2.fastq.gz
else
  D=$PB/data/$ID
  cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz
  cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
  R1=$SCR/R1.fastq.gz; R2=$SCR/R2.fastq.gz
fi
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $R1 $R2 -o $SCR/s.1fq > $O/encode.log 2>&1
[ $ID = kang2018_ctrl ] || rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz
export SINGLET_STALL_TIMEOUT=10800
CMD=($BIN $SCR/s.1fq --ref-base $REF --genome-dir $REF/$SPECIES/star_2.7.11b --exons $REF/$SPECIES/genes/genes.gtf
     --out-prefix $O/ --threads $T --pipeline --output-format mtx --raw-matrix)
echo "${CMD[*]}" > $O/command.txt
cd $SCR
/usr/bin/time -v -o $O/time.txt "${CMD[@]}" > $O/run.log 2>&1 || echo "[pub-singlet] exit $?"
grep -E "^\[singlet\] \.1fq:|cell_calling\] called_v2|Total:" $O/run.log | cut -c1-300
grep -E "Elapsed \(wall|Maximum resident" $O/time.txt $O/encode.time.txt
