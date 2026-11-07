#!/bin/bash
#SBATCH --job-name=s8-encode
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# S8: re-encode the two 10x test samples with the current encoder and the default ILLUMINA4 binned qualities (`singlet encode`, as
# `singlet download` writes today), for gates that need real, non-constant base qualities. Output: $W/data/s8/<sample>/<sample>_binned.1fq
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=${BIN:-$W/jobs/s8/bin_base2/singlet-gpu}
SAMPLES=(pbmc1k_v3 neuron1k_v3); s=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
case $s in pbmc1k_v3) F=$PB/data/pbmc1k_v3/pbmc_1k_v3_fastqs/pbmc_1k_v3_S1_L00; SFX=_R1_001.fastq.gz; LANES="1 2" ;; neuron1k_v3) F=; SFX=; LANES= ;; esac
SCR=/tmp/$USER-s8enc-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
if [ $s = pbmc1k_v3 ]; then
  for l in 1 2; do cat ${F}${l}_R1_001.fastq.gz >> $SCR/R1.fastq.gz; cat ${F}${l}_R2_001.fastq.gz >> $SCR/R2.fastq.gz; done
else
  # the GEX FASTQs listed in gex_fastqs.txt, R1 and R2 in file order
  grep -E "_R1_" $PB/data/$s/gex_fastqs.txt | while read f; do cat $f >> $SCR/R1.fastq.gz; done
  grep -E "_R2_" $PB/data/$s/gex_fastqs.txt | while read f; do cat $f >> $SCR/R2.fastq.gz; done
fi
O=$W/data/s8/$s; mkdir -p $O
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/${s}_binned.1fq --whitelist-dir $PB/whitelists > $O/encode.log 2>&1
mv $SCR/${s}_binned.1fq $O/${s}_binned.1fq
ls -l $O; grep -a -E "Done|qual|BC dict" $O/encode.log | cut -c1-160 | tail -5; grep -E "Elapsed|User time" $O/encode.time.txt
