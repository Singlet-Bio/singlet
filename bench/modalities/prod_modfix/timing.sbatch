#!/bin/bash
#SBATCH --job-name=mf-timing
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Same-node wall time of singlet-prod vs singlet-modfix (8 threads, the flags of validate.sbatch), one run after the
# other on an exclusive node: pbmc1k_v3 and pbmc5p_v1 (+ modfix --no-adapter-clip, + modfix --no-snps on pbmc1k).
# The validate.sbatch pbmc runs shared node c008 with the Kang run and other users' jobs (STAR mapping speed 3-4x lower).
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
G=$REF/GRCh38-2024-A; export SINGLET_STALL_TIMEOUT=7200
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "node=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p') $(nproc) cores"
for ID in pbmc1k_v3 pbmc5p_v1; do
  D=$PB/data/$ID
  cat $(grep "_R1_" $D/gex_fastqs.txt) > $SCR/R1.fastq.gz; cat $(grep "_R2_" $D/gex_fastqs.txt) > $SCR/R2.fastq.gz
  $PB/bin/singlet-modfix encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/$ID.1fq > /dev/null 2>&1
  RUNS="prod: modfix: modfix:--no-adapter-clip"; [ $ID = pbmc1k_v3 ] && RUNS="prod: modfix: modfix:--no-snps"
  for R in $RUNS; do
    B=${R%%:*}; X=${R#*:}; TAG=$B${X//-/_}
    O=$SCR/out_${ID}_$TAG; rm -rf $O; mkdir -p $O; cd $SCR
    /usr/bin/time -v -o $SCR/time.txt $PB/bin/singlet-$B $SCR/$ID.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $O/ --threads 8 --pipeline --output-format mtx --raw-matrix $X > $O/run.log 2>&1
    echo "$ID $TAG wall=$(grep 'Elapsed (wall' $SCR/time.txt | sed 's/.*: //') rss_kb=$(grep 'Maximum resident' $SCR/time.txt | sed 's/.*: //') star=$(grep -o 'STAR completed in [0-9.]*s' $O/run.log) map_speed=$(grep 'Mapping speed' $O/star_Log.final.out | sed 's/.*|//') uniq=$(grep 'Uniquely mapped reads %' $O/star_Log.final.out | sed 's/.*|//') export=$(grep -o '\[export\] Total: [0-9.]*s' $O/run.log)"
    grep -E "Pileup: [0-9]|demux\] (auto-K: selected|Best K)" $O/run.log | head -3
    rm -rf $O
  done
  rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz $SCR/$ID.1fq
done
