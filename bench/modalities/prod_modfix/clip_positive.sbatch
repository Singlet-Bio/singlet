#!/bin/bash
#SBATCH --job-name=mf-clippos
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Positive control for the 5' constant-prefix clip: SRR34789664 (mouse 10x v3, 50 bp constant prefix on the cDNA read;
# papers/manuscripts/singlify: 3 exon hits without the clip). First 4M spots via fastq-dump inside the job, encoded with
# bin/singlet-modfix from FASTQ (so the encoder's SRA-path metadata clip is not involved) and run with --pipeline:
# the spread probe must still clip ~50 bp. Also reruns with --no-clip5p to show what the clip buys.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/singlet-modfix
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCm39-2024-A
export HOME=$W/tools/home PATH=$W/envs/sratools/bin:$PATH SINGLET_STALL_TIMEOUT=7200
D=$W/modality-bench/clip_positive; mkdir -p $D $W/modality-bench/sj_nonhost/ncbi_cache/tmp
if [ ! -s $D/R2.fastq.gz ]; then
  cd $W/modality-bench/sj_nonhost/ncbi_cache/tmp
  rm -rf $D/raw; fastq-dump -X 4000000 --split-files -O $D/raw SRR34789664
  ls -la $D/raw
  for f in $D/raw/*.fastq; do echo "$f $(head -2 $f | tail -1 | awk '{print length($0)}')"; done
  # barcode read = 28 nt, cDNA read = the longest
  R1=$(for f in $D/raw/*.fastq; do echo "$(head -2 $f | tail -1 | awk '{print length($0)}') $f"; done | awk '$1==28{print $2}' | head -1)
  R2=$(for f in $D/raw/*.fastq; do echo "$(head -2 $f | tail -1 | awk '{print length($0)}') $f"; done | sort -n | tail -1 | cut -d' ' -f2)
  echo "R1=$R1 R2=$R2"; gzip -1 -c $R1 > $D/R1.fastq.gz; gzip -1 -c $R2 > $D/R2.fastq.gz; rm -rf $D/raw
fi
zcat $D/R2.fastq.gz | head -8 | sed -n '2p;6p'
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
$BIN encode --reads $D/R1.fastq.gz $D/R2.fastq.gz -o $SCR/s.1fq > $D/encode.log 2>&1; grep -i -E "protocol|prefix|clip" $D/encode.log | head
for MODE in clip noclip; do
  O=$D/run_$MODE; rm -rf $O; mkdir -p $O; X=""; [ $MODE = noclip ] && X="--no-clip5p"
  /usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
    --out-prefix $O/ --threads 8 --pipeline --output-format mtx $X > $O/run.log 2>&1 || echo "exit $?"
  echo "== $MODE"; grep -E "prefix|clip|Auto-clip" $O/run.log | head -8
  grep -E "Uniquely mapped reads %|Number of splices: Total" $O/star_Log.final.out
  grep -E "\"estimated_cells\"|\"median_umis_per_cell\"" $O/summary.json
done
