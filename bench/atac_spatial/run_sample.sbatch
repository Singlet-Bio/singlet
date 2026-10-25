#!/bin/bash
#SBATCH --job-name=as-run
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One Singlet run of a benchmark library from its 10x FASTQs, timed with /usr/bin/time -v.
#   RUN=<name> KIND=atac|gex|visium FQ=<fastq dir> GLOB=<file prefix> GENOME=GRCh38-2024-A|GRCm39-2024-A
#   [PROTO=10x-atac|10x-visium|...] [PAIRED_GEX=<gex run out dir>] [BIN=singlet-atacsp] [THREADS=16]
# Output: pipeline-benchmark/runs/atacsp/<RUN>/ (run.log, time.txt, pipeline outputs; ATAC keeps atac_sorted.bam).
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
G=$REF/$GENOME; BIN=$PB/bin/${BIN:-singlet-atacsp}; T=${THREADS:-16}; export SINGLET_STALL_TIMEOUT=10800
O=$PB/runs/atacsp/$RUN; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; export TMPDIR=$SCR
echo "node=$(hostname) $(nproc) cores; bin sha256 $(sha256sum $BIN | cut -c1-16)"
cat $(ls $FQ/${GLOB}*_R1_*.fastq.gz | sort) > $SCR/R1.fastq.gz
cat $(ls $FQ/${GLOB}*_R2_*.fastq.gz | sort) > $SCR/R2.fastq.gz
READS="$SCR/R1.fastq.gz $SCR/R2.fastq.gz"
if [ "$KIND" = atac ]; then cat $(ls $FQ/${GLOB}*_R3_*.fastq.gz | sort) > $SCR/R3.fastq.gz; READS="$READS $SCR/R3.fastq.gz"; fi
P=""; [ -n "${PROTO:-}" ] && P="--protocol $PROTO"
/usr/bin/time -v -o $O/time_encode.txt $BIN encode --reads $READS $P -o $SCR/x.1fq > $O/encode.log 2>&1 || { tail -20 $O/encode.log; exit 1; }
rm -f $SCR/R*.fastq.gz
X=""; [ -n "${PAIRED_GEX:-}" ] && X="--paired-gex $PAIRED_GEX"
[ "$KIND" = atac ] && export SINGLET_KEEP_ATAC_BAM=$O
cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $SCR/x.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
  --out-prefix $O/ --threads $T --pipeline --output-format mtx --raw-matrix $X > $O/run.log 2>&1
echo "exit=$? wall=$(grep 'Elapsed (wall' $O/time.txt | sed 's/.*: //') rss_kb=$(grep 'Maximum resident' $O/time.txt | sed 's/.*: //') encode_wall=$(grep 'Elapsed (wall' $O/time_encode.txt | sed 's/.*: //')"
grep -E "^\[atac\]|^\[visium\]|ATAC|Visium|cells called|ERROR" $O/run.log | head -40
ls $O | head -80
