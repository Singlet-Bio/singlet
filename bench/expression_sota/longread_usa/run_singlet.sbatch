#!/bin/bash
#SBATCH --job-name=lr-singlet
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:59:00
# Singlet (pinned release-gate binary d9834b8a, same command as release-gate/jobs/run_sample_rg.sbatch KIND=gex) on full pbmc1.
set -u
source /mnt/projects/debruinz_project/debruinz/singlet/release-gate/jobs/rg_common.sh; rg_check_bin
source /mnt/projects/debruinz_project/debruinz/singlet/expression-sota/longread_usa/code/common.sh
O=$X/singlet; rm -rf $O; mkdir -p $O; export SINGLET_STALL_TIMEOUT=10800
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; export TMPDIR=$SCR
# pbmc1 R1 is 27 nt (16 CB + 11 UMI); Singlet's 10x v3 geometry is 28 nt (auto-detect picked "celseq2" on the raw 27-mers).
# Pad every R1 with a constant 'A' (quality F): UMI = 11 real bases + a constant 12th, which changes no UMI identity or collapse.
pad() { zcat "$1" | awk 'NR%4==2{print $0 "A"; next} NR%4==0{print $0 "F"; next} {print}' | gzip -1 > "$2"; }
i=0; for f in ${R1//,/ }; do pad $f $SCR/R1_$i.fastq.gz & i=$((i+1)); done; wait
cat $SCR/R1_*.fastq.gz > $SCR/R1.fastq.gz; rm -f $SCR/R1_*.fastq.gz
cat ${R2//,/ } > $SCR/R2.fastq.gz
/usr/bin/time -v -o $O/time_encode.txt $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz --protocol 10x-3p-v3 -o $SCR/x.1fq > $O/encode.log 2>&1 || { tail -20 $O/encode.log; exit 1; }
rm -f $SCR/R*.fastq.gz; tail -5 $O/encode.log
cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $SCR/x.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf   --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix "${ENGINE_ARGS[@]}" > $O/run.log 2>&1
echo "exit=$?"; grep -E "Elapsed \(wall|Maximum resident" $O/time.txt
cp $(dirname $BIN)/commit $O/binary_commit.txt; rg_stream_ok $O/run.log | tee $O/stream_path.txt
rm -rf $O/private; ls $O
