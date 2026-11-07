#!/bin/bash
#SBATCH --job-name=ir-identity
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# input-routes: the FASTQ path stages the same .1fq as the SRA path. One SRA run (SRR, default SRR24098804: a 10x 3' run of the atlas
# re-run set, outside the benchmark corpus) is staged three ways with the binary in BIN (singlet-gpu of build.sbatch):
#   sra    singlet download <run>.sra                           (the SRA path, VDB)
#   fqall  singlet download --fastq-spec <every segment>        (fasterq-dump --split-files --include-technical of the same run)
#   fqgz   singlet download --r1 <bc>.fastq.gz --r2 <cdna>.fastq.gz  (the two biological reads only, gzipped, as a user would pass them)
# Every staging passes --source-label <run>, so the metadata block names the same source. Output: $W/jobs/input-routes/identity/<run>-<job>/
# (logs, sha256, cmp verdicts, decoded headers). The staged files are removed at the end.
#   sbatch --export=ALL,BIN=$W/jobs/input-routes/bin/<sha>/singlet-gpu[,SRR=...] identity.sbatch
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
BIN=${BIN:?BIN: singlet-gpu}; SRR=${SRR:-SRR24098804}; T=${SLURM_CPUS_PER_TASK:-16}
export PATH=$W/envs/pbench/bin:$PATH SINGLET_WHITELIST_DIR=$PB/whitelists
O=$W/jobs/input-routes/identity/$SRR-$SLURM_JOB_ID; mkdir -p $O
SCR=/tmp/$USER-ir-id-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "node=$(hostname) bin=$BIN commit=$(cat $(dirname $BIN)/commit-gpu 2>/dev/null) srr=$SRR $(date -Is)" | tee $O/summary.txt
cd $SCR
prefetch -q -O $SCR $SRR > $O/prefetch.log 2>&1 || { echo "prefetch failed" | tee -a $O/summary.txt; exit 1; }
SRA=$(ls $SCR/$SRR/*.sra* | head -1); echo "sra file $SRA $(stat -c %s $SRA) bytes" | tee -a $O/summary.txt
run() {   # name, args...
  local n=$1; shift
  /usr/bin/time -v -o $O/$n.time.txt $BIN download "$@" -o $SCR/$n.1fq --source-label $SRR --vdb-threads 8 > $O/$n.log 2>&1
  echo "$n: exit $? wall $(awk -F': ' '/Elapsed/{print $2}' $O/$n.time.txt) maxrss_kb $(awk -F': ' '/Maximum resident/{print $2}' $O/$n.time.txt) size $(stat -c %s $SCR/$n.1fq 2>/dev/null) sha256 $(sha256sum $SCR/$n.1fq 2>/dev/null | cut -c1-16)" | tee -a $O/summary.txt
  grep -aE "Protocol:|Done:|FASTQ layout|input spec|swap|Total spots" $O/$n.log | sed "s/^/  $n | /" | tee -a $O/summary.txt
}
run sra $SRA
fasterq-dump --split-files --include-technical -e $T -O $SCR/fq $SRA > $O/fasterq.log 2>&1
ls -l $SCR/fq | tee -a $O/summary.txt
SEGS=$(ls $SCR/fq/${SRR}_*.fastq | sort -t_ -k2 -n | paste -sd'|')
for f in $SCR/fq/${SRR}_*.fastq; do echo "$(basename $f): $(awk 'NR%4==2{print length($0)}' $f | head -100000 | sort | uniq -c | sort -rn | head -1)"; done | tee -a $O/summary.txt
run fqall --fastq-spec "$SEGS"
# the two longest segments, gzipped: barcode read first (the shorter of the two), then cDNA
mapfile -t L < <(for f in $SCR/fq/${SRR}_*.fastq; do echo "$(awk 'NR==2{print length($0)}' $f) $f"; done | sort -n | tail -2 | awk '{print $2}')
BCF=${L[0]}; CDF=${L[1]}
echo "fqgz: barcode $(basename $BCF), cDNA $(basename $CDF)" | tee -a $O/summary.txt
pigz -p $T -k $BCF $CDF
run fqgz --r1 $BCF.gz --r2 $CDF.gz
for n in fqall fqgz; do
  if cmp -s $SCR/sra.1fq $SCR/$n.1fq; then echo "VERDICT $n vs sra: IDENTICAL (byte for byte)" | tee -a $O/summary.txt
  else echo "VERDICT $n vs sra: DIFFERENT ($(cmp $SCR/sra.1fq $SCR/$n.1fq | head -1))" | tee -a $O/summary.txt; fi
done
for n in sra fqall fqgz; do $BIN decode $SCR/$n.1fq --header > $O/$n.header.txt 2>&1; done
diff $O/sra.header.txt $O/fqall.header.txt > $O/header.diff && echo "headers sra/fqall identical" | tee -a $O/summary.txt
echo "done $(date -Is)" | tee -a $O/summary.txt
