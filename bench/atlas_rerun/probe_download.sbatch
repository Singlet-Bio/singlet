#!/bin/bash
#SBATCH --job-name=ar-dl
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch probe_download.sbatch <GSM> <SRR,SRR,...>
# Timing probe: download every run of one sample from SRA (prefetch) and convert to FASTQ
# (fasterq-dump), recording seconds and bytes per step in $D/timing.tsv.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; export PATH=$ENV/bin:$PATH
export HOME=$W/cache/condahome   # sra-tools writes its config under $HOME
GSM=$1; RUNS=${2//,/ }
D=$W/data/atlas-rerun/depth/$GSM; mkdir -p $D/sra $D/fq; cd $D
echo -e "run\tstep\tseconds\tbytes" > timing.tsv
for r in $RUNS; do
  t0=$(date +%s); prefetch --max-size u -O sra $r > /dev/null; t1=$(date +%s)
  echo -e "$r\tprefetch\t$((t1-t0))\t$(du -sb sra/$r | cut -f1)" >> timing.tsv
  fasterq-dump -e 8 -t /tmp/$USER-$SLURM_JOB_ID --split-files --include-technical -O fq sra/$r/$r.sra > /dev/null; t2=$(date +%s)
  pigz -p 8 fq/${r}_*.fastq; t3=$(date +%s)
  echo -e "$r\tfasterq_dump\t$((t2-t1))\t$(du -cb fq/${r}_* | tail -1 | cut -f1)" >> timing.tsv
  echo -e "$r\tgzip\t$((t3-t2))\t0" >> timing.tsv
  rm -rf sra/$r
done
cat timing.tsv; ls -la fq | head
