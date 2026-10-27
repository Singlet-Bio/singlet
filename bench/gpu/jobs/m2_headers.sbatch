#!/bin/bash
#SBATCH --job-name=w11-hdr
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Header-only read of staged .1fq files (read counts, protocol, R2 length). Read-only.
W=/mnt/projects/debruinz_project/debruinz/singlet
BIN=$W/pipeline-benchmark/bin/singlet-w3
ls -l --time-style=long-iso $W/pipeline-benchmark/bin | sort -k6,7 | tail -5
$BIN decode --help 2>&1 | head -3
for f in $W/bench-public/data/kang_b1A/kang_b1A.1fq $W/modality-bench/genotype_pool/kang21.1fq \
         $W/data/atlas-rerun/perf/GSM*/SRR*.1fq $W/data/bam-stream/GSM*/SRR*.1fq $W/pipeline-benchmark/data/*/*.1fq; do
  echo "#### $f"; $BIN decode "$f" --header 2>&1 | grep -v '^$'
done
