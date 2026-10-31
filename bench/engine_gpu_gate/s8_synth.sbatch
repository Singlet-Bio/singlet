#!/bin/bash
#SBATCH --job-name=s8-synth
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: a test sample whose low-quality bases are NOT tied to N: the first 8 M reads of pbmc1k_v3 lane 1 with Q2 ('#') written on 2 non-N bases
# of 1 read in 3 (deterministic from the read number), then `singlet encode` with the default ILLUMINA4 binned qualities. The CPU engine's SNP / chrM
# base-quality filter and the GPU pileup's per-base masks must agree on it. Output: $W/data/s8/synthq/synthq.1fq
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=${BIN:-$W/jobs/s8/bin_base2/singlet-gpu}; PY=$W/envs/pbench/bin/python3
D=$PB/data/pbmc1k_v3/pbmc_1k_v3_fastqs; SCR=/tmp/$USER-s8syn-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
N=${NREADS:-8000000}
set +o pipefail   # zcat gets SIGPIPE when head has its lines
( zcat $D/pbmc_1k_v3_S1_L001_R1_001.fastq.gz | head -n $((4*N)) | gzip -1 > $SCR/R1.fastq.gz ) &
( zcat $D/pbmc_1k_v3_S1_L001_R2_001.fastq.gz | head -n $((4*N)) | $PY -I -c '
import sys
out = sys.stdout
n = 0
for hdr in sys.stdin:
    seq = sys.stdin.readline(); plus = sys.stdin.readline(); q = sys.stdin.readline().rstrip("\n")
    n += 1
    if (n * 2654435761) % 3 == 0:
        s = seq.rstrip("\n"); L = len(s); ql = list(q)
        for p in ((n * 13) % L, (n * 29 + 7) % L):
            if s[p] != "N": ql[p] = "#"
        q = "".join(ql)
    out.write(hdr + seq + plus + q + "\n")
' | gzip -1 > $SCR/R2.fastq.gz )
wait
set -o pipefail
O=$W/data/s8/synthq; mkdir -p $O
$BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/synthq.1fq --whitelist-dir $PB/whitelists > $O/encode.log 2>&1
mv $SCR/synthq.1fq $O/synthq.1fq; ls -l $O; grep -a "Done" $O/encode.log | tail -1
