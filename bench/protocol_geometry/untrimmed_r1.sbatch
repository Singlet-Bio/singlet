#!/bin/bash
#SBATCH --job-name=s14-untrimmed
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Builds tests/cpp/test_protocol_untrimmed_r1.cpp (header-only: protocol.h) and runs it with the real first spots of SRR23617320 (GSM7064234, untrimmed 3' v3 R1)
# and the real whitelists. Env: BRANCH (default fix/r1-untrimmed-protocol), ACC (default SRR23617320).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
T=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT; cd $T
git clone -q --depth 1 -b ${BRANCH:-fix/r1-untrimmed-protocol} git@github.com:Singlet-Bio/singlet.git repo 2>&1 | tail -1; (cd repo && git log --oneline -1)
ACC=${ACC:-SRR23617320}; SRA=$W/pilot/GSM7064234/sra/$ACC.sra
fastq-dump -X 20000 --split-files -O $T $SRA 2>&1 | tail -1
CXX=$W/envs/pbench/bin/x86_64-conda-linux-gnu-g++
$CXX -O1 -std=c++17 -Irepo/include repo/tests/cpp/test_protocol_untrimmed_r1.cpp -o t -Wl,-rpath,$W/envs/pbench/lib 2>&1 | tail -20
S14_R1_FASTQ=$T/${ACC}_2.fastq S14_R2_FASTQ=$T/${ACC}_3.fastq S14_WL_DIR=$W/pipeline-benchmark/whitelists ./t 2>&1 | cut -c1-220
