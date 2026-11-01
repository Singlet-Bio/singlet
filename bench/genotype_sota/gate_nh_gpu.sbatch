#!/bin/bash
#SBATCH --job-name=gsota-gatenhgpu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=160G
#SBATCH --time=03:55:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# nonhost-confirmed-reads: dpi2 on the new binary with --engine gpu; the member and the counts must equal the CPU run of gate_nh.sbatch.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
NEW=$W/jobs/gsota-r12/bin_nh; G=$REF/GRCh38-2024-A
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
R=$W/jobs/gsota-gatenh/dpi2; O=$R/gpu; rm -rf $O; mkdir -p $O
D=$W/modality-bench/sj_nonhost/data/dpi2
$NEW/singlet-gpu encode --reads $D/R1.fastq.gz $D/R2.fastq.gz -o $SCR/s.1fq > $O/encode.log 2>&1 || exit 1
$NEW/singlet-gpu $SCR/s.1fq --engine gpu --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $O/ --threads 18 --pipeline --output-format 1pz > $O/run.log 2>&1 || echo "EXIT $?"
grep -a "engine_gpu\|confirmed-species" $O/run.log | head
sha256sum $R/new/nonhost/nonhost_confirmed_reads.bin $O/nonhost/nonhost_confirmed_reads.bin
cmp $R/new/nonhost/nonhost_confirmed_reads.bin $O/nonhost/nonhost_confirmed_reads.bin && echo MEMBER_CPU_GPU_IDENTICAL
python3 $W/jobs/gsota-det/det_compare.py $R/new $O 2>&1 | head -40
