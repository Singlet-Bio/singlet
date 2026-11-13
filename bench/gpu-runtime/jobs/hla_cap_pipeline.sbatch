#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:quadro_rtx_8000:1 -c 18 --mem=170G --time=01:00:00 --exclude=g050 -J hla-cap-pipe -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# hla-cap-pipe: the cap inside the GPU pipeline on a 5' library typed on the node (--hla-gpu-5p type --hla-max-reads CAP): its genotype
# table against the one `singlet hla --max-reads CAP` wrote (REFTSV, from hla_cap_check.sbatch), byte for byte, and the run's HLA lines.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:?}; REFTSV=${REFTSV:?}; S=${S:-pbmc5p_v1}; CAP=${CAP:-500000}; TK=${TK:-$W/jobs/gpu-campaign/agents/hla-speed/t1k-p3b}; T=${SLURM_CPUS_PER_TASK:-18}; GD=$REF/GRCh38-2024-A
SCR=/tmp/$USER-hlacappipe-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf $SCR' EXIT; umask 077
export PATH=$W/envs/pbench/bin:$PATH SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_REF_BASE=$REF SINGLET_STALL_TIMEOUT=3600 TMPDIR=$SCR
F=$PB/data/$S/$S.1fq
echo "node=$(hostname) bin=$BIN sample=$S cap=$CAP $(date -Is)"
( cd $SCR && SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70 CUDA_VISIBLE_DEVICES=0 \
    /usr/bin/time -v -o $SCR/time.txt $BIN $F --ref-base $REF --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf --out-prefix $SCR/out/ --threads $T \
    --pipeline --output-format 1pz --engine gpu --hla-t1k $TK --no-editing --hla-gpu-5p type --hla-max-reads $CAP > $SCR/run.log 2>&1 ); rc=$?
echo "pipeline rc=$rc; $(grep -E 'Elapsed' $SCR/time.txt | tr -s ' \t' ' ')"
grep -a -E '^\[hla\]' $SCR/run.log | cut -c1-220
ls -la $SCR/out/private/hla/ 2>/dev/null | awk '{print $5, $9}' | grep -v '^ *$'
grep -o '"prefilter": {[^}]*}' $SCR/out/private/hla/hla_summary.json | cut -c1-300
if cmp -s $SCR/out/private/hla/hla_genotype.tsv $REFTSV; then echo "HLA-CAP-PIPELINE PASS: the pipeline's genotype table equals singlet hla --max-reads $CAP"; else echo "HLA-CAP-PIPELINE FAIL: the tables differ"; fi
