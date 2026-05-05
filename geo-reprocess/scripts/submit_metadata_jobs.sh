#!/bin/bash
#
# submit_metadata_jobs.sh — Submit metadata pipeline SLURM jobs
#
# Usage:
#   bash submit_metadata_jobs.sh <stage> [n_jobs] [walltime]
#
# Stages:
#   0, 1, 2a, 2b, 2c, 3  — Original per-GSM metadata pipeline
#   retry-2a, retry-2b, retry-2c — Retry failed/zero-match items with improved code
#   bulk-soft              — Extract SOFT metadata for ALL ~1.49M GSMs
#   bulk-desc              — Fetch NCBI descriptions for ALL ~24K GSEs
#
# Examples:
#   bash submit_metadata_jobs.sh 0           # Stage 0: 2 jobs, 2 days each
#   bash submit_metadata_jobs.sh 1 4 2-00    # Stage 1: 4 jobs, 2 days each
#   bash submit_metadata_jobs.sh retry-2a 4  # Retry Stage 2a: 4 jobs
#   bash submit_metadata_jobs.sh bulk-soft 8 # Bulk SOFT: 8 jobs
#   bash submit_metadata_jobs.sh bulk-desc 4 # Bulk NCBI: 4 jobs
#
set -euo pipefail

STAGE="${1:?Usage: $0 <stage> [n_jobs] [walltime]}"
N_JOBS="${2:-2}"
WALLTIME="${3:-2-00:00:00}"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="${SCRIPTS_DIR}/run_metadata_pipeline.py"
LOG_DIR="/mnt/projects/debruinz_project/cellarium/pipeline/logs"
mkdir -p "${LOG_DIR}"

# Resource requirements per stage
declare -A CPUS_MAP=( [0]=2 [1]=4 [2a]=4 [2b]=4 [2c]=4 [3]=2 \
                      [retry-2a]=4 [retry-2b]=4 [retry-2c]=4 \
                      [bulk-soft]=2 [bulk-desc]=2 )
declare -A MEM_MAP=( [0]=4G [1]=8G [2a]=32G [2b]=24G [2c]=32G [3]=4G \
                     [retry-2a]=32G [retry-2b]=24G [retry-2c]=32G \
                     [bulk-soft]=32G [bulk-desc]=4G )

CPUS="${CPUS_MAP[$STAGE]:-2}"
MEM="${MEM_MAP[$STAGE]:-8G}"

echo "═══════════════════════════════════════════════"
echo " Metadata Pipeline — Stage ${STAGE}"
echo " Jobs: ${N_JOBS}, Walltime: ${WALLTIME}"
echo " CPUs: ${CPUS}, Memory: ${MEM} per job"
echo "═══════════════════════════════════════════════"

JOB_IDS=()
for ((i=0; i<N_JOBS; i++)); do
    JOB_NAME="meta_s${STAGE}_b${i}"
    
    JOB_ID=$(sbatch \
        --partition=cpu \
        --time="${WALLTIME}" \
        --cpus-per-task="${CPUS}" \
        --mem="${MEM}" \
        --job-name="${JOB_NAME}" \
        --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
        --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
        --export=ALL \
        --wrap="
module load miniconda3/25.5.1
# Load R module for stages that need rpy2 (2b, 2c, and their retries)
if [[ '${STAGE}' == '2b' || '${STAGE}' == 'retry-2b' || '${STAGE}' == '2c' || '${STAGE}' == 'retry-2c' ]]; then module load r/4.5.2; fi
eval \"\$(conda shell.bash hook 2>/dev/null)\"
conda activate cellarium

echo \"Starting metadata pipeline stage ${STAGE}, batch ${i}/${N_JOBS}\"
echo \"Node: \$(hostname), Date: \$(date)\"
echo \"Python: \$(which python)\"

cd ${SCRIPTS_DIR}/..
python ${PIPELINE_SCRIPT} --stage ${STAGE} --batch ${i} --total-batches ${N_JOBS}

echo \"Finished: \$(date)\"
" | awk '{print $4}')
    
    JOB_IDS+=("${JOB_ID}")
    echo "  Submitted batch ${i}/${N_JOBS}: job ${JOB_ID} (${JOB_NAME})"
done

echo ""
echo "All jobs submitted: ${JOB_IDS[*]}"
echo "Monitor: squeue -u \$USER -n meta_s${STAGE}_b0"
echo "Logs:    tail -f ${LOG_DIR}/meta_s${STAGE}_b*"

# Submit a merge job that waits for all batch jobs to finish
# Choose the right merge command
MERGE_CMD="python ${PIPELINE_SCRIPT} --stage merge"
if [[ "${STAGE}" == "bulk-soft" ]]; then
    MERGE_CMD="python ${PIPELINE_SCRIPT} --stage bulk-merge-soft"
elif [[ "${STAGE}" == "bulk-desc" ]]; then
    MERGE_CMD="python ${PIPELINE_SCRIPT} --stage bulk-merge-desc"
fi

DEPEND_STR=$(IFS=:; echo "${JOB_IDS[*]}")
MERGE_ID=$(sbatch \
    --partition=cpu \
    --time=01:00:00 \
    --cpus-per-task=2 \
    --mem=32G \
    --job-name="meta_s${STAGE}_merge" \
    --output="${LOG_DIR}/meta_s${STAGE}_merge_%j.out" \
    --error="${LOG_DIR}/meta_s${STAGE}_merge_%j.err" \
    --dependency=afterany:${DEPEND_STR} \
    --export=ALL \
    --wrap="
module load miniconda3/25.5.1
eval \"\$(conda shell.bash hook 2>/dev/null)\"
conda activate cellarium

echo \"Merging for stage ${STAGE}\"
cd ${SCRIPTS_DIR}/..
${MERGE_CMD}

echo \"Merge finished: \$(date)\"
" | awk '{print $4}')

echo "Merge job: ${MERGE_ID} (runs after all batches complete)"
