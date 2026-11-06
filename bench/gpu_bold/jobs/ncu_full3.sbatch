#!/bin/bash
#SBATCH --job-name=gbold-ncufull3
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=00:40:00
#SBATCH --exclude=g052
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Advisor package pass. One solo pbmc1k_v3 run per ncu invocation, at the device budget of a 2-slot node (MEMGB=45), so the kernels are the ones a loaded node runs.
#  run 1: every prefill kernel launch, cheap metrics (duration, grid, block, registers) -> table of device time per kernel; the top kernel (and the second one
#         when it is within a factor of 2) is chosen from it
#  run 2 (and 3): --set full on two launches of that kernel from the middle of its launches, exported as .ncu-rep, then text exports (details, SASS source page)
# Parameters (--export): BINDIR, EXPECT_COMMIT, SAMPLE (pbmc1k_v3), MEMGB (45), RUNTMO (seconds per ncu run, 600).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; BIN=$BINDIR/singlet-gpu; EXPECT_COMMIT=${EXPECT_COMMIT:?}; SAMPLE=${SAMPLE:-pbmc1k_v3}; MEMGB=${MEMGB:-45}; RUNTMO=${RUNTMO:-600}
COMMIT=$(cat $BINDIR/commit-gpu); case $COMMIT in $EXPECT_COMMIT*) ;; *) echo "REFUSED: commit $COMMIT"; exit 2 ;; esac
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) bin=$BIN sha256=$(sha256sum $BIN | cut -d' ' -f1) commit=$COMMIT sample=$SAMPLE memgb=$MEMGB $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; /usr/local/cuda/bin/ncu --version | tail -1; /usr/local/cuda/bin/nvcc --version | tail -1
R=$W/jobs/gpu-bold/results/ncufull-$SLURM_JOB_ID; mkdir -p $R
SCR=/tmp/$USER-gbncuf-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf "${SCR:?}"' EXIT
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_SLOTS=1 SINGLET_ENGINE_GPU_MEM_GB=$MEMGB
G=$REF/GRCh38-2024-A; NCU=/usr/local/cuda/bin/ncu
cp $PB/data/$SAMPLE/$SAMPLE.1fq $SCR/
runncu() {   # tag ncu-args...
  local tag=$1; shift; rm -rf $SCR/out; mkdir -p $SCR/out
  ( cd $SCR && timeout $RUNTMO $NCU --target-processes all --cache-control none --clock-control none "$@" \
      $BIN $SCR/$SAMPLE.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads 14 --pipeline --output-format 1pz --engine gpu > $R/$tag.run.log 2>&1 ); echo "$tag exit $?"
}
KRE='regex:k_p1s|k_p1w|k_p1wp|k_p2|k_p3|k_p3a|k_p3w|k_p3e|k_p3lb|k_cost|k_clip|k_pack|k_cr4|k_hash|k_groups|k_heads|k_collide|k_rec_len|k_rec_copy|k_rec$|k_entries|k_eval|k_scatter|k_crkeys|k_pass_eval|k_pass_scatter|k_copy_status'
# the kernels of run 1 of job 800512 (results/ncufull-800512/run1_table.txt): lane B k_p3e 39.3%, lane A k_p3lb 13.4%, lane M k_p3w 12.8% of the prefill's kernel time.
# (job 800512's own --set full pass profiled nothing: its regex had doubled backslashes.) Two launches of each from the middle of its launches.
i=1
for spec in "k_p3e|20" "k_p3lb|38" "k_p3w|20"; do   # function names only: ncu matches the name without template arguments; at a 45 GB budget each name is one lane
  KR="regex:${spec%%|*}"; SK=${spec##*|}
  runncu full$i --kernel-name "$KR" --launch-skip $SK --launch-count 2 --set full --import-source yes --export $R/full$i --force-overwrite
  if [ -f $R/full$i.ncu-rep ]; then
    ls -la $R/full$i.ncu-rep
    $NCU --import $R/full$i.ncu-rep --page details --print-units base > $R/ncu_details_$i.txt 2>&1
    $NCU --import $R/full$i.ncu-rep --page source --print-source sass --csv > $R/ncu_source_$i.txt 2>&1
    $NCU --import $R/full$i.ncu-rep --page raw --csv --metrics launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,launch__stack_size,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__occupancy_limit_blocks > $R/ncu_launch_$i.csv 2>&1
  else echo "no report for kernel $i: $(grep -a 'No kernels' $R/full$i.run.log)"; fi
  i=$((i + 1))
done
ls -la $R
echo done $(date -Is)
