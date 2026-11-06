#!/bin/bash
#SBATCH --job-name=gbold-ncu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --exclude=g052
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Nsight Compute on the engine's hot kernels of one small sample (warp efficiency, achieved occupancy, throughput). One run, ncu kernel replay.
# Parameters (--export): BINDIR, EXPECT_COMMIT (8+ hex), SAMPLE (default pbmc2m), NCU_COUNT (kernel launches to profile, default 150), MEMGB (device budget, default 24).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; BIN=$BINDIR/singlet-gpu; EXPECT_COMMIT=${EXPECT_COMMIT:?}; SAMPLE=${SAMPLE:-pbmc2m}; NCU_COUNT=${NCU_COUNT:-150}; MEMGB=${MEMGB:-24}
COMMIT=$(cat $BINDIR/commit-gpu); case $COMMIT in $EXPECT_COMMIT*) ;; *) echo "REFUSED: commit $COMMIT"; exit 2 ;; esac
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) bin=$BIN sha256=$(sha256sum $BIN | cut -d' ' -f1) commit=$COMMIT sample=$SAMPLE count=$NCU_COUNT $(date -Is)"
R=$W/jobs/gpu-bold/results/ncu-$SLURM_JOB_ID; mkdir -p $R
SCR=/tmp/$USER-gbncu-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf "${SCR:?}"' EXIT
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_SLOTS=1 SINGLET_ENGINE_GPU_MEM_GB=$MEMGB
G=$REF/GRCh38-2024-A
cp $PB/data/$SAMPLE/$SAMPLE.1fq $SCR/
NCU=/usr/local/cuda/bin/ncu
KRE='regex:k_p1s|k_p1w|k_p1wp|k_p2|k_p3|k_p3a|k_p3w|k_p3e|k_cost|k_clip|k_pack|k_cr4|k_hash|k_groups|k_heads|k_collide|k_rec_len|k_rec_copy|k_rec$|k_entries|k_eval|k_scatter|k_crkeys|k_pass_eval|k_pass_scatter'
M=gpu__time_duration.sum,launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_blocks,launch__occupancy_limit_warps,sm__warps_active.avg.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__sass_average_branch_targets_threads_uniform.pct,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sector_hit_rate.pct,smsp__inst_executed.sum,sm__cycles_active.avg,smsp__issue_active.avg.pct_of_peak_sustained_active
( cd $SCR && timeout 4800 $NCU --target-processes all --kernel-name "$KRE" --launch-count $NCU_COUNT --metrics $M --csv --log-file $R/ncu.csv --cache-control none --clock-control none \
    $BIN $SCR/$SAMPLE.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads 8 --pipeline --output-format 1pz --engine gpu > $R/run.log 2>&1 ); echo "exit $?"
ls -la $R; grep -c . $R/ncu.csv
R=$R python3 - <<'EOF' > $R/ncu_summary.txt 2>&1
import csv, collections, os, re
R = os.environ["R"]
lines = [l for l in open(R + "/ncu.csv", errors="replace") if l.startswith('"')]
rows = list(csv.DictReader(lines))
print("rows", len(rows))
agg = collections.OrderedDict()
for r in rows:
    k = r.get("Kernel Name", "?")[:60]; m = r.get("Metric Name"); v = r.get("Metric Value", "0").replace(",", "")
    try: v = float(v)
    except ValueError: continue
    a = agg.setdefault(k, {}).setdefault(m, []); a.append(v)
def mean(x): return sum(x) / len(x) if x else float("nan")
cols = ["gpu__time_duration.sum", "launch__grid_size", "launch__block_size", "launch__registers_per_thread", "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__thread_inst_executed_per_inst_executed.ratio", "smsp__sass_average_branch_targets_threads_uniform.pct", "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed", "smsp__issue_active.avg.pct_of_peak_sustained_active"]
print("kernel | launches | " + " | ".join(c.split(".")[0].replace("launch__", "").replace("smsp__", "").replace("sm__", "")[:28] for c in cols))
for k, d in agg.items():
    n = len(d.get("launch__grid_size", []))
    print(k, "|", n, "|", " | ".join(f"{mean(d.get(c, [])):.4g}" for c in cols))
EOF
cat $R/ncu_summary.txt | head -60
echo done $(date -Is)
