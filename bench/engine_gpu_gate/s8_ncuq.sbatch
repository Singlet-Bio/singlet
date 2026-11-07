#!/bin/bash
#SBATCH --job-name=s8-ncuq
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: pull the roofline numbers out of Nsight Compute reports: REPS = space separated .ncu-rep paths
set -uo pipefail
N=/usr/local/cuda/bin/ncu
M=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,sm__inst_executed.avg.per_cycle_active,smsp__issue_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__occupancy_limit_registers,launch__block_size,launch__grid_size,sm__cycles_active.avg,dram__bytes_read.sum,dram__bytes_write.sum,smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct,smsp__warp_issue_stalled_no_instruction_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct
for r in $REPS; do
  echo "=== $r"
  $N -i $r --page raw --csv --metrics $M 2>&1 | python3 -c "
import csv,sys
rows=list(csv.reader(sys.stdin))
# find the header row
h=None
for i,r in enumerate(rows):
    if r and r[0]==\"ID\": h=i; break
if h is None: print(\"\n\".join(\",\".join(r) for r in rows[:5])); sys.exit()
hdr=rows[h]; units=rows[h+1]
for r in rows[h+2:]:
    name=r[hdr.index(\"Kernel Name\")][:60]
    print(\"--\",name,r[hdr.index(\"Grid Size\")],r[hdr.index(\"Block Size\")])
    for j,c in enumerate(hdr):
        if j>=11: print(\"   %-72s %s %s\"%(c,r[j],units[j]))
"
done
