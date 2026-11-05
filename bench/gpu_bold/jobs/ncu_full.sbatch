#!/bin/bash
#SBATCH --job-name=gbold-ncufull
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=00:50:00
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
runncu run1 --kernel-name-base demangled --kernel-name "$KRE" --metrics gpu__time_duration.sum,launch__grid_size,launch__block_size,launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__occupancy_limit_blocks --csv --log-file $R/run1.csv
python3 - $R <<'EOF'
import csv, collections, sys, re, json
R = sys.argv[1]
rows = list(csv.DictReader(l for l in open(R + "/run1.csv", errors="replace") if l.startswith('"')))
k = collections.OrderedDict()
for r in rows:
    nm = r["Kernel Name"]; m = r["Metric Name"]
    try: v = float(r["Metric Value"].replace(",", ""))
    except ValueError: continue
    d = k.setdefault(nm, collections.defaultdict(list)); d[m].append(v)
tab = []
for nm, d in k.items():
    t = d["gpu__time_duration.sum"]
    tab.append((sum(t), len(t), nm, d))
tab.sort(key=lambda x: -x[0])
tot = sum(x[0] for x in tab)
out = open(R + "/run1_table.txt", "w")
out.write(f"prefill kernels in run 1: {sum(x[1] for x in tab)} launches, summed device time {tot / 1e9:.3f} s\n")
out.write("| device s | share % | launches | mean ms | grid | block | regs | occ limit regs | occ limit warps | kernel |\n|---|---|---|---|---|---|---|---|---|---|\n")
mean = lambda x: sum(x) / len(x) if x else float("nan")
for s, n, nm, d in tab[:25]:
    out.write(f"| {s / 1e9:.3f} | {100 * s / tot:.1f} | {n} | {s / n / 1e6:.2f} | {mean(d['launch__grid_size']):.0f} | {mean(d['launch__block_size']):.0f} | {mean(d['launch__registers_per_thread']):.0f} | "
              f"{mean(d['launch__occupancy_limit_registers']):.0f} | {mean(d['launch__occupancy_limit_warps']):.0f} | {nm[:110]} |\n")
out.close()
pick = [tab[0]]
if len(tab) > 1 and tab[1][0] * 2 >= tab[0][0]: pick.append(tab[1])
json.dump([{"name": p[2], "launches": p[1], "device_s": p[0] / 1e9} for p in pick], open(R + "/pick.json", "w"))
for i, p in enumerate(pick):
    base = p[2].split("(")[0]
    skip = max(0, p[1] // 2 - 1)
    open(R + f"/pick{i + 1}.env", "w").write(f"KREGEX={json.dumps('regex:' + re.escape(base))}\nSKIP={skip}\n")
    print("pick", i + 1, base[:140], p[1], "launches, skip", skip)
EOF
cat $R/run1_table.txt | cut -c1-260
i=1
for f in $R/pick*.env; do
  [ -f $f ] || continue; . $f
  runncu full$i --kernel-name-base demangled --kernel-name "$KREGEX" --launch-skip $SKIP --launch-count 2 --set full --import-source yes --export $R/full$i --force-overwrite
  if [ -f $R/full$i.ncu-rep ]; then
    $NCU --import $R/full$i.ncu-rep --page details --print-units base > $R/ncu_details_$i.txt 2>&1
    $NCU --import $R/full$i.ncu-rep --page source --print-source sass --csv > $R/ncu_source_$i.txt 2>&1
    $NCU --import $R/full$i.ncu-rep --page raw --csv --metrics launch__grid_size,launch__block_size,launch__registers_per_thread,launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,launch__stack_size,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_warps,launch__occupancy_limit_blocks > $R/ncu_launch_$i.csv 2>&1
  else echo "no report for pick $i"; fi
  i=$((i + 1))
done
ls -la $R
echo done $(date -Is)
